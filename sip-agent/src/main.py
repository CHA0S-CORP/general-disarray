#!/usr/bin/env python3
"""
SIP AI Assistant - API-based Architecture
==========================================
All ML inference offloaded to dedicated services:
- Speaches API for STT (Whisper) and TTS (Piper/Kokoro)
- vLLM for LLM

This container is lightweight - just orchestration.
"""

import os
import time
import signal
import asyncio
import logging
import random
from typing import Optional, List, Dict

import numpy as np

from config import Config, get_config
from tool_manager import ToolManager
from llm_engine import create_llm_engine
from sip_handler import SIPHandler, PJSUA_AVAILABLE
from audio_pipeline import LowLatencyAudioPipeline, LatencyMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)


class SIPAIAssistant:
    """
    SIP AI Assistant with API-based ML inference.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        logger.info("Initializing SIP AI Assistant...")
        
        # Core components
        self.tool_manager = ToolManager(self)
        self.llm_engine = create_llm_engine(config, self.tool_manager)
        self.audio_pipeline = LowLatencyAudioPipeline(config)
        self.sip_handler = SIPHandler(config, self._on_call_received)
        
        # State
        self.conversation_history: List[Dict] = []
        self.current_call = None
        self._processing = False
        self._audio_loop_task = None
        self._call_lock = asyncio.Lock()
        
        # Pre-cached phrases for instant playback
        self.acknowledgments = [
            "Okay.", "Got it.", "One moment.", "Sure.", "Copy that.",
            "Alright.", "No problem.", "On it.", "You got it.", "Absolutely.",
            "Sure thing.", "Will do.", "Of course.", "Right away.", "Consider it done."
        ]
        
        self.thinking_phrases = [
            "Let me think about that.",
            "Give me a second.",
            "Let me check.",
            "One moment please.",
            "Hmm, let me see."
        ]
        
        self.greeting_phrases = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hello! How may I assist you?",
            "Hey! What can I help you with?"
        ]
        
        self.goodbye_phrases = [
            "Goodbye!",
            "Take care!",
            "Have a great day!",
            "Bye for now!",
            "Talk to you later!"
        ]
        
        self.error_phrases = [
            "Sorry, I didn't catch that.",
            "Could you repeat that please?",
            "I didn't quite get that.",
            "Sorry, can you say that again?"
        ]
        
        # Combined list for pre-caching
        self._phrases_to_cache = (
            self.acknowledgments + 
            self.thinking_phrases + 
            self.greeting_phrases + 
            self.goodbye_phrases +
            self.error_phrases
        )
        
    async def start(self):
        """Start all components."""
        logger.info("Starting SIP AI Assistant...")
        self.running = True
        
        # Start components
        logger.info("Starting LLM engine...")
        await self.llm_engine.start()
        
        logger.info("Starting audio pipeline...")
        await self.audio_pipeline.start()
        
        # Pre-cache common phrases
        await self._precache_phrases()
        
        logger.info("Starting SIP handler...")
        await self.sip_handler.start()
        
        logger.info("Starting tool manager...")
        await self.tool_manager.start()
        
        logger.info("SIP AI Assistant ready!")
        logger.info(f"SIP URI: sip:{self.config.sip_user}@{self.config.sip_domain}")
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
            
    async def stop(self):
        """Stop all components."""
        logger.info("Stopping...")
        self.running = False
        
        await self.tool_manager.stop()
        await self.sip_handler.stop()
        await self.audio_pipeline.stop()
        await self.llm_engine.stop()
        
        logger.info("Stopped.")
        
    async def _precache_phrases(self):
        """Pre-generate audio for common phrases."""
        logger.info(f"Pre-caching {len(self._phrases_to_cache)} phrases...")
        
        cached = 0
        for phrase in self._phrases_to_cache:
            try:
                audio = await self.audio_pipeline.synthesize(phrase)
                if audio:
                    cached += 1
            except Exception as e:
                logger.warning(f"Failed to cache '{phrase}': {e}")
                
        logger.info(f"Pre-cached {cached}/{len(self._phrases_to_cache)} phrases")
        
    def get_random_acknowledgment(self) -> str:
        """Get a random acknowledgment phrase."""
        return random.choice(self.acknowledgments)
        
    def get_random_thinking(self) -> str:
        """Get a random thinking/processing phrase."""
        return random.choice(self.thinking_phrases)
        
    def get_random_greeting(self) -> str:
        """Get a random greeting phrase."""
        return random.choice(self.greeting_phrases)
        
    def get_random_goodbye(self) -> str:
        """Get a random goodbye phrase."""
        return random.choice(self.goodbye_phrases)
        
    def get_random_error(self) -> str:
        """Get a random error/retry phrase."""
        return random.choice(self.error_phrases)
        
    async def _on_call_received(self, call_info):
        """Handle incoming call."""
        # Prevent duplicate handling
        if self._call_lock.locked():
            logger.warning("Call already being handled, ignoring duplicate callback")
            return
            
        async with self._call_lock:
            try:
                # Cancel any existing audio loop
                if self._audio_loop_task and not self._audio_loop_task.done():
                    self._audio_loop_task.cancel()
                    try:
                        await self._audio_loop_task
                    except asyncio.CancelledError:
                        pass
                
                remote_uri = getattr(call_info, 'remote_uri', 'unknown')
                logger.info(f"Call received from: {remote_uri}")
                
                self.current_call = call_info
                self.conversation_history = []
                self._processing = False
                
                # Play greeting
                await self._play_greeting()
                
                # Start listening (single task)
                logger.info("Listening...")
                self._audio_loop_task = asyncio.create_task(self._audio_processing_loop())
            except Exception as e:
                logger.error(f"Error handling call: {e}", exc_info=True)
        
    async def _play_greeting(self):
        """Play initial greeting (uses pre-cached audio)."""
        greeting = self.get_random_greeting()
        
        try:
            logger.info(f"Playing greeting: {greeting}")
            # This should hit the cache since we pre-cached it
            audio = await self.audio_pipeline.synthesize(greeting)
            if audio:
                await self._play_audio(audio)
        except Exception as e:
            logger.error(f"Error playing greeting: {e}")
            
    async def _audio_processing_loop(self):
        """Main audio processing loop."""
        logger.info("Audio processing loop started")
        
        while self.running and self.current_call:
            try:
                # Check call state
                if not getattr(self.current_call, 'is_active', False):
                    logger.info("Call ended, stopping audio loop")
                    break
                    
                # Wait for media to be ready
                if not getattr(self.current_call, 'media_ready', False):
                    await asyncio.sleep(0.1)
                    continue
                    
                try:
                    # Try to receive audio
                    audio_chunk = await self.sip_handler.receive_audio(
                        self.current_call, 
                        timeout=0.1
                    )
                    
                    if audio_chunk:
                        # Check for barge-in
                        if self._processing and self.audio_pipeline.has_speech(audio_chunk):
                            logger.info("Barge-in detected")
                            await self._handle_barge_in()
                            
                        # Process through VAD/STT
                        transcription = await self.audio_pipeline.process_audio(audio_chunk)
                        
                        if transcription:
                            await self._handle_transcription(transcription)
                            
                except Exception as e:
                    logger.debug(f"Audio read error: {e}")
                    
                await asyncio.sleep(0.05)  # 50ms polling interval
                    
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(0.1)
                
        logger.info("Audio processing loop ended")
                
    async def _handle_transcription(self, text: str):
        """Handle transcribed text."""
        text = text.strip()
        if not text or len(text) < 2:
            return
        
        # Prevent overlapping processing
        if self._processing:
            logger.debug(f"Already processing, queuing: {text}")
            return
            
        logger.info(f"User: {text}")
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        
        # Generate response
        self._processing = True
        
        try:
            response = await self._generate_response(text)
            
            if response:
                logger.info(f"Assistant: {response}")
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # Synthesize and play
                await self._speak(response)
                
        except Exception as e:
            logger.error(f"Response error: {e}")
            await self._speak(self.get_random_error())
            
        finally:
            self._processing = False
            
    async def _generate_response(self, user_input: str) -> str:
        """Generate LLM response."""
        try:
            caller_id = getattr(self.current_call, 'remote_uri', 'unknown') if self.current_call else 'unknown'
            response = await self.llm_engine.generate_response(
                self.conversation_history,
                {"caller_id": caller_id}
            )
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self.get_random_error()
            
    async def _speak(self, text: str):
        """Synthesize and play text."""
        try:
            # Check cache first
            cached = self.audio_pipeline.get_cached_audio(text)
            if cached:
                await self._play_audio(cached)
                return
                
            # Synthesize
            start = time.time()
            audio = await self.audio_pipeline.synthesize(text)
            elapsed = (time.time() - start) * 1000
            
            if audio:
                logger.info(f"TTS: {elapsed:.0f}ms for {len(text)} chars")
                await self._play_audio(audio)
            else:
                logger.warning("TTS returned no audio")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            
    async def _play_audio(self, audio: bytes):
        """Play audio to caller."""
        try:
            if self.current_call:
                await self.sip_handler.send_audio(self.current_call, audio)
        except Exception as e:
            logger.error(f"Playback error: {e}")
            
    async def _handle_barge_in(self):
        """Handle user interruption."""
        logger.info("Stopping playback for barge-in")
        # Stop playback via playlist player
        if self.current_call:
            player = self.sip_handler.get_playlist_player(self.current_call)
            if player:
                player.stop_all()
        self._processing = False
        
    async def make_outbound_call(self, uri: str, message: str):
        """Make an outbound call and play a message."""
        # Format URI if it's just an extension/number
        if not uri.startswith('sip:'):
            # Strip any existing sip: prefix or angle brackets
            clean_uri = uri.replace('<', '').replace('>', '').replace('sip:', '')
            # If it doesn't have an @, add the domain
            if '@' not in clean_uri:
                uri = f"sip:{clean_uri}@{self.config.sip_domain}"
            else:
                uri = f"sip:{clean_uri}"
                
        logger.info(f"Making outbound call to {uri}")
        try:
            call_info = await self.sip_handler.make_call(uri)
            if call_info:
                # Wait for call to connect (configurable ring timeout)
                ring_timeout = self.config.callback_ring_timeout_s
                start_time = asyncio.get_event_loop().time()
                
                # Poll for call to be answered
                while asyncio.get_event_loop().time() - start_time < ring_timeout:
                    if getattr(call_info, 'is_active', False):
                        break
                    await asyncio.sleep(0.5)
                else:
                    # Timed out waiting for answer
                    logger.warning(f"Call to {uri} not answered within {ring_timeout}s")
                    await self.sip_handler.hangup_call(call_info)
                    return
                
                # Small delay after answer for audio to stabilize
                await asyncio.sleep(1)
                
                # Play message
                audio = await self.audio_pipeline.synthesize(message)
                if audio:
                    await self.sip_handler.send_audio(call_info, audio)
                    # Wait for audio to play (estimate based on audio length)
                    audio_duration = len(audio) / (self.config.sample_rate * 2)
                    await asyncio.sleep(audio_duration + 1)
                    
                # Hang up
                await self.sip_handler.hangup_call(call_info)
                logger.info(f"Outbound call to {uri} completed")
            else:
                logger.error(f"Failed to connect outbound call to {uri}")
        except Exception as e:
            logger.error(f"Outbound call failed: {e}")
            raise
        
    async def schedule_callback(self, delay: int, message: str = "This is your scheduled callback.", destination: str = None):
        """Schedule a callback to the caller."""
        if destination == "CALLER_NUMBER" or destination is None:
            # Get caller's number from current call
            if self.current_call:
                destination = getattr(self.current_call, 'remote_uri', None)
                
        if not destination:
            logger.warning("No destination for callback")
            return
            
        logger.info(f"Scheduling callback in {delay}s to {destination}: {message}")
        
        # Use tool_manager's scheduler for proper task management
        await self.tool_manager.schedule_task(
            task_type="callback",
            delay_seconds=delay,
            message=message,
            target_uri=destination
        )


async def main():
    """Main entry point."""
    config = get_config()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
    
    assistant = SIPAIAssistant(config)
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(assistant.stop())
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)
        
    try:
        await assistant.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())
