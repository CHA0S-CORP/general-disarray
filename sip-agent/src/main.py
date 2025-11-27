#!/usr/bin/env python3
"""
Low-Latency SIP AI Assistant
=============================
Optimized for minimal response latency.

Key optimizations:
1. Faster STT with aggressive VAD
2. LLM token streaming → immediate TTS
3. Sentence-level TTS pipelining
4. Pre-cached acknowledgment audio
5. Parallel processing pipeline

Target: < 1 second from user stop speaking → first audio output
"""

import time
import random
import signal
import asyncio
import logging
from typing import Optional

import numpy as np

from config import Config

from tool_manager import ToolManager
from llm_engine import create_llm_engine
from sip_handler import SIPHandler, PJSUA_AVAILABLE
from audio_pipeline import LowLatencyAudioPipeline, StreamingResponsePipeline, LatencyMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sip_assistant.log')
    ]
)

logger = logging.getLogger(__name__)


class LowLatencySIPAssistant:
    """
    Low-latency SIP AI Assistant.
    
    Optimized for fastest possible response time.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        logger.info("Initializing Low-Latency SIP AI Assistant...")
        
        # Core components
        self.tool_manager = ToolManager(self)
        self.llm_engine = create_llm_engine(config, self.tool_manager)
        self.audio_pipeline = LowLatencyAudioPipeline(config)
        self.sip_handler = SIPHandler(config, self._on_call_received)
        
        # Parallel response pipeline
        self.response_pipeline: Optional[StreamingResponsePipeline] = None
        
        # State
        self.conversation_history = []
        self.current_call = None
        self._barge_in_counter = 0
        
        # Pre-cached acknowledgments (will be loaded from TTS cache)
        self.acknowledgments = [
            "Okay", "Acknowledged", "Got it", "One moment", "Copy that", "On it"
        ]
        
    async def start(self):
        """Start all components."""
        logger.info("Starting Low-Latency SIP AI Assistant...")
        self.running = True
        
        # Start components
        await self.llm_engine.start()
        await self.audio_pipeline.start()
        await self.sip_handler.start()
        await self.tool_manager.start()
        
        # Create response pipeline
        self.response_pipeline = StreamingResponsePipeline(
            self.llm_engine,
            self.audio_pipeline.tts,
            self.config
        )
        
        logger.info("Low-Latency SIP AI Assistant ready!")
        logger.info(f"SIP URI: sip:{self.config.sip_user}@{self.config.sip_domain}")
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
            
    async def stop(self):
        """Stop gracefully."""
        logger.info("Stopping...")
        self.running = False
        
        await self.tool_manager.stop()
        await self.sip_handler.stop()
        await self.audio_pipeline.stop()
        await self.llm_engine.stop()
        
        logger.info("Stopped.")
        
    async def _on_call_received(self, call):
        """Handle incoming call with fast greeting."""
        logger.info(f"Incoming call from: {call.remote_uri}")
        self.current_call = call
        self.conversation_history = []
        
        # Wait for media
        wait_start = time.time()
        while not call.media_ready and time.time() - wait_start < 3.0:
            await asyncio.sleep(0.05)
            
        if not call.media_ready:
            logger.warning("Media not ready")
            return
            
        logger.info("Media ready")
        
        # Initialize playlist player through SIPHandler (thread-safe)
        if PJSUA_AVAILABLE and call.pj_call and call.pj_call.aud_med:
            call.stream_player = self.sip_handler.get_playlist_player(call)
                
        # Play greeting (use cached if available)
        greeting = "Ready for commands"
        await self._play_response(call, greeting)
        
        # Start conversation loop
        await self._fast_conversation_loop(call)
        
    async def _fast_conversation_loop(self, call):
        """
        Optimized conversation loop with parallel processing.
        """
        # Reset state
        self.audio_pipeline.vad.reset()
        self.audio_pipeline.audio_buffer.clear()
        self._barge_in_counter = 0
        
        logger.info("Listening (low-latency mode)...")
        
        while call.is_active and self.running:
            try:
                # Receive audio (short timeout for responsiveness)
                audio_data = await self.sip_handler.receive_audio(call, timeout=0.03)
                if audio_data is None:
                    continue
                    
                # Check if bot is speaking
                is_bot_speaking = False
                if hasattr(call, 'stream_player') and call.stream_player:
                    is_bot_speaking = call.stream_player.is_playing
                    
                if is_bot_speaking:
                    # Barge-in detection
                    await self._check_barge_in(call, audio_data)
                else:
                    # Process speech
                    transcription = await self.audio_pipeline.process_audio(audio_data)
                    
                    if transcription and len(transcription.strip()) >= 2:
                        await self._handle_user_input(call, transcription)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                
        # Cleanup
        if hasattr(call, 'stream_player') and call.stream_player:
            call.stream_player.stop_all()
        self.current_call = None
        
    async def _check_barge_in(self, call, audio_data: bytes):
        """Fast barge-in detection."""
        samples = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32)**2)) if len(samples) > 0 else 0
        
        if energy > self.config.barge_in_energy_threshold:
            if self.audio_pipeline.has_speech(audio_data):
                self._barge_in_counter += len(audio_data)
                barge_ms = (self._barge_in_counter / (self.config.sample_rate * 2)) * 1000
                
                # Reduced threshold for faster interruption
                if barge_ms >= 400:  # 400ms instead of 700ms
                    logger.info(f"Barge-in! Energy={energy:.0f}")
                    call.stream_player.stop_all()
                    self._barge_in_counter = 0
            else:
                self._barge_in_counter = 0
        else:
            self._barge_in_counter = 0
            
    async def _handle_user_input(self, call, transcription: str):
        """
        Handle user input with parallel LLM + TTS streaming.
        """
        response_start = time.time()
        logger.info(f"User: {transcription}")
        
        # Immediate acknowledgment (use cached audio for instant playback)
        ack = random.choice(self.acknowledgments)
        cached_ack = self.audio_pipeline.get_cached_audio(ack)
        
        if cached_ack:
            # Instant playback from cache
            await self._play_audio_chunk(call, cached_ack)
            logger.debug(f"Played cached ack: {ack}")
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": transcription
        })
        
        # Check for hangup
        if self._should_hangup(transcription):
            goodbye_audio = self.audio_pipeline.get_cached_audio("goodbye")
            if goodbye_audio:
                await self._play_audio_chunk(call, goodbye_audio)
            else:
                await self._play_response(call, "Goodbye!")
            await asyncio.sleep(1.0)
            await self.sip_handler.hangup_call(call)
            return
            
        # Generate and stream response in parallel
        full_response = await self._generate_and_stream_response(
            call,
            {
                "remote_uri": call.remote_uri,
                "duration": call.duration
            }
        )
        
        total_time = (time.time() - response_start) * 1000
        logger.info(f"Total response time: {total_time:.0f}ms")
        
        if full_response:
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
    async def _generate_and_stream_response(self, call, call_context: dict) -> str:
        """
        Generate LLM response and stream TTS in parallel.
        
        Key optimization: Start TTS on first complete sentence while
        LLM is still generating rest of response.
        """
        full_response = ""
        sentence_buffer = ""
        first_audio = True
        metrics = LatencyMetrics()
        metrics.stt_end = time.time()
        
        # Stream from LLM
        async for token in self._stream_llm_tokens(call_context):
            if metrics.llm_first_token == 0:
                metrics.llm_first_token = time.time()
                ttft = (metrics.llm_first_token - metrics.stt_end) * 1000
                logger.debug(f"LLM TTFT: {ttft:.0f}ms")
                
            full_response += token
            sentence_buffer += token
            
            # Check for sentence boundary
            if self._is_sentence_complete(sentence_buffer):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                
                if sentence:
                    # Stream this sentence to TTS immediately
                    async for audio_chunk in self.audio_pipeline.synthesize_stream(sentence):
                        if first_audio:
                            metrics.tts_first_chunk = time.time()
                            tts_latency = (metrics.tts_first_chunk - metrics.llm_first_token) * 1000
                            logger.debug(f"TTS first chunk: {tts_latency:.0f}ms")
                            first_audio = False
                            
                        await self._play_audio_chunk(call, audio_chunk)
                        
        # Handle remaining text
        if sentence_buffer.strip():
            async for audio_chunk in self.audio_pipeline.synthesize_stream(sentence_buffer.strip()):
                await self._play_audio_chunk(call, audio_chunk)
                
        metrics.llm_complete = time.time()
        
        # Log metrics
        if metrics.speech_end == 0:
            metrics.speech_end = metrics.stt_end
        total = (metrics.tts_first_chunk - metrics.stt_end) * 1000 if metrics.tts_first_chunk else 0
        logger.info(f"Response latency: {total:.0f}ms (STT done → first audio)")
        
        return full_response
        
    async def _stream_llm_tokens(self, call_context: dict):
        """Stream tokens from LLM."""
        messages = [
            {"role": "system", "content": self.llm_engine._build_system_prompt(call_context)}
        ]
        messages.extend(self.conversation_history[-self.config.max_conversation_turns * 2:])
        
        if not self.llm_engine.client:
            # Mock for testing
            for word in "I understand. How can I help?".split():
                yield word + " "
                await asyncio.sleep(0.02)
            return
            
        try:
            stream = await self.llm_engine.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                stream=True
            )
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                    
        except Exception as e:
            logger.error(f"LLM error: {e}")
            yield "Sorry, I had trouble with that."
            
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text ends with complete sentence."""
        text = text.strip()
        if not text:
            return False
            
        if text[-1] in '.!?':
            # Avoid abbreviation false positives
            lower = text.lower()
            for abbrev in ['dr.', 'mr.', 'mrs.', 'ms.', 'jr.', 'sr.', 'etc.', 'e.g.', 'i.e.']:
                if lower.endswith(abbrev):
                    return False
            return True
        return False
        
    async def _play_response(self, call, text: str):
        """Play synthesized response."""
        async for chunk in self.audio_pipeline.synthesize_stream(text):
            await self._play_audio_chunk(call, chunk)
            
    async def _play_audio_chunk(self, call, audio_chunk: bytes):
        """Play audio chunk to call."""
        if not hasattr(call, 'stream_player') or not call.stream_player:
            return
            
        if not call.is_active:
            return
            
        import tempfile
        import os
        import wave
        
        # Write to temp file
        fd, path = tempfile.mkstemp(suffix='.wav', prefix='chunk_')
        os.close(fd)
        
        with wave.open(path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(audio_chunk)
            
        call.stream_player.enqueue_file(path)
        
    def _should_hangup(self, text: str) -> bool:
        """Check for hangup phrases."""
        phrases = ["goodbye", "bye", "hang up", "end call", "that's all"]
        lower = text.lower()
        return any(p in lower for p in phrases)
        
    # Tool methods
    async def schedule_callback(self, delay: int, message: str, destination: str = None):
        """Schedule callback."""
        target = destination
        if not target and self.current_call:
            target = self.current_call.remote_uri
        if not target:
            return
            
        target = target.replace("-", "").replace(" ", "")
        if "sip:" not in target:
            target = f"sip:{target}@{self.config.sip_domain}"
            
        await self.tool_manager.schedule_callback(delay, message, target)
        
    async def make_outbound_call(self, uri: str, message: str):
        """Make outbound call."""
        call = await self.sip_handler.make_call(uri)
        if not call:
            return
            
        # Wait for answer
        for _ in range(60):
            if call.is_active:
                break
            if call.call_id not in self.sip_handler.active_calls:
                return
            await asyncio.sleep(0.5)
            
        if call.is_active:
            self.current_call = call
            self.conversation_history = []
            
            if PJSUA_AVAILABLE and call.pj_call and call.pj_call.aud_med:
                call.stream_player = self.sip_handler.get_playlist_player(call)
                
            await asyncio.sleep(0.5)
            self.audio_pipeline.vad.reset()
            self.audio_pipeline.audio_buffer.clear()
            
            await self._play_response(call, message)
            await self._fast_conversation_loop(call)
        else:
            await self.sip_handler.hangup_call(call)


async def main():
    """Entry point."""
    config = Config()
    assistant = LowLatencySIPAssistant(config)
    
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(assistant.stop())
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
        
    try:
        await assistant.start()
    except KeyboardInterrupt:
        pass
    finally:
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())