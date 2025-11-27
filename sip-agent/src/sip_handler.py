"""
SIP Handler using PJSUA2
========================
Handles SIP registration, incoming/outgoing calls, and audio streams.

THREAD SAFETY:
- All PJSIP operations MUST go through the command queue
- The PJSIP library runs in a single dedicated thread
- External threads communicate via _queue_command()
- PlaylistPlayer uses polling instead of Timer threads
"""

import os
import asyncio
import logging
import time
import queue
import wave
import contextlib
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from collections import deque
import threading
import numpy as np

try:
    import pjsua2 as pj
    PJSUA_AVAILABLE = True
except ImportError:
    PJSUA_AVAILABLE = False
    print("WARNING: pjsua2 not available, using mock SIP handler")

from config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# Thread-Safe Registration (only called from PJSIP thread)
# ============================================================================

_pjsip_thread_id: Optional[int] = None


def is_pjsip_thread() -> bool:
    """Check if current thread is the PJSIP thread."""
    return threading.current_thread().ident == _pjsip_thread_id


def assert_pjsip_thread():
    """Assert we're in the PJSIP thread. Debug helper."""
    if _pjsip_thread_id and not is_pjsip_thread():
        logger.error(f"PJSIP operation from wrong thread! Current: {threading.current_thread().name}")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CallInfo:
    """Information about an active call."""
    call_id: str
    remote_uri: str
    is_active: bool
    start_time: float
    pj_call: Any = None
    audio_buffer: deque = None
    record_file: Optional[str] = None
    playback_queue: deque = None
    media_ready: bool = False
    stream_player: Any = None  # PlaylistPlayer
    # NEW: Ring buffer for real-time audio capture
    audio_ring_buffer: deque = None
    audio_ring_lock: Any = None
    
    def __post_init__(self):
        if self.audio_buffer is None:
            self.audio_buffer = deque(maxlen=1000)
        if self.playback_queue is None:
            self.playback_queue = deque()
        if self.audio_ring_buffer is None:
            # Ring buffer holds ~2 seconds of audio at 16kHz, 20ms chunks
            self.audio_ring_buffer = deque(maxlen=100)
        if self.audio_ring_lock is None:
            self.audio_ring_lock = threading.Lock()
            
    @property
    def duration(self) -> float:
        return time.time() - self.start_time


@dataclass
class PlaybackItem:
    """An item in the playback queue."""
    file_path: str
    duration: float
    start_time: float = 0
    started: bool = False


# ============================================================================
# PJSIP Call Class
# ============================================================================

class SIPCall(pj.Call if PJSUA_AVAILABLE else object):
    """Custom call class with callbacks."""
    
    def __init__(self, acc, call_id: int, handler: 'SIPHandler'):
        if PJSUA_AVAILABLE:
            super().__init__(acc, call_id)
        self.handler = handler
        self.call_info: Optional[CallInfo] = None
        self.aud_med: Any = None
        self.recorder: Any = None
        self.player: Any = None
        self.audio_sink: Any = None  # AudioCaptureSink for file-based capture
        self.direct_sink: Any = None  # DirectAudioCaptureSink if available
        
    def onCallState(self, prm):
        """Called when call state changes (PJSIP thread)."""
        ci = self.getInfo()
        logger.info(f"Call state: {ci.stateText}")
        
        if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            if self.call_info:
                self.call_info.is_active = True
                
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            if self.call_info:
                self.call_info.is_active = False
            self._cleanup_media()
            self.handler._on_call_ended(self)
            
    def onCallMediaState(self, prm):
        """Called when media state changes (PJSIP thread)."""
        ci = self.getInfo()
        
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and \
               mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                
                self.aud_med = self.getAudioMedia(mi.index)
                sample_rate = self.handler.config.sample_rate
                
                # Set up audio recording (file-based, reliable)
                try:
                    import tempfile
                    fd, record_path = tempfile.mkstemp(suffix='.wav', prefix='sip_in_')
                    os.close(fd)
                    
                    if self.call_info:
                        self.call_info.record_file = record_path
                    
                    self.recorder = pj.AudioMediaRecorder()
                    self.recorder.createRecorder(record_path)
                    self.aud_med.startTransmit(self.recorder)
                    
                    # Create the capture helper for reading from the file
                    if self.call_info:
                        self.audio_sink = AudioCaptureSink(self.call_info, sample_rate)
                    
                    logger.info(f"Recording to: {record_path} at {sample_rate}Hz")
                    
                except Exception as e:
                    logger.error(f"Failed to set up audio recording: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Try direct capture if available (may not work on all builds)
                if DIRECT_CAPTURE_AVAILABLE and self.call_info:
                    try:
                        self.direct_sink = DirectAudioCaptureSink(self.call_info, sample_rate)
                        
                        fmt = pj.MediaFormatAudio()
                        fmt.type = pj.PJMEDIA_TYPE_AUDIO
                        fmt.clockRate = sample_rate
                        fmt.channelCount = 1
                        fmt.bitsPerSample = 16
                        fmt.frameTimeUsec = 20000  # 20ms
                        
                        self.direct_sink.createPort("direct_capture", fmt)
                        self.aud_med.startTransmit(self.direct_sink)
                        
                        logger.info("Direct audio capture enabled")
                    except Exception as e:
                        logger.debug(f"Direct capture not available: {e}")
                        self.direct_sink = None
                
                if self.call_info:
                    self.call_info.media_ready = True
                    logger.info("Audio media ready")
                
    def _cleanup_media(self):
        """Clean up media resources (PJSIP thread only)."""
        try:
            # Clean up direct audio capture sink (if used)
            if self.direct_sink and self.aud_med:
                try:
                    self.aud_med.stopTransmit(self.direct_sink)
                except:
                    pass
                self.direct_sink = None
            
            # Clean up file-based audio capture
            # (AudioCaptureSink doesn't need stopTransmit - it reads from file)
            self.audio_sink = None
                
            if self.recorder and self.aud_med:
                try:
                    self.aud_med.stopTransmit(self.recorder)
                except:
                    pass
                self.recorder = None
                
            if self.player and self.aud_med:
                try:
                    self.player.stopTransmit(self.aud_med)
                except:
                    pass
                self.player = None
                
            self.aud_med = None
            
            # Clean up recording file
            if self.call_info and self.call_info.record_file:
                try:
                    if os.path.exists(self.call_info.record_file):
                        os.unlink(self.call_info.record_file)
                except:
                    pass
                self.call_info.record_file = None
                
        except Exception as e:
            logger.debug(f"Media cleanup error: {e}")


# ============================================================================
# Real-time Audio Capture Sink
# ============================================================================

class AudioCaptureSink:
    """
    Audio capture using a recorder + periodic file reading.
    
    PJSUA2's AudioMediaPort Python bindings can be unreliable,
    so we use a hybrid approach:
    1. Record to a WAV file
    2. Periodically read new audio data from the file
    3. Push chunks to a ring buffer for async processing
    
    This works reliably across different PJSUA2 builds.
    """
    
    def __init__(self, call_info: CallInfo, sample_rate: int = 16000):
        self.call_info = call_info
        self.sample_rate = sample_rate
        self._file_pos = 0
        self._header_size = 44  # Standard WAV header size
        self._frame_count = 0
        self._last_read_time = 0
        self._min_read_interval = 0.02  # 20ms minimum between reads
        
    def read_new_audio(self) -> Optional[bytes]:
        """
        Read any new audio data from the recording file.
        Call this periodically from the PJSIP thread.
        Returns None if no new data, or bytes of new audio.
        """
        if not self.call_info or not self.call_info.record_file:
            return None
            
        now = time.time()
        if now - self._last_read_time < self._min_read_interval:
            return None
        self._last_read_time = now
            
        try:
            file_path = self.call_info.record_file
            if not os.path.exists(file_path):
                return None
                
            file_size = os.path.getsize(file_path)
            
            # Skip if file hasn't grown
            if file_size <= self._file_pos:
                return None
                
            # First read: skip header
            if self._file_pos == 0:
                self._file_pos = self._header_size
                
            # Read new data
            with open(file_path, 'rb') as f:
                f.seek(self._file_pos)
                new_data = f.read(file_size - self._file_pos)
                
            if new_data:
                self._file_pos = file_size
                self._frame_count += 1
                
                # Push to ring buffer
                with self.call_info.audio_ring_lock:
                    self.call_info.audio_ring_buffer.append(new_data)
                    
                if self._frame_count % 50 == 0:
                    logger.debug(f"Captured {len(new_data)} bytes (read #{self._frame_count})")
                    
                return new_data
                
        except Exception as e:
            if "No such file" not in str(e):
                logger.error(f"Audio read error: {e}")
                
        return None


# Alternative: Direct AudioMediaPort approach (for newer PJSUA2 builds)
if PJSUA_AVAILABLE:
    try:
        class DirectAudioCaptureSink(pj.AudioMediaPort):
            """
            Direct audio capture using PJSUA2 AudioMediaPort.
            This may not work on all PJSUA2 Python builds.
            """
            
            def __init__(self, call_info: CallInfo, sample_rate: int = 16000):
                super().__init__()
                self.call_info = call_info
                self.sample_rate = sample_rate
                self._frame_count = 0
                
            def onFrameRequested(self, frame):
                """Called when PJSIP needs audio to send (outbound)."""
                pass
                
            def onFrameReceived(self, frame):
                """Called when audio frame is received from remote party."""
                if not self.call_info or not frame.buf:
                    return
                    
                try:
                    audio_data = bytes(frame.buf)
                    
                    with self.call_info.audio_ring_lock:
                        self.call_info.audio_ring_buffer.append(audio_data)
                        
                    self._frame_count += 1
                    if self._frame_count % 100 == 0:
                        logger.debug(f"Direct capture: {self._frame_count} frames")
                        
                except Exception as e:
                    logger.error(f"Direct capture error: {e}")
                    
        DIRECT_CAPTURE_AVAILABLE = True
    except Exception:
        DIRECT_CAPTURE_AVAILABLE = False
else:
    DIRECT_CAPTURE_AVAILABLE = False


# ============================================================================
# PJSIP Account Class
# ============================================================================

class SIPAccount(pj.Account if PJSUA_AVAILABLE else object):
    """Custom account class with callbacks."""
    
    def __init__(self, handler: 'SIPHandler'):
        if PJSUA_AVAILABLE:
            super().__init__()
        self.handler = handler
        
    def onRegState(self, prm):
        """Called when registration state changes."""
        ai = self.getInfo()
        logger.info(f"Registration state: {ai.regStatusText} (code={ai.regStatus})")
        
        if ai.regStatus == 200:
            logger.info(f"✓ Registered: {ai.uri}")
            self.handler._registered.set()
        elif ai.regStatus >= 400:
            logger.error(f"✗ Registration failed: {ai.regStatusText}")
            
    def onIncomingCall(self, prm):
        """Called for incoming calls (PJSIP thread)."""
        call = SIPCall(self, prm.callId, self.handler)
        ci = call.getInfo()
        logger.info(f"Incoming call from: {ci.remoteUri}")
        
        call_info = CallInfo(
            call_id=str(prm.callId),
            remote_uri=ci.remoteUri,
            is_active=False,
            start_time=time.time(),
            pj_call=call
        )
        call.call_info = call_info
        self.handler.active_calls[call_info.call_id] = call
        
        # Auto-answer
        try:
            answer_prm = pj.CallOpParam()
            answer_prm.statusCode = 200
            call.answer(answer_prm)
            call_info.is_active = True
            logger.info(f"Auto-answered call {call_info.call_id}")
        except Exception as e:
            logger.error(f"Error answering call: {e}")
        
        # Notify async handler
        if self.handler.loop:
            asyncio.run_coroutine_threadsafe(
                self.handler._handle_incoming_call(call),
                self.handler.loop
            )


# ============================================================================
# Thread-Safe Playlist Player
# ============================================================================

class PlaylistPlayer:
    """
    Thread-safe audio playlist player.
    
    IMPORTANT: This class does NOT directly call PJSIP.
    All PJSIP operations are routed through the SIPHandler command queue.
    The PJSIP thread polls pending_actions and executes them.
    """
    
    def __init__(self, handler: 'SIPHandler', call_id: str):
        self.handler = handler
        self.call_id = call_id
        
        # Thread-safe queue for files to play
        self.file_queue: queue.Queue = queue.Queue()
        
        # State (protected by lock)
        self._lock = threading.Lock()
        self._is_playing = False
        self._current_file: Optional[str] = None
        self._current_start: float = 0
        self._current_duration: float = 0
        self._stopped = False
        
        # PJSIP player reference (only accessed from PJSIP thread)
        self._pj_player: Any = None
        
    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing
            
    def enqueue_file(self, file_path: str):
        """Add file to playback queue (thread-safe, any thread)."""
        if self._stopped:
            return
            
        # Get duration
        duration = 0.5
        try:
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                duration = f.getnframes() / float(f.getframerate())
        except Exception as e:
            logger.warning(f"Could not read WAV duration: {e}")
            
        self.file_queue.put((file_path, duration))
        logger.debug(f"Enqueued: {file_path} ({duration:.2f}s)")
        
    def stop_all(self):
        """Stop all playback (thread-safe, any thread)."""
        with self._lock:
            self._stopped = True
            self._is_playing = False
            
        # Clear queue and delete files
        while True:
            try:
                file_path, _ = self.file_queue.get_nowait()
                try:
                    os.unlink(file_path)
                except:
                    pass
            except queue.Empty:
                break
                
    def _poll_and_update(self, pj_call: SIPCall):
        """
        Poll playback state and start next file if needed.
        MUST BE CALLED FROM PJSIP THREAD ONLY.
        """
        if self._stopped:
            self._cleanup_player(pj_call)
            return
            
        with self._lock:
            # Check if current file finished
            if self._is_playing and self._current_file:
                elapsed = time.time() - self._current_start
                if elapsed >= self._current_duration:
                    # Current file done
                    self._cleanup_player(pj_call)
                    self._delete_current_file()
                    self._is_playing = False
                    self._current_file = None
                    
            # Start next file if not playing
            if not self._is_playing:
                try:
                    file_path, duration = self.file_queue.get_nowait()
                    self._start_playback(pj_call, file_path, duration)
                except queue.Empty:
                    pass
                    
    def _start_playback(self, pj_call: SIPCall, file_path: str, duration: float):
        """Start playing a file (PJSIP thread only)."""
        if not pj_call.aud_med:
            logger.warning("No audio media for playback")
            try:
                os.unlink(file_path)
            except:
                pass
            return
            
        try:
            self._pj_player = pj.AudioMediaPlayer()
            self._pj_player.createPlayer(file_path, pj.PJMEDIA_FILE_NO_LOOP)
            self._pj_player.startTransmit(pj_call.aud_med)
            
            self._current_file = file_path
            self._current_duration = duration + 0.05  # Small buffer
            self._current_start = time.time()
            self._is_playing = True
            
            logger.debug(f"Playing: {file_path} ({duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
            try:
                os.unlink(file_path)
            except:
                pass
                
    def _cleanup_player(self, pj_call: SIPCall):
        """Clean up current player (PJSIP thread only)."""
        if self._pj_player:
            try:
                if pj_call.aud_med:
                    self._pj_player.stopTransmit(pj_call.aud_med)
            except:
                pass
            self._pj_player = None
            
    def _delete_current_file(self):
        """Delete current file."""
        if self._current_file:
            try:
                os.unlink(self._current_file)
            except:
                pass
            self._current_file = None


# ============================================================================
# Main SIP Handler
# ============================================================================

class SIPHandler:
    """
    Thread-safe SIP handler.
    
    All PJSIP operations are serialized through the command queue
    and executed in the dedicated PJSIP thread.
    """
    
    def __init__(self, config: Config, on_call_callback: Callable):
        self.config = config
        self.on_call_callback = on_call_callback
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        self.endpoint: Optional[pj.Endpoint] = None
        self.account: Optional[SIPAccount] = None
        self.active_calls: Dict[str, SIPCall] = {}
        
        # Playlist players per call
        self._playlist_players: Dict[str, PlaylistPlayer] = {}
        
        self._running = False
        self._pj_thread: Optional[threading.Thread] = None
        self._initialized = threading.Event()
        self._registered = threading.Event()
        
        # Command queue for thread-safe operations
        self._cmd_queue: queue.Queue = queue.Queue()
        self._result_queues: Dict[int, queue.Queue] = {}
        self._cmd_id = 0
        self._cmd_lock = threading.Lock()
        
    def get_playlist_player(self, call_info: CallInfo) -> PlaylistPlayer:
        """Get or create playlist player for a call."""
        if call_info.call_id not in self._playlist_players:
            self._playlist_players[call_info.call_id] = PlaylistPlayer(
                self, call_info.call_id
            )
        return self._playlist_players[call_info.call_id]
        
    def _queue_command(self, cmd: str, *args, **kwargs) -> Any:
        """Queue a command to run in PJSIP thread and wait for result."""
        if not self._running:
            return None
            
        with self._cmd_lock:
            cmd_id = self._cmd_id
            self._cmd_id += 1
            result_queue = queue.Queue()
            self._result_queues[cmd_id] = result_queue
            
        self._cmd_queue.put((cmd_id, cmd, args, kwargs))
        
        try:
            result = result_queue.get(timeout=10.0)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            logger.error(f"Command timeout: {cmd}")
            return None
        finally:
            with self._cmd_lock:
                if cmd_id in self._result_queues:
                    del self._result_queues[cmd_id]
                    
    def _process_commands(self):
        """Process pending commands (PJSIP thread only)."""
        # Process command queue
        processed = 0
        while not self._cmd_queue.empty() and processed < 10:
            try:
                cmd_id, cmd, args, kwargs = self._cmd_queue.get_nowait()
                result = self._execute_command(cmd, args, kwargs)
                
                with self._cmd_lock:
                    if cmd_id in self._result_queues:
                        self._result_queues[cmd_id].put(result)
                        
                processed += 1
            except queue.Empty:
                break
                
        # Poll playlist players
        for call_id, player in list(self._playlist_players.items()):
            if call_id in self.active_calls:
                call = self.active_calls[call_id]
                if call.aud_med:
                    player._poll_and_update(call)
            else:
                # Call ended, clean up player
                player.stop_all()
                del self._playlist_players[call_id]
                
        # Poll audio capture for each active call
        for call_id, call in list(self.active_calls.items()):
            if call.audio_sink and call.call_info:
                # Read new audio from file and push to ring buffer
                call.audio_sink.read_new_audio()
                
    def _execute_command(self, cmd: str, args: tuple, kwargs: dict) -> Any:
        """Execute a command (PJSIP thread only)."""
        try:
            if cmd == "answer":
                call = args[0]
                prm = pj.CallOpParam()
                prm.statusCode = 200
                call.answer(prm)
                return True
                
            elif cmd == "hangup":
                call = args[0]
                prm = pj.CallOpParam()
                call.hangup(prm)
                return True
                
            elif cmd == "make_call":
                uri = args[0]
                return self._do_make_call(uri)
                
            elif cmd == "play_file":
                call_id = args[0]
                wav_path = args[1]
                if call_id in self.active_calls:
                    call = self.active_calls[call_id]
                    return self._play_file_direct(call, wav_path)
                return False
                
            else:
                logger.warning(f"Unknown command: {cmd}")
                return None
                
        except Exception as e:
            logger.error(f"Command error ({cmd}): {e}")
            return e
            
    def _play_file_direct(self, call: SIPCall, wav_path: str) -> bool:
        """Play file directly (PJSIP thread only)."""
        if not call.aud_med:
            return False
            
        try:
            player = pj.AudioMediaPlayer()
            player.createPlayer(wav_path, pj.PJMEDIA_FILE_NO_LOOP)
            player.startTransmit(call.aud_med)
            call.player = player
            return True
        except Exception as e:
            logger.error(f"Play file error: {e}")
            return False
            
    async def start(self):
        """Start the SIP handler."""
        global _pjsip_thread_id
        
        self.loop = asyncio.get_event_loop()
        
        if not PJSUA_AVAILABLE:
            logger.warning("PJSUA2 not available - mock mode")
            self._running = True
            return
            
        self._running = True
        self._pj_thread = threading.Thread(
            target=self._pjsip_thread_main,
            name="PJSIP",
            daemon=True
        )
        self._pj_thread.start()
        
        if not self._initialized.wait(timeout=10.0):
            logger.error("PJSUA2 initialization timeout")
            return
            
        await asyncio.sleep(2)
        
        if self._registered.is_set():
            logger.info("SIP handler ready and registered")
        else:
            logger.warning("SIP handler ready, registration pending")
            
    def _pjsip_thread_main(self):
        """Main PJSIP thread function."""
        global _pjsip_thread_id
        _pjsip_thread_id = threading.current_thread().ident
        
        try:
            # Initialize
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 1
            ep_cfg.logConfig.consoleLevel = 1
            ep_cfg.uaConfig.maxCalls = 1
            ep_cfg.uaConfig.userAgent = "SIP-AI-Assistant/1.0"
            
            self.endpoint.libInit(ep_cfg)
            
            # Null audio device (headless)
            try:
                self.endpoint.audDevManager().setNullDev()
                logger.info("Using null audio device")
            except Exception as e:
                logger.warning(f"Could not set null device: {e}")
                
            # Transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = self.config.sip_port
            
            transport_type = {
                "udp": pj.PJSIP_TRANSPORT_UDP,
                "tcp": pj.PJSIP_TRANSPORT_TCP,
                "tls": pj.PJSIP_TRANSPORT_TLS
            }.get(self.config.sip_transport.lower(), pj.PJSIP_TRANSPORT_UDP)
            
            self.endpoint.transportCreate(transport_type, transport_cfg)
            self.endpoint.libStart()
            
            # Codecs
            for i, codec in enumerate(self.config.audio_codecs):
                try:
                    self.endpoint.codecSetPriority(codec, 255 - i)
                except:
                    pass
                    
            # Account
            self._create_account()
            
            logger.info("PJSUA2 initialized")
            self._initialized.set()
            
            # Main loop
            while self._running:
                # Handle SIP events (50ms timeout)
                self.endpoint.libHandleEvents(50)
                
                # Process our command queue
                self._process_commands()
                
        except Exception as e:
            logger.error(f"PJSIP thread error: {e}")
            import traceback
            traceback.print_exc()
            self._initialized.set()
            
    def _create_account(self):
        """Create SIP account (PJSIP thread only)."""
        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = f"sip:{self.config.sip_user}@{self.config.sip_domain}"
        
        registrar = self.config.sip_registrar or self.config.sip_domain
        acc_cfg.regConfig.registrarUri = f"sip:{registrar}"
        acc_cfg.regConfig.registerOnAdd = True
        acc_cfg.regConfig.timeoutSec = 300
        
        if self.config.sip_password:
            cred = pj.AuthCredInfo()
            cred.scheme = "digest"
            cred.realm = "*"
            cred.username = self.config.sip_user
            cred.dataType = 0
            cred.data = self.config.sip_password
            acc_cfg.sipConfig.authCreds.append(cred)
            
        acc_cfg.natConfig.iceEnabled = False
        acc_cfg.natConfig.turnEnabled = False
        acc_cfg.mediaConfig.transportConfig.port = 10000
        acc_cfg.mediaConfig.transportConfig.portRange = 100
        
        self.account = SIPAccount(self)
        self.account.create(acc_cfg)
        
        logger.info(f"Account created: {acc_cfg.idUri}")
        
    async def stop(self):
        """Stop the SIP handler."""
        self._running = False
        
        # Stop all playlist players
        for player in self._playlist_players.values():
            player.stop_all()
        self._playlist_players.clear()
        
        # Hangup calls
        for call in list(self.active_calls.values()):
            try:
                await self.hangup_call(call.call_info)
            except:
                pass
                
        if self.endpoint:
            try:
                self.endpoint.libDestroy()
            except:
                pass
                
        if self._pj_thread:
            self._pj_thread.join(timeout=5)
            
        logger.info("SIP handler stopped")
        
    async def _handle_incoming_call(self, call: SIPCall):
        """Handle incoming call (async context)."""
        if self.on_call_callback:
            await self.on_call_callback(call.call_info)
            
    def _on_call_ended(self, call: SIPCall):
        """Called when call ends (PJSIP thread)."""
        if call.call_info:
            call_id = call.call_info.call_id
            if call_id in self.active_calls:
                del self.active_calls[call_id]
            if call_id in self._playlist_players:
                self._playlist_players[call_id].stop_all()
                del self._playlist_players[call_id]
                
    async def hangup_call(self, call_info: CallInfo):
        """Hangup a call."""
        if not PJSUA_AVAILABLE:
            return
            
        if call_info.pj_call:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._queue_command,
                "hangup",
                call_info.pj_call
            )
            
    async def make_call(self, uri: str) -> Optional[CallInfo]:
        """Make an outbound call."""
        if not PJSUA_AVAILABLE:
            return None
            
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._queue_command,
            "make_call",
            uri
        )
        
        if isinstance(result, CallInfo):
            return result
        return None
        
    def _do_make_call(self, uri: str) -> Optional[CallInfo]:
        """Make call (PJSIP thread only)."""
        if not self.account:
            return None
            
        try:
            call = SIPCall(self.account, pj.PJSUA_INVALID_ID, self)
            
            call_info = CallInfo(
                call_id=str(id(call)),
                remote_uri=uri,
                is_active=False,
                start_time=time.time(),
                pj_call=call
            )
            call.call_info = call_info
            
            prm = pj.CallOpParam(True)
            call.makeCall(uri, prm)
            
            self.active_calls[call_info.call_id] = call
            logger.info(f"Outbound call initiated: {uri}")
            
            return call_info
            
        except Exception as e:
            logger.error(f"Make call error: {e}")
            return None
            
    async def receive_audio(self, call_info: CallInfo, timeout: float = 0.1) -> Optional[bytes]:
        """
        Receive audio from a call's ring buffer.
        
        Returns concatenated audio chunks if available, None if buffer is empty.
        The ring buffer is populated by AudioCaptureSink.read_new_audio().
        """
        if not call_info or not call_info.is_active:
            return None
            
        # Ensure ring buffer is initialized
        if not hasattr(call_info, 'audio_ring_buffer') or call_info.audio_ring_buffer is None:
            await asyncio.sleep(timeout)
            return None
            
        if not hasattr(call_info, 'audio_ring_lock') or call_info.audio_ring_lock is None:
            await asyncio.sleep(timeout)
            return None
            
        # Calculate expected chunk size for the timeout period
        # At 16kHz, 20ms frames = 640 bytes per frame
        # timeout of 0.03 = ~1-2 frames expected
        
        start_time = time.time()
        collected = []
        
        while time.time() - start_time < timeout:
            # Try to get audio from ring buffer
            try:
                with call_info.audio_ring_lock:
                    if call_info.audio_ring_buffer:
                        chunk = call_info.audio_ring_buffer.popleft()
                        collected.append(chunk)
                    else:
                        break
            except Exception as e:
                logger.debug(f"Ring buffer read error: {e}")
                break
                    
            # Small yield to prevent tight loop
            await asyncio.sleep(0.001)
            
        if collected:
            return b''.join(collected)
            
        # Wait a bit if nothing was collected
        await asyncio.sleep(timeout)
        
        # Try one more time after waiting
        try:
            with call_info.audio_ring_lock:
                if call_info.audio_ring_buffer:
                    chunks = list(call_info.audio_ring_buffer)
                    call_info.audio_ring_buffer.clear()
                    if chunks:
                        return b''.join(chunks)
        except Exception as e:
            logger.debug(f"Final ring buffer read error: {e}")
                    
        return None
        
    async def send_audio(self, call_info: CallInfo, audio_data: bytes):
        """Send audio to a call using the playlist player."""
        if not call_info or not call_info.is_active:
            return
            
        player = self.get_playlist_player(call_info)
        
        # Write to temp file
        import tempfile
        fd, wav_path = tempfile.mkstemp(suffix='.wav', prefix='sip_out_')
        os.close(fd)
        
        with wave.open(wav_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(audio_data)
            
        # Enqueue for playback
        player.enqueue_file(wav_path)