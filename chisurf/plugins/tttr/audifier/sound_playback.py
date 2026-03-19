"""
Sound Playback Module for TTTR Audifier

This module handles all audio playback functionality, providing a clean interface
for the GUI to control sound playback without managing the low-level details.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

try:
    from qtpy.QtMultimedia import QSound
except ImportError:
    QSound = None

try:
    from qtpy.QtCore import QTimer, Signal, QObject
except ImportError:
    # Fallback for non-Qt environments
    class QObject:
        pass
    class Signal:
        def __init__(self, *args):
            pass
        def emit(self, *args):
            pass
        def connect(self, slot):
            pass


@dataclass
class PlaybackState:
    """Current state of sound playback."""
    is_playing: bool = False
    is_paused: bool = False
    current_position: float = 0.0  # Current position in seconds
    duration: float = 0.0  # Total duration in seconds
    start_time: Optional[float] = None
    paused_time: Optional[float] = None


class SoundPlayer(QObject):
    """
    Handles audio playback for TTTR audifier.
    
    Provides play, pause, resume, stop functionality with position tracking
    and state management. Abstracts away QSound limitations.
    """
    
    # Signals for UI updates
    position_changed = Signal(float, float)  # current, duration
    state_changed = Signal(str)  # "playing", "paused", "stopped", "finished"
    error_occurred = Signal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Playback state
        self.state = PlaybackState()
        
        # Audio resources
        self.current_sound: Optional[QSound] = None
        self.temp_wav_path: Optional[str] = None
        
        # Position tracking timer
        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self._update_position)
        self.position_timer.setInterval(100)  # Update every 100ms
        
        # Cleanup on destruction
        self._cleanup_pending = False
    
    def load_audio(self, wav_data: np.ndarray, sample_rate: int) -> bool:
        """
        Load audio data for playback.
        
        Args:
            wav_data: Audio waveform data (float32, [-1, 1])
            sample_rate: Sample rate in Hz
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            # Clean up any existing audio
            self._cleanup_audio()
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                self.temp_wav_path = f.name
            
            # Write WAV data
            self._write_wav_file(self.temp_wav_path, wav_data, sample_rate)
            
            # Calculate duration
            self.state.duration = len(wav_data) / sample_rate
            
            # Load with QSound
            if QSound is not None:
                self.current_sound = QSound(self.temp_wav_path)
                return True
            else:
                self.error_occurred.emit("QtMultimedia not available for sound playback")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to load audio: {e}")
            self._cleanup_audio()
            return False
    
    def play(self) -> bool:
        """
        Start or resume audio playback.
        
        Returns:
            True if playback started successfully, False otherwise
        """
        if self.current_sound is None:
            self.error_occurred.emit("No audio loaded")
            return False
        
        try:
            if self.state.is_paused:
                return self._resume()
            else:
                return self._start_playback()
        except Exception as e:
            self.error_occurred.emit(f"Failed to play: {e}")
            return False
    
    def pause(self) -> bool:
        """
        Pause audio playback.
        
        Returns:
            True if paused successfully, False otherwise
        """
        if not self.state.is_playing or self.state.is_paused:
            return False
        
        try:
            # Stop the actual sound
            if self.current_sound is not None:
                self.current_sound.stop()
            
            # Record pause position
            if self.state.start_time is not None:
                self.state.paused_time = time.time() - self.state.start_time
                self.state.current_position = self.state.paused_time
            
            # Update state
            self.state.is_paused = True
            self.state.is_playing = False
            
            # Stop position tracking
            self.position_timer.stop()
            
            # Emit state change
            self.state_changed.emit("paused")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to pause: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop audio playback and reset to beginning.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            # Stop sound
            if self.current_sound is not None:
                self.current_sound.stop()
            
            # Stop timer
            self.position_timer.stop()
            
            # Reset state
            self.state = PlaybackState(duration=self.state.duration)
            
            # Emit state change and position
            self.state_changed.emit("stopped")
            self.position_changed.emit(0.0, self.state.duration)
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to stop: {e}")
            return False
    
    def revert(self) -> bool:
        """
        Revert playback to beginning (continue playing if was playing).
        
        Returns:
            True if reverted successfully, False otherwise
        """
        if self.state.duration == 0:
            return False
        
        was_playing = self.state.is_playing and not self.state.is_paused
        
        # Stop current playback
        if self.current_sound is not None:
            self.current_sound.stop()
        
        # Reset timing
        self.state.start_time = time.time()
        self.state.paused_time = None
        self.state.current_position = 0.0
        self.state.is_paused = False
        
        # Update position display
        self.position_changed.emit(0.0, self.state.duration)
        
        if was_playing:
            # Resume playing from start
            return self._start_playback()
        else:
            # Just reset position, don't start playing
            self.state.is_playing = False
            self.state_changed.emit("stopped")
            return True
    
    def get_position(self) -> tuple[float, float]:
        """
        Get current playback position.
        
        Returns:
            Tuple of (current_position, duration) in seconds
        """
        return self.state.current_position, self.state.duration
    
    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        return self.state
    
    def is_playing(self) -> bool:
        """Check if currently playing (not paused)."""
        return self.state.is_playing and not self.state.is_paused
    
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self.state.is_paused
    
    def _start_playback(self) -> bool:
        """Start new playback from beginning."""
        if self.current_sound is None:
            return False
        
        try:
            # Start playback
            self.current_sound.play()
            
            # Update state
            self.state.is_playing = True
            self.state.is_paused = False
            self.state.start_time = time.time()
            self.state.current_position = 0.0
            
            # Start position tracking
            self.position_timer.start()
            
            # Emit state change
            self.state_changed.emit("playing")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start playback: {e}")
            return False
    
    def _resume(self) -> bool:
        """Resume playback from paused position."""
        if self.current_sound is None or self.state.paused_time is None:
            return False
        
        try:
            # Restart playback (QSound limitation: can't seek)
            self.current_sound.play()
            
            # Adjust timing to account for paused position
            self.state.start_time = time.time() - self.state.paused_time
            self.state.is_playing = True
            self.state.is_paused = False
            self.state.paused_time = None
            
            # Resume position tracking
            self.position_timer.start()
            
            # Emit state change
            self.state_changed.emit("playing")
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to resume: {e}")
            return False
    
    def _update_position(self):
        """Update playback position and emit signals."""
        if self.state.start_time is None or self.state.duration == 0:
            return
        
        if self.state.is_paused:
            return  # Don't update position when paused
        
        # Calculate current position
        elapsed = time.time() - self.state.start_time
        self.state.current_position = elapsed
        
        if elapsed >= self.state.duration:
            # Playback finished
            self.position_timer.stop()
            self.state.is_playing = False
            self.state.is_paused = False
            self.state.current_position = self.state.duration
            self.state_changed.emit("finished")
            self.position_changed.emit(self.state.duration, self.state.duration)
        else:
            # Emit position update
            self.position_changed.emit(elapsed, self.state.duration)
    
    def _write_wav_file(self, path: str, data: np.ndarray, sample_rate: int):
        """Write numpy array to WAV file."""
        # Ensure data is in correct format
        data = np.asarray(data, dtype=np.float32)
        data = np.clip(data, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        pcm = (data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
    
    def _cleanup_audio(self):
        """Clean up audio resources."""
        # Stop playback
        if self.current_sound is not None:
            self.current_sound.stop()
            self.current_sound = None
        
        # Stop timer
        self.position_timer.stop()
        
        # Remove temporary file
        if self.temp_wav_path is not None and os.path.exists(self.temp_wav_path):
            try:
                os.unlink(self.temp_wav_path)
            except OSError:
                pass  # File might be locked or already deleted
            self.temp_wav_path = None
        
        # Reset state
        self.state = PlaybackState()
    
    def cleanup(self):
        """Clean up all resources. Call when destroying the player."""
        self._cleanup_audio()
        self._cleanup_pending = True
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if not self._cleanup_pending:
            self.cleanup()


# Convenience function for creating audio from TTTR data
def create_tttr_audio(
    data,
    channels: list[int],
    channel_cfg: dict,
    bin_width_s: float,
    sample_rate: int = 44100,
    env_mode: str = "log",
    master_gain: float = 0.8,
) -> tuple[np.ndarray, float]:
    """
    Create audio waveform from TTTR data.
    
    This is a convenience wrapper around the core TTTR audio generation
    functionality for use with the SoundPlayer.
    
    Args:
        data: TTTRData instance
        channels: List of routing channels to use
        channel_cfg: Channel configuration dictionary
        bin_width_s: Bin width in seconds
        sample_rate: Audio sample rate
        env_mode: Envelope mode ("linear", "sqrt", "log")
        master_gain: Master gain multiplier
        
    Returns:
        Tuple of (audio_data, duration) where audio_data is float32 numpy array
    """
    try:
        # Import core functions with fallback for direct execution
        try:
            from .core import tttr_to_wav
        except ImportError:
            # Fallback for direct execution
            from core import tttr_to_wav
        
        # Create temporary file for audio generation
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        # Generate audio using core functionality
        wav, edges, envelopes = tttr_to_wav(
            data=data,
            out_wav_path=temp_path,
            channels=channels,
            channel_cfg=channel_cfg,
            bin_width_s=bin_width_s,
            sample_rate=sample_rate,
            env_mode=env_mode,
            master_gain=master_gain
        )
        
        # Calculate duration
        duration = len(wav) / sample_rate
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        
        return wav, duration
        
    except Exception as e:
        raise RuntimeError(f"Failed to create TTTR audio: {e}")
