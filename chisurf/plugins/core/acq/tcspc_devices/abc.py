"""
Abstract Base Class for TCSPC Device Wrappers

This module defines the common interface that all TCSPC device wrappers must implement.
"""

from abc import ABC, abstractmethod
from qtpy.QtCore import QObject, Signal


class TCSPCDeviceABC(QObject):
    """Abstract Base Class for TCSPC device wrappers."""

    # Signal emitted when a message is logged
    message_logged = Signal(str)

    @abstractmethod
    def log_message(self, message):
        """Log a message and emit the message_logged signal."""
        pass

    @abstractmethod
    def detect_cards(self, simulation=False):
        """Detect available TCSPC cards/devices."""
        pass

    @abstractmethod
    def set_active_cards(self, card_numbers):
        """Set the active cards/devices."""
        pass

    @abstractmethod
    def get_active_cards(self):
        """Get the active cards/devices."""
        pass

    @abstractmethod
    def initialize(self, simulation=True):
        """Initialize the TCSPC device."""
        pass

    @abstractmethod
    def start_measurement(self):
        """Start measurement."""
        pass

    @abstractmethod
    def stop_measurement(self):
        """Stop measurement."""
        pass

    @abstractmethod
    def read_fifo(self, max_words=32768):
        """Read data from FIFO."""
        pass

    @abstractmethod
    def get_fifo_usage(self):
        """Get FIFO usage."""
        pass

    @abstractmethod
    def close(self):
        """Close the device."""
        pass
