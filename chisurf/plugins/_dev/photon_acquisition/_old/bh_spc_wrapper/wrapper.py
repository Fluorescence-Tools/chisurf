"""
BH SPC Wrapper Implementation

This module provides a direct wrapper for the Becker & Hickl SPC hardware.
It includes classes and functions for initializing and controlling
BH SPC devices, as well as for acquiring data from them.
"""

import os
import array
import ctypes
import enum
import contextlib
import tempfile
import numpy as np
from qtpy.QtCore import QObject, Signal
from abc import ABC, abstractmethod

from ...photon_sources import TCSPCDeviceABC

PICOQUANT_AVAILABLE = False

# Direct wrapper for the BH SPC DLL
class DLLOperationMode(enum.Enum):
    """Enum for the operation mode of SPCM-DLL."""
    HARDWARE = 0
    SIMULATE_SPC_600 = 600
    SIMULATE_SPC_630 = 630
    SIMULATE_SPC_700 = 700
    SIMULATE_SPC_730 = 730
    SIMULATE_SPC_130 = 130
    SIMULATE_SPC_830 = 830
    SIMULATE_SPC_140 = 140
    SIMULATE_SPC_930 = 930
    SIMULATE_DPC_230 = 230
    SIMULATE_SPC_150 = 150
    SIMULATE_SPC_150N = 151
    SIMULATE_SPC_150NX = 152
    SIMULATE_SPC_150NXX = 153
    SIMULATE_SPC_130EM = 131
    SIMULATE_SPC_130EMN = 132
    SIMULATE_SPC_130IN = 135
    SIMULATE_SPC_130INX = 136
    SIMULATE_SPC_130INXX = 137
    SIMULATE_SPC_160 = 160
    SIMULATE_SPC_160X = 161
    SIMULATE_SPC_160PCIE = 162
    SIMULATE_SPC_180N = 180
    SIMULATE_SPC_180NX = 181
    SIMULATE_SPC_180NXX = 182
    SIMULATE_SPC_QC_104 = 104
    SIMULATE_SPC_QC_004 = 4

class InitStatus(enum.Enum):
    """Enum for the initialization status of an SPC module."""
    INIT_OK = 0
    INIT_NOT_DONE = -1
    INIT_WRONG_EEP_CHKSUM = -2
    INIT_WRONG_MOD_ID = -3
    INIT_HARD_TEST_ERR = -4
    INIT_CANT_OPEN_PCI_CARD = -5
    INIT_MOD_IN_USE = -6
    INIT_WINDRVR_VER = -7
    INIT_WRONG_LICENSE = -8
    INIT_FIRMWARE_VER = -9
    INIT_NO_LICENSE = -10
    INIT_LICENSE_NOT_VALID = -11
    INIT_LICENSE_DATE_EXP = -12
    INIT_CANT_OPEN_USB_CARD = -13
    INIT_XILINX_ERR = -100

    def message(self):
        """Return a human-readable message for the initialization status."""
        messages = {
            self.INIT_OK: "Initialized",
            self.INIT_NOT_DONE: "Initialization not requested",
            self.INIT_WRONG_EEP_CHKSUM: "Incorrect EEPROM checksum",
            self.INIT_WRONG_MOD_ID: "Incorrect module identification code",
            self.INIT_HARD_TEST_ERR: "Hardware test failed",
            self.INIT_CANT_OPEN_PCI_CARD: "Cannot open PCI card",
            self.INIT_MOD_IN_USE: "Module in use elsewhere",
            self.INIT_WINDRVR_VER: "Incorrect WinDriver version",
            self.INIT_WRONG_LICENSE: "Corrupted license key",
            self.INIT_FIRMWARE_VER: "Incorrect firmware version",
            self.INIT_NO_LICENSE: "License key not found",
            self.INIT_LICENSE_NOT_VALID: "License key not applicable",
            self.INIT_LICENSE_DATE_EXP: "License key expired",
            self.INIT_CANT_OPEN_USB_CARD: "Cannot open USB card",
            self.INIT_XILINX_ERR: "FPGA configuration error"
        }
        return messages.get(self, "Unknown error")


class ParID(enum.Enum):
    """Enum of SPC parameter ids."""
    CFD_LIMIT_LOW = 0
    CFD_LIMIT_HIGH = 1
    CFD_ZC_LEVEL = 2
    CFD_HOLDOFF = 3
    SYNC_ZC_LEVEL = 4
    SYNC_FREQ_DIV = 5
    SYNC_HOLDOFF = 6
    SYNC_THRESHOLD = 7
    TAC_RANGE = 8
    TAC_GAIN = 9
    TAC_OFFSET = 10
    TAC_LIMIT_LOW = 11
    TAC_LIMIT_HIGH = 12
    ADC_RESOLUTION = 13
    EXT_LATCH_DELAY = 14
    COLLECT_TIME = 15
    DISPLAY_TIME = 16
    REPEAT_TIME = 17
    STOP_ON_TIME = 18
    STOP_ON_OVFL = 19
    DITHER_RANGE = 20
    COUNT_INCR = 21
    MEM_BANK = 22
    DEAD_TIME_COMP = 23
    SCAN_CONTROL = 24
    ROUTING_MODE = 25
    TAC_ENABLE_HOLD = 26
    MODE = 27
    SCAN_SIZE_X = 28
    SCAN_SIZE_Y = 29
    SCAN_ROUT_X = 30
    SCAN_ROUT_Y = 31
    SCAN_POLARITY = 32
    SCAN_FLYBACK = 33
    SCAN_BORDERS = 34
    PIXEL_TIME = 35
    PIXEL_CLOCK = 36
    LINE_COMPRESSION = 37
    TRIGGER = 38
    EXT_PIXCLK_DIV = 39
    RATE_COUNT_TIME = 40
    MACRO_TIME_CLK = 41
    ADD_SELECT = 42
    ADC_ZOOM = 43
    XY_GAIN = 44
    IMG_SIZE_X = 45
    IMG_SIZE_Y = 46
    IMG_ROUT_X = 47
    IMG_ROUT_Y = 48
    MASTER_CLOCK = 49
    ADC_SAMPLE_DELAY = 50
    DETECTOR_TYPE = 51
    TDC_CONTROL = 52
    CHAN_ENABLE = 53
    CHAN_SLOPE = 54
    CHAN_SPEC_NO = 55
    TDC_OFFSET1 = 56
    TDC_OFFSET2 = 57
    TDC_OFFSET3 = 58
    TDC_OFFSET4 = 59

class SPCMError(Exception):
    """Exception raised for errors in the SPCM DLL."""
    pass

class BHSPC:
    """Direct wrapper for the BH SPC DLL."""

    def __init__(self):
        """Initialize the wrapper."""
        self.dll = None
        self.dll_path = self._find_spcm_dll()
        if self.dll_path:
            self.dll = ctypes.WinDLL(self.dll_path)
            self._setup_function_prototypes()
        self.active_cards = []  # List of active card module numbers

    def _find_spcm_dll(self):
        """Find the SPCM DLL."""
        # Try common installation paths
        paths = [
            "C:\\Program Files\\Becker-Hickl\\SPCM\\DLL\\spcm64.dll",
            "C:\\Program Files (x86)\\BH\\SPCM\\DLL\\spcm64.dll",
        ]

        for path in paths:
            if os.path.exists(path):
                return path

        return None

    def _setup_function_prototypes(self):
        """Set up the function prototypes for the DLL."""
        if not self.dll:
            return

        # SPC_init
        self.dll.SPC_init.argtypes = [ctypes.c_char_p]
        self.dll.SPC_init.restype = ctypes.c_short

        # SPC_close
        self.dll.SPC_close.argtypes = []
        self.dll.SPC_close.restype = ctypes.c_short

        # SPC_get_init_status
        self.dll.SPC_get_init_status.argtypes = [ctypes.c_short]
        self.dll.SPC_get_init_status.restype = ctypes.c_short

        # SPC_set_parameter
        self.dll.SPC_set_parameter.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.c_float]
        self.dll.SPC_set_parameter.restype = ctypes.c_short

        # SPC_get_parameter
        self.dll.SPC_get_parameter.argtypes = [ctypes.c_short, ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.dll.SPC_get_parameter.restype = ctypes.c_short

        # SPC_start_measurement
        self.dll.SPC_start_measurement.argtypes = [ctypes.c_short]
        self.dll.SPC_start_measurement.restype = ctypes.c_short

        # SPC_stop_measurement
        self.dll.SPC_stop_measurement.argtypes = [ctypes.c_short]
        self.dll.SPC_stop_measurement.restype = ctypes.c_short

        # SPC_read_fifo
        self.dll.SPC_read_fifo.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ushort)]
        self.dll.SPC_read_fifo.restype = ctypes.c_short

        # SPC_get_fifo_usage
        self.dll.SPC_get_fifo_usage.argtypes = [ctypes.c_short, ctypes.POINTER(ctypes.c_float)]
        self.dll.SPC_get_fifo_usage.restype = ctypes.c_short

    def init(self, ini_file):
        """Initialize the SPCM DLL."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        if isinstance(ini_file, str):
            ini_file = ini_file.encode()

        result = self.dll.SPC_init(ini_file)
        if result < 0:
            raise SPCMError(f"Error initializing SPCM DLL: {result}")

    def close(self):
        """Close the SPCM DLL."""
        if not self.dll:
            return

        self.dll.SPC_close()

    def get_init_status(self, mod_no):
        """Get the initialization status of a module."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        result = self.dll.SPC_get_init_status(mod_no)
        return InitStatus(result)

    def set_parameter(self, mod_no, par_id, value):
        """Set a parameter for a module."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        if isinstance(par_id, ParID):
            par_id = par_id.value

        result = self.dll.SPC_set_parameter(mod_no, par_id, value)
        if result < 0:
            raise SPCMError(f"Error setting parameter: {result}")

    def get_parameter(self, mod_no, par_id):
        """Get a parameter from a module."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        if isinstance(par_id, ParID):
            par_id = par_id.value

        value = ctypes.c_float()
        result = self.dll.SPC_get_parameter(mod_no, par_id, ctypes.byref(value))
        if result < 0:
            raise SPCMError(f"Error getting parameter: {result}")

        return value.value

    def start_measurement(self, mod_no):
        """Start a measurement."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        result = self.dll.SPC_start_measurement(mod_no)
        if result < 0:
            raise SPCMError(f"Error starting measurement: {result}")

    def stop_measurement(self, mod_no):
        """Stop a measurement."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        result = self.dll.SPC_stop_measurement(mod_no)
        if result < 0:
            raise SPCMError(f"Error stopping measurement: {result}")

    def read_fifo(self, mod_no, max_words):
        """Read data from the FIFO buffer."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        count = ctypes.c_ulong(max_words)
        data = (ctypes.c_ushort * max_words)()

        result = self.dll.SPC_read_fifo(mod_no, ctypes.byref(count), data)
        if result < 0:
            raise SPCMError(f"Error reading FIFO: {result}")

        # Convert to array.array
        data_array = array.array('H')
        data_array.extend(data[:count.value])

        return data_array

    def get_fifo_usage(self, mod_no):
        """Get the FIFO usage."""
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        usage = ctypes.c_float()
        result = self.dll.SPC_get_fifo_usage(mod_no, ctypes.byref(usage))
        if result < 0:
            raise SPCMError(f"Error getting FIFO usage: {result}")

        return usage.value * 100  # Convert to percentage

    def detect_cards(self, max_cards=8):
        """Detect available SPC cards.

        Args:
            max_cards (int): Maximum number of cards to check.

        Returns:
            list: List of dictionaries with card information.
        """
        if not self.dll:
            raise SPCMError("SPCM DLL not found")

        cards = []

        # Create a temporary ini file for initialization
        with ini_file(minimal_spcm_ini(DLLOperationMode.HARDWARE)) as ini:
            self.init(ini)

            # Check each possible module number
            for mod_no in range(max_cards):
                init_status = self.get_init_status(mod_no)

                # If the card is initialized successfully, add it to the list
                if init_status == InitStatus.INIT_OK:
                    card_info = {
                        'module_number': mod_no,
                        'status': init_status,
                        'active': mod_no in self.active_cards
                    }
                    cards.append(card_info)
                # If we get a specific error that indicates no card, skip
                elif init_status in [InitStatus.INIT_WRONG_MOD_ID, InitStatus.INIT_CANT_OPEN_PCI_CARD, InitStatus.INIT_CANT_OPEN_USB_CARD]:
                    continue
                # For other errors, add the card with the error status
                else:
                    card_info = {
                        'module_number': mod_no,
                        'status': init_status,
                        'active': False
                    }
                    cards.append(card_info)

        return cards

    def set_active_cards(self, card_numbers):
        """Set the active cards.

        Args:
            card_numbers (list): List of module numbers to set as active.
        """
        self.active_cards = card_numbers

    def get_active_cards(self):
        """Get the active cards.

        Returns:
            list: List of active card module numbers.
        """
        return self.active_cards


# Helper functions for INI file creation
def minimal_spcm_ini(mode):
    """Return the text for a minimal .ini file for use with SPCM DLL."""
    if isinstance(mode, DLLOperationMode):
        mode = mode.value

    return f"""; SPCM
[spc_base]
simulation = {mode}
[spc_module]
"""

@contextlib.contextmanager
def ini_file(text):
    """Context manager providing a temporary .ini file with the given text."""
    with tempfile.TemporaryDirectory() as dirname:
        ininame = os.path.join(dirname, "spcm.ini")
        with open(ininame, mode="w") as inifile:
            inifile.write(text)
        yield ininame

class BHSPCDevice(TCSPCDeviceABC):
    """BH SPC device wrapper implementing the TCSPC device interface."""

    def __init__(self):
        """Initialize the BH SPC device wrapper."""
        super().__init__()
        self.bh_spc = BHSPC()
        self.initialized = False
        self.measurement_running = False
        self.active_cards = []
        self.available_cards = []

    def log_message(self, message):
        """Log a message and emit the message_logged signal."""
        print(f"BH SPC: {message}")
        self.message_logged.emit(message)

    def detect_cards(self, simulation=False):
        """Detect available BH SPC cards."""
        try:
            if simulation:
                # In simulation mode, create a single simulated card
                self.available_cards = [{
                    'module_number': 0,
                    'status': InitStatus.INIT_OK,
                    'active': True
                }]
            else:
                # Detect real hardware
                self.available_cards = self.bh_spc.detect_cards()

            return self.available_cards
        except Exception as e:
            self.log_message(f"Error detecting BH SPC cards: {e}")
            return []

    def set_active_cards(self, card_numbers):
        """Set the active cards."""
        self.active_cards = card_numbers
        self.bh_spc.set_active_cards(card_numbers)

    def get_active_cards(self):
        """Get the active cards."""
        return self.active_cards

    def initialize(self, simulation=True):
        """Initialize the BH SPC device."""
        try:
            # Detect available cards
            self.detect_cards(simulation)

            # If no cards are available, return False
            if not self.available_cards:
                self.log_message("No BH SPC cards detected")
                return False

            # If no cards are active, set the first available card as active
            if not self.active_cards and self.available_cards:
                self.set_active_cards([self.available_cards[0]['module_number']])

            # Create a temporary ini file for initialization
            mode = DLLOperationMode.SIMULATE_SPC_830 if simulation else DLLOperationMode.HARDWARE
            with ini_file(minimal_spcm_ini(mode)) as ini:
                self.bh_spc.init(ini)

            # Check if initialization was successful for all active cards
            success = True
            for mod_no in self.active_cards:
                init_status = self.bh_spc.get_init_status(mod_no)
                if init_status != InitStatus.INIT_OK:
                    self.log_message(f"Initialization failed for module {mod_no}: {init_status.message()}")
                    success = False
                else:
                    # Set FIFO mode
                    self.bh_spc.set_parameter(mod_no, ParID.MODE, 1)
                    # Disable stop_on_time
                    self.bh_spc.set_parameter(mod_no, ParID.STOP_ON_TIME, 0)

            if success:
                self.initialized = True
                self.log_message("BH SPC device initialized successfully")
                return True
            else:
                self.log_message("BH SPC initialization failed for one or more cards")
                return False
        except Exception as e:
            self.log_message(f"Error initializing BH SPC device: {e}")
            return False

    def start_measurement(self):
        """Start measurement."""
        if not self.initialized:
            self.log_message("BH SPC device not initialized")
            return False

        try:
            success = True
            for mod_no in self.active_cards:
                try:
                    self.bh_spc.start_measurement(mod_no)
                    self.log_message(f"Measurement started on BH SPC module {mod_no}")
                except Exception as e:
                    self.log_message(f"Error starting measurement on module {mod_no}: {e}")
                    success = False

            if success:
                self.measurement_running = True
                return True
            else:
                self.log_message("BH SPC measurement failed to start on one or more cards")
                return False
        except Exception as e:
            self.log_message(f"Error starting BH SPC measurement: {e}")
            return False

    def stop_measurement(self):
        """Stop measurement."""
        if not self.initialized or not self.measurement_running:
            return False

        try:
            success = True
            for mod_no in self.active_cards:
                try:
                    self.bh_spc.stop_measurement(mod_no)
                    self.log_message(f"Measurement stopped on BH SPC module {mod_no}")
                except Exception as e:
                    self.log_message(f"Error stopping measurement on module {mod_no}: {e}")
                    success = False

            self.measurement_running = False

            if success:
                self.log_message("BH SPC measurement stopped on all cards")
                return True
            else:
                self.log_message("BH SPC measurement failed to stop on one or more cards")
                return False
        except Exception as e:
            self.log_message(f"Error stopping BH SPC measurement: {e}")
            return False

    def read_fifo(self, max_words=32768):
        """Read data from FIFO."""
        if not self.initialized or not self.measurement_running:
            return np.array([], dtype=np.uint32)

        try:
            all_records = []

            for mod_no in self.active_cards:
                try:
                    buf = self.bh_spc.read_fifo(mod_no, max_words)
                    if len(buf):
                        # Convert to 32-bit records
                        records = np.array(buf, dtype=np.uint16).view(np.uint32)

                        # Add module number to bits 28-30 for routing
                        if mod_no > 0:
                            records = np.bitwise_and(records, ~(0b111 << 28))
                            records = np.bitwise_or(records, (mod_no & 0b111) << 28)

                        all_records.append(records)
                except Exception as e:
                    self.log_message(f"Error reading FIFO from BH SPC module {mod_no}: {e}")

            if all_records:
                combined_records = np.concatenate(all_records)
                return combined_records
            else:
                return np.array([], dtype=np.uint32)
        except Exception as e:
            self.log_message(f"Error reading BH SPC FIFO: {e}")
            return np.array([], dtype=np.uint32)

    def get_fifo_usage(self):
        """Get FIFO usage."""
        if not self.initialized:
            return {mod_no: -1 for mod_no in self.active_cards}

        try:
            usage = {}
            for mod_no in self.active_cards:
                try:
                    usage[mod_no] = self.bh_spc.get_fifo_usage(mod_no)
                except Exception as e:
                    self.log_message(f"Error getting FIFO usage for BH SPC module {mod_no}: {e}")
                    usage[mod_no] = -1
            return usage
        except Exception as e:
            self.log_message(f"Error getting BH SPC FIFO usage: {e}")
            return {mod_no: -1 for mod_no in self.active_cards}

    def close(self):
        """Close the device."""
        try:
            if self.measurement_running:
                self.stop_measurement()
            self.bh_spc.close()
            self.initialized = False
            self.active_cards = []
            self.available_cards = []
            self.log_message("BH SPC device closed")
        except Exception as e:
            self.log_message(f"Error closing BH SPC device: {e}")


# Define a wrapper class that is manufacturer-agnostic
class TCSPCDevice(QObject):
    """A manufacturer-agnostic wrapper for TCSPC devices."""

    # Signal emitted when a message is logged
    message_logged = Signal(str)

    def __init__(self, device_type="BH_SPC"):
        """Initialize the TCSPC device.

        Args:
            device_type (str): The type of TCSPC device. Supported: "BH_SPC", "PICOQUANT".
        """
        super().__init__()
        self.device_type = device_type

        # Import the appropriate device wrapper
        if self.device_type == "BH_SPC":
            self.device = BHSPCDevice()
        elif self.device_type == "PICOQUANT":
            try:
                from ..picoquant_wrapper import PicoQuantDevice as PQDevice
                self.device = PQDevice()
            except ImportError:
                raise RuntimeError("PicoQuant wrapper not available")
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

        # Connect device signals
        if self.device:
            self.device.message_logged.connect(self._forward_message)

    def _forward_message(self, message):
        """Forward messages from the device to our signal."""
        self.message_logged.emit(message)

    def log_message(self, message):
        """Log a message and emit the message_logged signal."""
        print(message)
        self.message_logged.emit(message)

    def detect_cards(self, simulation=False):
        """Detect available TCSPC cards/devices."""
        if self.device:
            return self.device.detect_cards(simulation)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return []

    def set_active_cards(self, card_numbers):
        """Set the active cards/devices."""
        if self.device:
            return self.device.set_active_cards(card_numbers)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def get_active_cards(self):
        """Get the active cards/devices."""
        if self.device:
            return self.device.get_active_cards()
        else:
            return []

    def initialize(self, simulation=True):
        """Initialize the TCSPC device."""
        if self.device:
            return self.device.initialize(simulation)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def start_measurement(self):
        """Start measurement."""
        if self.device:
            return self.device.start_measurement()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def stop_measurement(self):
        """Stop measurement."""
        if self.device:
            return self.device.stop_measurement()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def read_fifo(self, max_words=32768):
        """Read data from FIFO."""
        if self.device:
            return self.device.read_fifo(max_words)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return np.array([], dtype=np.uint32)

    def get_fifo_usage(self):
        """Get FIFO usage."""
        if self.device:
            return self.device.get_fifo_usage()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return {}

    def close(self):
        """Close the TCSPC device."""
        if self.device:
            self.device.close()

    @property
    def initialized(self):
        """Get the initialization status from the underlying device."""
        if self.device:
            return self.device.initialized
        return False
