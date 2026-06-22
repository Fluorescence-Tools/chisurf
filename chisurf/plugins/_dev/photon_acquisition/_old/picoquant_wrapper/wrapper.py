"""
PicoQuant Wrapper Implementation

This module provides a direct wrapper for PicoQuant TCSPC hardware by duplicating snAPI functionality.
It includes classes and functions for initializing and controlling
PicoQuant devices, as well as for acquiring data from them.
"""

import os
import sys
import json
import ctypes as ct
import numpy as np
import time
import typing
import traceback
from abc import ABC, abstractmethod

from ..photon_sources import TCSPCDeviceABC

# Copy constants from snAPI
class MeasMode:
    T2 = 0
    T3 = 1
    Histogram = 2

class RefSource:
    Internal = 0
    External = 1

class LogLevel:
    Api = 0
    Config = 1
    Device = 2
    DataFile = 3
    Manipulators = 4

class TrigMode:
    Edge = 0
    CFD = 1

class Color:
    Red = "\033[91m"
    End = "\033[0m"

# Duplicate snAPI DLL wrapper
class DuplicatedSnAPI:
    """Duplicated snAPI class - DLL wrapper for PicoQuant devices."""

    is_win = sys.platform.startswith("win")
    is_linux = sys.platform.startswith("linux")

    # Load the DLL - look for it in the same directory as this wrapper
    dll = None
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'snAPI64.dll'))
    if is_win and os.path.exists(dll_path):
        try:
            dll = ct.WinDLL(dll_path)
            # Set up function signatures
            dll.initAPI.argtypes = [ct.c_char_p]
            dll.initAPI.restype = ct.c_int
            dll.exitAPI.argtypes = []
            dll.exitAPI.restype = None
            dll.getDeviceIDs.argtypes = [ct.POINTER(ct.c_char * 8192)]
            dll.getDeviceIDs.restype = ct.c_int
            dll.getDevice.argtypes = [ct.c_char_p]
            dll.getDevice.restype = ct.c_int
            dll.initDevice.argtypes = [ct.c_int, ct.c_int]
            dll.initDevice.restype = ct.c_int
            dll.closeDevice.argtypes = [ct.c_int]
            dll.closeDevice.restype = None
            dll.getDeviceConfig.argtypes = [ct.POINTER(ct.c_char * 65535)]
            dll.getDeviceConfig.restype = ct.c_int
            dll.stopMeasure.argtypes = []
            dll.stopMeasure.restype = None
            dll.rawMeasure.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64), ct.c_uint64, ct.POINTER(ct.c_bool)]
            dll.rawMeasure.restype = ct.c_int
            dll.rawStartBlock.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_bool)]
            dll.rawStartBlock.restype = ct.c_int
            dll.rawGetBlock.argtypes = [ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64)]
            dll.rawGetBlock.restype = ct.c_int
        except:
            dll = None
    elif is_linux:
        dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libsnAPI4Linux.so'))
        if os.path.exists(dll_path):
            try:
                ct.cdll.LoadLibrary(dll_path)
                dll = ct.CDLL(dll_path, mode=ct.RTLD_GLOBAL)
                # Set up function signatures for Linux
                dll.initAPI.argtypes = [ct.c_char_p]
                dll.initAPI.restype = ct.c_int
                dll.exitAPI.argtypes = []
                dll.exitAPI.restype = None
                dll.getDeviceIDs.argtypes = [ct.POINTER(ct.c_char * 8192)]
                dll.getDeviceIDs.restype = ct.c_int
                dll.getDevice.argtypes = [ct.c_char_p]
                dll.getDevice.restype = ct.c_int
                dll.initDevice.argtypes = [ct.c_int, ct.c_int]
                dll.initDevice.restype = ct.c_int
                dll.closeDevice.argtypes = [ct.c_int]
                dll.closeDevice.restype = None
                dll.getDeviceConfig.argtypes = [ct.POINTER(ct.c_char * 65535)]
                dll.getDeviceConfig.restype = ct.c_int
                dll.stopMeasure.argtypes = []
                dll.stopMeasure.restype = None
                dll.rawMeasure.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64), ct.c_uint64, ct.POINTER(ct.c_bool)]
                dll.rawMeasure.restype = ct.c_int
                dll.rawStartBlock.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_uint32), ct.c_uint64, ct.POINTER(ct.c_bool)]
                dll.rawStartBlock.restype = ct.c_int
                dll.rawGetBlock.argtypes = [ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64)]
                dll.rawGetBlock.restype = ct.c_int
            except:
                dll = None

    deviceIDs = []
    deviceConfig = ()
    measDescription = ()

    def __init__(self, systemIni: typing.Union[str, None] = None):
        if not self.dll:
            raise RuntimeError("snAPI DLL not found")

        if systemIni is None:
            if self.is_win:
                systemIni = "\\".join(__file__.split("\\")[:-1])+'\\system.ini'
            if self.is_linux:
                systemIni = "/".join(__file__.split("/")[:-1])+'/system_linux.ini'

        self.raw = DuplicatedRaw(self)
        self.initAPI(systemIni)

    def __del__(self):
        self.exitAPI()

    def logPrint(self,*args, **kwargs):
        """Log print function."""
        summarized_args = " ".join(map(str, args))
        summarized_kwargs = " ".join([f"{key}={value}" for key, value in kwargs.items()])
        if hasattr(self.dll, 'logExternal'):
            self.dll.logExternal.argtypes = [ct.c_char_p]
            if summarized_args and summarized_kwargs:
                self.dll.logExternal(f"{summarized_args} {summarized_kwargs}".encode('utf-8'))
            elif summarized_args:
                self.dll.logExternal(f"{summarized_args}".encode('utf-8'))
            elif summarized_kwargs:
                self.dll.logExternal(f"{summarized_kwargs}".encode('utf-8'))

    def initAPI(self, systemIni: typing.Optional[str] = "system.ini"):
        """Initialize the API."""
        SBuf = systemIni.encode('utf-8')
        ok = self.dll.initAPI(SBuf)
        self.getDeviceConfig()
        return ok

    def exitAPI(self):
        """Exit the API."""
        if self.dll:
            self.dll.exitAPI()

    def getDeviceIDs(self):
        """Get device IDs."""
        devIDs = (ct.c_char * 8192)()
        found = self.dll.getDeviceIDs(devIDs)
        devIDs = str(devIDs, "utf-8").replace('\x00','')
        if found:
            self.deviceIDs = json.loads(devIDs)
            return True
        else:
            self.logPrint(devIDs)
            return False

    def getDevice(self, *dev):
        """Get a device."""
        if not dev: # no device parameter
            name = ""
            SBuf = name.ljust(8, '\0').encode('utf-8')
            if self.dll.getDevice(SBuf):
                return self.getDeviceConfig()
            else:
                self.logPrint(f"{Color.Red}Device not found!")

        elif (len(dev) == 1 and isinstance(dev[0], str)): # name of device
            name = dev[0]
            SBuf = name.ljust(8, '\0').encode('utf-8')
            if len(self.deviceIDs) == 0:
                self.getDeviceIDs()

            if self.dll.getDevice(SBuf):
                return self.getDeviceConfig()
            else:
                self.logPrint(Color.Red + "Device \"" +name+ "\" not found!")

        elif len(dev) == 1 and isinstance(dev[0], int): # index of device
            if len(self.deviceIDs) == 0:
                self.getDeviceIDs()

            if(dev[0] >= 0 and dev[0] < len(self.deviceIDs)) :
                name = self.deviceIDs[dev[0]]
                if(self.deviceIDs[dev[0]] != ""):
                    SBuf = self.deviceIDs[dev[0]].encode('utf-8')
                    if self.dll.getDevice(SBuf):
                        return self.getDeviceConfig()
                    else:
                        self.logPrint(f"{Color.Red}Error getting Device @ index: {dev[0]} \"{name}\"!")
                else:
                    self.logPrint(f"{Color.Red}No device at index: {dev[0]}")
            else:
                self.logPrint(f"{Color.Red}Device index: {dev[0]} out of bounds!")
        else:
            self.logPrint(f"{Color.Red}Invalid device parameter @ getDevice(): {dev[0]}")

        return False

    def initDevice(self, measMode: typing.Optional[MeasMode] = MeasMode.T2,
                   refSrc: typing.Optional[RefSource] = RefSource.Internal):
        """Initialize device."""
        if self.dll.initDevice(measMode.value, refSrc.value):
            ok = self.getDeviceConfig()
            return ok
        return False

    def closeDevice(self, allDevices: typing.Optional[bool] = True):
        """Close device."""
        self.dll.closeDevice(allDevices)

    def getDeviceConfig(self):
        """Get device config."""
        conf = (ct.c_char * 65535)()
        ok = self.dll.getDeviceConfig(conf)
        conf = str(conf, "utf-8").replace('\x00','')
        if ok:
            self.deviceConfig = json.loads(conf)
            return True
        else:
            self.logPrint(conf)
            return False

    def _stopMeasure(self):
        """Stop measurement (private)."""
        self.dll.stopMeasure()


class PicoQuantAPIError(Exception):
    """Exception for PicoQuant API errors."""
    pass


class DuplicatedRaw:
    """Duplicated Raw class from snAPI."""

    def __init__(self, parent):
        self.parent = parent
        self.data = ct.ARRAY(ct.c_uint32, 0)()
        self.finished = ct.pointer(ct.c_bool(False))
        self.idx = ct.pointer(ct.c_uint64(0))

    def measure(self, acqTime: typing.Optional[int] = 1000, size: typing.Optional[int] = 134217728,
                waitFinished: typing.Optional[bool] = True, savePTU: typing.Optional[bool] = False):
        """Measure raw data."""
        self.data = ct.ARRAY(ct.c_uint32, size)()
        if(self.parent.deviceConfig["MeasMode"] == MeasMode.Histogram.value):
            name = "Histogram"
            self.parent.logPrint(Color.Red + "measurement is not supported for Raw class in MeasMode:", name)
            return False
        self.parent.dll.rawMeasure.restype = ct.c_bool
        return self.parent.dll.rawMeasure(acqTime, waitFinished, savePTU, ct.byref(self.data), self.idx, ct.c_uint64(size), self.finished)

    def startBlock(self, acqTime: int = 1000, size: int = 134217728, savePTU: typing.Optional[bool] = False):
        """Start block measurement."""
        self.storeData = ct.ARRAY(ct.c_uint32, size)()
        self.data = ct.ARRAY(ct.c_uint32, size)()
        if(self.parent.deviceConfig["MeasMode"] == MeasMode.Histogram.value):
            name = "Histogram"
            self.parent.logPrint(Color.Red + "startBlock is not supported for Raw class in MeasMode:", name)
            return False
        self.parent.dll.rawStartBlock.restype = ct.c_bool
        return self.parent.dll.rawStartBlock(acqTime, savePTU, ct.byref(self.storeData), ct.c_uint64(size), self.finished)

    def getBlock(self):
        """Get block data."""
        size = ct.pointer(ct.c_uint64(0))
        if(self.parent.deviceConfig["MeasMode"] == MeasMode.Histogram.value):
            name = "Histogram"
            self.parent.logPrint(Color.Red + "getBlock is not supported for Raw class in MeasMode:", name)
            self.idx.contents.value = 0
        else:
            self.parent.dll.rawGetBlock(ct.byref(self.data), size)
            self.idx.contents.value = size.contents.value
        return self.getData()

    def getData(self, numRead: typing.Optional[int] = None):
        """Get data."""
        if not numRead:
            numRead = self.numRead()

        if(self.parent.deviceConfig["MeasMode"] == MeasMode.Histogram.value):
            name = "Histogram"
            self.parent.logPrint(Color.Red + "getData is not supported for Raw class in MeasMode:", name)
            return []
        return np.lib.stride_tricks.as_strided(self.data, shape=(1, numRead),
            strides=(ct.sizeof(self.data._type_) * numRead, ct.sizeof(self.data._type_)))[0]

    def numRead(self):
        """Get number of records read."""
        return self.idx.contents.value

    def isFinished(self):
        """Check if measurement is finished."""
        return self.finished.contents.value

    def stopMeasure(self):
        """Stop measurement."""
        self.parent._stopMeasure()

    def isSpecial(self, data: int):
        """Check if data record is special."""
        return ((0x80000000 & data) != 0)

    def channel(self, data: int):
        """Get channel from data record."""
        return ((data >> 25) & 0x0000003F) + 1


class PicoQuantAPIError(Exception):
    """Exception for PicoQuant API errors."""
    pass


class PicoQuantAPI:
    """Wrapper for duplicated PicoQuant snAPI."""

    def __init__(self):
        """Initialize the PicoQuant API wrapper."""
        try:
            self.sn = DuplicatedSnAPI()
            self.initialized = True
            self.measurement_running = False
            self.data_buffer = []
            self.max_buffer_size = 1000000  # Maximum buffer size for data accumulation
        except Exception as e:
            self.sn = None
            self.initialized = False
            raise PicoQuantAPIError(f"Failed to initialize duplicated snAPI: {e}")

    def init(self, config_path=None):
        """Initialize the duplicated snAPI.

        Args:
            config_path (str): Path to configuration file (optional).
        """
        if not self.initialized:
            raise PicoQuantAPIError("Duplicated snAPI not available")

        try:
            # Note: duplicated snAPI doesn't have loadIniConfig, so we'll skip this for now
            pass
        except Exception as e:
            raise PicoQuantAPIError(f"Failed to initialize duplicated snAPI: {e}")

    def close(self):
        """Close the duplicated snAPI."""
        if self.sn and self.initialized:
            try:
                if self.measurement_running:
                    self.stop_measurement()
                self.sn.closeDevice()
                self.sn.exitAPI()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.sn = None
                self.initialized = False

    def detect_devices(self):
        """Detect available PicoQuant devices.

        Returns:
            list: List of dictionaries with device information.
        """
        if not self.initialized:
            return []

        devices = []
        try:
            self.sn.getDeviceIDs()
            device_ids = self.sn.deviceIDs

            for i, device_id in enumerate(device_ids):
                if device_id:  # Non-empty device ID
                    device_info = {
                        'module_number': i,
                        'device_id': device_id,
                        'status': 'available',
                        'active': i == 0  # Default to first device as active
                    }
                    devices.append(device_info)

            # If no devices found, create a simulated device for testing
            if not devices:
                devices = [{
                    'module_number': 0,
                    'device_id': 'SIMULATED',
                    'status': 'simulated',
                    'active': True
                }]

        except Exception as e:
            print(f"Error detecting PicoQuant devices: {e}")
            # Return simulated device for development
            devices = [{
                'module_number': 0,
                'device_id': 'SIMULATED',
                'status': 'simulated',
                'active': True
            }]

        return devices

    def set_active_devices(self, device_indices):
        """Set the active devices.

        Args:
            device_indices (list): List of device indices to set as active.
        """
        # For PicoQuant, we can only work with one device at a time
        # Store the active device index
        self.active_devices = device_indices[:1] if device_indices else [0]

    def get_active_devices(self):
        """Get the active devices.

        Returns:
            list: List of active device indices.
        """
        return getattr(self, 'active_devices', [0])

    def initialize_device(self, device_index=0, simulation=False):
        """Initialize a specific device.

        Args:
            device_index (int): Device index to initialize.
            simulation (bool): Whether to use simulation mode.

        Returns:
            bool: True if successful.
        """
        if not self.initialized:
            return False

        try:
            if simulation:
                # For simulation, we'll use a mock device
                return True
            else:
                # Try to get the device
                devices = self.detect_devices()
                if device_index < len(devices):
                    device_id = devices[device_index]['device_id']
                    if device_id != 'SIMULATED':
                        self.sn.getDevice(device_id)
                        self.sn.initDevice(MeasMode.T3)  # Use T3 mode for time-resolved data
                        return True
                    else:
                        # Simulated device
                        return True
                return False
        except Exception as e:
            print(f"Error initializing PicoQuant device: {e}")
            return False

    def start_measurement(self, device_index=0):
        """Start measurement on a device.

        Args:
            device_index (int): Device index.

        Returns:
            bool: True if successful.
        """
        if not self.initialized:
            return False

        try:
            if not self.measurement_running:
                # Start continuous measurement
                self.sn.raw.startBlock(acqTime=0, size=self.max_buffer_size)
                self.measurement_running = True
                self.data_buffer = []
            return True
        except Exception as e:
            print(f"Error starting PicoQuant measurement: {e}")
            return False

    def stop_measurement(self, device_index=0):
        """Stop measurement on a device.

        Args:
            device_index (int): Device index.

        Returns:
            bool: True if successful.
        """
        if not self.initialized or not self.measurement_running:
            return False

        try:
            self.sn.raw.stopMeasure()
            self.measurement_running = False
            return True
        except Exception as e:
            print(f"Error stopping PicoQuant measurement: {e}")
            return False

    def read_data(self, device_index=0, max_records=10000):
        """Read data from the device.

        Args:
            device_index (int): Device index.
            max_records (int): Maximum number of records to read.

        Returns:
            numpy.ndarray: Array of 32-bit records.
        """
        if not self.initialized or not self.measurement_running:
            return np.array([], dtype=np.uint32)

        try:
            # Get data from duplicated snAPI raw interface
            data = self.sn.raw.getBlock()

            if len(data) > 0:
                # Convert to 32-bit unsigned integers
                # snAPI returns data as numpy array, but we need to ensure it's uint32
                records = data.astype(np.uint32)
                return records
            else:
                return np.array([], dtype=np.uint32)

        except Exception as e:
            print(f"Error reading PicoQuant data: {e}")
            return np.array([], dtype=np.uint32)

    def get_buffer_usage(self, device_index=0):
        """Get buffer usage percentage.

        Args:
            device_index (int): Device index.

        Returns:
            float: Buffer usage percentage (0-100), or -1 if error.
        """
        if not self.initialized:
            return -1

        try:
            # For PicoQuant, we don't have direct FIFO usage, but we can estimate
            # based on available data
            num_records = self.sn.raw.numRead()
            usage = min(100.0, (num_records / self.max_buffer_size) * 100.0)
            return usage
        except Exception:
            return -1


class PicoQuantDevice(TCSPCDeviceABC):
    """PicoQuant device wrapper compatible with the TCSPCDevice interface."""

    def __init__(self):
        """Initialize the PicoQuant device."""
        super().__init__()
        try:
            self.api = PicoQuantAPI()
            self.initialized = True
        except PicoQuantAPIError:
            self.api = None
            self.initialized = False
        self.measurement_running = False
        self.active_devices = [0]  # Default to device 0
        self.available_cards = []
