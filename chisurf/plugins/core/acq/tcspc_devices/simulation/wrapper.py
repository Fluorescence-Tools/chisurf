"""Simulation TCSPC Device

This module provides a simulation wrapper for TCSPC hardware using the Burbulator
single molecule diffusion simulator.
"""

import threading
import queue
import time
import os
import numpy as np
import logging
from datetime import datetime

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QMessageBox

from .burbulator_dll_wrapper import BurbulatorDLL, BurbulatorError
from ..abc import TCSPCDeviceABC


logger = logging.getLogger(__name__)


def make_json_serializable(data):
    """Recursively convert NumPy types/arrays to Python standard types for JSON serialization."""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(x) for x in data]
    elif hasattr(data, 'tolist'):  # Handles NumPy arrays and scalars
        return data.tolist()
    elif hasattr(data, 'item'):  # Handles NumPy scalars
        return data.item()
    else:
        return data


class SimulationSetupDialog:
    """Setup dialog for simulation parameters."""

    def __init__(self, parent=None):
        """Initialize the simulation setup dialog."""
        # For now, just show a simple message
        # TODO: Implement full parameter dialog
        pass

    def exec_(self):
        """Execute the dialog."""
        QMessageBox.information(
            None,
            "Simulation Setup",
            "Simulation device initialized with default parameters.\n"
            "Use the setup dialog to configure molecule species, diffusion, etc."
        )
        return 1  # Accepted


class BurbulatorSimulator:
    """Wrapper for the Burbulator DLL simulation functions."""

    # Focus type name to integer mapping (matches setup_dialog.py)
    FOCUS_TYPE_MAPPING = {
        "3D Gaussian, uniform CEF": 0,
        "3D Gaussian excitation and CEF": 1,
        "Rectangular excitation, uniform CEF": 2,
        "Cylindrical excitation, uniform CEF": 3,
        "Gaussian-Lorentzian excitation, pinhole CEF": 4,
        "Gaussian-Lorentzian excitation, cylindrical CEF": 5
    }

    def __init__(self):
        """Initialize the Burbulator simulator."""
        # High-level BurbulatorDLL wrapper instance
        self.dll = None
        # Backward-compatible alias used throughout this module
        self.burbulator = None
        self.dll_path = None
        self.dll_available = False

        # Prefer the high-level BurbulatorDLL wrapper, which handles
        # cross-platform library loading and uses the validated ctypes
        # prototypes shared with the test_dll_debug pipeline.
        try:
            burb = BurbulatorDLL()
            # Store wrapper in both attributes for compatibility
            self.dll = burb  # Primary handle
            self.burbulator = burb  # Alias used by older code paths
            self.dll_path = burb.path
            self.dll_available = True
            logger.info("SIMULATION: Loaded Burbulator DLL from %s", self.dll_path)
        except Exception as e:
            self.dll = None
            self.burbulator = None
            self.dll_available = False
            logger.error("SIMULATION: Burbulator DLL wrapper not available: %s", e)



    def get_focus_type_integer(self, focus_type):
        """Convert focus type name or integer to integer value for DLL.

        Args:
            focus_type: Either a string name or integer value

        Returns:
            int: Integer value for DLL call
        """
        if isinstance(focus_type, str):
            return self.FOCUS_TYPE_MAPPING.get(focus_type, 0)  # Default to 0 if not found
        elif isinstance(focus_type, int):
            return focus_type
        else:
            return 0  # Default fallback



    def is_available(self):
        """Check if the DLL is available."""
        return self.dll is not None

    def _simulate_with_dll(self, params):
        """Simulate photons using the Burbulator DLL.

        Args:
            params: Dictionary with simulation parameters

        Returns:
            numpy array: Simulated photon data in TCSPC format
        """
        if not self.burbulator or not self.dll_available:
            raise RuntimeError("Burbulator DLL not available")

        # Extract parameters with defaults
        N_species = params.get('N_species', 1)
        # Default to 5 molecules for single-species simulation
        M = params.get('M', [5.0])  # Initial number of molecules
        D = params.get('D', [1.0])    # Diffusion coefficients
        N_channels = params.get('N_channels', 2)
        q = params.get('q', [0.1, 0.1])  # Brightness per species per channel
        q_bg = params.get('q_bg', [0.01, 0.01])  # Background
        k_rad = params.get('k_rad', [0.0] * (N_species * N_species))   # Radiative rates (N_species x N_species)
        k_nrad = params.get('k_nrad', [0.0] * (N_species * N_species))  # Non-radiative rates
        box_xy = params.get('box_xy', 2.0)
        box_z = params.get('box_z', 4.0)
        focus_type = self.get_focus_type_integer(params.get('focus_type', 0))
        focus_param = params.get('focus_param', [0.5, 0.5])
        dt = params.get('dt', 1.0)
        N_ph_max = params.get('N_ph_max', 10000)

        logger.info("SIMULATION: Calling BurbulatorDLL.simulate_ov3 with N_ph_max = %d", N_ph_max)
        try:
            sim = self.burbulator.simulate_ov3(
                Nspecies=N_species,
                M=M,
                D=D,
                Nchannels=N_channels,
                q=q,
                q_bg=q_bg,
                k_rad=k_rad,
                k_nrad=k_nrad,
                box_xy=box_xy,
                box_z=box_z,
                focus_type=focus_type,
                focus_param=focus_param,
                dt=dt,
                N_ph_max=N_ph_max,
                rmt1seed=12345,  # Fixed seed for reproducible results
                rmt2seed=54321,  # Fixed seed for reproducible results
            )
        except BurbulatorError as e:
            logger.error("SIMULATION: simulate_ov3 failed: %s", e)
            return np.array([], dtype=np.uint32)

        N_ph = sim.get("N_ph", 0)
        logger.info(
            "SIMULATION: DLL returned %d photons, T0=%s, Nmolecules=%s",
            N_ph,
            sim.get("T0", 0),
            sim.get("Nmolecules", 0),
        )
        logger.debug("SIMULATION: First few data_T: %s", sim["data_T"][:5] if len(sim["data_T"]) > 0 else "None")
        logger.debug("SIMULATION: First few data_t: %s", sim["data_t"][:5] if len(sim["data_t"]) > 0 else "None")
        logger.debug("SIMULATION: First few data_N: %s", sim["data_N"][:5] if len(sim["data_N"]) > 0 else "None")

        # Convert to BH_SPC format using the high-level conversion helper.
        pulsed_exc = params.get('pulsed_exc', 0)  # CW excitation for simulation
        ch_conversion = params.get('ch_conversion', [8, 0, 9, 1, 10, 2])  # Match C# RunOpenVDll.cs
        N_tac_channels = params.get('N_tac_channels', 4096)
        tac_dt = params.get('tac_dt', 0.004069)  # ns per TAC channel
        laser_period = params.get('laser_period', 13.596)  # ns

        logger.debug(
            "SIMULATION: Converting with pulsed_exc=%s, ch_conversion=%s",
            pulsed_exc,
            ch_conversion,
        )
        logger.debug(
            "SIMULATION: N_tac_channels=%s, tac_dt=%s, laser_period=%s",
            N_tac_channels,
            tac_dt,
            laser_period,
        )
        logger.debug(
            "SIMULATION: data_N range: min=%s, max=%s, unique=%s",
            sim["data_N"].min(),
            sim["data_N"].max(),
            np.unique(sim["data_N"]),
        )

        # Debug: compare with test parameters
        logger.debug("SIMULATION: Parameters comparison with test_dll_debug.py:")
        logger.debug("  pulsed_exc: %s (test: 0)", pulsed_exc)
        logger.debug("  ch_conversion: %s (test: [8, 0, 9, 1, 10, 2])", ch_conversion)
        logger.debug("  N_tac_channels: %s (test: 4096)", N_tac_channels)
        logger.debug("  tac_dt: %s (test: 0.004069)", tac_dt)
        logger.debug("  laser_period: %s (test: 13.596)", laser_period)
        logger.debug("  tw/dt: %s (test: 0.01)", dt)

        try:
            spc_bytes, MT_ov, spc_i = self.burbulator.convert_to_spc132(
                pulsed_exc=pulsed_exc,
                Nchannels=N_channels,
                data_T=sim["data_T"],
                data_t=sim["data_t"],
                data_N=sim["data_N"],
                data_species=sim["data_species"],  # FIXED: was incorrectly set to sim["data_N"]
                data_molecule=sim["data_molecule"],
                tw=dt,
                ch_conversion=ch_conversion,
                N_tac_channels=N_tac_channels,
                tac_dt=tac_dt,
                laser_period=laser_period,
                N_photons=N_ph,
            )
            logger.info("SIMULATION: data2spc132_tac returned %s bytes, MT_ov=%s", spc_i, MT_ov)

            # Optional: write SPC-132 file for debugging if an output folder is provided
            spc_output_path = params.get('spc_output_path')
            if spc_output_path:
                try:
                    output_folder = os.path.abspath(spc_output_path)
                    os.makedirs(output_folder, exist_ok=True)

                    # Write simulation config JSON file to output folder
                    config_file_path = os.path.join(output_folder, "simulation_config.json")
                    try:
                        import json
                        serializable_params = make_json_serializable(params)
                        with open(config_file_path, 'w', encoding='utf-8') as f:
                            json.dump(serializable_params, f, indent=2)
                        logger.info("SIMULATION: Simulation config written to %s", config_file_path)
                    except Exception as e:
                        logger.error("SIMULATION: Failed to write simulation config file: %s", e)

                    # Write multiple files in batches like test_dll_debug.py
                    batch_size = params.get('N_ph_per_file', 1000)  # Photons per file (default to 1000)
                    # Cap batch_size at a reasonable maximum to avoid too few large files
                    max_batch_size = 100000  # Maximum 100k photons per file
                    if batch_size > max_batch_size:
                        batch_size = max_batch_size
                    filenumber = 0
                    photon_start = 0

                    while photon_start < N_ph:
                        photon_end = min(photon_start + batch_size, N_ph)
                        batch_photons = photon_end - photon_start

                        logger.info(
                            "SIMULATION: Writing batch %d: photons %d-%d",
                            filenumber,
                            photon_start,
                            photon_end - 1,
                        )

                        # Extract batch data
                        batch_data_T = sim["data_T"][photon_start:photon_end]
                        batch_data_t = sim["data_t"][photon_start:photon_end]
                        batch_data_N = sim["data_N"][photon_start:photon_end]
                        batch_data_species = sim["data_species"][photon_start:photon_end]
                        batch_data_molecule = sim["data_molecule"][photon_start:photon_end]

                        # Convert batch to SPC
                        spc_bytes_batch, MT_ov_batch, spc_i_batch = self.burbulator.convert_to_spc132(
                            pulsed_exc=pulsed_exc,
                            Nchannels=N_channels,
                            data_T=batch_data_T,
                            data_t=batch_data_t,
                            data_N=batch_data_N,
                            data_species=batch_data_species,
                            data_molecule=batch_data_molecule,
                            tw=dt,
                            ch_conversion=ch_conversion,
                            N_tac_channels=N_tac_channels,
                            tac_dt=tac_dt,
                            laser_period=laser_period,
                            N_photons=batch_photons,
                        )

                        # Write file
                        filename = os.path.join(
                            output_folder,
                            f"m{filenumber:03d}.spc",
                        )
                        logger.info(
                            "SIMULATION: Writing simulated SPC file %s with %d bytes",
                            filename,
                            len(spc_bytes_batch),
                        )
                        self.burbulator.write_spc132_file(filename, spc_bytes_batch)

                        filenumber += 1
                        photon_start = photon_end

                    logger.info("SIMULATION: Wrote %d SPC files to %s", filenumber, output_folder)

                except Exception as e:
                    logger.error(
                        "SIMULATION: Failed to write SPC files in folder %s: %s",
                        spc_output_path,
                        e,
                    )

        except BurbulatorError as e:
            logger.error("SIMULATION: convert_to_spc132 failed: %s", e)
            return np.array([], dtype=np.uint32)

        # Since we're writing files in batches above, we don't need to convert the full dataset
        # Just return an empty array to indicate success (files were written)
        logger.info("SIMULATION: DLL simulation completed, files written in batches")
        return np.array([], dtype=np.uint32)

    def _background_dll_generation(self, params, data_queue, stop_event):
        """Background thread: Generate photons from DLL in batches and push to queue.
        
        Generates data in real-time, writes SPC files, and streams to GUI.
        
        Args:
            params: Simulation parameters dict
            data_queue: Queue to push generated photon batches
            stop_event: Threading event to signal stop
        """
        import time
        
        try:
            N_ph_max = params.get('N_ph_max', 1000000)
            batch_size = 10000  # Generate 10k photons at a time for streaming
            photons_generated = 0
            file_write_buffer = []  # Accumulate for SPC file writing
            file_index = 0
            N_ph_per_file = params.get('N_ph_per_file', 100000)
            spc_output_path = params.get('spc_output_path', '')
            
            logger.info(
                "SIMULATION: Background DLL generation starting: %d photons in %d-photon batches",
                N_ph_max,
                batch_size,
            )
            
            while photons_generated < N_ph_max and not stop_event.is_set():
                # Calculate batch size (might be smaller for last batch)
                current_batch_size = min(batch_size, N_ph_max - photons_generated)
                
                # Create temporary params for this batch
                batch_params = params.copy()
                batch_params['N_ph_max'] = current_batch_size
                
                # Generate batch using DLL
                result = self.dll.simulate_ov3(
                    Nspecies=batch_params.get('N_species', 1),
                    M=batch_params.get('M', [50.0]),
                    D=batch_params.get('D', [3.0]),
                    Nchannels=batch_params.get('N_channels', 2),
                    q=batch_params.get('q', [50.0, 50.0]),
                    q_bg=batch_params.get('q_bg', [0.0, 0.0]),
                    k_rad=batch_params.get('k_rad', [1.0]),
                    k_nrad=batch_params.get('k_nrad', [0.0]),
                    box_xy=batch_params.get('box_xy', 2.0),
                    box_z=batch_params.get('box_z', 4.0),
                    focus_type=batch_params.get('focus_type', 0),
                    focus_param=batch_params.get('focus_param', [0.3, 2.0]),
                    dt=batch_params.get('dt', 0.01),
                    N_ph_max=current_batch_size,
                    rmt1seed=batch_params.get('rmt1seed', 12345) + photons_generated,  # Vary seed
                    rmt2seed=batch_params.get('rmt2seed', 54321) + photons_generated,
                )
                
                # Extract data from result dict
                data_T = result['data_T']
                data_t = result['data_t']
                data_N = result['data_N']
                data_species = result['data_species']
                data_molecule = result['data_molecule']
                n_ph = result['N_ph']
                
                # Convert to SPC format
                spc_bytes, MT_ov, spc_i = self.dll.convert_to_spc132(
                    pulsed_exc=batch_params.get('pulsed_exc', 0),
                    Nchannels=batch_params.get('N_channels', 2),
                    tw=batch_params.get('dt', 0.01),
                    N_tac_channels=batch_params.get('N_tac_channels', 4096),
                    tac_dt=batch_params.get('tac_dt', 0.004069),
                    laser_period=batch_params.get('laser_period', 13.6),
                    spc_data_bytes_per_photon=8,
                    data_T=data_T,
                    data_t=data_t,
                    data_N=data_N,
                    data_species=data_species,
                    data_molecule=data_molecule,
                    ch_conversion=batch_params.get('ch_conversion', [8, 0, 9, 1, 10, 2]),
                    F=None,
                    lookup=None,
                    N_photons=n_ph
                )
                
                # Convert bytes to uint32 array (BH SPC-130 format)
                spc_array = np.frombuffer(spc_bytes, dtype=np.uint32)
                
                # Add to file write buffer (raw SPC-130 records)
                file_write_buffer.append(spc_bytes)
                photons_generated += n_ph
                
                # Write SPC-132 file when buffer reaches N_ph_per_file
                # Each photon record is 4 bytes in SPC-130 format
                buffer_photon_count = sum(len(b) // 4 for b in file_write_buffer)
                if buffer_photon_count >= N_ph_per_file or photons_generated >= N_ph_max:
                    if spc_output_path:
                        try:
                            filename = os.path.join(spc_output_path, f"m{file_index:03d}.spc")
                            # Concatenate all buffered chunks and let the wrapper
                            # prepend a valid BH SPC-132 header for tttrlib.
                            spc_bytes_concat = b"".join(file_write_buffer)
                            # Use default macro_time_clock for now; reader can
                            # override if needed.
                            self.burbulator.write_spc132_file(filename, spc_bytes_concat)
                            file_index += 1
                        except Exception as e:
                            logger.error("SIMULATION: ERROR writing SPC file: %s", e)
                    
                    # Clear buffer
                    file_write_buffer = []
                
                # Pace simulation to real time if requested
                if batch_params.get("real_time_sim", False) and n_ph > 0:
                    # data_T represents macrotime ticks (default 50 ns per tick)
                    macrotime_clock = batch_params.get("macrotime_clock", 50e-9)
                    batch_duration_s = float(data_T[-1]) * macrotime_clock
                    if batch_duration_s > 0:
                        import time as _time
                        _time.sleep(batch_duration_s)

                # Push to queue for GUI (blocks if full)
                if not stop_event.is_set():
                    data_queue.put(spc_array, timeout=5.0)
                    
                    if photons_generated % 50000 == 0:  # Log every 50k
                        logger.info(
                            "SIMULATION: Background generation: %d/%d photons, %d files written",
                            photons_generated,
                            N_ph_max,
                            file_index,
                        )
            
            # Write any remaining data as a final SPC-132 file
            if file_write_buffer and spc_output_path:
                try:
                    filename = os.path.join(spc_output_path, f"m{file_index:03d}.spc")
                    spc_bytes_concat = b"".join(file_write_buffer)
                    self.burbulator.write_spc132_file(filename, spc_bytes_concat)
                    file_index += 1
                except Exception as e:
                    logger.error("SIMULATION: ERROR writing final SPC file: %s", e)
            
            # Signal completion by putting None
            data_queue.put(None)
            logger.info(
                "SIMULATION: Background DLL generation completed: %d photons, %d SPC files written",
                photons_generated,
                file_index,
            )
            
        except Exception as e:
            logger.error("SIMULATION: ERROR in background DLL generation: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            data_queue.put(None)  # Signal error/completion

    def simulate_photons_streaming(self, params, data_queue, stop_event):
        """Start streaming photon generation using Burbulator DLL in background thread.
        
        This is the new architecture: generate in batches, stream to queue.
        
        Args:
            params: Simulation parameters dict
            data_queue: Queue to receive photon batches
            stop_event: Threading event to signal stop
            
        Returns:
            bool: True if background generation started successfully
        """
        if not self.dll or not self.dll_available:
            logger.error("SIMULATION: ERROR: DLL not available for streaming simulation")
            return False
        
        # Test DLL once
        if not hasattr(self, '_dll_tested'):
            logger.info("SIMULATION: Testing DLL with actual function call...")
            try:
                # Note: DLL expects Nspecies, Nchannels (capital N)
                result = self.dll.simulate_ov3(
                    Nspecies=1, M=[1.0], D=[1.0], Nchannels=1,
                    q=[0.1], q_bg=[0.0], k_rad=[0.0], k_nrad=[0.0],
                    box_xy=2.0, box_z=4.0, focus_type=0, focus_param=[0.5, 0.5],
                    dt=1.0, N_ph_max=5, rmt1seed=12345, rmt2seed=54321
                )
                logger.info(
                    "SIMULATION: DLL test call succeeded, returned %d photons",
                    result["N_ph"],
                )
                self._dll_tested = True
            except Exception as e:
                logger.error("SIMULATION: DLL test call failed: %s", e)
                self.dll_available = False
                self._dll_tested = True
                return False
        
        # Start background generation thread
        import threading
        self.generation_thread = threading.Thread(
            target=self._background_dll_generation,
            args=(params, data_queue, stop_event),
            daemon=True
        )
        self.generation_thread.start()
        
        logger.info("SIMULATION: Started background DLL generation thread")
        return True


class SimulationDevice(TCSPCDeviceABC):
    """Simulation TCSPC device wrapper with queue-based streaming."""
    
    # Qt signal must be class attribute, not instance attribute
    message_logged = Signal(str)

    def __init__(self):
        """Initialize the simulation device."""
        super().__init__()
        self.simulator = BurbulatorSimulator()
        self.initialized = True
        self.measurement_running = False
        self.available_cards = [0]  # One virtual card
        self.active_cards = [0]
        self.device_type = "SIMULATION"
        
        # Queue-based streaming architecture
        self.data_queue = queue.Queue(maxsize=50)  # Buffer up to 50 batches
        self.stop_event = threading.Event()
        self.generation_thread = None
        
        # Current data buffer for read_fifo
        self.current_buffer = np.array([], dtype=np.uint32)
        self.buffer_index = 0
        
        # Default simulation parameters - optimized for realistic measurements
        self.simulation_params = {
            'N_species': 1,
            'M': [50.0],        # 50 molecules initial population
            'D': [3.0],          # Diffusion coefficient in μm²/s
            'N_channels': 2,     # Parallel and perpendicular detection
            'q': [50.0, 50.0],     # Brightness per species per channel (photons/molecule/μs)
            'q_bg': [0.001, 0.001],  # Background count rate per channel (counts/μs)
            'k_rad': [1.0],      # Radiative decay rate (μs⁻¹)
            'k_nrad': [0.0],     # Non-radiative decay rate
            'box_xy': 2.0,       # 2 μm lateral box size
            'box_z': 4.0,        # 4 μm axial box size (covers focus volume)
            'focus_type': 0,     # 3D Gaussian focus
            'focus_param': [0.3, 2.0],  # Focus parameters [waist_xy, waist_z] in μm
            'dt': 0.01,          # 10 ns diffusion time step
            'N_ph_max': 1000000,    # Total photons (1 million for realistic dataset)
            'N_ph_per_file': 100000,  # Photons per SPC file (10 files total)
            # BH_SPC conversion parameters
            'pulsed_exc': 0,
            'ch_conversion': [8, 0, 9, 1, 10, 2],
            'N_tac_channels': 4096,
            'tac_dt': 0.004069,  # TAC channel width in ns
            'laser_period': 13.596,  # Laser period in ns
        }

    def _data_generation_worker(self):
        """Background thread that generates photon data and pushes to queue."""
        try:
            # Generate all simulated photon data at once
            simulated_data = self.simulator.simulate_photons(self.simulation_params)
            total_photons = len(simulated_data)
            
            if total_photons == 0:
                self.log_message("No photons generated by simulator")
                return
            
            self.log_message(f"Generated {total_photons} simulated photons, starting background data feed")
            
            # Split data into chunks and push to queue
            chunk_size_photons = 1000  # Smaller chunks for more responsive feeding
            photon_index = 0
            
            while not self.data_generation_stop.is_set() and photon_index < total_photons:
                # Check if queue has space
                if self.data_queue.full():
                    time.sleep(0.001)  # Brief pause if queue is full
                    continue
                
                # Get next chunk
                end_index = min(photon_index + chunk_size_photons, total_photons)
                chunk = simulated_data[photon_index:end_index]
                photon_index = end_index
                
                # Put chunk in queue (blocks if full)
                try:
                    self.data_queue.put(chunk, timeout=1.0)
                except queue.Full:
                    self.log_message("Data queue full, dropping chunk")
                    continue
                
                # Small delay to simulate realistic data arrival rate
                time.sleep(0.01)  # 10ms delay between chunks
            
            self.log_message(f"Background data generation completed, fed {photon_index} photons")
            
        except Exception as e:
            self.log_message(f"Error in data generation thread: {e}")
        finally:
            # Signal end of data by putting empty array
            try:
                self.data_queue.put(np.array([], dtype=np.uint32), timeout=1.0)
            except queue.Full:
                pass

    def log_message(self, message):
        """Log a message.

        When running inside chisurf, messages are routed through the standard
        logging system instead of printing directly to stdout. The PyQt signal
        is still emitted so the GUI can display the same information.
        """
        logger.info("SIMULATION: %s", message)
        self.message_logged.emit(message)

    def detect_cards(self, simulation=False):
        """Detect simulated cards."""
        return self.available_cards

    def set_active_cards(self, card_numbers):
        """Set active cards."""
        self.active_cards = card_numbers

    def get_active_cards(self):
        """Get active cards."""
        return self.active_cards

    def initialize(self, simulation=True):
        """Initialize the simulation device."""
        self.initialized = True
        self.log_message("Simulation device initialized")
        return True

    def start_measurement(self):
        """Start simulated measurement with streaming data generation."""
        if not self.initialized:
            self.log_message("Simulation device not initialized")
            return False

        try:
            # Set defaults
            if 'N_ph_max' not in self.simulation_params:
                self.simulation_params['N_ph_max'] = 1000000
            if 'N_ph_per_file' not in self.simulation_params:
                self.simulation_params['N_ph_per_file'] = 100000

            # Create output directory for SPC file writing (done by background thread)
            spc_output_path = self.simulation_params.get('spc_output_path', '')
            if not spc_output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spc_output_path = f"simulation_output_{timestamp}"
                self.simulation_params['spc_output_path'] = spc_output_path
                os.makedirs(spc_output_path, exist_ok=True)

            # Write simulation config JSON file to output directory
            config_file_path = os.path.join(spc_output_path, "simulation_config.json")
            try:
                import json
                serializable_params = make_json_serializable(self.simulation_params)
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_params, f, indent=2)
                self.log_message(f"Simulation config written to {config_file_path}")
            except Exception as e:
                self.log_message(f"Failed to write simulation config file: {e}")
                logger.warning("SIMULATION: Failed to write simulation config file: %s", e)

            # Write simulation info file
            info_file_path = os.path.join(spc_output_path, "simulation_info.txt")
            try:
                with open(info_file_path, 'w') as f:
                    f.write("Simulation Settings\n")
                    f.write("==================\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("Simulation Parameters:\n")
                    f.write("-" * 25 + "\n")
                    for key, value in self.simulation_params.items():
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\nDevice Information:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Device Type: {self.device_type}\n")
                    f.write(f"Initialized: {self.initialized}\n")
                    
                self.log_message(f"Simulation info written to {info_file_path}")
            except Exception as e:
                self.log_message(f"Failed to write simulation info file: {e}")
                logger.warning("SIMULATION: Failed to write simulation info file: %s", e)

            self.log_message(f"Starting streaming simulation: {self.simulation_params['N_ph_max']} photons")
            logger.debug("SIMULATION: simulation_params = %s", self.simulation_params)

            # Clear queue and reset state
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    break
            
            self.stop_event.clear()
            self.current_buffer = np.array([], dtype=np.uint32)
            self.buffer_index = 0
            self.measurement_running = True

            # Start streaming generation in background
            success = self.simulator.simulate_photons_streaming(
                self.simulation_params,
                self.data_queue,
                self.stop_event
            )

            if not success:
                self.log_message("ERROR: Failed to start streaming simulation")
                self.measurement_running = False
                return False

            self.log_message("Streaming simulation started successfully")
            return True
            
        except Exception as e:
            self.log_message(f"Error starting simulation: {e}")
            logger.exception("SIMULATION: Exception in start_measurement")
            self.measurement_running = False
            return False


    def stop_measurement(self):
        """Stop simulated measurement."""
        self.measurement_running = False
        
        # Signal background thread to stop
        self.stop_event.set()
        
        # Wait for generation thread to finish
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=3.0)
            if self.generation_thread.is_alive():
                self.log_message("Background generation thread did not stop gracefully")
        
        # Clear queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break
        
        self.log_message("Simulation measurement stopped")
        return True

    def read_fifo(self, max_words=32768):
        """Read photon data from queue (streaming architecture).
        
        Pulls data from background generation thread's queue.
        No throttling needed - queue naturally paces the data flow.
        """
        import queue as queue_module
        
        if not self.initialized or not self.measurement_running:
            return np.array([], dtype=np.uint32)
        
        # max_words is number of 16-bit words, each photon is 2 words (32 bits)
        max_photons = max_words // 2
        
        # Try to fill from current buffer first
        if len(self.current_buffer) > self.buffer_index:
            remaining_in_buffer = len(self.current_buffer) - self.buffer_index
            to_return = min(remaining_in_buffer, max_photons)
            chunk = self.current_buffer[self.buffer_index:self.buffer_index + to_return]
            self.buffer_index += to_return
            
            # If we fully used the buffer, clear it
            if self.buffer_index >= len(self.current_buffer):
                self.current_buffer = np.array([], dtype=np.uint32)
                self.buffer_index = 0
            
            return chunk
        
        # Buffer empty, get next batch from queue
        try:
            # Non-blocking get with short timeout
            data = self.data_queue.get(timeout=0.01)
            
            if data is None:
                # None signals end of generation
                self.measurement_running = False
                self.log_message("Streaming generation completed")
                return np.array([], dtype=np.uint32)
            
            # Got new batch, store in buffer
            self.current_buffer = data
            self.buffer_index = 0
            
            # Return portion up to max_photons
            to_return = min(len(self.current_buffer), max_photons)
            chunk = self.current_buffer[:to_return]
            self.buffer_index = to_return
            
            return chunk
            
        except queue_module.Empty:
            # Queue empty but still generating - return empty array
            # This is normal, just means we're waiting for next batch
            return np.array([], dtype=np.uint32)

    def get_fifo_usage(self):
        """Get FIFO usage based on queue status."""
        if not self.initialized:
            return {0: -1}

        if not self.measurement_running:
            return {0: 0.0}

        # Report queue fill level as FIFO usage (0-100%)
        try:
            queue_size = self.data_queue.qsize()
            max_size = self.data_queue.maxsize if self.data_queue.maxsize > 0 else 50
            usage = min(100.0, (queue_size / max_size) * 100.0)
            return {0: usage}
        except:
            return {0: 0.0}

    def close(self):
        """Close the simulation device."""
        # Stop any running measurement
        if self.measurement_running:
            self.stop_measurement()
        
        self.initialized = False
        self.log_message("Simulation device closed")
