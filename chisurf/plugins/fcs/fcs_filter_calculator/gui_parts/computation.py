from __future__ import annotations

import pathlib
import numpy as np
from typing import List, Dict
import chisurf
from ..api import compute_filters, compute_filters_mfd, FilterResult

class FilterComputationMixin:
    """Mixin class containing all filter computation methods."""
    
    def _load_routing_channels(self, path: pathlib.Path) -> Dict[int, np.ndarray]:
        """Load and cache all routing channel histograms for a TTTR file.
        
        Returns dict mapping routing_channel_number -> histogram.
        """
        path_str = str(path.absolute())
        
        if path_str in self._routing_cache:
            return self._routing_cache[path_str]
        
        # Load file and extract all routing channels
        ext = path.suffix.lower()
        routing_histograms = {}
        
        if ext in ('.spc', '.ptu', '.ht3', '.tttr'):
            import tttrlib
            try:
                # Load TTTR data once
                if ext == '.spc':
                    try:
                        data = tttrlib.TTTR(str(path), 'SPC-130')
                    except:
                        data = tttrlib.TTTR(str(path))
                else:
                    data = tttrlib.TTTR(str(path))
                
                header = data.get_header()
                try:
                    n_tac = header.number_of_micro_time_channels
                except AttributeError:
                    try:
                        n_tac = header['number_of_micro_time_channels']
                    except (KeyError, TypeError):
                        n_tac = 4096
                
                microtimes = data.micro_times
                routing = data.routing_channels
                
                # Extract histogram for each routing channel
                unique_routing = np.unique(routing)
                for rch in unique_routing:
                    mask = (routing == rch) & (microtimes >= 0) & (microtimes < n_tac)
                    hist = np.zeros(n_tac, dtype=np.float64)
                    np.add.at(hist, microtimes[mask], 1)
                    routing_histograms[int(rch)] = hist
                
            except Exception as e:
                chisurf.logging.warning(f"Error loading routing channels from {path.name}: {e}")
        
        elif ext == '.bst':
            # For BST files, load the underlying TTTR and extract bursts
            from .data_loading import parse_bst_file
            tttr_path, ranges = parse_bst_file(path)
            
            if tttr_path and ranges:
                import tttrlib
                try:
                    data = tttrlib.TTTR(str(tttr_path))
                    header = data.get_header()
                    try:
                        n_tac = header.number_of_micro_time_channels
                    except AttributeError:
                        try:
                            n_tac = header['number_of_micro_time_channels']
                        except (KeyError, TypeError):
                            n_tac = 4096
                    
                    microtimes = data.micro_times
                    routing = data.routing_channels
                    
                    # Extract histograms for each routing channel from burst regions
                    unique_routing = np.unique(routing)
                    for rch in unique_routing:
                        hist = np.zeros(n_tac, dtype=np.float64)
                        for start, end in ranges:
                            if start < len(microtimes) and end <= len(microtimes):
                                burst_mt = microtimes[start:end]
                                burst_rt = routing[start:end]
                                mask = (burst_rt == rch) & (burst_mt >= 0) & (burst_mt < n_tac)
                                np.add.at(hist, burst_mt[mask], 1)
                        routing_histograms[int(rch)] = hist
                        
                except Exception as e:
                    chisurf.logging.warning(f"Error loading BST routing channels from {path.name}: {e}")
        
        # Cache the routing histograms
        self._routing_cache[path_str] = routing_histograms
        return routing_histograms

    def _load_and_sum_vectors(self, paths: List[pathlib.Path], chs: List[str] | None) -> np.ndarray:
        """Load and sum vectors with routing channel caching."""
        cache_key = self._get_cache_key(paths, chs)
        
        if cache_key in self._decay_cache:
            return self._decay_cache[cache_key].copy()
        
        # Determine which routing channels to use
        routing_channels = set()
        if chs:
            for ch_name in chs:
                if ch_name.startswith("routing_"):
                    try:
                        routing_channels.add(int(ch_name.split("_")[1]))
                    except:
                        pass
                elif self._detector_settings:
                    # Map detector name to routing channels
                    det_config = self._detector_settings.get("detectors", {}).get(ch_name, {})
                    det_chs = det_config.get("chs", [])
                    routing_channels.update(det_chs)
        
        # Load and combine from routing channel cache
        max_size = 0
        all_histograms = []
        
        for path in paths:
            ext = path.suffix.lower()
            
            if ext in ('.spc', '.ptu', '.ht3', '.tttr', '.bst'):
                # Use routing channel cache
                routing_hists = self._load_routing_channels(path)
                
                if routing_channels:
                    # Combine specified routing channels
                    combined = None
                    for rch in routing_channels:
                        if rch in routing_hists:
                            if combined is None:
                                combined = routing_hists[rch].copy()
                            else:
                                combined += routing_hists[rch]
                    if combined is not None:
                        all_histograms.append(combined)
                        max_size = max(max_size, combined.size)
                else:
                    # Use all routing channels
                    combined = None
                    for hist in routing_hists.values():
                        if combined is None:
                            combined = hist.copy()
                        else:
                            combined += hist
                    if combined is not None:
                        all_histograms.append(combined)
                        max_size = max(max_size, combined.size)
            else:
                # For text files, use old method
                from .data_loading import load_vector
                vec = load_vector(path, chs=chs, detector_settings=self._detector_settings)
                all_histograms.append(vec)
                max_size = max(max_size, vec.size)
        
        # Sum all histograms
        if max_size == 0 or not all_histograms:
            # No valid data - return empty array
            chisurf.logging.warning(f"No valid histogram data loaded for paths: {[p.name for p in paths]}")
            return np.zeros(4096, dtype=np.float64)
        
        summed = np.zeros(max_size, dtype=np.float64)
        for hist in all_histograms:
            if hist.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:hist.size] = hist
                summed += padded
            else:
                summed += hist
        
        # Cache the result
        self._decay_cache[cache_key] = summed.copy()
        return summed

    def _compute_filters(self) -> None:
        """Main filter computation dispatcher."""
        if not self._total_paths or self.lw_species.count() == 0:
            return

        try:
            # Detector filtering logic
            chs = None
            if self.detector_selection.checkboxes:
                chs = self.detector_selection.get_selected()
            
            # Check if Anisotropy mode is enabled (takes priority over multi-detector)
            anisotropy_mode = self.anisotropy_mode_cb.isChecked()
            if anisotropy_mode and chs and len(chs) >= 1:
                self._compute_filters_anisotropy(chs)
                return
            
            # Check if multiple detectors are selected (multi-detector stacking mode)
            # Only if NOT in anisotropy mode
            if chs and len(chs) > 1:
                self._compute_filters_multi_detector(chs)
                return
            
            # Standard single-channel mode
            # Load and sum total decay files (with caching)
            total_data = self._load_and_sum_vectors(self._total_paths, chs)
            
            # Load and sum species patterns (with caching)
            species_data = []
            species_names = []
            species_patterns = []
            
            for i in range(self.lw_species.count()):
                item = self.lw_species.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                
                paths = item.data(QtCore.Qt.UserRole)
                species_patterns.append([str(p.absolute()) for p in paths])
                
                # Load all vectors for this pattern (with caching)
                pattern_data = self._load_and_sum_vectors(paths, chs)
                species_data.append(pattern_data)
                species_names.append(item.text())

            # Validate against total - use the total size as the master size
            total_size = total_data.size
            for i in range(len(species_data)):
                sd = species_data[i]
                if sd.size != total_size:
                    # Adjust species pattern to match total size (truncate or pad)
                    if sd.size > total_size:
                        species_data[i] = sd[:total_size]
                    else:
                        padded = np.zeros(total_size, dtype=np.float64)
                        padded[:sd.size] = sd
                        species_data[i] = padded

            self._result = compute_filters(
                total_data, 
                species_data, 
                total_path=[str(p.absolute()) for p in self._total_paths],
                species_patterns=species_patterns
            )
            self._result_anisotropy = None  # Clear anisotropy result
            self._result_multi_detector = None  # Clear multi-detector result
            self._result_multi_anisotropy = None  # Clear multi-anisotropy result
            self._update_plots()
            self.btn_export.setEnabled(True)
            self._update_status("Filters computed successfully.")

        except Exception as e:
            import traceback
            chisurf.logging.error(f"Computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Computation Error", str(e))
            self._update_status(f"Error: {e}")

    def _compute_filters_anisotropy(self, chs: List[str]) -> None:
        """Compute Anisotropy filters for parallel and perpendicular channels.
        
        Uses detector wizard routing channel logic: channels alternate as par, perp, par, perp.
        For a detector with routing channels [8, 0], ch 8 = parallel, ch 0 = perpendicular.
        With multiple detectors - computes separate par/perp for each detector and stacks them.
        """
        from ..api import compute_filters_mfd
        
        try:
            # Check if multiple detectors selected - compute separately for each
            if len(chs) > 1:
                self._compute_filters_multi_anisotropy(chs)
                return
            
            # Single detector anisotropy mode
            det_name = chs[0]
            
            # Extract routing channels for this detector
            if not self._detector_settings:
                raise ValueError("Anisotropy mode requires detector settings")
            
            det_config = self._detector_settings.get("detectors", {}).get(det_name, {})
            routing_chs = det_config.get("chs", [])
            
            if len(routing_chs) < 2:
                raise ValueError(f"Detector '{det_name}' has <2 routing channels for Anisotropy mode")
            
            # Split into par/perp (even indices = parallel, odd = perpendicular)
            routing_par = []
            routing_perp = []
            for i, ch in enumerate(routing_chs):
                if i % 2 == 0:
                    routing_par.append(ch)
                else:
                    routing_perp.append(ch)
            
            ch_par = [f"routing_{ch}" for ch in routing_par]
            ch_perp = [f"routing_{ch}" for ch in routing_perp]
            
            # Load total decay for parallel and perpendicular channels (with caching)
            total_par = self._load_and_sum_vectors(self._total_paths, ch_par)
            total_perp = self._load_and_sum_vectors(self._total_paths, ch_perp)
            
            # Ensure same size
            max_size = max(total_par.size, total_perp.size)
            if total_par.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:total_par.size] = total_par
                total_par = padded
            if total_perp.size < max_size:
                padded = np.zeros(max_size, dtype=np.float64)
                padded[:total_perp.size] = total_perp
                total_perp = padded
            
            # Load species patterns for parallel and perpendicular channels (with caching)
            species_par = []
            species_perp = []
            species_patterns = []
            
            for i in range(self.lw_species.count()):
                item = self.lw_species.item(i)
                if item.checkState() != QtCore.Qt.Checked:
                    continue
                
                paths = item.data(QtCore.Qt.UserRole)
                species_patterns.append([str(p.absolute()) for p in paths])
                
                # Load for parallel channel (with caching)
                pattern_par = self._load_and_sum_vectors(paths, ch_par)
                if pattern_par.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:pattern_par.size] = pattern_par
                    pattern_par = padded
                species_par.append(pattern_par)
                
                # Load for perpendicular channel (with caching)
                pattern_perp = self._load_and_sum_vectors(paths, ch_perp)
                if pattern_perp.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:pattern_perp.size] = pattern_perp
                    pattern_perp = padded
                species_perp.append(pattern_perp)
            
            # Compute Anisotropy filters
            metadata = {
                "detector": chs[0],
                "routing_par": routing_par,
                "routing_perp": routing_perp,
            }
            self._result_anisotropy = compute_filters_mfd(
                total_par, total_perp,
                species_par, species_perp,
                metadata=metadata
            )
            self._result = None  # Clear single-channel result
            self._result_multi_detector = None  # Clear multi-detector result
            self._result_multi_anisotropy = None  # Clear multi-anisotropy result
            self._update_plots()
            self.btn_export.setEnabled(True)
            self._update_status(f"Anisotropy filters computed successfully ({chs[0]}: ch {routing_par} || ch {routing_perp}).")
            
        except Exception as e:
            import traceback
            chisurf.logging.error(f"Anisotropy computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Anisotropy Computation Error", str(e))
            self._update_status(f"Anisotropy Error: {e}")

    def _compute_filters_multi_anisotropy(self, chs: List[str]) -> None:
        """Compute Anisotropy filters separately for each detector and stack them.
        
        Each detector gets its own par/perp computation.
        """
        from ..api import compute_filters_mfd
        
        try:
            anisotropy_results = []
            
            for det_name in chs:
                # Extract routing channels for this detector
                if not self._detector_settings:
                    raise ValueError("Anisotropy mode requires detector settings")
                
                det_config = self._detector_settings.get("detectors", {}).get(det_name, {})
                routing_chs = det_config.get("chs", [])
                
                if len(routing_chs) < 2:
                    chisurf.logging.warning(f"Detector '{det_name}' has <2 routing channels, skipping")
                    continue
                
                # Split into par/perp
                routing_par = []
                routing_perp = []
                for i, ch in enumerate(routing_chs):
                    if i % 2 == 0:
                        routing_par.append(ch)
                    else:
                        routing_perp.append(ch)
                
                ch_par = [f"routing_{ch}" for ch in routing_par]
                ch_perp = [f"routing_{ch}" for ch in routing_perp]
                
                # Load total decay for this detector's par/perp
                total_par = self._load_and_sum_vectors(self._total_paths, ch_par)
                total_perp = self._load_and_sum_vectors(self._total_paths, ch_perp)
                
                # Ensure same size
                max_size = max(total_par.size, total_perp.size)
                if total_par.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:total_par.size] = total_par
                    total_par = padded
                if total_perp.size < max_size:
                    padded = np.zeros(max_size, dtype=np.float64)
                    padded[:total_perp.size] = total_perp
                    total_perp = padded
                
                # Load species patterns
                species_par = []
                species_perp = []
                species_patterns = []
                
                for i in range(self.lw_species.count()):
                    item = self.lw_species.item(i)
                    if item.checkState() != QtCore.Qt.Checked:
                        continue
                    
                    paths = item.data(QtCore.Qt.UserRole)
                    species_patterns.append([str(p.absolute()) for p in paths])
                    
                    pattern_par = self._load_and_sum_vectors(paths, ch_par)
                    if pattern_par.size < max_size:
                        padded = np.zeros(max_size, dtype=np.float64)
                        padded[:pattern_par.size] = pattern_par
                        pattern_par = padded
                    species_par.append(pattern_par)
                    
                    pattern_perp = self._load_and_sum_vectors(paths, ch_perp)
                    if pattern_perp.size < max_size:
                        padded = np.zeros(max_size, dtype=np.float64)
                        padded[:pattern_perp.size] = pattern_perp
                        pattern_perp = padded
                    species_perp.append(pattern_perp)
                
                # Compute anisotropy for this detector
                metadata = {
                    "detector": det_name,
                    "routing_par": routing_par,
                    "routing_perp": routing_perp,
                }
                result = compute_filters_mfd(
                    total_par, total_perp,
                    species_par, species_perp,
                    metadata=metadata
                )
                
                anisotropy_results.append({
                    'detector': det_name,
                    'result': result
                })
            
            if not anisotropy_results:
                raise ValueError("No valid detectors for Anisotropy mode")
            
            # Store multi-anisotropy results
            self._result_multi_anisotropy = anisotropy_results
            self._result = None
            self._result_anisotropy = None
            self._result_multi_detector = None
            self._update_plots()
            self.btn_export.setEnabled(True)
            detector_names = ", ".join([ar['detector'] for ar in anisotropy_results])
            self._update_status(f"Multi-detector Anisotropy filters computed successfully ({detector_names}).")
            
        except Exception as e:
            import traceback
            chisurf.logging.error(f"Multi-Anisotropy computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Multi-Anisotropy Computation Error", str(e))
            self._update_status(f"Multi-Anisotropy Error: {e}")

    def _compute_filters_multi_detector(self, chs: List[str]) -> None:
        """Compute filters for multiple detectors separately and stack them.
        
        Each detector's photons are isolated - red photons don't contribute to green decay.
        """
        try:
            # Store results for each detector
            detector_results = []
            
            for det_name in chs:
                # Load total decay for this detector only (with caching)
                total_data = self._load_and_sum_vectors(self._total_paths, [det_name])
                
                # Load species patterns for this detector only (with caching)
                species_data = []
                species_names = []
                species_patterns = []
                
                for i in range(self.lw_species.count()):
                    item = self.lw_species.item(i)
                    if item.checkState() != QtCore.Qt.Checked:
                        continue
                    
                    paths = item.data(QtCore.Qt.UserRole)
                    species_patterns.append([str(p.absolute()) for p in paths])
                    
                    # Load pattern for this detector only (with caching)
                    pattern_data = self._load_and_sum_vectors(paths, [det_name])
                    
                    # Ensure same size as total
                    if pattern_data.size != total_data.size:
                        if pattern_data.size > total_data.size:
                            pattern_data = pattern_data[:total_data.size]
                        else:
                            padded = np.zeros(total_data.size, dtype=np.float64)
                            padded[:pattern_data.size] = pattern_data
                            pattern_data = padded
                    
                    species_data.append(pattern_data)
                    species_names.append(item.text())
                
                # Compute filters for this detector
                result = compute_filters(
                    total_data,
                    species_data,
                    total_path=[str(p.absolute()) for p in self._total_paths],
                    species_patterns=species_patterns
                )
                
                detector_results.append({
                    'detector': det_name,
                    'result': result
                })
            
            # Store multi-detector results
            self._result_multi_detector = detector_results
            self._result = None  # Clear single-channel result
            self._result_anisotropy = None  # Clear anisotropy result
            self._result_multi_anisotropy = None  # Clear multi-anisotropy result
            self._update_plots()
            self.btn_export.setEnabled(True)
            detector_names = ", ".join([dr['detector'] for dr in detector_results])
            self._update_status(f"Multi-detector filters computed successfully ({detector_names}).")
            
        except Exception as e:
            import traceback
            chisurf.logging.error(f"Multi-detector computation error: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Multi-Detector Computation Error", str(e))
            self._update_status(f"Multi-Detector Error: {e}")
