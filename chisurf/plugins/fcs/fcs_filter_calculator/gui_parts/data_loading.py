from __future__ import annotations
import pathlib
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import chisurf as cs
try:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import load_detector_setups
    HAS_DETECTOR_WIZARD = True
except ImportError:
    HAS_DETECTOR_WIZARD = False
    load_detector_setups = None

def parse_bst_file(path: pathlib.Path) -> Tuple[pathlib.Path | None, List[Tuple[int, int]]]:
    """Return (tttr_path, list_of_(start, end)) for a Burst-ID .bst file.

    The underlying TTTR file is searched in the .bst folder and up to
    three parent folders by stripping the trailing ``.bst`` from the
    filename.
    """
    if not path.exists() or not path.is_file():
        return None, []

    base_with_ext = path.name[:-4]  # strip '.bst'
    candidates = [path.parent]
    try:
        if path.parent.parent:
            candidates.append(path.parent.parent)
        if path.parent.parent.parent:
            candidates.append(path.parent.parent.parent)
        if path.parent.parent.parent.parent:
            candidates.append(path.parent.parent.parent.parent)
    except Exception:
        pass

    tttr_path: pathlib.Path | None = None
    for folder in candidates:
        cand = folder / base_with_ext
        if cand.exists() and cand.is_file():
            tttr_path = cand
            break

    ranges: List[Tuple[int, int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    s = int(float(parts[0]))
                    e = int(float(parts[1]))
                    if e >= s:
                        ranges.append((s, e))
                except Exception:
                    continue
    except Exception:
        ranges = []

    return tttr_path, ranges

def load_vector(path: pathlib.Path, chs: List[str] | None = None, detector_settings: Dict[str, Any] | None = None) -> np.ndarray:
    ext = path.suffix.lower()
    
    if ext in ('.spc', '.ptu', '.ht3', '.tttr'):
        import tttrlib
        try:
            # tttrlib.TTTR handles most formats automatically
            if ext == '.spc':
                try:
                    data = tttrlib.TTTR(str(path), 'SPC-130')
                except:
                    data = tttrlib.TTTR(str(path))
            else:
                data = tttrlib.TTTR(str(path))
                
            header = data.get_header()
            n_tac = header.number_of_micro_time_channels
            hist = np.zeros(n_tac, dtype=np.float64)
            
            # Get microtimes and routing channels
            microtimes = data.micro_times
            routing = data.routing_channels
            
            # If chs (detector names) provided, we need to map them back to routing channels
            if chs is not None and len(chs) > 0:
                mask = np.zeros_like(routing, dtype=bool)
                for ch_name in chs:
                    if ch_name.startswith("routing_"):
                        try:
                            ch_idx = int(ch_name.split("_")[1])
                            mask |= (routing == ch_idx)
                        except: pass
                    elif detector_settings:
                        # Map from detector name to routing channel index using settings
                        try:
                            det_config = detector_settings.get("detectors", {}).get(ch_name, {})
                            # Check for 'chs' (from detector wizard), 'channels', or 'channel'
                            det_chs = det_config.get("chs") or det_config.get("channels")
                            if det_chs:
                                for c in det_chs:
                                    mask |= (routing == int(c))
                            else:
                                det_ch = det_config.get("channel")
                                if det_ch is not None:
                                    mask |= (routing == int(det_ch))
                        except: pass
                
                # If no channels matched (mask is all False), use all photons as fallback
                if not mask.any():
                    cs.logging.warning(f"No detector channels matched for {path.name}, using all photons")
                    valid = (microtimes >= 0) & (microtimes < n_tac)
                else:
                    valid = (microtimes >= 0) & (microtimes < n_tac) & mask
            else:
                valid = (microtimes >= 0) & (microtimes < n_tac)
                
            np.add.at(hist, microtimes[valid], 1)
            return hist.ravel()
        except Exception as e:
            cs.logging.error(f"tttrlib failed to load {path}: {e}")
            return np.fromfile(path, dtype=np.float64).ravel().astype(float)

    elif ext == '.bst':
        tttr_path, ranges = parse_bst_file(path)
        
        if tttr_path and tttr_path.exists():
            import tttrlib
            data = tttrlib.TTTR(str(tttr_path))
            header = data.get_header()
            n_tac = header.number_of_micro_time_channels
            
            microtimes = data.micro_times
            routing = data.routing_channels
            
            hist = np.zeros(n_tac, dtype=np.float64)
            for start, end in ranges:
                if start < len(microtimes):
                    burst_mt = microtimes[start:min(end + 1, len(microtimes))]
                    burst_rt = routing[start:min(end + 1, len(microtimes))]
                    
                    if chs is not None and len(chs) > 0:
                        mask = np.zeros_like(burst_rt, dtype=bool)
                        for ch_name in chs:
                            if ch_name.startswith("routing_"):
                                try:
                                    ch_idx = int(ch_name.split("_")[1])
                                    mask |= (burst_rt == ch_idx)
                                except: pass
                            elif detector_settings:
                                # Map from detector name to routing channel index using settings
                                try:
                                    det_config = detector_settings.get("detectors", {}).get(ch_name, {})
                                    # Check for 'chs' (from detector wizard), 'channels', or 'channel'
                                    det_chs = det_config.get("chs") or det_config.get("channels")
                                    if det_chs:
                                        for c in det_chs:
                                            mask |= (burst_rt == int(c))
                                    else:
                                        det_ch = det_config.get("channel")
                                        if det_ch is not None:
                                            mask |= (burst_rt == int(det_ch))
                                except: pass
                        # If no channels matched, use all photons as fallback
                        if not mask.any():
                            valid = (burst_mt >= 0) & (burst_mt < n_tac)
                        else:
                            valid = (burst_mt >= 0) & (burst_mt < n_tac) & mask
                    else:
                        valid = (burst_mt >= 0) & (burst_mt < n_tac)
                        
                    np.add.at(hist, burst_mt[valid], 1)
            return hist.ravel()
        else:
            # Fallback: maybe it IS a text histogram?
            try:
                data = np.loadtxt(path)
                return data.ravel().astype(float)
            except:
                return np.fromfile(path, dtype=np.float64).ravel().astype(float)

    else:
        # Default text loading
        try:
            data = np.loadtxt(path)
            return data.ravel().astype(float)
        except (UnicodeDecodeError, ValueError):
            # Fallback for unexpected binary files
            data = np.fromfile(path, dtype=np.float64)
            return data.ravel().astype(float)
