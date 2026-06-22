import numpy as np
import logging

logger = logging.getLogger(__name__)

# Standard wavelengths for simulation
WAVELENGTHS = np.arange(300, 901, 1, dtype=np.float64)

def interp(spec):
    """Interpolate (w, y) spectrum onto the standard grid."""
    if spec is None:
        return np.zeros_like(WAVELENGTHS)
    if isinstance(spec, tuple) and isinstance(spec[0], str) and spec[0] == "discrete":
        # Create delta peaks
        y = np.zeros_like(WAVELENGTHS)
        for wl in spec[1]:
            idx = np.abs(WAVELENGTHS - wl).argmin()
            if np.abs(WAVELENGTHS[idx] - wl) < 1.0:
                y[idx] += 1.0 # Simple unit peak
        return y
    w, y = spec
    return np.interp(WAVELENGTHS, w, y, left=0, right=0)

def calculate_r0(donor_em: np.ndarray, donor_qy: float, acceptor_abs: np.ndarray, acceptor_ec_max: float, kappa2: float = 2/3, n: float = 1.33) -> float:
    """Calculate the Forster radius R0 in Angstroms.
    
    Formula: R0 = 0.02108 * (kappa^2 * Phi_D * J * n^-4)^(1/6) * 10
    where J is the overlap integral in M^-1 cm^-1 nm^4.
    """
    # 1. Area-normalize donor emission
    area = np.trapz(donor_em, WAVELENGTHS)
    if area <= 0:
        return 0.0
    donor_em_norm = donor_em / area
    
    # 2. Overlap integral J
    # acceptor_abs is normalized to 1 at peak, so scale by extinction coefficient
    # In our case, we assume it's normalized to 1 and ec_max is supplied.
    J = np.trapz(donor_em_norm * acceptor_abs * acceptor_ec_max * (WAVELENGTHS**4), WAVELENGTHS)
    
    # 3. R0 in nm -> Angstrom
    r0_nm = 0.02108 * (kappa2 * donor_qy * J * (n**-4))**(1/6)
    return r0_nm * 10.0

def _get_input_spectral(input_spectra, port_name=None):
    """Merge spectral inputs from a specific port or all ports if None."""
    merged = {}
    if port_name:
        p_data = input_spectra.get(port_name, {})
        for src_id, data in p_data.items():
            if isinstance(data, np.ndarray):
                if src_id not in merged: merged[src_id] = np.zeros_like(WAVELENGTHS)
                merged[src_id] += data
    else:
        # Merge from all ports
        for p_data in input_spectra.values():
            if not isinstance(p_data, dict): continue
            for src_id, data in p_data.items():
                if isinstance(data, np.ndarray):
                    if src_id not in merged: merged[src_id] = np.zeros_like(WAVELENGTHS)
                    merged[src_id] += data
    return merged

def _get_input_dict(input_spectra, port_name=None):
    """Merge dictionary-based inputs (like Dye Data) from a specific port or all."""
    merged = {}
    if port_name:
        p_data = input_spectra.get(port_name, {})
        for src_id, data in p_data.items():
            if isinstance(data, dict):
                merged.update(data)
    else:
        for p_data in input_spectra.values():
            if not isinstance(p_data, dict): continue
            for src_id, data in p_data.items():
                if isinstance(data, dict):
                    merged.update(data)
    return merged

def _get_input_value(input_spectra, port_name, default=None):
    """Get the first numeric value found on a port."""
    p_data = input_spectra.get(port_name, {})
    if not isinstance(p_data, dict): return default
    for val in p_data.values():
        if isinstance(val, (int, float, np.number)):
            return val
    return default

def propagate_node(node_type, config, input_spectra, db):
    """
    Simulates the physics of a node.
    Returns: { "output_spectra": {port_name: {source_id: spectrum}}, "node_characteristic": spectrum }
    
    input_spectra is now hierarchical: { port_name: { source_id: data } }
    For backward compatibility, we'll also check if it's flat.
    """
    # 1. Normalize input_spectra if it's flat (old style)
    if input_spectra and not any(isinstance(v, dict) for v in input_spectra.values()):
        input_spectra = {"In": input_spectra}

    output_spectra = {}
    node_char = None
    
    if node_type == "light_source":
        mode = config.get("source_mode", "database")
        if mode == "database":
            sid = config.get("spectrum_id")
            if sid:
                with db:
                    s = db.get_spectrum(sid, "emission")
                    y = interp(s)
            else:
                y = np.zeros_like(WAVELENGTHS)
            output_spectra["Light"] = {"Light (Database)": y}
            node_char = y
        else:
            # Parse discrete lines, e.g., "488", "488:1.0, 561:0.5"
            lines_str = config.get("manual_lines", "")
            out_dict = {}
            sum_y = np.zeros_like(WAVELENGTHS)
            
            for part in lines_str.split(","):
                part = part.strip()
                if not part: continue
                
                pieces = part.split(":")
                try:
                    wl = float(pieces[0])
                    power = float(pieces[1]) if len(pieces) > 1 else 1.0
                except ValueError:
                    continue
                    
                y = interp(("discrete", [wl])) * power
                out_dict[f"{wl:g} nm"] = y
                sum_y += y
                
            output_spectra["Light"] = out_dict
            node_char = sum_y

    elif node_type == "sample":
        sids = config.get("spectrum_ids", [])
        if not sids and config.get("spectrum_id"):
            sids = [config["spectrum_id"]]
        
        sum_abs = np.zeros_like(WAVELENGTHS)
        sum_em = np.zeros_like(WAVELENGTHS)
        
        # Merge all spectral inputs regardless of port for sample excitation
        spectral_in = _get_input_spectral(input_spectra)
        
        if sids:
            dye_props = config.get("dye_properties", {})
            with db:
                out_dict = {}
                dye_data = {}
                for sid in sids:
                    abs_s_raw = db.get_spectrum(sid, "absorption")
                    em_s_raw = db.get_spectrum(sid, "emission")
                    props = db.get_standardized_optical_properties(sid)
                    item_data = db.get_probe_by_id(sid)
                    dye_name = item_data['chromophore_name'] if item_data else f"Probe_{sid}"
                    # QY and EC Defaults from DB or overrides from config
                    qy_str = props.get("qy", "1.0")
                    ec_str = props.get("ext_coeff", "1.0")
                    
                    sid_str = str(sid)
                    if sid_str in dye_props:
                        try:
                            qy = float(dye_props[sid_str].get("qy", qy_str))
                            ec = float(dye_props[sid_str].get("ec", ec_str))
                        except (ValueError, TypeError):
                            qy, ec = 1.0, 1.0
                    else:
                        try:
                            qy = float(qy_str)
                            ec = float(ec_str)
                        except (ValueError, TypeError):
                            qy, ec = 1.0, 1.0
                    
                    cur_abs_norm = interp(abs_s_raw)
                    cur_em_norm = interp(em_s_raw)
                    
                    # Store RAW (but grid-interpolated) spectra for Dye Data output
                    dye_data[dye_name] = {
                        "abs": cur_abs_norm.copy(),
                        "em": cur_em_norm.copy(),
                        "qy": qy,
                        "ec": ec
                    }
                    
                    # For signal propagation, scale them
                    norm = np.trapz(cur_em_norm, WAVELENGTHS)
                    if norm > 0:
                        cur_em_scaled = (cur_em_norm / norm) * qy
                    else:
                        cur_em_scaled = cur_em_norm
                        
                    cur_abs_scaled = cur_abs_norm * ec
                    
                    sum_abs += cur_abs_scaled
                    sum_em += cur_em_scaled
                    
                    for src_id, in_spec in spectral_in.items():
                        excitation_prob = np.trapz(in_spec * cur_abs_scaled, WAVELENGTHS)
                        out_dict[f"{dye_name} (ex {src_id})"] = excitation_prob * cur_em_scaled
                
            node_char = (sum_abs, sum_em)
            output_spectra["Out"] = out_dict
            output_spectra["Dye Data"] = {"Dyes": dye_data}
        else:
            output_spectra["Out"] = {}
            output_spectra["Dye Data"] = {}

    elif node_type == "filter":
        spectral_in = _get_input_spectral(input_spectra, "In")
        sid = config.get("spectrum_id")
        if sid:
            with db:
                s = db.get_spectrum(sid, "transmission")
                t_y = interp(s)
                node_char = t_y
                out_dict = {}
                for src_id, in_spec in spectral_in.items():
                    out_dict[src_id] = in_spec * t_y
                output_spectra["Out"] = out_dict
        else:
            output_spectra["Out"] = {k: v.copy() for k, v in spectral_in.items()}
            
    elif node_type == "splitter":
        spectral_in = _get_input_spectral(input_spectra, "In")
        sid = config.get("spectrum_id")
        if sid:
            with db:
                s = db.get_spectrum(sid, "transmission")
                t_y = interp(s)
                node_char = t_y
                t_dict = {}
                r_dict = {}
                for src_id, in_spec in spectral_in.items():
                    t_dict[src_id] = in_spec * t_y
                    r_dict[src_id] = in_spec * np.clip(1.0 - t_y, 0, 1)
                output_spectra["Transmission"] = t_dict
                output_spectra["Reflection"] = r_dict
        else:
            output_spectra["Transmission"] = {k: v.copy() for k, v in spectral_in.items()}
            output_spectra["Reflection"] = {}

    elif node_type == "detector":
        spectral_in = _get_input_spectral(input_spectra, "In")
        sid = config.get("spectrum_id")
        signals = {}
        if sid:
            with db:
                s = db.get_spectrum(sid, "quantum_efficiency")
                qe_y = interp(s)
                node_char = qe_y
                out_dict = {}
                for src_id, in_spec in spectral_in.items():
                    signals[src_id] = float(np.trapz(in_spec * qe_y, WAVELENGTHS))
                    out_dict[src_id] = in_spec * qe_y
                config["_last_signals"] = signals
                output_spectra["Out"] = out_dict
        else:
            config["_last_signals"] = {}
            output_spectra["Out"] = {}

    elif node_type == "combiner":
        spectral_in = _get_input_spectral(input_spectra) # Merge all incoming paths
        output_spectra["Out"] = {k: v.copy() for k, v in spectral_in.items()}

    elif node_type == "forster_radius":
        dye_data = _get_input_dict(input_spectra, "Dye Data")
        kappa2 = _get_input_value(input_spectra, "kappa2", config.get("kappa2", 0.6667))
        n = _get_input_value(input_spectra, "n", config.get("n", 1.33))
        
        results = []
        names = sorted(dye_data.keys())
        for d_name in names:
            donor = dye_data[d_name]
            for a_name in names:
                acceptor = dye_data[a_name]
                r0 = calculate_r0(
                    donor_em=donor["em"],
                    donor_qy=donor["qy"],
                    acceptor_abs=acceptor["abs"],
                    acceptor_ec_max=acceptor["ec"],
                    kappa2=float(kappa2),
                    n=float(n)
                )
                results.append({"donor": d_name, "acceptor": a_name, "r0": r0})
        
        config["_last_results"] = results
        output_spectra["Out"] = {}

    return output_spectra, node_char


# Note: Old compatibility functions (interpolate_spectra, etc.) removed 
# as they pointed to non-existent or circular references.
