from __future__ import annotations

import numpy as np

import chisurf.data
import chisurf.fio
import chisurf.fluorescence

from chisurf import typing


def _safe_float(value: typing.Any, default: typing.Optional[float] = None) -> typing.Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _annotate_anisotropy_meta(data_group, g_factor, l1, l2, source: str):
    group_meta = getattr(data_group, 'meta_data', None)
    if not isinstance(group_meta, dict):
        group_meta = {}
        data_group.meta_data = group_meta

    if g_factor is not None:
        group_meta['g_factor'] = float(g_factor)
    if l1 is not None:
        group_meta['l1'] = float(l1)
    if l2 is not None:
        group_meta['l2'] = float(l2)
    group_meta['anisotropy_calibration_source'] = source

    try:
        for curve in data_group:
            curve_meta = getattr(curve, 'meta_data', None)
            if not isinstance(curve_meta, dict):
                curve_meta = {}
                curve.meta_data = curve_meta
            if g_factor is not None:
                curve_meta.setdefault('g_factor', float(g_factor))
            if l1 is not None:
                curve_meta.setdefault('l1', float(l1))
            if l2 is not None:
                curve_meta.setdefault('l2', float(l2))
            curve_meta.setdefault('anisotropy_calibration_source', source)
    except Exception:
        pass


def read_tcspc_csv(
        filename: str = None,
        skiprows: int = None,
        rebin: typing.Tuple[int, int] = (1, 1),
        dt: float = 1.0,
        matrix_columns: typing.Tuple[int, int] = (0, 1),
        use_header: bool = False,
        is_jordi: bool = False,
        polarization: str = "vm",
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0,
        experiment: chisurf.experiments.core.Experiment = None,
        *args,
        **kwargs
) -> chisurf.data.DataCurveGroup:

    # Load data
    rebin_x, rebin_y = rebin

    if is_jordi:
        infer_delimiter = False
        mc = None
    else:
        mc = matrix_columns
        infer_delimiter = True

    csvSetup = chisurf.fio.ascii.Csv(
        *args,
        **kwargs
    )
    csvSetup.load(
        filename,
        skiprows=skiprows,
        use_header=use_header,
        usecols=mc,
        infer_delimiter=infer_delimiter
    )
    data = csvSetup.data

    calibration_source = 'reader'

    if is_jordi:
        # Read jordi file with the new format
        from chisurf.fio.jordi import read_jordi
        
        # Read the data with metadata
        data, meta = read_jordi(filename, split=True, return_metadata=True)
        
        # Get available channels
        available_channels = list(data.keys())
        
        # Get anisotropy calibration from metadata if available
        meta_g = _safe_float(meta.get('g_factor', None), None)
        meta_l1 = _safe_float(meta.get('l1', None), None)
        meta_l2 = _safe_float(meta.get('l2', None), None)
        if meta_g is not None:
            g_factor = meta_g
            calibration_source = 'jordi_metadata'
        if meta_l1 is not None:
            l1 = meta_l1
            calibration_source = 'jordi_metadata'
        if meta_l2 is not None:
            l2 = meta_l2
            calibration_source = 'jordi_metadata'
        
        # Convert data to numpy arrays
        n_data_sets = 1  # Default to 1 dataset
        
        # Handle different polarization cases
        if polarization == 'vv':
            if 'VV' not in available_channels:
                raise ValueError("VV channel not found in the jordi file")
            y = data['VV']
            if y.ndim == 1:
                y = y.reshape(1, -1)
                n_data_sets = 1
            else:
                n_data_sets = y.shape[0]
            ey = chisurf.fluorescence.tcspc.counting_noise(decay=y)
            
        elif polarization == 'vh':
            if 'VH' not in available_channels:
                raise ValueError("VH channel not found in the jordi file")
            y = data['VH']
            if y.ndim == 1:
                y = y.reshape(1, -1)
                n_data_sets = 1
            else:
                n_data_sets = y.shape[0]
                
            # Apply integer VH shift if provided by data_reader
            data_reader = kwargs.get('data_reader', None)
            vh_shift = int(getattr(data_reader, 'vh_shift', 0) or 0)
            if vh_shift != 0:
                if vh_shift > 0:
                    # Shift to the right: prepend zeros and trim end
                    pad = ((0, 0), (vh_shift, 0)) if n_data_sets > 1 else ((vh_shift, 0),)
                    y = np.pad(y, pad, mode='constant')
                    y = y[:, :-vh_shift] if n_data_sets > 1 else y[:-vh_shift]
                else:
                    # Shift to the left: append zeros and trim beginning
                    s = abs(vh_shift)
                    pad = ((0, 0), (0, s)) if n_data_sets > 1 else ((0, s),)
                    y = np.pad(y, pad, mode='constant')
                    y = y[:, s:] if n_data_sets > 1 else y[s:]
                    
            ey = chisurf.fluorescence.tcspc.counting_noise(decay=y)
            
        elif polarization == 'vm':
            if 'VM' not in available_channels:
                # Calculate VM from VV and VH if not directly available
                if 'VV' not in available_channels or 'VH' not in available_channels:
                    raise ValueError("VM channel not found and cannot be calculated (missing VV or VH)")
                vv = data['VV']
                vh = data['VH']
                y = vv + 2.0 * g_factor * vh
            else:
                y = data['VM']
                
            if y.ndim == 1:
                y = y.reshape(1, -1)
                n_data_sets = 1
            else:
                n_data_sets = y.shape[0]
                
            ey = chisurf.fluorescence.tcspc.counting_noise(decay=y)
            
        elif polarization == 'vv/vh':
            if 'VV' not in available_channels or 'VH' not in available_channels:
                raise ValueError("Both VV and VH channels are required for vv/vh polarization")
                
            vv = data['VV']
            vh = data['VH']
            
            # Handle single dataset case
            if vv.ndim == 1:
                vv = vv.reshape(1, -1)
                vh = vh.reshape(1, -1)
                n_data_sets = 1
            else:
                n_data_sets = vv.shape[0]
                
            # Stack VV and VH channels - this creates 2-channel stacked data
            y = np.vstack([vv, vh])
            
            # Calculate errors
            e1 = chisurf.fluorescence.tcspc.counting_noise(decay=vv)
            e2 = chisurf.fluorescence.tcspc.counting_noise(decay=vh)
            ey = np.vstack([e1, e2])
            
            # For VV,VH stacked data, we have 2 channels (VV and VH) regardless of n_data_sets
            n_channels = 2  # Fixed: VV and VH stacked
            
        else:  # Default to VM calculation
            if 'VM' in available_channels:
                y = data['VM']
            elif 'VV' in available_channels and 'VH' in available_channels:
                vv = data['VV']
                vh = data['VH']
                y = vv + 2.0 * g_factor * vh
            else:
                raise ValueError("Cannot determine polarization. Available channels: " + 
                               ", ".join(available_channels))
            
            if y.ndim == 1:
                y = y.reshape(1, -1)
                n_data_sets = 1
            else:
                n_data_sets = y.shape[0]
                
            ey = chisurf.fluorescence.tcspc.counting_noise(decay=y)
        
        # Apply rebinning
        n_data_points = y.shape[-1]
        new_channels = int(n_data_points / rebin_y)
        
        # For VV,VH stacked data, use n_channels=2, otherwise use n_data_sets
        if polarization == 'vv/vh':
            n_rebin_groups = 2  # VV and VH channels
        else:
            n_rebin_groups = n_data_sets
        
        # Reshape and sum for rebinning
        if n_rebin_groups > 1:
            y = y.reshape([n_rebin_groups, new_channels, rebin_y]).sum(axis=2)
            ey = ey.reshape([n_rebin_groups, new_channels, rebin_y]).sum(axis=2)
        else:
            y = y.reshape([1, new_channels, rebin_y]).sum(axis=2)
            ey = ey.reshape([1, new_channels, rebin_y]).sum(axis=2)
            
        n_data_points = y.shape[1]
        x = np.arange(
            n_data_points,
            dtype=np.float64
        ) * dt
    else:
        x = data[0] * dt
        y = data[1:]

        n_datasets, n_data_points = y.shape
        n_data_points = int(n_data_points / rebin_y)
        try:
            y = y.reshape(
                [n_datasets, n_data_points, rebin_y]
            ).sum(axis=2)
            ey = chisurf.fluorescence.tcspc.counting_noise(y)
            x = np.average(
                x.reshape([n_data_points, rebin_y]), axis=1
            ) / rebin_y
        except ValueError:
            print("Cannot reshape array")

    # TODO: in future adaptive binning of time axis
    #from scipy.stats import binned_statistic
    #dt = xn[1]-xn[0]
    #xb = np.logspace(np.log10(dt), np.log10(np.max(xn)), 512)
    #tmp = binned_statistic(xn, yn, statistic='sum', bins=xb)
    #xn = xb[:-1]
    #print xn
    #yn = tmp[0]
    #print tmp[1].shape
    x = x[n_data_points % rebin_y:]
    y = y[n_data_points % rebin_y:]

    # rebin along x-axis
    y_rebin = np.zeros_like(y)
    ib = 0
    for ix in range(0, y.shape[0], rebin_x):
        y_rebin[ib] += y[ix:ix+rebin_x, :].sum(axis=0)
        ib += 1
    y_rebin = y_rebin[:ib, :]
    data_curves = list()
    n_data_sets = y_rebin.shape[0]
    fn = csvSetup.filename
    ex = np.zeros(x.shape)  # Initialize ex variable
    
    # Create descriptive names based on polarization type
    if is_jordi and polarization == 'vv/vh':
        # For VV/VH stacked data, create VV and VH names
        if n_data_sets >= 2:
            base_name = fn
            # Add polarization suffixes
            polarization_names = ['VV', 'VH']
            for i, yi in enumerate(y_rebin[:2]):  # Only take first 2 for VV/VH
                eyi = ey[i]
                name = f'{base_name} {polarization_names[i]}'
                data = chisurf.data.DataCurve(
                    x=x,
                    y=yi,
                    ex=ex,
                    ey=eyi,
                    experiment=experiment,
                    name=name,
                    **kwargs
                )
                data.filename = filename
                data_curves.append(data)
        else:
            # Fallback for single dataset
            name = f'{fn} VV'  # Default to VV if only one dataset
            data = chisurf.data.DataCurve(
                x=x,
                y=y_rebin[0],
                ex=ex,
                ey=ey[0],
                experiment=experiment,
                name=name,
                **kwargs
            )
            data.filename = filename
            data_curves.append(data)
    elif is_jordi and polarization in ['vv', 'vh', 'vm']:
        # For single polarization data
        pol_name = polarization.upper()
        if n_data_sets > 1:
            for i, yi in enumerate(y_rebin):
                eyi = ey[i]
                name = f'{fn} {pol_name}_{i+1}'
                data = chisurf.data.DataCurve(
                    x=x,
                    y=yi,
                    ex=ex,
                    ey=eyi,
                    experiment=experiment,
                    name=name,
                    **kwargs
                )
                data.filename = filename
                data_curves.append(data)
        else:
            name = f'{fn} {pol_name}'
            data = chisurf.data.DataCurve(
                x=x,
                y=y_rebin[0],
                ex=ex,
                ey=ey[0],
                experiment=experiment,
                name=name,
                **kwargs
            )
            data.filename = filename
            data_curves.append(data)
    else:
        # Original naming logic for non-jordi or other cases
        for i, yi in enumerate(y_rebin):
            eyi = ey[i]
            if n_data_sets > 1:
                name = '{} {:d}_{:d}'.format(fn, i, n_data_sets)
            else:
                name = filename
            data = chisurf.data.DataCurve(
                x=x,
                y=yi,
                ex=ex,
                ey=eyi,
                experiment=experiment,
                name=name,
                **kwargs
            )
            data.filename = filename
            data_curves.append(data)
    data_group = chisurf.data.DataCurveGroup(data_curves, filename)
    _annotate_anisotropy_meta(
        data_group=data_group,
        g_factor=g_factor,
        l1=l1,
        l2=l2,
        source=calibration_source
    )
    return data_group
