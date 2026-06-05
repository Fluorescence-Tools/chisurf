# ChiSurf Potential Files Documentation

This directory contains potential energy functions used for protein structure calculations in ChiSurf.

## Available Potential Files

### Active Potentials (Used by Widgets)

| File | Description | Widget | Shape | Status |
|------|-------------|--------|-------|--------|
| `unres.npy` | UNRES (Unified Residue) potential lookup table | CEPotentialWidget (Iso-UNRES) | (20, 20, 400) | ✅ Active |
| `mj.npy` | Miyazawa-Jernigan statistical potential | MJPotentialWidget (Miyazawa-Jernigan) | (20, 20) | ✅ Active |
| `hb.npy` | Hydrogen bond potential matrix | HPotentialWidget (H-Bond) | (4, 800) | ✅ Active |
| `rama_ala_pro_gly.npy` | Ramachandran plot data for Ala, Pro, Gly | RamachandranWidget | (5, 129600) | ✅ Active |

### Additional Files (Not Currently Used)

| File | Description | Status | Notes |
|------|-------------|--------|-------|
| `repulsive_cb.npy` | Repulsive CB potential (based on unres.npy) | ❌ Empty | File is empty (0 bytes) - see UNUSED_FILES.md |

### Source Data Files

| File | Description | Format | Size | Status |
|------|-------------|--------|------|--------|
| `hb.csv` | H-bond potential source data | CSV | 32KB | ✅ Available |
| `mj.csv` | Miyazawa-Jernigan source data | CSV | 3.2KB | ✅ Available |
| `rama_ala_pro_gly.csv` | Ramachandran plot source data | CSV | 7.1MB | ✅ Available |
| `mj.gnumeric` | MJ potential in Gnumeric format | Gnumeric | 5.6KB | ✅ Available |
| `unres_1996.xlsx` | UNRES 1996 data in Excel format | Excel | 14KB | ✅ Available |
| `cont-eng.dat` | Contact energy data | DAT | 2.5KB | ✅ Available |

> **Note**: Unused potential files are documented in `UNUSED_FILES.md` with regeneration instructions.

## File Organization

The current organization is optimal:
- All potential files are centralized in `chisurf/structure/potential/database/`
- Clear naming convention with descriptive extensions
- README and documentation files present
- Easy to maintain and extend

## Integration with Widgets

All potential widgets now use the `chisurf.settings.path_utils.get_path('chisurf')` function to resolve paths dynamically:

```python
# Example from CEPotentialWidget
potential = str(get_path('chisurf') / 'structure/potential/database/unres.npy')
```

This ensures:
- ✅ Works regardless of installation location
- ✅ No hardcoded paths that can break
- ✅ Consistent path resolution across all widgets
- ✅ Error handling for missing files

## Adding New Potentials

To add a new potential:
1. Place the `.npy` file in this directory
2. Create or update the corresponding widget class
3. Use the same path resolution pattern
4. Add error handling for missing files
5. Update this documentation

## File Formats

- **.npy files**: NumPy binary format for efficient loading
- **.csv files**: Human-readable data (e.g., centroid data)
- **.xlsx files**: Excel format for manual inspection
- **.dat files**: Text format for specialized data

## Dependencies

- NumPy for loading .npy files
- All potential widgets handle missing files gracefully
- Error messages guide users to correct issues
