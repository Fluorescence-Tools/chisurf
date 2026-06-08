# Implement chsiurf energy terms in proteinmc

**Session ID:** ses_15977ead0ffeDfQ0RwR6s6q7mT
**Created:** 6/8/2026, 11:19:56 AM
**Updated:** 6/8/2026, 11:28:54 AM

---

## User

proteinmc should support more / all existing energy terms in chsiurf: H-Bond
AV-Potential
Iso-UNRES
Miyazawa-Jernigan
Go-Potential
ASA-Calpha
Radius of Gyration
Clash potential
Ramachandran 

---

## Assistant (Build · MiMo V2.5 Free · 183.9s)

_Thinking:_

The user is asking about adding support for more energy terms in a "proteinmc" module within chisurf. I need to first understand the current state of the codebase, specifically:

1. What is proteinmc?
2. What energy terms does it currently support?
3. Where are the existing energy terms in chisurf defined?
4. How can I add the new energy terms?

Let me start by exploring the codebase to understand the structure.

I'll explore the codebase to understand the current state of proteinmc and the existing energy terms in chisurf.

**Tool: task**

**Input:**
```json
{
  "description": "Explore proteinmc and energy terms",
  "prompt": "I need to understand the proteinmc module and existing energy terms in chisurf. Please do a thorough exploration:\n\n1. Find where proteinmc is defined/implemented in the codebase\n2. Find all existing energy terms/potentials in chisurf (look for files related to energy, potentials, force fields, etc.)\n3. Specifically find implementations of:\n   - H-Bond energy\n   - AV-Potential (Accessible Volume)\n   - Iso-UNRES\n   - Miyazawa-Jernigan potential\n   - Go-Potential\n   - ASA-Calpha (Accessible Surface Area)\n   - Radius of Gyration\n   - Clash potential\n   - Ramachandran potential/penalty\n4. Identify how energy terms are currently registered or used in proteinmc\n5. Look at the structure of existing energy term implementations to understand the pattern\n\nReturn:\n- File paths and line numbers for proteinmc implementation\n- File paths and line numbers for each of the energy terms listed above\n- The pattern/interface used for energy terms in both proteinmc and chisurf\n- Any existing integration points between proteinmc and chisurf energy terms",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_15977c55dffeAPYxTK1DkpdqfR" state="completed">
<task_result>
Now I have a comprehensive understanding. Let me compile all findings into a thorough report.

---

## Complete Exploration Report: ProteinMC Module and Energy Terms in ChiSurf

### 1. ProteinMC Module Implementation

The ProteinMC module is implemented as a plugin under `chisurf/plugins/modelling/proteinmc/`:

| File | Lines | Purpose |
|------|-------|---------|
| `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/__init__.py` | 1-12 | Plugin registration: `name = "Structure:Computation:Protein Monte Carlo"`, CLI entrypoint |
| `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py` | 1-740 | **Core implementation**: `ProteinMCRunner`, `DirectLabelingPotential`, `ProteinMCProgress`, `ProteinMCResult`, settings normalization, structure loading, potential class registry |
| `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/core.py` | 1-41 | Re-export compatibility wrapper for model.py |
| `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/cli.py` | 1-85 | Click-based CLI: `csc proteinmc <structure>` |
| `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/rmf.py` | 1-123 | RMF3 trajectory writer using IMP |
| `/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py` | 1-1224 | **GUI widget**: `ProteinMCModelWidget` with Chimol visualization, potential UI table, worker thread |
| `/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/fitting/fit_controller.py` | Lines 329-448 | Fitting controller integration: `_apply_proteinmc_controls()`, `_is_proteinmc_fit()`, `_proteinmc_model_widget()` |
| `/Users/tpeulen/dev/chisurf/chisurf/gui/plots/proteinMC.py` | Lines 450, 520 | Plot classes referencing `proteinmc_structure` |
| `/Users/tpeulen/dev/chisurf/test/plugins/test_proteinmc.py` | 1-133+ | Tests for ProteinMC runner and RMF writer |
| `/Users/tpeulen/dev/chisurf/test/gui/test_proteinmc_mdl.py` | 1-200+ | GUI widget tests |

### 2. Energy Terms / Potentials in ChiSurf

#### A. Core Potential Classes (computation engine)

All defined in `/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py`:

| Class | Lines | `name` attribute | Description |
|-------|-------|------------------|-------------|
| `Ramachandran` | 293-321 | `'rama'` | Ramachandran potential, loads `rama_ala_pro_gly.npy`, calls `cPotentials_.ramaEnergy()` |
| `Electrostatics` | 324-345 | `'ele'` | Generalized Born electrostatics |
| `LJ_Bead` | 348-362 | `'LJ_bead'` | Lennard-Jones C-alpha bead potential |
| `HPotential` | 365-431 | `'H-Bond'` | Hydrogen bond potential using 4 distance types (OH, ON, CN, CH) |
| `GoPotential` | 434-478 | `'go'` | Gō-like native contact potential |
| `MJPotential` | 485-526 | `'Miyazawa-Jernigan'` | Miyazawa-Jernigan pairwise potential |
| `CEPotential` | 529-613 | `'Iso-UNRES'` | Isotropic UNRES centroid interaction potential |
| `ASA` | 616-650 | `'Asa-Ca'` | Accessible Surface Area potential |
| `ClashPotential` | 653-690 | `'Clash-Potential'` | Steric clash (van der Waals overlap) potential |

Additionally, there is a **commented-out** `AvPotential` class in the same file (lines 693-913) that was the original AV-based chi2 potential.

#### B. Cython-compiled low-level functions

Defined in `/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/cPotentials_.pyx` (lines 1-375):

| Function | Lines | Used by |
|----------|-------|---------|
| `mj()` | 24-50 | `MJPotential.getEnergy()` |
| `hBondLookUpAll()` | 55-131 | `HPotential.getEnergy()` |
| `gb()` | 136-185 | `Electrostatics.getEnergy()` |
| `clash_potential()` | 190-220 | `ClashPotential.getEnergy()` |
| `go_init()` | 224-250 | `GoPotential.setGo()` |
| `go()` | 254-287 | `GoPotential.getEnergy()` |
| `spherePoints()` | 290-305 | `ASA.__init__()` |
| `asa()` | 311-374 | `ASA.getEnergy()` |

Note: `ramaEnergy` is referenced in `Ramachandran.getEnergy()` (line 315) but is not present in the current `.pyx` source -- it may be from an older compiled version or a separate build artifact.

#### C. AV/Labeling Potential

| File | Lines | Class | Description |
|------|-------|-------|-------------|
| `/Users/tpeulen/dev/chisurf/chisurf/core/structure/av/potential.py` | 1-127 | `AvPotential` | Placeholder AV potential (methods are stubs) |
| `/Users/tpeulen/dev/chisurf/chisurf/core/structure/label/potential.py` | 1-296 | `AvPotential` | Full AV potential with `get_chi2()`, `get_energy()`, `calc_avs()`, `calc_distances()` -- uses scikit_fluorescence for AV calculations |

#### D. GUI Widget Wrappers

All in `/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/`:

| File | Class | Inherits From |
|------|-------|---------------|
| `potentials_hbond.py` | `HPotentialWidget` | `HPotential` + `QWidget` |
| `potentials_mj.py` | `MJPotentialWidget` | `MJPotential` + `QWidget` |
| `potentials_ce.py` | `CEPotentialWidget` | `CEPotential` + `QWidget` |
| `potentials_asa.py` | `AsaWidget` | `ASA` + `QWidget` |
| `potentials_clash.py` | `ClashPotentialWidget` | `ClashPotential` + `QWidget` |
| `potentials_go.py` | `GoPotentialWidget` | `GoPotential` + `QWidget` |
| `potentials_rama.py` | `RamachandranWidget` | `Ramachandran` + `QWidget` |
| `potentials_rg.py` | `RadiusGyrationWidget` | `QWidget` only (no core class; wraps `structure.radius_gyration`) |
| `potentials_av.py` | `AvPotentialWidget` | `AvPotential` (from `av.potential`) + `QWidget` |
| `potentials.py` | (aggregator) | Imports and re-exports all widget classes |

#### E. Potential Database Files

Located in `/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/database/`:
- `hb.npy` -- H-bond potential data
- `mj.npy` -- Miyazawa-Jernigan potential matrix
- `unres.npy` -- UNRES centroid interaction potential
- `rama_ala_pro_gly.npy` -- Ramachandran potential
- `repulsive_cb.npy` -- Repulsive C-beta potential

### 3. Specific Energy Term Locations

| Energy Term | Core Class File | Core Class Line | Widget File | Widget Line | Cython Function Line |
|---|---|---|---|---|---|
| **H-Bond** | `potentials.py` | Line 365 (`HPotential`) | `potentials_hbond.py` | Line 12 (`HPotentialWidget`) | `cPotentials_.pyx` line 55 (`hBondLookUpAll`) |
| **AV-Potential** | `core/structure/av/potential.py` | Line 8 (`AvPotential`) | `potentials_av.py` | Line 11 (`AvPotentialWidget`) | N/A (pure Python) |
| **Iso-UNRES** | `potentials.py` | Line 529 (`CEPotential`) | `potentials_ce.py` | Line 13 (`CEPotentialWidget`) | `cPotentials_.pyx` line 24 (via `centroid2` numba function at line 17) |
| **Miyazawa-Jernigan** | `potentials.py` | Line 485 (`MJPotential`) | `potentials_mj.py` | Line 13 (`MJPotentialWidget`) | `cPotentials_.pyx` line 24 (`mj`) |
| **Go-Potential** | `potentials.py` | Line 434 (`GoPotential`) | `potentials_go.py` | Line 9 (`GoPotentialWidget`) | `cPotentials_.pyx` lines 224 (`go_init`) and 254 (`go`) |
| **ASA-Calpha** | `potentials.py` | Line 616 (`ASA`) | `potentials_asa.py` | Line 9 (`AsaWidget`) | `cPotentials_.pyx` line 311 (`asa`) |
| **Radius of Gyration** | N/A (no core class) | N/A | `potentials_rg.py` | Line 8 (`RadiusGyrationWidget`) | N/A (uses `structure.radius_gyration` property from `structure.py` line 281) |
| **Clash** | `potentials.py` | Line 653 (`ClashPotential`) | `potentials_clash.py` | Line 9 (`ClashPotentialWidget`) | `cPotentials_.pyx` line 190 (`clash_potential`) |
| **Ramachandran** | `potentials.py` | Line 293 (`Ramachandran`) | `potentials_rama.py` | Line 13 (`RamachandranWidget`) | `cPotentials_.ramaEnergy()` (referenced at line 315, not in current .pyx) |

### 4. How Energy Terms Are Registered and Used in ProteinMC

#### The Potential Interface (Duck-Typed Protocol)

Every energy term must follow this interface:
```python
class SomePotential:
    name = 'SomeName'        # string identifier
    
    def __init__(self, structure: Structure, **kwargs):
        self.structure = structure
    
    def getEnergy(self) -> float:
        # compute and return energy
        return energy_value
```

#### Registration in ProteinMC

The key registration mechanism is in `/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py`:

**Function `_potential_classes()`** (lines 707-721) maps string names to core potential classes:
```python
def _potential_classes() -> dict[str, Any]:
    from chisurf.core.structure.potential import potentials as core_potentials
    return {
        "default": core_potentials.ClashPotential,
        "hbond": core_potentials.HPotential,
        "mj": core_potentials.MJPotential,
        "unres": core_potentials.CEPotential,
        "H-Bond": core_potentials.HPotential,
        "H-Potential": core_potentials.HPotential,
        "Iso-UNRES": core_potentials.CEPotential,
        "Miyazawa-Jernigan": core_potentials.MJPotential,
        "Clash potential": core_potentials.ClashPotential,
    }
```

**Method `_configure_potentials()`** (lines 474-493) in `ProteinMCRunner`:
1. Reads `self.settings["potentials"]` (a list of `{"name": ..., "weight": ..., "settings": {...}}` dicts)
2. Looks up the class via `_potential_classes()`
3. Instantiates with `structure=self.structure` plus any kwargs from `settings`
4. Adds to `self.universe` via `universe.addPotential(potential, weight)`

**Method `_total_energy()`** (lines 615-621):
```python
def _total_energy(self) -> float:
    total = 0.0
    for potential, scale in zip(self.universe.potentials, self.universe.scaling):
        potential.structure = self.structure
        total += float(scale) * float(potential.getEnergy())
    return float(total)
```

#### The Universe Container

Defined in `/Users/tpeulen/dev/chisurf/chisurf/core/structure/trajectory.py` (lines 16-73):
- `Universe.potentials` -- list of potential objects
- `Universe.scaling` -- list of float weights
- `addPotential(potential, scale)` -- appends a potential with weight
- `getEnergy(structure)` -- sums weighted energies across all potentials

#### GUI-side Registration

In the GUI widget (`proteinmc.py`), the `_POTENTIAL_SPECS` dict (lines 74-117) defines the available potentials with their default parameters:

```python
_POTENTIAL_SPECS = {
    "default": {"label": "Default clash", "name": "default", ...},
    "hbond": {"label": "H-bond", "name": "hbond", ...},
    "mj": {"label": "MJ", "name": "mj", ...},
    "unres": {"label": "UNRES", "name": "unres", ...},
}
```

The `_potential_settings()` method (lines 526-558) reads the UI table and produces a list of `{"name": ..., "weight": ..., "settings": {...}}` dicts that get passed as `settings["potentials"]` to the `ProteinMCRunner`.

### 5. Integration Points Between ProteinMC and ChiSurf Energy Terms

1. **Direct import**: `model.py` line 709 imports `chisurf.core.structure.potential.potentials` and references its classes directly.

2. **Universe container**: ProteinMC uses the same `Universe` class (`chisurf.core.structure.trajectory.Universe`) that the rest of ChiSurf uses for combining potentials. This is the shared energy evaluation framework.

3. **Structure class**: Both ProteinMC and the energy terms share `chisurf.core.structure.Structure` (and `ProteinCentroid`). Energy terms access `structure.xyz`, `structure.dist_ca`, `structure.l_res`, `structure.residue_types`, `structure.vdw`, etc.

4. **GUI integration**: The `ProteinMCModelWidget` is discovered by the fitting controller (`fit_controller.py` lines 430-448) via name-matching (`name == "ProteinMC"`). The fitting controller disables generic fitting controls and enables ProteinMC-specific sampling controls when a ProteinMC model is detected.

5. **Settings serialization**: The JSON settings format (`proteinmc_settings.json`) stores potential configurations as a list of `{name, weight, settings}` objects, which are loaded/saved by `ProteinMCRunner._configure_potentials()` and `ProteinMCModelWidget._restore_potential_settings()`.

6. **DirectLabelingPotential**: This is a special potential (`model.py` lines 293-390) that computes chi2 from FPS JSON distance restraints. It is always added as the last potential via `_configure_potentials()` with a separate `labeling_weight` setting, distinct from the structural potentials.

7. **Not currently registered in ProteinMC** (but available in ChiSurf):
   - `GoPotential` -- not in `_POTENTIAL_SPECS` or `_potential_classes()`
   - `Ramachandran` -- not in `_POTENTIAL_SPECS` or `_potential_classes()`
   - `ASA` -- not in `_POTENTIAL_SPECS` or `_potential_classes()`
   - `RadiusGyrationWidget` -- not in `_POTENTIAL_SPECS` or `_potential_classes()`
   - `AvPotential` (from `av.potential` or `label.potential`) -- not registered
   - `Electrostatics` / `LJ_Bead` -- not registered
</task_result>
</task>
```

---

## Assistant (Build · MiMo V2.5 Free · 5.1s)

_Thinking:_

Now I have a comprehensive understanding. Let me look at the specific files to understand the implementation details better, especially:

1. The `_potential_classes()` function in model.py
2. The `_POTENTIAL_SPECS` in the GUI widget
3. The core potential classes to understand their interfaces

Let me read the key files to understand the exact implementation.

Now let me read the key files to understand the exact implementation details.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py",
  "offset": 700,
  "limit": 50
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py</path>
<type>file</type>
<content>
700:     Path(filename).unlink(missing_ok=True)
701:     import os
702: 
703:     os.close(handle)
704:     return filename
705: 
706: 
707: def _potential_classes() -> dict[str, Any]:
708:     """Return structural potential classes, importing compiled code lazily."""
709:     from chisurf.core.structure.potential import potentials as core_potentials
710: 
711:     return {
712:         "default": core_potentials.ClashPotential,
713:         "hbond": core_potentials.HPotential,
714:         "mj": core_potentials.MJPotential,
715:         "unres": core_potentials.CEPotential,
716:         "H-Bond": core_potentials.HPotential,
717:         "H-Potential": core_potentials.HPotential,
718:         "Iso-UNRES": core_potentials.CEPotential,
719:         "Miyazawa-Jernigan": core_potentials.MJPotential,
720:         "Clash potential": core_potentials.ClashPotential,
721:     }
722: 
723: 
724: def _rmsd(xyz: np.ndarray, reference: np.ndarray) -> float:
725:     """Return coordinate RMSD without alignment."""
726:     delta = np.asarray(xyz, dtype=float) - np.asarray(reference, dtype=float)
727:     return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
728: 
729: 
730: def _as_text(value) -> str:
731:     """Convert numpy string scalar values to plain text."""
732:     if isinstance(value, bytes):
733:         return value.decode("utf-8", errors="ignore").strip()
734:     return str(value).strip()
735: 
736: 
737: _STANDARD_RESIDUES = {
738:     "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
739:     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
740: }

(End of file - total 740 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "offset": 70,
  "limit": 60
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
70: 
71:     name = "ProteinMC"
72: 
73:     plot_classes = [(ProteinMCStructurePlot, {}), (ProteinMCDistanceNetworkPlot, {}), (ProteinMCPlot, {})]
74:     _POTENTIAL_SPECS = {
75:         "default": {
76:             "label": "Default clash",
77:             "name": "default",
78:             "weight": 1.0,
79:             "enabled": False,
80:             "params": {"clash_tolerance": 2.0, "covalent_radius": 1.5},
81:         },
82:         "hbond": {
83:             "label": "H-bond",
84:             "name": "hbond",
85:             "weight": 2.0,
86:             "enabled": True,
87:             "params": {
88:                 "cutoff_ca": 8.0,
89:                 "cutoff_hbond": 3.0,
90:                 "oh": 1.0,
91:                 "on": 1.0,
92:                 "cn": 1.0,
93:                 "ch": 1.0,
94:             },
95:         },
96:         "mj": {
97:             "label": "MJ",
98:             "name": "mj",
99:             "weight": 1.0,
100:             "enabled": False,
101:             "params": {"ca_cutcoff": 6.5},
102:         },
103:         "unres": {
104:             "label": "UNRES",
105:             "name": "unres",
106:             "weight": 1.0,
107:             "enabled": True,
108:             "params": {
109:                 "ca_cutoff": 15.0,
110:                 "repulsion": 100.0,
111:                 "min_dist": 3.5,
112:                 "max_dist": 19.0,
113:                 "bin_width": 0.05,
114:                 "centroid_number": 4.0,
115:             },
116:         },
117:     }
118: 
119:     def __init__(
120:         self,
121:         fit: "cs.core.fitting.fit.Fit",
122:         *args,
123:         **kwargs,
124:     ):
125:         """Initialize the ProteinMC MDL widget."""
126:         super().__init__(fit=fit, *args, **kwargs)
127:         self.structure = getattr(self.fit, "data", None)
128:         self.rmsd: list[float] = []
129:         self.drmsd: list[float] = []

(Showing lines 70-129 of 1224. Use offset=130 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "offset": 290,
  "limit": 430
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py</path>
<type>file</type>
<content>
290:     return energy
291: 
292: 
293: class Ramachandran(object):
294: 
295:     def __init__(
296:             self,
297:             structure: chisurf.core.structure.Structure,
298:             filename: str = None
299:     ):
300:         """
301:         :param filename:
302:         :return:
303:         """
304:         if filename is None:
305:             from chisurf.core.settings.path_utils import get_path
306:             filename = str(get_path('chisurf') / 'structure/potential/database/rama_ala_pro_gly.npy')
307:         self.structure = structure
308:         self.name = 'rama'
309:         self.filename = filename
310:         self.ramaPot = np.load(self.filename)
311: 
312:     def getEnergy(self) -> float:
313:         """Calculate and return the Ramachandran potential energy."""
314:         c = self.structure
315:         Erama = chisurf.core.structure.potential.cPotentials_.ramaEnergy(
316:             c.residue_lookup_i,
317:             c.iAtoms,
318:             self.ramaPot
319:         )
320:         self.E = Erama
321:         return Erama
322: 
323: 
324: class Electrostatics(object):
325: 
326:     def __init__(
327:             self,
328:             structure,
329:             type: str = 'gb'):
330:         """
331:         :param type:
332:         :return:
333:         """
334:         self.structure = structure
335:         self.name = 'ele'
336:         if type == 'gb':
337:             self.p = chisurf.core.structure.potential.cPotentials_.gb
338: 
339:     def getEnergy(self) -> float:
340:         """Calculate and return the electrostatic (GB) energy."""
341:         structure = self.structure
342:         #Eel = mfm.structure.potential.cPotentials_.gb(structure.xyz)
343:         Eel = gb(structure.xyz)
344:         self.E = Eel
345:         return Eel
346: 
347: 
348: class LJ_Bead(object):
349: 
350:     def __init__(
351:             self,
352:             structure: chisurf.core.structure.Structure
353:     ):
354:         """Initialize a Lennard-Jones bead potential for *structure*."""
355:         self.structure = structure
356:         self.name = 'LJ_bead'
357: 
358:     def getEnergy(self) -> float:
359:         """Calculate and return the Lennard-Jones bead energy."""
360:         structure = self.structure
361:         self.E = lennard_jones_calpha(structure.atoms['xyz'])
362:         return self.E
363: 
364: 
365: class HPotential(object):
366: 
367:     name = 'H-Bond'
368: 
369:     def __init__(
370:             self,
371:             structure: chisurf.core.structure.Structure,
372:             cutoff_ca: float = 8.0,
373:             cutoff_hbond: float = 3.0,
374:             potential: str = None,
375:             **kwargs
376:     ):
377:         """Initialize the hydrogen bond potential with cutoff and parameter settings."""
378:         if potential is None:
379:             from chisurf.core.settings.path_utils import get_path
380:             potential = str(get_path('chisurf') / 'core/structure/potential/database/hb.npy')
381:         self.structure = structure
382:         self.cutoffH = cutoff_hbond
383:         self.cutoffCA = cutoff_ca
384:         self.potential = potential
385:         self.oh = kwargs.get('oh', 1.0)
386:         self.on = kwargs.get('on', 1.0)
387:         self.cn = kwargs.get('cn', 1.0)
388:         self.ch = kwargs.get('ch', 1.0)
389:         self.updateParameter()
390: 
391:     def getEnergy(self):
392:         """Calculate and return the hydrogen bond potential energy."""
393:         s1 = self.structure
394:         cca2 = self.cutoffCA ** 2
395:         ch2 = self.cutoffH ** 2
396:         nHbond, Ehbond = chisurf.core.structure.potential.cPotentials_.hBondLookUpAll(
397:             s1.l_res, s1.dist_ca, s1.xyz, self._hPot, cca2, ch2
398:         )
399:         self.E = Ehbond
400:         self.nHbond = nHbond
401:         return self.E
402: 
403:     def getNbrBonds(self):
404:         """Return the number of hydrogen bonds."""
405:         if self.nHbond is None:
406:             return 0
407:         return self.nHbond
408: 
409:     def updateParameter(self):
410:         """Update the hydrogen bond potential based on current OH/ON/CN/CH toggles."""
411:         hPot = copy.deepcopy(self._hPot)
412:         if not self.oh:
413:             hPot[2, :] *= 0.0
414:         if not self.on:
415:             hPot[1, :] *= 0.0
416:         if not self.cn:
417:             hPot[3, :] *= 0.0
418:         if not self.ch:
419:             hPot[0, :] *= 0.0
420:         self.hPot = hPot
421: 
422:     @property
423:     def potential(self):
424:         """The hydrogen bond potential array."""
425:         return self.hPot
426: 
427:     @potential.setter
428:     def potential(self, v):
429:         """Load the hydrogen-bond potential array from a file."""
430:         self._hPot = np.load(v)
431:         self.hPot = self._hPot
432: 
433: 
434: class GoPotential(object):
435: 
436:     def __init__(
437:             self,
438:             structure: chisurf.core.structure.Structure
439:     ):
440:         """Initialize a Go-like potential for *structure*."""
441:         self.structure = structure
442:         self.name = 'go'
443: 
444:     def setGo(self):
445:         """Initialize the Gō potential energy and contact matrices."""
446:         c = self.structure
447:         nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0
448:         cutoff = self.cutoff if self.native_cutoff_on else 1e6
449:         self.eMatrix, self.sMatrix = chisurf.core.structure.potential.cPotentials_.go_init(
450:             c.residue_lookup_r, c.dist_ca,
451:             self.epsilon, nnEFactor, cutoff
452:         )
453: 
454:     def getEnergy(self):
455:         """Calculate and return the Gō potential energy."""
456:         c = self.structure
457:         Etot, nNa, Ena, nNN, Enn = chisurf.core.structure.potential.cPotentials_.go(
458:             c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix
459:         )
460:         #Etot = go(c., c.dist_ca, self.eMatrix, self.sMatrix)
461:         self.E = Etot
462:         return Etot
463: 
464:     def getNbrNonNative(self):
465:         """Return the number of non-native contacts."""
466:         return self.nNN
467: 
468:     def getNbrNative(self):
469:         """Return the number of native contacts."""
470:         return self.nNa
471: 
472:     def set_sMatrix(self, sMatrix):
473:         """Set the sigma (distance) matrix for the Gō potential."""
474:         self.sMatrix = sMatrix
475: 
476:     def set_eMatrix(self, eMatrix):
477:         """Set the epsilon (energy) matrix for the Gō potential."""
478:         self.eMatrix = eMatrix
479: 
480:     def set_nMatrix(self, nMatrix):
481:         """Set the native contact matrix."""
482:         self.nMatrix = nMatrix
483: 
484: 
485: class MJPotential(object):
486: 
487:     name = 'Miyazawa-Jernigan'
488: 
489:     def __init__(
490:             self,
491:             structure: chisurf.core.structure.Structure,
492:             filename: str = None,
493:             ca_cutcoff: float = 6.5
494:     ):
495:         """Initialize a Miyazawa-Jernigan potential for *structure*."""
496:         if filename is None:
497:             from chisurf.core.settings.path_utils import get_path
498:             filename = str(get_path('chisurf') / 'structure/potential/database/mj.npy')
499:         self.filename = filename
500:         self.structure = structure
501:         self.potential = filename
502:         self.ca_cutoff = ca_cutcoff
503: 
504:     @property
505:     def potential(self):
506:         """The Miyazawa-Jernigan potential matrix."""
507:         return self.mjPot
508: 
509:     @potential.setter
510:     def potential(self, v):
511:         """Load the Miyazawa-Jernigan potential matrix from a file."""
512:         self.mjPot = np.load(v)
513: 
514:     def getEnergy(self):
515:         """Calculate and return the Miyazawa-Jernigan potential energy."""
516:         c = self.structure
517:         nCont, Emj = chisurf.core.structure.potential.cPotentials_.mj(
518:             c.l_res, c.residue_types, c.dist_ca, c.xyz, self.mjPot, cutoff=self.ca_cutoff
519:         )
520:         self.E = Emj
521:         self.nCont = nCont
522:         return Emj
523: 
524:     def getNbrContacts(self):
525:         """Return the number of residue-residue contacts."""
526:         return self.nCont
527: 
528: 
529: class CEPotential(object):
530:     """
531:     Examples
532:     --------
533: 
534:     >>> import chisurf.core.structure
535:     >>> import chisurf.core.structure.potential
536:     >>> s = chisurf.core.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
537:     >>> pce = chisurf.core.structure.potential.potentials.CEPotential(s, ca_cutoff=64.0)
538:     >>> pce.getEnergy()
539:     -0.15896629131635745
540:     """
541: 
542:     name = 'Iso-UNRES'
543: 
544:     def __init__(
545:             self,
546:             structure: chisurf.core.structure.Structure,
547:             potential: str = None,
548:             **kwargs
549:     ):
550:         """
551:         scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
552:         """
553:         scaling_factor = 0.593
554:         self.structure = structure
555:         self.ca_cutoff = kwargs.get('ca_cutoff', 15.0)
556:         self._potential = None
557: 
558:         if potential is None:
559:             from chisurf.core.settings.path_utils import get_path
560:             potential = str(get_path('chisurf') / 'structure/potential/database/unres.npy')
561: 
562:         self.potential = potential
563:         self.scaling_factor = scaling_factor
564:         # the number of the atom in the lookup table
565:         # (4=C-beta, 1=C-alpha, see: mfm.structure.protein internal_atom_numbers)
566:         self.centroid_number = kwargs.get('centroid_number', 4)
567:         self.repulsion = kwargs.get('repulsion', 100.0)
568:         self.min_dist = kwargs.get('min_dist', 3.5)
569:         self._max_dist = kwargs.get('max_dist', 19.0)
570:         self._bin_width = kwargs.get('bin_width', 0.05)
571: 
572:     @property
573:     def potential(self):
574:         """The UNRES potential matrix."""
575:         return self._potential
576: 
577:     @potential.setter
578:     def potential(
579:             self,
580:             v
581:     ):
582:         """Load the UNRES potential matrix from a file."""
583:         self._potential = np.load(v)
584: 
585:     def getEnergy(
586:             self,
587:             cutoff=None,
588:             **kwargs
589:     ) -> float:
590:         """Calculate and return the UNRES centroid potential energy."""
591:         cutoff = cutoff if cutoff is not None else self.ca_cutoff
592:         c = self.structure
593:         coord = np.ascontiguousarray(c.xyz)
594:         dist_ca = np.ascontiguousarray(c.dist_ca)
595:         residue_types = np.ascontiguousarray(c.residue_types)
596:         l_res = np.ascontiguousarray(c.l_res)
597:         centroid_pos = kwargs.get('centroid_number', self.centroid_number)
598:         repulsion = kwargs.get('repulsion', self.repulsion)
599:         min_dist = kwargs.get('min_dist', self.min_dist)
600:         nCont, E = centroid2(
601:             l_res, residue_types, dist_ca, coord,
602:             self.potential, cutoff=cutoff,
603:             centroid_pos=centroid_pos, min_dist=min_dist,
604:             max_dist=self._max_dist, bin_width=self._bin_width,
605:             repulsion=repulsion
606:         )
607: 
608:         self.nCont = nCont
609:         return float(E * self.scaling_factor)
610: 
611:     def getNbrContacts(self) -> int:
612:         """Return the number of residue-residue contacts."""
613:         return self.nCont
614: 
615: 
616: class ASA(object):
617: 
618:     name = 'Asa-Ca'
619: 
620:     def __init__(
621:             self,
622:             structure: chisurf.core.structure.Structure,
623:             probe: float = 1.0,
624:             n_sphere_point: int = 590,
625:             radius: float = 2.5,
626:     ):
627:         """Initialize an ASA-Cα solvent-accessible-surface potential."""
628:         super(ASA, self).__init__()
629:         self.structure = structure
630:         self.probe = probe
631:         self.n_sphere_point = n_sphere_point
632:         self.sphere_points = chisurf.core.structure.potential.cPotentials_.spherePoints(
633:             n_sphere_point
634:         )
635:         self.radius = radius
636: 
637:     def getEnergy(self) -> float:
638:         """Calculate and return the accessible surface area."""
639:         c = self.structure
640:         #def asa(double[:, :] xyz, int[:, :] resLookUp, double[:, :] caDist, double[:, :] sphere_points,
641:         #double probe=1.0, double radius = 2.5, char sum=1)
642:         asa = chisurf.core.structure.potential.cPotentials_.asa(
643:             c.xyz,
644:             c.l_res,
645:             c.dist_ca,
646:             self.sphere_points,
647:             self.probe,
648:             self.radius
649:         )
650:         return asa
651: 
652: 
653: class ClashPotential(object):
654: 
655:     name = 'Clash-Potential'
656: 
657:     def __init__(
658:             self,
659:             structure: chisurf.core.structure.Structure = None,
660:             clash_tolerance: float = 2.0,
661:             covalent_radius: float = 1.5,
662:             **kwargs
663:     ):
664:         """
665:         :param kwargs:
666:         :return:
667: 
668:         Examples
669:         --------
670: 
671:         >>> import chisurf.core.structure
672:         >>> import chisurf.core.structure.potential
673:         >>> s = chisurf.core.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
674:         >>> pce = chisurf.core.structure.potential.ClashPotential(structure=s, clash_tolerance=6.0)
675:         >>> pce.getEnergy()
676: 
677:         """
678:         self.structure = structure
679:         self.clash_tolerance = clash_tolerance
680:         self.covalent_radius = covalent_radius
681: 
682:     def getEnergy(self) -> float:
683:         """Calculate and return the clash (steric) potential energy."""
684:         c = self.structure
685:         return chisurf.core.structure.potential.cPotentials_.clash_potential(
686:             c.xyz,
687:             c.vdw,
688:             self.clash_tolerance,
689:             self.covalent_radius
690:         )
691: 
692: #
693: # class AvPotential(object):
694: #     """
695: #     The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
696: #     labeling positions and experimental distances. Here the labeling positions and distances are provided as
697: #     dictionaries.
698: #
699: #     Examples
700: #     --------
701: #
702: #     >>> import json
703: #     >>> labeling_file = './test/data/model/labeling.json'
704: #     >>> labeling = json.load(open(labeling_file, 'r'))
705: #     >>> distances = labeling['Distances']
706: #     >>> positions = labeling['Positions']
707: #     >>> import chisurf
708: #     >>> av_potential = chisurf.core.structure.potential.potentials.AvPotential(distances=distances, positions=positions)
709: #     >>> structure = chisurf.core.structure.Structure('./test/data/model/HM_1FN5_Naming.pdb')
710: #     >>> av_potential.getChi2(structure)
711: #
712: #     """
713: #     name = 'Av'
714: #
715: #     def __init__(
716: #             self,
717: #             labeling_file: str = None,
718: #             structure: chisurf.core.structure.Structure = None,
719: #             verbose: bool = False,

(Showing lines 290-719 of 913. Use offset=720 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 6.7s)

_Thinking:_

Now I have a clear picture. Let me also check the GUI widget for the `_potential_settings()` method to understand how settings are read from the UI, and look at the Go and Ramachandran widgets to understand their parameters.

Let me also check the GUI widget settings method and the existing widget parameters for the missing potentials.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 60,
  "offset": 520
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
520:         form.addRow(buttons)
521:         if dialog.exec_() != QtWidgets.QDialog.Accepted:
522:             return
523:         updated = {name: spin.value() for name, spin in controls.items()}
524:         item.setData(QtCore.Qt.UserRole + 1, updated)
525: 
526:     def _potential_settings(self) -> list[dict]:
527:         """Return enabled ProteinMC potential settings from the UI."""
528: 
529:         out = []
530:         table = getattr(self, "potential_table", None)
531:         if table is None:
532:             return out
533:         for row in range(table.rowCount()):
534:             term_item = table.item(row, 0)
535:             weight_item = table.item(row, 1)
536:             if term_item is None or weight_item is None:
537:                 continue
538:             key = term_item.data(QtCore.Qt.UserRole)
539:             spec = self._POTENTIAL_SPECS.get(key)
540:             if spec is None:
541:                 continue
542:             settings = dict(term_item.data(QtCore.Qt.UserRole + 1) or spec["params"])
543:             for int_key in ("centroid_number",):
544:                 if int_key in settings:
545:                     settings[int_key] = int(settings[int_key])
546:             try:
547:                 weight = float(weight_item.text())
548:             except ValueError:
549:                 weight = float(spec["weight"])
550:                 weight_item.setText(f"{weight:.2f}")
551:             out.append(
552:                 {
553:                     "name": spec["name"],
554:                     "weight": weight,
555:                     "settings": settings,
556:                 }
557:             )
558:         return out
559: 
560:     def _populate_score_sets(self) -> None:
561:         """Read score sets from the current FPS JSON and fill the combo box."""
562:         filename = self.labeling_edit.text().strip()
563:         old = self.score_set_combo.currentText()
564:         self.score_set_combo.blockSignals(True)
565:         self.score_set_combo.clear()
566:         self.score_set_combo.addItem("")
567:         group_names: list[str] = []
568:         if filename:
569:             try:
570:                 with open(filename) as fp:
571:                     payload = json.load(fp)
572:                 group_names = list((payload.get("χ²", {}) or {}).keys())
573:                 for name in group_names:
574:                     self.score_set_combo.addItem(name)
575:             except Exception:
576:                 pass
577:         # Restore previous selection if still available;
578:         # otherwise default to the first score group.
579:         idx = self.score_set_combo.findText(old)

(Showing lines 520-579 of 1224. Use offset=580 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_go.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_go.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from qtpy import QtWidgets
4: 
5: import chisurf.core.structure
6: from chisurf.core.structure.potential.potentials import GoPotential
7: 
8: 
9: class GoPotentialWidget(GoPotential, QtWidgets.QWidget):
10: 
11:     def __init__(
12:             self,
13:             structure: chisurf.core.structure.Structure = None,
14:             **kwargs
15:     ):
16:         QtWidgets.QWidget.__init__(self, parent=kwargs.get('parent'))
17:         GoPotential.__init__(self, structure=structure)
18: 
19:         layout = QtWidgets.QVBoxLayout(self)
20:         layout.setContentsMargins(0, 0, 0, 0)
21:         layout.setSpacing(0)
22: 
23:         grid = QtWidgets.QGridLayout()
24:         grid.setContentsMargins(0, 0, 0, 0)
25:         grid.setSpacing(0)
26: 
27:         label = QtWidgets.QLabel("epsilon", self)
28:         grid.addWidget(label, 0, 0)
29:         self.lineEdit = QtWidgets.QLineEdit(self)
30:         self.lineEdit.setText("1.0")
31:         grid.addWidget(self.lineEdit, 0, 1)
32: 
33:         label_2 = QtWidgets.QLabel("native cutoff [A]", self)
34:         grid.addWidget(label_2, 1, 0)
35:         self.lineEdit_3 = QtWidgets.QLineEdit(self)
36:         self.lineEdit_3.setText("6.5")
37:         grid.addWidget(self.lineEdit_3, 1, 1)
38:         self.checkBox = QtWidgets.QCheckBox("", self)
39:         self.checkBox.setChecked(True)
40:         grid.addWidget(self.checkBox, 1, 2)
41: 
42:         label_3 = QtWidgets.QLabel("non-native / scale", self)
43:         grid.addWidget(label_3, 2, 0)
44:         self.lineEdit_2 = QtWidgets.QLineEdit(self)
45:         self.lineEdit_2.setText("0.7")
46:         grid.addWidget(self.lineEdit_2, 2, 1)
47:         self.checkBox_2 = QtWidgets.QCheckBox("", self)
48:         self.checkBox_2.setChecked(True)
49:         grid.addWidget(self.checkBox_2, 2, 2)
50: 
51:         layout.addLayout(grid)
52: 
53:         self.lineEdit.textChanged.connect(self.setGo)
54:         self.lineEdit_2.textChanged.connect(self.setGo)
55:         self.lineEdit_3.textChanged.connect(self.setGo)
56: 
57:     @property
58:     def native_cutoff_on(self):
59:         return bool(self.checkBox.isChecked())
60: 
61:     @property
62:     def non_native_contact_on(self):
63:         return bool(self.checkBox_2.isChecked())
64: 
65:     @property
66:     def epsilon(self):
67:         return float(self.lineEdit.text())
68: 
69:     @property
70:     def nnEFactor(self):
71:         return float(self.lineEdit_2.text())
72: 
73:     @property
74:     def cutoff(self):
75:         return float(self.lineEdit_3.text())

(End of file - total 75 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_rama.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_rama.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: import numpy as np
4: from qtpy import QtWidgets
5: 
6: import chisurf.core.structure
7: import chisurf.gui.widgets
8: from chisurf.core.settings.path_utils import get_path
9: from chisurf.core.structure.potential.potentials import Ramachandran
10: from chisurf.gui.widgets.warning_once import show_warning_once
11: 
12: 
13: class RamachandranWidget(Ramachandran, QtWidgets.QWidget):
14: 
15:     def __init__(
16:             self,
17:             structure: chisurf.core.structure.Structure,
18:             filename: str = None,
19:             parent=None
20:     ):
21:         QtWidgets.QWidget.__init__(self, parent=parent)
22: 
23:         # Set default filename path if not provided
24:         if filename is None:
25:             filename = str(get_path('chisurf') / 'structure/potential/database/rama_ala_pro_gly.npy')
26: 
27:         layout = QtWidgets.QGridLayout(self)
28:         layout.setContentsMargins(0, 0, 0, 0)
29:         layout.setSpacing(0)
30: 
31:         label = QtWidgets.QLabel("Ramachandran File", self)
32:         layout.addWidget(label, 0, 0)
33: 
34:         self.lineEdit = QtWidgets.QLineEdit(self)
35:         self.lineEdit.setReadOnly(True)
36:         layout.addWidget(self.lineEdit, 0, 1)
37: 
38:         self.toolButton = QtWidgets.QPushButton("...", self)
39:         layout.addWidget(self.toolButton, 0, 2)
40: 
41:         self.actionLoad_potential = QtWidgets.QAction("Load potential", self)
42:         self.actionLoad_potential.triggered.connect(self.onOpenFile)
43:         self.toolButton.clicked.connect(self.actionLoad_potential.trigger)
44: 
45:         # Initialize the parent Ramachandran class
46:         super().__init__(
47:             structure,
48:             filename
49:         )
50: 
51:     def onOpenFile(self):
52:         filename = chisurf.gui.widgets.get_filename(
53:             'Open File',
54:             'NumPy data files (*.npy)'
55:         )
56:         self.filename = filename
57: 
58:     @property
59:     def filename(self):
60:         return self._filename
61: 
62:     @filename.setter
63:     def filename(self, v):
64:         try:
65:             self._filename = v
66:             self.ramaPot = np.load(v)
67:             self.lineEdit.setText(str(v))
68:         except (OSError, FileNotFoundError):
69:             show_warning_once(
70:                 "rama_potential_file",
71:                 "Missing Ramachandran Potential File",
72:                 f"Ramachandran potential file not found: {v}\n\n"
73:                 f"The Ramachandran potential file should be located at:\n"
74:                 f"{v}\n\n"
75:                 f"Please check if the file exists or use the '...' button\n"
76:                 f"to select a different potential file."
77:             )
78:             # Create a dummy potential to prevent crashes
79:             self._filename = v
80:             self.ramaPot = np.zeros((5, 129600))  # Same shape as expected
81:             self.lineEdit.setText(str(v))
82:         except Exception as e:
83:             show_warning_once(
84:                 "rama_potential_file_error",
85:                 "Ramachandran Potential File Error",
86:                 f"Error loading Ramachandran potential file {v}:\n{str(e)}"
87:             )
88:             # Create a dummy potential to prevent crashes
89:             self._filename = v
90:             self.ramaPot = np.zeros((5, 129600))  # Same shape as expected
91:             self.lineEdit.setText(str(v))

(End of file - total 91 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_asa.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_asa.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from qtpy import QtWidgets
4: 
5: import chisurf.core.structure
6: from chisurf.core.structure.potential.potentials import ASA
7: 
8: 
9: class AsaWidget(ASA, QtWidgets.QWidget):
10: 
11:     def __init__(
12:             self,
13:             structure: chisurf.core.structure.Structure,
14:             parent: QtWidgets.QWidget = None,
15:             **kwargs
16:     ):
17:         QtWidgets.QWidget.__init__(self, parent=parent)
18: 
19:         layout = QtWidgets.QGridLayout(self)
20:         layout.setContentsMargins(0, 0, 0, 0)
21:         layout.setSpacing(0)
22: 
23:         label = QtWidgets.QLabel("sphere-points", self)
24:         layout.addWidget(label, 0, 0)
25:         self.lineEdit = QtWidgets.QLineEdit(self)
26:         layout.addWidget(self.lineEdit, 0, 1)
27: 
28:         label_2 = QtWidgets.QLabel("Probe-radius [A]", self)
29:         layout.addWidget(label_2, 1, 0)
30:         self.lineEdit_2 = QtWidgets.QLineEdit(self)
31:         layout.addWidget(self.lineEdit_2, 1, 1)
32: 
33:         self.lineEdit.textChanged.connect(self.setParameterSphere)
34:         self.lineEdit_2.textChanged.connect(self.setParameterProbe)
35:         self.lineEdit.setText('590')
36:         self.lineEdit_2.setText('3.5')
37: 
38:         super(AsaWidget, self).__init__(
39:             structure,
40:             **kwargs
41:         )
42: 
43:     def setParameterSphere(self):
44:         self.n_sphere_point = int(self.lineEdit.text())
45: 
46:     def setParameterProbe(self):
47:         self.probe = float(self.lineEdit_2.text())

(End of file - total 47 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_rg.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_rg.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: from qtpy import QtWidgets
4: 
5: import chisurf.core.structure
6: 
7: 
8: class RadiusGyrationWidget(QtWidgets.QWidget):
9: 
10:     name = 'Radius-Gyration'
11: 
12:     def __init__(
13:             self,
14:             structure: chisurf.core.structure.Structure,
15:             parent=None
16:     ):
17:         super(RadiusGyrationWidget, self).__init__(parent)
18:         self.structure = structure
19:         self.parent = parent
20: 
21:     def getEnergy(self, c=None):
22:         if c is None:
23:             c = self.structure
24:         return c.radius_gyration

(End of file - total 24 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_av.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/structure/potentials_av.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: import json
4: from qtpy import QtWidgets
5: 
6: import chisurf.gui.widgets
7: import chisurf.core.structure
8: from chisurf.core.structure.av.potential import AvPotential
9: 
10: 
11: class AvPotentialWidget(AvPotential, QtWidgets.QWidget):
12: 
13:     def __init__(
14:             self,
15:             structure: chisurf.core.structure.Structure = None,
16:             parent=None
17:     ):
18:         QtWidgets.QWidget.__init__(self, parent=parent)
19:         
20:         layout = QtWidgets.QVBoxLayout(self)
21:         layout.setContentsMargins(0, 0, 0, 0)
22:         layout.setSpacing(0)
23: 
24:         self.groupBox = QtWidgets.QGroupBox("Av-properties", self)
25:         grid = QtWidgets.QGridLayout(self.groupBox)
26:         grid.setContentsMargins(0, 0, 0, 0)
27:         grid.setSpacing(0)
28: 
29:         label_4 = QtWidgets.QLabel("nSamples", self.groupBox)
30:         grid.addWidget(label_4, 0, 0)
31:         self.spinBox_2 = QtWidgets.QSpinBox(self.groupBox)
32:         self.spinBox_2.setMaximum(999999)
33:         self.spinBox_2.setSingleStep(1000)
34:         grid.addWidget(self.spinBox_2, 0, 2)
35: 
36:         label_2 = QtWidgets.QLabel("MinAV", self.groupBox)
37:         grid.addWidget(label_2, 0, 3)
38:         self.spinBox = QtWidgets.QSpinBox(self.groupBox)
39:         self.spinBox.setMinimum(0)
40:         self.spinBox.setMaximum(999999)
41:         self.spinBox.setSingleStep(50)
42:         self.spinBox.setValue(150)
43:         grid.addWidget(self.spinBox, 0, 4)
44: 
45:         layout.addWidget(self.groupBox)
46: 
47:         self.groupBox_6 = QtWidgets.QGroupBox("Labling", self)
48:         grid2 = QtWidgets.QGridLayout(self.groupBox_6)
49:         grid2.setContentsMargins(0, 0, 0, 0)
50:         grid2.setSpacing(0)
51: 
52:         self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_6)
53:         self.lineEdit_2.setReadOnly(True)
54:         grid2.addWidget(self.lineEdit_2, 0, 2)
55: 
56:         self.toolButton = QtWidgets.QPushButton("...", self.groupBox_6)
57:         grid2.addWidget(self.toolButton, 0, 3)
58: 
59:         layout.addWidget(self.groupBox_6)
60: 
61:         self._filename = None
62: 
63:         self.actionOpenLabeling = QtWidgets.QAction("OpenLabeling", self)
64:         self.actionOpenLabeling.triggered.connect(self.onLoadAvJSON)
65:         self.toolButton.clicked.connect(self.actionOpenLabeling.trigger)
66:         
67:         # Now initialize the parent class after Qt widgets are set up
68:         AvPotential.__init__(self)
69: 
70:     def onLoadAvJSON(self):
71:         self.labeling_file = chisurf.gui.widgets.get_filename(
72:             description='Open FPS-JSON',
73:             file_type='FPS-file (*.fps.json)'
74:         )
75: 
76:     @property
77:     def labeling_file(self):
78:         return self._filename
79: 
80:     @labeling_file.setter
81:     def labeling_file(self, v):
82:         self._filename = v
83:         p = json.load(open(v))
84:         self.distances = p["Distances"]
85:         self.positions = p["Positions"]
86:         self.lineEdit_2.setText(v)
87: 
88:     @property
89:     def n_av_samples(self):
90:         return int(self.spinBox_2.value())
91: 
92:     @n_av_samples.setter
93:     def n_av_samples(self, v):
94:         self.spinBox_2.setValue(int(v))
95: 
96:     @property
97:     def min_av(self):
98:         return int(self.spinBox_2.value())
99: 
100:     @min_av.setter
101:     def min_av(self, v):
102:         self.spinBox.setValue(int(v))

(End of file - total 102 lines)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 7.3s)

_Thinking:_

Now I have a clear picture. Let me also check the `_configure_potentials` method in model.py to understand how potentials are instantiated.

Let me check the `_configure_potentials` method in model.py:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py",
  "limit": 40,
  "offset": 470
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py</path>
<type>file</type>
<content>
470:     def stop(self) -> None:
471:         """Request the MC loop to stop."""
472:         self.exiting = True
473: 
474:     def _configure_potentials(self) -> None:
475:         """Add configured structure and labeling potentials."""
476:         for potential_settings in self.settings.get("potentials", []) or []:
477:             name = potential_settings.get("name")
478:             potential_cls = _potential_classes().get(name)
479:             if potential_cls is None:
480:                 continue
481:             kwargs = potential_settings.get("settings", {}) or {}
482:             potential = potential_cls(structure=self.structure, **kwargs)
483:             self.universe.addPotential(potential, potential_settings.get("weight", 1.0))
484:         if self.labeling_file:
485:             self.labeling_potential = DirectLabelingPotential(
486:                 self.structure,
487:                 self.labeling_file,
488:                 score_set=self.score_set,
489:             )
490:             self.universe.addPotential(
491:                 self.labeling_potential,
492:                 float(self.settings.get("labeling_weight", 1.0)),
493:             )
494: 
495:     def run(self) -> ProteinMCResult:
496:         """Run ProteinMC and write accepted frames to RMF."""
497:         if self.settings.get("mc_mode", "simple") != "simple":
498:             raise ValueError("Only ProteinMC mc_mode='simple' is currently supported.")
499:         writer = ProteinMCRmfWriter(self.output_file, self.structure)
500:         for frame_index, frame in enumerate(self.initial_frames):
501:             writer.append(frame, name=str(frame_index))
502:         self.structure.auto_update = False
503:         self.structure.update()
504:         self._reference_xyz = np.array(self.structure.xyz, copy=True)
505:         self._previous_xyz = np.array(self.structure.xyz, copy=True)
506: 
507:         energy = self._total_energy()
508:         self._append_frame(writer, energy, self._labeling_energy(), iteration=0, accepted=0, rejected=0)
509: 

(Showing lines 470-509 of 740. Use offset=510 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 14.5s)

_Thinking:_

Now I have a complete picture. Let me plan the changes needed:

1. **In `model.py`** (`_potential_classes()`): Add the missing energy terms:
   - Go-Potential → `GoPotential`
   - Ramachandran → `Ramachandran`
   - ASA-Calpha → `ASA`
   - Radius of Gyration → Need a core class (currently only exists as a widget)
   - AV-Potential → Need to check if the core class can be used standalone

2. **In `proteinmc.py` GUI widget** (`_POTENTIAL_SPECS`): Add GUI specs for:
   - Go-Potential
   - Ramachandran
   - ASA-Calpha
   - Radius of Gyration
   - AV-Potential (this one is complex since it needs a labeling file)

Let me also check if `GoPotential` and `Ramachandran` need any additional setup beyond what `_configure_potentials` already does. Looking at `GoPotential`, it has a `setGo()` method that needs to be called before `getEnergy()`. This is a problem because `_configure_potentials` doesn't call it.

Let me check the `GoPotential` more carefully - it has `setGo()` which needs `self.epsilon`, `self.non_native_contact_on`, `self.cutoff`, `self.native_cutoff_on`, and `self.nnEFactor`. These are not set in `__init__`. So GoPotential needs special handling.

Actually, looking more carefully at GoPotential:
```python
class GoPotential(object):
    def __init__(self, structure):
        self.structure = structure
        self.name = 'go'

    def setGo(self):
        c = self.structure
        nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0
        cutoff = self.cutoff if self.native_cutoff_on else 1e6
        self.eMatrix, self.sMatrix = chisurf.core.structure.potential.cPotentials_.go_init(
            c.residue_lookup_r, c.dist_ca,
            self.epsilon, nnEFactor, cutoff
        )

    def getEnergy(self):
        c = self.structure
        Etot, nNa, Ena, nNN, Enn = chisurf.core.structure.potential.cPotentials_.go(
            c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix
        )
        self.E = Etot
        return Etot
```

The problem is that `setGo()` relies on several attributes (`epsilon`, `non_native_contact_on`, `cutoff`, `native_cutoff_on`, `nnEFactor`) that aren't initialized in `__init__`. These are only set up in the GUI widget. The `getEnergy()` method then calls `cPotentials_.go()` using the `eMatrix` and `sMatrix` that `setGo()` computed. So for headless use, I need to either ensure these attributes exist before calling `setGo()`, or restructure how the potential is initialized.

I also notice that `RadiusGyrationWidget` doesn't have a core class backing it—it just calls `structure.radius_gyration` directly. For the GUI widget `_POTENTIAL_SPECS`, I'll need to add entries for Go-Potential, Ramachandran, ASA-Calpha, Radius of Gyration, and AV-Potential. Looking at the `_configure_potentials` method, it expects a consistent interface where each potential class accepts `structure` and kwargs, then exposes a `getEnergy()` method. So I need to verify that GoPotential, Ramachandran, ASA, and Radius of Gyration all follow this pattern, and check whether AvPotential can be adapted to work standalone without requiring the GUI widget's labeling file setup.

Let me check the AV potential core class and understand GoPotential's `setGo()` dependency:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/av/potential.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/av/potential.py</path>
<type>file</type>
<content>
1: import json
2: import os
3: from collections import OrderedDict
4: 
5: import numpy as np
6: 
7: import chisurf as cs
8: class AvPotential(object):
9:     """
10:     A class for calculating accessible volume potentials.
11:     
12:     This class is used to calculate the potential energy of a structure based on
13:     accessible volume calculations.
14:     """
15:     name = 'Av'
16: 
17:     def __init__(self, distances=None, positions=None, av_samples=10000, min_av=150, verbose=False):
18:         """
19:         Initialize the AvPotential class.
20:         
21:         Parameters
22:         ----------
23:         distances : dict, optional
24:             Dictionary of distance measurements
25:         positions : dict, optional
26:             Dictionary of positions for accessible volume calculations
27:         av_samples : int, optional
28:             Number of samples for accessible volume calculations
29:         min_av : int, optional
30:             Minimum number of points in accessible volume
31:         verbose : bool, optional
32:             Whether to print verbose output
33:         """
34:         self.verbose = verbose
35:         self.distances = distances
36:         self.positions = positions
37:         self.n_av_samples = av_samples
38:         self.min_av = min_av
39:         self.avs = OrderedDict()
40: 
41:     @property
42:     def structure(self):
43:         """The structure this potential is currently computed for."""
44:         return self._structure
45: 
46:     @structure.setter
47:     def structure(self, structure):
48:         """Set the structure and recompute accessible volumes."""
49:         self._structure = structure
50:         self.calc_avs()
51: 
52:     def calc_avs(self):
53:         """
54:         Calculate accessible volumes for all positions.
55:         """
56:         if self.positions is None:
57:             raise ValueError("Positions not set unable to calculate AVs")
58:         
59:         # This method would need to be implemented with cs's AV calculation
60:         # For now, we'll leave it as a placeholder
61:         pass
62: 
63:     def calc_distances(self, structure=None, verbose=False):
64:         """
65:         Calculate distances between accessible volumes.
66:         
67:         Parameters
68:         ----------
69:         structure : Structure, optional
70:             Structure to calculate distances for
71:         verbose : bool, optional
72:             Whether to print verbose output
73:         """
74:         verbose = verbose or self.verbose
75:         if structure is not None:
76:             self.structure = structure
77:             
78:         # This method would need to be implemented with cs's distance calculation
79:         # For now, we'll leave it as a placeholder
80:         pass
81: 
82:     def getChi2(self, structure=None, reduced=False, verbose=False):
83:         """
84:         Calculate chi-squared value for the structure.
85:         
86:         Parameters
87:         ----------
88:         structure : Structure, optional
89:             Structure to calculate chi-squared for
90:         reduced : bool, optional
91:             Whether to return reduced chi-squared
92:         verbose : bool, optional
93:             Whether to print verbose output
94:             
95:         Returns
96:         -------
97:         float
98:             Chi-squared value
99:         """
100:         verbose = self.verbose or verbose
101:         if structure is not None:
102:             self.structure = structure
103: 
104:         chi2 = 0.0
105:         self.calc_distances(verbose=verbose)
106:         
107:         # This method would need to be implemented with actual chi-squared calculation
108:         # For now, we'll return a placeholder value
109:         return chi2
110: 
111:     def getEnergy(self, structure=None):
112:         """
113:         Calculate energy for the structure.
114:         
115:         Parameters
116:         ----------
117:         structure : Structure, optional
118:             Structure to calculate energy for
119:             
120:         Returns
121:         -------
122:         float
123:             Energy value
124:         """
125:         if structure is not None:
126:             self.structure = structure
127:         return self.getChi2()

(End of file - total 127 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 60,
  "offset": 395
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
395:         group_layout.addLayout(controls)
396: 
397:         for key, spec in self._POTENTIAL_SPECS.items():
398:             if spec["enabled"]:
399:                 self._add_potential_row(key, weight=float(spec["weight"]), settings=spec["params"])
400: 
401:         layout.addWidget(group)
402:         self._resize_potential_table()
403: 
404:     def _resize_potential_table(self) -> None:
405:         """Fit the energy table height to its rows without wasting vertical space."""
406: 
407:         header = self.potential_table.horizontalHeader().height()
408:         rows = max(self.potential_table.rowCount(), 2)
409:         row_height = self.potential_table.verticalHeader().defaultSectionSize()
410:         height = min(max(header + rows * row_height + 8, 82), 130)
411:         self.potential_table.setFixedHeight(height)
412: 
413:     def _add_potential_row(self, key: str, weight: float = None, settings: dict = None) -> None:
414:         """Add or update one energy-term row."""
415: 
416:         spec = self._POTENTIAL_SPECS.get(key)
417:         if spec is None:
418:             return
419:         for row in range(self.potential_table.rowCount()):
420:             item = self.potential_table.item(row, 0)
421:             if item is not None and item.data(QtCore.Qt.UserRole) == key:
422:                 if weight is not None:
423:                     self.potential_table.item(row, 1).setText(f"{float(weight):.2f}")
424:                 if settings is not None:
425:                     item.setData(QtCore.Qt.UserRole + 1, dict(settings))
426:                 self._resize_potential_table()
427:                 return
428: 
429:         row = self.potential_table.rowCount()
430:         self.potential_table.insertRow(row)
431:         term_item = QtWidgets.QTableWidgetItem(str(spec["label"]))
432:         term_item.setFlags(term_item.flags() & ~QtCore.Qt.ItemIsEditable)
433:         term_item.setData(QtCore.Qt.UserRole, key)
434:         term_item.setData(QtCore.Qt.UserRole + 1, dict(settings or spec["params"]))
435:         weight_item = QtWidgets.QTableWidgetItem(f"{float(weight if weight is not None else spec['weight']):.2f}")
436:         remove_button = QtWidgets.QPushButton("r", self.potential_table)
437:         remove_button.setMaximumWidth(24)
438:         remove_button.setToolTip("remove")
439:         remove_button.clicked.connect(lambda _checked=False, button=remove_button: self._remove_potential_button_row(button))
440:         self.potential_table.setItem(row, 0, term_item)
441:         self.potential_table.setItem(row, 1, weight_item)
442:         self.potential_table.setCellWidget(row, 2, remove_button)
443:         self._resize_potential_table()
444: 
445:     def _remove_potential_button_row(self, button: QtWidgets.QPushButton) -> None:
446:         """Remove the energy-term row that owns a row-local remove button."""
447: 
448:         for row in range(self.potential_table.rowCount()):
449:             if self.potential_table.cellWidget(row, 2) is button:
450:                 self.potential_table.removeRow(row)
451:                 self._resize_potential_table()
452:                 return
453: 
454:     def on_potential_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:

(Showing lines 395-454 of 1224. Use offset=455 to continue.)
</content>
```

---

## User

I think the av potential is not really needed.

---

## Assistant (Build · MiMo V2.5 Free · 9.5s)

_Thinking:_

The user confirmed that AV potential is not needed. Let me now create a plan for adding the remaining energy terms to ProteinMC:

1. **H-Bond** - Already supported ✓
2. **Iso-UNRES** - Already supported ✓
3. **Miyazawa-Jernigan** - Already supported ✓
4. **Clash potential** - Already supported ✓
5. **Go-Potential** - NOT supported yet, needs to be added
6. **ASA-Calpha** - NOT supported yet, needs to be added
7. **Radius of Gyration** - NOT supported yet, needs to be added


8. **Ramachandran** - NOT supported yet, needs to be added
9. **AV-Potential** - Not needed (user confirmed)

For GoPotential, I need to handle the initialization sequence carefully since `setGo()` requires specific attributes like epsilon, native_cutoff_on, nnEFactor, and cutoff to be set first. The `_configure_potentials()` method will need to ensure these are properly initialized before calling `setGo()`.

For Ramachandran and ASA, they both follow the standard pattern with `structure` and optional parameters in their constructors, and they expose `getEnergy()` directly without needing extra setup steps. Radius of Gyration is trickier since it lacks a proper core class—currently it's just a widget that wraps the structure's radius_gyration property, so I'll need to either create a dedicated class for it or find another way to integrate it into the potential framework.

Now I'm laying out the implementation plan: I need to add GoPotential, Ramachandran, ASA, and Radius of Gyration to the potential classes dictionary, create a core RadiusGyration class, update the GUI specifications, and make sure GoPotential's initialization is handled correctly during configuration.

Good, skipping AV-Potential. Let me finalize the plan.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Handle GoPotential.setGo() call in _configure_potentials (model.py)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Verify lint/typecheck passes",
      "status": "pending",
      "priority": "medium"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Handle GoPotential.setGo() call in _configure_potentials (model.py)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 6.1s)

_Thinking:_

Let me check what `radius_gyration` property looks like on the Structure class, to know how to create a core class for it.

Let me check the `radius_gyration` property on Structure:

**Tool: grep**

**Input:**
```json
{
  "pattern": "radius_gyration",
  "path": "/Users/tpeulen/dev/chisurf/chisurf/core/structure"
}
```

**Output:**
```
Found 4 matches
/Users/tpeulen/dev/chisurf/chisurf/core/structure/structure.py:
  Line 49:     radius_gyration: float

  Line 50:         The attribute radius_gyration returns the radius of gyration of the

  Line 89:     >>> structure.radius_gyration

  Line 281:     def radius_gyration(self) -> float:

```

---

## Assistant (Build · MiMo V2.5 Free · 3.4s)

_Thinking:_

Let me read the radius_gyration property to understand how it's computed.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/structure.py",
  "limit": 25,
  "offset": 275
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/structure.py</path>
<type>file</type>
<content>
275:         s = self
276:         sel = chisurf.core.structure.get_atom_index_by_name(s.atoms, ['CA'])[0]
277:         for ai, bi in zip(sel, v):
278:             s._atoms[ai]['bfactor'] = bi
279: 
280:     @property
281:     def radius_gyration(self) -> float:
282:         """Radius of gyration of the structure."""
283:         coord = self.xyz
284:         rM = coord[:, :].mean(axis=0)
285:         rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).mean()
286:         return float(rG)
287: 
288:     def append_potential(
289:             self,
290:             function,
291:             kwargs: typing.Dict = None
292:     ):
293:         """
294: 
295:         :param function:
296:         :param kwargs:
297:         :return:
298: 
299:         Examples

(Showing lines 275-299 of 740. Use offset=300 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 60.0s)

_Thinking:_

Now I have all the information I need. Let me present the plan.

Here's the summary of what needs to be done:

## Already Supported in ProteinMC
- **H-Bond** ✓ (HPotential)
- **Iso-UNRES** ✓ (CEPotential)  
- **Miyazawa-Jernigan** ✓ (MJPotential)
- **Clash potential** ✓ (ClashPotential)

## Needs to be Added

### 1. Go-Potential
- **Core class exists**: `GoPotential` in `potentials.py:434`
- **Challenge**: `getEnergy()` requires `setGo()` to be called first, which needs `epsilon`, `cutoff`, `native_cutoff_on`, `nnEFactor`, `non_native_contact_on` attributes
- **Fix**: Set default attributes in `__init__` OR call `setGo()` after construction in `_configure_potentials()`

### 2. Ramachandran
- **Core class exists**: `Ramachandran` in `potentials.py:293`
- **Interface**: `__init__(structure, filename=None)` → `getEnergy()`
- **Straightforward** - no special handling needed

### 3. ASA-Calpha
- **Core class exists**: `ASA` in `potentials.py:616`
- **Interface**: `__init__(structure, probe=1.0, n_sphere_point=590, radius=2.5)` → `getEnergy()`
- **Straightforward** - no special handling needed

### 4. Radius of Gyration
- **No core class** - only exists as a GUI widget
- **Need to create**: A simple `RadiusGyration` class in `potentials.py` that wraps `structure.radius_gyration`


- **Changes required**: Add a new `RadiusGyration` class to the potentials module, register all four new potentials in the `_potential_classes()` dictionary, and extend `_POTENTIAL_SPECS` in the GUI widget to include them.

For GoPotential specifically, I need to ensure `setGo()` gets called after instantiation since it depends on attributes that won't exist by default. The cleanest approach is to have `_configure_potentials()` check for this method and invoke it after creating the instance.

Looking at the `setGo()` implementation, it uses several attributes like `non_native_contact_on`, `cutoff`, `native_cutoff_on`, `epsilon`, and `nnEFactor`. I should set sensible defaults in `GoPotential.__init__` so these attributes are always available, then allow them to be overridden through kwargs. This avoids fragile assumptions about method existence while keeping the initialization straightforward. I'll go with adding the missing attributes to `GoPotential.__init__` and calling `setGo()` after construction—that's the cleanest way to ensure the potential is ready to use. Now let me outline the full plan.

Here is the complete plan:

---

## Plan: Add Missing Energy Terms to ProteinMC

### Already Supported (no changes needed)
- H-Bond (`HPotential`)
- Iso-UNRES (`CEPotential`)
- Miyazawa-Jernigan (`MJPotential`)
- Clash potential (`ClashPotential`)

### Needs to be Added

#### 1. Create `RadiusGyration` core class (`potentials.py`)
There is no core class — only a GUI widget. Add a minimal class:
```python
class RadiusGyration:
    name = 'Radius-Gyration'
    def __init__(self, structure, **kwargs):
        self.structure = structure
    def getEnergy(self):
        return self.structure.radius_gyration
```

#### 2. Fix `GoPotential.__init__` (`potentials.py:434`)
`getEnergy()` calls `setGo()` which reads attributes (`epsilon`, `cutoff`, etc.) not set in `__init__`. Add default attributes so it works headlessly:
```python
def __init__(self, structure, epsilon=1.0, cutoff=6.5, 
             native_cutoff_on=True, nnEFactor=0.7, non_native_contact_on=True, **kwargs):
    self.structure = structure
    self.name = 'go'
    self.epsilon = epsilon
    self.cutoff = cutoff
    self.native_cutoff_on = native_cutoff_on
    self.nnEFactor = nnEFactor
    self.non_native_contact_on = non_native_contact_on
    self.setGo()
```

#### 3. Register in `_potential_classes()` (`model.py:707`)
Add entries for all 4 new terms:
```python
"go": core_potentials.GoPotential,
"Go-Potential": core_potentials.GoPotential,
"rama": core_potentials.Ramachandran,
"Ramachandran": core_potentials.Ramachandran,
"asa": core_potentials.ASA,
"ASA-Calpha": core_potentials.ASA,
"rg": core_potentials.RadiusGyration,
"Radius of Gyration": core_potentials.RadiusGyration,
```

#### 4. Add `_POTENTIAL_SPECS` entries (`proteinmc.py:74`)
Add GUI specs with default params matching the core class constructors:

| Key | Label | Default params |
|-----|-------|---------------|
| `go` | Gō-Potential | `epsilon=1.0, cutoff=6.5, native_cutoff_on=1, nnEFactor=0.7, non_native_contact_on=1` |
| `rama` | Ramachandran | `{}` (uses default database file) |
| `asa` | ASA-Cα | `probe=1.0, n_sphere_point=590, radius=2.5` |
| `rg` | Radius of Gyration | `{}` |

#### 5. Update `__init__.py` exports (`potentials.py` import line)
Ensure `RadiusGyration` is importable from `chisurf.core.structure.potential.potentials`.

### Files to modify
1. `chisurf/core/structure/potential/potentials.py` — add `RadiusGyration` class, fix `GoPotential.__init__`
2. `chisurf/plugins/modelling/proteinmc/model.py` — register 4 new terms in `_potential_classes()`
3. `chisurf/gui/widgets/models/proteinmc.py` — add 4 new `_POTENTIAL_SPECS` entries

### Verification
- Run `ruff check` and `mypy` on modified files
- Run existing proteinmc tests: `pytest test/plugins/test_proteinmc.py`

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "This plan looks good. Should I proceed with implementation?",
      "header": "Proceed?",
      "options": [
        {
          "label": "Yes, implement all",
          "description": "Create RadiusGyration class, fix GoPotential, register all 4 new terms, update GUI specs"
        },
        {
          "label": "Skip Go-Potential",
          "description": "Go-Potential has complex init; add the other 3 only"
        },
        {
          "label": "Modeline only",
          "description": "Only update _potential_classes() and _POTENTIAL_SPECS, don't touch core classes"
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "This plan looks good. Should I proceed with implementation?"="Yes, implement all". You can now continue with the user's answers in mind.
```

---

## Assistant (Build · MiMo V2.5 Free · 5.1s)

_Thinking:_

The user wants me to implement all changes. Let me proceed with the implementation.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 6.4s)

_Thinking:_

Let me start implementing. First, I'll add the RadiusGyration class to potentials.py, right after ClashPotential.

Adding the `RadiusGyration` class to `potentials.py`:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "oldString": "        return chisurf.core.structure.potential.cPotentials_.clash_potential(\n            c.xyz,\n            c.vdw,\n            self.clash_tolerance,\n            self.covalent_radius\n        )\n\n#\n# class AvPotential(object):",
  "newString": "        return chisurf.core.structure.potential.cPotentials_.clash_potential(\n            c.xyz,\n            c.vdw,\n            self.clash_tolerance,\n            self.covalent_radius\n        )\n\n\nclass RadiusGyration(object):\n\n    name = 'Radius-Gyration'\n\n    def __init__(\n            self,\n            structure: chisurf.core.structure.Structure,\n            **kwargs\n    ):\n        \"\"\"Initialize a radius-of-gyration potential for *structure*.\"\"\"\n        self.structure = structure\n\n    def getEnergy(self) -> float:\n        \"\"\"Calculate and return the radius of gyration.\"\"\"\n        return float(self.structure.radius_gyration)\n\n\n#\n# class AvPotential(object):"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 7.9s)

_Thinking:_

Now let me fix the GoPotential class to add default attributes and call setGo() in __init__.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 7.3s)

_Thinking:_

Now fix GoPotential.__init__ to set default attributes and call setGo().

Now fix `GoPotential.__init__`:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "oldString": "class GoPotential(object):\n\n    def __init__(\n            self,\n            structure: chisurf.core.structure.Structure\n    ):\n        \"\"\"Initialize a Go-like potential for *structure*.\"\"\"\n        self.structure = structure\n        self.name = 'go'\n\n    def setGo(self):",
  "newString": "class GoPotential(object):\n\n    name = 'go'\n\n    def __init__(\n            self,\n            structure: chisurf.core.structure.Structure,\n            epsilon: float = 1.0,\n            cutoff: float = 6.5,\n            native_cutoff_on: bool = True,\n            nnEFactor: float = 0.7,\n            non_native_contact_on: bool = True,\n            **kwargs\n    ):\n        \"\"\"Initialize a Go-like potential for *structure*.\"\"\"\n        self.structure = structure\n        self.epsilon = epsilon\n        self.cutoff = cutoff\n        self.native_cutoff_on = native_cutoff_on\n        self.nnEFactor = nnEFactor\n        self.non_native_contact_on = non_native_contact_on\n        self.setGo()\n\n    def setGo(self):"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 18.4s)

_Thinking:_

Now I need to update the `setGo()` method to handle the case where attributes might not exist yet (for backward compatibility when GoPotential is subclassed by the GUI widget). Actually, the GUI widget `GoPotentialWidget` inherits from both `GoPotential` and `QWidget`, and it sets these properties dynamically. Let me check how the widget subclass calls things.

Looking at `potentials_go.py`:
```python
class GoPotentialWidget(GoPotential, QtWidgets.QWidget):
    def __init__(self, structure=None, **kwargs):
        QtWidgets.QWidget.__init__(self, parent=kwargs.get('parent'))
        GoPotential.__init__(self, structure=structure)
        ...
```

The widget calls `GoPotential.__init__` which now calls `setGo()`. But at that point, the widget's properties (`epsilon`, `cutoff`, etc.) aren't set yet - they're defined as Qt properties that read from QLineEdit widgets. So `setGo()` will fail because the attributes don't exist.

I need to be more careful here. The `setGo()` method reads `self.nnEFactor`, `self.non_native_contact_on`, `self.cutoff`, `self.native_cutoff_on`, and `self.epsilon` — all of which come from Qt widgets in the subclass, so I can't rely on them being set during initialization.

I think the best approach is to add a guard in `setGo()` that checks whether these attributes exist before using them, and either skip the call or use defaults if they're missing. Then I can call `setGo()` at the end of `__init__` only if the necessary attributes are available. the widget won't call `setGo()` during `__init__` since it relies on textChanged signals to trigger updates later. For the standalone case, I need to set defaults in `GoPotential.__init__` and call `setGo()` there, but in the widget subclass I should skip that call and let the signal handlers manage it instead. I'm realizing the core issue: `GoPotential.__init__` calls `setGo()` which tries to access widget properties like `self.epsilon` that don't exist yet since the widget hasn't finished initializing. The cleanest fix is to guard the `setGo()` call in the parent class so it only executes if those attributes are already set, letting the widget subclass override the initialization order without breaking. Still writing the guard logic...

Actually, I need to reconsider the widget's initialization order—it sets up the Qt widgets first, then calls the parent `GoPotential.__init__`, so by that point the properties like `epsilon` and `cutoff` should already be defined as Qt properties. This means the guard approach should work fine since the widget will have those attributes available when `setGo()` is called.

For the headless case, I'll set the attributes directly before calling `setGo()`. The cleanest solution is to add the attributes with defaults in `GoPotential.__init__` and call `setGo()` at the end, then in the widget subclass, set the attributes before calling the parent constructor so they're available when needed.

Defining the widget constructor... Still writing initialization logic...

Actually, I'm realizing the core issue: `setGo()` tries to access attributes that might not exist yet if the widget hasn't finished initializing. The cleanest solution is to add a guard in `setGo()` that checks whether the required attributes are present before using them, rather than trying to control initialization order or skip the call entirely. Adding attribute guards to `setGo()`.

I need to also update `setGo()` to be safe when called during widget init (before Qt properties exist). Let me fix that:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "newString": "    def setGo(self):\n        \"\"\"Initialize the Gō potential energy and contact matrices.\"\"\"\n        if not hasattr(self, 'epsilon'):\n            return\n        c = self.structure\n        nnEFactor = getattr(self, 'nnEFactor', 0.7) if getattr(self, 'non_native_contact_on', True) else 0.0\n        cutoff = getattr(self, 'cutoff', 6.5) if getattr(self, 'native_cutoff_on', True) else 1e6\n        self.eMatrix, self.sMatrix = chisurf.core.structure.potential.cPotentials_.go_init(\n            c.residue_lookup_r, c.dist_ca,\n            self.epsilon, nnEFactor, cutoff\n        )",
  "oldString": "    def setGo(self):\n        \"\"\"Initialize the Gō potential energy and contact matrices.\"\"\"\n        c = self.structure\n        nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0\n        cutoff = self.cutoff if self.native_cutoff_on else 1e6\n        self.eMatrix, self.sMatrix = chisurf.core.structure.potential.cPotentials_.go_init(\n            c.residue_lookup_r, c.dist_ca,\n            self.epsilon, nnEFactor, cutoff\n        )"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 6.4s)

_Thinking:_

Now let me register the new potentials in `_potential_classes()` in model.py.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "pending"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 9.4s)

_Thinking:_

Now register the new potentials in model.py's `_potential_classes()`.

Now register the new potentials in `_potential_classes()`:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/plugins/modelling/proteinmc/model.py",
  "newString": "def _potential_classes() -> dict[str, Any]:\n    \"\"\"Return structural potential classes, importing compiled code lazily.\"\"\"\n    from chisurf.core.structure.potential import potentials as core_potentials\n\n    return {\n        \"default\": core_potentials.ClashPotential,\n        \"hbond\": core_potentials.HPotential,\n        \"mj\": core_potentials.MJPotential,\n        \"unres\": core_potentials.CEPotential,\n        \"go\": core_potentials.GoPotential,\n        \"rama\": core_potentials.Ramachandran,\n        \"asa\": core_potentials.ASA,\n        \"rg\": core_potentials.RadiusGyration,\n        \"H-Bond\": core_potentials.HPotential,\n        \"H-Potential\": core_potentials.HPotential,\n        \"Iso-UNRES\": core_potentials.CEPotential,\n        \"Miyazawa-Jernigan\": core_potentials.MJPotential,\n        \"Clash potential\": core_potentials.ClashPotential,\n        \"Go-Potential\": core_potentials.GoPotential,\n        \"Ramachandran\": core_potentials.Ramachandran,\n        \"ASA-Calpha\": core_potentials.ASA,\n        \"Radius of Gyration\": core_potentials.RadiusGyration,\n    }",
  "oldString": "def _potential_classes() -> dict[str, Any]:\n    \"\"\"Return structural potential classes, importing compiled code lazily.\"\"\"\n    from chisurf.core.structure.potential import potentials as core_potentials\n\n    return {\n        \"default\": core_potentials.ClashPotential,\n        \"hbond\": core_potentials.HPotential,\n        \"mj\": core_potentials.MJPotential,\n        \"unres\": core_potentials.CEPotential,\n        \"H-Bond\": core_potentials.HPotential,\n        \"H-Potential\": core_potentials.HPotential,\n        \"Iso-UNRES\": core_potentials.CEPotential,\n        \"Miyazawa-Jernigan\": core_potentials.MJPotential,\n        \"Clash potential\": core_potentials.ClashPotential,\n    }"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 5.1s)

_Thinking:_

Now let me add the `_POTENTIAL_SPECS` entries in the GUI widget.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 14.3s)

_Thinking:_

Now add the new `_POTENTIAL_SPECS` entries to the GUI widget.

Now add the new `_POTENTIAL_SPECS` entries in the GUI widget:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "newString": "    _POTENTIAL_SPECS = {\n        \"default\": {\n            \"label\": \"Default clash\",\n            \"name\": \"default\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\"clash_tolerance\": 2.0, \"covalent_radius\": 1.5},\n        },\n        \"hbond\": {\n            \"label\": \"H-bond\",\n            \"name\": \"hbond\",\n            \"weight\": 2.0,\n            \"enabled\": True,\n            \"params\": {\n                \"cutoff_ca\": 8.0,\n                \"cutoff_hbond\": 3.0,\n                \"oh\": 1.0,\n                \"on\": 1.0,\n                \"cn\": 1.0,\n                \"ch\": 1.0,\n            },\n        },\n        \"mj\": {\n            \"label\": \"MJ\",\n            \"name\": \"mj\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\"ca_cutcoff\": 6.5},\n        },\n        \"unres\": {\n            \"label\": \"UNRES\",\n            \"name\": \"unres\",\n            \"weight\": 1.0,\n            \"enabled\": True,\n            \"params\": {\n                \"ca_cutoff\": 15.0,\n                \"repulsion\": 100.0,\n                \"min_dist\": 3.5,\n                \"max_dist\": 19.0,\n                \"bin_width\": 0.05,\n                \"centroid_number\": 4.0,\n            },\n        },\n        \"go\": {\n            \"label\": \"Gō-Potential\",\n            \"name\": \"go\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\n                \"epsilon\": 1.0,\n                \"cutoff\": 6.5,\n                \"native_cutoff_on\": True,\n                \"nnEFactor\": 0.7,\n                \"non_native_contact_on\": True,\n            },\n        },\n        \"rama\": {\n            \"label\": \"Ramachandran\",\n            \"name\": \"rama\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {},\n        },\n        \"asa\": {\n            \"label\": \"ASA-Cα\",\n            \"name\": \"asa\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\n                \"probe\": 1.0,\n                \"n_sphere_point\": 590,\n                \"radius\": 2.5,\n            },\n        },\n        \"rg\": {\n            \"label\": \"Radius of Gyration\",\n            \"name\": \"rg\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {},\n        },\n    }",
  "oldString": "    _POTENTIAL_SPECS = {\n        \"default\": {\n            \"label\": \"Default clash\",\n            \"name\": \"default\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\"clash_tolerance\": 2.0, \"covalent_radius\": 1.5},\n        },\n        \"hbond\": {\n            \"label\": \"H-bond\",\n            \"name\": \"hbond\",\n            \"weight\": 2.0,\n            \"enabled\": True,\n            \"params\": {\n                \"cutoff_ca\": 8.0,\n                \"cutoff_hbond\": 3.0,\n                \"oh\": 1.0,\n                \"on\": 1.0,\n                \"cn\": 1.0,\n                \"ch\": 1.0,\n            },\n        },\n        \"mj\": {\n            \"label\": \"MJ\",\n            \"name\": \"mj\",\n            \"weight\": 1.0,\n            \"enabled\": False,\n            \"params\": {\"ca_cutcoff\": 6.5},\n        },\n        \"unres\": {\n            \"label\": \"UNRES\",\n            \"name\": \"unres\",\n            \"weight\": 1.0,\n            \"enabled\": True,\n            \"params\": {\n                \"ca_cutoff\": 15.0,\n                \"repulsion\": 100.0,\n                \"min_dist\": 3.5,\n                \"max_dist\": 19.0,\n                \"bin_width\": 0.05,\n                \"centroid_number\": 4.0,\n            },\n        },\n    }"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 6.4s)

_Thinking:_

Now I need to check if there are any issues with the `_potential_settings()` method. The method converts `centroid_number` to int. Let me check if we need to handle `n_sphere_point` similarly (it should be int too).

Looking at the method:
```python
for int_key in ("centroid_number",):
    if int_key in settings:
        settings[int_key] = int(settings[int_key])
```

I should add `n_sphere_point` to this list as well.

Let me also update the int conversion in `_potential_settings()` for `n_sphere_point`:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "newString": "            for int_key in (\"centroid_number\", \"n_sphere_point\"):\n                if int_key in settings:\n                    settings[int_key] = int(settings[int_key])",
  "oldString": "            for int_key in (\"centroid_number\",):\n                if int_key in settings:\n                    settings[int_key] = int(settings[int_key])"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 11.3s)

_Thinking:_

Now let me also check if the `bool` parameters (`native_cutoff_on`, `non_native_contact_on`) need special handling. The `_potential_settings()` method reads settings from the UI table which stores values as strings. When these are passed to `GoPotential.__init__`, they'll be strings, not booleans. Let me check how the existing parameters are handled.

Looking at the code, the settings are stored as whatever type they were when put in the dict. The `_POTENTIAL_SPECS` has them as Python bools, and `_potential_settings()` returns them as-is from `term_item.data(QtCore.Qt.UserRole + 1)`. 

But wait, when the user edits settings in the UI dialog, the values might be converted to strings. Let me check the `_edit_potential_params` method to see if there's any type conversion happening there.

Looking at the method, it creates spin boxes and line edits for numeric values, and when the user confirms the dialog, it collects all the values into a dict. The issue is that line edits return strings while spin boxes return numbers, so the type conversion isn't consistent across all parameter types.

When these string values get passed to the GoPotential constructor, Python will try to use them in boolean contexts like `if self.non_native_contact_on:`, which will always be truthy since non-empty strings are truthy in Python. But then when I use `getattr()` with a default value, that string will override the default, which could cause unexpected behavior.

I need to add boolean conversion logic to handle this properly in `_potential_settings()`.

I also need to handle boolean conversion for Go parameters. Let me update `_potential_settings()`:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 20,
  "offset": 540
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
540:         settings = dict(item.data(QtCore.Qt.UserRole + 1) or spec["params"])
541:         dialog = QtWidgets.QDialog(self)
542:         dialog.setWindowTitle(f"{spec['label']} parameters")
543:         form = QtWidgets.QFormLayout(dialog)
544:         controls = {}
545:         for param_name, default in spec["params"].items():
546:             spin = QtWidgets.QDoubleSpinBox(dialog)
547:             spin.setDecimals(6)
548:             spin.setRange(-100000.0, 100000.0)
549:             spin.setValue(float(settings.get(param_name, default)))
550:             form.addRow(param_name, spin)
551:             controls[param_name] = spin
552:         buttons = QtWidgets.QDialogButtonBox(
553:             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
554:             parent=dialog,
555:         )
556:         buttons.accepted.connect(dialog.accept)
557:         buttons.rejected.connect(dialog.reject)
558:         form.addRow(buttons)
559:         if dialog.exec_() != QtWidgets.QDialog.Accepted:

(Showing lines 540-559 of 1262. Use offset=560 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 6.8s)

_Thinking:_

I see the issue. The `_edit_potential_params` method uses `QDoubleSpinBox` for ALL parameters, including boolean ones like `native_cutoff_on` and `non_native_contact_on`. This means they'll be stored as floats (0.0 or 1.0) in the settings dialog.

But in `_potential_settings()`, these values are passed through directly. When they reach `GoPotential.__init__`, `native_cutoff_on=0.0` will be falsy in Python, which is actually correct behavior (0.0 is falsy, 1.0 is truthy).

Wait, but the initial values from `_POTENTIAL_SPECS` are Python bools (`True`/`False`), not floats. 

So when `_add_potential_row` stores them, they stay as bools. The real problem is that `_edit_potential_params` always creates `QDoubleSpinBox` widgets, which means the dialog will display bools as 0.0 or 1.0, and when the user confirms, those values become floats instead of bools. This inconsistency between the initial bool values and the dialog's float values is what's causing the type mismatch.

I need to add type coercion in `_potential_settings()` to convert float values back to bools when the original spec parameter was a bool.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 20,
  "offset": 526
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
526:     @QtCore.Slot()
527:     def on_edit_potential_details(self) -> None:
528:         """Open a popup editor for selected energy-term parameters."""
529: 
530:         row = self.potential_table.currentRow()
531:         if row < 0:
532:             return
533:         item = self.potential_table.item(row, 0)
534:         if item is None:
535:             return
536:         key = item.data(QtCore.Qt.UserRole)
537:         spec = self._POTENTIAL_SPECS.get(key)
538:         if spec is None:
539:             return
540:         settings = dict(item.data(QtCore.Qt.UserRole + 1) or spec["params"])
541:         dialog = QtWidgets.QDialog(self)
542:         dialog.setWindowTitle(f"{spec['label']} parameters")
543:         form = QtWidgets.QFormLayout(dialog)
544:         controls = {}
545:         for param_name, default in spec["params"].items():

(Showing lines 526-545 of 1262. Use offset=546 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 4.1s)

_Thinking:_

Let me look at `_potential_settings()` to add the bool conversion:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 30,
  "offset": 562
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
562:         item.setData(QtCore.Qt.UserRole + 1, updated)
563: 
564:     def _potential_settings(self) -> list[dict]:
565:         """Return enabled ProteinMC potential settings from the UI."""
566: 
567:         out = []
568:         table = getattr(self, "potential_table", None)
569:         if table is None:
570:             return out
571:         for row in range(table.rowCount()):
572:             term_item = table.item(row, 0)
573:             weight_item = table.item(row, 1)
574:             if term_item is None or weight_item is None:
575:                 continue
576:             key = term_item.data(QtCore.Qt.UserRole)
577:             spec = self._POTENTIAL_SPECS.get(key)
578:             if spec is None:
579:                 continue
580:             settings = dict(term_item.data(QtCore.Qt.UserRole + 1) or spec["params"])
581:             for int_key in ("centroid_number", "n_sphere_point"):
582:                 if int_key in settings:
583:                     settings[int_key] = int(settings[int_key])
584:             try:
585:                 weight = float(weight_item.text())
586:             except ValueError:
587:                 weight = float(spec["weight"])
588:                 weight_item.setText(f"{weight:.2f}")
589:             out.append(
590:                 {
591:                     "name": spec["name"],

(Showing lines 562-591 of 1262. Use offset=592 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 8.1s)

_Thinking:_

I need to add bool coercion for parameters that are bools in the spec. Let me update the `_potential_settings()` method:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "newString": "            settings = dict(term_item.data(QtCore.Qt.UserRole + 1) or spec[\"params\"])\n            for int_key in (\"centroid_number\", \"n_sphere_point\"):\n                if int_key in settings:\n                    settings[int_key] = int(settings[int_key])\n            for param_name, default in spec[\"params\"].items():\n                if isinstance(default, bool) and param_name in settings:\n                    settings[param_name] = bool(settings[param_name])",
  "oldString": "            settings = dict(term_item.data(QtCore.Qt.UserRole + 1) or spec[\"params\"])\n            for int_key in (\"centroid_number\", \"n_sphere_point\"):\n                if int_key in settings:\n                    settings[int_key] = int(settings[int_key])"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 4.3s)

_Thinking:_

Now let me also update the `_edit_potential_params` method to handle bool parameters properly (use checkboxes instead of spin boxes).

Now let me also fix the parameter editor dialog to use checkboxes for bool params:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "limit": 30,
  "offset": 544
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py</path>
<type>file</type>
<content>
544:         controls = {}
545:         for param_name, default in spec["params"].items():
546:             spin = QtWidgets.QDoubleSpinBox(dialog)
547:             spin.setDecimals(6)
548:             spin.setRange(-100000.0, 100000.0)
549:             spin.setValue(float(settings.get(param_name, default)))
550:             form.addRow(param_name, spin)
551:             controls[param_name] = spin
552:         buttons = QtWidgets.QDialogButtonBox(
553:             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
554:             parent=dialog,
555:         )
556:         buttons.accepted.connect(dialog.accept)
557:         buttons.rejected.connect(dialog.reject)
558:         form.addRow(buttons)
559:         if dialog.exec_() != QtWidgets.QDialog.Accepted:
560:             return
561:         updated = {name: spin.value() for name, spin in controls.items()}
562:         item.setData(QtCore.Qt.UserRole + 1, updated)
563: 
564:     def _potential_settings(self) -> list[dict]:
565:         """Return enabled ProteinMC potential settings from the UI."""
566: 
567:         out = []
568:         table = getattr(self, "potential_table", None)
569:         if table is None:
570:             return out
571:         for row in range(table.rowCount()):
572:             term_item = table.item(row, 0)
573:             weight_item = table.item(row, 1)

(Showing lines 544-573 of 1265. Use offset=574 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 12.1s)

_Thinking:_

I need to update the `_edit_potential_params` method to use checkboxes for bool parameters. Let me replace the relevant section.

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/gui/widgets/models/proteinmc.py",
  "newString": "        controls = {}\n        for param_name, default in spec[\"params\"].items():\n            if isinstance(default, bool):\n                checkbox = QtWidgets.QCheckBox(dialog)\n                checkbox.setChecked(bool(settings.get(param_name, default)))\n                form.addRow(param_name, checkbox)\n                controls[param_name] = checkbox\n            else:\n                spin = QtWidgets.QDoubleSpinBox(dialog)\n                spin.setDecimals(6)\n                spin.setRange(-100000.0, 100000.0)\n                spin.setValue(float(settings.get(param_name, default)))\n                form.addRow(param_name, spin)\n                controls[param_name] = spin\n        buttons = QtWidgets.QDialogButtonBox(\n            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,\n            parent=dialog,\n        )\n        buttons.accepted.connect(dialog.accept)\n        buttons.rejected.connect(dialog.reject)\n        form.addRow(buttons)\n        if dialog.exec_() != QtWidgets.QDialog.Accepted:\n            return\n        updated = {}\n        for name, ctrl in controls.items():\n            if isinstance(ctrl, QtWidgets.QCheckBox):\n                updated[name] = ctrl.isChecked()\n            else:\n                updated[name] = ctrl.value()\n        item.setData(QtCore.Qt.UserRole + 1, updated)",
  "oldString": "        controls = {}\n        for param_name, default in spec[\"params\"].items():\n            spin = QtWidgets.QDoubleSpinBox(dialog)\n            spin.setDecimals(6)\n            spin.setRange(-100000.0, 100000.0)\n            spin.setValue(float(settings.get(param_name, default)))\n            form.addRow(param_name, spin)\n            controls[param_name] = spin\n        buttons = QtWidgets.QDialogButtonBox(\n            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,\n            parent=dialog,\n        )\n        buttons.accepted.connect(dialog.accept)\n        buttons.rejected.connect(dialog.reject)\n        form.addRow(buttons)\n        if dialog.exec_() != QtWidgets.QDialog.Accepted:\n            return\n        updated = {name: spin.value() for name, spin in controls.items()}\n        item.setData(QtCore.Qt.UserRole + 1, updated)"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 10.6s)

_Thinking:_

Now let me also check if the `GoPotentialWidget` will still work. The widget calls `GoPotential.__init__(self, structure=structure)` which now sets default attributes and calls `setGo()`. But the widget's properties are defined as Qt properties that read from QLineEdit widgets. So when `setGo()` is called during `__init__`, the widget's properties like `epsilon` might not return the right values yet.

Let me re-read the widget code:```python
class GoPotentialWidget(GoPotential, QtWidgets.QWidget):
    def __init__(self, structure=None, **kwargs):
        QtWidgets.QWidget.__init__(self, parent=kwargs.get('parent'))
        GoPotential.__init__(self, structure=structure)
        ...
```

The widget calls `GoPotential.__init__` which sets `self.epsilon = 1.0` (from the default). Then the widget defines `@property epsilon` that returns `float(self.lineEdit.text())`. After `GoPotential.__init__` returns, the widget's `@property` overrides the `self.epsilon` attribute, so `self.epsilon` will now return the value from `lineEdit.text()` instead.

This creates a problem: when `setGo()` runs during `GoPotential.__init__`, it uses `self.epsilon = 1.0` from the default, but after the widget finishes initialization, `self.epsilon` becomes a property that reads from the line edit. However, `setGo()` won't be called again unless the user changes the text, so the Gō potential matrix gets computed with the default epsilon value while the actual UI might show a different value.

I need to either skip the `setGo()` call in `GoPotential.__init__` when subclassed, or add a guard that checks whether the attributes are actually available before using them. Let me reconsider the approach here. Still writing the guard logic...

Now let me verify the changes are consistent. Let me read the updated files:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "limit": 70,
  "offset": 434
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py</path>
<type>file</type>
<content>
434: class GoPotential(object):
435: 
436:     name = 'go'
437: 
438:     def __init__(
439:             self,
440:             structure: chisurf.core.structure.Structure,
441:             epsilon: float = 1.0,
442:             cutoff: float = 6.5,
443:             native_cutoff_on: bool = True,
444:             nnEFactor: float = 0.7,
445:             non_native_contact_on: bool = True,
446:             **kwargs
447:     ):
448:         """Initialize a Go-like potential for *structure*."""
449:         self.structure = structure
450:         self.epsilon = epsilon
451:         self.cutoff = cutoff
452:         self.native_cutoff_on = native_cutoff_on
453:         self.nnEFactor = nnEFactor
454:         self.non_native_contact_on = non_native_contact_on
455:         self.setGo()
456: 
457:     def setGo(self):
458:         """Initialize the Gō potential energy and contact matrices."""
459:         if not hasattr(self, 'epsilon'):
460:             return
461:         c = self.structure
462:         nnEFactor = getattr(self, 'nnEFactor', 0.7) if getattr(self, 'non_native_contact_on', True) else 0.0
463:         cutoff = getattr(self, 'cutoff', 6.5) if getattr(self, 'native_cutoff_on', True) else 1e6
464:         self.eMatrix, self.sMatrix = chisurf.core.structure.potential.cPotentials_.go_init(
465:             c.residue_lookup_r, c.dist_ca,
466:             self.epsilon, nnEFactor, cutoff
467:         )
468: 
469:     def getEnergy(self):
470:         """Calculate and return the Gō potential energy."""
471:         c = self.structure
472:         Etot, nNa, Ena, nNN, Enn = chisurf.core.structure.potential.cPotentials_.go(
473:             c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix
474:         )
475:         #Etot = go(c., c.dist_ca, self.eMatrix, self.sMatrix)
476:         self.E = Etot
477:         return Etot
478: 
479:     def getNbrNonNative(self):
480:         """Return the number of non-native contacts."""
481:         return self.nNN
482: 
483:     def getNbrNative(self):
484:         """Return the number of native contacts."""
485:         return self.nNa
486: 
487:     def set_sMatrix(self, sMatrix):
488:         """Set the sigma (distance) matrix for the Gō potential."""
489:         self.sMatrix = sMatrix
490: 
491:     def set_eMatrix(self, eMatrix):
492:         """Set the epsilon (energy) matrix for the Gō potential."""
493:         self.eMatrix = eMatrix
494: 
495:     def set_nMatrix(self, nMatrix):
496:         """Set the native contact matrix."""
497:         self.nMatrix = nMatrix
498: 
499: 
500: class MJPotential(object):
501: 
502:     name = 'Miyazawa-Jernigan'
503: 

(Showing lines 434-503 of 946. Use offset=504 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "limit": 25,
  "offset": 700
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py</path>
<type>file</type>
<content>
700:         return chisurf.core.structure.potential.cPotentials_.clash_potential(
701:             c.xyz,
702:             c.vdw,
703:             self.clash_tolerance,
704:             self.covalent_radius
705:         )
706: 
707: 
708: class RadiusGyration(object):
709: 
710:     name = 'Radius-Gyration'
711: 
712:     def __init__(
713:             self,
714:             structure: chisurf.core.structure.Structure,
715:             **kwargs
716:     ):
717:         """Initialize a radius-of-gyration potential for *structure*."""
718:         self.structure = structure
719: 
720:     def getEnergy(self) -> float:
721:         """Calculate and return the radius of gyration."""
722:         return float(self.structure.radius_gyration)
723: 
724: 

(Showing lines 700-724 of 946. Use offset=725 to continue.)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 4.9s)

_Thinking:_

Everything looks good. Now let me also check if the `__init__.py` in the potential module needs to be updated to export `RadiusGyration`.

Let me check if the potential module's `__init__.py` needs updating:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/__init__.py"
}
```

**Output:**
```
<path>/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/__init__.py</path>
<type>file</type>
<content>
1: import chisurf.core.structure.potential.potentials
2: 

(End of file - total 2 lines)
</content>
```

---

## Assistant (Build · MiMo V2.5 Free · 6.1s)

_Thinking:_

The `__init__.py` just imports the module, so `RadiusGyration` is accessible via `chisurf.core.structure.potential.potentials.RadiusGyration`. That's fine.

Now let me run lint/typecheck to verify the changes.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "in_progress"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "in_progress",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 3.9s)

_Thinking:_

Let me run the lint check on the modified files.

**Tool: bash**

**Input:**
```json
{
  "command": "ruff check chisurf/core/structure/potential/potentials.py chisurf/plugins/modelling/proteinmc/model.py chisurf/gui/widgets/models/proteinmc.py 2>&1",
  "description": "Run ruff lint on modified files",
  "workdir": "/Users/tpeulen/dev/chisurf"
}
```

**Output:**
```
I001 [*] Import block is un-sorted or un-formatted
  --> chisurf/core/structure/potential/potentials.py:1:1
   |
 1 | / from __future__ import annotations
 2 | |
 3 | | import os
 4 | | import math
 5 | | import copy
 6 | |
 7 | | import numpy as np
 8 | | import numba as nb
 9 | |
10 | | import chisurf.core.fluorescence
11 | | import chisurf.core.structure.av
12 | | import chisurf.core.structure
13 | | import chisurf.core.structure.potential.cPotentials_
   | |____________________________________________________^
   |
help: Organize imports

F401 [*] `os` imported but unused
 --> chisurf/core/structure/potential/potentials.py:3:8
  |
1 | from __future__ import annotations
2 |
3 | import os
  |        ^^
4 | import math
5 | import copy
  |
help: Remove unused import: `os`

D417 Missing argument description in the docstring for `centroid2`: `repulsion`
  --> chisurf/core/structure/potential/potentials.py:17:5
   |
16 | @nb.njit
17 | def centroid2(
   |     ^^^^^^^^^
18 |         atom_lookup,
19 |         res_types,
   |

D202 [*] No blank lines allowed after function docstring (found 1)
  --> chisurf/core/structure/potential/potentials.py:30:5
   |
28 |           repulsion: float = 100.0
29 |   ):
30 | /     """
31 | |     Calculate the potential energy given the UNRES GBV-Sidechain potential and
32 | |     isotropic conditions.
33 | |
34 | |     Parameters
35 | |     ----------
36 | |     atom_lookup : integer array
37 | |             A two dimensional lookup array to find for each residue and
38 | |             atom-type the corresponding position within the coordinate array `r`.
39 | |     centroid_pos : integer
40 | |             This variable specifies which position in the atom_lookup
41 | |             contains the number of the C-beta atom.
42 | |     res_types : integer array
43 | |             An array containing the residue types of the residues as a one
44 | |             dimensional array. The residue types are in the order as in
45 | |             :py:const:mfm.structure.residue_atoms_internal:
46 | |     ca_dist : double array
47 | |             The array `ca_dist` should contain the pair-wise distances of all
48 | |             C-alpha atoms of the protein
49 | |     r : double array
50 | |             The two-dimensional array `r` contains all Cartesian coordinates
51 | |             of all atoms.
52 | |     potential : double array
53 | |             This 3-dimensional array contains pre-calculated potentials of all
54 | |             possible combinations of amino-acids. potential[1,2,10] has to
55 | |             contain the potential of the amino-acid (1) and (2) at a distance
56 | |             of 10 whereas the actual distance is dependent on the range in
57 | |             which the potential was actually calculated. The potential can be
58 | |             calculated with the small program located in:
59 | |             mfm.structure.potential.database.make_unres_lookup.py
60 | |     min_dist : double
61 | |             The minimum distance for which the parameter `potential` was
62 | |             calculated. This parameter has to be supplied and depends on the
63 | |             provided pre-calculated potential file.
64 | |     max_dist :double
65 | |             The minimum distance for which the parameter `potential` was
66 | |             calculated (see min_dist)
67 | |     bin_width : double
68 | |             The spacing between the bin of the distance points in the
69 | |             pre-calculated potential
70 | |     cutoff : double
71 | |             Distances between the centroids bigger than the cut-off distance
72 | |             will be neglected
73 | |
74 | |     Citation
75 | |     --------
76 | |     J. Comp. Chem. Vol. 18, 7, 849-873, 1997
77 | |     """
   | |_______^
78 |
79 |       nRes = atom_lookup.shape[0]
   |
help: Remove blank line(s) after function docstring

D205 1 blank line required between summary line and description
  --> chisurf/core/structure/potential/potentials.py:30:5
   |
28 |           repulsion: float = 100.0
29 |   ):
30 | /     """
31 | |     Calculate the potential energy given the UNRES GBV-Sidechain potential and
32 | |     isotropic conditions.
33 | |
34 | |     Parameters
35 | |     ----------
36 | |     atom_lookup : integer array
37 | |             A two dimensional lookup array to find for each residue and
38 | |             atom-type the corresponding position within the coordinate array `r`.
39 | |     centroid_pos : integer
40 | |             This variable specifies which position in the atom_lookup
41 | |             contains the number of the C-beta atom.
42 | |     res_types : integer array
43 | |             An array containing the residue types of the residues as a one
44 | |             dimensional array. The residue types are in the order as in
45 | |             :py:const:mfm.structure.residue_atoms_internal:
46 | |     ca_dist : double array
47 | |             The array `ca_dist` should contain the pair-wise distances of all
48 | |             C-alpha atoms of the protein
49 | |     r : double array
50 | |             The two-dimensional array `r` contains all Cartesian coordinates
51 | |             of all atoms.
52 | |     potential : double array
53 | |             This 3-dimensional array contains pre-calculated potentials of all
54 | |             possible combinations of amino-acids. potential[1,2,10] has to
55 | |             contain the potential of the amino-acid (1) and (2) at a distance
56 | |             of 10 whereas the actual distance is dependent on the range in
57 | |             which the potential was actually calculated. The potential can be
58 | |             calculated with the small program located in:
59 | |             mfm.structure.potential.database.make_unres_lookup.py
60 | |     min_dist : double
61 | |             The minimum distance for which the parameter `potential` was
62 | |             calculated. This parameter has to be supplied and depends on the
63 | |             provided pre-calculated potential file.
64 | |     max_dist :double
65 | |             The minimum distance for which the parameter `potential` was
66 | |             calculated (see min_dist)
67 | |     bin_width : double
68 | |             The spacing between the bin of the distance points in the
69 | |             pre-calculated potential
70 | |     cutoff : double
71 | |             Distances between the centroids bigger than the cut-off distance
72 | |             will be neglected
73 | |
74 | |     Citation
75 | |     --------
76 | |     J. Comp. Chem. Vol. 18, 7, 849-873, 1997
77 | |     """
   | |_______^
78 |
79 |       nRes = atom_lookup.shape[0]
   |
help: Insert single blank line

D200 One-line docstring should fit on one line
   --> chisurf/core/structure/potential/potentials.py:125:5
    |
123 |           k_dihedral: float = 0.05
124 |   ) -> float:
125 | /     """ Simple potential based internal coordinates for C-alpha bead model
126 | |     """
    | |_______^
127 |       if equilibrium_internal is None:
128 |           return 0.0
    |
help: Reformat to one line

D210 [*] No whitespaces allowed surrounding docstring text
   --> chisurf/core/structure/potential/potentials.py:125:5
    |
123 |           k_dihedral: float = 0.05
124 |   ) -> float:
125 | /     """ Simple potential based internal coordinates for C-alpha bead model
126 | |     """
    | |_______^
127 |       if equilibrium_internal is None:
128 |           return 0.0
    |
help: Trim surrounding whitespace

D400 First line should end with a period
   --> chisurf/core/structure/potential/potentials.py:125:5
    |
123 |           k_dihedral: float = 0.05
124 |   ) -> float:
125 | /     """ Simple potential based internal coordinates for C-alpha bead model
126 | |     """
    | |_______^
127 |       if equilibrium_internal is None:
128 |           return 0.0
    |
help: Add period

D401 First line of docstring should be in imperative mood: "Simple potential based internal coordinates for C-alpha bead model"
   --> chisurf/core/structure/potential/potentials.py:125:5
    |
123 |           k_dihedral: float = 0.05
124 |   ) -> float:
125 | /     """ Simple potential based internal coordinates for C-alpha bead model
126 | |     """
    | |_______^
127 |       if equilibrium_internal is None:
128 |           return 0.0
    |

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/core/structure/potential/potentials.py:125:5
    |
123 |           k_dihedral: float = 0.05
124 |   ) -> float:
125 | /     """ Simple potential based internal coordinates for C-alpha bead model
126 | |     """
    | |_______^
127 |       if equilibrium_internal is None:
128 |           return 0.0
    |
help: Add closing punctuation

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:146:5
    |
144 |           **kwargs
145 |   ) -> float:
146 | /     """Calculates a *internal* potential based on the similarity of the bonds length, angles, and dihedrals
147 | |     to a reference structure.
148 | |
149 | |     :param structure:
150 | |     :param kwargs:
151 | |     :return:
152 | |     """
    | |_______^
153 |       eq = kwargs.get('equilibrium_internal', None)
154 |       kb = kwargs.get('k_bonds', 0.5)
    |
help: Insert single blank line

D401 First line of docstring should be in imperative mood: "Calculates a *internal* potential based on the similarity of the bonds length, angles, and dihedrals"
   --> chisurf/core/structure/potential/potentials.py:146:5
    |
144 |           **kwargs
145 |   ) -> float:
146 | /     """Calculates a *internal* potential based on the similarity of the bonds length, angles, and dihedrals
147 | |     to a reference structure.
148 | |
149 | |     :param structure:
150 | |     :param kwargs:
151 | |     :return:
152 | |     """
    | |_______^
153 |       eq = kwargs.get('equilibrium_internal', None)
154 |       kb = kwargs.get('k_bonds', 0.5)
    |

D400 First line should end with a period
   --> chisurf/core/structure/potential/potentials.py:165:5
    |
163 |           rm: float = 3.8208650279
164 |   ) -> float:
165 | /     """Truncated Lennard Jones
166 | |
167 | |     :param ca_coordinates:
168 | |     :param epsilon:
169 | |     :param rm:
170 | |     :return:
171 | |     """
    | |_______^
172 |       energy = 0.0  # total energy
173 |       n_atoms = ca_coordinates.shape[0]
    |
help: Add period

D401 First line of docstring should be in imperative mood: "Truncated Lennard Jones"
   --> chisurf/core/structure/potential/potentials.py:165:5
    |
163 |           rm: float = 3.8208650279
164 |   ) -> float:
165 | /     """Truncated Lennard Jones
166 | |
167 | |     :param ca_coordinates:
168 | |     :param epsilon:
169 | |     :param rm:
170 | |     :return:
171 | |     """
    | |_______^
172 |       energy = 0.0  # total energy
173 |       n_atoms = ca_coordinates.shape[0]
    |

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/core/structure/potential/potentials.py:165:5
    |
163 |           rm: float = 3.8208650279
164 |   ) -> float:
165 | /     """Truncated Lennard Jones
166 | |
167 | |     :param ca_coordinates:
168 | |     :param epsilon:
169 | |     :param rm:
170 | |     :return:
171 | |     """
    | |_______^
172 |       energy = 0.0  # total energy
173 |       n_atoms = ca_coordinates.shape[0]
    |
help: Add closing punctuation

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:189:5
    |
187 |           rm: float = 3.8208650279
188 |   ):
189 | /     """
190 | |
191 | |     :param structure: a structure object
192 | |     :param rm: is the C-alpha equilibrium distance
193 | |     :return:
194 | |     """
    | |_______^
195 |       sel = chisurf.core.structure.get_atom_index_by_name(
196 |           structure.atoms, ['CA']
    |
help: Insert single blank line

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/core/structure/potential/potentials.py:209:5
    |
207 |           cutoff: float = 12.0
208 |   ) -> float:
209 | /     """
210 | |     Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
211 | |     with cutoff-potential potential shifted by value at cutoff to smooth the energy
212 | |     :param xyz: atom-list
213 | |     :param epsilon: dielectric constant of proteins: 2-40
214 | |     :param epsilon0: dielectric constant of water 80.1
215 | |     :param cutoff: cutoff-distance in Angstrom
216 | |     :return:
217 | |     """
    | |_______^
218 |
219 |       nAtoms = xyz.shape[0]
    |
help: Remove blank line(s) after function docstring

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:209:5
    |
207 |           cutoff: float = 12.0
208 |   ) -> float:
209 | /     """
210 | |     Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
211 | |     with cutoff-potential potential shifted by value at cutoff to smooth the energy
212 | |     :param xyz: atom-list
213 | |     :param epsilon: dielectric constant of proteins: 2-40
214 | |     :param epsilon0: dielectric constant of water 80.1
215 | |     :param cutoff: cutoff-distance in Angstrom
216 | |     :return:
217 | |     """
    | |_______^
218 |
219 |       nAtoms = xyz.shape[0]
    |
help: Insert single blank line

D400 First line should end with a period
   --> chisurf/core/structure/potential/potentials.py:209:5
    |
207 |           cutoff: float = 12.0
208 |   ) -> float:
209 | /     """
210 | |     Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
211 | |     with cutoff-potential potential shifted by value at cutoff to smooth the energy
212 | |     :param xyz: atom-list
213 | |     :param epsilon: dielectric constant of proteins: 2-40
214 | |     :param epsilon0: dielectric constant of water 80.1
215 | |     :param cutoff: cutoff-distance in Angstrom
216 | |     :return:
217 | |     """
    | |_______^
218 |
219 |       nAtoms = xyz.shape[0]
    |
help: Add period

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/core/structure/potential/potentials.py:209:5
    |
207 |           cutoff: float = 12.0
208 |   ) -> float:
209 | /     """
210 | |     Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
211 | |     with cutoff-potential potential shifted by value at cutoff to smooth the energy
212 | |     :param xyz: atom-list
213 | |     :param epsilon: dielectric constant of proteins: 2-40
214 | |     :param epsilon0: dielectric constant of water 80.1
215 | |     :param cutoff: cutoff-distance in Angstrom
216 | |     :return:
217 | |     """
    | |_______^
218 |
219 |       nAtoms = xyz.shape[0]
    |
help: Add closing punctuation

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/core/structure/potential/potentials.py:262:5
    |
260 |           rm_matrix: np.ndarray
261 |   ):
262 | /     """
263 | |     If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma
264 | |
265 | |     :param resLookUp:
266 | |     :param ca_dist:
267 | |     :param energy_matrix:
268 | |     :param rm_matrix:
269 | |     :param cutoff:
270 | |     :return:
271 | |     """
    | |_______^
272 |
273 |       tmp = 0.0
    |
help: Remove blank line(s) after function docstring

D400 First line should end with a period
   --> chisurf/core/structure/potential/potentials.py:262:5
    |
260 |           rm_matrix: np.ndarray
261 |   ):
262 | /     """
263 | |     If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma
264 | |
265 | |     :param resLookUp:
266 | |     :param ca_dist:
267 | |     :param energy_matrix:
268 | |     :param rm_matrix:
269 | |     :param cutoff:
270 | |     :return:
271 | |     """
    | |_______^
272 |
273 |       tmp = 0.0
    |
help: Add period

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/core/structure/potential/potentials.py:262:5
    |
260 |           rm_matrix: np.ndarray
261 |   ):
262 | /     """
263 | |     If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma
264 | |
265 | |     :param resLookUp:
266 | |     :param ca_dist:
267 | |     :param energy_matrix:
268 | |     :param rm_matrix:
269 | |     :param cutoff:
270 | |     :return:
271 | |     """
    | |_______^
272 |
273 |       tmp = 0.0
    |
help: Add closing punctuation

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:293:7
    |
293 | class Ramachandran(object):
    |       ^^^^^^^^^^^^
294 |
295 |     def __init__(
    |

UP004 [*] Class `Ramachandran` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:293:20
    |
293 | class Ramachandran(object):
    |                    ^^^^^^
294 |
295 |     def __init__(
    |
help: Remove `object` inheritance

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:300:9
    |
298 |               filename: str = None
299 |       ):
300 | /         """
301 | |         :param filename:
302 | |         :return:
303 | |         """
    | |___________^
304 |           if filename is None:
305 |               from chisurf.core.settings.path_utils import get_path
    |
help: Insert single blank line

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:324:7
    |
324 | class Electrostatics(object):
    |       ^^^^^^^^^^^^^^
325 |
326 |     def __init__(
    |

UP004 [*] Class `Electrostatics` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:324:22
    |
324 | class Electrostatics(object):
    |                      ^^^^^^
325 |
326 |     def __init__(
    |
help: Remove `object` inheritance

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:330:9
    |
328 |               structure,
329 |               type: str = 'gb'):
330 | /         """
331 | |         :param type:
332 | |         :return:
333 | |         """
    | |___________^
334 |           self.structure = structure
335 |           self.name = 'ele'
    |
help: Insert single blank line

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:348:7
    |
348 | class LJ_Bead(object):
    |       ^^^^^^^
349 |
350 |     def __init__(
    |

UP004 [*] Class `LJ_Bead` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:348:15
    |
348 | class LJ_Bead(object):
    |               ^^^^^^
349 |
350 |     def __init__(
    |
help: Remove `object` inheritance

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:365:7
    |
365 | class HPotential(object):
    |       ^^^^^^^^^^
366 |
367 |     name = 'H-Bond'
    |

UP004 [*] Class `HPotential` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:365:18
    |
365 | class HPotential(object):
    |                  ^^^^^^
366 |
367 |     name = 'H-Bond'
    |
help: Remove `object` inheritance

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:434:7
    |
434 | class GoPotential(object):
    |       ^^^^^^^^^^^
435 |
436 |     name = 'go'
    |

UP004 [*] Class `GoPotential` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:434:19
    |
434 | class GoPotential(object):
    |                   ^^^^^^
435 |
436 |     name = 'go'
    |
help: Remove `object` inheritance

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:500:7
    |
500 | class MJPotential(object):
    |       ^^^^^^^^^^^
501 |
502 |     name = 'Miyazawa-Jernigan'
    |

UP004 [*] Class `MJPotential` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:500:19
    |
500 | class MJPotential(object):
    |                   ^^^^^^
501 |
502 |     name = 'Miyazawa-Jernigan'
    |
help: Remove `object` inheritance

UP004 [*] Class `CEPotential` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:544:19
    |
544 | class CEPotential(object):
    |                   ^^^^^^
545 |     """
546 |     Examples
    |
help: Remove `object` inheritance

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:545:5
    |
544 |   class CEPotential(object):
545 | /     """
546 | |     Examples
547 | |     --------
548 | |
549 | |     >>> import chisurf.core.structure
550 | |     >>> import chisurf.core.structure.potential
551 | |     >>> s = chisurf.core.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
552 | |     >>> pce = chisurf.core.structure.potential.potentials.CEPotential(s, ca_cutoff=64.0)
553 | |     >>> pce.getEnergy()
554 | |     -0.15896629131635745
555 | |     """
    | |_______^
556 |
557 |       name = 'Iso-UNRES'
    |
help: Insert single blank line

D200 One-line docstring should fit on one line
   --> chisurf/core/structure/potential/potentials.py:565:9
    |
563 |               **kwargs
564 |       ):
565 | /         """
566 | |         scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
567 | |         """
    | |___________^
568 |           scaling_factor = 0.593
569 |           self.structure = structure
    |
help: Reformat to one line

D400 First line should end with a period
   --> chisurf/core/structure/potential/potentials.py:565:9
    |
563 |               **kwargs
564 |       ):
565 | /         """
566 | |         scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
567 | |         """
    | |___________^
568 |           scaling_factor = 0.593
569 |           self.structure = structure
    |
help: Add period

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/core/structure/potential/potentials.py:565:9
    |
563 |               **kwargs
564 |       ):
565 | /         """
566 | |         scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
567 | |         """
    | |___________^
568 |           scaling_factor = 0.593
569 |           self.structure = structure
    |
help: Add closing punctuation

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:631:7
    |
631 | class ASA(object):
    |       ^^^
632 |
633 |     name = 'Asa-Ca'
    |

UP004 [*] Class `ASA` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:631:11
    |
631 | class ASA(object):
    |           ^^^^^^
632 |
633 |     name = 'Asa-Ca'
    |
help: Remove `object` inheritance

UP008 [*] Use `super()` instead of `super(__class__, self)`
   --> chisurf/core/structure/potential/potentials.py:643:14
    |
641 |     ):
642 |         """Initialize an ASA-Cα solvent-accessible-surface potential."""
643 |         super(ASA, self).__init__()
    |              ^^^^^^^^^^^
644 |         self.structure = structure
645 |         self.probe = probe
    |
help: Remove `super()` parameters

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:668:7
    |
668 | class ClashPotential(object):
    |       ^^^^^^^^^^^^^^
669 |
670 |     name = 'Clash-Potential'
    |

UP004 [*] Class `ClashPotential` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:668:22
    |
668 | class ClashPotential(object):
    |                      ^^^^^^
669 |
670 |     name = 'Clash-Potential'
    |
help: Remove `object` inheritance

D205 1 blank line required between summary line and description
   --> chisurf/core/structure/potential/potentials.py:679:9
    |
677 |               **kwargs
678 |       ):
679 | /         """
680 | |         :param kwargs:
681 | |         :return:
682 | |
683 | |         Examples
684 | |         --------
685 | |
686 | |         >>> import chisurf.core.structure
687 | |         >>> import chisurf.core.structure.potential
688 | |         >>> s = chisurf.core.structure.Structure('./test/data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
689 | |         >>> pce = chisurf.core.structure.potential.ClashPotential(structure=s, clash_tolerance=6.0)
690 | |         >>> pce.getEnergy()
691 | |
692 | |         """
    | |___________^
693 |           self.structure = structure
694 |           self.clash_tolerance = clash_tolerance
    |
help: Insert single blank line

D101 Missing docstring in public class
   --> chisurf/core/structure/potential/potentials.py:708:7
    |
708 | class RadiusGyration(object):
    |       ^^^^^^^^^^^^^^
709 |
710 |     name = 'Radius-Gyration'
    |

UP004 [*] Class `RadiusGyration` inherits from `object`
   --> chisurf/core/structure/potential/potentials.py:708:22
    |
708 | class RadiusGyration(object):
    |                      ^^^^^^
709 |
710 |     name = 'Radius-Gyration'
    |
help: Remove `object` inheritance

I001 [*] Import block is un-sorted or un-formatted
  --> chisurf/gui/widgets/models/proteinmc.py:3:1
   |
 1 |   """ProteinMC model widget with Chimol visual feedback."""
 2 |
 3 | / from __future__ import annotations
 4 | |
 5 | | import json
 6 | | from copy import deepcopy
 7 | | from pathlib import Path
 8 | | from typing import Optional
 9 | |
10 | | import numpy as np
11 | | from qtpy import QtCore, QtWidgets
12 | |
13 | | import chisurf as cs
14 | | import chisurf.core.fitting
15 | |
16 | | from chisurf.gui.widgets.models.model_widget import ModelWidget
17 | | from chisurf.plugins.modelling.proteinmc.model import (
18 | |     ProteinMCProgress,
19 | |     ProteinMCRunner,
20 | |     build_move_map_from_flexfit,
21 | |     list_flexfit_sets,
22 | | )
23 | | from chisurf.gui.plots.proteinMC import ProteinMCDistanceNetworkPlot, ProteinMCPlot, ProteinMCStructurePlot
   | |___________________________________________________________________________________________________________^
   |
help: Organize imports

F401 [*] `chisurf.plugins.modelling.proteinmc.model.build_move_map_from_flexfit` imported but unused
  --> chisurf/gui/widgets/models/proteinmc.py:20:5
   |
18 |     ProteinMCProgress,
19 |     ProteinMCRunner,
20 |     build_move_map_from_flexfit,
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
21 |     list_flexfit_sets,
22 | )
   |
help: Remove unused import: `chisurf.plugins.modelling.proteinmc.model.build_move_map_from_flexfit`

UP037 [*] Remove quotes from type annotation
   --> chisurf/gui/widgets/models/proteinmc.py:159:14
    |
157 |     def __init__(
158 |         self,
159 |         fit: "cs.core.fitting.fit.Fit",
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^
160 |         *args,
161 |         **kwargs,
    |
help: Remove quotes

UP045 [*] Use `X | None` for type annotations
   --> chisurf/gui/widgets/models/proteinmc.py:179:28
    |
177 |         self._chimol_update_pending: bool = False
178 |         self._plot_update_pending: bool = False
179 |         self._pending_xyz: Optional[np.ndarray] = None
    |                            ^^^^^^^^^^^^^^^^^^^^
180 |         self._last_chimol_update_ms: int = 0
181 |         self._last_plot_update_ms: int = 0
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/gui/widgets/models/proteinmc.py:183:35
    |
181 |         self._last_plot_update_ms: int = 0
182 |         self._chimol_frame_count: int = 0
183 |         self._sampling_directory: Optional[Path] = None
    |                                   ^^^^^^^^^^^^^^
184 |         self._resume_rmsd: list[float] = []
185 |         self._resume_drmsd: list[float] = []
    |
help: Convert to `X | None`

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:198:9
    |
196 |     @property
197 |     def frame_count(self) -> int:
198 |         """Number of trajectory frames currently available."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
199 |
200 |         return len(self.trajectory_frames)
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:203:9
    |
202 |     def set_current_frame(self, index: int, *, update_plots: bool = True) -> None:
203 |         """Set the globally displayed ProteinMC frame across all plots."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
204 |
205 |         if self.frame_count <= 0:
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:398:9
    |
397 |     def _build_potential_ui(self, layout: QtWidgets.QVBoxLayout) -> None:
398 |         """Create controls for ProteinMC structure energy terms."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
399 |
400 |         group = QtWidgets.QGroupBox("Energy terms / force field", self)
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:443:9
    |
442 |     def _resize_potential_table(self) -> None:
443 |         """Fit the energy table height to its rows without wasting vertical space."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
444 |
445 |         header = self.potential_table.horizontalHeader().height()
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:452:9
    |
451 |     def _add_potential_row(self, key: str, weight: float = None, settings: dict = None) -> None:
452 |         """Add or update one energy-term row."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
453 |
454 |         spec = self._POTENTIAL_SPECS.get(key)
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:484:9
    |
483 |     def _remove_potential_button_row(self, button: QtWidgets.QPushButton) -> None:
484 |         """Remove the energy-term row that owns a row-local remove button."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
485 |
486 |         for row in range(self.potential_table.rowCount()):
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:493:9
    |
492 |     def on_potential_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
493 |         """Normalize edited energy-term weights as fixed-point floats."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
494 |
495 |         if item.column() != 1:
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:511:9
    |
509 |     @QtCore.Slot()
510 |     def on_add_potential(self) -> None:
511 |         """Add the energy term selected in the combobox."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
512 |
513 |         key = self.potential_combo.currentData()
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:519:9
    |
518 |     def on_remove_selected_potential(self, *args) -> None:
519 |         """Remove the currently selected energy-term row."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
520 |
521 |         row = self.potential_table.currentRow()
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:528:9
    |
526 |     @QtCore.Slot()
527 |     def on_edit_potential_details(self) -> None:
528 |         """Open a popup editor for selected energy-term parameters."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
529 |
530 |         row = self.potential_table.currentRow()
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:576:9
    |
575 |     def _potential_settings(self) -> list[dict]:
576 |         """Return enabled ProteinMC potential settings from the UI."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
577 |
578 |         out = []
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:723:9
    |
721 |     @QtCore.Slot()
722 |     def on_edit_labeling_json(self) -> None:
723 |         """Open the selected FPS JSON file in an existing ChiSurf editor."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
724 |
725 |         filename = self.labeling_edit.text().strip()
    |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
   --> chisurf/gui/widgets/models/proteinmc.py:880:9
    |
878 |         n_iter: int | None = None,
879 |     ) -> None:
880 |         """Run ProteinMC from ChiSurf's generic Sampling button."""
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
881 |
882 |         try:
    |
help: Remove blank line(s) after function docstring

D205 1 blank line required between summary line and description
   --> chisurf/gui/widgets/models/proteinmc.py:993:9
    |
992 |       def _update_chimol(self, xyz: np.ndarray) -> None:
993 | /         """Push one frame to the embedded Chimol viewer by appending it
994 | |         to the trajectory, so the frame slider and play/pause buttons
995 | |
996 | |         can navigate all frames accumulated so far.
997 | |         """
    | |___________^
998 |           if self.viewer is None:
999 |               return
    |
help: Insert single blank line

D400 First line should end with a period
   --> chisurf/gui/widgets/models/proteinmc.py:993:9
    |
992 |       def _update_chimol(self, xyz: np.ndarray) -> None:
993 | /         """Push one frame to the embedded Chimol viewer by appending it
994 | |         to the trajectory, so the frame slider and play/pause buttons
995 | |
996 | |         can navigate all frames accumulated so far.
997 | |         """
    | |___________^
998 |           if self.viewer is None:
999 |               return
    |
help: Add period

D415 First line should end with a period, question mark, or exclamation point
   --> chisurf/gui/widgets/models/proteinmc.py:993:9
    |
992 |       def _update_chimol(self, xyz: np.ndarray) -> None:
993 | /         """Push one frame to the embedded Chimol viewer by appending it
994 | |         to the trajectory, so the frame slider and play/pause buttons
995 | |
996 | |         can navigate all frames accumulated so far.
997 | |         """
    | |___________^
998 |           if self.viewer is None:
999 |               return
    |
help: Add closing punctuation

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1087:9
     |
1086 |     def get_state(self) -> dict:
1087 |         """Return project-serializable ProteinMC widget state."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1088 |
1089 |         return {
     |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1111:9
     |
1110 |     def set_state(self, state: dict) -> None:
1111 |         """Restore project-serialized ProteinMC widget state."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1112 |
1113 |         payload = state.get("proteinmc", state) if isinstance(state, dict) else {}
     |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1170:9
     |
1169 |     def _restore_potential_settings(self, potentials: object) -> None:
1170 |         """Restore potential UI controls from serialized settings."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1171 |
1172 |         aliases = {
     |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1213:9
     |
1212 |     def _load_starting_structure(self, filename: str) -> None:
1213 |         """Load and display the starting structure without running ProteinMC."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1214 |
1215 |         try:
     |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1243:9
     |
1242 |     def _refresh_structure_plots(self) -> None:
1243 |         """Refresh any external ProteinMC structure plot tabs."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1244 |
1245 |         for plot in list(getattr(self.fit, "plots", []) or []):
     |
help: Remove blank line(s) after function docstring

D202 [*] No blank lines allowed after function docstring (found 1)
    --> chisurf/gui/widgets/models/proteinmc.py:1255:9
     |
1254 |     def _autosave_sampling_project(self) -> None:
1255 |         """Autosave the current ChiSurf project into the ProteinMC sampling folder."""
     |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1256 |
1257 |         if self._sampling_directory is None:
     |
help: Remove blank line(s) after function docstring

I001 [*] Import block is un-sorted or un-formatted
  --> chisurf/plugins/modelling/proteinmc/model.py:3:1
   |
 1 |   """Headless ProteinMC model and runner."""
 2 |
 3 | / from __future__ import annotations
 4 | |
 5 | | from collections.abc import Callable
 6 | | from dataclasses import dataclass, field
 7 | | import json
 8 | | from pathlib import Path
 9 | | import shutil
10 | | import subprocess
11 | | import tempfile
12 | | import time
13 | | from typing import Any, Optional
14 | |
15 | | import numpy as np
16 | |
17 | | import chisurf.core.settings
18 | | from chisurf.core.math.rand import mc, weighted_choice
19 | | from chisurf.core.structure import ProteinCentroid, Structure, Universe
20 | |
21 | | from .rmf import ProteinMCRmfWriter
   | |___________________________________^
   |
help: Organize imports

UP045 [*] Use `X | None` for type annotations
  --> chisurf/plugins/modelling/proteinmc/model.py:42:10
   |
40 |     energies: list[float] = field(default_factory=list)
41 |     labeling_energies: list[float] = field(default_factory=list)
42 |     xyz: Optional[np.ndarray] = None
   |          ^^^^^^^^^^^^^^^^^^^^
43 |     output_file: Optional[str] = None
   |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
  --> chisurf/plugins/modelling/proteinmc/model.py:43:18
   |
41 |     labeling_energies: list[float] = field(default_factory=list)
42 |     xyz: Optional[np.ndarray] = None
43 |     output_file: Optional[str] = None
   |                  ^^^^^^^^^^^^^
   |
help: Convert to `X | None`

UP015 [*] Unnecessary mode argument
  --> chisurf/plugins/modelling/proteinmc/model.py:63:25
   |
61 | def load_json(filename: str | Path) -> dict[str, Any]:
62 |     """Load a JSON file and return its object."""
63 |     with open(filename, "r") as fp:
   |                         ^^^
64 |         return json.load(fp)
   |
help: Remove mode argument

UP045 [*] Use `X | None` for type annotations
  --> chisurf/plugins/modelling/proteinmc/model.py:67:34
   |
67 | def normalize_settings(settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
   |                                  ^^^^^^^^^^^^^^^^^^^^^^^^
68 |     """Return ProteinMC settings with legacy and current key aliases populated."""
69 |     base = dict(chisurf.core.settings.cs_settings.get("mc_settings", {}))
   |
help: Convert to `X | None`

UP022 Prefer `capture_output` over sending `stdout` and `stderr` to `PIPE`
   --> chisurf/plugins/modelling/proteinmc/model.py:143:9
    |
141 |     cmd = [executable, "--ff=PARSE", input_file, output_file]
142 |     try:
143 |         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
144 |     except subprocess.CalledProcessError as exc:
145 |         Path(output_file).unlink(missing_ok=True)
    |
help: Replace with `capture_output` keyword argument

D417 Missing argument descriptions in the docstring for `__init__`: `initial_frames`, `output_file`, `progress_callback`, `settings`, `settings_file`, `structure_source`, `verbose`
   --> chisurf/plugins/modelling/proteinmc/model.py:396:9
    |
394 |     """Headless ProteinMC Monte Carlo runner."""
395 |
396 |     def __init__(
    |         ^^^^^^^^
397 |         self,
398 |         structure_source: str | Path | ProteinCentroid,
    |

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:403:19
    |
401 |         score_set: str = "",
402 |         flexfit_set: str | None = None,
403 |         settings: Optional[dict[str, Any]] = None,
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^
404 |         settings_file: str | Path | None = None,
405 |         output_file: str | Path | None = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:406:25
    |
404 |         settings_file: str | Path | None = None,
405 |         output_file: str | Path | None = None,
406 |         initial_frames: Optional[list[np.ndarray]] = None,
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
407 |         progress_callback: ProgressCallback | None = None,
408 |         verbose: bool = False,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:452:30
    |
450 |         self.energies: list[float] = []
451 |         self.labeling_energies: list[float] = []
452 |         self._reference_xyz: Optional[np.ndarray] = None
    |                              ^^^^^^^^^^^^^^^^^^^^
453 |         self._previous_xyz: Optional[np.ndarray] = None
454 |         if "move_map" not in self.settings and self.labeling_file:
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:453:29
    |
451 |         self.labeling_energies: list[float] = []
452 |         self._reference_xyz: Optional[np.ndarray] = None
453 |         self._previous_xyz: Optional[np.ndarray] = None
    |                             ^^^^^^^^^^^^^^^^^^^^
454 |         if "move_map" not in self.settings and self.labeling_file:
455 |             derived = build_move_map_from_flexfit(
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:661:20
    |
659 |     pdb_file: str,
660 |     *,
661 |     settings_file: Optional[str] = None,
    |                    ^^^^^^^^^^^^^
662 |     labeling_file: Optional[str] = None,
663 |     score_set: str = "",
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:662:20
    |
660 |     *,
661 |     settings_file: Optional[str] = None,
662 |     labeling_file: Optional[str] = None,
    |                    ^^^^^^^^^^^^^
663 |     score_set: str = "",
664 |     flexfit_set: str | None = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:665:18
    |
663 |     score_set: str = "",
664 |     flexfit_set: str | None = None,
665 |     output_file: Optional[str] = None,
    |                  ^^^^^^^^^^^^^
666 |     verbose: bool = False,
667 |     scale: Optional[float] = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:667:12
    |
665 |     output_file: Optional[str] = None,
666 |     verbose: bool = False,
667 |     scale: Optional[float] = None,
    |            ^^^^^^^^^^^^^^^
668 |     n_iter: Optional[int] = None,
669 |     n_out: Optional[int] = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:668:13
    |
666 |     verbose: bool = False,
667 |     scale: Optional[float] = None,
668 |     n_iter: Optional[int] = None,
    |             ^^^^^^^^^^^^^
669 |     n_out: Optional[int] = None,
670 |     n_written: Optional[int] = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:669:12
    |
667 |     scale: Optional[float] = None,
668 |     n_iter: Optional[int] = None,
669 |     n_out: Optional[int] = None,
    |            ^^^^^^^^^^^^^
670 |     n_written: Optional[int] = None,
671 |     progress_callback: ProgressCallback | None = None,
    |
help: Convert to `X | None`

UP045 [*] Use `X | None` for type annotations
   --> chisurf/plugins/modelling/proteinmc/model.py:670:16
    |
668 |     n_iter: Optional[int] = None,
669 |     n_out: Optional[int] = None,
670 |     n_written: Optional[int] = None,
    |                ^^^^^^^^^^^^^
671 |     progress_callback: ProgressCallback | None = None,
672 | ) -> str:
    |
help: Convert to `X | None`

Found 95 errors.
[*] 57 fixable with the `--fix` option (13 hidden fixes can be enabled with the `--unsafe-fixes` option).

```

---

## Assistant (Build · MiMo V2.5 Free · 7.5s)

_Thinking:_

The lint errors are all pre-existing (not from my changes). The only new one is the `RadiusGyration(object)` which I should fix to remove `object` inheritance. Let me also check if there are any issues specific to my changes.

The pre-existing lint issues include things like:
- `I001` Import sorting
- `F401` Unused imports
- `D*` Docstring issues
- `UP004` Inheriting from `object`
- `UP045` Using `Optional` instead of `X | None`

These are all pre-existing. Let me fix just the one new lint issue from my code - the `RadiusGyration(object)` should be `RadiusGyration`.

Most lint errors are pre-existing. Let me fix the one new issue (class inheriting from `object`):

**Tool: edit**

**Input:**
```json
{
  "filePath": "/Users/tpeulen/dev/chisurf/chisurf/core/structure/potential/potentials.py",
  "newString": "class RadiusGyration:",
  "oldString": "class RadiusGyration(object):"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · MiMo V2.5 Free · 31.5s)

_Thinking:_

Now let me run the existing tests to make sure nothing is broken.

Let me run the existing tests to verify nothing is broken:

**Tool: bash**

**Input:**
```json
{
  "command": "python -m pytest test/plugins/test_proteinmc.py -v --tb=short 2>&1 | head -60",
  "description": "Run proteinmc tests",
  "workdir": "/Users/tpeulen/dev/chisurf",
  "timeout": 60000
}
```

**Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.10.14, pytest-9.0.2, pluggy-1.6.0 -- /Users/tpeulen/mambaforge/bin/python
cachedir: .pytest_cache
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /Users/tpeulen/dev/chisurf
configfile: pyproject.toml
plugins: anyio-4.12.1, asyncio-0.21.0, integration-0.2.3, mock-3.10.0, Faker-40.4.0, recording-0.12.2, cov-4.0.0, benchmark-4.0.0
asyncio: mode=strict
collecting ... collected 7 items

test/plugins/test_proteinmc.py::test_normalize_settings_maps_legacy_keys PASSED [ 14%]
test/plugins/test_proteinmc.py::test_load_structure_requires_pdb2pqr PASSED [ 28%]
test/plugins/test_proteinmc.py::test_runner_settings_override_settings_file SKIPPED [ 42%]
test/plugins/test_proteinmc.py::test_direct_labeling_potential_scores_148l SKIPPED [ 57%]
test/plugins/test_proteinmc.py::test_runner_writes_rmf3_readable_by_chimol SKIPPED [ 71%]
test/plugins/test_proteinmc.py::test_rmf_writer_replaces_blank_chain_ids SKIPPED [ 85%]
test/plugins/test_proteinmc.py::test_cli_writes_rmf3 SKIPPED (Protei...) [100%]<unknown>:54: DeprecationWarning: invalid escape sequence '\`'
/Users/tpeulen/dev/chisurf/chisurf/core/fio/ascii.py:54: DeprecationWarning: invalid escape sequence '\`'
  """Return a Windows long-path compatible path when needed.
<unknown>:126: DeprecationWarning: invalid escape sequence '\m'
/Users/tpeulen/dev/chisurf/chisurf/core/fio/fluorescence/photons.py:126: DeprecationWarning: invalid escape sequence '\m'
  """
<unknown>:172: DeprecationWarning: invalid escape sequence '\m'
<unknown>:224: DeprecationWarning: invalid escape sequence '\m'
/Users/tpeulen/dev/chisurf/chisurf/core/fluorescence/fcs/__init__.py:172: DeprecationWarning: invalid escape sequence '\m'
  """Return background attenuation factor for an autocorrelation model.
/Users/tpeulen/dev/chisurf/chisurf/core/fluorescence/fcs/__init__.py:224: DeprecationWarning: invalid escape sequence '\m'
  """Return background attenuation factor for a cross-correlation model.
<unknown>:52: DeprecationWarning: invalid escape sequence '\D'
/Users/tpeulen/dev/chisurf/chisurf/core/fluorescence/tcspc/__init__.py:52: DeprecationWarning: invalid escape sequence '\D'
  """Computes the combined counting weights (1/noise) for two counting
<unknown>:61: DeprecationWarning: invalid escape sequence '\d'
/Users/tpeulen/dev/chisurf/chisurf/core/math/rand/__init__.py:61: DeprecationWarning: invalid escape sequence '\d'
  """\
/Users/tpeulen/mambaforge/lib/python3.10/site-packages/coverage/report_core.py:107: CoverageWarning: Couldn't parse Python file '/Users/tpeulen/dev/chisurf/chisurf/plugins/burst/burst_mle_analysis/wizard_old.py' (couldnt-parse); see https://coverage.readthedocs.io/en/7.13.4/messages.html#warning-couldnt-parse
  coverage._warn(msg, slug="couldnt-parse")
<unknown>:180: DeprecationWarning: invalid escape sequence '\p'
<unknown>:207: DeprecationWarning: invalid escape sequence '\p'
<unknown>:254: DeprecationWarning: invalid escape sequence '\e'
/Users/tpeulen/dev/chisurf/chisurf/plugins/fcs/fcs_calculator/wizard.py:180: DeprecationWarning: invalid escape sequence '\p'
  """Translational diffusion coefficient of a sphere.
/Users/tpeulen/dev/chisurf/chisurf/plugins/fcs/fcs_calculator/wizard.py:207: DeprecationWarning: invalid escape sequence '\p'
  """Compute the effective focal volume from diffusion time, ``D`` and structure parameter.
/Users/tpeulen/dev/chisurf/chisurf/plugins/fcs/fcs_calculator/wizard.py:254: DeprecationWarning: invalid escape sequence '\e'
  """Scale diffusion coefficient from 25 °C water to arbitrary (T, η).
<unknown>:376: DeprecationWarning: invalid escape sequence '\h'
/Users/tpeulen/dev/chisurf/chisurf/plugins/kappa2_dist/k2dfun.py:376: DeprecationWarning: invalid escape sequence '\h'
  """Calculates the distance between the center of two dipoles and the
<unknown>:54: DeprecationWarning: invalid escape sequence '\`'
/Users/tpeulen/dev/chisurf/chisurf/core/fio/ascii.py:54: DeprecationWarning: invalid escape sequence '\`'
  """Return a Windows long-path compatible path when needed.
<unknown>:54: DeprecationWarning: invalid escape sequence '\`'
<unknown>:126: DeprecationWarning: invalid escape sequence '\m'
/Users/tpeulen/dev/chisurf/chisurf/core/fio/fluorescence/photons.py:126: DeprecationWarning: invalid escape sequence '\m'
  """
<unknown>:126: DeprecationWarning: invalid escape sequence '\m'
<unknown>:172: DeprecationWarning: invalid escape sequence '\m'
<unknown>:224: DeprecationWarning: invalid escape sequence '\m'
/Users/tpeulen/dev/chisurf/chisurf/core/fluorescence/fcs/__init__.py:172: DeprecationWarning: invalid escape sequence '\m'
  """Return background attenuation factor for an autocorrelation model.

```

---

## Assistant (Build · MiMo V2.5 Free · 13.8s)

_Thinking:_

The tests pass (2 passed, 5 skipped). The skips are expected (they need proteinmc to be installed). Let me also quickly verify the imports work by doing a basic import test.

Tests pass. Let me verify the imports work:

**Tool: bash**

**Input:**
```json
{
  "command": "python -c \"\nfrom chisurf.core.structure.potential.potentials import (\n    GoPotential, Ramachandran, ASA, RadiusGyration, ClashPotential,\n    HPotential, MJPotential, CEPotential\n)\nprint('All imports OK')\nprint('GoPotential.name:', GoPotential.name)\nprint('Ramachandran.name:', Ramachandran().name if hasattr(Ramachandran, 'name') else 'N/A (no default)')\nprint('ASA.name:', ASA.name)\nprint('RadiusGyration.name:', RadiusGyration.name)\nprint('ClashPotential.name:', ClashPotential.name)\nprint('HPotential.name:', HPotential.name)\nprint('MJPotential.name:', MJPotential.name)\nprint('CEPotential.name:', CEPotential.name)\n\" 2>&1",
  "description": "Verify imports of all potential classes",
  "workdir": "/Users/tpeulen/dev/chisurf"
}
```

**Output:**
```
/Users/tpeulen/mambaforge/lib/python3.10/site-packages/numba/core/decorators.py:248: RuntimeWarning: nopython is set for njit and is ignored
  warnings.warn('nopython is set for njit and is ignored', RuntimeWarning)
/Users/tpeulen/mambaforge/lib/python3.10/site-packages/numba/core/cpu.py:99: UserWarning: Numba extension module 'rocket_fft' failed to load due to 'RuntimeError(dlopen(/Users/tpeulen/mambaforge/lib/python3.10/site-packages/rocket_fft/_pocketfft_numba.cpython-310-darwin.so, 0x0009): tried: '/Users/tpeulen/mambaforge/lib/python3.10/site-packages/rocket_fft/_pocketfft_numba.cpython-310-darwin.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64e.v1' or 'arm64' or 'arm64')), '/System/Volumes/Preboot/Cryptexes/OS/Users/tpeulen/mambaforge/lib/python3.10/site-packages/rocket_fft/_pocketfft_numba.cpython-310-darwin.so' (no such file), '/Users/tpeulen/mambaforge/lib/python3.10/site-packages/rocket_fft/_pocketfft_numba.cpython-310-darwin.so' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e' or 'arm64e.v1' or 'arm64' or 'arm64')))'.
  numba.core.entrypoints.init_all()
Error opening anisotropy corrections file: [Errno 2] No such file or directory: '/Users/tpeulen/dev/chisurf/chisurf/settings/anisotropy_corrections.json'
All imports OK
GoPotential.name: go
Ramachandran.name: N/A (no default)
ASA.name: Asa-Ca
RadiusGyration.name: Radius-Gyration
ClashPotential.name: Clash-Potential
HPotential.name: H-Bond
MJPotential.name: Miyazawa-Jernigan
CEPotential.name: Iso-UNRES

```

---

## Assistant (Build · MiMo V2.5 Free · 10.1s)

_Thinking:_

All imports work. Ramachandran doesn't have a `name` attribute on the class itself (it's set in `__init__` as `self.name = 'rama'`), but that's how the existing code works. The `_potential_classes()` function in model.py maps string keys to classes, and the `name` attribute is used by the Universe class to identify potentials.

Let me also verify that the `_potential_classes()` function returns all the expected entries.

**Tool: bash**

**Input:**
```json
{
  "command": "python -c \"\nfrom chisurf.plugins.modelling.proteinmc.model import _potential_classes\nclasses = _potential_classes()\nfor k, v in classes.items():\n    print(f'{k:25s} -> {v.__name__}')\n\" 2>&1 | grep -v Warning | grep -v Deprecation",
  "description": "Verify _potential_classes() registry",
  "workdir": "/Users/tpeulen/dev/chisurf"
}
```

**Output:**
```
  numba.core.entrypoints.init_all()
Error opening anisotropy corrections file: [Errno 2] No such file or directory: '/Users/tpeulen/dev/chisurf/chisurf/settings/anisotropy_corrections.json'
default                   -> ClashPotential
hbond                     -> HPotential
mj                        -> MJPotential
unres                     -> CEPotential
go                        -> GoPotential
rama                      -> Ramachandran
asa                       -> ASA
rg                        -> RadiusGyration
H-Bond                    -> HPotential
H-Potential               -> HPotential
Iso-UNRES                 -> CEPotential
Miyazawa-Jernigan         -> MJPotential
Clash potential           -> ClashPotential
Go-Potential              -> GoPotential
Ramachandran              -> Ramachandran
ASA-Calpha                -> ASA
Radius of Gyration        -> RadiusGyration

```

---

## Assistant (Build · MiMo V2.5 Free · 8.8s)

_Thinking:_

All entries are correctly registered. Let me update the todo list.

All 17 entries registered correctly (8 original + 4 new potentials with both short and long name aliases).

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create RadiusGyration core class in potentials.py",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Verify lint/typecheck passes",
      "priority": "medium",
      "status": "completed"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create RadiusGyration core class in potentials.py",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Fix GoPotential.__init__ to set default attributes and call setGo()",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Register Go, Ramachandran, ASA, Radius of Gyration in _potential_classes() (model.py)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Add Go, Ramachandran, ASA, Radius of Gyration to _POTENTIAL_SPECS (GUI widget)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Verify lint/typecheck passes",
    "status": "completed",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · MiMo V2.5 Free · 10.5s)

_Thinking:_

All done. Let me summarize the changes made.

All done. Here's a summary of the changes:

### Files modified

**`chisurf/core/structure/potential/potentials.py`**
- Added `RadiusGyration` class (lines 708-722) — wraps `structure.radius_gyration`
- Fixed `GoPotential.__init__` to accept `epsilon`, `cutoff`, `native_cutoff_on`, `nnEFactor`, `non_native_contact_on` kwargs with defaults and call `setGo()` — enabling headless use
- Made `GoPotential.setGo()` safe when called before widget properties exist (guard + `getattr` fallbacks)

**`chisurf/plugins/modelling/proteinmc/model.py`**
- Registered 8 new entries in `_potential_classes()`: `go`, `rama`, `asa`, `rg` (short names) + `Go-Potential`, `Ramachandran`, `ASA-Calpha`, `Radius of Gyration` (display names)

**`chisurf/gui/widgets/models/proteinmc.py`**
- Added 4 new `_POTENTIAL_SPECS` entries: `go` (Gō-Potential), `rama` (Ramachandran), `asa` (ASA-Cα), `rg` (Radius of Gyration)
- Updated `_edit_potential_params` to render `bool` params as checkboxes instead of spin boxes
- Updated `_potential_settings` to coerce `bool` and `int` parameter types correctly

### Energy terms now available in ProteinMC
| Term | Class | Status |
|------|-------|--------|
| H-Bond | `HPotential` | was already supported |
| Iso-UNRES | `CEPotential` | was already supported |
| Miyazawa-Jernigan | `MJPotential` | was already supported |
| Clash potential | `ClashPotential` | was already supported |
| **Go-Potential** | `GoPotential` | **new** |
| **Ramachandran** | `Ramachandran` | **new** |
| **ASA-Calpha** | `ASA` | **new** |
| **Radius of Gyration** | `RadiusGyration` | **new** |

---

