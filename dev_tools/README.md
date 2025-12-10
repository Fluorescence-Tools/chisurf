# Developer tools for parameter registry

This directory contains small, maintenance-oriented scripts that help keep
`chisurf`'s global **fitting-parameter registry**
(`chisurf/settings/constants/fitting_parameters.json`) in sync with the
codebase and with the FCS/TCSPC model definitions.

These tools are **for developers** and are not part of the public API.
They are typically run from the repository root using the Python module
interface (e.g. `python -m dev_tools.export_fitting_parameters`).

---

## Scripts

- **`export_fitting_parameters.py`**  
  Scans the `chisurf` package (and optionally `test/`) for
  `FittingParameter(...)` calls using the Python AST, and (re)builds the
  central JSON registry of fitting-parameter metadata.

  - Collects parameter name, location (module/file/line, class, function),
    and selected keyword arguments (`description`, `label_text`, `fixed`,
    `bounds_on`, `lb`, `ub`).
  - Merges results into `fitting_parameters.json`, preserving any existing
    descriptions, keywords, aliases, and label texts.
  - Adds/updates a `sources` list per parameter with references to where
    that parameter is used in the code.

  **Typical usage:**

  ```bash
  # From the repository root
  python -m dev_tools.export_fitting_parameters \
      --root chisurf \
      --output chisurf/settings/constants/fitting_parameters.json

  # Optionally include tests in the scan
  python -m dev_tools.export_fitting_parameters --include-tests
  ```

- **`export_fcs_parameters.py`**  
  Parses the FCS model definitions from `models/fcs/models.yaml` and injects
  the discovered parameter names into the registry, under the `fcs.*`
  namespace.

  - Reads the FCS models YAML (`models/fcs/models.yaml` by default).
  - Builds a mapping `parameter -> set of model names` based on the
  `initial` sections.
  - For each parameter `p`, ensures there is a registry entry `fcs.p` with
  basic metadata and a FCS-specific `sources` entry that lists the models
  using it and the origin YAML path.
  - Preserves existing descriptions, keywords, aliases, and label texts.
  - The YAML file is the single source of truth for parse-based FCS models;
    this tool keeps the central parameter registry synchronized with those
    model definitions so that GUIs and other tooling know which parameters
    belong to which FCS models.

  **Typical usage:**

  ```bash
  python -m dev_tools.export_fcs_parameters \
      --root chisurf \
      --yaml chisurf/models/fcs/models.yaml \
      --output chisurf/settings/constants/fitting_parameters.json
  ```

- **`export_tcspc_parameters.py`**  
  Similar to `export_fcs_parameters.py`, but for parse-based TCSPC decay
  models defined in `models/tcspc/tcspc.models.json`. Parameters are
  registered under the `tcspc.*` namespace.

  - Reads `tcspc.models.json` by default.
  - Builds a mapping `parameter -> set of TCSPC model names` from the
  `initial` sections.
  - Ensures registry entries `tcspc.p` exist and records a TCSPC-specific
  `sources` entry listing models and the JSON origin path.
  - Preserves existing user-provided metadata where present.
  - The JSON file is the single source of truth for parse-based TCSPC decay
    models; running this tool ensures that the parameter registry reflects
    the parameters actually used by those models, so the GUI and fit engine
    can present consistent help text and groupings.

  **Typical usage:**

  ```bash
  python -m dev_tools.export_tcspc_parameters \
      --root chisurf \
      --json chisurf/models/tcspc/tcspc.models.json \
      --output chisurf/settings/constants/fitting_parameters.json
  ```

- **`fill_fcs_descriptions.py`**  
  Fills in **missing** descriptions and keywords for FCS-related entries in
  the registry (`fcs.*` keys) using simple, FCS-aware heuristics based on
  the parameter symbol (e.g. `N`, `td1`, `a`, `bt1`, etc.). Existing
  non-empty descriptions are left untouched.

  - Walks all `fcs.*` entries in `fitting_parameters.json`.
  - If `description` is empty/missing, generates a generic but physically
    meaningful text and merges FCS-related keywords.
  - Never overwrites user-provided descriptions.

  **Typical usage:**

  ```bash
  python -m dev_tools.fill_fcs_descriptions \
      --root chisurf \
      --output chisurf/settings/constants/fitting_parameters.json
  ```

- **`fill_tcspc_descriptions.py`**  
  Fills in **missing** descriptions and keywords for TCSPC-related
  parameters in the registry.

  This covers both:

  - `tcspc.*` keys introduced by `export_tcspc_parameters.py`, and
  - plain parameter names whose `sources` live in `chisurf.models.tcspc.*`
    modules.

  For each such parameter, a generic TCSPC-/lifetime-aware description is
  generated (or, for many common symbols, a more specific physical
  description), and TCSPC-related keywords are merged. Existing
  non-empty descriptions are preserved.

  **Typical usage:**

  ```bash
  python -m dev_tools.fill_tcspc_descriptions \
      --root chisurf \
      --output chisurf/settings/constants/fitting_parameters.json
  ```

---

## Typical workflow

When evolving models or adding new fitting parameters, a typical maintenance
sequence is:

1. **Refresh the base registry from code:**
   - **Run:** `python -m dev_tools.export_fitting_parameters`.
2. **Add/refresh FCS and TCSPC model parameters:**
   - **Run:** `python -m dev_tools.export_fcs_parameters`.
   - **Run:** `python -m dev_tools.export_tcspc_parameters`.
3. **Backfill missing descriptions:**
   - **Run:** `python -m dev_tools.fill_fcs_descriptions`.
   - **Run:** `python -m dev_tools.fill_tcspc_descriptions`.

All commands can be pointed at alternate roots or output paths via their
respective CLI options if needed for experiments or CI scripts.
