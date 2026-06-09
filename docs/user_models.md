# User-Defined Models in ChiSurf

This document explains how to add your own models to ChiSurf **without modifying the installed package**.

User models are loaded from your **user settings folder** and plugged into the normal experiment/model selection logic, so they appear in the GUI like built-in models.

---

## 1. Location of user model files

ChiSurf looks for user models in the folder:

- **Windows / Linux / macOS**: `~/.chisurf/models`

where `~` is your home directory.

Each `*.py` file in this folder (except those starting with `_`) is imported once when experiments are initialized.

> You do **not** need to add this folder to `PYTHONPATH` – ChiSurf will import the files directly.

---

## 2. How user models are wired into experiments

Internally, user models use a small registry in `chisurf.models`:

- `chisurf.models.register_user_model(experiments)` – decorator for user model classes.
- `chisurf.models.load_user_models()` – imports all user model modules from `~/.chisurf/models`.
- `chisurf.models.iter_user_models_for_experiment(exp_type, experiment_name)` – returns registered user models for a given experiment.

When the GUI starts and runs `Main.init_setups()` / `_setup_experiment(...)`:

1. The standard models for each experiment are added from `experiment_configs.yaml`.
2. `chisurf.models.load_user_models()` is called.
3. Any user models registered for that experiment are appended to the experiment’s `model_classes`.

As a result:

- They are visible in `Experiment.model_names`.
- They show up in the **Model** combobox in the main GUI for the corresponding experiments.

---

## 3. Mapping between experiment labels and your models

You attach a user model to one or more experiments using the `experiments` argument of `register_user_model`.

You can use:

- The **experiment type key** from `experiment_configs.yaml` (e.g. `"tcspc"`, `"fcs"`, `"pda"`).
- The **display name** from `experiment_types` (e.g. `"TCSPC"`, `"FCS"`, `"PDA"`).

Both are matched in a **case-insensitive** way.

Examples of valid labels:

- TCSPC experiment:
  - type key: `"tcspc"`
  - display name: `"TCSPC"`
- FCS experiment:
  - type key: `"fcs"`
  - display name: `"FCS"`

If you are unsure which keys are available, check `chisurf/settings/experiment_configs.yaml` under `experiment_types:` and the sections `tcspc:`, `fcs:`, `pda:`, etc.

---

## 4. Writing a simple user model

### 4.1. Minimal template

Create a file, for example:

```text
~/.chisurf/models/my_tcspc_models.py
```

Inside that file, you can define a model like this:

```python
import chisurf
import chisurf.models


@chisurf.models.register_user_model(["tcspc", "TCSPC"])
class MySimpleTCSPCModel(chisurf.models.ModelWidget):
    """Example user-defined TCSPC model."""

    # Name shown in the GUI model dropdown
    name = "MySimpleTCSPCModel"

    def __init__(self, fit: "chisurf.fitting.fit.FitGroup", *args, **kwargs):
        super().__init__(fit=fit, *args, **kwargs)
        # TODO: add parameters here (using chisurf.fitting.parameter.FittingParameter)

    def update_model(self, **kwargs):
        """Compute model curve based on current parameters and fit.data.

        This method must update self.y (and possibly self.x, ey, etc.) to
        represent the model evaluated at the current parameters.
        """
        # Example: identity model (copies data)
        data = self.fit.data
        self.x = data.x.copy()
        self.y = data.y.copy()
        self.ey = data.ey.copy() if getattr(data, "ey", None) is not None else None

    def update_widgets(self) -> None:
        """Refresh parameter widgets after parameter changes.

        If you add custom Qt widgets for parameters, update them here.
        """
        for p in self.parameters:
            try:
                p.update()
            except Exception:
                pass
```

Notes:

- Inherit from `chisurf.models.Model` for non-GUI models, or `chisurf.models.ModelWidget` if you want the standard plot integration and widget behavior.
- The `name` attribute is used by the GUI; choose something descriptive but short.
- `update_model` is where you implement the actual model calculation.
- `update_widgets` is responsible for keeping any parameter widgets in sync.

### 4.2. Attaching the same model to multiple experiments

You can attach a model to multiple experiment types by listing more labels:

```python
@chisurf.models.register_user_model(["tcspc", "TCSPC", "fcs", "FCS"])
class SharedModel(chisurf.models.ModelWidget):
    name = "SharedModel"
    # implementation as above
```

This model will appear in the model list for both TCSPC and FCS experiments.

---

## 5. Error handling and logging

- If a user model file raises an exception when imported, ChiSurf will log a warning and skip that file.
- A broken user model **does not** prevent built-in experiments or models from loading.
- Check the ChiSurf log file in your `~/.chisurf/logs` folder if your user model does not appear.

---

## 6. Quick checklist

- [ ] Create `~/.chisurf/models` if it does not exist.
- [ ] Add a `*.py` file there (e.g. `my_models.py`).
- [ ] Import `chisurf.models` and subclass `chisurf.models.Model` or `chisurf.models.ModelWidget`.
- [ ] Decorate your class with `@chisurf.models.register_user_model([...])` using appropriate experiment labels.
- [ ] Set a `name` attribute on your class.
- [ ] Implement at least `update_model` (and `update_widgets` for widget-based models).
- [ ] Restart ChiSurf and select the corresponding experiment; your model should appear in the model dropdown.

---

## 7. In-Place Model Code Editing (Front Face / Back Face)

ChiSurf provides a powerful feature to view and edit the source code of models directly from the GUI. This allows you to rapidly prototype, refine, and modify mathematical models and fitting logic without leaving the application.

### 7.1 Using the Code View
Inside any **Fit Window**, you will see a toggle button labeled **View Model Code** above the standard plots (the "Front Face"). Clicking this button flips the view to the "Back Face", revealing a fully functional Python code editor populated with the source code of the active model.

### 7.2 Saving and Applying Changes
Once you have modified the code in the editor, you can click **Save and Apply Model**. ChiSurf handles your changes automatically based on your permissions:
1. **Writable Installations**: If you have write access to the original source file (e.g., if you installed ChiSurf in editable mode `pip install -e .`), ChiSurf will save your changes directly to the original file.
2. **Versioned User Overrides**: Regardless of write access, ChiSurf will automatically save a backup of your overridden code to your user directory at `~/.chisurf/models/`. The filename will include the fully qualified module name and a datetime stamp (e.g., `chisurf.core.models.tcspc.fret__override__20260609_120000.py`).

### 7.3 Instant Application and Startup Injection
When you click "Save and Apply", the new code is dynamically injected into the running Python session, immediately updating the logic of your active fit.
Additionally, ChiSurf includes a **Startup Hook**. Every time you launch the application, it scans the `~/.chisurf/models/` directory for these `__override__` files. For each module, it automatically selects the latest version (based on the datetime stamp) and injects it over the built-in models. This ensures your modifications persist across sessions!

### 7.4 Reverting Changes
If you wish to revert to the factory default or an older version of your model:
- Navigate to your `~/.chisurf/models/` folder.
- Delete or rename the corresponding `__override__` files.
- Restart ChiSurf to load the default built-in models.
