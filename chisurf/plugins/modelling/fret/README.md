# FRET Modeling Plugin

A comprehensive FRET-restrained modeling, rigid-body docking, structure screening, and trajectory evaluation suite for ChiSurf. This plugin is a unified successor to the C# application **FPS** and the C++ application **OLGA**.

## Submodule Architecture

The plugin is designed with a strict separation of interface layers:
- **GUI (`gui/wizard.py`)**: Qt-based GUI containing parameter editors, docking controls, results analysis, and 3D visualization.
- **CLI (`cli/main.py`, `__main__.py`)**: Powerful `click`-based command-line interface.
- **API (`api/router.py`)**: Web-ready `fastapi` router and standalone app.

## Quick Start

### Running the GUI Wizard
Launch the wizard directly within the ChiSurf application or run it standalone:
```bash
python -m chisurf.plugins.modelling.fret.gui.wizard
```

### Running the CLI
Run the Click CLI for various tasks:
```bash
# Print info about active/available backends
python -m chisurf.plugins.modelling.fret info-backends

# Print summary of labeling parameters
python -m chisurf.plugins.modelling.fret info --fps examples/fps_4wj/4w_junction.fps.json
```

### Running the FastAPI Web API
Deploy or test the API standalone using `uvicorn`:
```bash
uvicorn chisurf.plugins.modelling.fret.api:app --reload
```

## Documentation

Detailed guides are available in the [docs](./docs) folder:
- [Command Line Interface Documentation](./docs/cli.md)
- [FastAPI Web API Documentation](./docs/api.md)
