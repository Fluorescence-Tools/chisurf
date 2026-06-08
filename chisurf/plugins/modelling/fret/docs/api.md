# FRET Modeling FastAPI Web API Reference

The FRET modeling plugin provides a FastAPI web service in `api/router.py`. It can be run standalone or included in larger web applications.

## Launching the Web Service
Start the API locally using `uvicorn`:
```bash
uvicorn chisurf.plugins.modelling.fret.api:app --host 127.0.0.1 --port 8000 --reload
```
You can view the interactive documentation at `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /fret/info-backends`
Get information about available and active AV backends.
- **Response**: `BackendInfoResponse`

### `POST /fret/info`
Parse and summarize an `fps.json` file.
- **Request Body**: `InfoRequest`
- **Response**: `InfoResponse`

### `POST /fret/dock`
Run rigid-body docking.
- **Request Body**: `DockRequest`

### `POST /fret/refine`
Run iterative structure refinement.
- **Request Body**: `RefineRequest`

### `POST /fret/bootstrap`
Run parametric bootstrap error analysis.
- **Request Body**: `BootstrapRequest`

### `POST /fret/sample`
Run Metropolis MC trajectory sampling.
- **Request Body**: `SampleRequest`

### `POST /fret/screen`
Screen structure library files.
- **Request Body**: `ScreenRequest`

### `POST /fret/evaluate`
Evaluate structure/trajectory parameters using OLGA evaluators.
- **Request Body**: `EvaluateRequest`

### `POST /fret/select-pairs`
Select informative FRET pairs from ensemble structure files.
- **Request Body**: `PairSelectRequest`
