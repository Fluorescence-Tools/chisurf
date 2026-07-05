# Subsystems

* [Core](core.md) - Domain objects, fitting, data, models, math, settings, actions, and the API facade.
* [Data model](data-model.md) - `Base`/`Data`/`DataCurve`, data groups, experiment readers, and dataset flow.
* [Data IO](data-io.md) - File loading, TTTR/photon readers, format registry, and slow-storage staging.
* [Fluorescence domain](fluorescence-domain.md) - Shared fluorescence math and algorithms used by models and plugins.
* [Fitting engine](fitting.md) - Fit/FitGroup, weighted residuals, global analysis, error analysis, and sampling.
* [Fitting models](models.md) - TCSPC/FCS/PDA/PCH/DEER/RICS/structure models and data-described editors.
* [Parameters](parameters.md) - Scalar parameters, bounds, links, dependency graph, and fit degrees of freedom.
* [GUI & AutoForm](gui-autoform.md) - The Qt application and the data-driven AutoForm UI framework.
* [Operation history](history.md) - Append-only action history, headless replay, and MFDB event-log projection.
* [Macros, CLI & scripting](macros-cli.md) - Macros, `csc`, GUI scripts, and the recording QtConsole.
* [Project persistence](project-persistence.md) - `.csp` archive format, UID-keyed project state, and UI-state capture.
* [Pipelines](pipeline.md) - Typed DAGs of transformer invocations persisted and replayed through MFDB provenance.
* [Compiled Modules](compiled-modules.md) - The C++ extensions in `modules/` that must be built before tests.
