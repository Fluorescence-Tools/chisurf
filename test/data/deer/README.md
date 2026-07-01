# DEER example data

Small experimental DEER/PELDOR traces used by the ChiSurf DEER reader/model
tests (`test/experiments/test_deer_reader.py`,
`test/gui/test_deer_model_editor.py`).

## Source / attribution

These files are copied from the **DeerAnalysis** test suite
(`thirdparty/DeerAnalysis/tests/data`, © 2026 Hugo Karas, MIT License; Jeschke
lab, ETH Zürich) and are redistributed here under that MIT License solely as
test fixtures.

| file | source | format | notes |
|------|--------|--------|-------|
| `deer_ringtest_4pdeer.DSC/.DTA` | `benchmark/data_multi_lab1` | Bruker BES3T | complex 4-pulse DEER ring-test trace (0–2.83 µs) |
| `deer_twostate.DSC/.DTA` | `population/example_twostate_data_1` | Bruker BES3T | complex two-state distance sample (0–4.70 µs) |
| `deer_trace.csv` | `test.csv` | ASCII | headerless two-column trace, time axis in ns |

Bruker BES3T files are loaded by the pure-numpy `eprload` reader; the CSV by
`csv_loader` (nanoseconds are auto-detected from the time span when the header
states no unit).
