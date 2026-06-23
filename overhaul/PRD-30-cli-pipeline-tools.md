# PRD-30: CLI Pipeline Tools with Unix Pipe Support

## Goal

Equip every burst/TTTR CLI tool with the ability to **read photon streams from
stdin** and **write results to stdout**, so users can compose ad-hoc processing
pipelines with Unix pipes:

```bash
# Current (file-based only):
csc burst-selection analyze data.ptu -o results.bur

# Desired (pipe-chained):
cat data.ptu | csc burst-select - | csc compute-fret - | csc fit-fret-dist - > results.json

# Mixed:
csc burst-select data.ptu | csc compute-fret - | tee fret.json | csc fit-fret-dist - > dist.json
```

This makes the CLI tools composable building blocks — the Unix philosophy
applied to burst analysis.

## Background

### Current state

The existing CLI tools (`csc burst-selection`, `csc trace-browser`,
`csc bva`, etc.) all take **file paths as arguments** and write results to
**files on disk**. There is no stdin/stdout data streaming:

- Input: `*.ptu`, `*.spc`, `*.bur` files referenced by path
- Output: written to `--output-dir` or specified file paths
- No tool reads from `/dev/stdin` or `-` (the pipe convention)
- No tool writes structured data to stdout

The supported file formats (tttrlib) all have defined binary formats — PTU,
SPC-130, etc. For pipe transport, we need a **serialized photon stream format**
that can be written to stdout and read from stdin.

### The pipe format problem

Binary TTTR formats (PTU, SPC) are **file-position-dependent** — they have
headers at specific offsets, lookup tables, and assume seekable streams. They
cannot be naively piped. A pipe-friendly format is needed.

Options:

| Format | Pros | Cons |
|--------|------|------|
| **msgpack frames** | Compact binary, typed, streamable, already used in MFDB payload codec | Need to define frame schema |
| **JSON lines (JSONL)** | Human-readable, inspectable with `head`, works with `jq` | Larger, slower for high-rate streams |
| **numpy .npy frames** | Compact, zero-copy for array data | Binary, not self-describing |
| **Custom binary frames** | Optimized for photon data | Yet another format |

**Recommendation: msgpack frames with a typed header.** Each frame is a
self-describing packet:
```python
{
    "frame_type": "photon_stream" | "burst_table" | "fret_result" | "eof",
    "schema": "photon_stream.v1",
    "metadata": {"macro_time_resolution": 50.0, ...},
    "data": {  # typed arrays
        "macro_times": [int, ...],
        "micro_times": [int, ...],
        "routing_channels": [int, ...],
        "detectors": [int, ...]
    }
}
```

This is essentially the MFDB payload format (PRD-030) adapted for streaming.

### Relationship to existing infrastructure

| Existing piece | How it maps |
|----------------|-------------|
| **tttrlib** | Reads/writes binary TTTR formats. The pipe-aware CLI wraps tttrlib to serialize/deserialize the pipe format. |
| **MFDB payload codec** (`payload_codec.py`) | Already has msgpack encode/decode for typed payloads — reuse for pipe frames. |
| **PRD-03 result registry** | `burst_table` payload model matches what a pipe would carry. |
| **PRD-16 transformer contract** | Each CLI tool maps to a transformer with typed input/output. |
| **PRD-22 pipeline engine** | A shell pipe is an ad-hoc pipeline; PRD-22's runner is the structured version. |
| **Click CLI framework** | All existing tools use Click — extend with `click.File` for stdin/stdout. |

## Design

### 1. Streamable frame format

Define a **pipe transport format** (`chisurf/core/io/pipe_frames.py`):

```python
@dataclass
class PipeFrame:
    frame_type: str       # "photon_stream", "burst_table", "fret_result", ...
    schema: str           # schema version string
    metadata: dict        # free-form metadata
    data: dict            # column_name -> numpy array (serialized via msgpack)

    def encode(self) -> bytes: ...
    @staticmethod
    def decode(data: bytes) -> "PipeFrame": ...
    def write_to_stream(stream: BinaryIO): ...  # length-prefixed
    @staticmethod
    def read_from_stream(stream: BinaryIO) -> "PipeFrame": ...
```

Frame encoding:
- Length-prefixed: 4-byte big-endian length + msgpack blob
- This allows the reader to frame correctly on a stream
- `write_to_stream(sys.stdout.buffer)` / `read_from_stream(sys.stdin.buffer)`

EOF is signalled by a zero-length frame or a dedicated `{"frame_type": "eof"}` frame.

### 2. Pipe-aware CLI pattern

Each Click command gains a **positional argument `input`** that accepts either a
file path or `-` for stdin:

```python
@click.command()
@click.argument("input", type=click.Path(exists=True, allow_dash=True))
@click.option("--output", "-o", type=click.Path(), default=None)
def burst_select(input, output):
    if input == "-":
        stream = FrameReader(sys.stdin.buffer)
        # read frames until EOF
    else:
        # open file with tttrlib
    # ... process ...
    if output:
        # write to file
    else:
        # write frames to stdout
        writer = FrameWriter(sys.stdout.buffer)
        writer.write(frame)
```

Key conventions:
- `-` means stdin/stdout (standard Unix convention)
- No `--output` means stdout (for pipe consumption)
- `--output file.bur` means write to file
- Stderr for logging/progress (never pollute stdout when piping)

### 3. Command inventory (pipe-enabled)

| Command | Input | Output | Status |
|---------|-------|--------|--------|
| `csc burst-select` | TTTR file or photon stream | burst_table frames | New |
| `csc photon-filter` | TTTR file or photon stream | photon stream frames | New |
| `csc compute-fret` | burst_table frames | fret_result frames | New |
| `csc fit-fret-dist` | fret_result frames | fit_result frames | New |
| `csc fit-decays` | burst_table frames | fit_result frames | New |
| `csc export-csv` | any frame type | CSV to stdout | Wraps existing |
| `csc export-json` | any frame type | JSON to stdout | Wraps existing |
| `csc info` | any frame type | Frame summary to stdout | Utility |

Existing commands that still accept file paths but **gain pipe support**:

| Existing command | New pipe mode |
|-----------------|---------------|
| `csc burst-selection analyze` | `csc burst-select` (lean pipe version) |
| `csc bva compute` | Add `-` input support |
| `csc trace-browser export-csv` | Add stdin frame input |

### 4. Pipeline examples

**Simple FRET pipeline:**
```bash
cat data.ptu | csc burst-select - | csc compute-fret - | tee fret.msgpack | csc fit-fret-dist - > dist.json
```

**Filter then burst-select:**
```bash
csc photon-filter data.ptu --dT-min 10 --dT-max 100 | csc burst-select -
```

**Batch processing with xargs:**
```bash
ls *.ptu | xargs -I{} sh -c 'csc burst-select {} | csc compute-fret - > {}.fret.msgpack'
```

**Inspect intermediate data:**
```bash
csc burst-select data.ptu | head -c 4096 | csc info -   # peek at first frames
csc burst-select data.ptu | csc export-csv - | column -t | head -20
```

**Network pipe (with socat):**
```bash
# Server acquires data, pipes to network
csc acquire | socat - TCP-LISTEN:9999

# Client processes
socat TCP:server:9999 - | csc burst-select - | csc compute-fret -
```

### 5. Implementation strategy

**Phase 1 — Pipe format + core utilities**

- Define `PipeFrame` dataclass, msgpack codec, stream framing
- `FrameReader` / `FrameWriter` classes for stdin/stdout
- Utility functions: `open_input(path_or_dash)` returns a `FrameReader` or
  tttrlib file handle
- Place in `chisurf/core/io/pipe_frames.py`

**Phase 2 — Core pipe commands**

- `csc photon-filter` — read TTTR, apply dT/microtime/routing filters, write
  photon stream frames
- `csc burst-select` — read TTTR or photon stream, run burst selection, write
  burst_table frames
- `csc compute-fret` — read burst_table frames, compute E/S/g-factor, write
  fret_result frames
- `csc info` — read any frame type, print summary to stderr/stdout

**Phase 3 — Analysis pipe commands**

- `csc fit-fret-dist` — read fret_result, fit distribution, write fit_result
- `csc fit-decays` — read burst_table, fit TCSPC decays, write fit_result
- `csc export-csv` — read any frame, write CSV
- `csc export-json` — read any frame, write JSON

**Phase 4 — Integration**

- Retrofit existing CLI tools (burst-selection, BVA, trace-browser) to accept
  `-` for stdin where applicable
- Add pipe format converters: `csc to-pipe *.ptu` converts files to stream
- Test the full pipeline: `cat file.ptu | csc burst-select - | csc compute-fret -`

## Non-goals

- **Real-time acquisition pipes** — the pipe format supports it in theory, but
  this PRD focuses on file-based pipelines. Real-time is future work.
- **Replacing the node editor** — pipes are the CLI complement to the visual
  pipeline (PRD-29).
- **Windows support** — Unix pipes are the primary target. Windows named pipes
  could be added later.
- **Binary compatibility with ttttrlib formats over pipes** — tttrlib formats
  are not streamable; the msgpack frame format is the pipe format.
- **Compression** — users can compress with `gzip`/`lz4` externally:
  `csc burst-select data.ptu | lz4 | ssh server "lz4 -d | csc compute-fret -"`

## Definition of Done

- [ ] `PipeFrame` dataclass + msgpack codec + stream framing implemented
- [ ] `FrameReader` / `FrameWriter` tested with stdin/stdout pipes
- [ ] `csc photon-filter` reads `.ptu` or `-`, writes photon stream frames to stdout
- [ ] `csc burst-select` reads `.ptu`, photon frames, or `-`, writes burst_table frames
- [ ] `csc compute-fret` reads burst_table frames, writes fret_result frames
- [ ] `csc info` reads any frame type, prints human-readable summary
- [ ] Full pipeline test: `cat file.ptu | csc burst-select - | csc compute-fret - > result.msgpack`
- [ ] All pipe commands accept `-` for stdin input
- [ ] No output argument → stdout; `--output file` → file
- [ ] Stderr used exclusively for logging/progress (stdout is clean data)

## Definition of Clean

- Zero-copy where possible: numpy arrays serialized via msgpack's `ext` type
- Frame format is self-describing (no out-of-band schema negotiation)
- Each command is a pure function of its input streams (no hidden state)
- `FrameReader`/`FrameWriter` tested with actual pipe subprocesses
- Existing file-only CLIs remain unchanged — `-` is an additive option
- Pipe format documented as a spec (frame types, schemas, example bytes)

## Appendix: Frame type schemas

### `photon_stream.v1`

```python
{
    "frame_type": "photon_stream",
    "schema": "photon_stream.v1",
    "metadata": {
        "macro_time_resolution": float,   # ns
        "micro_time_resolution": float,   # ps
        "micro_time_binning": int,
        "file_type": str,                 # "PTU", "SPC-130", etc.
        "n_photons": int,
        "source": str                     # original filename or description
    },
    "data": {
        "macro_times": np.ndarray[int64],
        "micro_times": np.ndarray[int64],
        "routing_channels": np.ndarray[uint8] | None,
        "detectors": np.ndarray[uint8] | None,
        "pulse_counts": np.ndarray[uint8] | None
    }
}
```

### `burst_table.v1`

```python
{
    "frame_type": "burst_table",
    "schema": "burst_table.v1",
    "metadata": {
        "setup_name": str | None,
        "burst_selection_parameters": dict,
        "n_bursts": int
    },
    "data": {
        "start": np.ndarray[int64],
        "stop": np.ndarray[int64],
        "duration": np.ndarray[float64],
        "n_photons_donor": np.ndarray[int64] | None,
        "n_photons_acceptor": np.ndarray[int64] | None,
        "E": np.ndarray[float64] | None,
        "S": np.ndarray[float64] | None,
        # ... additional columns
    }
}
```

### `fret_result.v1`, `fit_result.v1`, `distance_result.v1`

Follow the same pattern — typed metadata + typed columnar data, corresponding
to the existing MFDB payload models.
