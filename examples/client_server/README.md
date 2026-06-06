# ChiSurf Client-Server Architecture

ChiSurf uses a ZMQ/JSON-RPC 2.0 client-server architecture.

## Server

The server owns the runtime state (datasets, fits, parameters) and
exposes them via JSON-RPC over ZeroMQ.

Start the server in a terminal:

```bash
# Default ports (8765 cmd, 8766 pub, localhost):
python -m chisurf.server

# Custom ports:
python -m chisurf.server --cmd-port 18765 --pub-port 18766

# With debug logging:
python -m chisurf.server --log-level DEBUG
```

## Client

Connect from another process or even another machine:

```python
from chisurf.client import ChisurfClient

client = ChisurfClient(host="127.0.0.1", cmd_port=8765, pub_port=8766)

# Check server is alive
print(client.ping())

# List available methods
print(client.list_methods())

# List datasets
print(client.list_datasets())
```

## GUI

When the GUI starts (`python -m chisurf` or the `chisurf` launcher),
it automatically:

1. Finds two free TCP ports
2. Spawns ``python -m chisurf.server`` as a subprocess
3. Connects via ``ChisurfClient``
4. Installs proxy objects (``chisurf.fits``, ``chisurf.imported_datasets``)
   that delegate to the server

## Architecture

```
┌─────────────────────┐        ZMQ REQ/REP (JSON-RPC 2.0)
│  GUI / Client       │ ◄──────────────────────────────────► ┌──────────────────────┐
│                     │                                       │  Server Process       │
│  ChisurfClient      │                                       │  (python -m chisurf   │
│  ProxyFitList       │                                       │   .server)            │
│  ProxyDatasetList   │                                       │                       │
│                     │                                       │  SessionState         │
│  Proxy chisurf      │                                       │  ServiceDispatcher    │
│  module namespace   │                                       │  ZmqServer            │
└─────────────────────┘                                       └──────────────────────┘
      (one process)                      ZMQ PUB/SUB events       (separate process)
```

## Multiple Instances

Each GUI instance spawns its own private server on dynamically
assigned ports, so multiple instances do not interfere.
