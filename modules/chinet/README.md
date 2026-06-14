# chinet

chinet is the pure-Python graph and parameter runtime used by ChiSurf for
reactive model evaluation. It provides `Port`, `Node`, `Session`, and
`BaseObject` classes for building parameter dependency graphs, evaluating node
callbacks, and persisting sessions through JSONL or MFDB-backed adapters.

## Runtime API

```python
import chinet as cn

source = cn.Node(name="source")
source.add_output_port("out", cn.Port(1.25, is_output=True, name="out"))

follower = cn.Node(name="follower")
inp = cn.Port(0.0, is_bounded=True, lb=0.1, ub=10.0, name="in")
follower.add_input_port("in", inp)
inp.link = source.outputs["out"]

session = cn.Session({"source": source, "follower": follower})
session.update()
```

## Persistence

Local session persistence remains JSONL-compatible:

```python
session.save("session.jsonl")
restored = cn.Session.load("session.jsonl")
```

ChiSurf can also configure transparent MFDB persistence:

```python
session.connect_to_db(
    "mfdb",
    db_path="chisurf.mfdb.sqlite",
    operation_id="fit-operation-id",
    experiment_id="experiment-id",
    store_node_artifacts=False,
)
session.write_to_db()
```

MFDB storage writes one embedded `chinet_session` artifact by default, optional
per-node `chinet_node` artifacts, queryable `mfdb_parameter` rows, and
`parameter_depends_on` edges for port links.

## License

chinet is released under the open source MIT license.
