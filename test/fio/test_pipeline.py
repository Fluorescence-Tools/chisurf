"""PRD-22: the workflow / pipeline engine on the transformer contract.

A pipeline composes conformant transformers (PRD-16) into a type-checked DAG that a
headless runner executes as a recorded, queryable chain of operations (PRD-11/21).
The canonical pipeline is ``raw -> microtime_shift -> burst_selection``.

Execution is exercised through lightweight replay executors registered for the two
operation types (the same seam the real plugin executors use), so the test stays
hermetic and fast while genuinely recording the operation chain; the real
transformer executors are covered by the plugin ``test_replay`` suites.
"""

from __future__ import annotations

import os

import pytest

from chisurf.core.mfdb.provenance.compute_spec import (
    get_replay_executor,
    register_replay_executor,
    unregister_replay_executor,
)
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.provenance.result_registry import (
    register_raw_measurement,
    register_result,
    set_global_db,
)
from chisurf.core.pipeline import (
    Pipeline,
    PipelineEdge,
    PipelineNode,
    PipelineValidationError,
    get_pipeline,
    get_pipeline_run,
    list_pipelines,
    record_pipeline_run,
    run_pipeline,
    save_pipeline,
    topological_order,
    validate_pipeline,
)

# Importing the transformer modules self-registers both conformant transformers so
# the pipeline validator can resolve operation_type -> declared ports.
from chisurf.plugins.burst.burst_selection.api.transformer import (  # noqa: F401
    BurstSelectionTransformer,
)
from chisurf.plugins.tttr.tttr_microtime_shifter.api.transformer import (  # noqa: F401
    MicrotimeShifterTransformer,
)


def _make_executor(kind: str):
    """A replay executor that registers one derived artifact from the first source.

    Stands in for the real plugin executor: it records a real operation + output
    artifact (so the chain is genuine and queryable) without touching TTTR files.
    """

    def _exec(spec, database) -> str:
        parent = spec.source_artifact_ids[0]
        # metadata-only artifact: the placeholder payload is irrelevant here — the
        # point is the recorded operation + derived_from chain the runner produces.
        return register_result(
            kind=kind,
            data=None,
            parent_artifact_id=parent,
            operation_type=spec.operation_type,
            parameters=spec.parameters,
            db=database,
        )

    return _exec


@pytest.fixture
def fake_executors():
    """Register stand-in executors for both operation types; restore on teardown."""
    prior = {
        op: get_replay_executor(op) for op in ("microtime_shift", "burst_selection")
    }
    register_replay_executor("microtime_shift", _make_executor("processed_data"))
    register_replay_executor("burst_selection", _make_executor("burst_table"))
    try:
        yield
    finally:
        for op, fn in prior.items():
            if fn is None:
                unregister_replay_executor(op)
            else:
                register_replay_executor(op, fn)


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "pipeline.db"))
    try:
        yield database
    finally:
        set_global_db(None)
        database.close()


def _canonical_pipeline() -> Pipeline:
    return Pipeline(
        name="raw->shift->burst",
        nodes=(
            PipelineNode("shift", "microtime_shift", {"global_shift": 5}),
            PipelineNode("burst", "burst_selection", {}),
        ),
        edges=(PipelineEdge("shift", "shifted", "burst", "raw"),),
    )


# -- composition / type-checking (pure, no DB) -------------------------------


def test_canonical_pipeline_validates():
    # raw -> microtime_shift(out: shifted/processed_data) -> burst(in: raw) is valid
    # because burst's input port accepts processed_data (a shifted TTTR file).
    validate_pipeline(_canonical_pipeline())


def test_invalid_edge_kind_mismatch_is_rejected():
    # burst produces burst_table; microtime_shift's input accepts only TTTR kinds,
    # so wiring burst -> shift is a kind mismatch rejected at definition time.
    bad = Pipeline(
        name="bad",
        nodes=(
            PipelineNode("burst", "burst_selection", {}),
            PipelineNode("shift", "microtime_shift", {"global_shift": 1}),
        ),
        edges=(PipelineEdge("burst", "burst_table", "shift", "raw"),),
    )
    with pytest.raises(PipelineValidationError, match="incompatible edge"):
        validate_pipeline(bad)


def test_unknown_operation_type_is_rejected():
    p = Pipeline(name="x", nodes=(PipelineNode("n", "does_not_exist", {}),))
    with pytest.raises(PipelineValidationError, match="no registered transformer"):
        validate_pipeline(p)


def test_unknown_port_is_rejected():
    p = Pipeline(
        name="x",
        nodes=(
            PipelineNode("shift", "microtime_shift", {}),
            PipelineNode("burst", "burst_selection", {}),
        ),
        edges=(PipelineEdge("shift", "no_such_port", "burst", "raw"),),
    )
    with pytest.raises(PipelineValidationError, match="no output port"):
        validate_pipeline(p)


def test_cycle_is_rejected():
    # topological_order is the cycle detector; exercise it directly (a type-compatible
    # cycle can't be built from these transformers, and type-checks fire first).
    p = Pipeline(
        name="cyc",
        nodes=(
            PipelineNode("a", "microtime_shift", {}),
            PipelineNode("b", "microtime_shift", {}),
        ),
        edges=(
            PipelineEdge("a", "shifted", "b", "raw"),
            PipelineEdge("b", "shifted", "a", "raw"),
        ),
    )
    with pytest.raises(PipelineValidationError, match="cycle"):
        topological_order(p)


def test_topological_order_respects_edges():
    order = [n.name for n in topological_order(_canonical_pipeline())]
    assert order == ["shift", "burst"]


# -- execution (records a queryable operation chain) -------------------------


def test_run_pipeline_records_a_queryable_chain(db, fake_executors, tmp_path):
    f = os.path.join(tmp_path, "m.ptu")
    with open(f, "wb") as fh:
        fh.write(b"\x00\x01\x02")
    raw = register_raw_measurement(f, db=db)

    run = run_pipeline(_canonical_pipeline(), inputs={"shift": [raw]}, db=db)

    shifted = run.outputs("shift")[0]
    burst = run.outputs("burst")[0]
    assert shifted and burst and shifted != burst
    # two operations recorded, in execution order
    assert len(run.operation_ids) == 2

    # the chain is queryable via the lineage API: both derived artifacts are
    # descendants of the raw measurement, and burst descends from shifted.
    assert set(db.lineage.descendants(raw)) == {shifted, burst}
    assert db.lineage.descendants(shifted) == [burst]
    assert db.lineage.ancestors(burst) == [shifted, raw]


# -- persistence (saveable, shareable document + grouped runs) ---------------


def test_save_and_get_pipeline_round_trips(db):
    original = _canonical_pipeline()
    pid = save_pipeline(db, original, description="canonical demo")

    loaded = get_pipeline(db, pid)
    assert loaded is not None
    assert loaded.name == original.name
    assert [n.name for n in loaded.nodes] == ["shift", "burst"]
    assert loaded.node("shift").operation_type == "microtime_shift"
    assert loaded.node("shift").parameters == {"global_shift": 5}
    assert len(loaded.edges) == 1
    assert loaded.edges[0].source == "shift" and loaded.edges[0].target == "burst"
    # a stored definition is itself a valid composition
    validate_pipeline(loaded)
    assert any(p["pipeline_id"] == pid for p in list_pipelines(db, scope="all"))


def test_save_pipeline_replaces_prior_graph(db):
    pid = save_pipeline(db, _canonical_pipeline())
    smaller = Pipeline(
        name="raw->shift->burst",
        nodes=(PipelineNode("shift", "microtime_shift", {"global_shift": 1}),),
    )
    save_pipeline(db, smaller, pipeline_id=pid)
    loaded = get_pipeline(db, pid)
    assert [n.name for n in loaded.nodes] == ["shift"]
    assert loaded.edges == ()


def test_record_and_get_pipeline_run_groups_the_chain(db, fake_executors, tmp_path):
    f = os.path.join(tmp_path, "m3.ptu")
    with open(f, "wb") as fh:
        fh.write(b"\x00\x01\x02")
    raw = register_raw_measurement(f, db=db)

    pid = save_pipeline(db, _canonical_pipeline())
    run = run_pipeline(_canonical_pipeline(), inputs={"shift": [raw]}, db=db)
    run_id = record_pipeline_run(db, run, pipeline_id=pid)

    assert run.pipeline_run_id == run_id
    stored = get_pipeline_run(db, run_id)
    assert stored["pipeline_id"] == pid
    assert stored["status"] == "succeeded"
    # the run groups the recorded operation chain, in execution order
    assert stored["operation_ids"] == run.operation_ids
    assert len(stored["operation_ids"]) == 2


def test_run_pipeline_missing_executor_raises(db, tmp_path):
    # no executors registered for this fixture -> fail loud at the first node
    from chisurf.core.mfdb.provenance.compute_spec import NoReplayExecutorError

    f = os.path.join(tmp_path, "m2.ptu")
    with open(f, "wb") as fh:
        fh.write(b"\x00")
    raw = register_raw_measurement(f, db=db)
    prior = get_replay_executor("microtime_shift")
    unregister_replay_executor("microtime_shift")
    try:
        with pytest.raises(NoReplayExecutorError):
            run_pipeline(_canonical_pipeline(), inputs={"shift": [raw]}, db=db)
    finally:
        if prior is not None:
            register_replay_executor("microtime_shift", prior)
