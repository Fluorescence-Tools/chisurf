import json
import pathlib
import tempfile

import numpy as np

from chisurf.plugins.chat.index_docs import embed_texts, retrieve, EMBED_DIM
from chisurf.plugins.chat.agent.tools import get_tool, validate_args, execute_tool
from chisurf.plugins.chat.agent.loop import AgentLoop
from chisurf.plugins.chat.tools.anisotropy import run_anisotropy_analysis


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DATA = PROJECT_ROOT / 'test' / 'data'
SAMPLE_CSV = TEST_DATA / 'sample_anisotropy.csv'


def make_minimal_store(tmp: pathlib.Path):
    # Create a tiny RAG store with two chunks
    texts = [
        {"path": str((tmp / 'README.test.md').resolve()), "text": "G-factor controls polarization correction. Set G close to 1."},
        {"path": str(SAMPLE_CSV.resolve()), "text": "CSV columns: time, I_par, I_perp."},
    ]
    embs = embed_texts([t["text"] for t in texts])
    assert embs.shape == (2, EMBED_DIM)
    (tmp / 'embeddings.npy').parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp / 'embeddings.npy', embs)
    (tmp / 'texts.jsonl').write_text("\n".join(json.dumps(t) for t in texts), encoding='utf-8')
    meta = {"count": 2, "dim": EMBED_DIM, "paths": [t['path'] for t in texts], "store_type": "numpy"}
    (tmp / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


def test_rag_retrieval_formatting():
    with tempfile.TemporaryDirectory() as td:
        store = pathlib.Path(td)
        make_minimal_store(store)
        ctx = retrieve("How do I set G-factor?", top_k=2, store_dir=store)
        # Expect bracketed paths and --- separators
        assert "---" in ctx
        assert ctx.count("[") >= 2
        assert "G-factor" in ctx or "G" in ctx


def test_schema_validation_rejects_missing_required():
    tool = get_tool('run_anisotropy_analysis')
    assert tool is not None
    args, err = validate_args(tool, {})
    assert err is not None and 'Missing required' in err


def test_anisotropy_analysis_outputs():
    assert SAMPLE_CSV.is_file()
    with tempfile.TemporaryDirectory() as td:
        out_prefix = str(pathlib.Path(td) / 'anisotropy_out')
        res = run_anisotropy_analysis(
            csv_path=str(SAMPLE_CSV.resolve()),
            g_factor=1.0,
            time_col='time',
            Ipar_col='I_par',
            Iperp_col='I_perp',
            smooth_window=3,
            out_prefix=out_prefix,
        )
        assert isinstance(res, dict)
        assert pathlib.Path(res['out_csv']).is_file()
        assert pathlib.Path(res['out_png']).is_file()
        assert res['n_points'] > 0
        assert -1.0 <= res['r_mean'] <= 1.0


def test_agent_loop_confirmation_flow():
    # Step 1: list files in test/data (non-destructive)
    # Step 2: attempt batch_run_anisotropy (destructive) -> requires confirmation -> loop stops
    seq = []
    def llm_respond(messages):
        # Count steps by how many times we've been called
        idx = len(seq)
        if idx == 0:
            # Emit list_files toolcall
            payload = {
                "tool": "list_files",
                "args": {"dir_path": str(TEST_DATA.resolve()), "pattern": "*.csv", "max_items": 50},
            }
            seq.append('list')
            return "Plan: list files then run batch.\n\n```toolcall\n" + json.dumps(payload) + "\n```"
        else:
            # Emit batch_run_anisotropy next
            payload = {
                "tool": "batch_run_anisotropy",
                "args": {
                    "dir_path": str(TEST_DATA.resolve()),
                    "pattern": "*.csv",
                    "g_factor": 1.0,
                    "time_col": "time",
                    "Ipar_col": "I_par",
                    "Iperp_col": "I_perp",
                    "smooth_window": 3,
                    "out_prefix": "anisotropy_out",
                },
            }
            seq.append('batch')
            return "Continuing.\n\n```toolcall\n" + json.dumps(payload) + "\n```"

    loop = AgentLoop(max_steps=4)
    trace = loop.run(goal="Process all CSVs in test/data", llm_respond=llm_respond, confirm_destructive=False)
    # Expect at least 2 steps, with second having error requiring confirmation
    assert len(trace['trace']) >= 2
    last_obs = trace['trace'][-1].get('observation')
    assert isinstance(last_obs, dict)
    assert 'requires confirmation' in (last_obs.get('error') or '').lower()

    # Now run with confirmation allowed; reuse same llm_respond to emit again
    seq.clear()
    trace2 = loop.run(goal="Process all CSVs in test/data", llm_respond=llm_respond, confirm_destructive=True)
    # Should not error on destructive confirmation now
    obs2 = [s.get('observation') for s in trace2['trace'] if s.get('observation')]
    assert any('result' in o for o in obs2)
