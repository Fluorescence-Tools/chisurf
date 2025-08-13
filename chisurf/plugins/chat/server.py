import os
import uuid
import json
import time
import logging
import sqlite3
import pathlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    import httpx
except Exception:
    httpx = None

# ------------------------------- RAG utils -------------------------------
# We load these lazily/defensively so the server still runs if RAG isn't installed yet.
try:
    from .index_docs import (
        retrieve as rag_retrieve,
        DEFAULT_STORE as RAG_DEFAULT_STORE,
        build_index as rag_build_index,
    )
except Exception:
    rag_retrieve = None          # type: ignore
    RAG_DEFAULT_STORE = None     # type: ignore
    rag_build_index = None       # type: ignore

# ------------------------------- Config ---------------------------------
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:1b-instruct-q8_0")
EMBED_MODEL       = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
SYSTEM_PROMPT     = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a concise, helpful local assistant. Keep answers short unless asked for more detail.",
)
MAX_TURNS         = int(os.environ.get("MAX_TURNS", "80"))  # number of user/assistant messages kept

def _default_db_path() -> str:
    """Return default path for chat_history.db in chisurf user settings folder.
    Falls back to ~/.chisurf if chisurf.settings cannot be imported.
    Ensures the parent directory exists.
    """
    try:
        from chisurf.settings import get_path as _cs_get_path  # type: ignore
        p = _cs_get_path('settings') / 'chat_history.db'
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:
        home_settings = pathlib.Path.home() / '.chisurf'
        try:
            home_settings.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(home_settings / 'chat_history.db')

DB_PATH = os.environ.get("OLLAMA_CHAT_DB", _default_db_path())

# Resolve RAG store directory:
# 1) RAG_STORE_DIR env
# 2) index_docs.DEFAULT_STORE (plugin folder) if available
# 3) fallback to ./rag_store next to this server file
_RAG_STORE_ENV = os.environ.get("RAG_STORE_DIR")
if _RAG_STORE_ENV:
    RAG_STORE_DIR = pathlib.Path(_RAG_STORE_ENV)
elif RAG_DEFAULT_STORE is not None:
    RAG_STORE_DIR = pathlib.Path(RAG_DEFAULT_STORE)
else:
    # fallback: rag_store next to this file (plugin-local)
    RAG_STORE_DIR = pathlib.Path(__file__).resolve().parent / "rag_store"

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("chat.server")

APP = FastAPI()

# --------------------------- SQLite persistence --------------------------
_DB: Optional[sqlite3.Connection] = None

def db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        _DB = sqlite3.connect(DB_PATH, check_same_thread=False)
        _DB.execute("PRAGMA journal_mode=WAL;")
        _DB.execute("PRAGMA synchronous=NORMAL;")
        _DB.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT NOT NULL,
                turn_index     INTEGER NOT NULL,
                role           TEXT NOT NULL CHECK (role IN ('user','assistant')),
                content        TEXT NOT NULL,
                created_at     REAL NOT NULL,
                PRIMARY KEY (conversation_id, turn_index)
            )
            """
        )
        _DB.commit()
    return _DB

def db_next_index(conversation_id: str) -> int:
    cur = db().execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    return int(cur.fetchone()[0])

def db_append(conversation_id: str, role: str, content: str) -> None:
    idx = db_next_index(conversation_id)
    db().execute(
        "INSERT OR REPLACE INTO messages (conversation_id, turn_index, role, content, created_at) VALUES (?,?,?,?,?)",
        (conversation_id, idx, role, content, time.time()),
    )
    db().commit()

def db_get_history(conversation_id: str, limit: int) -> List[Dict[str, str]]:
    cur = db().execute(
        """
        SELECT role, content
          FROM messages
         WHERE conversation_id = ?
         ORDER BY turn_index DESC
         LIMIT ?
        """,
        (conversation_id, limit),
    )
    rows = cur.fetchall()
    rows.reverse()  # chronological
    return [{"role": r[0], "content": r[1]} for r in rows]

def db_trim(conversation_id: str, keep: int) -> None:
    db().execute(
        """
        DELETE FROM messages
         WHERE conversation_id = ?
           AND turn_index < (
                 SELECT MIN(turn_index)
                   FROM (
                         SELECT turn_index
                           FROM messages
                          WHERE conversation_id = ?
                          ORDER BY turn_index DESC
                          LIMIT ?
                        )
               )
        """,
        (conversation_id, conversation_id, keep),
    )
    db().commit()

def db_delete_conversation(conversation_id: str) -> None:
    db().execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    db().commit()

# ------------------------------- Ollama ----------------------------------

def _ollama_pull_model(model: str) -> bool:
    if httpx is None:
        log.warning("httpx not installed; cannot pull model")
        return False
    try:
        url = f"{OLLAMA_BASE_URL}/api/pull"
        with httpx.Client(timeout=None) as client:
            r = client.post(url, json={"name": model}, timeout=None)
            if r.status_code != 200:
                log.warning(f"Ollama pull HTTP {r.status_code}: {r.text[:300]}")
                return False
            return True
    except Exception as e:
        log.warning(f"Ollama pull exception: {e}")
        return False

def _ollama_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[str]:
    if httpx is None:
        log.info("Ollama chat skipped: httpx not available")
        return None
    url = f"{OLLAMA_BASE_URL}/api/chat"
    use_model = (model or OLLAMA_CHAT_MODEL)
    payload = {"model": use_model, "messages": messages, "stream": False}
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=payload)
            if r.status_code == 404 and "not found" in r.text.lower():
                log.info(f"Model '{use_model}' not present, attempting pull…")
                if _ollama_pull_model(use_model):
                    r = client.post(url, json=payload)
            if r.status_code != 200:
                log.warning(f"Ollama chat HTTP {r.status_code}: {r.text[:300]}")
                return None
            data = r.json()
            msg = (data.get("message") or {})
            return msg.get("content")
    except Exception as e:
        log.warning(f"Ollama chat exception: {e}")
        return None

# ------------------------------- Fallback --------------------------------

def _fallback_reply(query: str, __: Optional[Dict[str, Any]] = None) -> str:
    """Fallback response when Ollama is unavailable.
    If a RAG store is configured, include top retrieved snippets to still provide value.
    """
    try:
        rag_ctx = _build_rag_context(query)
    except Exception:
        rag_ctx = ""
    if rag_ctx:
        return (
            "[local fallback] I couldn't reach Ollama just now, but here are relevant snippets from the project RAG store.\n\n"
            + rag_ctx
        )
    return (
        "[local fallback] I couldn't reach Ollama just now. "
        "Please check that 'ollama serve' is running and the model is pulled."
    )

# ------------------------------ RAG Helpers ------------------------------

def _rag_store_has_index(p: pathlib.Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        # accept either meta.json or manifest.json as a signal the store exists
        return (p / "meta.json").exists() or (p / "manifest.json").exists()
    except Exception:
        return False

def _build_rag_context(query: str, top_k: int = 5) -> str:
    """Return retrieved context text or empty string on any error/missing store."""
    try:
        if not query or rag_retrieve is None or RAG_STORE_DIR is None:
            return ""
        if not _rag_store_has_index(RAG_STORE_DIR):
            return ""
        return rag_retrieve(query, top_k=top_k, store_dir=RAG_STORE_DIR)
    except Exception:
        return ""

# --------------------------------- API -----------------------------------

@APP.get("/health")
def health():
    info: Dict[str, Any] = {"backend": "ok", "db": "ok"}
    try:
        db().execute("SELECT 1")
    except Exception as e:
        info["db"] = f"error:{e.__class__.__name__}"
    # Report RAG store status
    try:
        info["rag_store"] = "present" if _rag_store_has_index(RAG_STORE_DIR) else "missing"
        info["rag_store_dir"] = str(RAG_STORE_DIR)
    except Exception:
        info["rag_store"] = "error"
    if httpx is None:
        info["ollama"] = "missing_httpx"
        return JSONResponse(info)
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            info["ollama"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        info["ollama"] = f"error:{e.__class__.__name__}"
    return JSONResponse(info)

@APP.get("/history/{conversation_id}")
def get_history(conversation_id: str):
    history = db_get_history(conversation_id, MAX_TURNS)
    return {"conversation_id": conversation_id, "history": history, "history_len": len(history)}

@APP.post("/rag/build")
def rag_build(payload: Dict[str, Any] = None):
    """Build or rebuild the RAG store.
    Optional JSON payload fields:
      - store_dir: custom output directory (string)
    Returns meta information from the indexer.
    """
    if payload is None:
        payload = {}
    try:
        # Determine store directory
        store_dir = payload.get("store_dir")
        if store_dir:
            out_dir = pathlib.Path(store_dir)
        else:
            out_dir = pathlib.Path(RAG_STORE_DIR)

        if rag_build_index is None:
            return JSONResponse({"error": "rag build unavailable (index_docs not importable)"}, status_code=503)

        out_dir.mkdir(parents=True, exist_ok=True)
        meta = rag_build_index(out_dir)

        # Normalize return payload
        if isinstance(meta, dict):
            meta = dict(meta)
        else:
            meta = {"meta": str(meta)}
        # Ensure path-like fields are serialized
        if isinstance(meta.get("paths"), list):
            meta["paths"] = [str(p) for p in meta["paths"]]
        meta["store_dir"] = str(out_dir)

        return JSONResponse({"ok": True, "meta": meta})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@APP.post("/rag/retrieve")
def rag_retrieve_api(payload: Dict[str, Any]):
    """Simple retrieval endpoint for debugging."""
    query = (payload.get("query") or "").strip()
    k = int(payload.get("top_k", 5))
    if not query:
        return JSONResponse({"error": "empty query"}, status_code=400)
    if rag_retrieve is None:
        return JSONResponse({"error": "retrieve unavailable (index_docs not importable)"}, status_code=503)
    if not _rag_store_has_index(RAG_STORE_DIR):
        return JSONResponse({"error": "rag store missing"}, status_code=404)
    try:
        ctx = rag_retrieve(query, top_k=k, store_dir=RAG_STORE_DIR)
        return JSONResponse({"ok": True, "context": ctx})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@APP.post("/reset")
def reset(payload: Dict[str, Any]):
    cid = payload.get("conversation_id")
    if cid:
        db_delete_conversation(cid)
    return {"ok": True}

@APP.post("/chat")
async def chat(payload: Dict[str, Any], request: Request):
    # Accept cid from JSON, header, or cookie; then set a cookie on the reply
    payload_cid = payload.get("conversation_id")
    header_cid  = request.headers.get("X-Conversation-ID")
    cookie_cid  = request.cookies.get("cid")
    conversation_id = payload_cid or header_cid or cookie_cid or str(uuid.uuid4())

    user_msg: str = (payload.get("message") or "").strip()
    model: Optional[str] = payload.get("model")
    context: Optional[Dict[str, Any]] = payload.get("context")

    if not user_msg:
        resp = JSONResponse({"error": "empty message", "conversation_id": conversation_id})
        resp.set_cookie("cid", conversation_id, httponly=False, samesite="Lax")
        return resp

    # Retrieve RAG context and build full message list.
    rag_ctx = _build_rag_context(user_msg)
    history = db_get_history(conversation_id, MAX_TURNS)
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    combined_user = ("CONTEXT:\n" + rag_ctx + "\n\nUSER QUESTION:\n" + user_msg) if rag_ctx else user_msg
    messages.append({"role": "user", "content": combined_user})

    assistant = _ollama_chat(messages, model=model)
    fallback_used = False
    if not assistant:
        assistant = _fallback_reply(user_msg, context=context)
        fallback_used = True

    # Persist both turns and trim
    db_append(conversation_id, "user", user_msg)
    db_append(conversation_id, "assistant", assistant)
    db_trim(conversation_id, MAX_TURNS)

    new_history = db_get_history(conversation_id, MAX_TURNS)

    data = {
        "reply": assistant,
        "conversation_id": conversation_id,
        "history": new_history,
        "history_len": len(new_history),
        "messages_sent_count": len(messages),  # includes system + history + new user
        "context": {"rag": bool(rag_ctx), "rag_preview": rag_ctx[:500] if rag_ctx else ""},
        "tool": None,
        "tool_result": None,
        "tool_error": None,
        "fallback": fallback_used,
    }

    resp = JSONResponse(data)
    resp.set_cookie("cid", conversation_id, httponly=False, samesite="Lax")
    return resp

@APP.post("/agent")
def agent_stub(payload: Dict[str, Any]):
    """Non-breaking stub so the UI's Agent Mode won't 404.
    Replace with your real agent loop when ready.
    """
    goal = (payload.get("goal") or "").strip()
    cid  = payload.get("conversation_id") or str(uuid.uuid4())
    if not goal:
        return JSONResponse({"error": "empty goal", "conversation_id": cid}, status_code=400)
    reply = (
        "Agent stub: no tools configured yet.\n"
        f"Goal: {goal}\n\n"
        "Tip: implement your planner/executor and return a 'trace' list here."
    )
    # Persist to history for symmetry with /chat
    db_append(cid, "user", f"[Agent goal] {goal}")
    db_append(cid, "assistant", reply)
    db_trim(cid, MAX_TURNS)
    return JSONResponse({
        "conversation_id": cid,
        "reply": reply,
        "trace": [],
        "tool": None,
        "tool_result": None,
        "tool_error": None,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        APP,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=bool(int(os.environ.get("RELOAD", "0"))),
    )
