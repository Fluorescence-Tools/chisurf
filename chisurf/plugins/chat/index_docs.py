import os
import re
import json
import time
import glob
import hashlib
import pathlib
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# Optional deps
try:
    import httpx
except Exception:
    httpx = None

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    faiss = None
    _FAISS_OK = False

try:
    from docx import Document  # python-docx
except Exception:
    Document = None

# ------------------------------- Paths -----------------------------------
PLUGIN_DIR = pathlib.Path(__file__).resolve().parent
# Store lives next to the plugin (requested behavior)
DEFAULT_STORE = PLUGIN_DIR / "rag_store"

# Try to locate repo/project roots sensibly; fall back to parent chains
def _guess_project_root() -> pathlib.Path:
    # typical: chisurf/chisurf/plugins/chat/index_docs.py -> project root two or three parents up
    for up in (3, 2, 4):
        try:
            p = PLUGIN_DIR.parents[up]
            return p
        except Exception:
            continue
    return PLUGIN_DIR.parents[2] if len(PLUGIN_DIR.parents) >= 3 else PLUGIN_DIR

PROJECT_ROOT = _guess_project_root()

# Source roots to scan
SOURCE_ROOTS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "chisurf",
    PROJECT_ROOT / "modules",
]

DOC_PATTERNS = ["docs/**/*.md", "README*.md", "docs/**/*.docx"]
CODE_PATTERNS = ["**/*.py", "**/*.cpp", "**/*.h", "**/*.hpp"]

# ----------------------------- Embeddings --------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_BATCH_ENV = int(os.environ.get("OLLAMA_EMBED_BATCH", "16"))
EMBED_DIM_DEFAULT = 768  # will be updated after first successful embed if needed

# ------------------------------ Chunking ---------------------------------
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# ------------------------------- Logging ---------------------------------
# use chisurf logging if available, otherwise print
try:
    import chisurf  # noqa: F401
    from chisurf import logging  # type: ignore
except Exception:
    class _Log:
        def info(self, *a, **k): print("[INFO]", *a)
        def warning(self, *a, **k): print("[WARN]", *a)
        def debug(self, *a, **k): print("[DEBUG]", *a)
        def error(self, *a, **k): print("[ERROR]", *a)
    logging = _Log()  # type: ignore


# ============================== Utilities ================================

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    # Deterministic pseudo-embedding based on SHA256 (fallback if Ollama unavailable)
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(h[:8], 'little', signed=False) % (2**32 - 1)
    rng = np.random.RandomState(seed)
    vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    return _normalize_vec(vec)


def _chunk_iterable(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _clean_for_embed(t: str, max_chars: int = 4000) -> str:
    # remove NULs, normalize whitespace, cap length
    t = t.replace("\u0000", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        t = " "
    return t[:max_chars]


class EmbeddingClient:
    """Robust client for Ollama /api/embeddings with batch→singleton fallback."""
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_EMBED_MODEL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        if httpx is None:
            raise RuntimeError("httpx not available")
        self.http = httpx.Client(timeout=timeout, follow_redirects=True, trust_env=True)

    def _embed_single(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        # Try modern "input"
        r = self.http.post(url, json={"model": self.model, "input": [text], "keep_alive": -1})
        r.raise_for_status()
        data = r.json()
        if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
            return data["embeddings"][0]
        if "embedding" in data and isinstance(data["embedding"], list):
            return data["embedding"]
        # Try legacy "prompt"
        r2 = self.http.post(url, json={"model": self.model, "prompt": text, "keep_alive": -1})
        r2.raise_for_status()
        data2 = r2.json()
        if "embeddings" in data2 and isinstance(data2["embeddings"], list) and data2["embeddings"]:
            return data2["embeddings"][0]
        if "embedding" in data2 and isinstance(data2["embedding"], list):
            return data2["embedding"]
        raise RuntimeError("No embedding in response")

    def embed(self, texts: List[str], batch_size: int = _BATCH_ENV) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM_DEFAULT), dtype=np.float32)

        cleaned = [_clean_for_embed(t) for t in texts]
        vectors: List[Optional[np.ndarray]] = [None] * len(cleaned)

        # try small batches; on any failure, fallback to singletons for that chunk
        bs = max(1, int(batch_size))
        url = f"{self.base_url}/api/embeddings"

        for chunk in _chunk_iterable(list(enumerate(cleaned)), bs):
            idxs, chunk_texts = zip(*chunk)
            try:
                if len(chunk_texts) == 1:
                    raise RuntimeError("skip_batch_for_singleton")
                r = self.http.post(url, json={"model": self.model, "input": list(chunk_texts), "keep_alive": -1})
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings")
                if not (isinstance(embs, list) and len(embs) == len(chunk_texts)):
                    raise RuntimeError("unexpected batch payload")
                for j, emb in enumerate(embs):
                    vectors[idxs[j]] = _normalize_vec(np.array(emb, dtype=np.float32))
            except Exception:
                # retry each item alone
                for i_single, t_single in chunk:
                    try:
                        emb = self._embed_single(t_single)
                        vectors[i_single] = _normalize_vec(np.array(emb, dtype=np.float32))
                    except Exception:
                        logging.warning(f"Ollama singleton embed failed for index {i_single}; using hash fallback.")
                        # We'll fill hash fallback later

        # determine dim
        dim = None
        for v in vectors:
            if isinstance(v, np.ndarray):
                dim = int(v.shape[0])
                break
        if dim is None:
            dim = EMBED_DIM_DEFAULT

        # fill missing with hash fallback of correct dim
        for i, v in enumerate(vectors):
            if v is None:
                vectors[i] = _hash_embedding(cleaned[i], dim)

        return np.vstack(vectors).astype(np.float32)


# ============================ Extraction =================================

_DOCSTRING_RE = re.compile(r'([\'\"]{3})([\s\S]*?)\1', re.MULTILINE)
_COMMENT_RE = re.compile(r'^[\s]*#(.*)$', re.MULTILINE)

def _extract_docx_text(path: pathlib.Path) -> str:
    if Document is None:
        return ""
    try:
        doc = Document(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception:
        return ""

def _extract_md_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _extract_code_text(path: pathlib.Path) -> str:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    parts = []
    # docstrings
    for m in _DOCSTRING_RE.finditer(src):
        content = (m.group(2) or "").strip()
        if content:
            parts.append(content)
    # line comments
    for m in _COMMENT_RE.finditer(src):
        c = (m.group(1) or "").strip()
        if c:
            parts.append(c)
    if parts:
        return "\n".join(parts)
    # fallback truncated entire file (kept as-is; chunker will cap)
    return src

def iter_sources() -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    # Docs & README
    for pattern in DOC_PATTERNS:
        for p in glob.glob(str(PROJECT_ROOT / pattern), recursive=True):
            path = pathlib.Path(p)
            if path.suffix.lower() == ".docx":
                text = _extract_docx_text(path)
            else:
                text = _extract_md_text(path)
            if text.strip():
                items.append((str(path.resolve()), text))

    # Code
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for pat in CODE_PATTERNS:
            for p in glob.glob(str(root / pat), recursive=True):
                path = pathlib.Path(p)
                text = _extract_code_text(path)
                if text.strip():
                    items.append((str(path.resolve()), text))

    return items


# ============================== Chunking =================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks


# ============================== Indexing =================================

def embed_texts(texts: List[str]) -> Tuple[np.ndarray, int]:
    """Return (embeddings, dim). Uses Ollama if available; falls back to deterministic hashing."""
    if not texts:
        return np.zeros((0, EMBED_DIM_DEFAULT), dtype=np.float32), EMBED_DIM_DEFAULT

    # Prefer Ollama when possible
    if httpx is not None:
        try:
            client = EmbeddingClient()
            embs = client.embed(texts, batch_size=_BATCH_ENV)
            dim = int(embs.shape[1]) if embs.ndim == 2 else EMBED_DIM_DEFAULT
            return embs.astype(np.float32), dim
        except Exception as e:
            logging.warning(f"Ollama embeddings unavailable, using hash fallback. Reason: {e}")

    # Fallback: deterministic hash vectors
    dim = EMBED_DIM_DEFAULT
    embs = np.vstack([_hash_embedding(_clean_for_embed(t), dim) for t in texts]).astype(np.float32)
    return embs, dim


def build_index(store_dir: pathlib.Path = DEFAULT_STORE) -> Dict[str, Any]:
    store_dir = pathlib.Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, str]] = []
    texts: List[str] = []
    paths: List[str] = []

    t0 = time.time()

    for abs_path, text in iter_sources():
        for ch in chunk_text(text):
            paths.append(abs_path)
            texts.append(ch)
            entries.append({"path": abs_path, "text": ch})

    if not texts:
        meta = {"count": 0, "dim": EMBED_DIM_DEFAULT, "paths": [], "store_type": "empty"}
        (store_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # also drop a tiny manifest so UI health check can detect store existence
        (store_dir / "manifest.json").write_text(json.dumps({"chunks": 0}, indent=2), encoding="utf-8")
        return meta

    embs, dim = embed_texts(texts)
    # Ensure embeddings are L2-normalized (cosine via dot)
    embs = np.vstack([_normalize_vec(v) for v in embs])

    # Persist embeddings & index
    if _FAISS_OK:
        index = faiss.IndexFlatIP(dim)  # inner product (with normalized = cosine)
        index.add(embs)
        faiss.write_index(index, str(store_dir / "index.faiss"))
        store_type = "faiss"
    else:
        store_type = "numpy"

    np.save(store_dir / "embeddings.npy", embs)
    (store_dir / "texts.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
                                           encoding="utf-8")

    meta: Dict[str, Any] = {
        "count": int(embs.shape[0]),
        "dim": int(dim),
        "paths": paths,
        "store_type": store_type,
    }
    (store_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (store_dir / "manifest.json").write_text(json.dumps({"chunks": int(embs.shape[0])}, indent=2), encoding="utf-8")

    meta["build_seconds"] = time.time() - t0
    return meta


# ============================== Retrieval ================================

def _load_store(store_dir: pathlib.Path = DEFAULT_STORE):
    store_dir = pathlib.Path(store_dir)
    meta_path = store_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    texts_path = store_dir / "texts.jsonl"
    texts: List[Dict[str, str]] = []
    if texts_path.exists():
        texts = [json.loads(l) for l in texts_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    embs_path = store_dir / "embeddings.npy"
    embs = np.load(embs_path) if embs_path.exists() else None

    index = None
    if meta.get("store_type") == "faiss" and _FAISS_OK:
        faiss_path = store_dir / "index.faiss"
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))

    return {"meta": meta, "texts": texts, "embs": embs, "index": index}


def retrieve(query: str, top_k: int = TOP_K, store_dir: pathlib.Path = DEFAULT_STORE) -> str:
    store = _load_store(store_dir)
    if store is None or int(store["meta"].get("count", 0)) == 0:
        return ""

    texts: List[Dict[str, str]] = store["texts"]
    paths: List[str] = store["meta"].get("paths", [])
    embs: Optional[np.ndarray] = store["embs"]
    index = store["index"]
    dim = int(store["meta"].get("dim", EMBED_DIM_DEFAULT))

    q_vecs, _dim_chk = embed_texts([query])
    q_emb = _normalize_vec(q_vecs[0])
    if _dim_chk != dim:
        # project or pad to match (normally shouldn't happen if using same model)
        if _dim_chk > dim:
            q_emb = q_emb[:dim]
        elif _dim_chk < dim:
            pad = np.zeros((dim - _dim_chk,), dtype=np.float32)
            q_emb = np.concatenate([q_emb, pad], axis=0)
        q_emb = _normalize_vec(q_emb)

    if index is not None:
        D, I = index.search(q_emb.reshape(1, -1).astype(np.float32), min(top_k, len(texts)))
        idxs = I[0].tolist()
    else:
        # cosine similarity (embeddings are normalized)
        sims = embs.dot(q_emb.astype(np.float32))
        idxs = np.argsort(-sims)[:min(top_k, len(texts))].tolist()

    parts = []
    for i in idxs:
        if i < 0 or i >= len(texts):
            continue
        rec = texts[i]
        path = rec.get("path", paths[i] if i < len(paths) else "")
        chunk = rec.get("text", "")
        parts.append(f"[{path}]\n{chunk}")
    return "\n\n---\n\n".join(parts)


# ================================ CLI ====================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build RAG store for ChiSurf Assistant")
    ap.add_argument("--store", type=str, default=str(DEFAULT_STORE), help="Output store directory")
    args = ap.parse_args()
    out = build_index(pathlib.Path(args.store))
    print(json.dumps(out, indent=2))
