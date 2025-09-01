from __future__ import annotations

import sys
import argparse
import json
import os
import re
import ast
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import httpx  # type: ignore
except Exception as e:  # pragma: no cover
    httpx = None

# ------------------------------ Config ---------------------------------
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "gemma:2b-instruct")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm:33m")

MAX_CHARS_PER_CHUNK = 1800
OVERLAP = 200
TOP_K = 6
EMBED_BATCH = 32  # batch size for embeddings

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".venv", "venv", "env", ".idea", ".vscode", ".tox"
}

# ------------------------------ Utils ----------------------------------

def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def sha1(s: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def norm_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n

# ------------------------------ Data -----------------------------------

@dataclass
class RagMeta:
    id: str
    file: str
    kind: str
    name: str
    preview: str
    text: str   # NEW: store full text

class RagIndex:
    def __init__(self, store_dir: pathlib.Path):
        self.dir = pathlib.Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.dir / "embeddings.npy"
        self.meta_path = self.dir / "meta.jsonl"
        self.emb: Optional[np.ndarray] = None
        self.meta: List[RagMeta] = []
        self._id2row: Dict[str, int] = {}

    def size(self) -> int:
        return 0 if self.meta is None else len(self.meta)

    def _rebuild_id_index(self):
        self._id2row = {m.id: i for i, m in enumerate(self.meta)}

    def load(self) -> None:
        if self.emb_path.exists():
            try:
                self.emb = np.load(self.emb_path)
            except Exception:
                self.emb = None
        else:
            self.emb = None

        self.meta = []
        if self.meta_path.exists():
            try:
                with self.meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        d = json.loads(s)
                        # Back-compat: if older entry lacks 'text', copy from preview
                        if "text" not in d:
                            d["text"] = d.get("preview", "")
                        self.meta.append(RagMeta(**d))
            except Exception:
                self.meta = []

        self._rebuild_id_index()

        # Keep emb/meta aligned (defensive)
        if self.emb is not None and len(self.emb) != len(self.meta):
            # truncate to shortest
            n = min(len(self.emb), len(self.meta))
            self.emb = self.emb[:n] if self.emb is not None else None
            self.meta = self.meta[:n]
            self._rebuild_id_index()

    def save(self) -> None:
        if self.emb is not None:
            np.save(self.emb_path, self.emb)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    def upsert_many(self, vectors: np.ndarray, metas: List[RagMeta]) -> None:
        """Insert or replace rows by id to keep the index fresh."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be (N, D)")
        vectors = norm_rows(vectors)

        if self.emb is None or self.size() == 0:
            self.emb = vectors
            self.meta = metas
            self._rebuild_id_index()
            self.save()
            return

        # Replace-or-append
        rows_to_replace: List[int] = []
        new_vecs: List[np.ndarray] = []
        new_metas: List[RagMeta] = []

        for v, m in zip(vectors, metas):
            if m.id in self._id2row:
                rows_to_replace.append(self._id2row[m.id])
                self.meta[self._id2row[m.id]] = m
            else:
                new_vecs.append(v[None, :])
                new_metas.append(m)

        # Perform replacements
        if rows_to_replace:
            for ridx, v in zip(rows_to_replace, vectors[:len(rows_to_replace)]):
                self.emb[ridx, :] = v

        # Append new
        if new_vecs:
            app = np.vstack(new_vecs)
            self.emb = np.vstack([self.emb, app])
            self.meta.extend(new_metas)

        self._rebuild_id_index()
        self.save()

# ------------------------------ Parser ---------------------------------

@dataclass
class Passage:
    id: str
    file: str
    kind: str
    name: str
    text: str

class PyDocParser:
    @staticmethod
    def parse_python_file(path: pathlib.Path) -> List[Passage]:
        src = read_text(path)
        passages: List[Passage] = []
        if not src.strip():
            return passages
        try:
            tree = ast.parse(src)
            module_doc = ast.get_docstring(tree) or ""
        except Exception:
            tree = None
            module_doc = ""
        if module_doc.strip():
            passages.append(
                Passage(
                    id=f"{path}::module::{sha1(module_doc)[:10]}",
                    file=str(path),
                    kind="module_doc",
                    name=path.name,
                    text=module_doc.strip(),
                )
            )
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    if doc.strip():
                        name = getattr(node, "name", "function")
                        passages.append(
                            Passage(
                                id=f"{path}::func::{name}::{sha1(doc)[:10]}",
                                file=str(path),
                                kind="func_doc",
                                name=name,
                                text=doc.strip(),
                            )
                        )
                elif isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node) or ""
                    if doc.strip():
                        name = getattr(node, "name", "Class")
                        passages.append(
                            Passage(
                                id=f"{path}::class::{name}::{sha1(doc)[:10]}",
                                file=str(path),
                                kind="class_doc",
                                name=name,
                                text=doc.strip(),
                            )
                        )
        # Code chunking (drop full-line comments but keep code)
        cleaned = re.sub(r"^[ \t]*#.*?$", "", src, flags=re.MULTILINE).strip()
        if cleaned:
            start = 0
            n = len(cleaned)
            while start < n:
                end = min(n, start + MAX_CHARS_PER_CHUNK)
                chunk = cleaned[start:end]
                if chunk.strip():
                    # STABLE ID: include hash of the chunk to avoid stale collisions
                    h = sha1(chunk)[:10]
                    passages.append(
                        Passage(
                            id=f"{path}::code::{start}-{end}::{h}",
                            file=str(path),
                            kind="code",
                            name=path.name,
                            text=chunk,
                        )
                    )
                if end >= n:
                    break
                start = end - OVERLAP
        return passages

# ------------------------------ Ollama ---------------------------------

class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embed_model: str):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama client")
        self.client = httpx.Client(timeout=120)

    def _models(self) -> List[str]:
        r = self.client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", [])]

    def check_ready(self) -> Tuple[bool, str]:
        try:
            models = self._models()
            ok_chat = any(self.chat_model.split(":")[0] in m for m in models)
            ok_embed = any(self.embed_model.split(":")[0] in m for m in models)
            msg = "OK"
            if not ok_chat:
                msg = f"Chat model '{self.chat_model}' not pulled."
            if not ok_embed:
                msg = f"Embed model '{self.embed_model}' not pulled."
            return ok_chat and ok_embed, msg
        except Exception as e:
            return False, f"Ollama not reachable: {e}"

    def pull_model(self, name: str, progress_cb: Optional[callable] = None) -> None:
        """Pull an Ollama model if missing. Blocks until completion. Raises on failure."""
        try:
            if progress_cb:
                progress_cb(f"Pulling model '{name}' (this may take a while)…")
            payload = {"name": name, "stream": False}
            r = self.client.post(f"{self.base_url}/api/pull", json=payload)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{name}': {e}")

    def ensure_models(self, progress_cb: Optional[callable] = None) -> None:
        ok, _ = self.check_ready()
        if ok:
            if progress_cb:
                progress_cb("Required models already available.")
            return
        try:
            models = self._models()
        except Exception as e:
            raise RuntimeError(f"Cannot contact Ollama: {e}")
        chat_short = self.chat_model.split(":")[0]
        embed_short = self.embed_model.split(":")[0]
        if not any(chat_short in m for m in models):
            self.pull_model(self.chat_model, progress_cb)
        # refresh
        models = self._models()
        if not any(embed_short in m for m in models):
            self.pull_model(self.embed_model, progress_cb)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts (Ollama currently takes one prompt per call, so we loop in batches)."""
        out_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), EMBED_BATCH):
            chunk = texts[i:i+EMBED_BATCH]
            for t in chunk:
                payload = {"model": self.embed_model, "prompt": t}
                r = self.client.post(f"{self.base_url}/api/embeddings", json=payload)
                r.raise_for_status()
                emb = r.json().get("embedding", [])
                out_vecs.append(np.asarray(emb, dtype=np.float32))
        X = np.vstack(out_vecs) if out_vecs else np.zeros((0, 0), dtype=np.float32)
        return norm_rows(X)

# ------------------------------ Build ----------------------------------

def iter_python_files(root: pathlib.Path) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for p in root.rglob("*.py"):
        if not p.is_file():
            continue
        # Skip excluded dirs
        parts = set(p.parts)
        if parts & EXCLUDE_DIRS:
            continue
        files.append(p)
    return files


def build_index(root: pathlib.Path, store: RagIndex, ollama: OllamaClient, status_cb=None) -> int:
    store.load()
    files = iter_python_files(root)
    total = len(files)
    added_or_updated = 0

    for i, path in enumerate(files, 1):
        if status_cb:
            status_cb(f"Parsing {path} [{i}/{total}]")
        try:
            passages = PyDocParser.parse_python_file(path)
        except Exception as e:
            if status_cb:
                status_cb(f"Skipping {path}: parse error: {e}")
            continue

        if not passages:
            continue

        # Build metas and texts
        metas: List[RagMeta] = []
        texts: List[str] = []
        for ps in passages:
            prev = ps.text[:200] + ("…" if len(ps.text) > 200 else "")
            metas.append(
                RagMeta(
                    id=ps.id,
                    file=ps.file,
                    kind=ps.kind,
                    name=ps.name,
                    preview=prev,
                    text=ps.text,  # full text
                )
            )
            texts.append(ps.text)

        # Embed and upsert (replace if id already there)
        vecs = ollama.embed_batch(texts)
        if vecs.size == 0:
            if status_cb:
                status_cb(f"Embedding failed for {path}")
            continue

        before = store.size()
        store.upsert_many(vecs, metas)
        after = store.size()

        # Rough count: if size grew, that many were new; otherwise assume replacements happened.
        delta = max(0, after - before)
        added_or_updated += max(delta, 1)  # count at least 1 to reflect work done

    return added_or_updated

# ------------------------------ Main -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ChiChat RAG build worker (subprocess)")
    ap.add_argument("--root", required=True, help="Folder to index")
    ap.add_argument("--store", required=True, help="RAG store directory (output)")
    ap.add_argument("--base-url", default=DEFAULT_OLLAMA_BASE)
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root)
    store_dir = pathlib.Path(args.store)
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Root folder not found: {root}")
        return 2

    try:
        ollama = OllamaClient(args.base_url, args.chat_model, args.embed_model)

        def progress(s: str):
            # Prefix so parent can filter
            print(f"PROGRESS: {s}")

        # Ensure required models, printing progress so the parent UI can relay it
        progress("Checking Ollama and required models…")
        ollama.ensure_models(progress_cb=progress)

        store = RagIndex(store_dir)
        added = build_index(root, store, ollama, status_cb=progress)
        print(f"DONE: {added}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
