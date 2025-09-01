from __future__ import annotations

from PyQt5 import QtWidgets, QtCore, QtGui
import pathlib
import sys
from typing import List, Dict

# Copied relevant functionality from cja.py to avoid importing it
# (OllamaClient, RagIndex, PyDocParser, build_index, prompt wiring, and Qt workers)
import os
import json
import re
import ast
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import httpx

# ------------------------------ Config ---------------------------------
DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_CHAT_MODEL  = os.environ.get("OLLAMA_CHAT_MODEL", "gemma:2b-instruct")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-minilm:33m")

# Chunking
MAX_CHARS_PER_CHUNK = 1800
OVERLAP = 200

# Retrieval
TOP_K = 6
SIM_FLOOR = 0.18         # NEW: still include best few even if scores are low
MAX_CONTEXT_CHARS = 18000  # NEW: trim context to avoid overlong prompts

# Decoding style
CHAT_OPTIONS = {          # NEW: make the model more decisive and less hedgy
    "temperature": 0.3,
    "top_p": 0.9,
}

# ------------------------------ Utilities ------------------------------

def norm_embed(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n


def cosine_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q shape (D,), M shape (N, D); both unit-normalized row-wise
    return M @ q


def read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# ------------------------------ RAG store ------------------------------

@dataclass
class RagMeta:
    id: str
    file: str
    kind: str
    name: str
    preview: str
    text: str           # NEW: store full passage text (the actual context we’ll show)

class RagIndex:
    def __init__(self, store_dir: pathlib.Path):
        self.dir = pathlib.Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.dir / "embeddings.npy"
        self.meta_path = self.dir / "meta.jsonl"
        self.emb: Optional[np.ndarray] = None
        self.meta: List[RagMeta] = []

    def size(self) -> int:
        return 0 if self.meta is None else len(self.meta)

    def load(self) -> None:
        # Load embeddings and metadata if available
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
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        # Backward-compat: if older index lacks 'text', fall back to 'preview'
                        if "text" not in d:
                            d["text"] = d.get("preview", "")
                        self.meta.append(RagMeta(**d))
            except Exception:
                self.meta = []

    def save(self) -> None:
        if self.emb is not None:
            np.save(self.emb_path, self.emb)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    def upsert(self, vectors: np.ndarray, metas: List[RagMeta]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be (N, D)")
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)

        if self.emb is None or self.size() == 0:
            self.emb = vectors
            self.meta = metas
        else:
            self.emb = np.vstack([self.emb, vectors])
            self.meta.extend(metas)
        self.save()

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K) -> List[Tuple[float, RagMeta]]:
        if self.emb is None or self.size() == 0:
            return []
        q = norm_embed(query_vec)
        sims = cosine_sim_matrix(q, self.emb)
        k = min(top_k, len(sims))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out = [(float(sims[i]), self.meta[i]) for i in idx]
        # NEW: if all below floor, still return top_k so we don't “give up” too early
        if all(s < SIM_FLOOR for s, _ in out) and len(sims) > k:
            # include a couple extra for breadth
            extra_k = min(k + 2, len(sims))
            idx2 = np.argpartition(-sims, extra_k-1)[:extra_k]
            idx2 = idx2[np.argsort(-sims[idx2])]
            out = [(float(sims[i]), self.meta[i]) for i in idx2][:extra_k]
        return out

# ------------------------------ Doc parsing ----------------------------

@dataclass
class Passage:
    id: str
    file: str
    kind: str  # "module_doc", "class_doc", "func_doc", "code"
    name: str
    text: str

class PyDocParser:
    @staticmethod
    def parse_python_file(path: pathlib.Path) -> List[Passage]:
        src = read_text(path)
        passages: List[Passage] = []
        if not src.strip():
            return passages

        # 1) AST docstrings
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

        # 2) Code chunking (strip comment-only lines but keep real code)
        cleaned = re.sub(r"^[ \t]*#.*?$", "", src, flags=re.MULTILINE).strip()
        if cleaned:
            start = 0
            n = len(cleaned)
            while start < n:
                end = min(n, start + MAX_CHARS_PER_CHUNK)
                chunk = cleaned[start:end]
                if chunk.strip():
                    passages.append(
                        Passage(
                            id=f"{path}::code::{start}-{end}",
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

# ------------------------------ Ollama backend -------------------------

class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE,
                 chat_model: str = DEFAULT_CHAT_MODEL,
                 embed_model: str = DEFAULT_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
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
                # if chat also missing, prefer reporting both but embed last overwrites message
                msg = f"Embed model '{self.embed_model}' not pulled."
            return ok_chat and ok_embed, msg
        except Exception as e:
            return False, f"Ollama not reachable: {e}"

    def pull_model(self, name: str, progress_cb: Optional[callable] = None) -> None:
        """Ensure a model is pulled locally. Blocks until done. Raises on failure."""
        try:
            if progress_cb:
                progress_cb(f"Pulling model '{name}' (this may take a while)…")
            payload = {"name": name, "stream": False}
            r = self.client.post(f"{self.base_url}/api/pull", json=payload)
            r.raise_for_status()
            # Some versions return a JSON with status; we ignore content here.
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{name}': {e}")

    def ensure_models(self, progress_cb: Optional[callable] = None) -> None:
        ok, _ = self.check_ready()
        if ok:
            return
        # refresh model list
        try:
            models = self._models()
        except Exception as e:
            raise RuntimeError(f"Cannot contact Ollama: {e}")
        chat_short = self.chat_model.split(":")[0]
        embed_short = self.embed_model.split(":")[0]
        if not any(chat_short in m for m in models):
            self.pull_model(self.chat_model, progress_cb)
        # refresh again after pulling chat
        models = self._models()
        if not any(embed_short in m for m in models):
            self.pull_model(self.embed_model, progress_cb)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for t in texts:
            payload = {"model": self.embed_model, "prompt": t}
            r = self.client.post(f"{self.base_url}/api/embeddings", json=payload)
            r.raise_for_status()
            emb = r.json().get("embedding", [])
            vecs.append(np.asarray(emb, dtype=np.float32))
        X = np.vstack(vecs)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": CHAT_OPTIONS,   # NEW
        }
        r = self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        # Support both new/old shapes defensively
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return data.get("message", {}).get("content", "")
            # Some builds return a list of 'messages'
            if "messages" in data and isinstance(data["messages"], list):
                for m in reversed(data["messages"]):
                    if m.get("role") == "assistant":
                        return m.get("content", "")
        return ""

# ------------------------------ Index builder --------------------------

def iter_python_files(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]


def build_index(root: pathlib.Path, store: RagIndex, ollama: OllamaClient,
                status_cb: Optional[callable] = None) -> int:
    store.load()
    files = iter_python_files(root)
    added = 0
    for i, path in enumerate(files, 1):
        if status_cb:
            status_cb(f"Parsing {path} [{i}/{len(files)}]")
        passages = PyDocParser.parse_python_file(path)
        if not passages:
            continue
        existing_ids = set(m.id for m in store.meta)
        new_passages = [ps for ps in passages if ps.id not in existing_ids]
        if not new_passages:
            continue
        texts = [ps.text for ps in new_passages]
        vecs = ollama.embed(texts)
        metas = [
            RagMeta(
                id=ps.id,
                file=ps.file,
                kind=ps.kind,
                name=ps.name,
                preview=(ps.text[:200] + ("…" if len(ps.text) > 200 else "")),
                text=ps.text,  # NEW: keep full text
            )
            for ps in new_passages
        ]
        store.upsert(vecs, metas)
        added += len(new_passages)
    return added

# ------------------------------ Prompt wiring --------------------------

def _truncate_context(chunks: List[str], limit_chars: int = MAX_CONTEXT_CHARS) -> List[str]:
    # NEW: ensure we don’t blow the prompt window; keep most similar first
    total = 0
    out = []
    for c in chunks:
        if total + len(c) > limit_chars:
            # trim the last chunk to fit if it helps
            remain = max(0, limit_chars - total)
            if remain > 400:  # keep a meaningful tail if possible
                out.append(c[:remain] + "\n…")
                total = limit_chars
            break
        out.append(c)
        total += len(c)
    return out

def make_rag_prompt(user_query: str, hits: List[Tuple[float, RagMeta]]) -> Tuple[str, str]:
    """Return (system_instructions, context_block)."""
    if not hits:
        return (
            "You are a concise assistant that helps a user in a fluorescence analysis software. "
            "Answer clearly and correctly.",
            "",
        )

    ctx_lines = []
    for score, meta in hits:
        ctx_lines.append(
            f"# Source: {meta.file} [{meta.kind}] {meta.name}\n{meta.preview}"
        )

    context_block = "\n\n".join(ctx_lines)
    sys_msg = (
        "You are a concise assistant in fluorescence spectroscopy. Use the provided CONTEXT when helpful. "
        "If the answer is not in the context, use your own knowledge, but prefer the context. "
        "Cite filenames in your answer when drawing directly from context."
    )
    return sys_msg, context_block

# ------------------------------ Qt Workers -----------------------------

class BuildIndexWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, str)  # added_count, message
    progress = QtCore.pyqtSignal(str)       # status text

    def __init__(self, root: pathlib.Path, store: RagIndex, ollama: OllamaClient):
        super().__init__()
        self.root = root
        self.store = store
        self.ollama = ollama

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Ensure models are available; emit progress to UI
            try:
                self.ollama.ensure_models(progress_cb=self.progress.emit)
            except Exception as e:
                self.finished.emit(0, f"Model setup failed: {e}")
                return
            count = build_index(self.root, self.store, self.ollama, status_cb=self.progress.emit)
            self.finished.emit(count, f"Indexed {count} passages.")
        except Exception as e:
            self.finished.emit(0, f"Index error: {e}")


class ChatWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)        # assistant reply
    error = QtCore.pyqtSignal(str)

    def __init__(self, ollama: OllamaClient, store: RagIndex,
                 history: List[Dict[str, str]], user_text: str, use_rag: bool):
        super().__init__()
        self.ollama = ollama
        self.store = store
        self.history = history
        self.user_text = user_text
        self.use_rag = use_rag

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Ensure models present for embed/chat
            try:
                self.ollama.ensure_models()
            except Exception as e:
                self.error.emit(f"Model setup failed: {e}")
                return
            msgs = list(self.history)
            if self.use_rag and self.store.size() > 0:
                qv = self.ollama.embed([self.user_text])[0]
                hits = self.store.search(qv, top_k=TOP_K)
                sys_msg, ctx = make_rag_prompt(self.user_text, hits)

                # Put system + context before prior turns.
                msgs = (
                    [{"role": "system", "content": sys_msg}]
                    + ([{"role": "system", "content": f"CONTEXT:\n{ctx}"}] if ctx else [])
                    + msgs
                )

            # Always append the current user message last
            msgs = msgs + [{"role": "user", "content": self.user_text}]
            reply = self.ollama.chat(msgs)
            self.finished.emit(reply if reply else "(No response)")
        except Exception as e:
            self.error.emit(str(e))

# ------------------------------ UI (unchanged except for using above) --

class EnterAwarePlainTextEdit(QtWidgets.QPlainTextEdit):
    sendRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        f = self.font()
        f.setFamily("Consolas")
        self.setFont(f)
        self.setTabChangesFocus(True)
        self.setPlaceholderText("Ask something… (Enter to send, Shift+Enter for newline)")

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not (e.modifiers() & QtCore.Qt.ShiftModifier):
            e.accept()
            self.sendRequested.emit()
            return
        super().keyPressEvent(e)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Chato Settings")
        self.setModal(True)
        self.resize(520, 0)

        self._initial = initial or {}

        form = QtWidgets.QGridLayout(self)

        row = 0
        def add_row(label: str, widget: QtWidgets.QWidget):
            nonlocal row
            form.addWidget(QtWidgets.QLabel(label, self), row, 0)
            form.addWidget(widget, row, 1)
            row += 1

        # Ollama
        self.base_url = QtWidgets.QLineEdit(self)
        self.chat_model = QtWidgets.QLineEdit(self)
        self.embed_model = QtWidgets.QLineEdit(self)
        add_row("Ollama base URL:", self.base_url)
        add_row("Chat model:", self.chat_model)
        add_row("Embed model:", self.embed_model)

        # Retrieval / decoding
        self.top_k = QtWidgets.QSpinBox(self); self.top_k.setRange(1, 50)
        self.sim_floor = QtWidgets.QDoubleSpinBox(self); self.sim_floor.setRange(0.0, 1.0); self.sim_floor.setSingleStep(0.01); self.sim_floor.setDecimals(3)
        self.max_ctx = QtWidgets.QSpinBox(self); self.max_ctx.setRange(1000, 200000)
        self.temperature = QtWidgets.QDoubleSpinBox(self); self.temperature.setRange(0.0, 2.0); self.temperature.setSingleStep(0.1)
        self.top_p = QtWidgets.QDoubleSpinBox(self); self.top_p.setRange(0.0, 1.0); self.top_p.setSingleStep(0.05)
        add_row("Top K:", self.top_k)
        add_row("Similarity floor:", self.sim_floor)
        add_row("Max context chars:", self.max_ctx)
        add_row("Temperature:", self.temperature)
        add_row("Top-p:", self.top_p)

        # RAG toggle
        self.use_rag = QtWidgets.QCheckBox("Use RAG if index is available", self)
        form.addWidget(self.use_rag, row, 0, 1, 2)
        row += 1

        # Store dir
        self.store_dir = QtWidgets.QLineEdit(self)
        self.store_browse = QtWidgets.QPushButton("Browse…", self)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.store_dir, 1)
        h.addWidget(self.store_browse, 0)
        form.addWidget(QtWidgets.QLabel("RAG store directory:", self), row, 0)
        form.addLayout(h, row, 1)
        row += 1

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        form.addWidget(btns, row, 0, 1, 2)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Hook browse
        self.store_browse.clicked.connect(self._on_browse_store)

        # Load initial values
        self._load_initial()

    def _on_browse_store(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            sel = dlg.selectedFiles()
            if sel:
                self.store_dir.setText(sel[0])

    def _load_initial(self):
        get = self._initial.get
        self.base_url.setText(str(get("base_url", "")))
        self.chat_model.setText(str(get("chat_model", "")))
        self.embed_model.setText(str(get("embed_model", "")))
        self.top_k.setValue(int(get("top_k", 6)))
        self.sim_floor.setValue(float(get("sim_floor", 0.18)))
        self.max_ctx.setValue(int(get("max_ctx", 18000)))
        self.temperature.setValue(float(get("temperature", 0.3)))
        self.top_p.setValue(float(get("top_p", 0.9)))
        self.use_rag.setChecked(bool(get("use_rag", True)))
        self.store_dir.setText(str(get("store_dir", "")))

    def get_values(self) -> dict:
        return {
            "base_url": self.base_url.text().strip(),
            "chat_model": self.chat_model.text().strip(),
            "embed_model": self.embed_model.text().strip(),
            "top_k": int(self.top_k.value()),
            "sim_floor": float(self.sim_floor.value()),
            "max_ctx": int(self.max_ctx.value()),
            "temperature": float(self.temperature.value()),
            "top_p": float(self.top_p.value()),
            "use_rag": bool(self.use_rag.isChecked()),
            "store_dir": self.store_dir.text().strip(),
        }


class ChiChatDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Chato", parent)
        self.setObjectName("ChatoDock")

        # Try set window icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
            if os.path.exists(icon_path):
                self.setWindowIcon(QtGui.QIcon(icon_path))
        except Exception:
            pass

        # State
        self.history: List[Dict[str, str]] = []
        # Save RAG under the ChiChat plugin folder
        self.store_dir = pathlib.Path(__file__).parent / ".rag_store"
        self.store = RagIndex(self.store_dir)
        self.ollama = OllamaClient()
        self._rag_folder: pathlib.Path | None = None

        # UI
        w = QtWidgets.QWidget(self)
        self.setWidget(w)

        self.transcript = QtWidgets.QTextBrowser(w)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(260)

        self.input = EnterAwarePlainTextEdit(w)
        self.send_btn = QtWidgets.QPushButton("Send", w)
        self.settings_btn = QtWidgets.QPushButton("Settings", w)

        # RAG controls (based on existing chat plugin)
        self.rag_folder_lbl = QtWidgets.QLabel("No folder selected", w)
        self.rag_folder_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.pick_rag_btn = QtWidgets.QPushButton("Choose Folder…", w)
        self.build_rag_btn = QtWidgets.QPushButton("Build RAG", w)
        self.index_size_lbl = QtWidgets.QLabel("Index: 0 passages", w)

        # Layouts
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.settings_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.send_btn)

        rag_row = QtWidgets.QHBoxLayout()
        rag_row.addWidget(self.rag_folder_lbl, 1)
        rag_row.addWidget(self.pick_rag_btn, 0)
        rag_row.addWidget(self.build_rag_btn, 0)
        rag_row.addWidget(self.index_size_lbl, 0)

        layout = QtWidgets.QVBoxLayout(w)

        # Create a splitter between transcript (responses) and the input area
        input_panel = QtWidgets.QWidget(w)
        input_vbox = QtWidgets.QVBoxLayout(input_panel)
        input_vbox.setContentsMargins(0, 0, 0, 0)
        input_vbox.setSpacing(6)
        # Ensure the input area has a sensible minimum height
        try:
            self.input.setMinimumHeight(80)
        except Exception:
            pass
        input_vbox.addWidget(self.input)
        input_vbox.addLayout(btn_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, w)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.transcript)
        splitter.addWidget(input_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        layout.addLayout(rag_row)

        # Signals
        self.send_btn.clicked.connect(self.on_send)
        self.settings_btn.clicked.connect(self.on_settings)
        self.input.sendRequested.connect(self.on_send)
        self.pick_rag_btn.clicked.connect(self.on_pick_rag_folder)
        self.build_rag_btn.clicked.connect(self.on_build_rag)

        # Init state
        self.use_rag_enabled = True
        self._init_and_apply_settings()

        # Init UI
        self._append_sys("Olá! Eu sou chato o ChiSurf chatbot. Vou responder em ingles. Construa um índice (RAG). Pergunte à vontade.")
        self._refresh_index_size()

    # ---------- UI helpers ----------
    def _append_html(self, html: str):
        self.transcript.append(html)
        sb = self.transcript.verticalScrollBar()
        sb.setValue(sb.maximum())

    # Bubble theming (same as before)
    def _is_dark_palette(self) -> bool:
        p = self.transcript.palette()
        c = p.color(QtGui.QPalette.Base)
        r, g, b = c.redF(), c.greenF(), c.blueF()
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return lum < 0.5

    def _mix(self, a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
        inv = 1.0 - t
        return QtGui.QColor(
            int(a.red() * inv + b.red() * t),
            int(a.green() * inv + b.green() * t),
            int(a.blue() * inv + b.blue() * t)
        )

    def _derive_bubble_colors(self, is_user: bool):
        p = self.transcript.palette()
        base = p.color(QtGui.QPalette.Base)
        text = p.color(QtGui.QPalette.Text)
        dark = self._is_dark_palette()
        user_tint = QtGui.QColor(90, 150, 255)
        asst_tint = QtGui.QColor(200, 200, 210)
        t_user = 0.26 if dark else 0.14
        t_asst = 0.18 if dark else 0.08
        bg = self._mix(base, user_tint if is_user else asst_tint, t_user if is_user else t_asst)
        if dark:
            border = self._mix(bg, QtGui.QColor(255, 255, 255), 0.22)
        else:
            border = self._mix(bg, QtGui.QColor(0, 0, 0), 0.12)
        return bg, border, text

    def _insert_code_block(self, parent_cursor: QtGui.QTextCursor, code_text: str):
        dark = self._is_dark_palette()
        bg_base, border, _ = self._derive_bubble_colors(is_user=False)
        if dark:
            bg = self._mix(bg_base, QtGui.QColor(255, 255, 255), 0.08)
        else:
            bg = self._mix(bg_base, QtGui.QColor(0, 0, 0), 0.06)
        tfmt = QtGui.QTextTableFormat()
        tfmt.setBorder(0.6)
        tfmt.setBorderBrush(QtGui.QBrush(border))
        tfmt.setCellSpacing(0)
        tfmt.setCellPadding(6)
        tfmt.setWidth(QtGui.QTextLength(QtGui.QTextLength.PercentageLength, 100))
        table = parent_cursor.insertTable(1, 1, tfmt)
        cell = table.cellAt(0, 0)
        cfmt = cell.format().toTableCellFormat()
        cfmt.setBackground(QtGui.QBrush(bg))
        cell.setFormat(cfmt)
        ccur = cell.firstCursorPosition()
        mono = QtGui.QTextCharFormat()
        mono.setFontFamily("Consolas")
        mono.setFontFixedPitch(True)
        mono.setForeground(self.transcript.palette().brush(QtGui.QPalette.Text))
        ccur.insertText(code_text, mono)
        parent_cursor.movePosition(QtGui.QTextCursor.End)

    def _render_message_fragments(self, text: str):
        import html as _html, re as _re
        fragments = []
        s = text
        fence = _re.compile(r"```(.*?)```", _re.DOTALL)
        pos = 0
        for m in fence.finditer(s):
            before = s[pos:m.start()]
            if before.strip():
                before_esc = _html.escape(before)
                before_esc = _re.sub(
                    r"`([^`]+)`", lambda m: f"<code>{_html.escape(m.group(1))}</code>", before_esc
                )
                before_esc = before_esc.replace("\n", "<br>")
                fragments.append(("text", before_esc))
            code_body = m.group(1)
            parts = code_body.split("\n", 1)
            if len(parts) == 2 and len(parts[0]) < 20:
                code_body = parts[1]
            fragments.append(("codeblock", code_body))
            pos = m.end()
        after = s[pos:]
        if after.strip():
            after_esc = _html.escape(after)
            after_esc = _re.sub(
                r"`([^`]+)`", lambda m: f"<code>{_html.escape(m.group(1))}</code>", after_esc
            )
            after_esc = after_esc.replace("\n", "<br>")
            fragments.append(("text", after_esc))
        return fragments

    def _append_bubble(self, role: str, text: str):
        is_user = (role == "user")
        title = "You" if is_user else "Assistant"
        bg, border, text_color = self._derive_bubble_colors(is_user)
        cur = self.transcript.textCursor()
        cur.movePosition(QtGui.QTextCursor.End)
        cur.insertBlock()
        bfmt = QtGui.QTextBlockFormat()
        bfmt.setAlignment(QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft)
        cur.setBlockFormat(bfmt)
        tfmt = QtGui.QTextTableFormat()
        tfmt.setAlignment(QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft)
        tfmt.setBorder(0.9)
        tfmt.setBorderBrush(QtGui.QBrush(border))
        tfmt.setCellSpacing(0)
        tfmt.setCellPadding(8)
        tfmt.setWidth(QtGui.QTextLength(QtGui.QTextLength.PercentageLength, 85))
        table = cur.insertTable(1, 1, tfmt)
        cell = table.cellAt(0, 0)
        cfmt = cell.format().toTableCellFormat()
        cfmt.setBackground(QtGui.QBrush(bg))
        cell.setFormat(cfmt)
        ccur = cell.firstCursorPosition()
        title_fmt = QtGui.QTextCharFormat()
        title_fmt.setFontWeight(QtGui.QFont.Bold)
        tcol = QtGui.QColor(text_color)
        tcol.setAlpha(220)
        title_fmt.setForeground(QtGui.QBrush(tcol))
        ccur.insertText(title, title_fmt)
        ccur.insertBlock()
        fragments = self._render_message_fragments(text)
        for ftype, content in fragments:
            if ftype == "text":
                ccur.insertHtml(content)
            elif ftype == "codeblock":
                ccur.insertBlock()
                self._insert_code_block(ccur, content)
                ccur.insertBlock()
        self.transcript.moveCursor(QtGui.QTextCursor.End)
        cur2 = self.transcript.textCursor()
        cur2.insertBlock()

    def _append_user(self, text: str):
        self._append_bubble("user", text)

    def _append_assistant(self, text: str):
        self._append_bubble("assistant", text)

    def _append_sys(self, text: str):
        safe = QtGui.QTextDocumentFragment.fromPlainText(text).toHtml()
        self._append_html(f"<div style='color:#999;'><i>{safe}</i></div>")

    def _refresh_index_size(self):
        try:
            self.store.load()
            self.index_size_lbl.setText(f"Index: {self.store.size()} passages")
        except Exception:
            self.index_size_lbl.setText("Index: (error)")

    # ---------- Settings management ----------
    def _default_settings(self) -> dict:
        return {
            "base_url": DEFAULT_OLLAMA_BASE,
            "chat_model": DEFAULT_CHAT_MODEL,
            "embed_model": DEFAULT_EMBED_MODEL,
            "top_k": TOP_K,
            "sim_floor": SIM_FLOOR,
            "max_ctx": MAX_CONTEXT_CHARS,
            "temperature": CHAT_OPTIONS.get("temperature", 0.3),
            "top_p": CHAT_OPTIONS.get("top_p", 0.9),
            "use_rag": True,
            "store_dir": str(self.store_dir),
        }

    def _load_settings(self) -> dict:
        settings = QtCore.QSettings("ChiSurf", "Chato")
        d = self._default_settings()
        for k in list(d.keys()):
            v = settings.value(k, d[k])
            # QSettings returns strings; cast as needed
            if k in ("top_k", "max_ctx"):
                try:
                    v = int(v)
                except Exception:
                    v = d[k]
            elif k in ("sim_floor", "temperature", "top_p"):
                try:
                    v = float(v)
                except Exception:
                    v = d[k]
            elif k == "use_rag":
                v = str(v).lower() not in ("false", "0", "no", "")
            d[k] = v
        return d

    def _save_settings(self, values: dict) -> None:
        settings = QtCore.QSettings("ChiSurf", "Chato")
        for k, v in values.items():
            settings.setValue(k, v)

    def _apply_settings(self, values: dict) -> None:
        global TOP_K, SIM_FLOOR, MAX_CONTEXT_CHARS, CHAT_OPTIONS
        # Update RAG/decoding params
        TOP_K = int(values.get("top_k", TOP_K))
        SIM_FLOOR = float(values.get("sim_floor", SIM_FLOOR))
        MAX_CONTEXT_CHARS = int(values.get("max_ctx", MAX_CONTEXT_CHARS))
        # CHAT_OPTIONS is a dict; update keys
        CHAT_OPTIONS = dict(CHAT_OPTIONS)
        CHAT_OPTIONS["temperature"] = float(values.get("temperature", CHAT_OPTIONS.get("temperature", 0.3)))
        CHAT_OPTIONS["top_p"] = float(values.get("top_p", CHAT_OPTIONS.get("top_p", 0.9)))
        # RAG toggle
        self.use_rag_enabled = bool(values.get("use_rag", True))
        # Store dir and index object
        new_store = pathlib.Path(values.get("store_dir", str(self.store_dir))).resolve()
        try:
            new_store.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.store_dir = new_store
        self.store = RagIndex(self.store_dir)
        # Ollama connection
        base = values.get("base_url", DEFAULT_OLLAMA_BASE)
        chat = values.get("chat_model", DEFAULT_CHAT_MODEL)
        emb = values.get("embed_model", DEFAULT_EMBED_MODEL)
        # Update existing client in place
        self.ollama.base_url = str(base).rstrip("/")
        self.ollama.chat_model = chat
        self.ollama.embed_model = emb
        # refresh UI index label
        self._refresh_index_size()

    def _init_and_apply_settings(self) -> None:
        vals = self._load_settings()
        self._apply_settings(vals)

    def on_settings(self):
        current = self._load_settings()
        dlg = SettingsDialog(self, initial=current)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            vals = dlg.get_values()
            # Fill defaults if fields left empty
            defs = self._default_settings()
            for k, v in list(vals.items()):
                if (isinstance(v, str) and v == "") and k in defs:
                    vals[k] = defs[k]
            self._save_settings(vals)
            self._apply_settings(vals)
            self._append_sys("Settings applied.")

    # ---------- Actions ----------
    def on_pick_rag_folder(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            sel = dlg.selectedFiles()
            if sel:
                self._rag_folder = pathlib.Path(sel[0])
                self.rag_folder_lbl.setText(str(self._rag_folder))

    def on_build_rag(self):
        if not self._rag_folder or not self._rag_folder.exists():
            QtWidgets.QMessageBox.warning(self, "Chato", "Please choose a folder to index.")
            return
        # Prevent concurrent builds
        if getattr(self, "_rag_proc", None) is not None:
            QtWidgets.QMessageBox.information(self, "Chato", "RAG build already in progress.")
            return

        self._append_sys(f"Indexing (separate process): {self._rag_folder} …")
        self.build_rag_btn.setEnabled(False)

        # Launch subprocess using QProcess to isolate potential crashes
        proc = QtCore.QProcess(self)
        self._rag_proc = proc
        # Use current Python interpreter
        python_exe = sys.executable
        # Module path for the worker
        module = "chisurf.plugins.chato.rag_build_process"
        store_dir = str(self.store_dir)
        args = [
            "-m", module,
            "--root", str(self._rag_folder),
            "--store", store_dir,
        ]
        # Capture stdout for progress
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        def on_ready_read():
            try:
                data = bytes(proc.readAllStandardOutput()).decode(errors="ignore")
            except Exception:
                return
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("PROGRESS:"):
                    self._append_sys(line[len("PROGRESS:"):].strip())
                elif line.startswith("DONE:"):
                    n = line[len("DONE:"):].strip()
                    self._append_sys(f"Indexed {n} passages.")
                elif line.startswith("ERROR:"):
                    err_msg = line[len("ERROR:"):].strip()
                    self._append_sys(f"Index error: {err_msg}")
                else:
                    self._append_sys(line)

        def on_finished(exit_code, exit_status):
            try:
                if exit_code != 0:
                    self._append_sys(f"RAG build process exited with code {exit_code}.")
                self._refresh_index_size()
            finally:
                self.build_rag_btn.setEnabled(True)
                self._rag_proc = None

        proc.readyReadStandardOutput.connect(on_ready_read)
        proc.finished.connect(on_finished)

        # Start process
        proc.start(python_exe, args)
        if not proc.waitForStarted(3000):
            self._append_sys("Failed to start RAG build process.")
            self.build_rag_btn.setEnabled(True)
            self._rag_proc = None

    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._append_user(text)

        use_rag = bool(getattr(self, "use_rag_enabled", True))
        try:
            self.store.load()
            use_rag = use_rag and (self.store.size() > 0)
        except Exception:
            pass

        worker = ChatWorker(self.ollama, self.store, list(self.history), text, use_rag)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _finish(reply: str):
            try:
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                self._append_assistant(reply)
            finally:
                worker.deleteLater()
                thread.quit()
                thread.wait()
                thread.deleteLater()

        def _error(err: str):
            _finish(f"[Error] {err}")

        worker.finished.connect(lambda r: _finish(r))
        worker.error.connect(lambda e: _error(e))
        thread.start()


def create_dock(parent=None):
    return ChiChatDock(parent=parent)
