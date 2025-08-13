from PyQt5 import QtWidgets, QtCore, QtGui
import json
import httpx
import html
import sys
import subprocess
import time
import re
import traceback

BACKEND_URL = "http://127.0.0.1:8000"


class EnterAwarePlainTextEdit(QtWidgets.QPlainTextEdit):
    """
    Chat-style input:
    - Enter / Ctrl+Enter sends
    - Shift+Enter inserts newline
    - Ctrl+L clears input
    """
    sendRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setPlaceholderText("Type a message… (Enter to send, Shift+Enter for newline)")
        self.setTabChangesFocus(True)
        self._max_lines = 8
        self.document().contentsChanged.connect(self._auto_resize)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if e.modifiers() & QtCore.Qt.ShiftModifier:
                # newline
                return super().keyPressEvent(e)
            # Ctrl+Enter or plain Enter -> send
            e.accept()
            self.sendRequested.emit()
            return
        if e.key() == QtCore.Qt.Key_L and (e.modifiers() & QtCore.Qt.ControlModifier):
            self.clear()
            e.accept()
            return
        return super().keyPressEvent(e)

    def _auto_resize(self):
        # Grow with content up to max lines
        doc = self.document()
        m = self.contentsMargins()
        h = int(doc.size().height() + self.frameWidth() * 2 + m.top() + m.bottom())
        # Estimate line height
        fm = QtGui.QFontMetrics(self.font())
        line = fm.lineSpacing()
        max_h = int(line * self._max_lines + self.frameWidth() * 2 + m.top() + m.bottom())
        self.setFixedHeight(min(max(h, line * 2 + 14), max_h))


class ChiSurfChatDock(QtWidgets.QDockWidget):

    # Shared, process-wide resources to ensure single assistant backend/session
    _shared_http = None
    _shared_backend_proc = None
    _app_quit_hooked = False

    def changeEvent(self, e: QtCore.QEvent):
        super().changeEvent(e)
        if e.type() == QtCore.QEvent.PaletteChange:
            # Re-render the transcript to adopt the new palette/theme
            try:
                self._reload_entire_transcript()
            except Exception:
                pass

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

        # Tints for user/assistant bubbles. Keep subtle in light theme,
        # make a bit stronger in dark theme for readability.
        user_tint = QtGui.QColor(90, 150, 255)
        asst_tint = QtGui.QColor(200, 200, 210)
        t_user = 0.26 if dark else 0.14
        t_asst = 0.18 if dark else 0.08
        bg = self._mix(base, user_tint if is_user else asst_tint, t_user if is_user else t_asst)

        # Borders slightly lighter in dark theme, slightly darker in light theme
        if dark:
            border = self._mix(bg, QtGui.QColor(255, 255, 255), 0.22)
        else:
            border = self._mix(bg, QtGui.QColor(0, 0, 0), 0.12)

        return bg, border, text

    def _insert_code_block(self, parent_cursor: QtGui.QTextCursor, code_text: str):
        # Slightly tinted background for code blocks
        dark = self._is_dark_palette()
        bg_base, border, _ = self._derive_bubble_colors(is_user=False)
        if dark:
            # In dark theme: lighten slightly vs bubble to improve contrast
            bg = self._mix(bg_base, QtGui.QColor(255, 255, 255), 0.08)
        else:
            # In light theme: darken subtly
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

    def __init__(self, parent=None):
        super().__init__("ChiSurf Assistant", parent)
        self.setObjectName("ChiSurfChatDock")

        # persistent session + cid; backend/session are shared across all docks
        self._cid = None

        self._confirm_pending = False
        self._last_payload = None

        # Ensure the app-exit hook is installed once across the process
        try:
            self.register_app_quit_hook()
        except Exception:
            pass

        # ---------- UI ----------
        w = QtWidgets.QWidget(self)
        self.setWidget(w)

        self.transcript = QtWidgets.QTextBrowser(w)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setReadOnly(True)
        # Minimal context menu addition for copy support
        self.transcript.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.transcript.customContextMenuRequested.connect(self._on_transcript_context_menu)
        self._init_transcript_theme()

        self.input = EnterAwarePlainTextEdit(w)
        self.send_btn = QtWidgets.QPushButton("Send", w)
        self.agent_mode = QtWidgets.QCheckBox("Agent Mode", w)
        self.confirm_btn = QtWidgets.QPushButton("Confirm Action", w)
        self.confirm_btn.setVisible(False)
        self.clear_btn = QtWidgets.QPushButton("New Chat", w)
        self.build_rag_btn = QtWidgets.QPushButton("Build RAG", w)

        # Status bar
        self.status_lbl = QtWidgets.QLabel("—", w)
        self.status_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.agent_mode)
        btn_row.addStretch(1)
        btn_row.addWidget(self.build_rag_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.confirm_btn)
        btn_row.addWidget(self.send_btn)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(self.transcript)
        layout.addWidget(self.input)
        layout.addLayout(btn_row)
        layout.addWidget(self.status_lbl)

        # Signals
        self.send_btn.clicked.connect(self.on_send)
        self.confirm_btn.clicked.connect(self.on_confirm)
        self.clear_btn.clicked.connect(self.on_new_chat)
        self.build_rag_btn.clicked.connect(self.on_build_rag)
        self.input.sendRequested.connect(self.on_send)

        # Session; defer history load until widget is shown (palette ready)
        self._ensure_session()
        self._restore_cid()
        self._history_loaded = False

        # Nice default sizes
        self.transcript.setMinimumHeight(240)
        self.input.setFixedHeight(60)

        # health polling
        self._health_timer = QtCore.QTimer(self)
        self._health_timer.setInterval(10000)  # 10s
        self._health_timer.timeout.connect(self._poll_health_once)
        self._poll_health_once()
        self._health_timer.start()

    # ---------- Lifecycle ----------
    def closeEvent(self, event):
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'removeDockWidget'):
                try:
                    parent.removeDockWidget(self)
                except Exception:
                    pass
            # Only clean up UI connections and state; do NOT shutdown backend here
            self.spin_down()
        finally:
            event.accept()
            self.deleteLater()

    def showEvent(self, event: QtGui.QShowEvent):
        try:
            if not getattr(self, "_history_loaded", False):
                self._reload_entire_transcript()
                self._history_loaded = True
        except Exception:
            pass
        finally:
            super().showEvent(event)

    def spin_down(self):
        try:
            for sig in (self.send_btn.clicked, self.confirm_btn.clicked, self.build_rag_btn.clicked, self.input.sendRequested):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            self.confirm_btn.setVisible(False)
            self._confirm_pending = False
            self._last_payload = None
            try:
                self.transcript.clear()
            except Exception:
                pass
        finally:
            # Do not shutdown shared backend or HTTP session here; they live until app exit
            pass

    # ---------- Theming & Rendering ----------
    def _init_transcript_theme(self):
        self.transcript.clear()
        f = self.transcript.font()
        f.setPointSize(10)
        self.transcript.setFont(f)

    def _reload_entire_transcript(self):
        """Clear and reload the transcript so bubble colors match the current palette."""
        try:
            self.transcript.clear()
            self._init_transcript_theme()
            self._load_history()
        except Exception:
            pass

    def _append_bubble(self, role: str, text: str):
        is_user = (role == "user")
        title = "You" if is_user else "Assistant"

        bg, border, text_color = self._derive_bubble_colors(is_user)

        cur = self.transcript.textCursor()
        cur.movePosition(QtGui.QTextCursor.End)
        cur.insertBlock()

        # Align the bubble
        bfmt = QtGui.QTextBlockFormat()
        bfmt.setAlignment(QtCore.Qt.AlignRight if is_user else QtCore.Qt.AlignLeft)
        cur.setBlockFormat(bfmt)

        # Outer bubble as a table
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

        # Title
        title_fmt = QtGui.QTextCharFormat()
        title_fmt.setFontWeight(QtGui.QFont.Bold)
        tcol = QtGui.QColor(text_color)
        tcol.setAlpha(220)
        title_fmt.setForeground(QtGui.QBrush(tcol))
        ccur.insertText(title, title_fmt)
        ccur.insertBlock()

        # Insert fragments
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

    def _render_message_fragments(self, text: str):
        """
        Split message into [(type, content)] where type is 'text' or 'codeblock'.
        Inline code is converted into <code> spans in the 'text' fragments.
        """
        import html as _html, re as _re
        fragments = []
        s = text

        # Regex to split out ```fenced code``` blocks
        fence = _re.compile(r"```(.*?)```", _re.DOTALL)
        pos = 0
        for m in fence.finditer(s):
            before = s[pos:m.start()]
            if before.strip():
                # inline code: `code`
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

    def _render_message_html(self, text: str) -> str:
        """Convert basic markdown-ish text to Qt-safe HTML (no CSS)."""
        import html as _html, re as _re
        s = text

        # Pull out ```fenced code``` first
        fence = _re.compile(r"```(.*?)```", _re.DOTALL)
        blocks = []

        def repl_fence(m):
            block = m.group(1)
            parts = block.split("\n", 1)
            body = parts[1] if len(parts) == 2 and len(parts[0]) < 20 else block
            blocks.append(body)
            return f"[[[FENCE_{len(blocks) - 1}]]]"

        s = fence.sub(repl_fence, s)

        # Escape the rest
        s = _html.escape(s)

        # Inline code: `code`
        s = _re.sub(r"`([^`]+)`", lambda m: f"<code>{_html.escape(m.group(1))}</code>", s)

        # Restore code blocks as <pre> (Qt supports <pre> nicely)
        for i, body in enumerate(blocks):
            body_esc = _html.escape(body)
            s = s.replace(f"[[[FENCE_{i}]]]", f"<pre>{body_esc}</pre>")

        # Newlines to <br> (outside <pre>)
        s = s.replace("\n", "<br>")
        return s

    def _on_transcript_context_menu(self, pos: QtCore.QPoint):
        menu = self.transcript.createStandardContextMenu()
        menu.addSeparator()
        act_copy = QtWidgets.QAction("Copy selection", self.transcript)
        act_copy.triggered.connect(self.transcript.copy)
        menu.addAction(act_copy)
        menu.exec_(self.transcript.mapToGlobal(pos))

    def append_text(self, title: str, text: str):
        # Keep this wrapper for error paths; route to bubble renderer
        role = "assistant" if title.lower().startswith("assistant") else \
            "user" if title.lower().startswith("you") else "assistant"
        self._append_bubble(role, text)

    # ---------- Settings ----------
    def _settings(self):
        return QtCore.QSettings("ChiSurf", "AssistantChat")

    def _restore_cid(self):
        cid = self._settings().value("conversation_id", type=str)
        if cid:
            self._cid = cid

    def _save_cid(self, cid: str):
        self._cid = cid
        self._settings().setValue("conversation_id", cid)

    # ---------- Backend process mgmt ----------
    @classmethod
    def _start_backend(cls):
        if cls._shared_backend_proc is not None and cls._shared_backend_proc.poll() is None:
            return True
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "chisurf.plugins.chat.server:APP",
                "--host", "127.0.0.1",
                "--port", "8000",
            ]
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            cls._shared_backend_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
            )
            time.sleep(0.8)
            return True
        except Exception as e:
            try:
                print(f"Assistant backend failed to start: {e}")
            except Exception:
                pass
            return False

    @classmethod
    def _shutdown_backend(cls):
        proc = getattr(cls, "_shared_backend_proc", None)
        if not proc:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        finally:
            cls._shared_backend_proc = None

    @classmethod
    def register_app_quit_hook(cls):
        if cls._app_quit_hooked:
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        try:
            app.aboutToQuit.connect(cls._on_app_quit)
            cls._app_quit_hooked = True
        except Exception:
            pass

    @classmethod
    def _on_app_quit(cls):
        # Gracefully close shared resources when the application exits
        try:
            if cls._shared_http is not None:
                try:
                    cls._shared_http.close()
                except Exception:
                    pass
                cls._shared_http = None
        finally:
            try:
                cls._shutdown_backend()
            except Exception:
                pass

    # ---------- HTTP ----------
    @classmethod
    def _ensure_session(cls):
        if cls._shared_http is None:
            cls._shared_http = httpx.Client(timeout=50.0, follow_redirects=True, trust_env=True)

    def _http_post(self, path: str, json_payload: dict):
        self._ensure_session()
        url = BACKEND_URL + path
        headers = {}
        if self._cid:
            headers["X-Conversation-ID"] = self._cid
        try:
            r = self._shared_http.post(url, json=json_payload, headers=headers)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}: {r.text[:500]}"
            return r.json(), None
        except Exception as e:
            msg = str(e)
            if "WinError 10061" in msg or "Connection refused" in msg or "ConnectError" in msg:
                if self._start_backend():
                    try:
                        time.sleep(0.7)
                        r2 = self._shared_http.post(url, json=json_payload, headers=headers)
                        if r2.status_code != 200:
                            return None, f"HTTP {r2.status_code}: {r2.text[:500]}"
                        return r2.json(), None
                    except Exception as e2:
                        return None, f"Backend start attempted but request failed: {e2}"
            return None, msg

    def _http_get(self, path: str, params: dict = None):
        self._ensure_session()
        url = BACKEND_URL + path
        headers = {}
        if self._cid:
            headers["X-Conversation-ID"] = self._cid
        try:
            r = self._shared_http.get(url, params=params, headers=headers)
            if r.status_code != 200:
                return None, f"HTTP {r.status_code}: {r.text[:500]}"
            return r.json(), None
        except Exception as e:
            return None, str(e)

    # ---------- Actions ----------
    def on_build_rag(self):
        self.build_rag_btn.setEnabled(False)
        try:
            data, err = self._http_post("/rag/build", {})
        finally:
            self.build_rag_btn.setEnabled(True)
        if err:
            # If the backend doesn't have the endpoint yet, try local fallback build
            if isinstance(err, str) and ("HTTP 404" in err or "Not Found" in err):
                try:
                    from chisurf.plugins.chat.index_docs import build_index, DEFAULT_STORE
                    import pathlib as _pl
                    out_dir = _pl.Path(DEFAULT_STORE)
                    meta = build_index(out_dir)
                    cnt = meta.get("count", 0)
                    dur = meta.get("build_seconds")
                    store_dir = str(out_dir)
                    dstr = f"{dur:.2f}s" if isinstance(dur, (int, float)) else "?s"
                    msg = (
                        "Backend missing /rag/build endpoint; built locally.\n"
                        f"RAG store built: {cnt} chunks in {dstr}\nStore: {store_dir}"
                    )
                    self._append_bubble("assistant", msg)
                    return
                except Exception as _e:
                    tb = "".join(traceback.format_exc())
                    self._append_bubble("assistant", f"RAG build error (fallback failed): {_e}\n\n{tb}")
                    return
            self._append_bubble("assistant", f"RAG build error: {err}")
            return
        meta = (data or {}).get("meta") or {}
        if meta:
            cnt = meta.get("count", 0)
            dur = meta.get("build_seconds")
            store_dir = meta.get("store_dir") or "(unknown)"
            if isinstance(dur, (int, float)):
                dstr = f"{dur:.2f}s"
            else:
                dstr = "?s"
            msg = f"RAG store built: {cnt} chunks in {dstr}\nStore: {store_dir}"
        else:
            msg = "RAG build done."
        self._append_bubble("assistant", msg)

    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._append_bubble("user", text)
        self.send_btn.setEnabled(False)
        self.confirm_btn.setVisible(False)
        self._confirm_pending = False
        self._last_payload = None

        if self.agent_mode.isChecked():
            payload = {
                "goal": text,
                "conversation_id": self._cid or ""
            }
            path = "/agent"  # requires server-side implementation/stub
        else:
            payload = {
                "message": text,
                "conversation_id": self._cid or ""
            }
            path = "/chat"

        self._last_payload = (path, payload)
        data, err = self._http_post(path, payload)
        self.send_btn.setEnabled(True)
        if err:
            self._append_bubble("assistant", f"Error: {err}")
            return
        self._handle_response(data)

    def on_confirm(self):
        if not self._last_payload:
            return
        path, payload = self._last_payload
        payload2 = dict(payload)
        payload2["confirm_destructive"] = True
        self.send_btn.setEnabled(False)
        data, err = self._http_post(path, payload2)
        self.send_btn.setEnabled(True)
        if err:
            self._append_bubble("assistant", f"Error: {err}")
            return
        self._confirm_pending = False
        self.confirm_btn.setVisible(False)
        self._handle_response(data)

    def on_new_chat(self):
        old = self._cid
        if old:
            self._http_post("/reset", {"conversation_id": old})
        self._save_cid("")
        self.transcript.clear()
        self._init_transcript_theme()
        self._append_bubble("assistant", "Started a new conversation.")

    # ---------- History & responses ----------
    def _load_history(self):
        if not self._cid:
            return
        data, err = self._http_get(f"/history/{self._cid}")
        if err or not data:
            return
        for turn in data.get("history", []):
            role = turn.get("role", "")
            content = turn.get("content", "")
            self._append_bubble(role, content)

    def _handle_response(self, data: dict):
        cid = data.get("conversation_id")
        if isinstance(cid, str) and cid:
            self._save_cid(cid)

        if "reply" in data:
            self._append_bubble("assistant", data.get("reply", ""))
            tool = data.get("tool")
            if tool:
                self._append_bubble("assistant", "Tool\n" + json.dumps(tool, indent=2))
            if data.get("tool_result") is not None:
                self._append_bubble("assistant", "Tool Result\n" + json.dumps(data.get("tool_result"), indent=2))
            if data.get("tool_error"):
                err = data.get("tool_error")
                self._append_bubble("assistant", "Tool Error\n" + str(err))
                if "requires confirmation" in (err or "").lower():
                    self.confirm_btn.setVisible(True)
                    self._confirm_pending = True
        elif "trace" in data:
            for step in data.get("trace", []):
                self._append_bubble("assistant", step.get("assistant", ""))
                if step.get("tool"):
                    self._append_bubble("assistant", "Tool\n" + json.dumps(step.get("tool"), indent=2))
                if step.get("observation"):
                    obs = step.get("observation")
                    self._append_bubble("assistant", "Observation\n" + json.dumps(obs, indent=2))
                    err = (obs or {}).get("error")
                    if isinstance(err, str) and "requires confirmation" in err.lower():
                        self.confirm_btn.setVisible(True)
                        self._confirm_pending = True
        else:
            self._append_bubble("assistant", json.dumps(data, indent=2))

    # ---------- Health ----------
    def _poll_health_once(self):
        try:
            data, err = self._http_get("/health")
            if err or not data:
                self.status_lbl.setText("Health: backend unreachable")
                return
            rag_dir = data.get("rag_store_dir") or ""
            rag_status = data.get("rag_store", "missing")
            txt = f"Ollama: {data.get('ollama','?')}  |  DB: {data.get('db','?')}  |  RAG: {rag_status}"
            if rag_dir:
                txt += f"  ({rag_dir})"
            self.status_lbl.setText(txt)
        except Exception:
            self.status_lbl.setText("Health: error")

# ---- factory ----
def create_dock(parent=None) -> ChiSurfChatDock:
    # Ensure the backend shutdown is tied to app exit
    try:
        ChiSurfChatDock.register_app_quit_hook()
    except Exception:
        pass

    # Keep a single dock per main window (parent). Reuse if exists.
    if parent is not None:
        try:
            existing = getattr(parent, "_assistant_chat_dock", None)
            if isinstance(existing, ChiSurfChatDock):
                try:
                    existing.show()
                    existing.raise_()
                    existing.activateWindow()
                except Exception:
                    pass
                return existing
        except Exception:
            pass

    dock = ChiSurfChatDock(parent)
    if parent is not None:
        try:
            setattr(parent, "_assistant_chat_dock", dock)
            # Clear reference when destroyed
            dock.destroyed.connect(lambda: setattr(parent, "_assistant_chat_dock", None))
        except Exception:
            pass
    return dock
