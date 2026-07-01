"""GUI-free backing model for the AI Settings AutoForm view.

The model holds the live state for a single provider and exposes the zero/one-arg
methods the ``ai_settings.view.json`` sections invoke (choice ``call`` and
``button_row`` actions). It contains no Qt imports so it stays headlessly
testable; the tool wires :attr:`AISettingsModel.on_change` to refresh the form.
"""

from __future__ import annotations

import logging
import pathlib
import typing
import webbrowser

from chisurf.core.dataspec import load_view_spec
from chisurf.core.settings import ai_settings
from chisurf.core.settings.ai_settings import DEFAULT_PROVIDER_SETTINGS, PROVIDERS

_LOG = logging.getLogger(__name__)

_VIEW = pathlib.Path(__file__).with_name("ai_settings.view.json")

#: Editable fields that are auto-persisted the moment they change. ``provider`` is
#: excluded on purpose: switching it *loads* another provider rather than editing
#: the current one, so it must not save the current fields under the new key.
_PERSIST_FIELDS = frozenset(
    {"base_url", "api_key", "text_model", "image_model", "temperature", "top_p", "max_tokens"}
)


class AISettingsModel:
    """Editable bag bound to the AI settings AutoForm scheme.

    Field edits auto-persist (:meth:`_persist`) so a pasted token is never lost,
    and pasting a token also verifies the endpoint (:meth:`apply_token`).
    """

    def __init__(self) -> None:
        # Guards read by __setattr__; set before any persisted attribute exists.
        self._ready = False
        self._loading = False
        self._saving = False
        #: Optional refresh callback, set by the tool (kept Qt-free here).
        self.on_change: typing.Callable[[], None] | None = None
        #: Optional off-thread runner injected by the tool: ``runner(work, done)``
        #: runs ``work()`` on a background thread and calls ``done(result)`` back on
        #: the GUI thread. ``None`` runs synchronously (headless/tests).
        self.async_runner: (
            typing.Callable[
                [typing.Callable[[], typing.Any], typing.Callable[[typing.Any], None]], None
            ]
            | None
        ) = None
        self.status_html: str = ""
        self._text_models: list[str] = []
        self._image_models: list[str] = []
        # Populate every attribute from the currently selected provider.
        self._apply_settings(ai_settings.get_api_settings())
        self._ready = True

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """Set the attribute, then auto-persist when an editable field changes."""
        object.__setattr__(self, name, value)
        if (
            name in _PERSIST_FIELDS
            and getattr(self, "_ready", False)
            and not getattr(self, "_loading", False)
            and not getattr(self, "_saving", False)
        ):
            self._persist(silent=True)

    # -- view -----------------------------------------------------------------
    def view_spec(self):
        """Return the parsed ``ai_settings.view.json`` model view."""
        return load_view_spec(_VIEW)

    # -- option sources (editable combos) -------------------------------------
    def available_text_models(self) -> list[str]:
        """Return the fetched text-capable model ids for the text-model combo."""
        return list(self._text_models)

    def available_image_models(self) -> list[str]:
        """Return the fetched image-capable model ids for the image-model combo."""
        return list(self._image_models)

    # -- status ---------------------------------------------------------------
    def status_source(self) -> str:
        """Live HTML for the status ``info`` section."""
        return self.status_html

    # -- provider -------------------------------------------------------------
    def set_provider(self, provider: str) -> None:
        """Load the saved settings for *provider* into the visible fields."""
        provider = ai_settings.normalize_provider_key(provider)
        self._apply_settings(ai_settings.get_api_settings(provider))
        # Fetched model lists belong to the previous endpoint — drop them.
        self._text_models = []
        self._image_models = []
        self.status_html = ""
        self._notify()

    def sign_in(self) -> None:
        """Open the selected provider's API-key console in the browser."""
        for _display, (key, _url, api_url, _env) in PROVIDERS.items():
            if key == self.provider and api_url:
                webbrowser.open(api_url)
                self._set_status(f"Opened {api_url} in browser", "blue")
                return
        self._set_status("No browser sign-in available for this provider", "orange")

    # -- network (runs off the UI thread when an async_runner is injected) -----
    def fetch_models(self) -> None:
        """Query the endpoint's ``/models`` list and split it by capability."""
        base_url = (self.base_url or "").strip()
        if not base_url:
            self._set_status("Enter a base URL first.", "red")
            return
        self._set_status("Fetching models…", "blue")
        key = (self.api_key or "").strip()
        provider = self.provider

        def work():
            try:
                return ("ok", self._get_models(base_url, key))
            except Exception as exc:  # network / parse errors are user-facing
                return ("error", str(exc))

        def done(result):
            kind, payload = result
            if kind == "error":
                self._set_status(f"Failed: {payload}", "red")
                return
            model_data = payload.get("data", []) if isinstance(payload, dict) else []
            text_models, image_models = ai_settings.split_models_by_capability(
                model_data, provider=provider
            )
            all_models = sorted(
                {
                    str((m.get("id") or m.get("name") or "") if isinstance(m, dict) else m).strip()
                    for m in model_data
                }
                - {""}
            )
            self._text_models = text_models or all_models
            self._image_models = image_models
            self._set_status(
                f"Found {len(self._text_models)} text and {len(self._image_models)} image models.",
                "green",
            )

        self._run(work, done)

    def test_connection(self) -> None:
        """Check the endpoint answers a ``/models`` request."""
        base_url = (self.base_url or "").strip()
        if not base_url:
            self._set_status("No base URL provided.", "red")
            return
        self._set_status("Testing connection…", "blue")
        key = (self.api_key or "").strip()

        def work():
            try:
                self._get_models(base_url, key)
                return ("Connection successful!", "green")
            except Exception as exc:
                return (f"Connection failed: {exc}", "red")

        self._run(work, lambda result: self._set_status(*result))

    def apply_token(self, value: str | None = None) -> None:
        """Paste-and-go: the token is already auto-saved, so just verify it.

        Wired to the API-key field's ``call`` so entering/pasting a key (on
        focus-out or Enter) immediately tests the endpoint — no Save/Test clicks.
        """
        self.test_connection()

    # -- persistence ----------------------------------------------------------
    def save(self) -> None:
        """Persist the current provider's settings and report the outcome."""
        ok = self._persist(silent=False)
        if not ok:
            return

    def reset(self) -> None:
        """Reset the current provider's fields to their defaults (and persist)."""
        defaults = DEFAULT_PROVIDER_SETTINGS.get(self.provider, DEFAULT_PROVIDER_SETTINGS["openai"])
        self._apply_settings({**defaults, "provider": self.provider})
        self._text_models = []
        self._image_models = []
        self._persist(silent=True)
        self._set_status("Settings reset to defaults.", "blue")

    def _persist(self, silent: bool = False) -> bool:
        """Write the current provider's fields to the settings JSON.

        ``silent`` suppresses the status message (used by auto-save on every field
        change); the explicit Save button uses ``silent=False`` for feedback. The
        ``_saving`` guard stops the normalized ``base_url`` write-back from
        re-triggering :meth:`__setattr__` auto-save.
        """
        self._saving = True
        try:
            base_url = (self.base_url or "").strip()
            if self.provider != "custom" and not base_url:
                base_url = DEFAULT_PROVIDER_SETTINGS.get(self.provider, {}).get("base_url", "")
            ok = ai_settings.save_api_settings(
                {
                    "provider": self.provider,
                    "base_url": base_url,
                    "api_key": (self.api_key or "").strip(),
                    "text_model": (self.text_model or "").strip(),
                    "image_model": (self.image_model or "").strip(),
                    "temperature": float(self.temperature),
                    "top_p": float(self.top_p),
                    "max_tokens": int(self.max_tokens),
                }
            )
            if ok and base_url != self.base_url:
                self.base_url = base_url  # show the resolved URL (guarded: no re-save)
        finally:
            self._saving = False
        if not silent:
            if ok:
                self._set_status("Settings saved.", "green")
            else:
                self._set_status("Failed to save settings.", "red")
        return ok

    # -- internals ------------------------------------------------------------
    def _run(
        self, work: typing.Callable[[], typing.Any], done: typing.Callable[[typing.Any], None]
    ) -> None:
        """Run ``work`` off-thread via the injected runner, else synchronously.

        ``work`` must be self-contained (no ``self`` mutation, no Qt); ``done``
        receives its result on the GUI thread and performs the state/status update.
        """
        runner = self.async_runner
        if callable(runner):
            runner(work, done)
        else:
            done(work())

    def _apply_settings(self, settings: typing.Mapping[str, typing.Any]) -> None:
        """Copy a settings mapping onto the field attributes (no auto-save)."""
        self._loading = True
        try:
            self.provider = ai_settings.normalize_provider_key(settings.get("provider"))
            self.base_url = str(settings.get("base_url", "") or "")
            self.api_key = str(settings.get("api_key", "") or "")
            self.text_model = str(settings.get("text_model", settings.get("model", "")) or "")
            self.image_model = str(settings.get("image_model", "") or "")
            self.temperature = float(settings.get("temperature", 0.3))
            self.top_p = float(settings.get("top_p", 0.9))
            self.max_tokens = int(settings.get("max_tokens", 4096))
        finally:
            self._loading = False

    @staticmethod
    def _get_models(base_url: str, api_key: str) -> dict:
        """GET ``{base_url}/models`` and return the parsed JSON (raises on error)."""
        import requests

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/") + "/models"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:100]}")
        return response.json()

    def _set_status(self, message: str, color: str = "black") -> None:
        """Record an HTML status message and refresh the view."""
        self.status_html = f"<span style='color: {color};'>{message}</span>"
        self._notify()

    def _notify(self) -> None:
        """Invoke the tool-supplied refresh callback, if any."""
        if callable(self.on_change):
            try:
                self.on_change()
            except Exception:  # pragma: no cover - defensive
                _LOG.warning("AISettingsModel.on_change failed", exc_info=True)
