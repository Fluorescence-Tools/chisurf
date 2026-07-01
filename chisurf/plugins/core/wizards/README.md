# Wizards

A two-panel hub for ChiSurf's guided wizards: pick a wizard on the left, use it
embedded on the right.

```
┌────────────────┬───────────────────────────────────────┐
│ 🔬 Anisotropy  │  <selected wizard embedded here>       │
│ 📋 Batch …     │                                        │
│                │                                        │
└────────────────┴───────────────────────────────────────┘
```

Each wizard is an embeddable `QWidget` referenced by dotted path in the Qt-free
`core/registry.py`; the hub resolves and constructs it lazily on first selection.
Adding a wizard is one `WizardEntry` in `default_wizards()`.

## Architecture (new plugin standard)

```
wizards/
  manifest.json        plugin metadata + gui entrypoint
  core/registry.py     Qt-free catalogue of embeddable wizards (WizardEntry)
  gui/tool.py          WizardHub — selector list + lazy stacked embed
  test/                headless tests (registry + manifest)
```
