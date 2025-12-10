from qtpy import QtWidgets

from . import NodeEditorWidget


def _create_editor_with_json(parent, json_path: str, title: str) -> NodeEditorWidget:
    """Helper to create a NodeEditorWidget and load a JSON graph if present."""

    editor = NodeEditorWidget(parent)
    try:
        from pathlib import Path

        base = Path(__file__).resolve().parent
        p = Path(json_path)
        if not p.is_absolute():
            p = base / p
        if p.is_file():
            try:
                editor.load_graph_from_file(str(p))
            except Exception:
                pass
    except Exception:
        pass

    editor.setWindowTitle(title)
    return editor


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)

    if NodeEditorWidget is None:
        w = QtWidgets.QWidget()
        w.setWindowTitle("Node Editor (legacy widget not found)")
        w.resize(600, 200)
        w.show()
        sys.exit(app.exec_())

    # Use a tabbed window hosting multiple example scenes so that large
    # example graphs can be split into focused views.
    tabs = QtWidgets.QTabWidget()
    tabs.setWindowTitle("Node Editor Examples")

    # Tab 1: full example graph (original JSON)
    full_editor = NodeEditorWidget(tabs)
    tabs.addTab(full_editor, "Full example")

    # Tab 2: math-only example (no chinet nodes)
    math_editor = _create_editor_with_json(tabs, "examples/example_math.json", "Math example")
    tabs.addTab(math_editor, "Math example")

    # Tab 3: chinet-focused example (PT nodes only)
    chinet_editor = _create_editor_with_json(tabs, "examples/example_chinet.json", "Chinet example")
    tabs.addTab(chinet_editor, "Chinet example")

    tabs.resize(1100, 650)
    tabs.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
