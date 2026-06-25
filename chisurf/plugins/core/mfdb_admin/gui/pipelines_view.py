"""Self-contained pipeline/workflow view (PRD-22).

A thin Qt widget over the ``mfdb.pipelines.*`` RPC handlers (PRD-23: logic in the
backend, view only renders): list stored pipeline definitions, and on selection show
the pipeline's structure (nodes + typed wiring) and its recorded runs (each grouping a
chain of operations). Read-only — pipelines are authored in code / via the runner, not
hand-built here. Standalone, like ``StudiesView`` / ``ReagentLotsView``.
"""

from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

_PIPE_COLUMNS = ("Name", "Version", "Description", "Pipeline ID")
_PIPE_KEYS = ("name", "version", "description", "pipeline_id")
_RUN_COLUMNS = ("Run", "Status", "Ops", "Run ID")
_RUN_KEYS = ("name", "status", "operation_count", "pipeline_run_id")


class PipelinesView(QtWidgets.QWidget):
    """List pipelines + show the selected one's structure and runs."""

    def __init__(self, client: Any, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._client = client

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Pipelines:"))
        self.pipeline_table = QtWidgets.QTableWidget(0, len(_PIPE_COLUMNS))
        self.pipeline_table.setHorizontalHeaderLabels(list(_PIPE_COLUMNS))
        self.pipeline_table.horizontalHeader().setStretchLastSection(True)
        self.pipeline_table.itemSelectionChanged.connect(self._on_select)
        layout.addWidget(self.pipeline_table)

        layout.addWidget(QtWidgets.QLabel("Nodes (transformer invocations):"))
        self.node_table = QtWidgets.QTableWidget(0, 2)
        self.node_table.setHorizontalHeaderLabels(["Node", "Operation type"])
        self.node_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.node_table)

        layout.addWidget(QtWidgets.QLabel("Wiring (typed edges):"))
        self.edge_table = QtWidgets.QTableWidget(0, 2)
        self.edge_table.setHorizontalHeaderLabels(["From", "To"])
        self.edge_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.edge_table)

        layout.addWidget(QtWidgets.QLabel("Runs (recorded operation chains):"))
        self.run_table = QtWidgets.QTableWidget(0, len(_RUN_COLUMNS))
        self.run_table.setHorizontalHeaderLabels(list(_RUN_COLUMNS))
        self.run_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.run_table)

        self.message_label = QtWidgets.QLabel("")
        layout.addWidget(self.message_label)

        self.refresh()

    # -- data plumbing (via the client) --------------------------------------

    def refresh(self) -> None:
        pipelines = self._client.list_pipelines() or []
        self.pipeline_table.setRowCount(len(pipelines))
        for row, p in enumerate(pipelines):
            for col, key in enumerate(_PIPE_KEYS):
                value = p.get(key)
                self.pipeline_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem("" if value is None else str(value))
                )
        self.pipeline_table.resizeColumnsToContents()
        for table in (self.node_table, self.edge_table, self.run_table):
            table.setRowCount(0)

    def _selected_pipeline_id(self) -> str:
        items = self.pipeline_table.selectedItems()
        if not items:
            return ""
        return self.pipeline_table.item(items[0].row(), 3).text()

    def _on_select(self) -> None:
        pipeline_id = self._selected_pipeline_id()
        if not pipeline_id:
            return
        detail = self._client.get_pipeline(pipeline_id) or {}
        if detail.get("error"):
            self.message_label.setText(detail["error"])
            return

        nodes = detail.get("nodes", [])
        self.node_table.setRowCount(len(nodes))
        for row, n in enumerate(nodes):
            self.node_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(n.get("name", ""))))
            self.node_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(n.get("operation_type", "")))
            )
        self.node_table.resizeColumnsToContents()

        edges = detail.get("edges", [])
        self.edge_table.setRowCount(len(edges))
        for row, e in enumerate(edges):
            self.edge_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(f"{e.get('source')}.{e.get('source_port')}")
            )
            self.edge_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{e.get('target')}.{e.get('target_port')}")
            )
        self.edge_table.resizeColumnsToContents()

        runs = self._client.list_pipeline_runs(pipeline_id) or []
        self.run_table.setRowCount(len(runs))
        for row, r in enumerate(runs):
            for col, key in enumerate(_RUN_KEYS):
                value = r.get(key)
                self.run_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem("" if value is None else str(value))
                )
        self.run_table.resizeColumnsToContents()
