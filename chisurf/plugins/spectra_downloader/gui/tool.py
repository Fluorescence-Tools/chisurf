"""Spectra tool — the spectra downloader & browser on the shared navigation shell.

Uses :class:`chisurf.gui.widgets.navigation.NavigationPanelTool` (the same shell
as Settings, FCS Tools, Burst/Decay Analysis, …): a left navigation list and the
selected panel on the right. Panels reuse existing widgets — the AutoForm
overview, the data browser, and the download/push panel — rather than
duplicating them.
"""

from __future__ import annotations

from chisurf.gui.widgets.navigation import NavigationPanelTool


def _make_overview(db):
    def factory(parent):
        from chisurf.plugins.spectra_downloader.gui.overview_panel import OverviewPanel

        return OverviewPanel(db, parent)

    return factory


def _make_browser(db):
    def factory(parent):
        from chisurf.plugins.spectra_downloader.browser import SpectraBrowserWidget

        return SpectraBrowserWidget(db, parent)

    return factory


def _make_download(db):
    def factory(parent):
        from chisurf.plugins.spectra_downloader.download_manager import DownloadPanel

        return DownloadPanel(db, parent)

    return factory


def _make_add_to_mfdb(db):
    def factory(parent):
        from chisurf.plugins.spectra_downloader.gui.add_to_mfdb_panel import AddToMfdbPanel

        return AddToMfdbPanel(db, parent)

    return factory


class SpectraTool(NavigationPanelTool):
    """Spectra downloader & browser (left navigation + selected panel)."""

    def __init__(self, db=None, parent=None):
        if db is None:
            from chisurf.plugins.spectra_downloader import get_db

            db = get_db()
            db.connect()
        self._db = db

        panels = [
            {
                "name": "Overview",
                "icon": "📊",
                "factory": _make_overview(db),
                "description": "Summary of the staging spectra database "
                               "(counts by category and source).",
            },
            {
                "name": "Browse",
                "icon": "🔎",
                "factory": _make_browser(db),
                "description": "Browse scraped components — filter, inspect "
                               "metadata, view spectra, push to the MFDB.",
            },
            {
                "name": "Download",
                "icon": "⬇️",
                "factory": _make_download(db),
                "description": "Run source scrapers into the staging DB.",
            },
            {
                "name": "Add to MFDB",
                "icon": "⬆️",
                "factory": _make_add_to_mfdb(db),
                "description": "Choose an endpoint (local file or server), "
                               "authenticate, and add the staging components to the MFDB.",
            },
        ]
        super().__init__(
            title="Spectra",
            panels=panels,
            parent=parent,
            minimum_size=(950, 600),
            initial_size=(1180, 760),
            navigation_width=200,
        )


def main() -> None:
    """Launch the Spectra tool standalone."""
    import sys

    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = SpectraTool()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
