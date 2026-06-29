"""Headless tests for the additive FRET Line Generator GUI tool.

Each press of "+ Add FRET line" snapshots the current mixture/sweep and appends
it to a collection that is overlaid on the plots — successive presses accumulate
rather than replace. These tests drive that accumulation logic without a display
(offscreen Qt platform).
"""

import utils
import sys
import unittest
import pathlib

from qtpy.QtWidgets import QApplication

TOPDIR = pathlib.Path(__file__).parent.parent

utils.set_search_paths(TOPDIR)

from chisurf.plugins.fret_line.gui.tool import FRETLineTool


app = QApplication.instance() or QApplication(sys.argv)


class Tests(unittest.TestCase):

    def setUp(self):
        self.tool = FRETLineTool()
        self.tool._refresh_sweep_targets()
        self.assertGreater(self.tool._sweep_combo.count(), 0)

    def _compute_one(self, sweep_index=0, lo=20.0, hi=60.0, n=8):
        idx = min(sweep_index, self.tool._sweep_combo.count() - 1)
        self.tool._sweep_combo.setCurrentIndex(idx)
        self.tool._min_spin.setValue(lo)
        self.tool._max_spin.setValue(hi)
        self.tool._n_pts_spin.setValue(n)
        self.tool._do_compute()

    def test_starts_empty(self):
        self.assertEqual(len(self.tool._lines), 0)
        self.assertEqual(self.tool._lines_list.count(), 0)
        self.assertFalse(self.tool._save_btn.isEnabled())
        self.assertFalse(self.tool._push_btn.isEnabled())

    def test_compute_is_additive(self):
        self._compute_one(0)
        self.assertEqual(len(self.tool._lines), 1)
        self.assertEqual(self.tool._lines_list.count(), 1)
        self.assertTrue(self.tool._save_btn.isEnabled())
        self.assertTrue(self.tool._push_btn.isEnabled())

        # second press appends rather than replaces
        self._compute_one(1)
        self.assertEqual(len(self.tool._lines), 2)
        self.assertEqual(self.tool._lines_list.count(), 2)

        # each line carries a distinct colour and a populated result
        colors = [ln["color"] for ln in self.tool._lines]
        self.assertEqual(len(set(colors)), 2)
        for ln in self.tool._lines:
            self.assertEqual(len(ln["result"]["parameter_values"]), 8)

    def test_remove_and_clear(self):
        self._compute_one(0)
        self._compute_one(1)
        self._compute_one(0)
        self.assertEqual(len(self.tool._lines), 3)

        self.tool._lines_list.setCurrentRow(1)
        self.tool._remove_line()
        self.assertEqual(len(self.tool._lines), 2)
        self.assertEqual(self.tool._lines_list.count(), 2)

        self.tool._clear_lines()
        self.assertEqual(len(self.tool._lines), 0)
        self.assertFalse(self.tool._save_btn.isEnabled())
        self.assertFalse(self.tool._push_btn.isEnabled())

    def test_visibility_checkboxes(self):
        from qtpy.QtCore import Qt

        self._compute_one(0)
        self._compute_one(1)
        # both start visible and checked
        self.assertTrue(all(ln["visible"] for ln in self.tool._lines))
        self.assertEqual(
            self.tool._lines_list.item(0).checkState(), Qt.Checked
        )

        # unticking a checkbox hides that line but keeps it in the list
        self.tool._lines_list.item(0).setCheckState(Qt.Unchecked)
        self.assertFalse(self.tool._lines[0]["visible"])
        self.assertTrue(self.tool._lines[1]["visible"])
        self.assertEqual(len(self.tool._lines), 2)

        # hide-all / show-all helpers
        self.tool._set_all_visible(False)
        self.assertTrue(all(not ln["visible"] for ln in self.tool._lines))
        self.assertEqual(self.tool._lines_list.item(1).checkState(), Qt.Unchecked)
        self.tool._set_all_visible(True)
        self.assertTrue(all(ln["visible"] for ln in self.tool._lines))

    def test_editing_mixture_keeps_lines(self):
        # Computed lines are snapshots: changing the editor must not drop them.
        self._compute_one(0)
        self.assertEqual(len(self.tool._lines), 1)
        self.tool._add_component()
        self.assertEqual(len(self.tool._lines), 1)
        self.assertTrue(self.tool._save_btn.isEnabled())

    def test_save_csv_all_lines(self):
        import tempfile, os

        self._compute_one(0)
        self._compute_one(1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lines.csv")
            # bypass the file dialog by writing directly through the same path
            self.tool._lines  # noqa: B018  (ensure populated)
            from unittest import mock

            with mock.patch.object(
                self.tool, "_on_save", wraps=self.tool._on_save
            ):
                with mock.patch(
                    "qtpy.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(path, "CSV (*.csv)"),
                ), mock.patch("qtpy.QtWidgets.QMessageBox.information"):
                    self.tool._on_save()
            with open(path) as fh:
                text = fh.read()
            self.assertIn("Line 1", text)
            self.assertIn("Line 2", text)
            # header documents the tidy/long layout with a per-row line column
            self.assertIn("line,sweep,log,components,parameter", text)


if __name__ == "__main__":
    unittest.main()
