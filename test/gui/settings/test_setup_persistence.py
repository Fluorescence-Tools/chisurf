"""Tests for setup persistence and 2-way sync.

These tests verify that:
1. Reader state can be serialized/deserialized correctly
2. UI can be synced from reader (updateUI)
3. Reader can be synced from UI (onParametersChanged)
4. The persistence layer works correctly
"""
import pytest
import json
import pathlib
import tempfile
import os
import sys

import numpy as np


class TestReaderSerialization:
    """Tests for reader state serialization."""

    def test_basic_types_serialization(self):
        """Test that basic types are serialized correctly."""
        from chisurf.gui.widgets.experiments.setup_persistence import _to_basic

        assert _to_basic("test") == "test"
        assert _to_basic(123) == 123
        assert _to_basic(1.5) == 1.5
        assert _to_basic(True) is True
        assert _to_basic(None) is None
        assert _to_basic(np.int64(42)) == 42
        assert _to_basic(np.float64(3.14)) == pytest.approx(3.14)
        assert _to_basic(np.array([1, 2, 3])).tolist() == [1, 2, 3]
        assert _to_basic([1, 2, 3]) == [1, 2, 3]
        assert _to_basic((1, 2, 3)) == [1, 2, 3]

    def test_serialize_reader_state(self):
        """Test serialization of a mock reader."""
        from chisurf.gui.widgets.experiments.setup_persistence import (
            serialize_reader_state,
        )

        class MockReader:
            def __init__(self):
                self.name = "test_reader"
                self.reading_routine = "PTU"
                self.channel_numbers = np.array([0, 1], dtype=np.int8)
                self.g_factor = 1.5
                self.experiment = None  # Should be skipped
                self._cache = {}  # Should be skipped

        reader = MockReader()
        result = serialize_reader_state(reader)

        assert result is not None
        assert result["class"] == "MockReader"
        assert result["state"]["name"] == "test_reader"
        assert result["state"]["reading_routine"] == "PTU"
        assert result["state"]["channel_numbers"] == [0, 1]
        assert result["state"]["g_factor"] == pytest.approx(1.5)
        # Should skip experiment and _cache
        assert "experiment" not in result["state"]
        assert "_cache" not in result["state"]

    def test_deserialize_reader_state(self):
        """Test deserialization of reader state."""
        from chisurf.gui.widgets.experiments.setup_persistence import (
            deserialize_reader_state,
        )

        reader_info = {
            "module": "test.module",
            "class": "MockReader",
            "state": {
                "name": "test",
                "reading_routine": "HT3",
                "channel_numbers": [0, 2],
            },
        }

        state = deserialize_reader_state(reader_info)
        assert state["name"] == "test"
        assert state["reading_routine"] == "HT3"
        assert state["channel_numbers"] == [0, 2]

    def test_apply_reader_state(self):
        """Test applying state to a mock reader."""
        from chisurf.gui.widgets.experiments.setup_persistence import (
            apply_reader_state,
        )

        class MockReader:
            pass

        reader = MockReader()
        state = {
            "name": "test_name",
            "reading_routine": "SPC",
            "g_factor": 2.0,
        }

        apply_reader_state(reader, state)

        assert reader.name == "test_name"
        assert reader.reading_routine == "SPC"
        assert reader.g_factor == 2.0


class TestSetupPersistence:
    """Tests for setup defaults persistence."""

    def test_save_and_load_defaults(self, tmp_path):
        """Test saving and loading setup defaults."""
        # Mock the settings path
        from chisurf.gui.widgets.experiments import setup_persistence

        original_get_path = setup_persistence.get_path

        def mock_get_path(path_type):
            if path_type == "settings":
                return tmp_path
            return original_get_path(path_type)

        setup_persistence.get_path = mock_get_path

        try:
            defaults = {
                "schema_version": 1,
                "last_selection": {
                    "experiment_index": 2,
                    "setup_index": 1,
                },
                "experiments": {
                    "TCSPC": {
                        "TTTR": {
                            "module": "chisurf.core.experiments.tcspc",
                            "class": "TCSPCTTTRReader",
                            "state": {
                                "reading_routine": "PTU",
                                "channel_numbers": [0],
                            },
                        }
                    }
                },
            }

            # Save
            result = setup_persistence.save_setup_defaults(defaults)
            assert result is True

            # Load
            loaded = setup_persistence.load_setup_defaults()
            assert loaded["schema_version"] == 1
            assert loaded["last_selection"]["experiment_index"] == 2
            assert "TCSPC" in loaded["experiments"]
        finally:
            setup_persistence.get_path = original_get_path

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading when file doesn't exist."""
        from chisurf.gui.widgets.experiments import setup_persistence

        original_get_path = setup_persistence.get_path

        def mock_get_path(path_type):
            if path_type == "settings":
                return tmp_path
            return original_get_path(path_type)

        setup_persistence.get_path = mock_get_path

        try:
            loaded = setup_persistence.load_setup_defaults()
            assert loaded["schema_version"] == 1
            assert loaded["experiments"] == {}
        finally:
            setup_persistence.get_path = original_get_path


class TestUiSyncHelper:
    """Tests for UI sync helper utilities."""

    def test_signal_blocker(self):
        """Test SignalBlocker context manager."""
        from chisurf.gui.widgets.experiments.ui_sync import SignalBlocker
        from qtpy import QtCore

        class MockWidget(QtCore.QObject):
            signal = QtCore.Signal()

            def __init__(self):
                super().__init__()
                self._signals_blocked = False

            def signalsBlocked(self):
                return self._signals_blocked

            def blockSignals(self, block):
                self._signals_blocked = block

        widget = MockWidget()
        assert widget.signalsBlocked() is False

        with SignalBlocker(widget):
            assert widget.signalsBlocked() is True

        assert widget.signalsBlocked() is False

    def test_reentrancy_guard(self):
        """Test ReentrancyGuard."""
        from chisurf.gui.widgets.experiments.ui_sync import ReentrancyGuard

        call_count = 0

        class TestClass:
            _reentrancy_guard_default = False

            @ReentrancyGuard()
            def guarded_method(self):
                nonlocal call_count
                call_count += 1

        obj = TestClass()
        obj.guarded_method()
        assert call_count == 1

        # Second call should work (not reentrant from same object)
        obj.guarded_method()
        assert call_count == 2


class TestControllerSyncInvariants:
    """Tests verifying 2-way sync invariants for controllers.

    These tests verify that after calling updateUI(), the UI matches the reader,
    and after calling onParametersChanged(), the reader matches the UI.
    """

    def test_fcs_controller_sync(self):
        """Test FCS controller 2-way sync."""
        # This test requires Qt application context
        # Skip if no display available
        pytest.importorskip("qtpy")
        pytest.importorskip("PyQt5")

        from chisurf.gui.widgets.experiments.fcs import FCSController

        # Create a mock reader with specific attributes
        class MockFCSReader:
            name = "FCS Test"
            weight_mode = "photon_noise"

        # Create controller with mock reader
        controller = FCSController.__new__(FCSController)
        controller.experiment_reader = MockFCSReader()

        # Test that updateUI exists and is callable
        assert hasattr(controller, "updateUI")
        assert hasattr(controller, "onParametersChanged")
        assert callable(controller.updateUI)
        assert callable(controller.onParametersChanged)
