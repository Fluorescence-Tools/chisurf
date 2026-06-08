from __future__ import annotations
import pytest
from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer, MockWindow
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

def test_chimol_animation_mset():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    cmd.do("mset 1 x50")
    assert viewer.get_total_frames() == 50

def test_chimol_animation_frame():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    viewer.mset(1, 50)
    
    cmd.do("frame 25")
    assert viewer.get_current_frame() == 24 # 0-indexed internal

def test_chimol_animation_play_stop():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    viewer.mset(1, 50)
    
    cmd.do("mplay")
    assert viewer._animation_running
    
    cmd.do("mstop")
    assert not viewer._animation_running
    assert viewer.get_current_frame() == 0
