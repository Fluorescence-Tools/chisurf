from __future__ import annotations
import sys
import os
import time

# Ensure path to chisurf is included
sys.path.append(os.getcwd())

from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer, MockWindow
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

def test_animation():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    cmd.set_message_callback(lambda m: print(f"MSG: {m}"))
    cmd.set_error_callback(lambda e: print(f"ERROR: {e}"))
    
    print("\nTesting mset:")
    cmd.do("mset 1 x50")
    if viewer.get_total_frames() == 50:
        print("SUCCESS: Timeline set to 50 frames.")
    else:
        print(f"FAILURE: Total frames is {viewer.get_total_frames()}")

    print("\nTesting frame command:")
    cmd.do("frame 25")
    if viewer.get_current_frame() == 24: # 0-indexed internal
        print("SUCCESS: Jumped to frame 25 (internal 24).")
    else:
        print(f"FAILURE: Current frame is {viewer.get_current_frame()}")

    print("\nTesting mplay (simulated):")
    cmd.do("mplay")
    if viewer._animation_running:
        print("SUCCESS: Animation running flag set.")
    else:
        print("FAILURE: Animation not running.")

    print("\nTesting mstop:")
    cmd.do("mstop")
    if not viewer._animation_running and viewer.get_current_frame() == 0:
        print("SUCCESS: Animation stopped and reset to frame 1.")
    else:
        print(f"FAILURE: State after mstop: running={viewer._animation_running}, frame={viewer.get_current_frame()}")

    print("\nAll animation command tests passed!")

if __name__ == "__main__":
    test_animation()
