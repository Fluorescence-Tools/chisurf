from __future__ import annotations
import sys
import argparse
from pathlib import Path
from ..cmd.core import Cmd
from ..testing.mock_viewer import MockWindow

def main():
    parser = argparse.ArgumentParser(description="ChiMol Headless CLI")
    parser.add_argument("-e", "--execute", type=str, help="Execute a semicolon-separated list of commands and exit")
    parser.add_argument("-s", "--script", type=str, help="Run a script file and exit")
    args = parser.parse_args()

    win = MockWindow()
    cmd = Cmd(win)
    
    # Print to stdout/stderr
    cmd.set_message_callback(lambda msg: print(f"[Chimol] {msg}"))
    cmd.set_error_callback(lambda err: print(f"[Error] {err}", file=sys.stderr))

    if args.execute:
        for line in args.execute.split(";"):
            cmd.do(line.strip())
        return

    if args.script:
        cmd.do(f"@{args.script}")
        return

    # REPL mode
    print("ChiMol Headless CLI (PyMOL Parity Layer)")
    print("Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("chimol> ").strip()
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break
            cmd.do(line)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Runtime error: {e}")

if __name__ == "__main__":
    main()
