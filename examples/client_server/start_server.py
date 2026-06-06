#!/usr/bin/env python3
"""Example: start a ChiSurf ZMQ server and interact with it."""

import threading
import time

from chisurf.server.app import ChiSurfServer


def main():
    server = ChiSurfServer(cmd_port=18765, pub_port=18766)

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    print(f"Server running on ports {server.cmd_port}/{server.pub_port}")
    print(f"Methods: {server.dispatcher.list_methods()}")
    print(f"State: {server.state.to_dict()}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
