#!/usr/bin/env python3
"""Example: connect to a running ChiSurf server via ChisurfClient."""

import sys
import time

from chisurf.api._client import ChisurfClient


def main():
    host = "127.0.0.1"
    cmd_port = 18765
    pub_port = 18766

    client = ChisurfClient(host=host, cmd_port=cmd_port, pub_port=pub_port, timeout_ms=5000)

    try:
        client.connect()

        # Ping the server
        pong = client.ping()
        print(f"Ping response: {pong}")
        assert pong.get("ok"), "Server not healthy"

        # List available RPC methods
        methods = client.list_methods()
        print(f"Server exposes {len(methods)} methods:")
        for m in methods:
            print(f"  - {m}")

        # Datasets are empty initially
        datasets = client.list_datasets()
        print(f"Datasets: {len(datasets)}")

        # Fits are empty initially
        fits = client.list_fits()
        print(f"Fits: {len(fits)}")

        print("\nSuccessfully connected to ChiSurf server!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
