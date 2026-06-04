from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and start the ChiSurf ZMQ/JSON-RPC server.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If ``None``, ``sys.argv`` is used.

    """
    parser = argparse.ArgumentParser(description="ChiSurf ZMQ/JSON-RPC server")
    parser.add_argument(
        "--cmd-port", type=int, default=8765,
        help="TCP port for the REQ/REP command socket (default: 8765)",
    )
    parser.add_argument(
        "--pub-port", type=int, default=8766,
        help="TCP port for the PUB event socket (default: 8766)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from chisurf.server.app import ChiSurfServer

    server = ChiSurfServer(
        cmd_port=args.cmd_port,
        pub_port=args.pub_port,
        host=args.host,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Server shutting down (SIGINT)")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
