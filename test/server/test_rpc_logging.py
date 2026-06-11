from __future__ import annotations

"""Tests for client-side RPC logging helpers."""

import logging

from chisurf.server.rpc_logging import RpcLogWriter
from test.server.helpers import find_free_port


def test_rpc_log_writer_falls_back_to_local_logging(caplog):
    logger_name = "chisurf.test.rpc_fallback"
    writer = RpcLogWriter(
        logger_name,
        cmd_port=find_free_port(),
        timeout_ms=1,
    )

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        writer.debug("fallback diagnostic", file="m006.spc")

    assert "fallback diagnostic" in caplog.text
    assert "m006.spc" in caplog.text
