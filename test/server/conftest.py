from __future__ import annotations

import pytest


@pytest.fixture
def mock_fit():
    """Create a minimal mock fit object with UID support."""
    import uuid
    from unittest.mock import MagicMock

    fit = MagicMock()
    fit.unique_identifier = str(uuid.uuid4())
    fit.name = "MockFit"
    fit.chi2 = 1.5
    fit.data = MagicMock()
    fit.data.name = "MockData"
    fit.model = MagicMock()
    fit.model.parameters_all_dict = {}
    fit.model.parameter_values = []
    fit.model.parameter_bounds = []
    fit.grouped_fits = []
    return fit


@pytest.fixture
def mock_dataset():
    """Create a minimal mock dataset with UID support."""
    import uuid
    from unittest.mock import MagicMock

    ds = MagicMock()
    ds.unique_identifier = str(uuid.uuid4())
    ds.name = "MockDataset"
    ds.experiment = MagicMock()
    ds.experiment.name = "TCSPC"
    return ds


@pytest.fixture
def mock_experiment():
    """Create a minimal mock experiment."""
    from unittest.mock import MagicMock

    exp = MagicMock()
    exp.name = "TCSPC"
    exp.readers = []
    return exp


@pytest.fixture
def zmq_server_port():
    """Find a free port for ZMQ tests."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
