import pathlib

import numpy as np
import pytest
import tempfile

import chisurf.core.data


@pytest.fixture
def sample_curve():
    x = np.linspace(0, 10, 11)
    y = np.sin(x)
    return chisurf.core.data.DataCurve(x=x, y=y)


@pytest.fixture
def sample_curve_short():
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    return chisurf.core.data.DataCurve(x=x, y=y, name="S")


@pytest.fixture
def sample_data_group(sample_curve_short):
    c1 = sample_curve_short
    c2 = chisurf.core.data.DataCurve(x=np.array([5.0, 6.0]), y=np.array([7.0, 8.0]), name="B")
    return chisurf.core.data.DataGroup([c1, c2])


class TestExperimentalData:

    def test_init_defaults(self):
        d = chisurf.core.data.ExperimentalData()
        assert d.data_reader is None
        assert d._experiment is None

    def test_init_with_reader_and_experiment(self):
        d = chisurf.core.data.ExperimentalData(data_reader="mock", experiment="mock_exp")
        assert d.data_reader == "mock"
        assert d._experiment == "mock_exp"

    def test_experiment_property(self):
        d = chisurf.core.data.ExperimentalData()
        d.experiment = "exp1"
        assert d.experiment == "exp1"

    def test_getstate(self):
        d = chisurf.core.data.ExperimentalData(name="test")
        state = d.__getstate__()
        assert isinstance(state, dict)
        assert "name" in state

    def test_to_dict(self):
        d = chisurf.core.data.ExperimentalData(name="test_dict")
        result = d.to_dict()
        assert result["name"] == "test_dict"


class TestDataCurve:

    def test_init_defaults(self):
        c = chisurf.core.data.DataCurve()
        assert c.ex is not None
        assert c.ey is not None
        assert c.mask is not None

    def test_init_with_data(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([3.0, 4.0, 5.0])
        c = chisurf.core.data.DataCurve(x=x, y=y)
        np.testing.assert_array_equal(c.x, x)
        np.testing.assert_array_equal(c.y, y)

    def test_init_with_errors(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        ex = np.array([0.1, 0.1])
        ey = np.array([0.2, 0.2])
        c = chisurf.core.data.DataCurve(x=x, y=y, ex=ex, ey=ey)
        np.testing.assert_array_equal(c.ex, ex)
        np.testing.assert_array_equal(c.ey, ey)

    def test_init_with_mask(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        mask = np.array([1.0, 0.0, 1.0])
        c = chisurf.core.data.DataCurve(x=x, y=y, mask=mask)
        np.testing.assert_array_equal(c.mask, mask)

    def test_data_property_getter(self, sample_curve):
        data = sample_curve.data
        assert data.shape[0] == 5  # x, y, ex, ey, mask
        assert data.shape[1] == 11

    def test_data_property_setter(self, sample_curve_short):
        """set_data replaces arrays; data setter unpacks stacked input."""
        x = np.array([10.0, 20.0])
        y = np.array([30.0, 40.0])
        ex = np.array([0.5, 0.5])
        ey = np.array([0.6, 0.6])
        mask = np.array([0.0, 1.0])
        sample_curve_short.data = np.vstack([x, y, ex, ey, mask])
        np.testing.assert_array_equal(sample_curve_short.x, x)
        np.testing.assert_array_equal(sample_curve_short.ey, ey)
        np.testing.assert_array_equal(sample_curve_short.mask, mask)

    def test_to_dict_roundtrip(self, sample_curve):
        d = sample_curve.to_dict()
        c2 = chisurf.core.data.DataCurve()
        c2.from_dict(d)
        np.testing.assert_array_equal(c2.x, sample_curve.x)
        np.testing.assert_array_equal(c2.y, sample_curve.y)
        np.testing.assert_array_equal(c2.ex, sample_curve.ex)
        np.testing.assert_array_equal(c2.ey, sample_curve.ey)

    def test_set_data_in_place(self, sample_curve_short):
        """set_data with matching-length arrays updates in place."""
        x = np.array([10.0, 20.0])
        y = np.array([30.0, 40.0])
        sample_curve_short.set_data(x, y)
        np.testing.assert_array_equal(sample_curve_short.x, x)
        np.testing.assert_array_equal(sample_curve_short.y, y)
        np.testing.assert_array_equal(sample_curve_short.ex, np.ones_like(x))
        np.testing.assert_array_equal(sample_curve_short.ey, np.ones_like(y))

    def test_set_data_with_errors(self, sample_curve_short):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        ex = np.array([0.5, 0.5])
        ey = np.array([0.6, 0.6])
        mask = np.array([0.0, 1.0])
        sample_curve_short.set_data(x, y, ex, ey, mask)
        np.testing.assert_array_equal(sample_curve_short.ex, ex)
        np.testing.assert_array_equal(sample_curve_short.ey, ey)
        np.testing.assert_array_equal(sample_curve_short.mask, mask)

    def test_set_weights(self, sample_curve):
        w = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        sample_curve.set_weights(w)
        np.testing.assert_array_almost_equal(sample_curve.ey, 1.0 / w)

    def test_getitem_int(self, sample_curve):
        result = sample_curve[0]
        assert len(result) == 5
        assert result[0] == sample_curve.x[0]
        assert result[1] == sample_curve.y[0]

    def test_getitem_slice(self, sample_curve):
        result = sample_curve[2:5]
        assert len(result) == 5
        np.testing.assert_array_equal(result[0], sample_curve.x[2:5])
        np.testing.assert_array_equal(result[1], sample_curve.y[2:5])

    def test_str(self, sample_curve):
        s = str(sample_curve)
        assert "Dataset:" in s
        assert "filename" in s

    def test_save_and_load_csv(self, sample_curve):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            tmpname = f.name
        try:
            sample_curve.save(tmpname, file_type="csv")
            c2 = chisurf.core.data.DataCurve(
                x=np.zeros_like(sample_curve.x),
                y=np.zeros_like(sample_curve.y)
            )
            c2.load(tmpname, file_type="csv")
            np.testing.assert_array_almost_equal(c2.x, sample_curve.x)
            np.testing.assert_array_almost_equal(c2.y, sample_curve.y)
        finally:
            pathlib.Path(tmpname).unlink(missing_ok=True)

    def test_load_csv_5col(self):
        """Load CSV with 5 columns (x, y, ex, ey, mask)."""
        with tempfile.NamedTemporaryFile(
                suffix=".csv", mode="w", delete=False
        ) as f:
            f.write("1.0,10.0,0.1,0.2,1.0\n2.0,20.0,0.1,0.2,0.0\n")
            tmpname = f.name
        try:
            c = chisurf.core.data.DataCurve(x=np.array([0.0, 0.0]), y=np.array([0.0, 0.0]))
            c.load(tmpname, file_type="csv")
            np.testing.assert_array_almost_equal(c.x, np.array([1.0, 2.0]))
            np.testing.assert_array_almost_equal(c.y, np.array([10.0, 20.0]))
        finally:
            pathlib.Path(tmpname).unlink(missing_ok=True)


class TestDataGroup:

    def test_init_empty(self):
        g = chisurf.core.data.DataGroup([])
        assert len(g) == 0

    def test_names(self, sample_data_group):
        assert sample_data_group.names == ["S", "B"]

    def test_current_dataset(self, sample_data_group):
        assert sample_data_group.current_dataset.name == "S"
        sample_data_group.current_dataset = 1
        assert sample_data_group.current_dataset.name == "B"

    def test_current_dataset_empty_raises(self):
        g = chisurf.core.data.DataGroup([])
        with pytest.raises(IndexError):
            _ = g.current_dataset

    def test_name_default_is_class_name(self):
        g = chisurf.core.data.DataGroup([])
        assert g.name == "DataGroup"

    def test_name_uses_explicit_name_via_dict(self, sample_data_group):
        sample_data_group.__dict__["name"] = "MyGroup"
        assert sample_data_group.name == "MyGroup"

    def test_name_explicit_from_init(self):
        g = chisurf.core.data.DataGroup([], name="Explicit")
        assert g.name == "Explicit"

    def test_filename_empty(self):
        g = chisurf.core.data.DataGroup([])
        assert g.filename == "Empty group"

    def test_append_single(self, sample_data_group):
        c3 = chisurf.core.data.DataCurve(name="C")
        sample_data_group.append(c3)
        assert len(sample_data_group) == 3

    def test_append_list(self, sample_data_group):
        c3 = chisurf.core.data.DataCurve(name="C")
        c4 = chisurf.core.data.DataCurve(name="D")
        sample_data_group.append([c3, c4])
        assert len(sample_data_group) == 4

    def test_append_non_experimental_ignored(self, sample_data_group):
        sample_data_group.append("not-a-dataset")
        assert len(sample_data_group) == 2

    def test_to_yaml(self, sample_data_group):
        yaml_str = sample_data_group.to_yaml()
        assert isinstance(yaml_str, str)


class TestDataCurveGroup:

    def test_init(self, sample_curve):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve, c2])
        assert len(g) == 2

    def test_property_proxying(self, sample_curve):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve, c2])
        np.testing.assert_array_equal(g.x, sample_curve.x)
        np.testing.assert_array_equal(g.y, sample_curve.y)
        np.testing.assert_array_equal(g.ex, sample_curve.ex)
        np.testing.assert_array_equal(g.ey, sample_curve.ey)
        np.testing.assert_array_equal(g.mask, sample_curve.mask)

    def test_property_setters(self, sample_curve_short):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve_short, c2])
        new_x = np.array([99.0, -99.0])
        g.x = new_x
        np.testing.assert_array_equal(sample_curve_short.x, new_x)

    def test_getitem_slice(self, sample_curve_short):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve_short, c2])
        result = g[:]
        assert len(result) == 5
        np.testing.assert_array_equal(result[0], sample_curve_short.x)

    def test_getitem_int(self, sample_curve_short):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve_short, c2])
        item = g[0]
        assert item is sample_curve_short

    def test_str(self, sample_curve_short):
        c2 = chisurf.core.data.DataCurve(x=np.array([0.0]), y=np.array([1.0]), name="B")
        g = chisurf.core.data.DataCurveGroup([sample_curve_short, c2])
        result = g.__str__()
        assert isinstance(result, list)
        assert len(result) == 2


class TestExperimentDataGroup:

    def test_init(self):
        d = chisurf.core.data.ExperimentalData(name="E1", experiment="mock_exp")
        g = chisurf.core.data.ExperimentDataGroup([d])
        assert len(g) == 1

    def test_experiment_property(self):
        d = chisurf.core.data.ExperimentalData(name="E1", experiment="mock_exp")
        g = chisurf.core.data.ExperimentDataGroup([d])
        assert g.experiment == "mock_exp"


class TestExperimentDataCurveGroup:

    def test_init(self):
        c = chisurf.core.data.DataCurve(name="C1", experiment="exp1")
        g = chisurf.core.data.ExperimentDataCurveGroup([c])
        assert len(g) == 1


class TestGetData:

    def test_get_data_experiment_type(self):
        d1 = chisurf.core.data.ExperimentalData(name="A")
        d2 = chisurf.core.data.ExperimentalData(name="Global-fit")
        result = chisurf.core.data.get_data(curve_type="experiment", data_set=[d1, d2])
        assert len(result) == 1
        assert result[0].name == "A"

    def test_get_data_all(self):
        d1 = chisurf.core.data.ExperimentalData(name="A")
        d2 = chisurf.core.data.ExperimentalData(name="Global-fit")
        result = chisurf.core.data.get_data(curve_type="all", data_set=[d1, d2])
        assert len(result) == 2
