# -*- coding: utf-8 -*-
"""Test mvpa model."""

import numpy as np
import pytest

from fmri_decoder.model import MVPA


@pytest.fixture
def mvpa_model() -> MVPA:
    """Dummy model using the iris data set."""
    return MVPA.from_iris()


def test_shape(mvpa_model: MVPA):
    """Check input data shape."""
    assert mvpa_model.n_batch == 10
    assert np.array_equal(mvpa_model.label_names, [0, 1])
    for i, f in enumerate(mvpa_model.feature_names):
        assert f == f"feature {i}"


def test_scaling(mvpa_model: MVPA):
    """Check scaling methods."""
    # test feature scaling
    mvpa_model.scale_features("min_max")
    dtf = mvpa_model.dtf
    for i in range(mvpa_model.n_batch):
        data = dtf.loc[dtf["batch"] == i, mvpa_model.feature_names]
        assert np.nanmin(data) == 0.0
        assert np.nanmax(data) == 1.0
    mvpa_model.scale_features("standard")
    dtf = mvpa_model.dtf
    for i in range(mvpa_model.n_batch):
        data = dtf.loc[dtf["batch"] == i, mvpa_model.feature_names]
        assert np.isclose(np.nanmean(data), 0.0)
        assert np.isclose(np.nanstd(data), 1.0)

    # test sample scaling
    mvpa_model.scale_samples("norm")
    arr = mvpa_model.dtf.loc[:, mvpa_model.feature_names].to_numpy()
    for x in arr:
        assert np.isclose(np.sqrt(np.sum(x**2)), 1.0)
    mvpa_model.scale_samples("standard")
    arr = mvpa_model.dtf.loc[:, mvpa_model.feature_names].to_numpy()
    for x in arr:
        assert np.isclose(np.nanmean(x), 0.0)
        assert np.isclose(np.nanstd(x), 1.0)


def test_fit_predict(mvpa_model: MVPA):
    """Test output shapes of the fit and predict method."""
    # test fit
    assert [type(fit) == MVPA for fit in mvpa_model.fit]
    assert len(mvpa_model.fit) == mvpa_model.n_batch
    # test predict
    assert len(mvpa_model.predict()) == mvpa_model.n_batch


def test_accuracy(mvpa_model: MVPA):
    """Test accuracy metric."""
    n = 100
    length = 1000
    for _ in range(n):
        y_test = _get_random_events(length)
        y_predict = _get_random_events(length)
        result = len(y_test[y_test == y_predict]) / length
        assert np.isclose(mvpa_model._accuracy(y_test, y_predict), result)


def _get_random_events(length: int) -> np.ndarray:
    """Helper method to generate random label vectors."""
    label = np.zeros(length, dtype=int)
    label[: int(length / 2)] = 1
    np.random.shuffle(label)
    return label
