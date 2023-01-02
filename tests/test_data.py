# -*- coding: utf-8 -*-
"""Test data parser."""
from pathlib import Path

import pytest

from fmri_decoder.data import TimeseriesData

# Path to resources folder
DIR_RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture
def file_series() -> list:
    """Dummy fmri time series."""
    file_data = "/home/daniel/dummy/fmri.nii"
    return [file_data] * 10


@pytest.fixture
def file_events() -> list:
    """Exemplary condition file."""
    file_data = DIR_RESOURCES / "events.mat"
    return [file_data] * 10


def test_load_events(file_series: list, file_events: list):
    """Check right parsing of event data."""
    timeseries = TimeseriesData(file_series, file_events)
    names, onsets, durations = timeseries.load_events(run=0)
    assert len(names) == len(onsets)
    assert len(onsets) == len(durations)
    assert [isinstance(n_, str) for n_ in names]
    assert [isinstance(o_, list) for o_ in onsets]
    assert [isinstance(d_, float) for d_ in durations]
