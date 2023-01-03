# -*- coding: utf-8 -*-
"""Classes for parameter and data deserialization."""

from dataclasses import dataclass
from typing import Optional

import nibabel as nb
import numpy as np
import yaml
from fmri_tools.io.surf import read_mgh
from nibabel.freesurfer.io import read_geometry, read_label
from scipy.io.matlab import loadmat

__all__ = ["TimeseriesData", "SurfaceData", "DataConfig", "ModelConfig"]


class TimeseriesData:
    """Load fmri time series volume data and corresponding experimental events.

    Attributes:
        file_series: List of time series file names.
        file_events: List of corresponding event files.

    """

    def __init__(self, file_series: list[str], file_events: list[str]) -> None:
        self.file_series = file_series
        self.file_events = file_events

    def load_timeseries(self, run: int) -> nb.Nifti1Image:
        """Load a single fmri time series.

        Args:
            run: Selected file from the list of input time series.

        Returns:
            Nibabel image object that contains both image data and header information.
        """
        return nb.load(self.file_series[run])

    def load_events(self, run: int) -> tuple:
        """Load a single event file.

        Args:
            run: Selected file from the list of input event files.

        Returns:
            A tuple that contains names of experimental conditions, onset times of
            single blocks and corresponding durations.
        """
        cond_ = loadmat(self.file_events[run])
        names = [list(l_)[0] for l_ in cond_["names"][0]]
        onsets = [list(l_)[0] for l_ in cond_["onsets"][0]]
        durations = [list(l_)[0][0] for l_ in cond_["durations"][0]]
        return names, onsets, durations

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Load from dictionary."""
        return cls(
            file_series=data_dict["file_series"],
            file_events=data_dict["file_events"],
        )

    @classmethod
    def from_yaml(cls, file_yaml: str):
        """Deserialize from yaml file."""
        with open(file_yaml, "r", encoding="utf8") as yml:
            config = {}
            try:
                config = yaml.safe_load(yml)
            except yaml.YAMLError as exc:
                print(exc)

            return cls(
                file_series=config["file_series"],
                file_events=config["file_events"],
            )

    @property
    def file_series(self) -> list[str]:
        """Get files series."""
        return self._file_series

    @file_series.setter
    def file_series(self, fs_: list[str]) -> None:
        """Set file series."""
        if len(fs_) <= 1:
            raise ValueError("More than one time series must be given!")
        self._file_series = fs_

    @property
    def file_events(self) -> list[str]:
        """Get file events."""
        return self._file_events

    @file_events.setter
    def file_events(self, fe_: list[str]) -> None:
        """Set file events."""
        if len(self.file_series) != len(fe_):
            raise ValueError("The number of timeseries and event files do not match!")
        self._file_events = fe_


class SurfaceData:
    """Load surface data.

    Attributes:
        file_layer: Dictionary with a list of surface file names per hemisphere.
        file_localizer: Dictionary with a list of overlay file names per hemisphere.
        file_label: Dictionary with a list of labels file names per hemisphere.

    """

    def __init__(
        self,
        file_layer: dict,
        file_localizer: dict,
        file_label: dict,
    ) -> None:
        self.file_layer = file_layer
        self.file_localizer = file_localizer
        self.file_label = file_label

    def load_layer(self, hemi: str, n: int) -> tuple:
        """Load surface data for a single hemisphere and cortical depth.

        Args:
            hemi: Hemisphere
            n: Cortical depth (layer number)

        Returns:
            A tuple that contains an vertex array and a corresponding face array.
        """
        return read_geometry(self.file_layer[hemi][n])

    def load_localizer(self, hemi: str, sess: int) -> np.ndarray:
        """Load localizer data for a single hemisphere.

        Args:
            hemi: Hemisphere
            sess: Selected file from the list of input localizers.

        Returns:
            Array of vertex-wise data points.
        """
        return read_mgh(self.file_localizer[hemi][sess])[0]

    def load_localizer_average(self, hemi: str) -> np.ndarray:
        """Load the mean localizer data for a single hemisphere that is averaged across
        sessions.

        Args:
            hemi: Hemisphere

        Returns:
            Array of vertex-wise data points.
        """
        data = np.zeros(len(self.load_layer(hemi, 0)[0]))
        n_files = len(self.file_localizer[hemi])
        for i in range(n_files):
            data += self.load_localizer(hemi, i)
        data /= n_files
        return data

    def load_label(self, hemi: str, sess: int) -> np.ndarray:
        """Load label data for a single hemisphere.

        Args:
            hemi: Hemisphere.
            sess: Selected file from the list of input labels.

        Returns:
            Array of labeled vertex indices.
        """
        label, _ = read_label(self.file_label[hemi][sess], read_scalars=True)
        return label

    def load_label_intersection(self, hemi: str) -> np.ndarray:
        """Load the intersection of multiple label files for a single hemisphere.

        Args:
            hemi: Hemisphere.

        Returns:
            Array of labeled vertex indices.
        """
        data = np.arange(len(self.load_layer(hemi, 0)[0]))
        for i in range(len(self.file_label[hemi])):
            data = np.intersect1d(data, self.load_label(hemi, i))
        return data

    def _compare(self, lst1: list[str], lst2: list[str]) -> bool:
        """Helper method that checks the equivalence of two lists."""
        lst1_clean = list(self._remove_hemisphere(lst1))
        lst2_clean = list(self._remove_hemisphere(lst2))
        return lst1_clean == lst2_clean

    @staticmethod
    def _remove_hemisphere(lst: list[str]) -> list[str]:
        """Helper method that removes the indicated hemisphere ('lh.' or 'rh.') from the
        basename of the input file name."""
        return [l_[3:] for l_ in lst]

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Load from dictionary."""
        return cls(
            file_layer=data_dict["file_layer"],
            file_localizer=data_dict["file_localizer"],
            file_label=data_dict["file_label"],
        )

    @classmethod
    def from_yaml(cls, file_yaml: str):
        """Deserialize from yaml file."""
        with open(file_yaml, "r", encoding="utf8") as yml:
            config = {}
            try:
                config = yaml.safe_load(yml)
            except yaml.YAMLError as exc:
                print(exc)

            return cls(
                file_layer=config["file_layer"],
                file_localizer=config["file_localizer"],
                file_label=config["file_label"],
            )

    @property
    def file_layer(self) -> dict:
        """Get file layer."""
        return self._file_layer

    @file_layer.setter
    def file_layer(self, fl_: dict) -> None:
        """Set file layer."""
        if self._compare(fl_["lh"], fl_["rh"]):
            raise ValueError("Layers in left and right hemisphere do not match!")
        self._file_layer = fl_

    @property
    def file_localizer(self) -> dict:
        """Get file localizers."""
        return self._file_localizer

    @file_localizer.setter
    def file_localizer(self, fl_: dict) -> None:
        """Set file localizers."""
        if self._compare(fl_["lh"], fl_["rh"]):
            raise ValueError("Localizers in left and right hemisphere do not match!")
        self._file_localizer = fl_

    @property
    def file_label(self) -> dict:
        """Get file label."""
        return self._file_label

    @file_label.setter
    def file_label(self, fl_: dict) -> None:
        """Set file label."""
        if self._compare(fl_["lh"], fl_["rh"]):
            raise ValueError("Label data in left and right hemisphere do not match!")
        self._file_label = fl_


@dataclass
class DataConfig:
    """Parameters for data preprocessing.

    Attributes:
        tr: Repetition time in seconds.
        n_skip: Number of skipped initial volumes per experimental condition.
        cutoff_sec: Cutoff frequency for time series detrending in 1/Hz.
        filter_size: Bandpass filter size. Default to None.
        file_deformation: File name of deformation field that contains the
        transformation of surface coordinates to the space of fmri time series. Defaults
        to None.

    """

    tr: float
    n_skip: int
    cutoff_sec: float
    filter_size: Optional[float] = None
    file_deformation: Optional[str] = None

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Load from dictionary."""
        return cls(
            tr=data_dict["TR"],
            n_skip=data_dict["n_skip"],
            cutoff_sec=data_dict["cutoff_sec"],
            filter_size=data_dict["filter_size"],
            file_deformation=data_dict["file_deformation"],
        )

    @classmethod
    def from_yaml(cls, file_yaml: str):
        """Deserialize from yaml file."""
        with open(file_yaml, "r", encoding="utf8") as yml:
            config = {}
            try:
                config = yaml.safe_load(yml)
                _filter_size = (
                    None if config["filter_size"] == "none" else config["filter_size"]
                )
                _file_deformation = (
                    None
                    if config["file_deformation"] == "none"
                    else config["file_deformation"]
                )
            except yaml.YAMLError as exc:
                print(exc)

            return cls(
                tr=config["TR"],
                n_skip=config["n_skip"],
                cutoff_sec=config["cutoff_sec"],
                filter_size=_filter_size,
                file_deformation=_file_deformation,
            )


@dataclass
class ModelConfig:
    """Parameters for setting up the MVPA model.

    Attributes:
        nmax: Number of considered features (data points).
        radius: Minimum distance between selected features.
        feature_scaling: Feature scaling method. Defaults to None.
        sample_scaling: Sample scaling method. Defaults to None.

    """

    nmax: Optional[int]
    radius: Optional[float]
    feature_scaling: Optional[str] = None  # min_max | standard | None
    sample_scaling: Optional[str] = None  # norm | standard | None

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Load from dictionary."""
        return cls(
            nmax=data_dict["nmax"],
            radius=data_dict["radius"],
            feature_scaling=data_dict["feature_scaling"],
            sample_scaling=data_dict["sample_scaling"],
        )

    @classmethod
    def from_yaml(cls, file_yaml: str):
        """Deserialize from yaml file."""
        with open(file_yaml, "r", encoding="utf8") as yml:
            config = {}
            try:
                config = yaml.safe_load(yml)
                _nmax = None if config["nmax"] == "none" else config["nmax"]
                _radius = None if config["radius"] == "none" else config["radius"]
                _feature_scaling = (
                    None
                    if config["feature_scaling"] == "none"
                    else config["feature_scaling"]
                )
                _sample_scaling = (
                    None
                    if config["sample_scaling"] == "none"
                    else config["sample_scaling"]
                )
            except yaml.YAMLError as exc:
                print(exc)

            return cls(
                nmax=_nmax,
                radius=_radius,
                feature_scaling=_feature_scaling,
                sample_scaling=_sample_scaling,
            )
