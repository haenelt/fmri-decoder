# -*- coding: utf-8 -*-
"""Preprocessing utilities to prepare fmri time series for the decoding analysis. This
module does not provide a complete preprocessing pipeline for fmri data (e.g. no tools
for motion correction, distortion correction or slice-time are provided here and are
assumed to be applied beforehand if necessary for the analysis)."""

import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import nibabel as nb
import numpy as np
import yaml
from fmri_tools.io.hdf5 import write_hdf5
from fmri_tools.io.surf import write_label
from fmri_tools.mapping.map_timeseries import map_timeseries
from fmri_tools.preprocessing.timeseries import FilterTimeseries, ScaleTimeseries
from fmri_tools.surface.filter import LaplacianGaussian
from fmri_tools.surface.mesh import Mesh
from joblib import Parallel, delayed

from fmri_decoder.data import SurfaceData, TimeseriesData

__all__ = [
    "TimeseriesPreproc",
    "TimeseriesSampling",
    "FeatureSelection",
]


# number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()


class TimeseriesPreproc(TimeseriesData):
    """Fmri time series preprocessing.

    This class contains several methods for preprocessing of fmri timseries data and
    correponding event files. During object instantiation, all fmri time series and
    event files are loaded into memory. This is done for parallelization purposes of
    preprocessing steps. But be aware that enough memory can be allocated.

    Attributes:
        file_series: List of time series file names.
        file_events: List of corresponding event files.
        tr: Repetition time in seconds.
        data: Loaded fmri time series arrays.
        conds: Loaded corresponding event files.

    """

    def __init__(
        self, file_series: list[str], file_events: list[str], tr: float
    ) -> None:
        super().__init__(file_series, file_events)
        self.tr = tr

        # initialize attributes
        self.data: list[np.ndarray] | Any = [
            self.load_timeseries(i).get_fdata() for i, _ in enumerate(self.file_series)
        ]
        _onsets: list[Any] = [
            self.load_events(i)[1] for i, _ in enumerate(self.file_events)
        ]
        _durations: list[Any] = [
            self.load_events(i)[2] for i, _ in enumerate(self.file_events)
        ]
        self.conds: list[Any] = [
            self.parse_events(_ons, _durs) for _ons, _durs in zip(_onsets, _durations)
        ]

    def detrend_timeseries(
        self, tr: float, cutoff_sec: float
    ) -> list[np.ndarray] | Any:
        """Apply high-pass filtering to all voxel time courses of all loaded fmri time
        series to remove baseline drifts.

        Args:
            tr: Repetition time in seconds.
            cutoff_sec: Cutoff frequency for time series detrending in 1/Hz.

        Returns:
            Detrendend fmri time series arrays.
        """
        print("Detrend timeseries ...")
        _data = Parallel(n_jobs=NUM_CORES)(
            delayed(self._detrend)(_d, tr, cutoff_sec) for _d in self.data
        )
        self.data = _data
        return _data

    def demean_timeseries(self) -> list[np.ndarray] | Any:
        """Remove the teporal mean from all voxel time courses of all loaded fmri time
        series.

        Returns:
            Demeaned fmri time series arrays.
        """
        print("Demean timeseries ...")
        _data = Parallel(n_jobs=NUM_CORES)(
            delayed(self._demean)(_d) for _d in self.data
        )
        self.data = _data
        return _data

    def crop_data(self, n_skip: int) -> tuple:
        """Remove transient time points from fmri time series and corresponding event
        arrays.

        Args:
            n_skip: Number of skipped initial volumes per experimental condition.

        Returns:
            Cropped fmri time series and event arrays.
        """
        print("Crop timeseries ...")
        _res: Any = Parallel(n_jobs=NUM_CORES)(
            delayed(self._crop)(_d, _c, n_skip) for _d, _c in zip(self.data, self.conds)
        )
        # sort resulting array
        self.data = [np.array(r[0]) for r in _res]
        self.conds = [np.array(r[1]) for r in _res]
        return self.data, self.conds

    def parse_events(self, onsets: list[Any], durations: list[Any]) -> np.ndarray:
        """Transform information from the condition array into a label vector.

        Args:
            onsets: Nested list with onset times for each experimental condition.
            durations: Nested list with durations for each experimental condition.

        Returns:
            Label vector that indicates the experimental condition for each fmri volume.
        """
        # durations in number of volumes
        dur: Any = [int(i / self.tr) for i in durations]
        labels = [[i] * len(l_) for i, l_ in enumerate(onsets)]

        ons = self._flatten(onsets)
        labels = self._flatten(labels)
        labels = [x for _, x in sorted(zip(ons, labels))]

        return np.array([l_ for l_ in labels for _ in range(dur[l_])])

    def _demean(self, arr: np.ndarray) -> np.ndarray:
        """Helper function for demeaning used for parallelization purposes."""
        scale = ScaleTimeseries(arr)
        return scale.demean()

    def _detrend(self, arr: np.ndarray, tr: float, cutoff: float) -> np.ndarray:
        """Helper function for detrending used for parallelization purposes."""
        filt = FilterTimeseries(arr, tr)
        return filt.detrend(cutoff, store_dc=True)

    def _crop(self, arr: np.ndarray, events: np.ndarray, n_skip: int) -> tuple:
        """Helper function for data cropping used for parallelization purposes."""
        # remove transient volumes
        flag = np.ones(len(events))
        for i, (last, curr) in enumerate(
            zip(events[:-n_skip], events[1 : -n_skip + 1])
        ):
            if last != curr:
                flag[i + 1 : i + n_skip + 1] = 0
        arr = arr[:, :, :, flag == 1]
        events = events[flag == 1]

        # remove baseline condition
        flag = np.ones(len(events))
        flag[events == 0] = 0
        arr = arr[:, :, :, flag == 1]
        events = events[flag == 1]
        return arr, events

    @staticmethod
    def _flatten(lst: list) -> list:
        """Helper function to flatten nested list."""
        return [item for sublist in lst for item in sublist]

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Load from dictionary."""
        return cls(
            file_series=data_dict["file_series"],
            file_events=data_dict["file_events"],
            tr=data_dict["TR"],
        )

    @classmethod
    def from_yaml(cls, file_yaml: str):
        """Deserialize from yaml file."""
        with open(file_yaml, "r", encoding="utf8") as yml:
            config = dict()
            try:
                config = yaml.safe_load(yml)
            except yaml.YAMLError as exc:
                print(exc)

            return cls(
                file_series=config["file_series"],
                file_events=config["file_events"],
                tr=config["TR"],
            )

    @property
    def tr(self) -> float:
        """Get TR."""
        return self._tr

    @tr.setter
    def tr(self, tr_: float) -> None:
        """Set TR."""
        if tr_ <= 0.0:
            raise ValueError("TR must be greater than zero!")
        self._tr = tr_


class TimeseriesSampling:
    """Fmri time series sampling.

    This class provides methods to sample fmri time series data in volume format onto
    a surface mesh. If both mesh and volume are not aligned, surface vertices can first
    be transformed to the space of fmri time series using a coordinate mapping.
    Optionally, sampled time series can spatially be filtered using a bandpass filter
    (Laplacian of Gaussian).

    Attributes:
        vtx: Array of surface vertices.
        fac: Array of corresponding faces.
        data: List of fmri time series array.
        data_sampled: Initialized list of sampled time series arrays.

    """

    # number of iterations for bandpas filter size center frequency estimation.
    N_ITER = 1

    def __init__(
        self, vtx: np.ndarray, fac: np.ndarray, data: list[np.ndarray]
    ) -> None:
        self.verts = vtx
        self.faces = fac
        self.data = data

        self.data_sampled: list[np.ndarray] | Any = []

    def sample_timeseries(
        self,
        file_deformation: Optional[str] = None,
        file_reference: Optional[str] = None,
    ) -> list[np.ndarray] | Any:
        """Sample fmri time series onto a surface mesh.

        Args:
            file_deformation: File name of deformation field that contains the
            transformation of surface coordinates to the space of fmri time series.
            Defaults to None.
            file_reference: File name of volume in target space to infer array
            dimensions and voxel sizes. Defaults to None.

        Returns:
            Array of sampled time series points.
        """
        print("Sample timeseries ...")
        mesh = Mesh(self.verts, self.faces)
        if file_deformation:
            mesh.transform_coords(file_deformation, file_reference)
        header = nb.load(file_reference).header
        dims = header["dim"][1:4]
        ds = header["pixdim"][1:4]
        _data = Parallel(n_jobs=NUM_CORES)(
            delayed(self._sample)(mesh.verts, _d, dims, ds) for _d in self.data
        )
        self.data_sampled = _data
        return _data

    def filter_timeseries(
        self, label: np.ndarray, filter_size: float
    ) -> list[np.ndarray] | Any:
        """Apply spatial bandpass filter to each time point of a sampled fmri time
        series.

        Args:
            label: Array of vertex indices to restrict the application of the bandpass
            filter within a region of interest (due to computational reaons). All other
            data points will be set to nan.
            filter_size: Bandpass filter size.

        Raises:
            ValueError: If time series sampling was not done.

        Returns:
            Array of filtered time series points.
        """
        if not self.data_sampled:
            raise ValueError("Time series sampling not done!")
        print("Filter timeseries ...")
        # check dimensions
        mesh = Mesh(self.verts, self.faces)
        surf_roi = mesh.remove_vertices(label)
        verts = surf_roi[0]
        faces = surf_roi[1]
        _res: Any = Parallel(n_jobs=NUM_CORES)(
            delayed(self._filter)(verts, faces, filter_size, _d[label])
            for _d in self.data_sampled
        )
        _data0 = np.empty_like(self.data_sampled[0])
        _data: list[np.ndarray] = []
        for d in _res:
            _data0[:] = np.nan
            _data0[label] = d
            _data.append(_data0)
        self.data_sampled = _data
        return _data

    def save_timeseries(self, file_out: str | Path, run: int) -> None:
        """Save sampled time series to disk. The numpy array is saved in .hdf5 file
        format.

        Args:
            file_out: File name of saved time series file.
            run: Select the time series to be saved from a list of input time series.

        Raises:
            ValueError: If time series sampling was not done.
        """
        if self.sample_timeseries is None:
            raise ValueError("Timeseries sampling not done!")
        write_hdf5(file_out, self.data_sampled[run])

    def filter_scale(self, label: np.ndarray, filter_size: float) -> dict:
        """Estimate the bandwidth of the applied Laplacian of gaussian bandpass filter.
        The center frequency of the bandpass filter is estimated by applying the filter
        to white noise sampled onto a surface mesh and computing the mean distance
        between local minima and maxima.

        Args:
            label: Array of vertex indices to restrict the application of the bandpass
            filter within a region of interest (due to computational reaons). All other
            data points will be set to nan.
            filter_size: Bandpass filter size.

        Returns:
            Dictionary collecting the output under the following keys

            * period (float): Mean spatial cycle period of the filter.
            * freq (float): Mean spatial frequency of the filter.
            * mean (float): Expected value of the log-normal distribution.
            * variance (float): Variance of the log-normal distribution.
            * length (np.ndarray): Min-Max distance distribution.
        """
        mesh = Mesh(self.verts, self.faces)
        surf_roi = mesh.remove_vertices(label)
        verts = surf_roi[0]
        faces = surf_roi[1]
        filt = LaplacianGaussian(verts, faces, filter_size)
        return filt.spatial_scale(n_iter=TimeseriesSampling.N_ITER)

    def _sample(
        self, _v: np.ndarray, _d: np.ndarray, dims: tuple, ds: tuple
    ) -> np.ndarray:
        """Helper function for time series sampling used for parallelization
        purposes."""
        return map_timeseries(_v, _d, dims, ds)

    def _filter(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        filter_size: float,
        arr_sampled: np.ndarray,
    ) -> np.ndarray:
        """Helper function for time series filtering used for parallelization
        purposes."""
        filt = LaplacianGaussian(verts, faces, filter_size)
        return filt.apply(arr_sampled)


class FeatureSelection(SurfaceData):
    """Feature selection.

    This class performs feature selection based on separate localizer data. From the
    localizer data, the best (most responsive) data points are selected, i.e., the
    vertices whose data is most responsive. Optionally, a minimum distances between
    neighboring vertices can be set.

    Attributes:
        file_layer: Dictionary with a list of surface file names per hemisphere.
        file_localizer: Dictionary with a list of overlay file names per hemisphere.
        file_label: Dictionary with a list of labels file names per hemisphere.
        features: Initialized dictionary of selected features.

    """

    def __init__(
        self,
        file_layer: dict,
        file_localizer: dict,
        file_label: dict,
    ) -> None:
        super().__init__(file_layer, file_localizer, file_label)

        # initialize attributes
        self.features: dict = {}
        self.features["hemi"] = np.empty(0)
        self.features["label"] = np.empty(0)
        self.features["coords"] = np.empty((0, 3))

    def sort_features(
        self, radius: Optional[float] = None, nmax: Optional[int] = None
    ) -> dict:
        """Selects a set of features (vertices) on the basis of separate localizer data.
        Localizer data is sorted in reverse order (max -> min) and the best data points
        that show highest response are selected. Since data selection is performed on
        data after sampling onto a surface mesh, the same data point can be selected
        multiple times depending on the resolution of the surface mesh. Therefore,
        a radius can be set that enforces a minimum distance between neighboring
        selected vertices.

        Args:
            radius: Minimum distance between selected features. Defaults to None.
            nmax: Number of considered features (data points). Defaults to None.

        Returns:
            Dictionary collecting the output under the following keys

            * hemi (np.ndarray): Array that with the hemisphere of selected features.
            * label (np.ndarray): Array with label indices of selected features.
            * coords (np.ndarray): Array with vertex coordinates of selected features.
        """
        # load data array, index array, and label array
        data, hemisphere, label = self._get_data()

        # sort index array
        index = np.arange(len(data), dtype=int)
        index_sorted = np.array([x for _, x in sorted(zip(data, index), reverse=True)])
        # sort hemisphere and label arrays
        tmp_hemi = hemisphere[index_sorted]
        tmp_label = label[index_sorted]

        # initialize corresponding coords from the first surface mesh in the list of
        # provided surfaces. This should correspond to the white matter surface if input
        # surface are sorted in cortical depth from deep to superficial layers.
        tmp_coords = np.zeros((len(tmp_hemi), 3))
        for i, hemi in enumerate(["lh", "rh"]):
            vtx = self.load_layer(hemi, n=0)[0]
            tmp_coords[tmp_hemi == i] = vtx[tmp_label[tmp_hemi == i]]

        # add features to the final feature list with optional minimum distance between
        # neighboring features
        while True:
            # append
            self.features["hemi"] = np.append(self.features["hemi"], tmp_hemi[0])
            self.features["label"] = np.append(self.features["label"], tmp_label[0])
            self.features["coords"] = np.append(
                self.features["coords"], [tmp_coords[0, :]], axis=0
            )
            # remove
            tmp_hemi = tmp_hemi[1:]
            tmp_label = tmp_label[1:]
            tmp_coords = tmp_coords[1:]

            if radius:
                x_dist = (tmp_coords[:, 0] - self.features["coords"][-1, 0]) ** 2
                y_dist = (tmp_coords[:, 1] - self.features["coords"][-1, 1]) ** 2
                z_dist = (tmp_coords[:, 2] - self.features["coords"][-1, 2]) ** 2
                dist = np.sqrt(x_dist + y_dist + z_dist)
                dist[dist <= radius] = 0
                dist[dist != 0] = 1
                # remove all inside sphere
                tmp_hemi = tmp_hemi[dist == 1]
                tmp_label = tmp_label[dist == 1]
                tmp_coords = tmp_coords[dist == 1]

            if not tmp_label.size > 0:
                break

        if nmax:
            self.features["hemi"] = self.features["hemi"][:nmax]
            self.features["label"] = self.features["label"][:nmax]
            self.features["coords"] = self.features["coords"][:nmax]

        return self.features

    def save_features(self, file_out: str | Path) -> None:
        """Save selected features as .label file to disk.

        Args:
            file_out: File name of written label file.

        Raises:
            ValueError: If features selection was not done.
            ValueError: If hemisphere was not specified in the file name.
        """
        if not self.features["hemi"].size > 0 or not self.features["label"].size > 0:
            raise ValueError("Feature selection not done!")
        hemi = os.path.basename(file_out)[:2]
        if hemi not in ["lh", "rh"]:
            raise ValueError("Hemisphere not specified in file name!")
        i = 0 if hemi == "lh" else 1
        write_label(
            file_out, self.features["label"][np.where(self.features["hemi"] == i)[0]]
        )

    def _get_data(self) -> tuple:
        """Load localizer and label data.

        Returns:
            Flattened numpy arrays of localizer data, corresponding hemispheres, and
            corresponding vertex indices.
        """
        data = [self.load_localizer_average(hemi) for hemi in ["lh", "rh"]]
        label = [self.load_label_intersection(hemi) for hemi in ["lh", "rh"]]

        data = [data[i][label[i]] for i in range(len(data))]
        hemisphere = [[i] * len(data[i]) for i in range(len(data))]
        data = self._flatten(data)
        hemisphere = self._flatten(hemisphere)
        label = self._flatten(label)
        return np.array(data), np.array(hemisphere), np.array(label)

    @staticmethod
    def _flatten(lst: list) -> list:
        """Helper function to flatten nested list."""
        return [item for sublist in lst for item in sublist]
