# -*- coding: utf-8 -*-
"""Module for performing binary classification from multi-voxel pattern using a
support vector machine."""

import csv
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, shapiro
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC

__all__ = ["MVPA"]

plt.style.use(Path(__file__).parent / "default.mplstyle")


class MVPA:
    """Multi-voxel pattern analysis (MVPA).

    This class performs binary classification on fmri time series data sampled onto a
    surface mesh. A dataframe is loaded which is used for fitting and predicting using a
    leave-one-run-out cross-validataion procedure. The dataframe must contain the
    following columns: batch, label, feature 0, feature 1, ...
    The batch column assigns each row its corresponding fold, i.e., folds are already
    set while loading and no random splitting ist performed. This is done to prevent
    any information leakage due to finite auto-correlation in time series data. The
    label column contains class labels for binary classification. These are expected to
    be either 0 or 1. All other columns (feature 0, feature 1, ...) contain sample data
    per feature.

    Attributes:
        dtf: Dataframe with sample data.
        model_trained: List of objects containing already trained SVM models. Defaults
        to None.
        nmax: Number of considered features (data points). Defaults to None.
        y_predict: Initialize array of predicted class labels.

    """

    # hyperparameters
    C: float = 1.0  # SVM regularization parameter
    kernel: str = "linear"  # SVM kernel type

    # color cycle for plots
    colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def __init__(
        self,
        dtf: pd.DataFrame,
        model_trained: Optional[list] = None,
        nmax: Optional[int] = None,
    ) -> None:
        self.dtf = dtf
        self.model_trained = model_trained  # trained svm model
        self.nmax = nmax
        self.y_predict: Optional[list] = None  # predicted classification

    @property
    def n_batch(self) -> int:
        """Number of batches."""
        return np.unique(self.dtf["batch"]).size

    @property
    def label_names(self) -> np.ndarray:
        """Class label names."""
        return np.unique(self.dtf["label"])

    @property
    def feature_names(self) -> list[str]:
        """Feature names."""
        return list(self.dtf.columns[2:])

    def describe_data(self, features: list[str], fold: int) -> None:
        """Print some descriptive statistics to console. Descriptive statistics is
        computed for a list of selected features and for one fold.

        Args:
            features: Select the list of features to be analyzed.
            fold: Select the fold to be analyzed.

        Raises:
            ValueError: If an invalid fold is selected.
        """
        if fold < 0 or fold >= self.n_batch:
            raise ValueError("Fold does not match number of batches!")
        data_summary = pd.DataFrame()
        data = self.dtf.loc[self.dtf["batch"] != fold, features].to_numpy()
        data_summary["counts"] = [len(data[:, j]) for j in range(len(features))]
        data_summary["mean"] = np.mean(data, axis=0)
        data_summary["std"] = np.std(data, axis=0)
        data_summary["min"] = np.min(data, axis=0)
        data_summary["max"] = np.max(data, axis=0)
        data_summary["q(25%)"] = np.quantile(data, 0.25, axis=0)
        data_summary["q(50%)"] = np.quantile(data, 0.50, axis=0)
        data_summary["q(75%)"] = np.quantile(data, 0.75, axis=0)
        data_summary["r (shapiro)"], data_summary["p (shapiro)"] = zip(
            *[shapiro(data[:, j]) for j in range(len(features))]
        )
        print(data_summary)

    def check_balance(self, fold: int) -> None:
        """Check balance between class labels.

        Args:
            fold: Select the fold to be analyzed.

        Raises:
            ValueError: If an invalid fold is selected.
        """
        if fold < 0 or fold >= self.n_batch:
            raise ValueError("Fold does not match number of batches!")
        label = self.dtf.loc[self.dtf["batch"] != fold, "label"]
        balance_0 = len(label[label == 1]) / len(label) * 100
        balance_1 = 100 - balance_0
        print(f"Balance (class 0/class 1) in pu: {balance_0:.1f}/{balance_1:.1f}")

    def explore_data(self, features: list[str], fold: int) -> None:
        """Show some descriptive plots (histogram, scatter) to visualize the sample
        distribution between classes for a list of selected features and for one fold.

        Args:
            features: Select the list of features to be analyzed.
            fold: Select the fold to be analyzed.

        Raises:
            ValueError: If an invalid fold is selected.
        """
        if fold < 0 or fold >= self.n_batch:
            raise ValueError("Fold does not match number of batches!")
        __, axs = plt.subplots(len(features), len(features))
        axs = np.array(axs)
        for i, feat_i in enumerate(features):
            for j, feat_j in enumerate(features):
                data = self.dtf.loc[self.dtf["batch"] != fold, feat_i].to_numpy()
                data2 = self.dtf.loc[self.dtf["batch"] != fold, feat_j].to_numpy()
                label = self.dtf.loc[self.dtf["batch"] != fold, "label"].to_numpy()

                if i == j:
                    for class_ in self.label_names:
                        self._show_hist(
                            data[label == class_],
                            axis=axs[i, j],
                        )

                if i != j and i > j:
                    for class_ in self.label_names:
                        self._show_scatter(
                            data[label == class_],
                            data2[label == class_],
                            axis=axs[i, j],
                        )

                # add annotations
                if i == j:
                    axs[i, j].text(
                        0.75,
                        0.75,
                        feat_i,
                        fontsize=12,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axs[i, j].transAxes,
                    )
                if i < len(features) - 1:
                    axs[i, j].set_xticks([])
                if j > 0:
                    axs[i, j].set_yticks([])
                if i != j and i <= j:
                    axs[i, j].set_visible(False)

                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=f"Class {c}",
                        markerfacecolor=MVPA.colors[c],
                        markersize=5,
                    )
                    for c in self.label_names
                ]
                axs[len(features) - 1, 0].legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    fancybox=False,
                    shadow=False,
                    ncol=2,
                )
        plt.show()

    def select_features(self, fold: int, y_train: Optional[np.ndarray] = None) -> list:
        """Select nmax best features based on ANOVA F-values of the training sample.

        Args:
            fold: Select the fold to be analyzed.
            y_train: Alternative class label, e.g. for estimation of a null distribution
            using shuffled sampled. Defaults to None.

        Raises:
            ValueError: If not enough features are available.

        Returns:
            List of selected feature names.
        """
        if self.nmax and len(self.dtf.columns) < 2 + self.nmax:
            raise ValueError("Not enough features available or nmax is not set!")
        X_train = np.array(self.dtf.loc[self.dtf["batch"] != fold, self.feature_names])
        if y_train is None:
            y_train = np.array(self.dtf.loc[self.dtf["batch"] != fold, "label"])

        # discard features with non-numeric values
        _feature_index = np.sum(X_train, axis=0)
        _feature_names = [
            feat
            for feat, idx in zip(self.feature_names, _feature_index)
            if np.isfinite(idx)
        ]
        X_train = np.array(self.dtf.loc[self.dtf["batch"] != fold, _feature_names])

        f_statistic = f_classif(X_train, y_train)[0]
        index = np.arange(len(_feature_names))
        index_sorted = np.array(
            [x for _, x in sorted(zip(f_statistic, index), reverse=True)]
        )
        index_sorted = index_sorted[: self.nmax]
        return [_feature_names[i] for i in index_sorted]

    def scale_features(self, method: str) -> None:
        """Scale features per fold. Within one fold, each feature is scaled
        independently.

        Args:
            method: min_max (scale in the range [0, 1]) or standard (standardize using
            standard score).

        Raises:
            ValueError: If an unknown method is selected.
        """
        for i in range(self.n_batch):
            data = self.dtf.loc[self.dtf["batch"] == i, self.feature_names]

            min_ = np.tile(np.min(data, axis=0), (len(data), 1))
            max_ = np.tile(np.max(data, axis=0), (len(data), 1))
            mu_ = np.tile(np.mean(data, axis=0), (len(data), 1))
            sd_ = np.tile(np.std(data, axis=0), (len(data), 1))
            if method == "min_max":
                data = (data - min_) / (max_ - min_)
            elif method == "standard":
                data = (data - mu_) / sd_
            else:
                raise ValueError("Unknown method!")

            self.dtf.loc[self.dtf["batch"] == i, self.feature_names] = data

    def scale_samples(self, method: str) -> None:
        """Scale samples. Samples are scaled individually across features. See
        `Haynes`_ for reference of the norm method used in MVPA analysis.

        Args:
            method: norm (normalize by dividing by its l2 norm) or standard (standardize
            using standard score)

        Raises:
            ValueError: If an unknown method is selected.

        .._Haynes:
            Haynes, J-D, Rees, G, Predicting the orientation of invisible stimuli from
            acrivity in human primary visual cortex, Nature Neurosci 8, 5, 686--691
            (2005).
        """
        n_features = len(self.feature_names)
        data = self.dtf[self.feature_names]
        mu_ = np.tile(np.mean(data, axis=1), (n_features, 1)).T
        sd_ = np.tile(np.std(data, axis=1), (n_features, 1)).T
        norm_ = np.tile(np.sqrt(np.sum(data**2, axis=1)), (n_features, 1)).T
        if method == "norm":
            data /= norm_
        elif method == "standard":
            data = (data - mu_) / sd_
        else:
            raise ValueError("Unknown method!")

        self.dtf[self.feature_names] = data

    @property
    def fit(self) -> list[SVC]:
        """Fit individual models per batch.

        Returns:
            List of trained models.
        """
        self.model_trained = []
        for i in range(self.n_batch):
            model = SVC(C=MVPA.C, kernel=MVPA.kernel)
            if self.nmax:
                _feature_names = self.select_features(i)
            else:
                _feature_names = self.feature_names
            X_train = np.array(self.dtf.loc[self.dtf["batch"] != i, _feature_names])
            y_train = np.array(self.dtf.loc[self.dtf["batch"] != i, "label"])
            self.model_trained.append(model.fit(X_train, y_train))
        return self.model_trained

    def predict(self, dtf: Optional[pd.DataFrame] = None) -> list[float]:
        """Predict new class labels based on a fitted model.

        Args:
            dtf: Test on a dataframe from another dataset. By default the left out
            fold from the same dataset is used for prediction. Defaults to None.

        Raises:
            ValueError: If fitting was not performed.

        Returns:
            Array of predicted class labels
        """
        if not self.model_trained or len(self.model_trained) != self.n_batch:
            raise ValueError(
                "Trained models do not match batch size! Did you forget to train?"
            )
        if not dtf:
            dtf = self.dtf.copy()
        self.y_predict = []
        for i in range(self.n_batch):
            if self.nmax:
                _feature_names = self.select_features(i)
            else:
                _feature_names = self.feature_names
            X_test = np.array(dtf.loc[dtf["batch"] == i, _feature_names])
            self.y_predict.append(self.model_trained[i].predict(X_test))
        return self.y_predict

    @property
    def scores(self) -> NamedTuple:
        """For each fold, some evaluation metrics are computed.

        Returns:
            NamedTuple collecting outputs under the following fields

            * accuracy : accuracy score.
            * sensitivity : true positive rate.
            * specificity : true negative rate.
            * f1 : balanced F-measure.
        """
        if not self.y_predict:
            raise ValueError("No model predictions were found!")
        accuracy = []
        sensitivity = []
        specificity = []
        f1 = []
        for i in range(self.n_batch):
            y_test = np.array(self.dtf.loc[self.dtf["batch"] == i, "label"])

            accuracy.append(self._accuracy(y_test, self.y_predict[i]))
            sensitivity.append(self._sensitivity(y_test, self.y_predict[i]))
            specificity.append(self._specificity(y_test, self.y_predict[i]))
            f1.append(self._f1(y_test, self.y_predict[i]))

        res = namedtuple("res", ["accuracy", "sensitivity", "specificity", "f1"])
        return res(accuracy, sensitivity, specificity, f1)

    def null_dist(self, n_iter: int = 1000) -> NamedTuple:
        """Generate a null distribution for each fold by permutation of class labels. It
        should be kept in mind that this method ignores any autocorrealtion in the
        fmri time series signal.

        Args:
            n_iter: Number of permutation iterations. Defaults to 1000.

        Returns:
            NamedTuple collecting outputs under the following fields

            * accuracy : null distributions for accuracy score.
            * sensitivity : null distribution for true positive rate.
            * specificity : null distribution for true negative rate.
            * f1 : null distritbution for balanced F-measure.
        """
        null_accuracy = np.zeros((n_iter, self.n_batch))
        null_sensitivity = np.zeros((n_iter, self.n_batch))
        null_specificity = np.zeros((n_iter, self.n_batch))
        null_f1 = np.zeros((n_iter, self.n_batch))
        for i in range(n_iter):
            # initialize svm model
            model = SVC(C=MVPA.C, kernel=MVPA.kernel)

            # shuffle class labels within each fold
            _y = np.empty(shape=0)
            for j in range(self.n_batch):
                y_batch = np.array(self.dtf.loc[self.dtf["batch"] == j, "label"])
                np.random.shuffle(y_batch)
                _y = np.append(_y, y_batch)

            for j in range(self.n_batch):
                y_train = _y[self.dtf["batch"] != j]
                y_test = _y[self.dtf["batch"] == j]
                # select features
                if self.nmax:
                    _feature_names = self.select_features(j, y_train)
                else:
                    _feature_names = self.feature_names
                # training
                X_train = np.array(self.dtf.loc[self.dtf["batch"] != j, _feature_names])
                model_trained = model.fit(X_train, y_train)
                # testing
                X_test = np.array(self.dtf.loc[self.dtf["batch"] == j, _feature_names])
                y_predict = model_trained.predict(X_test)
                # get score
                null_accuracy[i, j] = self._accuracy(y_test, y_predict)
                null_sensitivity[i, j] = self._sensitivity(y_test, y_predict)
                null_specificity[i, j] = self._specificity(y_test, y_predict)
                null_f1[i, j] = self._f1(y_test, y_predict)

        res = namedtuple("res", ["accuracy", "sensitivity", "specificity", "f1"])
        return res(null_accuracy, null_sensitivity, null_specificity, null_f1)

    @property
    def evaluate(self) -> NamedTuple:
        """Train and test the model using a cross-validation procedure in one row.

        Returns:
            Resulting scores (accuracy, sensitivity, specificity, f1).
        """
        _ = self.fit
        _ = self.predict()
        return self.scores

    def evaluate_stats(self, n_iter: int = 1000, metric: str = "accuracy"):
        """Tests mean (across folds) scoring metric for statistical significance. A null
        distribution is generated using permutation sampling based on which a one-sided
        p-value is computed.

        Args:
            n_iter: Number of permutation iterations. Defaults to 1000.
            metric: Selected metric (accuracy, sensitivity, specificity, f1). Defaults
            to "accuracy".

        Returns:
            One-sided p-value.
        """
        _ = self.fit
        _ = self.predict()
        score = self.scores
        null = self.null_dist(n_iter)  # generate null distribution
        res = np.mean(getattr(score, metric))  # average score
        res_null = np.mean(getattr(null, metric), axis=1)  # average null distribution
        return len(res_null[res_null > res]) / n_iter

    def show_results(self, metric: str = "accuracy") -> None:
        """Print metric to console.

        Args:
            metric: Selected metric (accuracy, sensitivity, specificity, f1). Defaults
            to "accuracy".
        """
        res = self.scores
        print(f"{metric} (mean): {np.mean(getattr(res, metric)):.6f}")
        print(f"{metric} (std): {np.std(getattr(res, metric)):.6f}")

    def save_results(self, file_out: str | Path, metric: str = "accuracy") -> None:
        """Save metric to disk (*.csv file format). Note that data is appended to the
        file if already exists.

        Args:
            file_out: File name of written file.
            metric: Selected metric (accuracy, sensitivity, specificity, f1). Defaults
            to "accuracy".
        """
        res = self.scores
        with open(file_out, "a", encoding="utf-8") as fid:
            writer = csv.writer(fid)  # create the csv writer
            writer.writerow(getattr(res, metric))  # append a row

    def save_stats(
        self, file_out: str | Path, n_iter: int = 1000, metric: str = "accuracy"
    ) -> None:
        """Save one-sided p-value for scoring metric to disc (*csv file format). Note
        that data is appended to the file if already exists.

        Args:
            file_out: File name of written file.
            n_iter: Number of permutation iterations. Defaults to 1000.
            metric: Selected metric (accuracy, sensitivity, specificity, f1). Defaults
            to "accuracy".
        """
        with open(file_out, "a", encoding="utf-8") as fid:
            writer = csv.writer(fid)  # create the csv writer
            writer.writerow([self.evaluate_stats(n_iter, metric)])

    def save_model(self, file_out: str | Path) -> None:
        """Dump list of fitted models (*.z file format) to disk.

        Args:
            file_out: File name of written file.

        Raises:
            ValueError: If fitting was not performed.
        """
        if not self.model_trained:
            raise ValueError("No trained model found!")
        joblib.dump(self.model_trained, file_out)

    def save_dataframe(self, file_out: str | Path) -> None:
        """Save dataframe to disk (*.parquet file format).

        Args:
            file_out: File name of written file.
        """
        self.dtf.to_parquet(file_out, engine="pyarrow")

    def _accuracy(self, y_test: np.ndarray, y_predict: np.ndarray) -> float:
        """Accuracy."""
        tp_ = self._tp(y_test, y_predict)
        tn_ = self._tn(y_test, y_predict)
        fp_ = self._fp(y_test, y_predict)
        fn_ = self._fn(y_test, y_predict)
        return self._safe_division(tp_ + tn_, tp_ + tn_ + fp_ + fn_)

    def _sensitivity(self, y_test: np.ndarray, y_predict: np.ndarray) -> float:
        """True positive rate (recall)."""
        tp_ = self._tp(y_test, y_predict)
        fn_ = self._fn(y_test, y_predict)
        return self._safe_division(tp_, tp_ + fn_)

    def _specificity(self, y_test: np.ndarray, y_predict: np.ndarray) -> float:
        """True negative rate."""
        tn_ = self._tn(y_test, y_predict)
        fp_ = self._fp(y_test, y_predict)
        return self._safe_division(tn_, tn_ + fp_)

    def _positive_predictive_value(
        self, y_test: np.ndarray, y_predict: np.ndarray
    ) -> float:
        """Positive predictive value (precision)."""
        tp_ = self._tp(y_test, y_predict)
        fp_ = self._fp(y_test, y_predict)
        return self._safe_division(tp_, tp_ + fp_)

    def _negative_predictive_value(
        self, y_test: np.ndarray, y_predict: np.ndarray
    ) -> float:
        """Negative predictive value."""
        tn_ = self._tn(y_test, y_predict)
        fn_ = self._fn(y_test, y_predict)
        return self._safe_division(tn_, tn_ + fn_)

    def _f1(self, y_test: np.ndarray, y_predict: np.ndarray) -> float:
        """F1 score."""
        precision_ = self._positive_predictive_value(y_test, y_predict)
        recall_ = self._sensitivity(y_test, y_predict)
        return self._safe_division(2 * precision_ * recall_, precision_ + recall_)

    @staticmethod
    def _tp(y_test: np.ndarray, y_predict: np.ndarray) -> int:
        """True positives."""
        tmp = y_test + y_predict
        return len(tmp[tmp == 2])

    @staticmethod
    def _tn(y_test: np.ndarray, y_predict: np.ndarray) -> int:
        """True negatives."""
        tmp = y_test + y_predict
        return len(tmp[tmp == 0])

    @staticmethod
    def _fp(y_test: np.ndarray, y_predict: np.ndarray) -> int:
        """False positives."""
        tmp = y_test + 2 * y_predict
        return len(tmp[tmp == 2])

    @staticmethod
    def _fn(y_test: np.ndarray, y_predict: np.ndarray) -> int:
        """False negatives."""
        tmp = 2 * y_test + y_predict
        return len(tmp[tmp == 2])

    @staticmethod
    def _safe_division(x: float, y: float) -> float:
        """Helper function to handle division by zero."""
        return np.divide(x, y) if y != 0 else np.nan

    @staticmethod
    def _show_scatter(
        x_data: np.ndarray,
        y_data: np.ndarray,
        axis: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_line: Optional[bool] = False,
    ) -> None:
        """Helper method for scatter plot."""
        slope, y_intercept = np.polyfit(x_data, y_data, 1)
        xx_min = np.min(x_data)
        xx_max = np.max(x_data)
        xx_ = np.linspace(xx_min, xx_max, 1000)
        yy_ = slope * xx_ + y_intercept

        axis.scatter(x_data, y_data)
        if show_line:
            axis.plot(xx_, yy_)
        if xlabel:
            axis.set_xlabel(xlabel)
        if ylabel:
            axis.set_ylabel(ylabel)

    @staticmethod
    def _show_hist(
        data: np.ndarray,
        axis: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_kde: bool = False,
    ) -> None:
        """Helper method for histogram plot."""
        kernel = gaussian_kde(data)
        _, bins, _ = axis.hist(data, edgecolor="white", alpha=0.5)
        bin_width = bins[1] - bins[0]
        xx_min = np.min(bins) - bin_width / 2
        xx_max = np.max(bins) + bin_width / 2
        xx_ = np.linspace(xx_min, xx_max, 1000)
        yy_ = kernel.evaluate(xx_) * bin_width * 1000
        if show_kde:
            axis.plot(xx_, yy_)
        if xlabel:
            axis.set_xlabel(xlabel)
        if ylabel:
            axis.set_ylabel(ylabel)

    @classmethod
    def from_file(
        cls,
        file_dataframe: str,
        file_model: Optional[str] = None,
        nmax: Optional[int] = None,
    ):
        """Alternative constructor for MVPA from saved file.

        Args:
            file_dataframe: File name of saved pandas dataframe.
            file_model: File name of already fitted models. Defaults to None.
            nmax: Number of considered features (data points). Defaults to None.
        """
        dtf = pd.read_parquet(file_dataframe, engine="pyarrow")
        model = joblib.load(file_model) if file_model else None
        return cls(dtf, model, nmax)

    @classmethod
    def from_selected_data(
        cls,
        data: dict,
        features: dict,
        label: list[np.ndarray],
        model_trained: Optional[list] = None,
    ):
        """Alternative contructor for MVPA from a loaded data dictionary. Data from
        single hemispheres are separated in the dictionary by the keys lh and rh. The
        feature dictionary is expected to have the keys label and hemi which contain
        selected vertex indices and corresponding hemisphere. Hemisphere must be
        indicated by integer values (0: lh and 1: rh).

        Args:
            data: Data arrays.
            features: Selected features.
            label: List of class label arrays.
            model_trained: List of already fitted models. Defaults to None.
        """
        n_features = len(features["label"])
        columns = ["batch", "label"] + [f"feature {i}" for i in range(n_features)]
        dtf = pd.DataFrame(columns=columns)
        for i, y in enumerate(label):
            arr = np.zeros((len(y), n_features + 2))
            arr_tmp = np.zeros((len(y), n_features))
            arr[:, 0] = i
            arr[:, 1] = y - 1  # rescale class labels [1, 2] -> [0, 1]
            for j, hemi in enumerate(["lh", "rh"]):
                idx = np.array(features["label"][features["hemi"] == j], dtype=int)
                arr_tmp[:, features["hemi"] == j] = data[hemi][i][idx, :].T
            arr[:, 2:] = arr_tmp[:, :n_features]
            dtf = pd.concat([dtf, pd.DataFrame(arr, columns=columns)])

        dtf["batch"] = dtf["batch"].astype(int)
        dtf["label"] = dtf["label"].astype(int)

        return cls(dtf, model_trained, None)

    @classmethod
    def from_data(
        cls,
        data: dict,
        label: list[np.ndarray],
        model_trained: Optional[list] = None,
        nmax: Optional[int] = None,
        remove_nan: bool = False,
    ):
        """Alternative contructor for MVPA from a loaded data dictionary. Data from
        single hemispheres are separated in the dictionary by the keys lh and rh.

        Args:
            data: Data arrays.
            label: List of class label arrays.
            model_trained: List of already fitted models. Defaults to None.
            nmax: Number of considered features (data points). Defaults to None.
            remove_nan: Discard non-numeric columns in data.
        """
        if remove_nan is True:
            for hemi in ["lh", "rh"]:
                ind = np.sum(data[hemi], axis=(0, 2))
                data[hemi] = [
                    data[hemi][x][~np.isnan(ind), :] for x in range(len(data[hemi]))
                ]

        n_features = np.size(data["lh"][0], 0) + np.size(data["rh"][0], 0)
        columns = ["batch", "label"] + [f"feature {i}" for i in range(n_features)]
        dtf = pd.DataFrame(columns=columns)
        for i, y in enumerate(label):
            arr = np.zeros((len(y), n_features + 2))
            arr[:, 0] = i
            arr[:, 1] = y - 1  # rescale class labels [1, 2] -> [0, 1]
            arr[:, 2:] = np.concatenate((data["lh"][i], data["rh"][i])).T
            dtf = pd.concat([dtf, pd.DataFrame(arr, columns=columns)])

        dtf["batch"] = dtf["batch"].astype(int)
        dtf["label"] = dtf["label"].astype(int)

        return cls(dtf, model_trained, nmax)

    @classmethod
    def from_iris(cls, noise_sd: Optional[float] = None):
        """Alternative constructor for MVPA on iris example data set.

        The iris data set is a famous database from `Fisher`_, which is used here as an
        example data set. This data set is comprised of 3 classes of 50 instances each.
        Each class refers to a type of iris plant (iris setosa, iris versicolour, iris
        virginica). Each sample contains 4 attributes (sepal length in cm, sepal
        width in cm, petal lengh in cm, petal width in cm). The full data set has thus a
        shape of (150 samples, 4 attributes).
        For the current demonstration of binary classification, only the first two
        classes are considered. Furthermore, samples are shuffled and divided into 10
        batches of the same size, which are used as training/test sets in a
        cross-validation procedure. Optionally, gaussian noise can be added to data
        samples to complicate correct classification.

        Args:
            noise_sd: Standard deviation of added gaussian noise. Defaults to None.

        .._Fisher:
            Fisher, RA, The use of multiple measurements in taxonomic problems, Annual
            Eugenics 7, Part II, 179--188 (1936).
        """
        X, y = load_iris(return_X_y=True)  # fetch iris data set
        X = np.array(X)
        y = np.array(y)
        n_batch = 10  # number of folds
        feature_names = [f"feature {i}" for i in range(np.shape(X)[1])]

        # add gaussian noise
        if noise_sd:
            np.random.seed(42)
            noise = np.random.normal(0, noise_sd, np.shape(X))
            X += noise

        # return pandas dataframe
        # the third class with label 2 is removed for binary classification
        # all samples are shuffled
        dtf = pd.DataFrame(X, columns=feature_names)
        dtf.insert(0, "label", y)
        dtf.drop(dtf.loc[dtf["label"] == 2].index, inplace=True)
        dtf = dtf.sample(frac=1, random_state=42).reset_index(drop=True)

        # add batch column
        # the data set contains 100 samples in total and is divided into 10 batches
        # therefore, each batch contains 10 samples
        data_length = float(dtf.shape[0])
        batch_ = np.repeat(np.arange(n_batch), int(data_length / n_batch))
        dtf.insert(0, "batch", batch_)
        return cls(dtf, None, None)

    @property
    def dtf(self) -> pd.DataFrame:
        """Get dtf."""
        return self._dtf

    @dtf.setter
    def dtf(self, df_: pd.DataFrame) -> None:
        """Set dtf."""
        nb_ = np.unique(df_["batch"]).size  # number of batches
        ln_ = np.unique(df_["label"])  # class label names
        fn_ = list(df_.columns[2:])  # feature names

        if not isinstance(df_, pd.DataFrame):
            raise TypeError("dtf must be a pandas dataframe!")
        if not np.isfinite(df_.iloc[:, 2:].to_numpy()).all():
            raise ValueError("dtf contains non-numeric data!")
        if not isinstance(nb_, int):
            raise TypeError("Number of batches must be an integer!")
        if not nb_ > 0:
            raise ValueError("Number of batches must be positive!")
        if not isinstance(ln_, np.ndarray):
            raise ValueError("Label names must be a numpy array!")
        if not ln_.ndim == 1 or not ln_.size == 2:
            raise ValueError("Label names must have shape=(2,)!")
        if any(i not in [0, 1] for i in ln_):
            raise ValueError("Label names must be 0 and 1 for binary classification!")
        if not isinstance(fn_, list):
            raise TypeError("Feature names must be a list!")
        if any(f != f"feature {i}" for i, f in enumerate(fn_)):
            raise ValueError("Please check feature names!")

        self._dtf = df_
