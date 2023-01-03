# -*- coding: utf-8 -*-
"""Decoding tools for functional magnetic resonance imaging data. Execute via the
command line with python -m fmri_decoder <args>."""

import argparse
import os
from pathlib import Path

import fmri_decoder
from fmri_decoder.data import DataConfig, ModelConfig, SurfaceData, TimeseriesData
from fmri_decoder.model import MVPA
from fmri_decoder.preprocessing import (
    FeatureSelection,
    TimeseriesPreproc,
    TimeseriesSampling,
)

# description
PARSER_DESCRIPTION = (
    "This program will apply a decoding analysis (binary classification using a "
    "support vector machine) on a set of fmri time series. Fmri time series data are "
    "preprocessed and sampled onto a surface mesh. Multiple surfaces can be used as "
    "input to analyze the decoding performace across cortical depth. Features "
    "(vertices within a region of interest) are selected based on separate localizer "
    "data. Samples are single fMRI time series points and transient responses between "
    "experimental condition are neglected. Optionally, a spatial bandpass filter "
    "(Laplacian of Gaussian) can be applied to individual samples. Please be aware "
    "that multiple files are written to disk."
)
IN_HELP = (
    "file name of a yaml configuration file that contains all selected parameters and "
    "input file names."
)
OUT_HELP = "path of output directory to which all resulting files are written."
PREPROC_HELP = (
    "only perform all preprocessing steps without running the decoding analysis "
    "(default: False)."
)
VERB_HELP = "save all supporting files to disk (default: False)."

# parse arguments from command line
parser = argparse.ArgumentParser(description=PARSER_DESCRIPTION)
parser.add_argument("-i", "--in", type=str, help=IN_HELP, dest="in_", metavar="IN")
parser.add_argument("-o", "--out", type=str, help=OUT_HELP)
parser.add_argument(
    "-p", "--only-preprocessing", default=False, action="store_true", help=PREPROC_HELP
)
parser.add_argument(
    "-v", "--verbose", default=False, action="store_true", help=VERB_HELP
)
args = parser.parse_args()

# run
term_size = os.get_terminal_size()
print("=" * term_size.columns)
print("FMRI DECODER\n".center(term_size.columns))
print(f"author: {fmri_decoder.__author__}")
print(f"version: {fmri_decoder.__version__}")
print("=" * term_size.columns)

# arguments
# ---
# args.in_
# args.out
# args.only_preprocessing
# args.verbose

# make output directory
dir_out = Path(args.out)
dir_out.mkdir(parents=True, exist_ok=True)

dir_res = dir_out / "res"
dir_res.mkdir(parents=True, exist_ok=True)

dir_label = dir_out / "label"
dir_model = dir_out / "model"
if args.verbose:
    dir_label.mkdir(parents=True, exist_ok=True)
    dir_model.mkdir(parents=True, exist_ok=True)

# load data
time_data = TimeseriesData.from_yaml(args.in_)
surf_data = SurfaceData.from_yaml(args.in_)
config_data = DataConfig.from_yaml(args.in_)
config_model = ModelConfig.from_yaml(args.in_)

# features selection
features = FeatureSelection.from_yaml(args.in_)
features_selected = features.sort_features(config_model.radius, config_model.nmax)
if args.verbose:
    for hemi in ["lh", "rh"]:
        features.save_features(dir_label / f"{hemi}.features.label")

# timeseries preprocessing
preproc = TimeseriesPreproc.from_yaml(args.in_)
# detrend time series
_ = preproc.detrend_timeseries(config_data.tr, config_data.cutoff_sec)
# crop time series
data_vol, events = preproc.crop_data(config_data.n_skip)

# iterate over surfaces (layers)
n_surf = len(surf_data.file_layer["lh"])
for i in range(n_surf):
    data_sampled = {}
    for hemi in ["lh", "rh"]:
        vtx, fac = surf_data.load_layer(hemi, i)
        sampler = TimeseriesSampling(vtx, fac, data_vol)
        # sample time series
        file_deformation = config_data.file_deformation
        file_reference = time_data.file_series[0]
        data_sampled[hemi] = sampler.sample_timeseries(file_deformation, file_reference)
        # filter time series
        if config_data.filter_size:
            label = surf_data.load_label_intersection(hemi)
            data_sampled[hemi] = sampler.filter_timeseries(
                label, config_data.filter_size
            )

    mvpa = MVPA.from_data(data_sampled, features_selected, events)
    if args.verbose or args.only_preprocessing:
        mvpa.save_dataframe(dir_out / f"sample_data_{i}.parquet")

    # model preparation and fitting
    if not args.only_preprocessing:
        # scaling
        if config_model.feature_scaling:
            mvpa.scale_features(config_model.feature_scaling)
        if config_model.sample_scaling:
            mvpa.scale_samples(config_model.sample_scaling)
        _ = mvpa.evaluate
        if args.verbose:
            # check balance
            for fold in range(len(time_data.file_series)):
                mvpa.check_balance(fold)
            # save model
            mvpa.save_model(dir_model / f"model_{i}.z")
            # show results
            mvpa.show_results("accuracy")
        # save results
        mvpa.save_results(dir_res / "accuracy.csv", "accuracy")
        mvpa.save_results(dir_res / "sensitivity.csv", "sensitivity")
        mvpa.save_results(dir_res / "specificity.csv", "specificity")
        mvpa.save_results(dir_res / "f1.csv", "f1")

print("Done.")
