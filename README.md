# fmri-decoder

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python package for multivariate pattern analysis of fMRI data, intended to analyze high-resolution fMRI data across cortical depth.

Author
---
Daniel Haenelt &lt;<daniel.haenelt@gmail.com>&gt;

Installation
---
I recommend to use `Miniconda` to create a new python environment with `Python ~= 3.10`. Then, clone this repository and run the following line from the directory in which the repository was cloned with the environment being activated:

```
pip install .
```

To install the package in editable mode, you can use the command `pip install -e .` or `poetry install` if [poetry](https://python-poetry.org/) is installed.

Usage
---

The indented way to use this package is by its command-line interface.

```
python -m fmri_decoder --in <yaml_file> --out <output directory>
```

| Flags   | |
|---------|-|
| `--in`  | The location of a .yaml file that contains all input file names and parameter settings. Fruther information about the configuration is given in the exemplary yaml file in the root directory of this repository [config_decoder.yaml](https://github.com/haenelt/fmri-decoder/blob/main/config_decoder.yaml). |
| `--out` | The location of the output directory to which results are written. |

| Optional flags | |
|---------|-|
| `--only-preprocessing` | Do not perform MVPA analysis and only perform preprocessing steps. A pandas dataframe preprocessed data will be saved to disk. Defaults to False. |
| `--verbose` | Dump out additional files and print more information to the console. Defaults to False. |

Processing steps
---

### Feature selection
Selected features are vertices on a given surface mesh from which raw fMRI time series volumes are sampled. First, a region of interest (ROI) is specified by one or multiple FreeSurfer .label files. If multiple files are given, the intersection of labels is used. Within the ROI, data from an independent dataset is read from a FreeSurfer MGH overlay and ranked by its magnitude. This can be data from an independent localizer contrast. If multiple overlays are given, the average across files is computed before ranking. The most responsive `nmax` vertices across hemispheres are then taken as features. Since features are defined on a surface mesh which can have a higher mesh resolution compared to fMRI data, a minimum `radius` can be specified that enforces neighboring features to keep a minimum distance to each other.

### Time series preprocessing
Data samples are individual fMRI raw volumes (single time points) mapped onto a given surface mesh. All raw volumes (within and between time series) are expected to be aligned to each other (motion correction). Class labels are read from condition files in .mat file format that are compatible to SPM12. These are expected to contain a baseline condition and two experimental conditions for binary classificiation.

Before time series mapping, large-scale trends are removed from individual time series. Baseline volumes are then removed and the `n_skip` volumes at the start of each experimental condition are discarded to remove any transient responses from further analysis.

Fmri time series are then mapped onto the surface geometry. Optionally, sampled time series can spatially be filtered using a bandpass filter (Laplacian of Gaussian). [[1]](#1)

### MVPA
The MVPA analysis is initiated by summararizing samples from all features within a pandas dataframe. The dataframe will contain the following columns: 

| batch | label | feature 0    | feature 1    | ...          |
| ----- | ----- | ------------ | ------------ | ------------ |
| 0     | 1     | &lt;data&gt; | &lt;data&gt; | &lt;data&gt; |
| 0     | 0     | &lt;data&gt; | &lt;data&gt; | &lt;data&gt; |
| ...   | ...   | ...          | ...          | ...          |

The first column *batch* indicates the belonging to individual fMRI time series with consecutive integer numbers. The batch information is used in the cross-validation procedure. The *label* column contains the class label and is either 0 or 1. All other columns contain sample data for individual selected features.

Sample data can be scaled before training. Scaling can be performed for individual features and/or individual samples. Individual features can be scaled using the *min_max* or the *standard* method, which either rescales individual features into the range [0, 1] or standardizes them using the standard *z*-score. Individual samples can be scaled using the *norm* or the *standard* method, which normalizes individual samples by their L2-norm or standardizes samples using the standard *z*-score, respectively.

For binary classification, a support vector machine is used in a leave-one-run out cross-validation procedure. I.e., all but one batch is used for training of the classifier and the remaining batch is used for prediction. This procedure is repeated until all runs have been used for prediction. Resulting scores (accuracy, sensitivity, specificity, f1) are saved as .csv file to disk.

Surface sampling and MVPA is performed for each given cortical surface geometry. Resulting scores from the repeated analysis are appended in the .csv file in consecutive rows.

If the `--verbose` flag is used, the following additional files are saved in subdirectories of the output directory:

- selected features as FreeSurfer .label files
- pandas dataframe as .parquet file
- trained SVM models as .z file (for each fold in the cross-validation procedure)

Some snippets
---
With the following code, the mean spatial cycle period of the applied spatial bandpass filter can be computed. The cycle period is estimated by applying the bandpass filter to spatial white noise and computing the mean distance between found local minima and maxima of the filtered image.

```python
from nibabel.freesurfer.io import read_geometry, read_label
from fmri_decoder.preprocessing import TimeseriesSampling

file_surf = ""  # file name of FreeSurfer geometry
file_label = ""  # file name of FreeSurfer label file
t = 0.5  # gaussian filter size

# load data
vtx, fac = read_geometry(file_surf)
label = read_label(file_label)

sampler = TimeseriesSampling(vtx, fac, [])
res = sampler.filter_scale(label, t)
print(res["period"])  # mean spatial cycle period of the filter
```

References
---
<a id="1">[1]</a> Chen, Y., Cichy, R., Stannat, W. & Haynes, J.-D.. Scale-specific analysis of fMRI data on the irregular cortical surface. *NeuroImage* **181** (2018). 