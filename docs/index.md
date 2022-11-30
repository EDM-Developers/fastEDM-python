# fastEDM Python Package

## Package Description

<img src="assets/logo-lorenz.svg" align="right" height="200px" width="200px" alt="logo" />

_Empirical Dynamic Modeling (EDM)_ is a way to perform _causal analysis on time series data_. 
The `fastEDM` Python package implements a series of EDM tools, including the convergent cross-mapping algorithm. 

Key features of the package:

- powered by a fast multi-threaded _C++ backend_,
- able to process panel data, a.k.a. _multispatial EDM_,
- able to handle _missing data_ using new `dt` algorithms or by dropping points.

<!-- 
- _factor variables_ can be added to the analysis,
- _multiple distance functions_ available (Euclidean, Mean Absolute Error, Wasserstein),
- [_GPU acceleration_](/gpu) available.
- so-called _coprediction_ is also available,
- forecasting methods will soon be added (WIP).
- training/testing splits can be made in a variety of ways including _cross-validation_,
-->

## Installation

To install the latest version from [Github](https://github.com/EDM-Developers/fastEDM-python/) using `pip` run:

``` bash
pip install 'fastEDM @ git+https://github.com/EDM-Developers/fastEDM-python'
```

## Example: Chicago crime levels and temperature

This example, looking at the causal links between Chicago’s temperature
and crime rates, is described in full in our
[paper](https://jinjingli.github.io/edm/edm-wp.pdf):

``` python
from fastEDM import easy_edm
import pandas as pd

url = "https://github.com/EDM-Developers/fastEDM-r/raw/main/vignettes/chicago.csv"
chicago = pd.read_csv(url)
chicago["Crime"] = chicago["Crime"].diff()

crimeCCMCausesTemp = easy_edm("Crime", "Temperature", data=chicago, verbosity=0)
#> No evidence of CCM causation from Crime to Temperature found.

tempCCMCausesCrime = easy_edm("Temperature", "Crime", data=chicago, verbosity=0)
#> Some evidence of CCM causation from Temperature to Crime found.
```

## Stata & R packages

We have created the [edm Stata package](https://edm-developers.github.io/edm-stata/) and are currently developing this package alongside the [fastEDM R package](https://edm-developers.github.io/fastEDM-r/). The `fastEDM` packages are direct ports of the Stata package to R & Python.
As all the packages share the same underlying C++ code, their behaviour will be identical.

## Other Resources

This site serves as the primary source of documentation for the package, though there is also:

- our [Stata Journal paper](https://jinjingli.github.io/edm/edm-wp.pdf) which explains the package and the overall causal framework, and
- Jinjing's QMNET seminar on the package, the recording is on [YouTube](https://youtu.be/kZv85k1YUVE) and the [slides are here](pdfs/EDM-talk-QMNET.pdf).

## Authors

- [Patrick Laub](https://pat-laub.github.io/) (author, maintainer),
- [Jinjing Li](https://www.jinjingli.com/) (author),
- [Michael Zyphur](https://business.uq.edu.au/profile/14074/michael-zyphur) (author),
- Edoardo Tescari (contributor),
- Simon Mutch (contributor),
- Rishi Dhushiyandan (contributor),
- George Sugihara (originator)

## Citation

Jinjing Li, Michael J. Zyphur, George Sugihara, Patrick J. Laub (2021), _Beyond Linearity, Stability, and Equilibrium: The edm Package for Empirical Dynamic Modeling and Convergent Cross Mapping in Stata_, Stata Journal, 21(1), pp. 220-258

``` bibtex
@article{edm-stata,
  title={Beyond linearity, stability, and equilibrium: The edm package for empirical dynamic modeling and convergent cross-mapping in {S}tata},
  author={Li, Jinjing and Zyphur, Michael J and Sugihara, George and Laub, Patrick J},
  journal={The Stata Journal},
  volume={21},
  number={1},
  pages={220--258},
  year={2021},
}
```
