# fastEDM

The `fastEDM` Python package implements a series of *Empirical Dynamic
Modeling* tools that can be used for *causal analysis of time series*
data.

Key features of the package:

- powered by a fast multi-threaded *C++ backend*,
- able to process panel data, a.k.a. *multispatial EDM*,
- able to handle *missing data* using new `dt` algorithms or by dropping
  points.

## Installation

You can install the development version of fastEDM from
[GitHub](https://github.com/EDM-Developers/fastEDM-python/) with:

``` bash
pip install 'fastEDM @ git+https://github.com/EDM-Developers/fastEDM-python'
```

## Stata & R Packages

This Python package, and the R `fastEDM` package, is port of our [EDM Stata
package](https://edm-developers.github.io/edm-stata/). As all packages
share the same underlying C++ code, their behaviour will be identical.
If you plan to adjust some of the various low-level EDM parameters,
check out the documentation of the Stata package for more details on
their options and behaviours.

## Other Resources

This site serves as the primary source of documentation for the package, though there is also:

- our [Stata Journal paper](https://jinjingli.github.io/edm/edm-wp.pdf) which explains the package and the overall causal framework, and
- Jinjing's QMNET seminar on the package, the recording is on [YouTube](https://youtu.be/kZv85k1YUVE) and the [slides are here](https://github.com/EDM-Developers/edm-stata/raw/main/docs/pdfs/EDM-talk-QMNET.pdf).

## Authors

- [Jinjing Li](https://www.jinjingli.com/) (author),
- [Michael Zyphur](https://business.uq.edu.au/profile/14074/michael-zyphur) (author),
- [Patrick Laub](https://pat-laub.github.io/) (author, maintainer),
- Edoardo Tescari (contributor),
- Simon Mutch (contributor),
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
