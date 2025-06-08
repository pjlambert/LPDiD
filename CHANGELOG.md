# Changelog

All notable changes to the lpdid package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-XX

### Added - Initial Release

#### Core Features
- Local Projections Difference-in-Differences (LP-DiD) estimator
- Event study and pooled treatment effect estimation
- Support for absorbing and non-absorbing treatments
- Multiple control group options (never-treated, not-yet-treated, clean controls)
- Pre-mean differencing (PMD) baseline options
- Parallel processing support via joblib

#### Formula Interface
- R-style formula specification for controls and fixed effects
- Three-part formula syntax for instrumental variables
- Separate formulas for clustering variables
- Interaction specification for treatment effect heterogeneity

#### Instrumental Variables
- Built-in 2SLS support via pyfixest
- First-stage F-statistics and weak instrument diagnostics
- Kleibergen-Paap statistics when available
- First-stage coefficient reporting
- Compatible with wild bootstrap and all other features

#### Treatment Effect Heterogeneity
- Interaction terms between treatment and observables
- Support for continuous and categorical moderators
- Automatic inclusion of main effects
- Full inference on interaction coefficients
- Can be combined with IV for heterogeneous LATEs

#### Inference
- Analytical clustered standard errors
- Wild cluster bootstrap inference
- Multi-way clustering support
- Fallback wild bootstrap implementation
- Proper handling of weights in bootstrap

#### Output and Reporting
- Structured results object with DataFrames
- Comprehensive summary method
- IV diagnostics reporting
- First-stage results
- No plotting functionality (by design)

#### Additional Features
- Flexible weighting options
- Variance-weighted or equally-weighted ATE
- User-provided weights
- Outcome lags (levels and differences)
- Additional fixed effects absorption
- Backward compatibility with list-based interface

### Dependencies
- numpy >= 1.19.0
- pandas >= 1.1.0
- pyfixest >= 0.18.0
- scipy >= 1.5.0
- joblib >= 1.0.0
- matplotlib >= 3.3.0
- wildboottest >= 0.1.0 (optional)

### Known Limitations
- Binary treatment only (continuous treatment not yet supported)
- Single treatment variable (no multiple treatments)
- Requires balanced or nearly balanced panels for PMD

### Future Enhancements (Planned)
- GPU acceleration for large datasets
- Additional IV diagnostics (Anderson-Rubin)
- Support for continuous treatments
- Machine learning for covariate selection
- Integration with other causal inference packages

## Notes

This is the initial release of lpdid, providing a Python implementation of the LP-DiD estimator from Dube, Girardi, Jordà, and Taylor (2023). The package aims to replicate and extend the functionality of the Stata lpdid package while leveraging Python's strengths for parallel processing and modern interfaces.