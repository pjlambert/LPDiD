# LP-DiD Python Package Structure

This package implements Local Projections Difference-in-Differences (LP-DiD) for Python, extending the functionality of the Stata `lpdid` package with IV support and treatment effect heterogeneity.

## Directory Structure

```
lpdid/
├── __init__.py            # Package initialization
├── lpdid.py              # Main LP-DiD implementation
├── wildboot_fallback.py  # Wild bootstrap fallback implementation
├── utils.py              # Utility functions for data generation and testing
├── setup.py              # Package setup configuration
├── requirements.txt      # Package dependencies
├── README.md            # Package documentation
├── LICENSE              # License file (add your preferred license)
├── .gitignore           # Git ignore patterns
├── test_lpdid.py        # Unit tests
├── example_usage.py     # Example usage scripts
├── example_iv_interactions.py  # IV and interaction examples
└── PACKAGE_STRUCTURE.md  # This file
```

## Key Components

### lpdid.py
The main module containing:
- `LPDiD`: Main estimator class with formula interface
- `LPDiDResults`: Results container with summary method (no plotting)
- `parse_formula()`: Parse R-style formulas including IV specification
- `parse_interactions()`: Parse interaction specifications
- `parse_cluster_formula()`: Parse clustering formula
- Core methods:
  - `_validate_inputs()`: Input validation
  - `_prepare_data()`: Data preparation and lag creation
  - `_identify_clean_controls()`: Clean control identification
  - `_generate_long_differences()`: Long difference generation
  - `_compute_weights()`: Weight computation for reweighting
  - `_build_regression_formula()`: Construct pyfixest formula with IV/interactions
  - `_run_single_regression()`: Single horizon regression with full diagnostics
  - `fit()`: Main estimation method

### wildboot_fallback.py
Fallback implementation of wild cluster bootstrap when the `wildboottest` package is not available.

### Key Features

1. **Formula Interface**:
   - R-style formulas: `"~ controls | fixed_effects | endog ~ instruments"`
   - Interaction specification: `"~ var1 + var2"`
   - Clustering formula: `"~ cluster1 + cluster2"`
   - Intuitive specification of complex models

2. **Instrumental Variables**:
   - Native support via pyfixest
   - Automatic first-stage diagnostics
   - Weak instrument detection
   - Compatible with all other features

3. **Treatment Effect Heterogeneity**:
   - Interaction terms with treatment
   - Support for continuous and categorical moderators
   - Full inference on interaction effects
   - Combined with IV for heterogeneous LATEs

4. **Wild Bootstrap Inference**:
   - Cluster-robust wild bootstrap standard errors
   - Adapted for IV estimation
   - Handles weighted regression
   - Fallback implementation if package not available

5. **Multi-way Clustering**:
   - Support for multiple clustering variables
   - Proper variance-covariance matrix computation

6. **Parallel Processing**: 
   - Uses `joblib` for parallel computation across horizons
   - Controlled by `n_jobs` parameter
   - Significant speedup for large panels

7. **Flexible Treatment Types**:
   - Absorbing treatment (default)
   - Non-absorbing treatment with clean control windows

8. **Control Group Options**:
   - Not-yet-treated (default)
   - Never-treated only
   - Clean controls for non-absorbing

9. **Estimation Options**:
   - Variance-weighted ATE (default)
   - Equally-weighted ATE (with `rw=True`)
   - Pre-mean differencing (PMD)
   - User-provided weights

10. **Rich Output**:
    - Event study coefficients with interactions
    - Pooled pre/post estimates
    - IV diagnostics and first-stage results
    - Comprehensive summary method
    - No plotting functionality (as requested)

## Installation

### From Source
```bash
git clone <repository>
cd lpdid
pip install -e .
```

### Via pip (when published)
```bash
pip install lpdid
```

## Usage Examples

### Basic Usage with Formula Interface
```python
from lpdid import LPDiD

# Specify controls and fixed effects using formula
lpdid = LPDiD(
    data=df,
    depvar='outcome',
    unit='entity_id',
    time='period',
    treat='treatment',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry + region",
    n_jobs=-1
)

results = lpdid.fit()
results.summary()
```

### IV Estimation
```python
lpdid_iv = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | FE | D_treat ~ instrument1 + instrument2",
    wildbootstrap=999,
    n_jobs=-1
)

results = lpdid_iv.fit()
print(results.iv_diagnostics)  # Check first-stage F-stats
```

### Treatment Effect Heterogeneity
```python
lpdid_het = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    interactions="~ size + i.region",  # Explore heterogeneity
    n_jobs=-1
)
```

### Combined IV + Heterogeneity
```python
lpdid_iv_het = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ controls | FE | D_treat ~ instruments",
    interactions="~ moderator1 + moderator2",
    wildbootstrap=999,
    n_jobs=-1
)
```

## Testing

Run unit tests:
```bash
python -m pytest test_lpdid.py
# or
python test_lpdid.py
```

## Performance Notes

- Parallel processing provides significant speedup for large panels
- IV estimation adds computational overhead
- Wild bootstrap can be intensive; consider fewer iterations for exploration
- Many interaction terms increase computation time
- Memory usage scales with panel size and number of horizons
- For very large datasets, consider:
  - Reducing the number of horizons
  - Using `only_event=True` or `only_pooled=True`
  - Setting `n_jobs` to control memory usage
  - Using analytical SEs instead of bootstrap for initial exploration

## Differences from Stata Version

1. **Enhanced Interface**: Formula-based specification for controls, fixed effects, clustering, and IV
2. **IV Implementation**: Uses pyfixest's native 2SLS instead of manual implementation
3. **Interaction Support**: Built-in support for treatment effect heterogeneity
4. **Regression Backend**: Uses `pyfixest` instead of `reghdfe`
5. **Bootstrap**: Full wild cluster bootstrap implementation
6. **Multi-way Clustering**: Native support for multiple clustering variables
7. **Output**: Returns structured results object with comprehensive diagnostics
8. **No Plotting**: Plotting functionality removed as requested
9. **Parallel Processing**: Built-in parallel computation support

## API Reference

### Main Class
```python
LPDiD(data, depvar, unit, time, treat, 
      pre_window=None, post_window=None,
      formula=None, interactions=None, 
      cluster_formula=None, ...)
```

### Key Methods
- `fit()`: Run estimation
- `results.summary()`: Print comprehensive results

### Formula Syntax
- Controls only: `"~ x1 + x2"`
- With fixed effects: `"~ x1 + x2 | fe1 + fe2"`
- With IV: `"~ x1 | fe1 | endog ~ inst1 + inst2"`
- Interactions: `interactions="~ var1 + i.var2"`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this package, please cite:

Dube, A., D. Girardi, Ò. Jordà and A. M. Taylor. 2023. "A Local Projections Approach to Difference-in-Differences." NBER Working Paper 31184.