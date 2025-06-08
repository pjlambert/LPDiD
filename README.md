# LPDiD: Local Projections Difference-in-Differences for Python

A Python implementation of the Local Projections Difference-in-Differences (LP-DiD) estimator proposed by Dube, Girardi, Jordà, and Taylor (2023).  Developed by Peter John Lambert and Yannick Schindler.

## Features

- **Binary treatment support**: Both absorbing and non-absorbing treatments
- **Instrumental Variables**: Built-in IV support for endogenous treatment
- **Treatment effect heterogeneity**: Interaction terms to explore effect variation
- **Formula interface**: R-style formulas for controls, fixed effects, and IV
- **Wild bootstrap inference**: Cluster-robust wild bootstrap standard errors
- **Multi-way clustering**: Support for multiple clustering variables
- **Flexible estimation**: Event study and pooled treatment effects
- **Parallel processing**: Built-in support for parallel computation
- **Multiple control groups**: Never-treated, not-yet-treated, or clean controls
- **Reweighting options**: Variance-weighted or equally-weighted ATEs
- **Pre-mean differencing**: Alternative baseline specifications
- **Comprehensive diagnostics**: First-stage F-stats, weak IV tests, and more
- **Sample Weights**: Permits sample weights to allow for matching/PSM/CEM

## Installation

```bash
git clone https://github.com/yourusername/lpdid.git
cd lpdid
pip install -e .
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.1.0
- pyfixest >= 0.18.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0
- joblib >= 1.0.0
- wildboottest >= 0.1.0 (optional, for wild bootstrap)

## Quick Start

```python
import pandas as pd
from lpdid import LPDiD

# Basic LP-DiD estimation
lpdid = LPDiD(
    data=data,
    depvar='outcome',           # Outcome variable
    unit='entity_id',           # Unit identifier
    time='period',              # Time variable
    treat='treatment',          # Binary treatment indicator
    pre_window=5,              # Pre-treatment periods
    post_window=10,            # Post-treatment periods
    formula="~ x1 + x2 | fe1 + fe2",  # Controls and fixed effects
    n_jobs=-1                  # Use all CPU cores
)

# Fit the model
results = lpdid.fit()

# View results
print(results.event_study)
print(results.pooled)

# Summary report
results.summary()
```

## Advanced Usage

### Instrumental Variables

For endogenous treatment, use the three-part formula syntax:

```python
# IV specification
lpdid_iv = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ controls | fixed_effects | D_treat ~ instrument1 + instrument2",
    wildbootstrap=999,
    n_jobs=-1
)

results = lpdid_iv.fit()

# Check IV diagnostics
print(results.iv_diagnostics)  # First-stage F-statistics
print(results.first_stage)      # First-stage coefficients
```

### Treatment Effect Heterogeneity

Explore how treatment effects vary with observable characteristics:

```python
# Continuous interaction
lpdid_het = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    interactions="~ size",  # Creates D_treat × size interaction
    n_jobs=-1
)

# Multiple interactions
lpdid_multi = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    interactions="~ size + i.region",  # Size and region interactions
    n_jobs=-1
)

results = lpdid_het.fit()

# Extract interaction effects
event_study = results.event_study
print(event_study[['horizon', 'coefficient', 'size_interaction', 'size_interaction_se']])
```

### IV with Heterogeneous Effects

Combine IV and interactions:

```python
lpdid_iv_het = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ controls | FE | D_treat ~ instruments",
    interactions="~ size + i.industry",
    wildbootstrap=999,
    n_jobs=-1
)

results = lpdid_iv_het.fit()
results.summary()  # Comprehensive output including all diagnostics
```

### Formula Interface

The package supports R-style formulas with three parts:

```python
# Basic formula
formula="~ control1 + control2"  # Only controls

# With fixed effects
formula="~ control1 + control2 | fe1 + fe2"  # Controls and FEs

# With IV
formula="~ controls | FE | endog ~ instruments"  # Full specification

# Factor variables (categorical)
formula="~ x1 + i.category | FE"  # i.category creates dummies
```

### Wild Bootstrap Inference

```python
# Wild bootstrap with single clustering
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    wildbootstrap=999,        # Number of bootstrap iterations
    seed=123,                 # For reproducibility
    n_jobs=-1
)
```

### Multi-way Clustering

```python
# Multi-way clustering
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | fe1",
    cluster_formula="~ cluster1 + cluster2",  # Multiple clusters
    wildbootstrap=999,  # Uses first cluster for wild bootstrap
    n_jobs=-1
)
```

### Weighted Estimation

```python
# User-provided weights
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    weights='population_weight',  # Weight variable
    n_jobs=-1
)

# Combine with reweighting for equally-weighted ATE
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    weights='population_weight',
    rw=True,                     # Also reweight for equal weights
    n_jobs=-1
)
```

### Non-absorbing Treatment

```python
# For treatments that can turn on and off
lpdid_nonabs = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    nonabsorbing=(5, False, False),  # (L, notyet, firsttreat)
    cluster_formula="~ cluster_var",
    wildbootstrap=999,
    n_jobs=-1
)
```

### Control Group Selection

```python
# Use only never-treated units as controls
lpdid_never = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | fe1",
    nevertreated=True,
    n_jobs=-1
)
```

### Pre-mean Differencing

```python
# Use average of all pre-treatment periods as baseline
lpdid_pmd = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    pmd='max',              # Use all available pre-periods
    wildbootstrap=999,
    n_jobs=-1
)
```

## Main Parameters

- `data`: Panel DataFrame
- `depvar`: Outcome variable name
- `unit`: Unit identifier variable
- `time`: Time variable
- `treat`: Binary treatment indicator
- `pre_window`: Pre-treatment periods to estimate (≥2)
- `post_window`: Post-treatment periods to estimate (≥0)
- `formula`: R-style formula for controls, fixed effects, and IV
- `interactions`: Variables to interact with treatment
- `cluster_formula`: R-style formula for clustering variables
- `ylags`: Number of outcome lags to include
- `dylags`: Number of first-differenced outcome lags
- `nonabsorbing`: Tuple (L, notyet, firsttreat) for non-absorbing treatment
- `nevertreated`: Use only never-treated as controls
- `nocomp`: Avoid composition changes
- `rw`: Reweight for equally-weighted ATE
- `pmd`: Pre-mean differencing specification
- `wildbootstrap`: Number of wild bootstrap iterations
- `weights`: Weight variable
- `n_jobs`: Number of parallel jobs

## Output

The `fit()` method returns a `LPDiDResults` object containing:

- `event_study`: DataFrame with period-by-period treatment effects
  - Includes main effects and interaction terms if specified
- `pooled`: DataFrame with averaged pre/post effects
- `iv_diagnostics`: DataFrame with first-stage F-statistics and weak IV indicators
- `first_stage`: DataFrame with first-stage regression coefficients
- `summary()`: Method to print comprehensive results

Each results DataFrame includes:
- `coefficient`: Point estimate
- `se`: Standard error
- `t`: t-statistic
- `p`: p-value
- `ci_low`, `ci_high`: Confidence interval
- `obs`: Number of observations
- Interaction coefficients and their SEs (if interactions specified)
- IV diagnostics (if IV specified)

## Interpreting Results

### Basic Model
- `coefficient`: Average treatment effect on the treated (ATT)

### With Interactions
- `coefficient`: Base effect (when interaction variables = 0)
- `var_interaction`: How effect changes with the variable
- Total effect = `coefficient + Σ(interaction_coef × var_value)`

### With IV
- `coefficient`: Local average treatment effect (LATE) for compliers
- Check `first_stage_F > 10` to avoid weak instruments
- `iv_diagnostics` provides diagnostic information

## Citation

If you use this package, please cite:

Dube, A., D. Girardi, Ò. Jordà and A. M. Taylor. 2023. "A Local Projections Approach to Difference-in-Differences." NBER Working Paper 31184.
Maintained by Peter John Lambert (p.j.lambert@lse.ac.uk).

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
