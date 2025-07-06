# LPDiD: Local Projections Difference-in-Differences for Python

**NOTE:** This is still in the early building stages!

A Python implementation of the Local Projections Difference-in-Differences (LP-DiD) estimator proposed by Dube, Girardi, Jordà, and Taylor (2023). Developed by Peter John Lambert and Yannick Schindler.

## Features

- **Binary treatment support**: Both absorbing and non-absorbing treatments
- **Treatment effect heterogeneity**: Interaction terms to explore effect variation  
- **Formula interface**: R-style formulas for controls and fixed effects
- **Wild bootstrap inference**: Cluster-robust wild bootstrap for p-values and t-statistics
- **Multi-way clustering**: Support for multiple clustering variables (2+ clusters)
- **Flexible estimation**: Event study and pooled treatment effects
- **Ray-based parallel processing**: Distributed computing with Ray for 2-8x speedups
- **Multiple control groups**: Never-treated or clean controls
- **Sample weights**: Permits sample weights to allow for matching/PSM/CEM
- **Poisson estimation**: LPDiDPois for count data and log-linear models
- **Data building**: Separate data preparation from estimation for flexibility
- **Long-format export**: Access to prepared long-difference data for custom analyses

## Installation

```bash
git clone https://github.com/pjlambert/LPDiD.git
cd LPDiD
pip install -e .
```

## Requirements

- Python >= 3.7
- numpy >= 1.19.0
- pandas >= 1.1.0
- pyfixest >= 0.18.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0
- ray >= 2.0.0
- tqdm >= 4.60.0

## Quick Start

```python
import pandas as pd
from LPDiD import LPDiD

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
    n_jobs=-1                  # Use all CPU cores with Ray
)

# Fit the model
results = lpdid.fit()

# View results
print(results.event_study)

# Summary report
results.summary()
```

## Advanced Usage

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
print(event_study[['horizon', 'coefficient', 'size_coef', 'size_se']])
```

### Formula Interface

The package supports R-style formulas:

```python
# Basic formula
formula="~ control1 + control2"  # Only controls

# With fixed effects
formula="~ control1 + control2 | fe1 + fe2"  # Controls and FEs

# Factor variables (categorical)
formula="~ x1 + i.category | FE"  # i.category creates dummies
```

### Wild Bootstrap Inference

**Note:** Wild bootstrap adjusts p-values and t-statistics only. Standard errors and confidence intervals remain analytical.

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
    n_jobs=-1
)

# Three or more clustering variables
lpdid_3way = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | fe1",
    cluster_formula="~ state + industry + year",  # 3-way clustering
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

### Control Period Selection

The package provides advanced options for selecting control periods in long differences:

```python
# Use minimum time controls for more comparable control periods
lpdid_min_controls = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    min_time_controls=True,  # Use earlier period for controls
    n_jobs=-1
)

# Filter units based on conditions at control periods
lpdid_selection = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    min_time_selection='alive==1',  # Only include units alive at control period
    n_jobs=-1
)

# Swap pre-treatment difference direction
lpdid_swap = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    swap_pre_diff=True,  # Use y_{t-1} - y_{t-h} for pre-treatment
    n_jobs=-1
)

# Combine multiple features
lpdid_combined = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    min_time_controls=True,
    min_time_selection='status>0',  # Custom boolean condition
    swap_pre_diff=True,
    n_jobs=-1
)
```

**Control Period Logic:**
- `min_time_controls=True`: Uses the earlier period of the long-difference for control variables
  - For post-treatment (t-1 to t+h): controls from t-1
  - For pre-treatment (t-h to t-1): controls from t-h
- `min_time_selection`: Filters units based on boolean condition at the control period
- `swap_pre_diff=True`: For pre-treatment periods, computes `y_{t-1} - y_{t-h}` instead of `y_{t-h} - y_{t-1}`

### Data Building and Custom Analysis

Build data without running regressions:

```python
# Build long-format data
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry"
)

# Build data only
lpdid.build()

# Access the long-format data
long_data = lpdid.get_long_diff_data()
print(f"Generated {len(long_data)} observations")

# Use for custom analysis
import statsmodels.formula.api as smf
pooled = smf.ols('Dy ~ D_treat + C(h)', data=long_data).fit()

# Later run regressions if desired
results = lpdid.fit()  # Uses already built data
```

### Poisson Estimation (LPDiDPois)

For count data or when you need log-linear models:

```python
from LPDiD import LPDiDPois

# Poisson LP-DiD for count outcomes
lpdid_pois = LPDiDPois(
    data=data,
    depvar='event_count',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    n_jobs=-1
)

results_pois = lpdid_pois.fit()
results_pois.summary()
```

### Memory Optimization

```python
# For large datasets, use memory-efficient options
lpdid = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    lean=True,       # Memory-efficient regression objects (default)
    copy_data=False, # Don't copy data in pyfixest (default)
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
- `formula`: R-style formula for controls and fixed effects
- `interactions`: Variables to interact with treatment
- `cluster_formula`: R-style formula for clustering variables
- `ylags`: Number of outcome lags to include
- `dylags`: Number of first-differenced outcome lags
- `nonabsorbing`: Tuple (L, notyet, firsttreat) for non-absorbing treatment
- `nevertreated`: Use only never-treated as controls
- `min_time_controls`: Use earlier period for control variables (default: False)
- `min_time_selection`: Boolean condition for unit inclusion at control periods
- `swap_pre_diff`: Swap pre-treatment difference direction (default: False)
- `wildbootstrap`: Number of wild bootstrap iterations
- `weights`: Weight variable
- `n_jobs`: Number of parallel jobs (-1 for all cores)
- `lean`: Memory-efficient regression objects (default: True)
- `copy_data`: Copy data in pyfixest (default: False)
- `seed`: Random seed for reproducibility

## Output

The `fit()` method returns a `LPDiDResults` object containing:

- `event_study`: DataFrame with period-by-period treatment effects
  - Includes main effects and interaction terms if specified
- `summary()`: Method to print comprehensive results

Each results DataFrame includes:
- `horizon`: Event time
- `coefficient`: Point estimate
- `se`: Standard error
- `t`: t-statistic
- `p`: p-value (adjusted by wild bootstrap if used)
- `ci_low`, `ci_high`: Confidence interval
- `obs`: Number of observations

For models with interactions, additional columns:
- `{var}_{value}_coef`: Group-specific coefficients
- `{var}_{value}_se`: Group-specific standard errors
- `{var}_{value}_p`: Group-specific p-values

## Interpreting Results

### Basic Model
- `coefficient`: Average treatment effect on the treated (ATT)
- Pre-treatment coefficients test parallel trends assumption
- Horizon -1 is normalized to zero

### With Interactions
- Group-specific effects are reported as `{variable}_{value}_coef`
- Total effect for a group = that group's specific coefficient

## Performance Tips

1. **Ray Parallelization**: The package uses Ray for distributed computing
   - Automatically initialized when `n_jobs > 1`
   - Handles memory more efficiently than traditional multiprocessing
   
2. **Vectorized Operations**: Long differences are computed using optimized NumPy operations
   - Significantly faster for large datasets
   - Memory-efficient implementation

3. **Data Building**: Use `build()` to prepare data once, then run multiple analyses

## Citation

If you use this package, please cite:

Dube, A., D. Girardi, Ò. Jordà and A. M. Taylor. 2023. "A Local Projections Approach to Difference-in-Differences." NBER Working Paper 31184.

Maintained by Peter John Lambert (p.j.lambert@lse.ac.uk).

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
