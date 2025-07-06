# LP-DiD Quick Start Guide

## Installation

```bash
pip install numpy pandas pyfixest scipy ray matplotlib tqdm
# Then install lpdid (from source for now)
git clone https://github.com/pjlambert/LPDiD.git
cd LPDiD
pip install -e .
```

## Basic Example (5 minutes)

```python
import pandas as pd
from LPDiD import LPDiD

# Load your data
df = pd.read_csv('your_panel_data.csv')

# Run LP-DiD
lpdid = LPDiD(
    data=df,
    depvar='outcome',
    unit='firm_id',
    time='year',
    treat='treated',
    pre_window=3,
    post_window=5,
    formula="~ control1 + control2 | industry",
    n_jobs=-1  # Use all CPU cores with Ray
)

# Get results
results = lpdid.fit()

# View event study
print(results.event_study)

# Get summary
results.summary()
```

## Common Use Cases

### 1. Simple DiD with Controls

```python
lpdid = LPDiD(
    data=df,
    depvar='revenue',
    unit='firm_id',
    time='quarter',
    treat='policy',
    pre_window=4,
    post_window=8,
    formula="~ size + age | industry + region"
)
results = lpdid.fit()
```

### 2. Heterogeneous Treatment Effects

```python
# How do effects vary by firm characteristics?
lpdid_het = LPDiD(
    data=df,
    depvar='productivity',
    unit='firm_id',
    time='year',
    treat='training_program',
    pre_window=2,
    post_window=4,
    formula="~ baseline_prod | industry",
    interactions="~ size + tech_intensity"
)
results = lpdid_het.fit()

# Check interaction terms - columns are named as {var}_{value}_coef
print(results.event_study[['horizon', 'coefficient', 'size_1_coef', 'size_1_se']])
```

### 3. Robust Standard Errors

```python
# Wild bootstrap for better inference
lpdid_robust = LPDiD(
    data=df,
    depvar='wages',
    unit='firm_id',
    time='year',
    treat='min_wage_increase',
    pre_window=3,
    post_window=5,
    formula="~ employment | industry + county",
    cluster_formula="~ state",  # Cluster at state level
    wildbootstrap=999,  # Wild bootstrap iterations
    seed=123  # For reproducibility
)
results = lpdid_robust.fit()

# Note: Wild bootstrap adjusts p-values and t-statistics only
# Standard errors and confidence intervals remain analytical
```

### 4. Multi-way Clustering

```python
# Two-way clustering
lpdid_2way = LPDiD(
    data=df,
    depvar='output',
    unit='firm_id',
    time='year',
    treat='policy',
    pre_window=3,
    post_window=5,
    formula="~ controls | industry",
    cluster_formula="~ state + year",  # 2-way clustering
    n_jobs=-1
)

# Three or more clustering variables
lpdid_3way = LPDiD(
    data=df,
    depvar='employment',
    unit='firm_id',
    time='year',
    treat='regulation',
    pre_window=3,
    post_window=5,
    formula="~ size | industry",
    cluster_formula="~ state + industry + year",  # 3-way clustering
    n_jobs=-1
)
```

### 5. Advanced Control Period Selection

```python
# Use minimum time controls for better comparability
lpdid_min_controls = LPDiD(
    data=df,
    depvar='employment',
    unit='firm_id',
    time='year',
    treat='policy_change',
    pre_window=4,
    post_window=6,
    formula="~ size + industry_controls | region",
    min_time_controls=True,  # Use earlier period for controls
    n_jobs=-1
)
results = lpdid_min_controls.fit()

# Filter units based on status at control periods
lpdid_selection = LPDiD(
    data=df,
    depvar='profits',
    unit='firm_id',
    time='year',
    treat='regulation',
    pre_window=3,
    post_window=5,
    formula="~ controls | industry",
    min_time_selection='active==1',  # Only firms active at control period
    n_jobs=-1
)
results = lpdid_selection.fit()

# Swap pre-treatment difference direction
lpdid_swap = LPDiD(
    data=df,
    depvar='wages',
    unit='firm_id',
    time='year',
    treat='policy',
    pre_window=4,
    post_window=6,
    formula="~ size + controls | industry",
    swap_pre_diff=True,  # Use y_{t-1} - y_{t-h} for pre-treatment periods
    n_jobs=-1
)
results = lpdid_swap.fit()

# Combine multiple advanced features
lpdid_combined = LPDiD(
    data=df,
    depvar='revenue',
    unit='firm_id',
    time='year',
    treat='treatment',
    pre_window=4,
    post_window=6,
    formula="~ size + age | industry",
    min_time_controls=True,
    min_time_selection='operational==1',  # Custom boolean condition
    swap_pre_diff=True,
    n_jobs=-1
)
results = lpdid_combined.fit()
```

**When to use these features:**
- `min_time_controls=True`: When you want control variables from the earlier period of each long-difference
- `min_time_selection`: When units may exit/enter the sample and you need them to satisfy conditions at control periods
- `swap_pre_diff=True`: When you want to reverse the direction of pre-treatment differences
- Common conditions: `'alive==1'`, `'employed==1'`, `'status>0'`, `'active==True'`

### 6. Data Building for Custom Analysis

```python
# Build data without running regressions
lpdid = LPDiD(
    data=df,
    depvar='sales',
    unit='firm_id',
    time='year',
    treat='treatment',
    pre_window=4,
    post_window=6,
    formula="~ size | industry"
)

# Just build the long-format data
lpdid.build()

# Access the prepared data
long_data = lpdid.get_long_diff_data()
print(f"Generated {len(long_data)} observations")

# Use for custom analysis
import statsmodels.formula.api as smf
pooled_model = smf.ols('Dy ~ D_treat + C(h)', data=long_data).fit()
print(pooled_model.summary())

# Later run LP-DiD regressions if desired
results = lpdid.fit()  # Uses already built data - fast!
```

### 7. Poisson Model for Count Data

```python
from LPDiD import LPDiDPois

# For count outcomes (e.g., number of patents, accidents)
lpdid_pois = LPDiDPois(
    data=df,
    depvar='patent_count',
    unit='firm_id',
    time='year',
    treat='r_and_d_subsidy',
    pre_window=3,
    post_window=5,
    formula="~ size + rd_stock | industry",
    n_jobs=-1
)

results_pois = lpdid_pois.fit()
results_pois.summary()
```

## Interpreting Results

### Event Study Output

```
horizon  coefficient    se      p     ci_low   ci_high   obs
-3       0.021        0.045   0.642  -0.067   0.109     1200
-2       0.015        0.038   0.694  -0.059   0.089     1250
-1       0.000        0.000   NaN     0.000   0.000     1300
0        0.234        0.072   0.001   0.093   0.375     1300
1        0.456        0.084   0.000   0.291   0.621     1250
2        0.523        0.091   0.000   0.345   0.701     1200
```

- **Pre-treatment** (negative horizons): Should be near zero (parallel trends)
- **Horizon -1**: Reference period (always zero)
- **Post-treatment** (positive horizons): Treatment effects over time

### Heterogeneous Effects

With interactions, columns are named as `{variable}_{value}_coef`:

```
horizon  coefficient  industry_1_coef  industry_1_p  industry_2_coef  industry_2_p
0        0.150       0.250           0.001         0.180           0.023
1        0.289       0.456           0.000         0.334           0.001
2        0.356       0.523           0.000         0.412           0.000
```

- Each group gets its own coefficient column
- Total effect for group = that group's specific coefficient

## Performance Tips

1. **Ray Parallelization**: Set `n_jobs=-1` for automatic parallel processing
2. **Memory Efficiency**: Default settings (`lean=True`, `copy_data=False`) optimize memory
3. **Data Building**: Use `build()` to prepare data once for multiple analyses
4. **Large Datasets**: Use vectorized NumPy operations (automatic)

## Common Issues

### "Regression failed"
- Check for sufficient variation in treatment
- Ensure panel is balanced enough
- Try fewer horizons or controls

### Slow performance
- Ensure `n_jobs=-1` for parallelization
- Reduce wild bootstrap iterations for testing
- Consider using `build()` first if running multiple models

### Wild Bootstrap Note
- Wild bootstrap only adjusts p-values and t-statistics
- Standard errors and confidence intervals remain analytical
- This is intentional and follows best practices

## Need Help?

- Full documentation: See README.md
- GitHub issues: https://github.com/pjlambert/LPDiD/issues
