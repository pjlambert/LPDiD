# LP-DiD Quick Start Guide

## Installation

```bash
pip install numpy pandas pyfixest scipy joblib matplotlib
# Then install lpdid (from source for now)
git clone https://github.com/yourusername/lpdid.git
cd lpdid
pip install -e .
```

## Basic Example (5 minutes)

```python
import pandas as pd
from lpdid import LPDiD

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
    n_jobs=-1  # Use all CPU cores
)

# Get results
results = lpdid.fit()

# View event study
print(results.event_study)

# View pooled effects
print(results.pooled)

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

# Check interaction terms
print(results.event_study[['horizon', 'coefficient', 'size_interaction']])
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
```

### 4. Advanced Control Period Selection

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
    min_time_controls=True,  # Use min(t-1, t+h) for control periods
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
    swap_pre_diff=True,  # Use y_t-1 - y_t+h for pre-treatment periods
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
- `min_time_controls=True`: When you want more comparable control periods for long pre-treatment horizons
- `min_time_selection`: When units may exit/enter the sample and you need them to satisfy conditions at control periods
- `swap_pre_diff=True`: When you want to reverse the direction of pre-treatment differences (y_t-1 - y_t+h instead of y_t+h - y_t-1)
- Common conditions: `'alive==1'`, `'employed==1'`, `'status>0'`, `'active==True'`

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

```
horizon  coefficient  size_interaction  size_interaction_p
0        0.150       0.082            0.023
1        0.289       0.134            0.001
2        0.356       0.156            0.000
```

- **coefficient**: Effect for baseline (size = 0)
- **size_interaction**: How effect changes with size
- **Total effect** = coefficient + size_interaction × size_value

## Tips

1. **Start simple**: Run without interactions first
2. **Check pre-trends**: Look at negative horizons
3. **Use parallel**: Set `n_jobs=-1` for speed
4. **Bootstrap iterations**: Start with 499, increase for final results
5. **Save results**: `results_df = results.event_study.copy()`

## Common Issues

### "Regression failed"
- Check for sufficient variation in treatment
- Ensure panel is balanced enough
- Try fewer horizons or controls

### Slow performance
- Reduce horizons
- Use fewer bootstrap iterations initially
- Ensure `n_jobs=-1` for parallelization

## Need Help?

- Full documentation: See README.md
- Examples: Check examples/ directory
- Issues: GitHub issues page