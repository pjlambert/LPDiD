# LP-DiD Python Package Enhancements

## Overview

The enhanced LP-DiD package now includes full support for:
1. **Wild cluster bootstrap inference**
2. **Formula-based model specification**
3. **Multi-way clustering**
4. **Advanced weighting options**

## New Features

### 1. Formula Interface

Specify controls and fixed effects using R-style formulas:

```python
# Old way (still supported for backward compatibility)
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    controls=['x1', 'x2'],
    absorb=['industry', 'region']
)

# New way (recommended)
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    formula="~ x1 + x2 | industry + region"
)
```

Formula syntax:
- `"~ controls"` - Only control variables
- `"~ | fixed_effects"` - Only fixed effects
- `"~ controls | fixed_effects"` - Both controls and fixed effects

### 2. Cluster Formula

Specify clustering variables with a formula:

```python
# Single clustering
lpdid = LPDiD(
    ...,
    cluster_formula="~ state"
)

# Multi-way clustering
lpdid = LPDiD(
    ...,
    cluster_formula="~ state + year"
)
```

### 3. Wild Bootstrap Inference

Robust cluster bootstrap standard errors:

```python
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    formula="~ x1 + x2 | industry",
    cluster_formula="~ state",
    wildbootstrap=999,  # Number of bootstrap iterations
    seed=123  # For reproducibility
)
```

Features:
- Rademacher weights at cluster level
- Handles weighted regression
- Automatic fallback if wildboottest package not available
- Works with both event study and pooled estimates

### 4. Enhanced Weighting

Combine user weights with reweighting:

```python
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    formula="~ x1 + x2 | industry",
    weights='population_weight',  # User-provided weights
    rw=True,  # Also reweight for equally-weighted ATE
    wildbootstrap=999
)
```

## Implementation Details

### Wild Bootstrap Algorithm

The package implements wild cluster bootstrap following:
1. Estimate original model and obtain residuals
2. Generate Rademacher weights at cluster level
3. Create bootstrap samples by multiplying residuals with weights
4. Re-estimate model for each bootstrap sample
5. Compute bootstrap standard errors and confidence intervals

### Multi-way Clustering

When multiple clusters are specified:
- Uses appropriate variance-covariance matrix specification in pyfixest
- For wild bootstrap, uses the first cluster variable
- Properly handles nested and non-nested clustering structures

### Formula Parsing

The formula parser:
- Handles leading `~` (optional)
- Splits controls and fixed effects by `|`
- Parses multiple variables separated by `+`
- Strips whitespace and handles edge cases

## Backward Compatibility

The package maintains full backward compatibility:

```python
# Old interface still works
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    controls=['x1', 'x2'],  # Deprecated but supported
    absorb=['industry'],    # Deprecated but supported
    cluster='state',        # Deprecated but supported
    bootstrap=999          # Deprecated but supported
)
```

Deprecation warnings guide users to the new interface.

## Performance Considerations

### Wild Bootstrap
- Computationally intensive for large datasets
- Use fewer iterations (99-499) for exploration
- Use more iterations (999-9999) for final results
- Parallel processing helps with multiple horizons

### Multi-way Clustering
- Standard errors computation is more complex
- May increase estimation time
- Consider the trade-off between robustness and speed

## Installation Requirements

The enhanced package requires:
- `wildboottest>=0.1.0` (optional, fallback available)
- All other dependencies remain the same

Install with:
```bash
pip install wildboottest  # For optimal wild bootstrap performance
```

## Testing

The test suite has been expanded to cover:
- Formula parsing edge cases
- Wild bootstrap inference
- Multi-way clustering
- Weighted estimation combinations
- Backward compatibility

Run tests:
```bash
python test_lpdid.py
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic formula interface usage
- Wild bootstrap with multi-way clustering
- Weighted estimation combinations
- Specification comparisons
- Performance benchmarks

## Future Enhancements

Potential future additions:
- Support for more complex formulas (interactions, transformations)
- Additional bootstrap methods (pairs, block)
- Integration with other inference packages
- GPU acceleration for large datasets