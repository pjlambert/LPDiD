# Parallel Processing in LPDiD

The LPDiD package now supports parallel processing to significantly speed up estimation for large datasets. This feature utilizes multiple CPU cores to parallelize both data preparation and regression estimation phases.

## Overview

Parallel processing is controlled by the `n_jobs` parameter:
- `n_jobs=1`: Sequential processing (default)
- `n_jobs=2,3,4,...`: Use specific number of cores
- `n_jobs=-1`: Use all available CPU cores

## What Gets Parallelized

### 1. Data Reshaping Operations
- **Lag Creation**: When `ylags > 2` or `dylags > 2`, lag computations are distributed across cores
- **Long Differences Generation**: The most computationally intensive part, creating differences for all horizons is parallelized

### 2. Regression Estimation
- Each horizon's regression is run independently in parallel
- This was already implemented and remains unchanged

## Performance Benefits

The speedup depends on several factors:
- **Dataset size**: Larger datasets (more units) benefit more
- **Number of horizons**: More pre/post periods mean more parallel tasks
- **Number of lags**: More lags mean more parallel lag computations
- **Available CPU cores**: More cores allow more parallelization

Typical speedups range from 2-8x depending on these factors.

## Usage Example

```python
from LPDiD import LPDiD

# Sequential processing
lpdid_seq = LPDiD(
    data=df,
    depvar='y',
    unit='unit', 
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    ylags=5,
    n_jobs=1  # Sequential
)

# Parallel processing with all cores
lpdid_par = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time', 
    treat='treat',
    pre_window=10,
    post_window=20,
    ylags=5,
    n_jobs=-1  # Use all cores
)

# Parallel processing with 4 cores
lpdid_4 = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat', 
    pre_window=10,
    post_window=20,
    ylags=5,
    n_jobs=4  # Use 4 cores
)
```

## Progress Monitoring

When parallel processing is active for data operations, you'll see progress bars:
```
Creating 10 lags in parallel...
100%|████████████| 10000/10000 [00:05<00:00, 1876.42it/s]

Generating long differences in parallel for 31 horizons...
100%|████████████| 31000/31000 [00:12<00:00, 2516.73it/s]
```

## When to Use Parallel Processing

### Recommended for:
- Large datasets (>1000 units)
- Many horizons (pre_window + post_window > 10)
- Many lags (ylags or dylags > 5)
- Computationally intensive operations

### May not benefit:
- Small datasets (<100 units)
- Few horizons (<5 total)
- Simple models without lags
- Systems with limited cores

## Technical Details

### Implementation
- Uses `joblib.Parallel` with the 'loky' backend for robustness
- Automatically detects optimal parallelization points
- Maintains exact numerical consistency with sequential results
- Handles memory efficiently through chunking

### Parallel Thresholds
The package automatically decides when to parallelize:
- Lag creation: Parallel when `ylags > 2` or `dylags > 2`
- Long differences: Parallel when total horizons > 3
- Regressions: Always respects n_jobs setting

### Platform Compatibility
- Works on Windows, macOS, and Linux
- Automatically selects appropriate backend for each platform
- Handles OpenMP warnings on macOS

## Best Practices

1. **Start with `n_jobs=-1`** to use all cores and see maximum speedup
2. **Monitor CPU usage** to ensure cores are being utilized
3. **For very large datasets**, consider using fewer cores to avoid memory issues
4. **Results are identical** whether using parallel or sequential processing

## Troubleshooting

### No speedup observed
- Check if your dataset/model is large enough to benefit
- Verify CPU cores are available (not busy with other tasks)
- Try with a simpler model first (no fixed effects)

### Memory issues
- Reduce n_jobs to use fewer cores
- Consider processing in batches for extremely large datasets

### Verification
To verify parallel results match sequential:
```python
# Run both versions
result_seq = lpdid_seq.fit()
result_par = lpdid_par.fit()

# Compare coefficients
import numpy as np
seq_coef = result_seq.event_study['coefficient'].values
par_coef = result_par.event_study['coefficient'].values
max_diff = np.max(np.abs(seq_coef - par_coef))
print(f"Maximum difference: {max_diff}")  # Should be < 1e-10
```

## Example Scripts

See the following files for complete examples:
- `examples/parallel_processing_example.py`: Basic usage example
- `internal_tests/test_parallel_processing.py`: Comprehensive benchmarking script
