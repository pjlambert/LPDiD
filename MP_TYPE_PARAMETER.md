# Multiprocessing Type Parameter (mp_type)

## Overview

The `mp_type` parameter has been added to both `LPDiD` and `LPDiDPois` classes to address Polars fork() warnings on Linux systems and provide explicit control over multiprocessing behavior.

## Problem Solved

When running LPDiD on Linux systems, you may encounter this warning:
```
RuntimeWarning: Using fork() can cause Polars to deadlock in the child process.
In addition, using fork() with Python in general is a recipe for mysterious
deadlocks and crashes.
```

This happens during step 4 (parallel regression execution) when the multiprocessing library uses the default 'fork' method on Linux.

## Solution

Use the `mp_type='spawn'` parameter to explicitly set the multiprocessing start method:

```python
from LPDiD import LPDiD

# For LPDiD
model = LPDiD(
    data=data,
    depvar='outcome',
    unit='unit_id',
    time='time_period',
    treat='treatment',
    pre_window=3,
    post_window=3,
    n_jobs=-1,           # Use all available cores
    mp_type='spawn'      # Prevents Polars fork() warnings on Linux
)

# For LPDiDPois  
model_pois = LPDiDPois(
    data=count_data,
    depvar='count_outcome',
    unit='unit_id', 
    time='time_period',
    treat='treatment',
    pre_window=3,
    post_window=3,
    n_jobs=4,
    mp_type='spawn'      # Same parameter works for Poisson version
)
```

## Available Options

The `mp_type` parameter accepts three values:

- **'spawn'** (recommended for Linux): Creates a fresh Python interpreter process
  - Pros: Avoids fork() issues, more robust
  - Cons: Slightly higher overhead due to process startup
  
- **'fork'** (default on Unix/Linux): Copies the parent process
  - Pros: Lower startup overhead
  - Cons: Can cause issues with certain libraries like Polars
  
- **'forkserver'** (Unix only): Uses a server process to create new processes
  - Pros: Combines benefits of fork and spawn
  - Cons: More complex, Unix only

- **None** (default): Uses system default method

## Platform Behavior

- **Linux**: Default is 'fork' (causes Polars warnings) → Use `mp_type='spawn'`
- **macOS**: Default is 'spawn' (no issues) → `mp_type` parameter optional
- **Windows**: Only supports 'spawn' → `mp_type` parameter has no effect

## Usage Recommendations

### For Linux Users (Recommended)
```python
model = LPDiD(
    data=data,
    # ... other parameters ...
    mp_type='spawn'  # Prevents Polars warnings
)
```

### For Cross-Platform Code
```python
import platform

mp_method = 'spawn' if platform.system() == 'Linux' else None

model = LPDiD(
    data=data,
    # ... other parameters ...
    mp_type=mp_method
)
```

### When Parallel Processing is Not Used
If `n_jobs=1`, the `mp_type` parameter has no effect since no multiprocessing occurs.

## Error Handling

The implementation includes robust error handling:

- **Invalid method**: Raises `ValueError` with valid options
- **Unsupported method**: Warns and falls back to system default  
- **Already set method**: Attempts to change with force, warns if unsuccessful
- **Configuration failure**: Warns and continues with system default

## Performance Considerations

- **'spawn'** has slightly higher startup overhead but is more robust
- **'fork'** is faster to start but may cause issues with certain libraries
- The performance difference is typically negligible for the parallel regression step
- Use 'spawn' unless you have specific performance requirements that necessitate 'fork'

## Example Output

When the parameter is used successfully, you'll see output like:
```
Set multiprocessing start method to 'spawn'

Parallel Processing Configuration:
  System: Linux x86_64
  Available CPU cores: 8
  Cores to be used: 8
  Total regressions to run: 6
  Multiprocessing start method: spawn
```

This confirms that the multiprocessing method has been set correctly and should prevent Polars fork() warnings.
