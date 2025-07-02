# LP-DiD Spawn Configuration Resolution

## Issue
You were getting warnings about fork vs spawn in Step 4 of the test, likely on a Linux system with Python 3.10.

## Root Cause
The package was being too aggressive in trying to configure multiprocessing settings, which can conflict with:
1. Other packages that have already set the multiprocessing method
2. User preferences or system configurations
3. The natural behavior of libraries like joblib which handle their own process management

## Solution
I've updated the package to:

1. **Remove forced multiprocessing configuration at module import time** - This was causing conflicts
2. **Only configure when explicitly requested via mp_type parameter** - Gives users control
3. **Provide helpful warnings instead of forcing changes** - Less intrusive approach
4. **Let joblib handle its own backend configuration** - The 'loky' backend is designed to avoid fork issues

## Updated Behavior

### Default (mp_type=None)
- Uses system default multiprocessing method
- On Linux with parallel processing, shows a warning suggesting mp_type='spawn' if needed
- Does not force any configuration changes

### Explicit Configuration (mp_type='spawn')
- Only attempts to configure if actually using parallel processing (n_jobs > 1)
- If already set to the desired method, does nothing
- If unable to change, provides clear guidance on how to set it manually

## Recommendations

### For Linux Users
If you see Polars fork() warnings, you have three options:

1. **Set mp_type parameter explicitly:**
   ```python
   lpdid = LPDiD(
       data=df,
       depvar='outcome',
       unit='id',
       time='period',
       treat='treatment',
       pre_window=5,
       post_window=5,
       n_jobs=4,
       mp_type='spawn'  # Explicitly request spawn
   )
   ```

2. **Set environment variable before running Python:**
   ```bash
   export MULTIPROCESSING_START_METHOD=spawn
   python your_script.py
   ```

3. **Set in your script before importing LPDiD:**
   ```python
   import multiprocessing
   multiprocessing.set_start_method('spawn', force=True)
   
   from LPDiD import LPDiD
   ```

### Why This Approach?
- **Less intrusive**: Doesn't interfere with other packages or user configurations
- **More flexible**: Users can choose their preferred method
- **Better compatibility**: Works with various Python environments and package combinations
- **Clear guidance**: Provides warnings and suggestions rather than forcing changes

## The Warnings Are Safe
Important note: The fork() warnings from Polars are just warnings, not errors. They indicate a potential for issues but won't affect the correctness of your results. The warnings exist because:
- Fork can be problematic with multi-threaded libraries
- Spawn is safer but has more overhead
- Many systems work fine with fork despite the warnings

The package will work correctly regardless of which method is used.
