# LP-DiD Memory Management Fix

## Issue
`TerminatedWorkerError` with `SIGKILL(-9)` when running LP-DiD on large datasets (40+ million observations) with parallel processing.

## Solution Implemented

**Graceful Error Handling**: The code now includes a try-except block that catches `TerminatedWorkerError` and `MemoryError`, automatically falling back to sequential processing if parallel processing fails.

```python
try:
    # Parallel processing attempt
    with Parallel(...) as parallel:
        results = parallel(...)
except (TerminatedWorkerError, MemoryError) as e:
    print(f"\nParallel processing failed: {type(e).__name__}")
    print("Falling back to sequential processing...")
    print("Consider reducing n_jobs or using a machine with more memory.")
    # Fall back to sequential processing
```

## Key Changes

1. **No artificial size limits**: The code doesn't make assumptions about your hardware capabilities
2. **Automatic fallback**: If parallel processing fails due to memory issues, it automatically switches to sequential
3. **Clear messaging**: Users see what happened and get suggestions for resolution
4. **Increased timeouts**: Timeout increased to 600 seconds for large datasets
5. **Dynamic max_nbytes**: Uses '1G' for datasets over 1M observations, '100M' otherwise

## Recommendations for Large Datasets

### If you encounter memory issues:

1. **Reduce n_jobs**: Use fewer parallel workers
   ```python
   n_jobs=2  # Instead of 10
   ```

2. **Use sequential processing**: Set `n_jobs=1`

3. **Optimize memory usage**:
   - Use `lean=True` (default)
   - Use `copy_data=False`

4. **Monitor system resources**: Use tools like `htop` to watch memory usage

5. **For machines with lots of memory** (like your 1TB systems):
   - You can use high n_jobs values
   - The code will use your resources without artificial restrictions
   - Only falls back if actual errors occur

## Example Usage

```python
# For high-memory machines (1TB RAM)
custom_lpdid = LPDiD(
    data=full_panel,
    depvar='in_dnb',
    unit='dunsnumber',
    time='dbyear',
    treat='treatment',
    pre_window=5,
    post_window=5,
    n_jobs=10,  # Use as many cores as you want
    lean=True,
    copy_data=False,
    cluster_formula='~ dbyear + dunsnumber',
    mp_type='spawn',
)

# If memory issues occur, consider:
# 1. Reducing n_jobs
# 2. Using n_jobs=1 for sequential
# 3. Checking system memory usage
```

The code respects your hardware capabilities and only intervenes when actual errors occur.
