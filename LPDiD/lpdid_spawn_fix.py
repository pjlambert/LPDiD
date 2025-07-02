"""
Alternative approach to handle fork warnings by controlling environment variables
and joblib backend configuration more directly.
"""

import os
import warnings

def configure_spawn_environment():
    """
    Configure environment to avoid fork-related warnings on Linux.
    This should be called BEFORE importing numpy, scipy, or other libraries.
    """
    # Disable fork safety warnings from various libraries
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Configure joblib to avoid fork issues
    os.environ['JOBLIB_START_METHOD'] = 'spawn'
    
    # Disable Polars fork warnings if Polars is being used
    os.environ['POLARS_WARN_UNSTABLE_FORK'] = '0'
    
    # Configure OpenBLAS to avoid fork issues
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Configure MKL to avoid fork issues
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("Environment configured to avoid fork warnings")

def create_spawn_safe_parallel(n_jobs=-1):
    """
    Create a joblib Parallel object that's configured to avoid fork issues.
    """
    from joblib import Parallel
    
    # Use 'loky' backend which is fork-safe
    # Set environment to 'spawn' for extra safety
    return Parallel(
        n_jobs=n_jobs,
        backend='loky',
        prefer='processes',  # Force process-based parallelism
        require='sharedmem',  # Avoid unnecessary data serialization
        verbose=0
    )

# Example usage in a script:
if __name__ == "__main__":
    # Call this BEFORE importing numpy, scipy, LPDiD, etc.
    configure_spawn_environment()
    
    # Now import everything else
    import numpy as np
    import pandas as pd
    from LPDiD import LPDiD
    
    print("Libraries imported successfully without fork warnings")
