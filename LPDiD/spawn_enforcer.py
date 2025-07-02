"""
Spawn Enforcer for LP-DiD
Forces the use of spawn multiprocessing for memory efficiency
"""

import os
import sys
import platform
import multiprocessing
import warnings


def enforce_spawn_configuration(force=False, verbose=True):
    """
    Aggressively enforce spawn multiprocessing configuration for memory efficiency.
    
    This function should be called BEFORE importing any heavy scientific libraries
    or creating the LPDiD instance.
    
    Parameters
    ----------
    force : bool, default False
        If True, forces spawn even if another method is already set
    verbose : bool, default True
        If True, prints configuration messages
    
    Returns
    -------
    bool
        True if spawn was successfully configured, False otherwise
    """
    # Set environment variables first (these must be set before numpy/polars import)
    env_vars = {
        'POLARS_WARN_UNSTABLE_FORK': '0',
        'OMP_NESTED': 'FALSE',
        'OMP_MAX_ACTIVE_LEVELS': '1',
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
        'NUMBA_DISABLE_INTEL_SVML': '1',  # Prevents some fork-related issues
    }
    
    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
    
    # Only enforce spawn on systems where it's beneficial
    if platform.system() not in ['Linux', 'Darwin']:  # Darwin = macOS
        if verbose:
            print("Spawn enforcement only applies to Linux/macOS systems")
        return False
    
    try:
        # Check current method
        current_method = multiprocessing.get_start_method(allow_none=True)
        
        if current_method == 'spawn':
            if verbose:
                print("✓ Spawn multiprocessing already configured")
            return True
        
        if current_method is not None and not force:
            if verbose:
                print(f"⚠ Multiprocessing method already set to '{current_method}'")
                print("  Use force=True to override, or restart Python session")
            return False
        
        # Force spawn configuration
        if force and current_method is not None:
            # This requires careful handling as it can break existing processes
            if verbose:
                print(f"🔄 Forcing change from '{current_method}' to 'spawn'...")
        
        multiprocessing.set_start_method('spawn', force=force)
        
        # Verify the change
        new_method = multiprocessing.get_start_method()
        if new_method == 'spawn':
            if verbose:
                print("✓ Successfully configured spawn multiprocessing")
                print("  Benefits: Lower memory usage, better stability with scientific libraries")
            return True
        else:
            if verbose:
                print(f"✗ Failed to set spawn (current: {new_method})")
            return False
            
    except Exception as e:
        if verbose:
            print(f"✗ Error configuring spawn: {e}")
        return False


def get_optimal_joblib_backend():
    """
    Get the optimal joblib backend configuration for spawn processing.
    
    Returns
    -------
    str
        The recommended joblib backend
    """
    current_method = multiprocessing.get_start_method(allow_none=True)
    
    if current_method == 'spawn':
        # With spawn configured, we can use multiprocessing backend
        return 'multiprocessing'
    else:
        # Fall back to loky (which has its own process management)
        return 'loky'


def configure_for_memory_efficiency():
    """
    Complete configuration for memory-efficient parallel processing.
    
    This is a convenience function that sets up the entire environment
    for optimal memory usage with LP-DiD.
    """
    print("=" * 60)
    print("Configuring LP-DiD for Memory Efficiency")
    print("=" * 60)
    
    # Step 1: Enforce spawn
    spawn_success = enforce_spawn_configuration(force=False, verbose=True)
    
    # Step 2: Recommend joblib backend
    backend = get_optimal_joblib_backend()
    print(f"✓ Recommended joblib backend: {backend}")
    
    # Step 3: Memory optimization tips
    print("\n📋 Memory Optimization Tips:")
    print("  • Use spawn multiprocessing (configured above)")
    print("  • Set OMP_NUM_THREADS=1 for many parallel jobs")  
    print("  • Consider n_jobs=4-8 instead of n_jobs=-1 for large datasets")
    print("  • Use lean=True (default) to reduce memory overhead")
    
    # Step 4: Warn about restart if needed
    if not spawn_success:
        print("\n⚠ IMPORTANT: Restart your Python session for spawn to take effect")
        print("   Then run this configuration before importing LPDiD")
    
    print("=" * 60)
    
    return spawn_success, backend


# Auto-configure on import if running on Linux
if platform.system() == 'Linux' and 'pytest' not in sys.modules:
    # Only auto-configure if not running tests and multiprocessing not yet set
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method is None:
        try:
            multiprocessing.set_start_method('spawn')
            # Set environment variables
            if 'POLARS_WARN_UNSTABLE_FORK' not in os.environ:
                os.environ['POLARS_WARN_UNSTABLE_FORK'] = '0'
        except:
            pass  # Silently handle any issues
