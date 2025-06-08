"""
Local Projections Difference-in-Differences (LP-DiD) for Python
Based on Dube, Girardi, Jordà, and Taylor (2023)
"""

# Set OpenMP environment variables BEFORE importing numpy/scipy
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'
# Additional OMP settings that might help
os.environ['OMP_NESTED'] = 'FALSE'
os.environ['OMP_THREAD_LIMIT'] = '1'

import warnings
# Explicitly suppress all OMP-related warnings
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")
warnings.filterwarnings("ignore", message=".*OMP.*nested.*deprecated.*")
warnings.filterwarnings("ignore", message=".*OMP_NESTED.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Cannot form a team.*")
warnings.filterwarnings("ignore", message=".*Consider unsetting.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*omp.*")

# Also try to suppress at the system level
import sys
if hasattr(sys, 'stderr'):
    class WarningFilter:
        def __init__(self, stream):
            self.stream = stream
            
        def write(self, data):
            # Filter out OMP warnings
            if any(phrase in data for phrase in [
                "OMP: Info #276", "OMP: Info #268", "OMP: Warning #96", "OMP: Hint"
            ]):
                return
            self.stream.write(data)
            
        def flush(self):
            self.stream.flush()
    
    # Temporarily replace stderr to filter OMP warnings
    original_stderr = sys.stderr
    sys.stderr = WarningFilter(original_stderr)

import numpy as np
import pandas as pd
import pyfixest as pf
from typing import Optional, Union, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from dataclasses import dataclass
from scipy import stats
import re

# Try to import wildboottest, fall back to local implementation if not available
try:
    from wildboottest import wildboottest
except ImportError:
    warnings.warn("wildboottest package not found. Using fallback implementation. "
                  "For better performance, install wildboottest: pip install wildboottest")
    from .wildboot_fallback import wildboottest


@dataclass
class LPDiDResults:
    """Container for LP-DiD results"""
    event_study: Optional[pd.DataFrame] = None
    pooled: Optional[pd.DataFrame] = None
    depvar: str = ""
    pre_window: Optional[int] = None
    post_window: Optional[int] = None
    control_group: str = ""
    treated_group: str = ""
    iv_diagnostics: Optional[pd.DataFrame] = None
    first_stage: Optional[pd.DataFrame] = None
    
    def summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("LP-DiD Results Summary")
        print("="*60)
        
        print(f"\nDependent variable: {self.depvar}")
        print(f"Pre-treatment window: {self.pre_window}")
        print(f"Post-treatment window: {self.post_window}")
        print(f"Control group: {self.control_group}")
        print(f"Treated group: {self.treated_group}")
        
        if self.event_study is not None:
            print("\n" + "-"*40)
            print("Event Study Estimates")
            print("-"*40)
            print(self.event_study.to_string(index=False))
        
        if self.pooled is not None:
            print("\n" + "-"*40)
            print("Pooled Estimates")
            print("-"*40)
            print(self.pooled.to_string(index=False))
        
        if self.iv_diagnostics is not None:
            print("\n" + "-"*40)
            print("IV Diagnostics")
            print("-"*40)
            print(self.iv_diagnostics.to_string(index=False))
            
            # Check for weak instruments
            weak_iv = self.iv_diagnostics[self.iv_diagnostics['first_stage_F'] < 10]
            if not weak_iv.empty:
                print("\nWarning: Weak instruments detected (F < 10) for horizons:")
                print(weak_iv['horizon'].tolist())
        
        if self.first_stage is not None:
            print("\n" + "-"*40)
            print("First Stage Results (Selected Horizons)")
            print("-"*40)
            # Show first stage for key horizons
            key_horizons = [-1, 0, 3, 5] if 5 <= self.post_window else [-1, 0, self.post_window]
            first_stage_subset = self.first_stage[self.first_stage['horizon'].isin(key_horizons)]
            if not first_stage_subset.empty:
                print(first_stage_subset.to_string(index=False))


def parse_formula(formula: str) -> Tuple[List[str], List[str], Optional[Dict[str, List[str]]]]:
    """
    Parse formula string like "~ controls | FE | endog ~ instruments"
    
    Returns
    -------
    controls : list
        Control variables
    fixed_effects : list
        Fixed effects
    iv_spec : dict or None
        Dict with 'endog' and 'instruments' keys if IV specified
    """
    if not formula or formula.strip() == "":
        return [], [], None
    
    # Remove leading ~ if present
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()
    
    # Split by | (up to 3 parts for IV)
    parts = [p.strip() for p in formula.split("|")]
    
    # Parse controls
    controls = []
    if len(parts) > 0 and parts[0]:
        controls = [c.strip() for c in parts[0].split("+") if c.strip()]
    
    # Parse fixed effects
    fixed_effects = []
    if len(parts) > 1 and parts[1]:
        fixed_effects = [fe.strip() for fe in parts[1].split("+") if fe.strip()]
    
    # Parse IV specification (third part)
    iv_spec = None
    if len(parts) > 2 and parts[2]:
        # Format: "endog ~ instruments"
        iv_parts = parts[2].split("~")
        if len(iv_parts) == 2:
            endog_vars = [e.strip() for e in iv_parts[0].split("+") if e.strip()]
            instruments = [i.strip() for i in iv_parts[1].split("+") if i.strip()]
            iv_spec = {
                'endog': endog_vars,
                'instruments': instruments
            }
    
    return controls, fixed_effects, iv_spec


def parse_interactions(interactions: str) -> List[str]:
    """
    Parse interaction specification like "~ var1 + var2"
    
    Returns
    -------
    interact_vars : list
        Variables to interact with treatment
    """
    if not interactions or interactions.strip() == "":
        return []
    
    # Remove leading ~ if present
    interactions = interactions.strip()
    if interactions.startswith("~"):
        interactions = interactions[1:].strip()
    
    # Parse interaction variables
    interact_vars = [v.strip() for v in interactions.split("+") if v.strip()]
    return interact_vars


def parse_cluster_formula(formula: str) -> List[str]:
    """
    Parse cluster formula like "~ cluster1 + cluster2"
    
    Returns
    -------
    clusters : list
        Clustering variables
    """
    if not formula or formula.strip() == "":
        return []
    
    # Remove leading ~ if present
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()
    
    # Parse clusters
    clusters = [c.strip() for c in formula.split("+") if c.strip()]
    return clusters


class LPDiD:
    """
    Local Projections Difference-in-Differences Estimator
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    depvar : str
        Dependent variable name
    unit : str
        Unit identifier variable
    time : str
        Time variable
    treat : str
        Binary treatment indicator
    pre_window : int, optional
        Pre-treatment periods (>=2)
    post_window : int, optional
        Post-treatment periods (>=0)
    formula : str, optional
        Formula for controls, fixed effects, and IV like "~ controls | FE | endog ~ instruments"
    interactions : str, optional
        Variables to interact with treatment like "~ var1 + var2"
    cluster_formula : str, optional
        Formula for clustering like "~ cluster1 + cluster2"
    ylags : int, optional
        Number of outcome lags to include
    dylags : int, optional
        Number of first-differenced outcome lags to include
    nonabsorbing : tuple, optional
        (L, notyet, firsttreat) for non-absorbing treatment
    nevertreated : bool, default False
        Use only never-treated as controls
    nocomp : bool, default False
        Rule out composition changes
    rw : bool, default False
        Reweight for equally-weighted ATE
    pmd : Union[int, str], optional
        Pre-mean differencing specification
    wildbootstrap : int, optional
        Wild bootstrap iterations
    seed : int, optional
        Random seed
    weights : str, optional
        Weight variable
    n_jobs : int, default 1
        Number of parallel jobs (-1 for all cores)
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 depvar: str,
                 unit: str,
                 time: str,
                 treat: str,
                 pre_window: Optional[int] = None,
                 post_window: Optional[int] = None,
                 formula: Optional[str] = None,
                 interactions: Optional[str] = None,
                 cluster_formula: Optional[str] = None,
                 ylags: Optional[int] = None,
                 dylags: Optional[int] = None,
                 nonabsorbing: Optional[Tuple[int, bool, bool]] = None,
                 nevertreated: bool = False,
                 nocomp: bool = False,
                 rw: bool = False,
                 pmd: Optional[Union[int, str]] = None,
                 wildbootstrap: Optional[int] = None,
                 seed: Optional[int] = None,
                 weights: Optional[str] = None,
                 n_jobs: int = 1,
                 # Backward compatibility parameters
                 controls: Optional[List[str]] = None,
                 absorb: Optional[List[str]] = None,
                 cluster: Optional[str] = None,
                 bootstrap: Optional[int] = None):
        
        # Store parameters
        self.data = data.copy()
        self.depvar = depvar
        self.unit = unit
        self.time = time
        self.treat = treat
        self.pre_window = pre_window
        self.post_window = post_window
        
        # Handle backward compatibility
        if controls is not None or absorb is not None:
            warnings.warn("Using deprecated parameters 'controls' and 'absorb'. "
                         "Please use the formula parameter instead: "
                         "formula='~ control1 + control2 | fe1 + fe2'", 
                         DeprecationWarning)
            self.controls = controls or []
            self.absorb = absorb or []
            self.iv_spec = None
        elif formula:
            self.controls, self.absorb, self.iv_spec = parse_formula(formula)
        else:
            self.controls = []
            self.absorb = []
            self.iv_spec = None
        
        # Parse interactions
        if interactions:
            self.interact_vars = parse_interactions(interactions)
        else:
            self.interact_vars = []
        
        # Handle cluster backward compatibility
        if cluster is not None:
            warnings.warn("Using deprecated parameter 'cluster'. "
                         "Please use cluster_formula instead: "
                         "cluster_formula='~ cluster1 + cluster2'",
                         DeprecationWarning)
            self.cluster_vars = [cluster]
        elif cluster_formula:
            self.cluster_vars = parse_cluster_formula(cluster_formula)
        else:
            self.cluster_vars = [unit]  # Default to unit clustering
        
        # Handle bootstrap backward compatibility
        if bootstrap is not None:
            warnings.warn("Using deprecated parameter 'bootstrap'. "
                         "Please use 'wildbootstrap' instead.",
                         DeprecationWarning)
            self.wildbootstrap = bootstrap
        else:
            self.wildbootstrap = wildbootstrap
            
        self.ylags = ylags
        self.dylags = dylags
        self.nonabsorbing = nonabsorbing
        self.nevertreated = nevertreated
        self.nocomp = nocomp
        self.rw = rw
        self.pmd = pmd
        self.seed = seed
        self.weights = weights
        self.n_jobs = n_jobs
        
        # Validate inputs
        self._validate_inputs()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Prepare data
        self._prepare_data()
        
    def _validate_inputs(self):
        """Validate input parameters"""
        # Check if pre_window or post_window is specified
        if self.pre_window is None and self.post_window is None:
            raise ValueError("Please specify either post_window, pre_window, or both.")
        
        # Set defaults
        if self.pre_window is None:
            self.pre_window = 2
        if self.post_window is None:
            self.post_window = 0
            
        # Validate windows
        if self.pre_window < 2:
            raise ValueError("pre_window must be >= 2")
        if self.post_window < 0:
            raise ValueError("post_window must be >= 0")
        
        # Check binary treatment
        treat_vals = self.data[self.treat].dropna().unique()
        if not set(treat_vals).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0/1)")
        
        # Parse nonabsorbing
        if self.nonabsorbing:
            L, notyet, firsttreat = self.nonabsorbing
            if not isinstance(L, int) or L < 1:
                raise ValueError("L in nonabsorbing must be a positive integer")
            if notyet and self.nevertreated:
                raise ValueError("Cannot specify both notyet and nevertreated")
    
    def _prepare_data(self):
        """Prepare data for estimation"""
        # Sort data
        self.data = self.data.sort_values([self.unit, self.time])
        
        # Set as panel
        self.data = self.data.set_index([self.unit, self.time])
        
        # Create first difference of treatment
        self.data['D_treat'] = self.data.groupby(level=0)[self.treat].diff()
        
        # Create lags if needed
        if self.ylags:
            for lag in range(1, self.ylags + 1):
                self.data[f'L{lag}_{self.depvar}'] = (
                    self.data.groupby(level=0)[self.depvar].shift(lag)
                )
                self.controls.append(f'L{lag}_{self.depvar}')
        
        if self.dylags:
            # First create the first difference
            self.data[f'D_{self.depvar}'] = self.data.groupby(level=0)[self.depvar].diff()
            for lag in range(1, self.dylags + 1):
                self.data[f'L{lag}_D_{self.depvar}'] = (
                    self.data.groupby(level=0)[f'D_{self.depvar}'].shift(lag)
                )
                self.controls.append(f'L{lag}_D_{self.depvar}')
        
        # Reset index for easier manipulation
        self.data = self.data.reset_index()

    def _identify_clean_controls(self):
        """Identify clean control samples and create indicators"""
        # Maximum horizons needed
        pre_CCS = max(self.pre_window, self.pre_window if hasattr(self, 'pre_pooled_end') else 0)
        post_CCS = max(self.post_window, self.post_window if hasattr(self, 'post_pooled_end') else 0)
        
        # Absorbing treatment case
        if self.nonabsorbing is None:
            for h in range(post_CCS + 1):
                self.data[f'CCS_{h}'] = 0
                # Treatment switchers or future controls
                self.data.loc[
                    (self.data['D_treat'] == 1) | 
                    (self.data.groupby(self.unit)[self.treat].shift(-h) == 0),
                    f'CCS_{h}'
                ] = 1
            
            # Pre-treatment clean controls
            for h in range(1, pre_CCS + 1):
                self.data[f'CCS_m{h}'] = self.data['CCS_0']
        
        # Non-absorbing treatment case
        else:
            L, notyet, firsttreat = self.nonabsorbing
            
            # Create CCS_0
            self.data['CCS_0'] = 0
            mask = (self.data['D_treat'].isin([0, 1]))
            
            # Check no switches in window [-L, -1]
            for k in range(1, L + 1):
                mask &= (self.data.groupby(self.unit)['D_treat'].shift(k).abs() != 1)
            
            self.data.loc[mask, 'CCS_0'] = 1
            
            # Forward CCS
            for h in range(1, post_CCS + 1):
                self.data[f'CCS_{h}'] = 0
                mask = (self.data[f'CCS_{h-1}'] == 1) & (
                    self.data.groupby(self.unit)['D_treat'].shift(-h).abs() != 1
                )
                self.data.loc[mask, f'CCS_{h}'] = 1
            
            # Backward CCS
            self.data['CCS_m1'] = self.data['CCS_0']
            for h in range(2, pre_CCS + 1):
                self.data[f'CCS_m{h}'] = 0
                mask = (
                    (self.data[f'CCS_m{h-1}'] == 1) & 
                    (self.data.groupby(self.unit)[f'CCS_m{h-1}'].shift(1) == 1)
                )
                self.data.loc[mask, f'CCS_m{h}'] = 1
        
        # Apply control group restrictions
        if self.nevertreated:
            # Only never treated units
            max_treat = self.data.groupby(self.unit)[self.treat].max()
            never_treated = (max_treat == 0).astype(int)
            never_treated_df = never_treated.reset_index()
            never_treated_df.columns = [self.unit, 'never_treated']
            self.data = self.data.merge(never_treated_df, on=self.unit)
            
            # Update CCS indicators
            for h in range(post_CCS + 1):
                self.data.loc[
                    (self.data['D_treat'] == 0) & (self.data['never_treated'] == 0),
                    f'CCS_{h}'
                ] = 0
            for h in range(2, pre_CCS + 1):
                self.data.loc[
                    (self.data['D_treat'] == 0) & (self.data['never_treated'] == 0),
                    f'CCS_m{h}'
                ] = 0

    def _generate_long_differences(self):
        """Generate long differences for dependent variable"""
        if self.pmd is None:
            # Standard long differences
            for h in range(self.post_window + 1):
                self.data[f'D{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                    self.data.groupby(self.unit)[self.depvar].shift(1)
                )
            
            for h in range(2, self.pre_window + 1):
                self.data[f'Dm{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(h) - 
                    self.data.groupby(self.unit)[self.depvar].shift(1)
                )
        
        elif self.pmd == 'max':
            # Use all available pre-treatment periods
            def calc_pre_mean(group):
                cumsum = group[self.depvar].expanding().sum()
                count = group[self.depvar].expanding().count()
                return cumsum.shift(1) / count.shift(1)
            
            self.data['aveLY'] = self.data.groupby(self.unit).apply(
                calc_pre_mean
            ).reset_index(drop=True)
            
            # Create differences
            for h in range(self.post_window + 1):
                self.data[f'D{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                    self.data['aveLY']
                )
            
            for h in range(2, self.pre_window + 1):
                self.data[f'Dm{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(h) - 
                    self.data['aveLY']
                )
        
        else:
            # Moving average over [-pmd, -1]
            def calc_ma(group, window):
                return group[self.depvar].rolling(window=window, min_periods=window).mean()
            
            self.data['aveLY'] = self.data.groupby(self.unit).apply(
                lambda x: calc_ma(x, self.pmd).shift(1)
            ).reset_index(drop=True)
            
            # Create differences
            for h in range(self.post_window + 1):
                self.data[f'D{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                    self.data['aveLY']
                )
            
            for h in range(2, self.pre_window + 1):
                self.data[f'Dm{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(h) - 
                    self.data['aveLY']
                )

    def _compute_weights(self):
        """Compute weights for reweighted estimation"""
        if not self.rw:
            # No reweighting - equal weights
            if self.weights:
                base_weight = self.data[self.weights]
            else:
                base_weight = 1
                
            for h in range(max(self.post_window, self.post_window) + 1):
                self.data[f'reweight_{h}'] = base_weight
            return
        
        # Simple reweighting implementation
        for h in range(max(self.post_window, self.post_window) + 1):
            self.data[f'reweight_{h}'] = 1

    def _build_regression_formula(self, y_var: str, fe_vars: List[str]) -> str:
        """Build regression formula with potential IV and interactions"""
        # Start with dependent variable
        formula_parts = [y_var + " ~"]
        
        # Add treatment (if not endogenous)
        if self.iv_spec is None or 'D_treat' not in self.iv_spec['endog']:
            formula_parts.append("D_treat")
        
        # Add interaction terms
        if self.interact_vars:
            for var in self.interact_vars:
                # Add main effect if not already in controls
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                if clean_var not in self.controls and var not in self.controls:
                    self.controls.append(var)
                
                # Add interaction term
                formula_parts.append(f"D_treat:{var}")
        
        # Add controls
        if self.controls:
            formula_parts.extend(self.controls)
        
        # Build exogenous part
        if len(formula_parts) > 1:
            exog_part = " + ".join(formula_parts[1:])
        else:
            exog_part = "1"
        
        # Add fixed effects
        if fe_vars:
            formula = f"{y_var} ~ {exog_part} | {' + '.join(fe_vars)}"
        else:
            formula = f"{y_var} ~ {exog_part}"
        
        # Handle IV specification
        if self.iv_spec:
            endog_formula = " + ".join(self.iv_spec['endog'])
            instruments_formula = " + ".join(self.iv_spec['instruments'])
            formula += f" | {endog_formula} ~ {instruments_formula}"
        
        return formula

    def _run_single_regression(self, horizon, is_pre=False):
        """Run a single LP-DiD regression"""
        # Determine variables
        if is_pre:
            y_var = f'Dm{horizon}y'
            ccs_var = f'CCS_m{horizon}'
            weight_var = 'reweight_0' if self.rw else None
        else:
            y_var = f'D{horizon}y'
            ccs_var = f'CCS_{horizon}'
            weight_var = f'reweight_{horizon}' if self.rw else None
        
        # Filter data
        mask = self.data[ccs_var] == 1
        reg_data = self.data[mask].copy()
        
        if reg_data.shape[0] == 0:
            return None
        
        # Build formula
        fe_vars = [self.time] + self.absorb
        formula = self._build_regression_formula(y_var, fe_vars)
        
        # Set up clustering
        if len(self.cluster_vars) == 1:
            vcov = {'CRV1': self.cluster_vars[0]}
        else:
            # Multi-way clustering
            vcov = 'twoway'  # pyfixest will handle the cluster variables
        
        # Add weights if specified
        if self.weights and weight_var:
            # Combine user weights with reweighting
            reg_data['combined_weight'] = reg_data[self.weights] * reg_data[weight_var]
            use_weights = 'combined_weight'
        elif self.weights:
            use_weights = self.weights
        elif weight_var and weight_var in reg_data.columns:
            use_weights = weight_var
        else:
            use_weights = None
        
        # Run regression
        try:
            if use_weights is not None:
                fit = pf.feols(
                    formula, 
                    data=reg_data, 
                    weights=use_weights,
                    vcov=vcov
                )
            else:
                fit = pf.feols(
                    formula, 
                    data=reg_data,
                    vcov=vcov
                )
            
            # Extract main treatment coefficient
            coef = fit.coef().loc['D_treat']
            nobs = len(reg_data)  # Use len(reg_data) instead of fit.nobs
            
            # Store results dictionary
            results = {
                'horizon': -horizon if is_pre else horizon,
                'coefficient': coef,
                'obs': nobs
            }
            
            # Extract interaction coefficients if present
            if self.interact_vars:
                for var in self.interact_vars:
                    interaction_term = f"D_treat:{var}"
                    if interaction_term in fit.coef().index:
                        results[f'interact_{var}'] = fit.coef().loc[interaction_term]
                        results[f'interact_{var}_se'] = fit.se().loc[interaction_term]
                        results[f'interact_{var}_p'] = fit.pvalue().loc[interaction_term]  # Changed from pvalues() to pvalue()
            
            # Store IV diagnostics if applicable
            if self.iv_spec:
                # Get first-stage F-statistic
                try:
                    first_stage_F = fit.fitstat('ivf1')
                    results['first_stage_F'] = first_stage_F
                    results['weak_iv'] = first_stage_F < 10
                    
                    # Try to get additional diagnostics
                    if hasattr(fit, 'kpr'):
                        results['kleibergen_paap'] = fit.kpr
                    
                    # Store first stage coefficients for main instrument
                    if hasattr(fit, 'first_stage'):
                        fs_results = {}
                        for inst in self.iv_spec['instruments']:
                            if inst in fit.first_stage.coef().index:
                                fs_results[f'fs_{inst}_coef'] = fit.first_stage.coef().loc[inst]
                                fs_results[f'fs_{inst}_se'] = fit.first_stage.se().loc[inst]
                        results.update(fs_results)
                except:
                    results['first_stage_F'] = np.nan
                    results['weak_iv'] = True
            
            # Handle inference
            if self.wildbootstrap:
                # Run wild bootstrap
                try:
                    # For IV, we need the reduced form for bootstrapping
                    if self.iv_spec:
                        # Get reduced form: Y ~ Z + controls
                        rf_formula_parts = [y_var + " ~"]
                        rf_formula_parts.extend(self.iv_spec['instruments'])
                        if self.controls:
                            rf_formula_parts.extend(self.controls)
                        rf_formula = " + ".join(rf_formula_parts) + " | " + " + ".join(fe_vars)
                        
                        # Estimate reduced form
                        rf_fit = pf.feols(rf_formula, data=reg_data, weights=use_weights)
                        X = rf_fit.model_matrix_x
                        y = rf_fit.model_response
                    else:
                        X = fit.model_matrix_x
                        y = fit.model_response
                    
                    cluster = reg_data[self.cluster_vars[0]].values
                    
                    # Get position of D_treat coefficient
                    param_idx = list(fit.coef().index).index('D_treat')
                    
                    # Run wild bootstrap
                    wb_result = wildboottest(
                        X=X,
                        y=y,
                        cluster=cluster,
                        B=self.wildbootstrap,
                        param=param_idx,
                        weights=use_weights,
                        seed=self.seed
                    )
                    
                    results['se'] = wb_result['se']
                    results['t'] = wb_result['t_stat']
                    results['p'] = wb_result['p_value']
                    results['ci_low'] = wb_result['CI'][0]
                    results['ci_high'] = wb_result['CI'][1]
                    
                except Exception as e:
                    warnings.warn(f"Wild bootstrap failed, using analytical SEs: {e}")
                    # Fall back to analytical SEs
                    results['se'] = fit.se().loc['D_treat']
                    results['t'] = fit.tstat().loc['D_treat']
                    results['p'] = fit.pvalue().loc['D_treat']  # Changed from pvalues() to pvalue()
                    ci = fit.confint().loc['D_treat']
                    results['ci_low'] = ci.iloc[0]  # Use iloc instead of positional indexing
                    results['ci_high'] = ci.iloc[1]  # Use iloc instead of positional indexing
            else:
                # Use analytical standard errors
                results['se'] = fit.se().loc['D_treat']
                results['t'] = fit.tstat().loc['D_treat']
                results['p'] = fit.pvalue().loc['D_treat']  # Changed from pvalues() to pvalue()
                ci = fit.confint().loc['D_treat']
                results['ci_low'] = ci.iloc[0]  # Use iloc instead of positional indexing
                results['ci_high'] = ci.iloc[1]  # Use iloc instead of positional indexing
            
            return results
            
        except Exception as e:
            warnings.warn(f"Regression failed for horizon {horizon}: {e}")
            return None

    def fit(self):
        """
        Fit the LP-DiD model
        
        Returns
        -------
        LPDiDResults
            Results object containing event study and pooled estimates
        """
        # Identify clean control samples
        self._identify_clean_controls()
        
        # Generate long differences
        self._generate_long_differences()
        
        # Compute weights if needed
        self._compute_weights()
        
        # Run event study regressions
        event_study_results = []
        
        # Pre-treatment regressions
        for h in range(2, self.pre_window + 1):
            result = self._run_single_regression(h, is_pre=True)
            if result:
                event_study_results.append(result)
        
        # Post-treatment regressions  
        for h in range(self.post_window + 1):
            result = self._run_single_regression(h, is_pre=False)
            if result:
                event_study_results.append(result)
        
        # Convert to DataFrame
        if event_study_results:
            event_study_df = pd.DataFrame(event_study_results)
            # Sort by horizon
            event_study_df = event_study_df.sort_values('horizon').reset_index(drop=True)
        else:
            # Create empty DataFrame with expected columns
            event_study_df = pd.DataFrame(columns=['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs'])
        
        # Create results object
        results = LPDiDResults(
            event_study=event_study_df,
            depvar=self.depvar,
            pre_window=self.pre_window,
            post_window=self.post_window,
            control_group="Controls",
            treated_group="Treated"
        )
        
        return results