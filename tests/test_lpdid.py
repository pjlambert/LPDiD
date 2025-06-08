"""
Unit tests for lpdid package
"""

import unittest
import numpy as np
import pandas as pd
from lpdid import LPDiD, LPDiDResults
import warnings


class TestLPDiD(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create test data"""
        np.random.seed(42)
        
        # Generate simple panel data
        n_units = 50
        n_periods = 20
        treatment_period = 10
        
        units = np.repeat(range(n_units), n_periods)
        time = np.tile(range(n_periods), n_units)
        
        # Add clusters
        cluster1 = np.repeat(np.random.randint(0, 10, n_units), n_periods)
        cluster2 = np.repeat(np.random.randint(0, 5, n_units), n_periods)
        
        # Half units are treated
        treated_units = range(n_units // 2)
        treat = ((np.isin(units, treated_units)) & 
                (time >= treatment_period)).astype(int)
        
        # Simple outcome with treatment effect
        y = (units + time + 
             5 * treat +  # Treatment effect
             np.random.normal(0, 1, len(units)))
        
        # Add categorical FE
        industry = np.repeat(np.random.choice(['A', 'B', 'C'], n_units), n_periods)
        
        cls.df = pd.DataFrame({
            'unit': units,
            'time': time,
            'treat': treat,
            'y': y,
            'x1': np.random.normal(0, 1, len(units)),
            'x2': np.random.binomial(1, 0.5, len(units)),
            'industry': industry,
            'cluster1': cluster1,
            'cluster2': cluster2,
            'weight': np.random.uniform(0.5, 2.0, len(units))
        })
    
    def test_basic_estimation(self):
        """Test basic LP-DiD estimation"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit()
        
        # Check results structure
        self.assertIsInstance(results, LPDiDResults)
        self.assertIsNotNone(results.event_study)
        self.assertIsNotNone(results.pooled)
        
        # Check dimensions
        # Should have -3, -2, -1 (reference), 0, 1, 2, 3, 4, 5
        self.assertEqual(len(results.event_study), 9)
        self.assertEqual(len(results.pooled), 2)
        
        # Check reference period
        ref_period = results.event_study[results.event_study['horizon'] == -1]
        self.assertAlmostEqual(ref_period['coefficient'].iloc[0], 0)
    
    def test_formula_interface(self):
        """Test formula interface"""
        # Controls only
        lpdid1 = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 + x2"
        )
        results1 = lpdid1.fit()
        self.assertIsNotNone(results1.event_study)
        
        # Fixed effects only
        lpdid2 = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ | industry"
        )
        results2 = lpdid2.fit()
        self.assertIsNotNone(results2.event_study)
        
        # Both controls and FE
        lpdid3 = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 + x2 | industry"
        )
        results3 = lpdid3.fit()
        self.assertIsNotNone(results3.event_study)
    
    def test_cluster_formula(self):
        """Test cluster formula interface"""
        # Single cluster
        lpdid1 = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            cluster_formula="~ cluster1"
        )
        results1 = lpdid1.fit()
        self.assertIsNotNone(results1.event_study)
        
        # Multi-way clustering
        lpdid2 = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            cluster_formula="~ cluster1 + cluster2"
        )
        results2 = lpdid2.fit()
        self.assertIsNotNone(results2.event_study)
    
    def test_wild_bootstrap(self):
        """Test wild bootstrap inference"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry",
            wildbootstrap=99,  # Small number for testing
            seed=123
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
        
        # Check that SEs are different from analytical
        lpdid_analytical = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry"
        )
        results_analytical = lpdid_analytical.fit()
        
        # SEs should generally be different
        se_wb = results.event_study[results.event_study['horizon']==3]['se'].values[0]
        se_analytical = results_analytical.event_study[results_analytical.event_study['horizon']==3]['se'].values[0]
        
        # They might be similar but shouldn't be exactly the same
        self.assertNotAlmostEqual(se_wb, se_analytical, places=5)
    
    def test_weighted_estimation(self):
        """Test weighted estimation"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1",
            weights='weight'
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
    
    def test_reweighting(self):
        """Test reweighted estimation"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1",
            rw=True
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
    
    def test_weighted_and_reweighted(self):
        """Test combined weighting and reweighting"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1",
            weights='weight',
            rw=True
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
    
    def test_never_treated(self):
        """Test never-treated control group"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            nevertreated=True
        )
        
        results = lpdid.fit()
        self.assertEqual(results.control_group, "Never treated units")
    
    def test_lags(self):
        """Test with outcome lags"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            ylags=2
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
    
    def test_pmd(self):
        """Test pre-mean differencing"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            pmd='max'
        )
        
        results = lpdid.fit()
        self.assertIsNotNone(results.event_study)
    
    def test_parallel_processing(self):
        """Test parallel processing gives same results"""
        # Sequential
        lpdid_seq = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry",
            n_jobs=1
        )
        results_seq = lpdid_seq.fit()
        
        # Parallel
        lpdid_par = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry",
            n_jobs=2
        )
        results_par = lpdid_par.fit()
        
        # Compare coefficients
        coef_seq = results_seq.event_study['coefficient'].values
        coef_par = results_par.event_study['coefficient'].values
        
        np.testing.assert_array_almost_equal(coef_seq, coef_par)
    
    def test_only_event(self):
        """Test only event study estimation"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit(only_event=True)
        self.assertIsNotNone(results.event_study)
        self.assertIsNone(results.pooled)
    
    def test_only_pooled(self):
        """Test only pooled estimation"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit(only_pooled=True)
        self.assertIsNone(results.event_study)
        self.assertIsNotNone(results.pooled)
    
    def test_custom_pooled_windows(self):
        """Test custom pooled windows"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=5,
            post_window=8
        )
        
        results = lpdid.fit(
            post_pooled=(0, 3),
            pre_pooled=(2, 4)
        )
        
        self.assertIsNotNone(results.pooled)
    
    def test_plot(self):
        """Test plotting functionality"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit()
        
        # Should not raise error
        fig, ax = results.plot()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
    
    def test_nonabsorbing_treatment(self):
        """Test non-absorbing treatment"""
        # Create non-absorbing treatment data
        df_nonabs = self.df.copy()
        # Add some units that switch out of treatment
        mask = (df_nonabs['unit'] < 10) & (df_nonabs['time'] > 15)
        df_nonabs.loc[mask, 'treat'] = 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpdid = LPDiD(
                data=df_nonabs,
                depvar='y',
                unit='unit',
                time='time',
                treat='treat',
                pre_window=3,
                post_window=5,
                nonabsorbing=(3, False, False)
            )
            
            results = lpdid.fit()
            self.assertIsNotNone(results.event_study)
    
    def test_input_validation(self):
        """Test input validation"""
        # No windows specified
        with self.assertRaises(ValueError):
            LPDiD(
                data=self.df,
                depvar='y',
                unit='unit',
                time='time',
                treat='treat'
            )
        
        # Invalid pre_window
        with self.assertRaises(ValueError):
            LPDiD(
                data=self.df,
                depvar='y',
                unit='unit',
                time='time',
                treat='treat',
                pre_window=1,
                post_window=5
            )
        
        # Non-binary treatment
        df_bad = self.df.copy()
        df_bad.loc[0, 'treat'] = 2
        with self.assertRaises(ValueError):
            LPDiD(
                data=df_bad,
                depvar='y',
                unit='unit',
                time='time',
                treat='treat',
                pre_window=3,
                post_window=5
            )
    
    def test_iv_specification(self):
        """Test IV specification"""
        # Add instruments to test data
        self.df['instrument1'] = np.random.normal(0, 1, len(self.df))
        self.df['instrument2'] = np.random.binomial(1, 0.5, len(self.df))
        
        lpdid_iv = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry | D_treat ~ instrument1 + instrument2"
        )
        
        # Check that IV spec was parsed correctly
        self.assertIsNotNone(lpdid_iv.iv_spec)
        self.assertEqual(lpdid_iv.iv_spec['endog'], ['D_treat'])
        self.assertEqual(lpdid_iv.iv_spec['instruments'], ['instrument1', 'instrument2'])
        
        results = lpdid_iv.fit()
        self.assertIsNotNone(results.event_study)
        self.assertIsNotNone(results.iv_diagnostics)
        
        # Check that IV diagnostics are present
        self.assertIn('first_stage_F', results.iv_diagnostics.columns)
        self.assertIn('weak_iv', results.iv_diagnostics.columns)
    
    def test_interactions(self):
        """Test interaction terms"""
        lpdid_interact = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry",
            interactions="~ x2"
        )
        
        # Check that interactions were parsed
        self.assertEqual(lpdid_interact.interact_vars, ['x2'])
        
        results = lpdid_interact.fit()
        self.assertIsNotNone(results.event_study)
        
        # Check that interaction results are present
        self.assertIn('x2_interaction', results.event_study.columns)
        self.assertIn('x2_interaction_se', results.event_study.columns)
        self.assertIn('x2_interaction_p', results.event_study.columns)
    
    def test_multiple_interactions(self):
        """Test multiple interaction terms"""
        lpdid_multi = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1",
            interactions="~ x2 + industry"
        )
        
        results = lpdid_multi.fit()
        
        # Check multiple interactions
        self.assertIn('x2_interaction', results.event_study.columns)
        self.assertIn('industry_interaction', results.event_study.columns)
    
    def test_iv_with_interactions(self):
        """Test combined IV and interactions"""
        self.df['instrument'] = np.random.normal(0, 1, len(self.df))
        
        lpdid_iv_het = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5,
            formula="~ x1 | industry | D_treat ~ instrument",
            interactions="~ x2"
        )
        
        results = lpdid_iv_het.fit()
        
        # Should have both IV diagnostics and interactions
        self.assertIsNotNone(results.iv_diagnostics)
        self.assertIn('x2_interaction', results.event_study.columns)
    
    def test_summary_method(self):
        """Test summary method"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit()
        
        # Should not raise error
        results.summary()
    
    def test_no_plotting(self):
        """Test that plotting is removed"""
        lpdid = LPDiD(
            data=self.df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=5
        )
        
        results = lpdid.fit()
        
        # plot method should not exist
        self.assertFalse(hasattr(results, 'plot'))


if __name__ == '__main__':
    unittest.main()