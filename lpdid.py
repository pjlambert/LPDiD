    def fit(self):
        """
        Fit the Local Projections Difference-in-Differences (LP-DiD) model.

        This method executes the full LP-DiD estimation pipeline:
        1. Identifies clean control units that are never treated.
        2. Constructs long-differenced outcomes for each horizon.
        3. Computes weights for the regression if specified.
        4. Runs separate regressions for each pre- and post-treatment horizon.
        5. Collates the results into an `LPDiDResults` object.

        Returns
        -------
        LPDiDResults
            An object containing the estimated event-study coefficients and other relevant information.
        """
        # Step 1: Identify units that are never treated to serve as clean controls.
        self._identify_clean_controls()
        
        # Step 2: Create the long-differenced outcome variable for each horizon.
        self._generate_long_differences()
        
        # Step 3: Compute regression weights if a weighting variable is specified.
        self._compute_weights()
        
        # Step 4: Run event-study regressions for each specified horizon.
        event_study_results = []
        
        # Run regressions for each pre-treatment period (h=2 to pre_window).
        # h=1 is the period just before treatment, which is normalized to 0 and omitted here.
        for h in range(2, self.pre_window + 1):
            result = self._run_single_regression(h, is_pre=True)
            if result:
                event_study_results.append(result)
        
        # Run regressions for each post-treatment period (h=0 to post_window).
        # h=0 is the treatment period itself.
        for h in range(self.post_window + 1):
            result = self._run_single_regression(h, is_pre=False)
            if result:
                event_study_results.append(result)
        
        # Step 5: Collate results into a structured DataFrame.
        if event_study_results:
            event_study_df = pd.DataFrame(event_study_results)
            # Sort results by horizon for chronological plotting and analysis.
            event_study_df = event_study_df.sort_values('horizon').reset_index(drop=True)
        else:
            # If no regressions were successful, create an empty DataFrame with the expected structure.
            event_study_df = pd.DataFrame(columns=['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs'])
        
        # Normalize the event-study plot by setting the coefficient for h=-1 to 0.
        # This is a standard convention to show treatment effects relative to the period just before treatment.
        if -1 not in event_study_df['horizon'].values:
            h_minus_1_row = {
                'horizon': -1,
                'coefficient': 0.0,
                'se': np.nan,
                't': np.nan,
                'p': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'obs': np.nan
            }
            event_study_df = pd.concat([pd.DataFrame([h_minus_1_row]), event_study_df], ignore_index=True)
            event_study_df = event_study_df.sort_values('horizon').reset_index(drop=True)
        
        # Step 6: Package the results into a dedicated LPDiDResults object for easy access and plotting.
        results = LPDiDResults(
            event_study=event_study_df,
            depvar=self.depvar,
            pre_window=self.pre_window,
            post_window=self.post_window,
            control_group="Controls",
            treated_group="Treated"
        )
        
        return results
