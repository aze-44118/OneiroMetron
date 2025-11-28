"""
Empirical Analysis Module
==========================
All regression specifications and robustness checks
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class EmpiricalAnalyzer:
    """Run all empirical specifications"""
    
    def __init__(self, data: pd.DataFrame, config):
        self.data = data.copy()
        self.config = config
        
        # Drop any rows with missing key variables
        key_vars = ['log_sales_proxy', 'log_views', 'barrier_index']
        self.data = self.data.dropna(subset=key_vars)
        
        logger.info(f"Analysis dataset: {len(self.data)} observations")
    
    def run_model1_pooled_ols(self) -> Dict:
        """
        Model 1: Pooled OLS (Baseline)
        Y_it = β₀ + β₁·X_it + β₂·Controls + ε_it
        """
        logger.info("Estimating Model 1: Pooled OLS...")
        
        formula = 'log_sales_proxy ~ log_views + unemployment_rate + gas_price'
        
        model = smf.ols(formula, data=self.data).fit(cov_type='HC1')
        
        results = {
            'model': model,
            'formula': formula,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'beta_views': float(model.params['log_views']),
            'se_views': float(model.bse['log_views']),
            'pvalue_views': float(model.pvalues['log_views']),
            'coef_table': model.summary2().tables[1]
        }
        
        return results
    
    def run_model2_fixed_effects(self) -> Dict:
        """
        Model 2: Fixed Effects (Sport FE + Time FE)
        Y_it = β₁·X_it + β₂·Controls + α_i + λ_t + ε_it
        """
        logger.info("Estimating Model 2: Fixed Effects...")
        
        formula = 'log_sales_proxy ~ log_views + unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)'
        
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['sport']}
        )
        
        # Calculate within R²
        # Demean by sport and time
        data_demeaned = self.data.copy()
        for var in ['log_sales_proxy', 'log_views']:
            data_demeaned[f'{var}_demeaned'] = (
                data_demeaned.groupby('sport')[var].transform(lambda x: x - x.mean())
            )
        
        # Simple regression on demeaned data
        model_within = smf.ols(
            'log_sales_proxy_demeaned ~ log_views_demeaned - 1',
            data=data_demeaned
        ).fit()
        
        r_squared_within = float(model_within.rsquared)
        
        results = {
            'model': model,
            'formula': formula,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'r_squared_within': r_squared_within,
            'beta_views': float(model.params['log_views']),
            'se_views': float(model.bse['log_views']),
            'pvalue_views': float(model.pvalues['log_views']),
            'coef_table': model.summary2().tables[1]
        }
        
        return results
    
    def run_model3_interaction(self) -> Dict:
        """
        Model 3: FE + Interaction (CORE MODEL)
        Y_it = β₁·X_it + β₃·(Z_i × X_it) + β₄·Controls + α_i + λ_t + ε_it
        """
        logger.info("Estimating Model 3: FE + Interaction (CORE MODEL)...")
        
        formula = ('log_sales_proxy ~ log_views + log_views:barrier_index + '
                   'unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)')
        
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['sport']}
        )
        
        # Extract key coefficients
        beta_views = float(model.params['log_views'])
        beta_interaction = float(model.params['log_views:barrier_index'])
        
        # Calculate implied elasticities at different barrier levels
        barriers = [1.2, 3.0, 5.0, 6.0, 8.7]
        elasticities = {
            f'barrier_{b}': beta_views + beta_interaction * b
            for b in barriers
        }
        
        results = {
            'model': model,
            'formula': formula,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'beta_views': beta_views,
            'se_views': float(model.bse['log_views']),
            'pvalue_views': float(model.pvalues['log_views']),
            'beta_interaction': beta_interaction,
            'se_interaction': float(model.bse['log_views:barrier_index']),
            'pvalue_interaction': float(model.pvalues['log_views:barrier_index']),
            'elasticities': elasticities,
            'coef_table': model.summary2().tables[1]
        }
        
        return results
    
    def run_model4_iv(self) -> Dict:
        """
        Model 4: Instrumental Variables (2SLS)
        First stage: X_it = π₀ + π₁·IV_it + π₂·Controls + ν_it
        Second stage: Y_it = β₁·X̂_it + β₂·Controls + ε_it
        """
        logger.info("Estimating Model 4: IV/2SLS...")
        
        # First stage
        first_stage_formula = ('log_views ~ documentary_release + post_documentary + '
                                'unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)')
        
        first_stage = smf.ols(first_stage_formula, data=self.data).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.data['sport']}
        )
        
        first_stage_f = float(first_stage.fvalue)
        
        # Prepare data for IV2SLS
        # Create dummy variables for FE
        sport_dummies = pd.get_dummies(self.data['sport'], prefix='sport', drop_first=True)
        year_dummies = pd.get_dummies(self.data['year'], prefix='year', drop_first=True)
        quarter_dummies = pd.get_dummies(self.data['quarter'], prefix='quarter', drop_first=True)
        
        # Exogenous variables (controls + FE)
        exog = pd.concat([
            self.data[['unemployment_rate', 'gas_price']],
            sport_dummies,
            year_dummies,
            quarter_dummies
        ], axis=1)
        
        # Add constant
        exog.insert(0, 'const', 1)
        
        # Endogenous variable
        endog = self.data[['log_views']]
        
        # Instruments
        instruments = self.data[['documentary_release', 'post_documentary']]
        
        # Dependent variable
        dependent = self.data['log_sales_proxy']
        
        try:
            # Run 2SLS
            iv_model = IV2SLS(
                dependent=dependent,
                exog=exog,
                endog=endog,
                instruments=instruments
            ).fit(cov_type='clustered', cov_kwds={'groups': self.data['sport']})
            
            beta_views_iv = float(iv_model.params['log_views'])
            se_views_iv = float(iv_model.std_errors['log_views'])
            pvalue_views_iv = float(iv_model.pvalues['log_views'])
            
        except Exception as e:
            logger.warning(f"IV estimation failed: {e}. Using OLS estimates.")
            beta_views_iv = np.nan
            se_views_iv = np.nan
            pvalue_views_iv = np.nan
        
        results = {
            'first_stage_model': first_stage,
            'first_stage_formula': first_stage_formula,
            'first_stage_f': first_stage_f,
            'weak_instrument': first_stage_f < self.config.WEAK_INSTRUMENT_THRESHOLD,
            'beta_views_iv': beta_views_iv,
            'se_views_iv': se_views_iv,
            'pvalue_views_iv': pvalue_views_iv,
            'n_obs': int(first_stage.nobs)
        }
        
        return results
    
    def run_heterogeneity_analysis(self) -> Dict:
        """
        Split sample analysis by barrier level
        """
        logger.info("Running heterogeneity analysis (split sample)...")
        
        # Define groups
        low_barrier = self.data[self.data['barrier_index'] < self.config.BARRIER_LOW_THRESHOLD]
        medium_barrier = self.data[
            (self.data['barrier_index'] >= self.config.BARRIER_LOW_THRESHOLD) &
            (self.data['barrier_index'] <= self.config.BARRIER_HIGH_THRESHOLD)
        ]
        high_barrier = self.data[self.data['barrier_index'] > self.config.BARRIER_HIGH_THRESHOLD]
        
        formula = 'log_sales_proxy ~ log_views + unemployment_rate + gas_price + C(year) + C(quarter)'
        
        results = {}
        
        # Low barrier
        if len(low_barrier) > 30:
            model_low = smf.ols(formula, data=low_barrier).fit(
                cov_type='cluster',
                cov_kwds={'groups': low_barrier['sport']}
            )
            results['low'] = {
                'model': model_low,
                'n_obs': int(model_low.nobs),
                'elasticity': float(model_low.params['log_views']),
                'se': float(model_low.bse['log_views']),
                'pvalue': float(model_low.pvalues['log_views']),
                'sports': list(low_barrier['sport'].unique())
            }
        else:
            results['low'] = {'elasticity': np.nan, 'n_obs': 0}
        
        # Medium barrier
        if len(medium_barrier) > 30:
            model_medium = smf.ols(formula, data=medium_barrier).fit(
                cov_type='cluster',
                cov_kwds={'groups': medium_barrier['sport']}
            )
            results['medium'] = {
                'model': model_medium,
                'n_obs': int(model_medium.nobs),
                'elasticity': float(model_medium.params['log_views']),
                'se': float(model_medium.bse['log_views']),
                'pvalue': float(model_medium.pvalues['log_views']),
                'sports': list(medium_barrier['sport'].unique())
            }
        else:
            results['medium'] = {'elasticity': np.nan, 'n_obs': 0}
        
        # High barrier
        if len(high_barrier) > 30:
            model_high = smf.ols(formula, data=high_barrier).fit(
                cov_type='cluster',
                cov_kwds={'groups': high_barrier['sport']}
            )
            results['high'] = {
                'model': model_high,
                'n_obs': int(model_high.nobs),
                'elasticity': float(model_high.params['log_views']),
                'se': float(model_high.bse['log_views']),
                'pvalue': float(model_high.pvalues['log_views']),
                'sports': list(high_barrier['sport'].unique())
            }
        else:
            results['high'] = {'elasticity': np.nan, 'n_obs': 0}
        
        # Summary
        results['low_elasticity'] = results['low']['elasticity']
        results['medium_elasticity'] = results['medium']['elasticity']
        results['high_elasticity'] = results['high']['elasticity']
        
        return results
    
    def run_robustness_checks(self) -> Dict:
        """
        Run all robustness checks
        """
        logger.info("Running robustness checks...")
        
        results = {}
        
        # 1. Quadratic specification
        try:
            self.data['log_views_sq'] = self.data['log_views'] ** 2
            formula_quad = ('log_sales_proxy ~ log_views + log_views_sq + '
                            'unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)')
            model_quad = smf.ols(formula_quad, data=self.data).fit(
                cov_type='cluster',
                cov_kwds={'groups': self.data['sport']}
            )
            results['quadratic'] = {
                'beta_linear': float(model_quad.params['log_views']),
                'beta_quadratic': float(model_quad.params['log_views_sq']),
                'r_squared': float(model_quad.rsquared)
            }
        except Exception as e:
            logger.warning(f"Quadratic specification failed: {e}")
            results['quadratic'] = {}
        
        # 2. Lagged effects
        try:
            self.data = self.data.sort_values(['sport', 'year', 'quarter'])
            self.data['log_views_lag1'] = self.data.groupby('sport')['log_views'].shift(1)
            data_lagged = self.data[self.data['log_views_lag1'].notna()].copy()
            
            formula_lag = ('log_sales_proxy ~ log_views + log_views_lag1 + '
                           'unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)')
            model_lag = smf.ols(formula_lag, data=data_lagged).fit(
                cov_type='cluster',
                cov_kwds={'groups': data_lagged['sport']}
            )
            results['lagged'] = {
                'beta_current': float(model_lag.params['log_views']),
                'beta_lag1': float(model_lag.params['log_views_lag1']),
                'r_squared': float(model_lag.rsquared)
            }
        except Exception as e:
            logger.warning(f"Lagged specification failed: {e}")
            results['lagged'] = {}
        
        # 3. Pre/Post COVID
        try:
            data_pre = self.data[self.data['year'] < 2020]
            data_post = self.data[self.data['year'] >= 2020]
            
            formula_simple = 'log_sales_proxy ~ log_views + unemployment_rate + gas_price + C(sport)'
            
            model_pre = smf.ols(formula_simple, data=data_pre).fit(
                cov_type='cluster',
                cov_kwds={'groups': data_pre['sport']}
            )
            model_post = smf.ols(formula_simple, data=data_post).fit(
                cov_type='cluster',
                cov_kwds={'groups': data_post['sport']}
            )
            
            results['covid_split'] = {
                'beta_pre': float(model_pre.params['log_views']),
                'beta_post': float(model_post.params['log_views']),
                'n_pre': int(model_pre.nobs),
                'n_post': int(model_post.nobs)
            }
        except Exception as e:
            logger.warning(f"COVID split failed: {e}")
            results['covid_split'] = {}
        
        # 4. Alternative clustering (by year)
        try:
            formula_base = 'log_sales_proxy ~ log_views + unemployment_rate + gas_price + C(sport) + C(year) + C(quarter)'
            model_cluster_year = smf.ols(formula_base, data=self.data).fit(
                cov_type='cluster',
                cov_kwds={'groups': self.data['year']}
            )
            results['cluster_year'] = {
                'beta_views': float(model_cluster_year.params['log_views']),
                'se_views': float(model_cluster_year.bse['log_views']),
                'r_squared': float(model_cluster_year.rsquared)
            }
        except Exception as e:
            logger.warning(f"Alternative clustering failed: {e}")
            results['cluster_year'] = {}
        
        return results
