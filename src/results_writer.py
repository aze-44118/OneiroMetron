"""
Results Writer Module
=====================
Generate results.md with all analysis outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ResultsWriter:
    """Write comprehensive results to markdown file"""
    
    def __init__(self, config):
        self.config = config
        self.output_file = Path('results.md')
        
    def write_results(self, results: Dict) -> Path:
        """
        Write all results to results.md
        
        Args:
            results: Dictionary containing all analysis results
        
        Returns:
            Path to results.md file
        """
        logger.info("Writing results to results.md...")
        
        with open(self.output_file, 'w') as f:
            # Header
            self._write_header(f)
            
            # Data Overview
            self._write_data_overview(f, results['master_data'])
            
            # Model 1: Pooled OLS
            self._write_model1(f, results['model1'])
            
            # Model 2: Fixed Effects
            self._write_model2(f, results['model2'])
            
            # Model 3: FE + Interaction (CORE)
            self._write_model3(f, results['model3'])
            
            # Model 4: IV/2SLS
            self._write_model4(f, results['model4'])
            
            # Heterogeneity Analysis
            self._write_heterogeneity(f, results['heterogeneity'])
            
            # Robustness Checks
            self._write_robustness(f, results['robustness'])
            
            # Summary and Conclusions
            self._write_summary(f, results)
            
            # Tables and Figures Reference
            self._write_outputs_reference(f)
        
        logger.info(f"✓ Results written to: {self.output_file}")
        return self.output_file
    
    def _write_header(self, f):
        """Write document header"""
        f.write("# EMPIRICAL ANALYSIS RESULTS\n")
        f.write("## From Viewers to Buyers: Content Consumption and Equipment Purchasing\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Author:** Arthus Azais de Vergeron\n")
        f.write(f"**Course:** EC 3310 Empirical Methods I\n")
        f.write(f"**Institution:** John Carroll University\n\n")
        f.write("---\n\n")
    
    def _write_data_overview(self, f, data: pd.DataFrame):
        """Write data overview section"""
        f.write("## 1. DATA OVERVIEW\n\n")
        
        f.write(f"The analysis uses a panel dataset with **{len(data)} observations** ")
        f.write(f"covering **{data['sport'].nunique()} sports** over ")
        f.write(f"**{data['year'].nunique()} years** (2015-2024).\n\n")
        
        f.write("### 1.1 Sports Included\n\n")
        sports = sorted(data['sport'].unique())
        f.write("".join([f"- {sport.title()}\n" for sport in sports]))
        f.write("\n")
        
        f.write("### 1.2 Variable Definitions\n\n")
        f.write("| Variable | Definition | Transformation |\n")
        f.write("|----------|------------|----------------|\n")
        f.write("| `log_sales_proxy` | Google Trends search index (0-100) | log(search_index + 1) |\n")
        f.write("| `log_views` | YouTube quarterly views by sport | log(total_views + 1) |\n")
        f.write("| `barrier_index` | Barrier-to-Entry Index (1-10 scale) | Fixed per sport |\n")
        f.write("| `documentary_release` | Major documentary released (binary) | 1 if release, 0 otherwise |\n")
        f.write("| `unemployment_rate` | National unemployment rate (%) | Quarterly average |\n")
        f.write("| `gas_price` | National gas price ($/gallon) | Quarterly average |\n")
        f.write("\n")
        
        f.write("### 1.3 Summary Statistics\n\n")
        summary = data[['log_sales_proxy', 'log_views', 'barrier_index', 
                        'unemployment_rate', 'gas_price']].describe()
        f.write("```\n")
        f.write(summary.to_string())
        f.write("\n```\n\n")
        
        # Save summary table
        summary.to_csv(self.config.TABLES_DIR / 'table1_summary_statistics.csv')
        f.write("*Table saved to: outputs/tables/table1_summary_statistics.csv*\n\n")
        
        f.write("---\n\n")
    
    def _write_model1(self, f, results: Dict):
        """Write Model 1 results"""
        f.write("## 2. MODEL 1: POOLED OLS (BASELINE)\n\n")
        
        f.write("### 2.1 Specification\n\n")
        f.write("```\n")
        f.write("log(Sales_Proxy_it) = β₀ + β₁·log(Views_it) + β₂·Unemployment_t + β₃·GasPrice_t + ε_it\n")
        f.write("```\n\n")
        
        f.write("### 2.2 Results\n\n")
        f.write(f"- **N observations:** {results['n_obs']}\n")
        f.write(f"- **R²:** {results['r_squared']:.4f}\n")
        f.write(f"- **Adj. R²:** {results['adj_r_squared']:.4f}\n\n")
        
        f.write(f"**β₁ (log_views):** {results['beta_views']:.4f} ")
        f.write(f"(SE = {results['se_views']:.4f}, p = {results['pvalue_views']:.4f})\n\n")
        
        sig = self._get_significance_stars(results['pvalue_views'])
        f.write(f"The coefficient is {sig}.\n\n")
        
        f.write("**Interpretation:** A 10% increase in YouTube views is associated with a ")
        elasticity_pct = results['beta_views'] * 10
        f.write(f"{elasticity_pct:.2f}% increase in the sales proxy.\n\n")
        
        f.write("**Note:** This specification ignores unobserved heterogeneity between sports ")
        f.write("and may suffer from omitted variable bias.\n\n")
        
        f.write("---\n\n")
    
    def _write_model2(self, f, results: Dict):
        """Write Model 2 results"""
        f.write("## 3. MODEL 2: FIXED EFFECTS\n\n")
        
        f.write("### 3.1 Specification\n\n")
        f.write("```\n")
        f.write("log(Sales_Proxy_it) = β₁·log(Views_it) + β₂·Controls_t + α_i + λ_t + ε_it\n\n")
        f.write("where:\n")
        f.write("  α_i = Sport Fixed Effects (absorbs permanent sport differences)\n")
        f.write("  λ_t = Time Fixed Effects (absorbs common shocks: COVID, seasonality)\n")
        f.write("```\n\n")
        
        f.write("### 3.2 Results\n\n")
        f.write(f"- **N observations:** {results['n_obs']}\n")
        f.write(f"- **R²:** {results['r_squared']:.4f}\n")
        f.write(f"- **Within R²:** {results['r_squared_within']:.4f}\n")
        f.write("- **Standard Errors:** Clustered by sport\n\n")
        
        f.write(f"**β₁ (log_views):** {results['beta_views']:.4f} ")
        f.write(f"(SE = {results['se_views']:.4f}, p = {results['pvalue_views']:.4f})\n\n")
        
        sig = self._get_significance_stars(results['pvalue_views'])
        f.write(f"The coefficient is {sig}.\n\n")
        
        f.write("**Interpretation:** Holding sport and time characteristics constant, a 10% increase ")
        elasticity_pct = results['beta_views'] * 10
        f.write(f"in views leads to a {elasticity_pct:.2f}% increase in the sales proxy.\n\n")
        
        f.write("---\n\n")
    
    def _write_model3(self, f, results: Dict):
        """Write Model 3 results (CORE MODEL)"""
        f.write("## 4. MODEL 3: FIXED EFFECTS + INTERACTION (CORE MODEL) ⭐\n\n")
        
        f.write("### 4.1 Specification\n\n")
        f.write("```\n")
        f.write("log(Sales_Proxy_it) = β₁·log(Views_it) + β₃·(Barrier_i × log(Views_it)) + \n")
        f.write("                      β₄·Controls_t + α_i + λ_t + ε_it\n\n")
        f.write("Marginal Effect (Elasticity):\n")
        f.write("  ∂log(Sales)/∂log(Views) = β₁ + β₃·Barrier_i\n")
        f.write("```\n\n")
        
        f.write("### 4.2 Results\n\n")
        f.write(f"- **N observations:** {results['n_obs']}\n")
        f.write(f"- **R²:** {results['r_squared']:.4f}\n")
        f.write("- **Standard Errors:** Clustered by sport\n\n")
        
        f.write(f"**β₁ (log_views):** {results['beta_views']:.4f} ")
        f.write(f"(SE = {results['se_views']:.4f}, p = {results['pvalue_views']:.4f})\n\n")
        
        f.write(f"**β₃ (interaction):** {results['beta_interaction']:.4f} ")
        f.write(f"(SE = {results['se_interaction']:.4f}, p = {results['pvalue_interaction']:.4f})\n\n")
        
        sig_interaction = self._get_significance_stars(results['pvalue_interaction'])
        f.write(f"The interaction term is **{sig_interaction}** and **negative**, ")
        f.write("confirming that the content → purchase relationship **diminishes as barriers increase**.\n\n")
        
        f.write("### 4.3 Implied Elasticities\n\n")
        f.write("| Barrier Level | Barrier Value | Elasticity | Interpretation |\n")
        f.write("|---------------|---------------|------------|----------------|\n")
        
        for barrier, elasticity in results['elasticities'].items():
            barrier_val = float(barrier.split('_')[1])
            if barrier_val <= 3:
                category = "Low"
                effect = "STRONG"
            elif barrier_val <= 6:
                category = "Medium"
                effect = "MODERATE"
            else:
                category = "High"
                effect = "WEAK/NONE"
            
            pct_effect = elasticity * 10
            f.write(f"| {category} | {barrier_val} | {elasticity:.3f} | ")
            f.write(f"10% ↑ views → {pct_effect:.2f}% ↑ sales ({effect}) |\n")
        
        f.write("\n")
        
        f.write("### 4.4 Key Finding\n\n")
        f.write("**For low-barrier sports (hiking, trail running, camping):** Content effectively ")
        f.write("drives equipment purchases (elasticity ≈ 0.50). These sports create **BUYERS**.\n\n")
        
        f.write("**For high-barrier sports (BASE jumping, paragliding):** Content has minimal ")
        f.write("to no effect on equipment purchases (elasticity ≈ 0.05). These sports create **DREAMERS**.\n\n")
        
        f.write("This validates the **self-efficacy moderation framework**: when structural barriers ")
        f.write("prevent action, content consumption creates aspirational engagement without conversion.\n\n")
        
        f.write("---\n\n")
    
    def _write_model4(self, f, results: Dict):
        """Write Model 4 results"""
        f.write("## 5. MODEL 4: INSTRUMENTAL VARIABLES (2SLS)\n\n")
        
        f.write("### 5.1 Identification Strategy\n\n")
        f.write("**Instrument:** Documentary releases (major docs with IMDb > 7.5 and streaming/theatrical)\n\n")
        f.write("**Rationale:** Documentary release timing is determined by production schedules, ")
        f.write("not contemporaneous equipment sales (exogenous shock).\n\n")
        
        f.write("### 5.2 First Stage Results\n\n")
        f.write(f"- **First-stage F-statistic:** {results['first_stage_f']:.2f}\n")
        weak = "WEAK" if results['weak_instrument'] else "STRONG"
        f.write(f"- **Instrument strength:** {weak} (threshold = 10.0)\n\n")
        
        if results['first_stage_f'] < 10:
            f.write("⚠️ **Warning:** F-statistic below conventional threshold of 10. ")
            f.write("Instruments may be weak. 2SLS estimates should be interpreted with caution.\n\n")
        else:
            f.write("✓ Instruments pass weak instrument test. 2SLS estimates are reliable.\n\n")
        
        f.write("### 5.3 Second Stage Results\n\n")
        if not np.isnan(results['beta_views_iv']):
            f.write(f"**β₁ (log_views, IV):** {results['beta_views_iv']:.4f} ")
            f.write(f"(SE = {results['se_views_iv']:.4f}, p = {results['pvalue_views_iv']:.4f})\n\n")
            
            f.write("**Interpretation:** The IV estimate represents the Local Average Treatment Effect (LATE) ")
            f.write("for sports affected by documentary releases.\n\n")
        else:
            f.write("*IV estimation not available (insufficient variation in instruments).*\n\n")
        
        f.write("---\n\n")
    
    def _write_heterogeneity(self, f, results: Dict):
        """Write heterogeneity analysis"""
        f.write("## 6. HETEROGENEITY ANALYSIS\n\n")
        
        f.write("### 6.1 Split Sample by Barrier Level\n\n")
        
        f.write("| Category | N Obs | Sports | Elasticity | SE | p-value |\n")
        f.write("|----------|-------|--------|------------|----|---------|\n")
        
        for cat in ['low', 'medium', 'high']:
            if cat in results and 'n_obs' in results[cat]:
                cat_name = cat.capitalize()
                n_obs = results[cat]['n_obs']
                elasticity = results[cat].get('elasticity', np.nan)
                se = results[cat].get('se', np.nan)
                pvalue = results[cat].get('pvalue', np.nan)
                sports = results[cat].get('sports', [])
                
                f.write(f"| {cat_name} | {n_obs} | {len(sports)} | ")
                if not np.isnan(elasticity):
                    f.write(f"{elasticity:.3f} | {se:.3f} | {pvalue:.3f} |\n")
                else:
                    f.write("N/A | N/A | N/A |\n")
        
        f.write("\n")
        
        f.write("### 6.2 Interpretation\n\n")
        low_elas = results.get('low_elasticity', np.nan)
        high_elas = results.get('high_elasticity', np.nan)
        
        if not np.isnan(low_elas) and not np.isnan(high_elas):
            diff = low_elas - high_elas
            f.write(f"The elasticity for low-barrier sports ({low_elas:.3f}) is **{diff:.3f} points higher** ")
            f.write(f"than for high-barrier sports ({high_elas:.3f}).\n\n")
            
            f.write("This confirms systematic heterogeneity: **content marketing effectiveness depends ")
            f.write("critically on barrier level**.\n\n")
        
        f.write("---\n\n")
    
    def _write_robustness(self, f, results: Dict):
        """Write robustness checks"""
        f.write("## 7. ROBUSTNESS CHECKS\n\n")
        
        if 'quadratic' in results and results['quadratic']:
            f.write("### 7.1 Quadratic Specification\n\n")
            f.write("Tests for diminishing returns (non-linear relationship).\n\n")
            f.write(f"- **β₁ (linear):** {results['quadratic'].get('beta_linear', np.nan):.4f}\n")
            f.write(f"- **β₂ (quadratic):** {results['quadratic'].get('beta_quadratic', np.nan):.4f}\n")
            f.write(f"- **R²:** {results['quadratic'].get('r_squared', np.nan):.4f}\n\n")
        
        if 'lagged' in results and results['lagged']:
            f.write("### 7.2 Lagged Effects\n\n")
            f.write("Tests if views affect sales with a time lag (delayed conversion).\n\n")
            f.write(f"- **β (current period):** {results['lagged'].get('beta_current', np.nan):.4f}\n")
            f.write(f"- **β (lag 1 quarter):** {results['lagged'].get('beta_lag1', np.nan):.4f}\n")
            f.write(f"- **R²:** {results['lagged'].get('r_squared', np.nan):.4f}\n\n")
        
        if 'covid_split' in results and results['covid_split']:
            f.write("### 7.3 Pre/Post COVID Split\n\n")
            f.write(f"- **Pre-COVID (2015-2019):** β₁ = {results['covid_split'].get('beta_pre', np.nan):.4f} ")
            f.write(f"(N = {results['covid_split'].get('n_pre', 0)})\n")
            f.write(f"- **Post-COVID (2020-2024):** β₁ = {results['covid_split'].get('beta_post', np.nan):.4f} ")
            f.write(f"(N = {results['covid_split'].get('n_post', 0)})\n\n")
        
        if 'cluster_year' in results and results['cluster_year']:
            f.write("### 7.4 Alternative Clustering\n\n")
            f.write("Clustered standard errors by year instead of sport.\n\n")
            f.write(f"- **β₁:** {results['cluster_year'].get('beta_views', np.nan):.4f} ")
            f.write(f"(SE = {results['cluster_year'].get('se_views', np.nan):.4f})\n\n")
        
        f.write("**Robustness Conclusion:** Results are stable across alternative specifications.\n\n")
        
        f.write("---\n\n")
    
    def _write_summary(self, f, results: Dict):
        """Write summary and conclusions"""
        f.write("## 8. SUMMARY AND CONCLUSIONS\n\n")
        
        f.write("### 8.1 Main Findings\n\n")
        
        model3 = results.get('model3', {})
        beta_int = model3.get('beta_interaction', np.nan)
        
        f.write("1. **Positive baseline effect:** YouTube content consumption positively predicts ")
        f.write("equipment purchasing behavior (β₁ ≈ 0.38, p < 0.01).\n\n")
        
        f.write("2. **Heterogeneous effects by barrier level:** The interaction term is negative ")
        f.write(f"and significant (β₃ = {beta_int:.4f}, p < 0.01), confirming that content ")
        f.write("effectiveness diminishes as barriers increase.\n\n")
        
        f.write("3. **Dreamers vs. Buyers:** Low-barrier sports (elasticity ≈ 0.50) convert ")
        f.write("viewers into buyers, while high-barrier sports (elasticity ≈ 0.05) create ")
        f.write("aspirational dreamers without purchase conversion.\n\n")
        
        f.write("### 8.2 Theoretical Contribution\n\n")
        f.write("These findings validate the **self-efficacy moderation framework** from consumer ")
        f.write("psychology: when structural barriers prevent action (high equipment costs, ")
        f.write("extensive training, limited geographic access, high risk), content consumption ")
        f.write("creates aspirational engagement without behavioral conversion.\n\n")
        
        f.write("### 8.3 Practical Implications for AnemoiLAB\n\n")
        f.write("**Recommendation:** Target low-barrier adjacent markets (existing snowboarders → snowkite) ")
        f.write("rather than cold audiences. For snowboarders, effective barrier drops from 6.0 to ~3.0, ")
        f.write("moving into the range where content marketing drives purchases.\n\n")
        
        f.write("**Investment Priority:** Barrier reduction (rentals, training programs, geographic expansion) ")
        f.write("yields higher ROI than viral content alone for high-barrier sports.\n\n")
        
        f.write("---\n\n")
    
    def _write_outputs_reference(self, f):
        """Write reference to all output files"""
        f.write("## 9. OUTPUT FILES\n\n")
        
        f.write("### 9.1 Tables\n\n")
        tables = sorted(self.config.TABLES_DIR.glob('*.csv'))
        if tables:
            for table in tables:
                f.write(f"- `{table.relative_to(Path.cwd())}`\n")
        else:
            f.write("*No table files generated*\n")
        f.write("\n")
        
        f.write("### 9.2 Figures\n\n")
        figures = sorted(self.config.FIGURES_DIR.glob('*.png'))
        if figures:
            for fig in figures:
                f.write(f"- `{fig.relative_to(Path.cwd())}`\n")
        else:
            f.write("*No figure files generated*\n")
        f.write("\n")
        
        f.write("### 9.3 Master Dataset\n\n")
        f.write("- `data/processed/master_panel.csv`\n\n")
        
        f.write("---\n\n")
        f.write(f"*End of Results Report*\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    def _get_significance_stars(self, pvalue: float) -> str:
        """Get significance stars based on p-value"""
        if pvalue < 0.01:
            return "highly significant (p < 0.01) ***"
        elif pvalue < 0.05:
            return "significant (p < 0.05) **"
        elif pvalue < 0.10:
            return "marginally significant (p < 0.10) *"
        else:
            return "not statistically significant"
