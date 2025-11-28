"""
EMPIRICAL ANALYSIS: From Viewers to Buyers
==========================================
Content Consumption and Equipment Purchasing in Outdoor Sports

Author: Arthus Azais de Vergeron
Course: EC 3310 Empirical Methods I
Institution: John Carroll University
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_processing import DataProcessor
from src.empirical_analysis import EmpiricalAnalyzer
from src.visualization import Visualizer
from src.results_writer import ResultsWriter
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_header():
    """Print analysis header"""
    header = """
╔════════════════════════════════════════════════════════════════════════════╗
║        EMPIRICAL ANALYSIS: From Viewers to Buyers                         ║
║        Content → Purchase in Outdoor Sports                                ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)
    logger.info("Starting empirical analysis pipeline")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main execution pipeline"""
    
    print_header()
    
    try:
        # ========================================================================
        # PHASE 1: DATA PROCESSING
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DATA PROCESSING & CLEANING")
        logger.info("="*80)
        
        processor = DataProcessor(Config)
        
        # Load and clean all data sources
        logger.info("\n[1/6] Loading YouTube data...")
        youtube_data = processor.process_youtube_data()
        logger.info(f"✓ Processed {len(youtube_data)} sport-quarter observations")
        
        logger.info("\n[2/6] Loading Google Trends data...")
        trends_data = processor.process_google_trends()
        logger.info(f"✓ Processed {len(trends_data)} sport-quarter observations")
        
        logger.info("\n[3/6] Loading Barrier Index...")
        barrier_data = processor.load_barrier_index()
        logger.info(f"✓ Loaded {len(barrier_data)} sports")
        
        logger.info("\n[4/6] Loading IMDB documentary data...")
        doc_data = processor.process_documentary_data()
        logger.info(f"✓ Processed {len(doc_data)} documentary releases")
        
        logger.info("\n[5/6] Loading FRED economic controls...")
        fred_data = processor.process_fred_data()
        logger.info(f"✓ Processed {len(fred_data)} quarters of economic data")
        
        logger.info("\n[6/6] Merging into master panel dataset...")
        master_data = processor.create_master_panel(
            youtube_data, trends_data, barrier_data, doc_data, fred_data
        )
        logger.info(f"✓ Created master panel: {len(master_data)} observations")
        logger.info(f"  - Sports: {master_data['sport'].nunique()}")
        logger.info(f"  - Quarters: {master_data['year'].nunique() * 4}")
        logger.info(f"  - Date range: {master_data['date'].min()} to {master_data['date'].max()}")
        
        # Save master dataset
        master_path = Config.PROCESSED_DATA_DIR / 'master_panel.csv'
        master_data.to_csv(master_path, index=False)
        logger.info(f"✓ Saved master panel to: {master_path}")
        
        # ========================================================================
        # PHASE 2: EMPIRICAL ANALYSIS
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: EMPIRICAL ANALYSIS")
        logger.info("="*80)
        
        analyzer = EmpiricalAnalyzer(master_data, Config)
        
        # Run all regression models
        logger.info("\n[1/6] Running Model 1: Pooled OLS...")
        model1_results = analyzer.run_model1_pooled_ols()
        logger.info(f"✓ Model 1 complete. R² = {model1_results['r_squared']:.3f}")
        
        logger.info("\n[2/6] Running Model 2: Fixed Effects...")
        model2_results = analyzer.run_model2_fixed_effects()
        logger.info(f"✓ Model 2 complete. Within R² = {model2_results['r_squared_within']:.3f}")
        
        logger.info("\n[3/6] Running Model 3: FE + Interaction (CORE MODEL)...")
        model3_results = analyzer.run_model3_interaction()
        logger.info(f"✓ Model 3 complete. β₃ (interaction) = {model3_results['beta_interaction']:.4f}")
        
        logger.info("\n[4/6] Running Model 4: IV/2SLS...")
        model4_results = analyzer.run_model4_iv()
        logger.info(f"✓ Model 4 complete. First-stage F = {model4_results['first_stage_f']:.2f}")
        
        logger.info("\n[5/6] Running heterogeneity analysis (split sample)...")
        heterogeneity_results = analyzer.run_heterogeneity_analysis()
        logger.info(f"✓ Split sample analysis complete")
        logger.info(f"  - Low barrier elasticity: {heterogeneity_results['low_elasticity']:.3f}")
        logger.info(f"  - Medium barrier elasticity: {heterogeneity_results['medium_elasticity']:.3f}")
        logger.info(f"  - High barrier elasticity: {heterogeneity_results['high_elasticity']:.3f}")
        
        logger.info("\n[6/6] Running robustness checks...")
        robustness_results = analyzer.run_robustness_checks()
        logger.info(f"✓ Robustness checks complete ({len(robustness_results)} specifications)")
        
        # ========================================================================
        # PHASE 3: VISUALIZATION
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: VISUALIZATION")
        logger.info("="*80)
        
        visualizer = Visualizer(master_data, Config)
        
        logger.info("\n[1/5] Creating Figure 1: Time series by barrier category...")
        fig1_path = visualizer.create_timeseries_plot(model2_results)
        logger.info(f"✓ Saved to: {fig1_path}")
        
        logger.info("\n[2/5] Creating Figure 2: Scatter views vs sales...")
        fig2_path = visualizer.create_scatter_plot(model2_results)
        logger.info(f"✓ Saved to: {fig2_path}")
        
        logger.info("\n[3/5] Creating Figure 3: Marginal effects plot...")
        fig3_path = visualizer.create_marginal_effects_plot(model3_results)
        logger.info(f"✓ Saved to: {fig3_path}")
        
        logger.info("\n[4/5] Creating Figure 4: Documentary event study...")
        fig4_path = visualizer.create_event_study_plot(model4_results)
        logger.info(f"✓ Saved to: {fig4_path}")
        
        logger.info("\n[5/5] Creating Figure 5: Coefficient comparison...")
        fig5_path = visualizer.create_coefficient_comparison(heterogeneity_results)
        logger.info(f"✓ Saved to: {fig5_path}")
        
        # ========================================================================
        # PHASE 4: RESULTS COMPILATION
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: RESULTS COMPILATION")
        logger.info("="*80)
        
        writer = ResultsWriter(Config)
        
        # Compile all results
        all_results = {
            'master_data': master_data,
            'model1': model1_results,
            'model2': model2_results,
            'model3': model3_results,
            'model4': model4_results,
            'heterogeneity': heterogeneity_results,
            'robustness': robustness_results
        }
        
        logger.info("\nGenerating results.md...")
        results_path = writer.write_results(all_results)
        logger.info(f"✓ Results saved to: {results_path}")
        
        # ========================================================================
        # COMPLETION
        # ========================================================================
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("📊 OUTPUTS GENERATED:")
        print("="*80)
        print(f"\n📁 Master Dataset:")
        print(f"   {master_path}")
        print(f"\n📊 Tables: {len(list(Config.TABLES_DIR.glob('*.csv')))} CSV files")
        print(f"   {Config.TABLES_DIR}")
        print(f"\n📈 Figures: {len(list(Config.FIGURES_DIR.glob('*.png')))} PNG files")
        print(f"   {Config.FIGURES_DIR}")
        print(f"\n📝 Results Report:")
        print(f"   {results_path}")
        print("\n" + "="*80)
        print("\n✅ All analyses completed successfully!")
        print("Ready for paper writing! 🚀\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {str(e)}", exc_info=True)
        print(f"\n❌ Analysis failed. Check analysis.log for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
