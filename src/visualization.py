"""
Visualization Module
====================
Generate all figures for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class Visualizer:
    """Generate all visualizations"""
    
    def __init__(self, data: pd.DataFrame, config):
        self.data = data.copy()
        self.config = config
        
    def create_timeseries_plot(self, model_results: Dict) -> Path:
        """
        Figure 1: Time series of views and sales proxy by barrier category
        """
        logger.info("Creating Figure 1: Time series plots...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Aggregate by barrier category and quarter
        df_agg = self.data.groupby(['date', 'barrier_category']).agg({
            'log_views': 'mean',
            'log_sales_proxy': 'mean'
        }).reset_index()
        
        # Plot views
        for category in ['Low (1-3)', 'Medium (3-6)', 'High (6-10)']:
            cat_data = df_agg[df_agg['barrier_category'] == category]
            if len(cat_data) > 0:
                axes[0].plot(cat_data['date'], cat_data['log_views'],
                            label=category, marker='o', markersize=3)
        
        axes[0].set_xlabel('Quarter', fontweight='bold')
        axes[0].set_ylabel('Log(YouTube Views)', fontweight='bold')
        axes[0].set_title('Content Consumption Over Time by Barrier Category',
                         fontweight='bold', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot sales proxy
        for category in ['Low (1-3)', 'Medium (3-6)', 'High (6-10)']:
            cat_data = df_agg[df_agg['barrier_category'] == category]
            if len(cat_data) > 0:
                axes[1].plot(cat_data['date'], cat_data['log_sales_proxy'],
                            label=category, marker='o', markersize=3)
        
        axes[1].set_xlabel('Quarter', fontweight='bold')
        axes[1].set_ylabel('Log(Sales Proxy)', fontweight='bold')
        axes[1].set_title('Equipment Purchasing Intent Over Time by Barrier Category',
                         fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.FIGURES_DIR / 'figure1_timeseries.png'
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_scatter_plot(self, model_results: Dict) -> Path:
        """
        Figure 2: Scatter plot of views vs sales with fitted lines
        """
        logger.info("Creating Figure 2: Scatter plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter for each barrier category
        colors = {'Low (1-3)': 'green', 'Medium (3-6)': 'orange', 'High (6-10)': 'red'}
        
        for category in ['Low (1-3)', 'Medium (3-6)', 'High (6-10)']:
            cat_data = self.data[self.data['barrier_category'] == category]
            if len(cat_data) > 0:
                ax.scatter(cat_data['log_views'], cat_data['log_sales_proxy'],
                          alpha=0.5, s=50, c=colors[category], label=category)
                
                # Fit line
                if len(cat_data) > 2:
                    z = np.polyfit(cat_data['log_views'], cat_data['log_sales_proxy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(cat_data['log_views'].min(),
                                        cat_data['log_views'].max(), 100)
                    ax.plot(x_line, p(x_line), '--', color=colors[category],
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Log(YouTube Views)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Log(Sales Proxy)', fontweight='bold', fontsize=12)
        ax.set_title('Content Consumption vs. Equipment Purchasing by Barrier Level',
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.FIGURES_DIR / 'figure2_scatter.png'
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_marginal_effects_plot(self, model_results: Dict) -> Path:
        """
        Figure 3: Marginal effects of views on sales across barrier index
        """
        logger.info("Creating Figure 3: Marginal effects plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract coefficients
        beta1 = model_results['beta_views']
        beta3 = model_results['beta_interaction']
        se_interaction = model_results['se_interaction']
        
        # Create barrier range
        barrier_range = np.linspace(1, 9, 100)
        
        # Calculate marginal effect
        marginal_effect = beta1 + beta3 * barrier_range
        
        # Calculate confidence interval (approximate)
        # SE varies with barrier level, but this is approximate
        ci_upper = marginal_effect + 1.96 * se_interaction * barrier_range
        ci_lower = marginal_effect - 1.96 * se_interaction * barrier_range
        
        # Plot
        ax.plot(barrier_range, marginal_effect, linewidth=3, color='navy',
               label='Marginal Effect')
        ax.fill_between(barrier_range, ci_lower, ci_upper, alpha=0.2, color='navy',
                       label='95% CI')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label='Zero Effect')
        
        # Add vertical lines for barrier categories
        ax.axvline(x=3, color='green', linestyle=':', alpha=0.5, linewidth=2)
        ax.axvline(x=6, color='orange', linestyle=':', alpha=0.5, linewidth=2)
        
        # Add labels
        ax.text(1.5, ax.get_ylim()[1]*0.9, 'LOW\nBARRIER', ha='center',
               fontsize=10, style='italic', color='green')
        ax.text(4.5, ax.get_ylim()[1]*0.9, 'MEDIUM\nBARRIER', ha='center',
               fontsize=10, style='italic', color='orange')
        ax.text(7.5, ax.get_ylim()[1]*0.9, 'HIGH\nBARRIER', ha='center',
               fontsize=10, style='italic', color='red')
        
        ax.set_xlabel('Barrier-to-Entry Index (1-10)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Elasticity (∂log(Sales)/∂log(Views))', fontweight='bold', fontsize=12)
        ax.set_title('Marginal Effect of Content on Purchases Across Barrier Levels',
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.config.FIGURES_DIR / 'figure3_marginal_effects.png'
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_event_study_plot(self, model_results: Dict) -> Path:
        """
        Figure 4: Event study around documentary releases
        """
        logger.info("Creating Figure 4: Documentary event study...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Aggregate around documentary releases
        # For sports with documentaries, look at ±4 quarters around release
        doc_sports = self.data[self.data['documentary_release'] == 1]['sport'].unique()
        
        if len(doc_sports) > 0:
            # Create event time
            event_data = []
            for sport in doc_sports:
                sport_data = self.data[self.data['sport'] == sport].copy()
                sport_data = sport_data.sort_values('date')
                doc_quarters = sport_data[sport_data['documentary_release'] == 1].index
                
                if len(doc_quarters) > 0:
                    doc_q = doc_quarters[0]  # First documentary
                    sport_data['event_time'] = sport_data.index - doc_q
                    sport_data['event_time'] = sport_data['event_time'].apply(lambda x: x if -4 <= x <= 4 else np.nan)
                    event_data.append(sport_data[sport_data['event_time'].notna()])
            
            if len(event_data) > 0:
                event_df = pd.concat(event_data)
                
                # Aggregate
                agg_event = event_df.groupby('event_time').agg({
                    'log_views': ['mean', 'sem'],
                    'log_sales_proxy': ['mean', 'sem']
                }).reset_index()
                
                # Plot views
                ax.plot(agg_event['event_time'], agg_event['log_views']['mean'],
                       marker='o', linewidth=2, label='Views', color='blue')
                ax.fill_between(
                    agg_event['event_time'],
                    agg_event['log_views']['mean'] - 1.96*agg_event['log_views']['sem'],
                    agg_event['log_views']['mean'] + 1.96*agg_event['log_views']['sem'],
                    alpha=0.2, color='blue'
                )
                
                # Plot sales proxy (on secondary axis)
                ax2 = ax.twinx()
                ax2.plot(agg_event['event_time'], agg_event['log_sales_proxy']['mean'],
                        marker='s', linewidth=2, label='Sales Proxy', color='red')
                ax2.fill_between(
                    agg_event['event_time'],
                    agg_event['log_sales_proxy']['mean'] - 1.96*agg_event['log_sales_proxy']['sem'],
                    agg_event['log_sales_proxy']['mean'] + 1.96*agg_event['log_sales_proxy']['sem'],
                    alpha=0.2, color='red'
                )
                
                # Add vertical line at event
                ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7,
                          label='Documentary Release')
                
                ax.set_xlabel('Quarters Relative to Documentary Release', fontweight='bold', fontsize=12)
                ax.set_ylabel('Log(YouTube Views)', fontweight='bold', fontsize=12, color='blue')
                ax2.set_ylabel('Log(Sales Proxy)', fontweight='bold', fontsize=12, color='red')
                ax.set_title('Event Study: Documentary Releases as Natural Experiment',
                           fontweight='bold', fontsize=14)
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
                
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient documentary data for event study',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.config.FIGURES_DIR / 'figure4_event_study.png'
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_coefficient_comparison(self, heterogeneity_results: Dict) -> Path:
        """
        Figure 5: Coefficient comparison across barrier levels
        """
        logger.info("Creating Figure 5: Coefficient comparison...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract elasticities
        categories = []
        elasticities = []
        std_errors = []
        
        for cat in ['low', 'medium', 'high']:
            if cat in heterogeneity_results and 'elasticity' in heterogeneity_results[cat]:
                categories.append(cat.capitalize())
                elasticities.append(heterogeneity_results[cat]['elasticity'])
                std_errors.append(heterogeneity_results[cat].get('se', 0))
        
        if len(categories) > 0:
            x = np.arange(len(categories))
            colors = ['green', 'orange', 'red'][:len(categories)]
            
            # Create bars
            bars = ax.bar(x, elasticities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add error bars
            ax.errorbar(x, elasticities, yerr=[1.96*se for se in std_errors],
                       fmt='none', ecolor='black', capsize=5, linewidth=2)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, elasticities)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylabel('Elasticity (β₁)', fontweight='bold', fontsize=12)
            ax.set_xlabel('Barrier Category', fontweight='bold', fontsize=12)
            ax.set_title('Content → Purchase Elasticity by Barrier Level',
                        fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{cat} Barrier' for cat in categories])
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for coefficient comparison',
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.config.FIGURES_DIR / 'figure5_coefficient_comparison.png'
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        return output_path
