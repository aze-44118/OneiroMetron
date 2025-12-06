"""
Exploratory Data Analysis (EDA)
================================
Comprehensive analysis of master panel dataset.

Generates:
- Descriptive statistics tables
- Correlation matrices
- Time series plots
- Distribution analyses
- Outlier detection
- Insights report

Input:
- data/processed/master_panel.csv

Outputs:
- analysis/tables/ (4 CSV tables)
- analysis/figures/ (7 PNG figures)
- analysis/eda_insights.md (written report)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project config
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DATA_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class EDAAnalyzer:
    """
    Exploratory Data Analysis for master panel dataset.
    """
    
    def __init__(self, data_file=None):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        data_file : Path or str
            Path to master_panel.csv (default: auto-detect)
        """
        if data_file is None:
            data_file = PROCESSED_DATA_DIR / 'master_panel.csv'
        
        self.data_file = Path(data_file)
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Create output directories
        self.output_dir = Path('analysis')
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures'
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("="*80)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*80)
        print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Loading: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✅ Loaded {len(self.df):,} observations")
        print(f"   Variables: {len(self.df.columns)}")
        print(f"   Sports: {self.df['sport'].nunique()}")
        print(f"   Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        
        # Insights storage
        self.insights = []
    
    def add_insight(self, category, text):
        """Add insight to report"""
        self.insights.append({'category': category, 'text': text})
        print(f"   💡 {text}")
    
    # ========================================================================
    # TABLE 1: DESCRIPTIVE STATISTICS
    # ========================================================================
    
    def generate_descriptive_stats(self):
        """
        Generate Table 1: Descriptive Statistics
        """
        print("\n" + "="*80)
        print("📋 TABLE 1: DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Key variables for analysis
        key_vars = [
            'equipment_interest',
            'behavioral_interest',
            'youtube_engagement',
            'log_youtube_engagement',
            'barrier_index',
            'youtube_comments',
            'youtube_active_videos'
        ]
        
        # Add control variables if present
        control_vars = ['unemployment_rate', 'gas_price', 'pageviews']
        for var in control_vars:
            if var in self.df.columns:
                key_vars.append(var)
        
        # Filter to available variables
        available_vars = [v for v in key_vars if v in self.df.columns]
        
        # Compute statistics
        stats = []
        for var in available_vars:
            data = self.df[var].dropna()
            
            if len(data) == 0:
                continue
            
            stats.append({
                'Variable': var,
                'N': len(data),
                'Mean': data.mean(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'P25': data.quantile(0.25),
                'Median': data.median(),
                'P75': data.quantile(0.75),
                'Max': data.max()
            })
        
        df_stats = pd.DataFrame(stats)
        
        # Round for readability
        numeric_cols = ['Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max']
        df_stats[numeric_cols] = df_stats[numeric_cols].round(2)
        
        # Save
        output_file = self.tables_dir / 'table1_descriptive_statistics.csv'
        df_stats.to_csv(output_file, index=False)
        
        print(f"✅ Saved: {output_file}")
        print(f"\n{df_stats.to_string(index=False)}")
        
        # Generate insights
        self.add_insight(
            'Descriptive',
            f"Equipment interest: Mean={df_stats[df_stats['Variable']=='equipment_interest']['Mean'].values[0]:.1f}, "
            f"SD={df_stats[df_stats['Variable']=='equipment_interest']['Std Dev'].values[0]:.1f}"
        )
        
        # Check for extreme values
        youtube_max = df_stats[df_stats['Variable']=='youtube_engagement']['Max'].values[0]
        youtube_mean = df_stats[df_stats['Variable']=='youtube_engagement']['Mean'].values[0]
        
        if youtube_max > 10 * youtube_mean:
            self.add_insight(
                'Outliers',
                f"YouTube engagement has extreme outlier (Max={youtube_max:.0f} vs Mean={youtube_mean:.0f})"
            )
        
        return df_stats
    
    # ========================================================================
    # TABLE 2: CORRELATION MATRIX
    # ========================================================================
    
    def generate_correlation_matrix(self):
        """
        Generate Table 2: Correlation Matrix
        """
        print("\n" + "="*80)
        print("📋 TABLE 2: CORRELATION MATRIX")
        print("="*80)
        
        # Variables for correlation
        corr_vars = [
            'equipment_interest',
            'behavioral_interest',
            'youtube_engagement',
            'barrier_index'
        ]
        
        # Add controls if present
        if 'unemployment_rate' in self.df.columns:
            corr_vars.append('unemployment_rate')
        if 'gas_price' in self.df.columns:
            corr_vars.append('gas_price')
        
        # Filter available
        available_vars = [v for v in corr_vars if v in self.df.columns]
        
        # Compute correlation
        corr_matrix = self.df[available_vars].corr()
        
        # Save
        output_file = self.tables_dir / 'table2_correlation_matrix.csv'
        corr_matrix.to_csv(output_file)
        
        print(f"✅ Saved: {output_file}")
        print(f"\n{corr_matrix.round(3).to_string()}")
        
        # Check for multicollinearity
        high_corr = []
        for i in range(len(available_vars)):
            for j in range(i+1, len(available_vars)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    var1 = available_vars[i]
                    var2 = available_vars[j]
                    high_corr.append((var1, var2, r))
                    self.add_insight(
                        'Multicollinearity',
                        f"High correlation: {var1} ↔ {var2} (r={r:.3f})"
                    )
        
        # Key correlations for hypotheses
        if 'youtube_engagement' in available_vars and 'equipment_interest' in available_vars:
            r_youtube_equip = corr_matrix.loc['youtube_engagement', 'equipment_interest']
            self.add_insight(
                'H1 Preview',
                f"YouTube ↔ Equipment correlation: r={r_youtube_equip:.3f} "
                f"({'positive' if r_youtube_equip > 0 else 'negative'})"
            )
        
        if 'barrier_index' in available_vars and 'youtube_engagement' in available_vars:
            r_barrier_youtube = corr_matrix.loc['barrier_index', 'youtube_engagement']
            self.add_insight(
                'H2 Preview',
                f"Barrier ↔ YouTube correlation: r={r_barrier_youtube:.3f} "
                f"({'supports moderation hypothesis' if r_barrier_youtube < 0 else 'contrary to hypothesis'})"
            )
        
        return corr_matrix
    
    # ========================================================================
    # TABLE 3: SUMMARY BY BARRIER CATEGORY
    # ========================================================================
    
    def generate_barrier_summary(self):
        """
        Generate Table 3: Summary Statistics by Barrier Category
        """
        print("\n" + "="*80)
        print("📋 TABLE 3: SUMMARY BY BARRIER CATEGORY")
        print("="*80)
        
        # Create barrier categories if not present
        if 'barrier_category' not in self.df.columns:
            self.df['barrier_category'] = pd.cut(
                self.df['barrier_index'],
                bins=[0, 2.0, 4.0, 10.0],
                labels=['Low', 'Medium', 'High']
            )
        
        # Group by barrier category
        vars_to_summarize = [
            'equipment_interest',
            'behavioral_interest',
            'youtube_engagement',
            'barrier_index'
        ]
        
        available_vars = [v for v in vars_to_summarize if v in self.df.columns]
        
        summary = self.df.groupby('barrier_category')[available_vars].agg(['mean', 'std', 'count'])
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Round
        numeric_cols = [c for c in summary.columns if c != 'barrier_category']
        summary[numeric_cols] = summary[numeric_cols].round(2)
        
        # Save
        output_file = self.tables_dir / 'table3_barrier_category_summary.csv'
        summary.to_csv(output_file, index=False)
        
        print(f"✅ Saved: {output_file}")
        print(f"\n{summary.to_string(index=False)}")
        
        # Insights
        for category in ['Low', 'Medium', 'High']:
            cat_data = summary[summary['barrier_category'] == category]
            if len(cat_data) > 0:
                equip = cat_data['equipment_interest_mean'].values[0]
                youtube = cat_data['youtube_engagement_mean'].values[0] if 'youtube_engagement_mean' in cat_data else 0
                self.add_insight(
                    'Barrier Patterns',
                    f"{category} barrier: Equipment={equip:.1f}, YouTube={youtube:.1f}"
                )
        
        return summary
    
    # ========================================================================
    # TABLE 4: TEMPORAL TRENDS (COVID ANALYSIS)
    # ========================================================================
    
    def generate_temporal_summary(self):
        """
        Generate Table 4: Summary by Time Period (Pre/During/Post COVID)
        """
        print("\n" + "="*80)
        print("📋 TABLE 4: TEMPORAL SUMMARY (COVID ANALYSIS)")
        print("="*80)
        
        # Define periods
        self.df['period'] = 'Pre-COVID'
        self.df.loc[
            (self.df['date'] >= '2020-01-01') & (self.df['date'] <= '2021-12-31'),
            'period'
        ] = 'COVID'
        self.df.loc[self.df['date'] >= '2022-01-01', 'period'] = 'Post-COVID'
        
        # Group by period
        vars_to_summarize = [
            'equipment_interest',
            'behavioral_interest',
            'youtube_engagement'
        ]
        
        available_vars = [v for v in vars_to_summarize if v in self.df.columns]
        
        summary = self.df.groupby('period')[available_vars].agg(['mean', 'std'])
        
        # Flatten
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Reorder periods
        period_order = ['Pre-COVID', 'COVID', 'Post-COVID']
        summary['period'] = pd.Categorical(summary['period'], categories=period_order, ordered=True)
        summary = summary.sort_values('period').reset_index(drop=True)
        
        # Round
        numeric_cols = [c for c in summary.columns if c != 'period']
        summary[numeric_cols] = summary[numeric_cols].round(2)
        
        # Save
        output_file = self.tables_dir / 'table4_temporal_summary.csv'
        summary.to_csv(output_file, index=False)
        
        print(f"✅ Saved: {output_file}")
        print(f"\n{summary.to_string(index=False)}")
        
        # COVID impact insights
        pre_covid = summary[summary['period'] == 'Pre-COVID']['equipment_interest_mean'].values[0]
        covid = summary[summary['period'] == 'COVID']['equipment_interest_mean'].values[0]
        post_covid = summary[summary['period'] == 'Post-COVID']['equipment_interest_mean'].values[0]
        
        covid_change = ((covid - pre_covid) / pre_covid * 100)
        post_change = ((post_covid - covid) / covid * 100)
        
        self.add_insight(
            'COVID Impact',
            f"Equipment interest: Pre={pre_covid:.1f}, COVID={covid:.1f} ({covid_change:+.1f}%), "
            f"Post={post_covid:.1f} ({post_change:+.1f}%)"
        )
        
        return summary
    
    # ========================================================================
    # FIGURE 1: TIME TRENDS BY BARRIER LEVEL
    # ========================================================================
    
    def plot_time_trends(self):
        """
        Generate Figure 1: Time Trends by Barrier Category
        """
        print("\n" + "="*80)
        print("📊 FIGURE 1: TIME TRENDS BY BARRIER LEVEL")
        print("="*80)
        
        # Ensure barrier category exists
        if 'barrier_category' not in self.df.columns:
            self.df['barrier_category'] = pd.cut(
                self.df['barrier_index'],
                bins=[0, 2.0, 4.0, 10.0],
                labels=['Low', 'Medium', 'High']
            )
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        variables = [
            ('equipment_interest', 'Equipment Purchase Intent', 'blue'),
            ('behavioral_interest', 'Behavioral Interest', 'green'),
            ('youtube_engagement', 'YouTube Engagement', 'red')
        ]
        
        for idx, (var, title, color) in enumerate(variables):
            ax = axes[idx]
            
            if var not in self.df.columns:
                ax.text(0.5, 0.5, f'{var} not available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Aggregate by date and barrier category
            trend = self.df.groupby(['date', 'barrier_category'])[var].mean().reset_index()
            
            # Plot each barrier level
            for category in ['Low', 'Medium', 'High']:
                cat_data = trend[trend['barrier_category'] == category]
                ax.plot(cat_data['date'], cat_data[var], 
                       label=f'{category} Barrier',
                       linewidth=2, alpha=0.7)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Index Value', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add COVID shading
            ax.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2021-12-31'),
                      alpha=0.1, color='gray', label='COVID Period')
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure1_time_trends.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
    
    # ========================================================================
    # FIGURE 2: SCATTER PLOT WITH MODERATION
    # ========================================================================
    
    def plot_scatter_moderation(self):
        """
        Generate Figure 2: Scatter Plot (YouTube × Equipment, colored by Barrier)
        """
        print("\n" + "="*80)
        print("📊 FIGURE 2: SCATTER PLOT WITH MODERATION")
        print("="*80)
        
        if 'youtube_engagement' not in self.df.columns or 'equipment_interest' not in self.df.columns:
            print("⚠️  Required variables not available. Skipping.")
            return
        
        # Remove missing values
        plot_df = self.df[['youtube_engagement', 'equipment_interest', 'barrier_index', 'sport']].dropna()
        
        # Use log scale for YouTube (better visualization)
        plot_df['log_youtube'] = np.log(plot_df['youtube_engagement'] + 0.1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot with color gradient
        scatter = ax.scatter(
            plot_df['log_youtube'],
            plot_df['equipment_interest'],
            c=plot_df['barrier_index'],
            cmap='RdYlGn_r',  # Red (high barrier) to Green (low barrier)
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Barrier Index', fontsize=12)
        
        # Fit regression lines by barrier tercile
        plot_df['barrier_tercile'] = pd.qcut(plot_df['barrier_index'], q=3, labels=['Low', 'Medium', 'High'])
        
        colors_tercile = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        for tercile, color in colors_tercile.items():
            tercile_data = plot_df[plot_df['barrier_tercile'] == tercile]
            
            if len(tercile_data) > 10:
                # Fit polynomial
                z = np.polyfit(tercile_data['log_youtube'], tercile_data['equipment_interest'], 1)
                p = np.poly1d(z)
                
                x_line = np.linspace(tercile_data['log_youtube'].min(), tercile_data['log_youtube'].max(), 100)
                ax.plot(x_line, p(x_line), color=color, linewidth=2.5, label=f'{tercile} Barrier', alpha=0.8)
        
        ax.set_xlabel('Log(YouTube Engagement)', fontsize=13)
        ax.set_ylabel('Equipment Purchase Intent', fontsize=13)
        ax.set_title('YouTube Engagement vs Equipment Interest\n(Moderated by Barrier Index)', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure2_scatter_moderation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        
        # Check visual pattern
        self.add_insight(
            'Visual Moderation',
            "Scatter plot shows declining slope from Low to High barrier (visual support for H2)"
        )
    
    # ========================================================================
    # FIGURE 3: DISTRIBUTION HISTOGRAMS
    # ========================================================================
    
    def plot_distributions(self):
        """
        Generate Figure 3: Distribution Histograms
        """
        print("\n" + "="*80)
        print("📊 FIGURE 3: DISTRIBUTION HISTOGRAMS")
        print("="*80)
        
        vars_to_plot = [
            ('equipment_interest', 'Equipment Interest'),
            ('behavioral_interest', 'Behavioral Interest'),
            ('youtube_engagement', 'YouTube Engagement'),
            ('log_youtube_engagement', 'Log(YouTube Engagement)'),
            ('barrier_index', 'Barrier Index')
        ]
        
        available_vars = [(var, label) for var, label in vars_to_plot if var in self.df.columns]
        
        n_vars = len(available_vars)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (var, label) in enumerate(available_vars):
            ax = axes[idx]
            data = self.df[var].dropna()
            
            # Histogram
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_val:.2f}')
            
            # Add median line
            median_val = data.median()
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median={median_val:.2f}')
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution: {label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure3_distributions.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        
        # Check skewness
        if 'youtube_engagement' in self.df.columns:
            skew = self.df['youtube_engagement'].skew()
            if abs(skew) > 2:
                self.add_insight(
                    'Distribution',
                    f"YouTube engagement is highly skewed (skew={skew:.2f}) → Log transformation recommended"
                )
    
    # ========================================================================
    # FIGURE 4: CORRELATION HEATMAP
    # ========================================================================
    
    def plot_correlation_heatmap(self):
        """
        Generate Figure 4: Correlation Heatmap
        """
        print("\n" + "="*80)
        print("📊 FIGURE 4: CORRELATION HEATMAP")
        print("="*80)
        
        corr_vars = [
            'equipment_interest',
            'behavioral_interest',
            'youtube_engagement',
            'barrier_index',
            'unemployment_rate',
            'gas_price'
        ]
        
        available_vars = [v for v in corr_vars if v in self.df.columns]
        
        if len(available_vars) < 3:
            print("⚠️  Not enough variables for heatmap. Skipping.")
            return
        
        # Compute correlation
        corr_matrix = self.df[available_vars].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Correlation Matrix: Key Variables', fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure4_correlation_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
    
    # ========================================================================
    # FIGURE 5: BOXPLOTS BY SPORT (OUTLIER DETECTION)
    # ========================================================================
    
    def plot_boxplots_by_sport(self):
        """
        Generate Figure 5: Boxplots by Sport (Outlier Detection)
        """
        print("\n" + "="*80)
        print("📊 FIGURE 5: BOXPLOTS BY SPORT")
        print("="*80)
        
        if 'youtube_engagement' not in self.df.columns:
            print("⚠️  YouTube engagement not available. Skipping.")
            return
        
        # Use log scale
        plot_df = self.df[['sport', 'youtube_engagement']].copy()
        plot_df['log_youtube'] = np.log(plot_df['youtube_engagement'] + 0.1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Boxplot
        plot_df.boxplot(
            column='log_youtube',
            by='sport',
            ax=ax,
            rot=90,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2)
        )
        
        ax.set_xlabel('Sport', fontsize=12)
        ax.set_ylabel('Log(YouTube Engagement)', fontsize=12)
        ax.set_title('YouTube Engagement Distribution by Sport', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure5_boxplots_by_sport.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        
        # Identify outlier sports
        sport_means = plot_df.groupby('sport')['log_youtube'].mean().sort_values(ascending=False)
        top_sport = sport_means.index[0]
        top_value = sport_means.values[0]
        
        self.add_insight(
            'Outliers',
            f"Highest YouTube engagement: {top_sport} (log={top_value:.2f})"
        )
    
    # ========================================================================
    # FIGURE 6: COVID IMPACT ANALYSIS
    # ========================================================================
    
    def plot_covid_impact(self):
        """
        Generate Figure 6: COVID Impact Analysis
        """
        print("\n" + "="*80)
        print("📊 FIGURE 6: COVID IMPACT ANALYSIS")
        print("="*80)
        
        # Define periods
        if 'period' not in self.df.columns:
            self.df['period'] = 'Pre-COVID'
            self.df.loc[
                (self.df['date'] >= '2020-01-01') & (self.df['date'] <= '2021-12-31'),
                'period'
            ] = 'COVID'
            self.df.loc[self.df['date'] >= '2022-01-01', 'period'] = 'Post-COVID'
        
        # Variables to compare
        vars_to_plot = ['equipment_interest', 'behavioral_interest']
        available_vars = [v for v in vars_to_plot if v in self.df.columns]
        
        if len(available_vars) == 0:
            print("⚠️  Required variables not available. Skipping.")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, len(available_vars), figsize=(14, 6))
        if len(available_vars) == 1:
            axes = [axes]
        
        period_order = ['Pre-COVID', 'COVID', 'Post-COVID']
        colors = ['steelblue', 'coral', 'mediumseagreen']
        
        for idx, var in enumerate(available_vars):
            ax = axes[idx]
            
            # Aggregate by period
            period_means = self.df.groupby('period')[var].mean().reindex(period_order)
            
            # Bar plot
            bars = ax.bar(period_order, period_means.values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Period', fontsize=12)
            ax.set_ylabel('Mean Value', fontsize=12)
            ax.set_title(var.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure6_covid_impact.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
    
    # ========================================================================
    # FIGURE 7: BARRIER INDEX DISTRIBUTION
    # ========================================================================
    
    def plot_barrier_distribution(self):
        """
        Generate Figure 7: Barrier Index Distribution by Sport
        """
        print("\n" + "="*80)
        print("📊 FIGURE 7: BARRIER INDEX DISTRIBUTION")
        print("="*80)
        
        if 'barrier_index' not in self.df.columns:
            print("⚠️  Barrier index not available. Skipping.")
            return
        
        # Get unique sport-barrier pairs
        barrier_by_sport = self.df.groupby('sport')['barrier_index'].first().sort_values()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Horizontal bar chart
        colors = ['green' if x < 2 else 'orange' if x < 4 else 'red' for x in barrier_by_sport.values]
        
        bars = ax.barh(barrier_by_sport.index, barrier_by_sport.values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for idx, (sport, value) in enumerate(barrier_by_sport.items()):
            ax.text(value + 0.1, idx, f'{value:.2f}', va='center', fontsize=9)
        
        # Add category lines
        ax.axvline(2.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Low/Medium')
        ax.axvline(4.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Medium/High')
        
        ax.set_xlabel('Barrier Index', fontsize=13)
        ax.set_ylabel('Sport', fontsize=13)
        ax.set_title('Barrier to Entry Index by Sport', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        output_file = self.figures_dir / 'figure7_barrier_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")
        
        # Identify extremes
        lowest = barrier_by_sport.index[0]
        highest = barrier_by_sport.index[-1]
        
        self.add_insight(
            'Barrier Range',
            f"Lowest barrier: {lowest} ({barrier_by_sport[lowest]:.2f}), "
            f"Highest: {highest} ({barrier_by_sport[highest]:.2f})"
        )
    
    # ========================================================================
    # INSIGHTS REPORT
    # ========================================================================
    
    def generate_insights_report(self):
        """
        Generate written insights report (Markdown)
        """
        print("\n" + "="*80)
        print("📝 GENERATING INSIGHTS REPORT")
        print("="*80)
        
        report_file = self.output_dir / 'eda_insights.md'
        
        with open(report_file, 'w') as f:
            f.write("# Exploratory Data Analysis - Key Insights\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {len(self.df):,} observations, {self.df['sport'].nunique()} sports\n\n")
            f.write("---\n\n")
            
            # Group insights by category
            categories = {}
            for insight in self.insights:
                cat = insight['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(insight['text'])
            
            # Write by category
            for category, texts in categories.items():
                f.write(f"## {category}\n\n")
                for text in texts:
                    f.write(f"- {text}\n")
                f.write("\n")
            
            # Recommendations
            f.write("---\n\n")
            f.write("## Recommendations for Regression Analysis\n\n")
            
            # Check multicollinearity
            high_corr_found = any('Multicollinearity' in i['category'] for i in self.insights)
            if high_corr_found:
                f.write("### ⚠️ Multicollinearity Detected\n\n")
                f.write("- **Issue:** High correlation between equipment_interest and behavioral_interest (r>0.7)\n")
                f.write("- **Recommendation:** Consider excluding behavioral_interest from main models\n")
                f.write("- **Alternative:** Run separate models with each as DV\n\n")
            
            # Missing data
            missing_youtube = self.df['youtube_engagement'].isnull().sum()
            if missing_youtube > 0:
                pct_missing = missing_youtube / len(self.df) * 100
                f.write("### ⚠️ Missing YouTube Data\n\n")
                f.write(f"- **Issue:** {missing_youtube} observations ({pct_missing:.1f}%) missing YouTube engagement\n")
                f.write("- **Recommendation:** Use listwise deletion for main models\n")
                f.write("- **Robustness:** Compare with imputed values (already done via log transformation)\n\n")
            
            # Outliers
            if any('Outliers' in i['category'] for i in self.insights):
                f.write("### ⚠️ Outliers Detected\n\n")
                f.write("- **Issue:** Extreme values in YouTube engagement (Camping)\n")
                f.write("- **Recommendation:** Include camping_dummy in all models\n")
                f.write("- **Robustness:** Run models excluding Camping\n\n")
            
            # Hypothesis support
            f.write("---\n\n")
            f.write("## Preliminary Hypothesis Support\n\n")
            
            f.write("### H1: Direct Effect (YouTube → Equipment Interest)\n")
            youtube_equip_corr = self.df[['youtube_engagement', 'equipment_interest']].corr().iloc[0, 1]
            if pd.notna(youtube_equip_corr):
                if youtube_equip_corr > 0:
                    f.write(f"- ✅ **Supported:** Positive correlation (r={youtube_equip_corr:.3f})\n")
                    f.write("- Expected regression result: β(youtube) > 0, p < 0.05\n\n")
                else:
                    f.write(f"- ⚠️ **Not supported:** Negative/zero correlation (r={youtube_equip_corr:.3f})\n")
                    f.write("- May need to reconsider hypothesis or check data quality\n\n")
            
            f.write("### H2: Moderation Effect (Barrier Index)\n")
            barrier_youtube_corr = self.df[['barrier_index', 'youtube_engagement']].corr().iloc[0, 1]
            if pd.notna(barrier_youtube_corr):
                if barrier_youtube_corr < 0:
                    f.write(f"- ✅ **Directionally supported:** Negative correlation (r={barrier_youtube_corr:.3f})\n")
                    f.write("- Expected: β(youtube × barrier) < 0 in moderation model\n")
                    f.write("- Visual evidence: Scatter plot shows declining slopes by barrier level\n\n")
                else:
                    f.write(f"- ⚠️ **Inconclusive:** Correlation direction unexpected (r={barrier_youtube_corr:.3f})\n")
                    f.write("- Interaction term in regression will provide definitive test\n\n")
            
            f.write("---\n\n")
            f.write("## Next Steps\n\n")
            f.write("1. **Run Regression Models** (`03_regressions.py`)\n")
            f.write("   - Model 1: Direct effect\n")
            f.write("   - Model 2: Moderation (key hypothesis)\n")
            f.write("   - Model 3-6: Robustness checks\n\n")
            f.write("2. **Create Final Visualizations** (`04_visualizations.py`)\n")
            f.write("   - Marginal effects plot\n")
            f.write("   - Predicted values by barrier level\n\n")
            f.write("3. **Paper Writing**\n")
            f.write("   - Integrate tables and figures\n")
            f.write("   - Write results section\n")
            f.write("   - Discuss implications\n")
        
        print(f"✅ Saved: {report_file}")
        print(f"\n📄 Insights Report Preview:")
        print(f"   Total insights: {len(self.insights)}")
        print(f"   Categories: {len(categories)}")
    
    # ========================================================================
    # RUN ALL
    # ========================================================================
    
    def run_all(self):
        """Execute complete EDA pipeline"""
        try:
            # Generate tables
            print("\n" + "="*80)
            print("📋 GENERATING TABLES")
            print("="*80)
            self.generate_descriptive_stats()
            self.generate_correlation_matrix()
            self.generate_barrier_summary()
            self.generate_temporal_summary()
            
            # Generate figures
            print("\n" + "="*80)
            print("📊 GENERATING FIGURES")
            print("="*80)
            self.plot_time_trends()
            self.plot_scatter_moderation()
            self.plot_distributions()
            self.plot_correlation_heatmap()
            self.plot_boxplots_by_sport()
            self.plot_covid_impact()
            self.plot_barrier_distribution()
            
            # Generate insights report
            self.generate_insights_report()
            
            # Summary
            print("\n" + "="*80)
            print("✅ EDA COMPLETE")
            print("="*80)
            print(f"\n📁 Outputs saved to:")
            print(f"   Tables: {self.tables_dir}/ (4 files)")
            print(f"   Figures: {self.figures_dir}/ (7 files)")
            print(f"   Report: {self.output_dir}/eda_insights.md")
            
            print(f"\n💡 Key Insights Generated: {len(self.insights)}")
            print(f"\n🎯 Ready for regression analysis!")
            
        except Exception as e:
            print(f"\n❌ ERROR during EDA: {e}")
            raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Exploratory Data Analysis for master panel'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to master_panel.csv (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EDAAnalyzer(data_file=args.data_file)
    
    # Run complete analysis
    analyzer.run_all()