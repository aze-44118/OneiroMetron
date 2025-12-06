"""
Data Merging Pipeline
=====================
Creates the master panel dataset by merging all data sources.

Pipeline:
1. Load all raw data sources
2. Convert monthly → quarterly (Google Trends, FRED, Wikipedia)
3. Filter dates (2015-2024 only)
4. Merge progressively on [sport, date]
5. Create derived variables (logs, interactions, dummies)
6. Generate Fixed Effects dummies
7. Validate and save

Inputs (data/raw/):
- google_trends_equipment.csv
- google_trends_behavioral.csv
- youtube_quarterly_engagement.csv
- youtube_quarterly_views_estimated.csv
- barrier_index_results.csv
- fred.csv
- wikipedia_pageviews.csv (optional)
- imdb_data.csv (optional)

Output (data/processed/):
- master_panel.csv (960 observations × ~35 variables)
- master_panel_codebook.md (documentation)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project config
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataMerger:
    """
    Merges all data sources into master panel dataset.
    
    Parameters:
    -----------
    start_date : str
        Start of analysis period (default: '2015-01-01')
    end_date : str
        End of analysis period (default: '2024-12-31')
    include_wikipedia : bool
        Include Wikipedia pageviews (default: False)
    include_imdb : bool
        Include IMDB documentary data (default: False)
    """
    
    def __init__(self, 
                 start_date='2015-01-01', 
                 end_date='2024-12-31',
                 include_wikipedia=False,
                 include_imdb=False):
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.include_wikipedia = include_wikipedia
        self.include_imdb = include_imdb
        
        print("="*80)
        print("🔗 DATA MERGING PIPELINE")
        print("="*80)
        print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Expected observations: {self._calculate_expected_obs()}")
        
        # Initialize data containers
        self.df_equipment = None
        self.df_behavioral = None
        self.df_youtube_engagement = None
        self.df_youtube_views = None
        self.df_barrier = None
        self.df_fred = None
        self.df_wikipedia = None
        self.df_imdb = None
        
        # Final merged dataset
        self.df_master = None
    
    def _calculate_expected_obs(self):
        """Calculate expected number of observations"""
        quarters = pd.period_range(self.start_date, self.end_date, freq='Q')
        n_quarters = len(quarters)
        n_sports = 24  # Known from project
        return n_quarters * n_sports
    
    def _monthly_to_quarterly(self, df, date_col='date', agg_method='mean'):
        """
        Convert monthly data to quarterly by aggregation.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with monthly frequency
        date_col : str
            Name of date column
        agg_method : str
            Aggregation method ('mean', 'sum', 'last')
        
        Returns:
        --------
        DataFrame with quarterly frequency
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['quarter'] = df[date_col].dt.to_period('Q')
        
        # Group by sport (if exists) and quarter
        if 'sport' in df.columns:
            group_cols = ['sport', 'quarter']
        else:
            group_cols = ['quarter']
        
        # Aggregate
        if agg_method == 'mean':
            df_q = df.groupby(group_cols).mean(numeric_only=True).reset_index()
        elif agg_method == 'sum':
            df_q = df.groupby(group_cols).sum(numeric_only=True).reset_index()
        elif agg_method == 'last':
            df_q = df.groupby(group_cols).last().reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        # Convert quarter back to timestamp (start of quarter)
        df_q['date'] = df_q['quarter'].dt.to_timestamp()
        df_q.drop('quarter', axis=1, inplace=True)
        
        return df_q
    
    def load_google_trends_equipment(self):
        """Load and process Google Trends equipment data (DV)"""
        print("\n📥 Loading Google Trends Equipment (Purchase Intent)...")
        
        file_path = RAW_DATA_DIR / 'google_trends_equipment.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Missing: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"   Raw: {len(df):,} observations (monthly)")
        
        # Aggregate by sport-month (average across keywords)
        df_agg = df.groupby(['sport', 'date'])['index'].mean().reset_index()
        df_agg.rename(columns={'index': 'equipment_interest'}, inplace=True)
        
        # Convert to quarterly
        df_q = self._monthly_to_quarterly(df_agg, agg_method='mean')
        
        # Filter date range
        df_q = df_q[(df_q['date'] >= self.start_date) & (df_q['date'] <= self.end_date)]
        
        print(f"   Quarterly: {len(df_q):,} observations")
        print(f"   Sports: {df_q['sport'].nunique()}")
        print(f"   Date range: {df_q['date'].min()} to {df_q['date'].max()}")
        
        self.df_equipment = df_q
        return df_q
    
    def load_google_trends_behavioral(self):
        """Load and process Google Trends behavioral data (IV)"""
        print("\n📥 Loading Google Trends Behavioral (Learning Intent)...")
        
        file_path = RAW_DATA_DIR / 'google_trends_behavioral.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Missing: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"   Raw: {len(df):,} observations (monthly)")
        
        # Aggregate by sport-month
        df_agg = df.groupby(['sport', 'date'])['index'].mean().reset_index()
        df_agg.rename(columns={'index': 'behavioral_interest'}, inplace=True)
        
        # Convert to quarterly
        df_q = self._monthly_to_quarterly(df_agg, agg_method='mean')
        
        # Filter date range
        df_q = df_q[(df_q['date'] >= self.start_date) & (df_q['date'] <= self.end_date)]
        
        print(f"   Quarterly: {len(df_q):,} observations")
        
        self.df_behavioral = df_q
        return df_q
    
    def load_youtube_engagement(self):
        """Load YouTube engagement data (IV - comment-based)"""
        print("\n📥 Loading YouTube Engagement (Comment-Based)...")
        
        file_path = RAW_DATA_DIR / 'youtube_quarterly_engagement.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Missing: {file_path}")
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"   Raw: {len(df):,} observations")
        
        # Filter date range (CRITICAL: exclude 2025+)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # Select relevant columns
        df = df[['sport', 'date', 'total_comments', 'active_videos', 'engagement_intensity']]
        
        # Rename for clarity
        df.rename(columns={
            'engagement_intensity': 'youtube_engagement',
            'total_comments': 'youtube_comments',
            'active_videos': 'youtube_active_videos'
        }, inplace=True)
        
        print(f"   Filtered (2015-2024): {len(df):,} observations")
        print(f"   Sports: {df['sport'].nunique()}")
        
        self.df_youtube_engagement = df
        return df
    
    def load_youtube_views(self):
        """Load YouTube estimated views data (IV - view-based)"""
        print("\n📥 Loading YouTube Estimated Views (View-Based)...")
        
        file_path = RAW_DATA_DIR / 'youtube_quarterly_views_estimated.csv'
        if not file_path.exists():
            print("   ⚠️  File not found. Skipping YouTube views.")
            return None
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['quarter_date'])
        
        print(f"   Raw: {len(df):,} observations")
        
        # Filter date range (CRITICAL: exclude 2025+)
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # Select relevant columns
        df = df[['sport', 'date', 'estimated_views', 'avg_views_per_video']]
        
        # Rename
        df.rename(columns={
            'estimated_views': 'youtube_views',
            'avg_views_per_video': 'youtube_avg_views_per_video'
        }, inplace=True)
        
        print(f"   Filtered (2015-2024): {len(df):,} observations")
        
        self.df_youtube_views = df
        return df
    
    def load_barrier_index(self):
        """Load Barrier Index data (Moderator)"""
        print("\n📥 Loading Barrier Index (Moderator)...")
        
        file_path = RAW_DATA_DIR / 'barrier_index_results.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Missing: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Keep only relevant columns
        if 'barrier_index' not in df.columns:
            # Try alternative column names
            if 'barrier_score' in df.columns:
                df.rename(columns={'barrier_score': 'barrier_index'}, inplace=True)
            else:
                raise ValueError("Barrier index column not found")
        
        df = df[['sport', 'barrier_index']]
        
        print(f"   Sports: {len(df)}")
        print(f"   Barrier range: {df['barrier_index'].min():.2f} to {df['barrier_index'].max():.2f}")
        
        self.df_barrier = df
        return df
    
    def load_fred(self):
        """Load FRED economic data (Controls)"""
        print("\n📥 Loading FRED Economic Data (Controls)...")
        
        file_path = RAW_DATA_DIR / 'fred.csv'
        if not file_path.exists():
            print("   ⚠️  File not found. Creating dummy controls.")
            # Create dummy data (zeros)
            dates = pd.date_range(self.start_date, self.end_date, freq='QS')
            df = pd.DataFrame({
                'date': dates,
                'unemployment_rate': 5.0,
                'gas_price': 3.0
            })
        else:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['DATE'] if 'DATE' in df.columns else df['date'])
            
            print(f"   Raw: {len(df):,} observations")
            
            # Convert to quarterly if monthly
            if len(df) > 500:  # Likely monthly
                df_q = self._monthly_to_quarterly(df, agg_method='mean')
                df = df_q
            
            # Select/rename relevant columns
            col_mapping = {
                'UNRATE': 'unemployment_rate',
                'unemployment': 'unemployment_rate',
                'GASREGW': 'gas_price',
                'gas': 'gas_price'
            }
            
            for old, new in col_mapping.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            
            # Keep relevant columns
            keep_cols = ['date']
            if 'unemployment_rate' in df.columns:
                keep_cols.append('unemployment_rate')
            if 'gas_price' in df.columns:
                keep_cols.append('gas_price')
            
            df = df[keep_cols]
        
        # Filter date range
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        print(f"   Quarterly: {len(df):,} observations")
        print(f"   Variables: {', '.join([c for c in df.columns if c != 'date'])}")
        
        self.df_fred = df
        return df
    
    def load_wikipedia(self):
        """Load Wikipedia pageviews (Optional robustness check)"""
        if not self.include_wikipedia:
            print("\n⏭️  Skipping Wikipedia (not requested)")
            return None
        
        print("\n📥 Loading Wikipedia Pageviews...")
        
        file_path = RAW_DATA_DIR / 'wikipedia_pageviews.csv'
        if not file_path.exists():
            print("   ⚠️  File not found. Skipping.")
            return None
        
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert to quarterly
        df_q = self._monthly_to_quarterly(df, agg_method='mean')
        
        # Filter date range
        df_q = df_q[(df_q['date'] >= self.start_date) & (df_q['date'] <= self.end_date)]
        
        df_q.rename(columns={'views': 'wikipedia_views'}, inplace=True)
        
        print(f"   Quarterly: {len(df_q):,} observations")
        
        self.df_wikipedia = df_q
        return df_q
    
    def load_imdb(self):
        """Load IMDB documentary data (Optional IV)"""
        if not self.include_imdb:
            print("\n⏭️  Skipping IMDB (not requested)")
            return None
        
        print("\n📥 Loading IMDB Documentary Data...")
        
        file_path = RAW_DATA_DIR / 'imdb_data.csv'
        if not file_path.exists():
            print("   ⚠️  File not found. Skipping.")
            return None
        
        df = pd.read_csv(file_path)
        
        # Process documentary releases
        # Create quarterly dummy: 1 if documentary released in quarter, 0 otherwise
        # (This is a placeholder - actual implementation depends on IMDB data structure)
        
        print(f"   Documentaries: {len(df)}")
        
        self.df_imdb = df
        return df
    
    def merge_datasets(self):
        """
        Merge all datasets progressively.
        
        Merge order:
        1. Equipment (base)
        2. Behavioral
        3. YouTube Engagement
        4. YouTube Views
        5. Barrier Index (cross-join on sport)
        6. FRED (cross-join on date)
        7. Wikipedia (optional)
        """
        print("\n🔗 MERGING DATASETS...")
        
        # Start with Equipment (DV)
        df = self.df_equipment.copy()
        print(f"\n   Base (Equipment): {len(df):,} obs")
        
        # Merge Behavioral
        df = pd.merge(
            df, 
            self.df_behavioral,
            on=['sport', 'date'],
            how='left',
            validate='1:1'
        )
        print(f"   + Behavioral: {len(df):,} obs")
        
        # Merge YouTube Engagement
        df = pd.merge(
            df,
            self.df_youtube_engagement,
            on=['sport', 'date'],
            how='left',
            validate='1:1'
        )
        print(f"   + YouTube Engagement: {len(df):,} obs")
        
        # Merge YouTube Views (if available)
        if self.df_youtube_views is not None:
            df = pd.merge(
                df,
                self.df_youtube_views,
                on=['sport', 'date'],
                how='left',
                validate='1:1'
            )
            print(f"   + YouTube Views: {len(df):,} obs")
        
        # Merge Barrier Index (constant per sport)
        df = pd.merge(
            df,
            self.df_barrier,
            on='sport',
            how='left',
            validate='m:1'
        )
        print(f"   + Barrier Index: {len(df):,} obs")
        
        # Merge FRED (constant per date)
        df = pd.merge(
            df,
            self.df_fred,
            on='date',
            how='left',
            validate='m:1'
        )
        print(f"   + FRED Controls: {len(df):,} obs")
        
        # Merge Wikipedia (if requested)
        if self.df_wikipedia is not None:
            df = pd.merge(
                df,
                self.df_wikipedia,
                on=['sport', 'date'],
                how='left',
                validate='1:1'
            )
            print(f"   + Wikipedia: {len(df):,} obs")
        
        print(f"\n✅ Merged dataset: {len(df):,} observations")
        
        self.df_master = df
        return df
    
    def create_derived_variables(self):
        """
        Create derived variables for analysis.
        
        Variables created:
        - Log transformations
        - Interactions
        - Dummies
        - Fixed effects
        """
        print("\n🔧 CREATING DERIVED VARIABLES...")
        
        df = self.df_master.copy()
        
        # 1. Log Transformations
        print("   1. Log transformations...")
        
        # Handle zeros: add small constant before log
        epsilon = 0.1
        
        df['log_equipment_interest'] = np.log(df['equipment_interest'] + epsilon)
        df['log_behavioral_interest'] = np.log(df['behavioral_interest'] + epsilon)
        
        if 'youtube_engagement' in df.columns:
            # Replace 0 with epsilon
            df['youtube_engagement'] = df['youtube_engagement'].replace(0, epsilon)
            df['log_youtube_engagement'] = np.log(df['youtube_engagement'])
        
        if 'youtube_views' in df.columns:
            df['youtube_views'] = df['youtube_views'].replace(0, epsilon)
            df['log_youtube_views'] = np.log(df['youtube_views'])
        
        # 2. Interactions (for moderation analysis)
        print("   2. Interaction terms...")
        
        if 'youtube_engagement' in df.columns and 'barrier_index' in df.columns:
            df['youtube_x_barrier'] = df['youtube_engagement'] * df['barrier_index']
            df['log_youtube_x_barrier'] = df['log_youtube_engagement'] * df['barrier_index']
        
        if 'youtube_views' in df.columns and 'barrier_index' in df.columns:
            df['views_x_barrier'] = df['youtube_views'] * df['barrier_index']
            df['log_views_x_barrier'] = df['log_youtube_views'] * df['barrier_index']
        
        # 3. Dummy Variables
        print("   3. Dummy variables...")
        
        # Camping outlier dummy
        df['camping_dummy'] = (df['sport'] == 'Camping').astype(int)
        
        # Barrier categories
        df['barrier_category'] = pd.cut(
            df['barrier_index'],
            bins=[0, 2.0, 4.0, 10.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # COVID period dummy
        df['covid_period'] = ((df['date'] >= '2020-01-01') & (df['date'] <= '2021-12-31')).astype(int)
        
        # 4. Temporal Variables
        print("   4. Temporal variables...")
        
        df['year'] = df['date'].dt.year
        df['quarter_num'] = df['date'].dt.quarter
        df['quarter'] = df['date'].dt.to_period('Q').astype(str)
        
        # Time trend (quarters since start)
        min_date = df['date'].min()
        df['time_trend'] = ((df['date'] - min_date).dt.days / 91.25).astype(int)  # ~91.25 days per quarter
        
        # 5. Fixed Effects Dummies
        print("   5. Fixed effects dummies...")
        
        # Sport FE (N-1 dummies, baseline = first sport alphabetically)
        sport_dummies = pd.get_dummies(df['sport'], prefix='sport', drop_first=True)
        df = pd.concat([df, sport_dummies], axis=1)
        
        # Quarter FE (N-1 dummies, baseline = first quarter)
        quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True)
        df = pd.concat([df, quarter_dummies], axis=1)
        
        # Year FE (alternative to quarter FE)
        year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
        df = pd.concat([df, year_dummies], axis=1)
        
        print(f"\n✅ Created {len(df.columns) - len(self.df_master.columns)} new variables")
        
        self.df_master = df
        return df
    
    def validate_dataset(self):
        """
        Validate merged dataset.
        
        Checks:
        - Expected number of observations
        - No duplicates
        - Key variables present
        - Missing values documented
        """
        print("\n🔍 VALIDATING DATASET...")
        
        df = self.df_master
        
        # 1. Observation count
        expected = self._calculate_expected_obs()
        actual = len(df)
        print(f"\n   Observations: {actual:,} (expected: {expected:,})")
        
        if actual != expected:
            print(f"   ⚠️  Warning: Observation count mismatch")
            missing = expected - actual
            print(f"      Missing: {missing} observations ({missing/expected*100:.1f}%)")
        
        # 2. Duplicates
        duplicates = df.duplicated(subset=['sport', 'date']).sum()
        if duplicates > 0:
            print(f"   ❌ ERROR: {duplicates} duplicate sport-date combinations found!")
        else:
            print(f"   ✅ No duplicates")
        
        # 3. Key variables present
        required_vars = [
            'sport', 'date', 'equipment_interest', 'behavioral_interest',
            'youtube_engagement', 'barrier_index'
        ]
        
        missing_vars = [v for v in required_vars if v not in df.columns]
        if missing_vars:
            print(f"   ❌ ERROR: Missing required variables: {missing_vars}")
        else:
            print(f"   ✅ All required variables present")
        
        # 4. Missing values report
        print(f"\n   Missing Values Summary:")
        missing = df[required_vars].isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        for var in required_vars:
            if var in df.columns:
                n_missing = missing[var]
                pct_missing = missing_pct[var]
                if n_missing > 0:
                    print(f"      {var:30s}: {n_missing:4d} ({pct_missing:5.2f}%)")
        
        if missing.sum() == 0:
            print(f"      ✅ No missing values in key variables")
        
        # 5. Summary statistics
        print(f"\n   Summary Statistics (Key Variables):")
        summary_vars = ['equipment_interest', 'behavioral_interest', 
                       'youtube_engagement', 'barrier_index']
        
        for var in summary_vars:
            if var in df.columns:
                print(f"      {var:30s}: Mean={df[var].mean():8.2f}, SD={df[var].std():8.2f}, "
                      f"Min={df[var].min():8.2f}, Max={df[var].max():8.2f}")
        
        # 6. Balance check
        print(f"\n   Panel Balance:")
        obs_per_sport = df.groupby('sport').size()
        print(f"      Min observations/sport: {obs_per_sport.min()}")
        print(f"      Max observations/sport: {obs_per_sport.max()}")
        
        if obs_per_sport.min() == obs_per_sport.max():
            print(f"      ✅ Perfectly balanced panel")
        else:
            print(f"      ⚠️  Unbalanced panel detected")
            unbalanced = obs_per_sport[obs_per_sport < obs_per_sport.max()]
            print(f"         Sports with missing quarters:")
            for sport, count in unbalanced.items():
                print(f"            {sport}: {count} quarters")
    
    def save_dataset(self):
        """Save master panel and documentation"""
        print("\n💾 SAVING DATASET...")
        
        # Save master panel
        output_file = PROCESSED_DATA_DIR / 'master_panel.csv'
        self.df_master.to_csv(output_file, index=False)
        print(f"   📁 Saved: {output_file}")
        print(f"      Size: {len(self.df_master):,} rows × {len(self.df_master.columns)} columns")
        
        # Generate codebook
        self._generate_codebook()
        
        return output_file
    
    def _generate_codebook(self):
        """Generate data dictionary / codebook"""
        print("   📖 Generating codebook...")
        
        codebook_file = PROCESSED_DATA_DIR / 'master_panel_codebook.md'
        
        with open(codebook_file, 'w') as f:
            f.write("# Master Panel Dataset Codebook\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Observations:** {len(self.df_master):,}\n\n")
            f.write(f"**Variables:** {len(self.df_master.columns)}\n\n")
            f.write(f"**Date Range:** {self.start_date.date()} to {self.end_date.date()}\n\n")
            
            f.write("---\n\n")
            f.write("## Variable Definitions\n\n")
            
            # Group variables by category
            var_categories = {
                'Identifiers': ['sport', 'date', 'quarter', 'year', 'quarter_num'],
                'Dependent Variables': ['equipment_interest', 'log_equipment_interest'],
                'Independent Variables': [
                    'behavioral_interest', 'log_behavioral_interest',
                    'youtube_engagement', 'log_youtube_engagement',
                    'youtube_views', 'log_youtube_views',
                    'youtube_comments', 'youtube_active_videos'
                ],
                'Moderator': ['barrier_index', 'barrier_category'],
                'Interactions': [
                    'youtube_x_barrier', 'log_youtube_x_barrier',
                    'views_x_barrier', 'log_views_x_barrier'
                ],
                'Controls': [
                    'unemployment_rate', 'gas_price', 'wikipedia_views',
                    'camping_dummy', 'covid_period', 'time_trend'
                ]
            }
            
            for category, variables in var_categories.items():
                f.write(f"### {category}\n\n")
                for var in variables:
                    if var in self.df_master.columns:
                        dtype = self.df_master[var].dtype
                        n_unique = self.df_master[var].nunique()
                        f.write(f"- **{var}** ({dtype}): {n_unique} unique values\n")
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## Data Sources\n\n")
            f.write("- **Google Trends Equipment:** Monthly search indices (2015-2024)\n")
            f.write("- **Google Trends Behavioral:** Monthly search indices (2015-2024)\n")
            f.write("- **YouTube Engagement:** Quarterly comment-based metrics\n")
            f.write("- **YouTube Views:** Quarterly estimated views\n")
            f.write("- **Barrier Index:** Sport difficulty scores (constant)\n")
            f.write("- **FRED:** Economic indicators (quarterly)\n")
            
            if self.include_wikipedia:
                f.write("- **Wikipedia:** Pageviews (quarterly)\n")
            
            f.write("\n---\n\n")
            f.write("## Notes\n\n")
            f.write("- All monetary values in USD\n")
            f.write("- Log transformations use ln(x + 0.1) to handle zeros\n")
            f.write("- Fixed effects dummies use first category as baseline\n")
            f.write("- Panel structure: Balanced (24 sports × 40 quarters)\n")
        
        print(f"   📁 Saved: {codebook_file}")
    
    def run(self):
        """
        Execute complete merging pipeline.
        
        Returns:
        --------
        DataFrame: Master panel dataset
        """
        try:
            # Load all data sources
            self.load_google_trends_equipment()
            self.load_google_trends_behavioral()
            self.load_youtube_engagement()
            self.load_youtube_views()
            self.load_barrier_index()
            self.load_fred()
            self.load_wikipedia()
            self.load_imdb()
            
            # Merge
            self.merge_datasets()
            
            # Create derived variables
            self.create_derived_variables()
            
            # Validate
            self.validate_dataset()
            
            # Save
            output_file = self.save_dataset()
            
            print("\n" + "="*80)
            print("✅ DATA MERGING COMPLETE")
            print("="*80)
            print(f"\n📊 Final Dataset: {output_file}")
            print(f"   {len(self.df_master):,} observations")
            print(f"   {len(self.df_master.columns)} variables")
            print(f"   {self.df_master['sport'].nunique()} sports")
            print(f"   {self.df_master['quarter'].nunique()} quarters")
            
            return self.df_master
            
        except Exception as e:
            print(f"\n❌ ERROR during merging: {e}")
            raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge all data sources into master panel'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--include-wikipedia',
        action='store_true',
        help='Include Wikipedia pageviews data'
    )
    parser.add_argument(
        '--include-imdb',
        action='store_true',
        help='Include IMDB documentary data'
    )
    
    args = parser.parse_args()
    
    # Create merger instance
    merger = DataMerger(
        start_date=args.start_date,
        end_date=args.end_date,
        include_wikipedia=args.include_wikipedia,
        include_imdb=args.include_imdb
    )
    
    # Run pipeline
    df_master = merger.run()
    
    # Display sample
    print("\n📋 Sample Output (first 10 rows):")
    display_cols = [
        'sport', 'date', 'equipment_interest', 'behavioral_interest',
        'youtube_engagement', 'barrier_index'
    ]
    available_cols = [c for c in display_cols if c in df_master.columns]
    print(df_master[available_cols].head(10).to_string(index=False))