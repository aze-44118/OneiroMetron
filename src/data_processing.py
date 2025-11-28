"""
Data Processing Module
======================
Handles loading, cleaning, and merging all data sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean all data sources"""
    
    def __init__(self, config):
        self.config = config
        
    def process_youtube_data(self) -> pd.DataFrame:
        """
        Process YouTube channel data into sport-quarter totals
        
        Returns:
            DataFrame with columns: sport, year, quarter, total_views, avg_subscribers
        """
        df = pd.read_csv(self.config.YOUTUBE_FILE)
        
        # Standardize sport names
        df['sport'] = df['sport'].str.lower().str.strip()
        df['sport'] = df['sport'].replace(self.config.SPORT_NAME_MAPPING)
        
        # Ensure data types are consistent
        df['year'] = df['year'].astype(int)
        
        # Handle quarter: might be "Q1", "Q2" or already integers
        if df['quarter'].dtype == 'object':
            # Extract number from "Q1" format
            df['quarter'] = df['quarter'].str.replace('Q', '').astype(int)
        else:
            df['quarter'] = df['quarter'].astype(int)
        
        # Aggregate to sport-quarter level
        agg_df = df.groupby(['sport', 'year', 'quarter']).agg({
            'total_views_in_quarter': 'sum',
            'subscribers_end_quarter': 'mean',
            'videos_uploaded_in_quarter': 'sum'
        }).reset_index()
        
        agg_df.columns = ['sport', 'year', 'quarter', 'total_views', 
                          'avg_subscribers', 'videos_uploaded']
        
        # Create log transformations (add 1 to handle zeros)
        agg_df['log_views'] = np.log(agg_df['total_views'] + 1)
        
        return agg_df
    
    def process_google_trends(self) -> pd.DataFrame:
        """
        Process Google Trends search data into sport-quarter sales proxy
        
        Returns:
            DataFrame with columns: sport, year, quarter, search_index, log_sales_proxy
        """
        df = pd.read_csv(self.config.GOOGLE_TRENDS_FILE)
        
        # Map search terms to sports
        df['sport'] = df['search_term'].map(self.config.SEARCH_TERM_MAPPING)
        
        # Drop unmapped terms
        df = df[df['sport'].notna()].copy()
        
        # Convert month to quarter
        df['month'] = pd.to_datetime(df['month'])
        df['year'] = df['month'].dt.year.astype(int)
        df['quarter'] = df['month'].dt.quarter.astype(int)
        
        # Filter date range
        df = df[(df['year'] >= self.config.START_YEAR) & 
                (df['year'] <= self.config.END_YEAR)].copy()
        
        # Average search index per sport per quarter (across multiple search terms)
        agg_df = df.groupby(['sport', 'year', 'quarter']).agg({
            'search_index': 'mean'
        }).reset_index()
        
        # Create log transformation
        agg_df['log_sales_proxy'] = np.log(agg_df['search_index'] + 1)
        
        return agg_df
    
    def load_barrier_index(self) -> pd.DataFrame:
        """
        Load barrier-to-entry index
        
        Returns:
            DataFrame with columns: sport, barrier_index, barrier_category
        """
        df = pd.read_csv(self.config.BARRIER_INDEX_FILE)
        
        # Standardize sport names - apply mapping to handle Title Case
        df['sport'] = df['sport'].replace(self.config.SPORT_NAME_MAPPING)
        # Then lowercase everything
        df['sport'] = df['sport'].str.lower().str.strip()
        
        # Keep only necessary columns
        df = df[['sport', 'barrier_index', 'barrier_category']].copy()
        
        return df
    
    def process_documentary_data(self) -> pd.DataFrame:
        """
        Process IMDB documentary data into release indicators
        
        Returns:
            DataFrame with columns: sport, year, quarter, documentary_release
        """
        df = pd.read_csv(self.config.IMDB_FILE)
        
        # Standardize sport names
        df['sport'] = df['sport'].str.lower().str.strip()
        df['sport'] = df['sport'].replace({
            'climbing': 'rock climbing',
            'mountaineering': 'rock climbing',
            'skiing': 'snowboarding',  # Close enough for aggregation
            'skiing-snowboarding': 'snowboarding',
            'ultrarunning': 'trail running',
            'mountain biking': 'mountain biking',
            'BASE jumping': 'base jumping',
            'adventure travel': None  # Drop generic
        })
        
        # Drop NA sports
        df = df[df['sport'].notna()].copy()
        
        # Parse release date - handle any format issues
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df[df['release_date'].notna()].copy()  # Drop rows with invalid dates
        
        df['year'] = df['release_date'].dt.year.astype(int)
        df['quarter'] = df['release_date'].dt.quarter.astype(int)
        
        # Filter for major documentaries only
        # Handle NA in box_office_usd and streaming_platform
        df['has_box_office'] = (df['box_office_usd'].notna()) & (df['box_office_usd'] != 'NA')
        df['has_streaming'] = (df['streaming_platform'].notna()) & (df['streaming_platform'] != 'NA')
        
        df['major_doc'] = (
            (df['imdb_rating'] >= self.config.MAJOR_DOC_RATING_THRESHOLD) &
            (df['has_box_office'] | df['has_streaming'])
        ).astype(int)
        
        # Create indicators by sport-quarter
        doc_indicators = df[df['major_doc'] == 1].groupby(['sport', 'year', 'quarter']).agg({
            'documentary_title': 'count'
        }).reset_index()
        
        doc_indicators.columns = ['sport', 'year', 'quarter', 'documentary_release']
        doc_indicators['documentary_release'] = (doc_indicators['documentary_release'] > 0).astype(int)
        
        return doc_indicators
    
    def process_fred_data(self) -> pd.DataFrame:
        """
        Process FRED economic data into quarterly national averages
        
        Returns:
            DataFrame with columns: year, quarter, unemployment_rate, gas_price
        """
        df = pd.read_csv(self.config.FRED_FILE)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year.astype(int)
        df['quarter'] = df['date'].dt.quarter.astype(int)
        
        # Filter date range
        df = df[(df['year'] >= self.config.START_YEAR) & 
                (df['year'] <= self.config.END_YEAR)].copy()
        
        # Pivot to get variables as columns
        pivot_df = df.pivot_table(
            index=['year', 'quarter'],
            columns='variable',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Ensure we have unemployment_rate column
        if 'unemployment_rate' not in pivot_df.columns:
            # If not available, use national average
            pivot_df['unemployment_rate'] = 5.0  # Fallback value
        
        # Handle gas price if available
        if 'gas_price' in pivot_df.columns:
            pivot_df['gas_price'] = pivot_df['gas_price'].fillna(pivot_df['gas_price'].mean())
        else:
            pivot_df['gas_price'] = 2.50  # Fallback value
        
        return pivot_df[['year', 'quarter', 'unemployment_rate', 'gas_price']]
    
    def create_master_panel(
        self,
        youtube_data: pd.DataFrame,
        trends_data: pd.DataFrame,
        barrier_data: pd.DataFrame,
        doc_data: pd.DataFrame,
        fred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all data sources into master panel
        
        Returns:
            Master panel DataFrame with all variables
        """
        # Create time skeleton
        years = range(self.config.START_YEAR, self.config.END_YEAR + 1)
        quarters = range(1, 5)
        
        # Get unique sports from both YouTube and Trends
        youtube_sports = set(youtube_data['sport'].unique())
        trends_sports = set(trends_data['sport'].unique())
        barrier_sports = set(barrier_data['sport'].unique())
        
        # Take intersection of all three
        common_sports = youtube_sports & trends_sports & barrier_sports
        
        logger.info(f"Common sports across all datasets: {sorted(common_sports)}")
        
        # Create skeleton
        skeleton = pd.DataFrame([
            {'sport': sport, 'year': year, 'quarter': quarter}
            for sport in common_sports
            for year in years
            for quarter in quarters
        ])
        
        # Merge YouTube data
        master = skeleton.merge(
            youtube_data[['sport', 'year', 'quarter', 'total_views', 'log_views']],
            on=['sport', 'year', 'quarter'],
            how='left'
        )
        
        # Merge Google Trends data
        master = master.merge(
            trends_data[['sport', 'year', 'quarter', 'search_index', 'log_sales_proxy']],
            on=['sport', 'year', 'quarter'],
            how='left'
        )
        
        # Merge Barrier Index (sport-level, no time dimension)
        master = master.merge(
            barrier_data[['sport', 'barrier_index', 'barrier_category']],
            on='sport',
            how='left'
        )
        
        # Merge Documentary indicators
        master = master.merge(
            doc_data,
            on=['sport', 'year', 'quarter'],
            how='left'
        )
        master['documentary_release'] = master['documentary_release'].fillna(0).astype(int)
        
        # Merge FRED controls (quarter-level, no sport dimension)
        master = master.merge(
            fred_data,
            on=['year', 'quarter'],
            how='left'
        )
        
        # Create date variable
        master['date'] = pd.to_datetime(
            master['year'].astype(str) + '-' + 
            (master['quarter'] * 3).astype(str) + '-01'
        )
        
        # Create interaction term
        master['log_views_x_barrier'] = master['log_views'] * master['barrier_index']
        
        # Create time trend
        master['time_trend'] = (master['year'] - self.config.START_YEAR) * 4 + master['quarter']
        
        # Create COVID period dummy
        master['covid_period'] = (
            ((master['year'] == 2020) & (master['quarter'] >= 2)) |
            ((master['year'] == 2021) & (master['quarter'] <= 2))
        ).astype(int)
        
        # Create quarter dummies
        for q in [1, 2, 3]:
            master[f'Q{q}'] = (master['quarter'] == q).astype(int)
        
        # Create post-documentary counters
        master = master.sort_values(['sport', 'year', 'quarter'])
        master['post_documentary'] = master.groupby('sport')['documentary_release'].cumsum()
        
        # Handle missing values
        # For views and sales_proxy, fill with 0 (no activity)
        master['total_views'] = master['total_views'].fillna(0)
        master['log_views'] = master['log_views'].fillna(0)
        master['search_index'] = master['search_index'].fillna(0)
        master['log_sales_proxy'] = master['log_sales_proxy'].fillna(0)
        
        # Sort by sport and date
        master = master.sort_values(['sport', 'date']).reset_index(drop=True)
        
        logger.info(f"\nMaster panel statistics:")
        logger.info(f"  Total observations: {len(master)}")
        logger.info(f"  Sports: {master['sport'].nunique()}")
        logger.info(f"  Years: {master['year'].nunique()}")
        logger.info(f"  Missing values:\n{master.isnull().sum()[master.isnull().sum() > 0]}")
        
        return master