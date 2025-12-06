"""
YouTube Views Estimator
========================
Estimates quarterly view distribution from comment patterns and total views.

Methodology:
- Uses comment temporal distribution as proxy for view distribution
- Applies decay correction for early-view surge (first 6 months)
- Validates estimates against known total views

Input:
- data/raw/youtube_videos.csv (video metadata with total views)
- data/raw/youtube_comments/*.csv (timestamped comments)

Output:
- data/processed/youtube_quarterly_views_estimated.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import project config
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class YouTubeViewsEstimator:
    """
    Estimates quarterly view distribution using comment patterns.
    
    Parameters:
    -----------
    decay_beta : float
        Early boost parameter (default: 2.0)
        Higher = more weight to early views
    
    decay_alpha : float
        Decay rate parameter (default: 0.3)
        Higher = faster decay of early boost
    """
    
    def __init__(self, decay_beta=2.0, decay_alpha=0.3):
        self.decay_beta = decay_beta
        self.decay_alpha = decay_alpha
        
        # Load data
        self.videos_file = RAW_DATA_DIR / 'youtube_videos.csv'
        self.comments_dir = RAW_DATA_DIR / 'youtube_comments'
        
        if not self.videos_file.exists():
            raise FileNotFoundError(f"Videos file not found: {self.videos_file}")
        
        if not self.comments_dir.exists():
            raise FileNotFoundError(f"Comments directory not found: {self.comments_dir}")
        
        # Load videos metadata
        self.df_videos = pd.read_csv(self.videos_file)
        self.df_videos['published_at'] = pd.to_datetime(self.df_videos['published_at'], utc=True).dt.tz_localize(None)
        
        print(f"✅ Loaded {len(self.df_videos)} videos")
        print(f"📁 Comments directory: {self.comments_dir}")
        
        # Load all comments
        self.df_comments = self._load_all_comments()
        
    def _load_all_comments(self):
        """Load and concatenate all comment files"""
        print("\n📥 Loading comments...")
        
        comment_files = list(self.comments_dir.glob('*_comments.csv'))
        if not comment_files:
            raise FileNotFoundError(f"No comment files found in {self.comments_dir}")
        
        all_comments = []
        for file in comment_files:
            try:
                df = pd.read_csv(file, engine='python', on_bad_lines='warn')
            except Exception as e:
                print(f"   ⚠️ Error reading {file.name}: {e}")
                continue
            df['published_at'] = pd.to_datetime(df['published_at'], utc=True).dt.tz_localize(None)
            all_comments.append(df)
            print(f"   {file.name}: {len(df):,} comments")
        
        df_combined = pd.concat(all_comments, ignore_index=True)
        print(f"\n✅ Total comments loaded: {len(df_combined):,}")
        
        return df_combined
    
    def _decay_factor(self, age_months):
        """
        Calculate decay factor for view adjustment.
        
        Early videos get boost (more views/comment ratio).
        Decays exponentially to 1.0 over ~6 months.
        """
        return 1.0 + self.decay_beta * np.exp(-self.decay_alpha * age_months)
    
    def _estimate_views_for_video(self, video_id, video_total_views, video_published_at):
        """
        Estimate quarterly view distribution for a single video.
        
        Returns:
        --------
        DataFrame with columns: [quarter, estimated_views, comment_count, decay_factor]
        """
        # Get comments for this video
        video_comments = self.df_comments[self.df_comments['video_id'] == video_id].copy()
        
        if len(video_comments) == 0:
            # No comments = assume all views in publication quarter
            pub_quarter = pd.Period(video_published_at, freq='Q')
            return pd.DataFrame({
                'video_id': [video_id],
                'quarter': [str(pub_quarter)],
                'estimated_views': [video_total_views],
                'comment_count': [0],
                'decay_factor': [1.0],
                'method': ['no_comments_fallback']
            })
        
        # Extract quarter from comment dates
        video_comments['quarter'] = video_comments['published_at'].dt.to_period('Q')
        
        # Count comments per quarter
        comments_per_quarter = video_comments.groupby('quarter').size().reset_index(name='comment_count')
        comments_per_quarter['quarter'] = comments_per_quarter['quarter'].astype(str)
        
        # Calculate comment rate (proportion of total comments)
        total_comments = len(video_comments)
        comments_per_quarter['comment_rate'] = comments_per_quarter['comment_count'] / total_comments
        
        # Calculate age in months for each quarter
        comments_per_quarter['quarter_start'] = pd.to_datetime(comments_per_quarter['quarter']).dt.tz_localize(None)
        
        # Ensure video_published_at is naive
        if hasattr(video_published_at, 'tzinfo') and video_published_at.tzinfo is not None:
            video_published_at = video_published_at.tz_localize(None)

        comments_per_quarter['age_months'] = (
            (comments_per_quarter['quarter_start'] - video_published_at).dt.days / 30.44
        )
        
        # Apply decay correction
        comments_per_quarter['decay_factor'] = comments_per_quarter['age_months'].apply(
            self._decay_factor
        )
        
        # Estimate raw views (proportional to comments)
        comments_per_quarter['raw_views'] = (
            video_total_views * comments_per_quarter['comment_rate']
        )
        
        # Apply decay adjustment
        comments_per_quarter['adjusted_views'] = (
            comments_per_quarter['raw_views'] * comments_per_quarter['decay_factor']
        )
        
        # Normalize to match total views (sanity check)
        total_estimated = comments_per_quarter['adjusted_views'].sum()
        if total_estimated > 0:
            normalization_factor = video_total_views / total_estimated
            comments_per_quarter['estimated_views'] = (
                comments_per_quarter['adjusted_views'] * normalization_factor
            )
        else:
            comments_per_quarter['estimated_views'] = 0
        
        # Keep relevant columns
        result = comments_per_quarter[[
            'quarter', 'estimated_views', 'comment_count', 'decay_factor'
        ]].copy()
        result['video_id'] = video_id
        result['method'] = 'comment_distribution'
        
        return result
    
    def estimate_all_views(self):
        """
        Estimate quarterly views for all videos.
        
        Returns:
        --------
        DataFrame with estimated views per video per quarter
        """
        print("\n🔄 Estimating quarterly views...")
        
        all_estimates = []
        
        for idx, video in self.df_videos.iterrows():
            video_id = video['video_id']
            total_views = video['views']
            published_at = video['published_at']
            
            if pd.isna(total_views) or total_views == 0:
                continue  # Skip videos with no views
            
            estimates = self._estimate_views_for_video(
                video_id, total_views, published_at
            )
            
            # Add video metadata
            estimates['sport'] = video['sport']
            estimates['channel_id'] = video['channel_id']
            estimates['channel_name'] = video['channel_name']
            estimates['total_views'] = total_views
            estimates['published_at'] = published_at
            
            all_estimates.append(estimates)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(self.df_videos)} videos...")
        
        # Combine all estimates
        df_views = pd.concat(all_estimates, ignore_index=True)
        
        print(f"\n✅ Estimated views for {df_views['video_id'].nunique()} videos")
        print(f"   Total quarters covered: {df_views['quarter'].nunique()}")
        
        return df_views
    
    def aggregate_by_sport_quarter(self, df_views):
        """
        Aggregate estimated views at sport-quarter level.
        
        Returns:
        --------
        DataFrame with total estimated views per sport per quarter
        """
        print("\n📊 Aggregating by sport-quarter...")
        
        # Convert quarter string back to datetime for sorting
        df_views['quarter_date'] = pd.PeriodIndex(df_views['quarter'], freq='Q').to_timestamp()
        
        # Aggregate
        df_agg = df_views.groupby(['sport', 'quarter', 'quarter_date']).agg({
            'estimated_views': 'sum',
            'comment_count': 'sum',
            'video_id': 'nunique',  # Count unique videos
            'total_views': 'sum'    # Sum of all video total views (for validation)
        }).reset_index()
        
        # Rename columns
        df_agg.rename(columns={
            'video_id': 'video_count',
            'total_views': 'total_views_all_videos'
        }, inplace=True)
        
        # Sort
        df_agg = df_agg.sort_values(['sport', 'quarter_date'])
        
        # Calculate average views per video
        df_agg['avg_views_per_video'] = (
            df_agg['estimated_views'] / df_agg['video_count']
        )
        
        print(f"✅ Created {len(df_agg)} sport-quarter observations")
        
        return df_agg
    
    def validate_estimates(self, df_views):
        """
        Validate estimates against known total views.
        
        Prints diagnostics and flags videos with large discrepancies.
        """
        print("\n🔍 Validating estimates...")
        
        # Sum estimated views per video
        validation = df_views.groupby('video_id').agg({
            'estimated_views': 'sum',
            'total_views': 'first'
        }).reset_index()
        
        # Calculate error
        validation['error_pct'] = (
            (validation['estimated_views'] - validation['total_views']) 
            / validation['total_views'] * 100
        )
        
        # Summary statistics
        mean_error = validation['error_pct'].mean()
        median_error = validation['error_pct'].median()
        std_error = validation['error_pct'].std()
        
        print(f"   Mean Error: {mean_error:.2f}%")
        print(f"   Median Error: {median_error:.2f}%")
        print(f"   Std Dev: {std_error:.2f}%")
        
        # Flag large errors (>20%)
        large_errors = validation[abs(validation['error_pct']) > 20]
        if len(large_errors) > 0:
            print(f"\n⚠️  {len(large_errors)} videos with >20% error:")
            print(large_errors[['video_id', 'total_views', 'estimated_views', 'error_pct']].head(10))
        else:
            print("\n✅ All videos within ±20% tolerance")
        
        return validation
    
    def run(self, save=True):
        """
        Run complete estimation pipeline.
        
        Steps:
        1. Estimate views for all videos
        2. Aggregate by sport-quarter
        3. Validate estimates
        4. Save results
        
        Returns:
        --------
        tuple: (df_views_detailed, df_sport_quarter_aggregated)
        """
        print("="*80)
        print("🚀 YOUTUBE VIEWS ESTIMATOR")
        print("="*80)
        print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Estimate views
        df_views = self.estimate_all_views()
        
        # Step 2: Aggregate
        df_agg = self.aggregate_by_sport_quarter(df_views)
        
        # Step 3: Validate
        validation = self.validate_estimates(df_views)
        
        # Step 4: Save
        if save:
            # Detailed (video-level)
            output_detailed = PROCESSED_DATA_DIR / 'youtube_quarterly_views_detailed.csv'
            df_views.to_csv(output_detailed, index=False)
            print(f"\n💾 Saved detailed estimates: {output_detailed}")
            
            # Aggregated (sport-quarter level)
            output_agg = PROCESSED_DATA_DIR / 'youtube_quarterly_views_estimated.csv'
            df_agg.to_csv(output_agg, index=False)
            print(f"💾 Saved aggregated estimates: {output_agg}")
            
            # Validation report
            output_validation = PROCESSED_DATA_DIR / 'youtube_views_validation.csv'
            validation.to_csv(output_validation, index=False)
            print(f"💾 Saved validation report: {output_validation}")
        
        print("\n" + "="*80)
        print("✅ ESTIMATION COMPLETE")
        print("="*80)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total videos: {df_views['video_id'].nunique()}")
        print(f"   Total sports: {df_views['sport'].nunique()}")
        print(f"   Total quarters: {df_views['quarter'].nunique()}")
        print(f"   Total estimated views: {df_agg['estimated_views'].sum():,.0f}")
        print(f"   Average views/quarter/sport: {df_agg['estimated_views'].mean():,.0f}")
        
        return df_views, df_agg


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Estimate quarterly YouTube views from comment patterns'
    )
    parser.add_argument(
        '--decay-beta',
        type=float,
        default=2.0,
        help='Early boost parameter (default: 2.0)'
    )
    parser.add_argument(
        '--decay-alpha',
        type=float,
        default=0.3,
        help='Decay rate parameter (default: 0.3)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files (validation only)'
    )
    
    args = parser.parse_args()
    
    # Run estimator
    estimator = YouTubeViewsEstimator(
        decay_beta=args.decay_beta,
        decay_alpha=args.decay_alpha
    )
    
    df_views, df_agg = estimator.run(save=not args.no_save)
    
    # Optional: Print sample
    print("\n📋 Sample Output (Aggregated):")
    print(df_agg.head(10).to_string(index=False))