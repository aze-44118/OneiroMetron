"""
YouTube Quarterly Aggregator - ÉTAPE 3
=======================================
Agrège les comments par quarter en utilisant la fenêtre d'engagement active.

Méthodologie:
1. Pour chaque quarter Q:
   - Identifier videos "actives" = publiées dans les 6 derniers mois (2 quarters)
   - Compter tous les comments publiés pendant Q sur ces videos actives
   - Calculer engagement_intensity = total_comments / nb_videos_actives

2. Variables finales:
   - youtube_total_comments: Raw count
   - youtube_active_videos: Nombre de videos actives
   - youtube_engagement_intensity: Comments normalisés par video

Pas d'API calls. Processing local uniquement.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class YouTubeQuarterlyAggregator:
    """Agrégation des comments par quarter avec fenêtre active"""
    
    def __init__(self, window_quarters=2):
        """
        Parameters:
        - window_quarters: Fenêtre d'activité (en quarters). Défaut=2 (6 mois)
        """
        self.window_quarters = window_quarters
        self.window_months = window_quarters * 3
        
        # Load videos
        videos_file = RAW_DATA_DIR / 'youtube_videos.csv'
        if not videos_file.exists():
            raise FileNotFoundError(f"❌ Videos file not found: {videos_file}")
        
        self.videos_df = pd.read_csv(videos_file)
        # Convert to datetime and remove timezone info for comparison
        self.videos_df['published_at'] = pd.to_datetime(self.videos_df['published_at']).dt.tz_localize(None)
        
        print(f"✅ Loaded {len(self.videos_df)} videos")
        
        # Load comments
        self.comments_dir = RAW_DATA_DIR / 'youtube_comments'
        if not self.comments_dir.exists():
            raise FileNotFoundError(f"❌ Comments directory not found: {self.comments_dir}")
        
        self.comments_files = list(self.comments_dir.glob('*_comments.csv'))
        print(f"✅ Found {len(self.comments_files)} comment files")
        
        # Load all comments into memory (may be large!)
        self.all_comments = self._load_all_comments()
        
    def _load_all_comments(self):
        """Load and combine all comment files"""
        print("\n📥 Loading comments into memory...")
        
        all_dfs = []
        total_comments = 0
        
        for file in self.comments_files:
            try:
                # Use python engine for robustness against buffer overflows
                # Skip bad lines (malformed CSVs)
                df = pd.read_csv(file, engine='python', on_bad_lines='skip')
                
                # Ensure date parsing works and remove timezone
                if 'published_at' in df.columns:
                    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce').dt.tz_localize(None)
                    df = df.dropna(subset=['published_at'])
                    
                all_dfs.append(df)
                total_comments += len(df)
                print(f"   {file.name}: {len(df):,} comments")
            except Exception as e:
                print(f"   ❌ Error reading {file.name}: {e}")
        
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ Total comments loaded: {total_comments:,}")
        
        return combined
    
    def get_active_videos(self, quarter_end, sport):
        """
        Retourne les videos "actives" pour un quarter donné.
        
        Active = publiée dans les [window_months] mois avant quarter_end
        """
        window_start = quarter_end - pd.DateOffset(months=self.window_months)
        
        active = self.videos_df[
            (self.videos_df['sport'] == sport) &
            (self.videos_df['published_at'] >= window_start) &
            (self.videos_df['published_at'] < quarter_end)
        ]
        
        return active
    
    def calculate_channel_weights(self, sport):
        """
        Calculate weights for each channel in a sport based on total views.
        Returns: dict {channel_id: weight}
        """
        sport_videos = self.videos_df[self.videos_df['sport'] == sport]
        channel_views = sport_videos.groupby('channel_id')['views'].sum()
        total_views = channel_views.sum()
        
        if total_views == 0:
            return {cid: 1.0/len(channel_views) for cid in channel_views.index}
            
        weights = channel_views / total_views
        return weights.to_dict()

    def aggregate_quarter(self, quarter_start, quarter_end, sport, weighted=False):
        """
        Agrège l'engagement pour un sport pendant un quarter.
        
        Returns: dict avec metrics
        """
        # Get active videos (from all channels in this sport)
        active_videos = self.get_active_videos(quarter_end, sport)
        
        if len(active_videos) == 0:
            return {
                'total_comments': 0,
                'active_videos': 0,
                'engagement_intensity': 0,
                'avg_video_age_days': 0
            }
        
        # Get comments for these videos
        video_ids = active_videos['video_id'].tolist()
        
        quarter_comments = self.all_comments[
            (self.all_comments['video_id'].isin(video_ids)) &
            (self.all_comments['published_at'] >= quarter_start) &
            (self.all_comments['published_at'] < quarter_end)
        ]
        
        total_comments = len(quarter_comments)
        active_count = len(active_videos)
        
        if not weighted:
            # Standard aggregation (pooled)
            engagement_intensity = total_comments / active_count if active_count > 0 else 0
        else:
            # Weighted aggregation
            weights = self.calculate_channel_weights(sport)
            channels = active_videos['channel_id'].unique()
            weighted_intensity = 0
            
            # We need to consider ALL channels that exist for this sport, 
            # even if they don't have active videos?
            # If a channel has no active videos, its intensity is 0.
            # And it contributes 0 * weight to the sum.
            # So we iterate over all channels defined in weights.
            
            for channel_id, weight in weights.items():
                # Check if this channel has active videos in this quarter
                channel_active = active_videos[active_videos['channel_id'] == channel_id]
                n_active = len(channel_active)
                
                if n_active > 0:
                    # Get comments for this channel's active videos
                    c_video_ids = channel_active['video_id'].tolist()
                    c_comments = quarter_comments[quarter_comments['video_id'].isin(c_video_ids)]
                    n_comments = len(c_comments)
                    
                    intensity = n_comments / n_active
                else:
                    intensity = 0
                    
                weighted_intensity += intensity * weight
                
            engagement_intensity = weighted_intensity
        
        # Average video age (days since publication to quarter_end)
        avg_age = (quarter_end - active_videos['published_at']).dt.days.mean()
        
        return {
            'total_comments': total_comments,
            'active_videos': active_count,
            'engagement_intensity': engagement_intensity,
            'avg_video_age_days': avg_age
        }
    
    def run(self, start_date='2015-01-01', end_date='2024-12-31', weighted=False):
        """
        Run quarterly aggregation for all sports.
        
        Returns: DataFrame with quarterly engagement metrics
        """
        print("\n" + "="*80)
        print("🚀 YOUTUBE QUARTERLY AGGREGATION - ÉTAPE 3")
        print("="*80)
        print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🪟 Active window: {self.window_months} months ({self.window_quarters} quarters)")
        print(f"⚖️  Weighted Mode: {'ON' if weighted else 'OFF'}")
        print(f"🏆 Sports: {self.videos_df['sport'].nunique()}\n")
        
        # Generate quarters
        quarters = pd.date_range(start_date, end_date, freq='QS')
        sports = sorted(self.videos_df['sport'].unique())
        
        print(f"📊 Total observations to generate: {len(quarters)} quarters × {len(sports)} sports = {len(quarters) * len(sports):,}\n")
        
        results = []
        
        for sport_idx, sport in enumerate(sports, 1):
            print(f"[{sport_idx}/{len(sports)}] Processing {sport}...")
            
            for quarter_idx, quarter_start in enumerate(quarters):
                quarter_end = quarter_start + pd.DateOffset(months=3)
                
                metrics = self.aggregate_quarter(quarter_start, quarter_end, sport, weighted=weighted)
                
                results.append({
                    'date': quarter_start.strftime('%Y-%m-%d'),
                    'quarter': f"{quarter_start.year}-Q{quarter_start.quarter}",
                    'sport': sport,
                    'total_comments': metrics['total_comments'],
                    'active_videos': metrics['active_videos'],
                    'engagement_intensity': round(metrics['engagement_intensity'], 2),
                    'avg_video_age_days': round(metrics['avg_video_age_days'], 1)
                })
            
            # Progress update
            if sport_idx % 5 == 0:
                print(f"   ✅ {sport_idx}/{len(sports)} sports completed")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add log-transformed variable
        df['log_engagement_intensity'] = np.log1p(df['engagement_intensity'])
        
        # Save results
        output_file = RAW_DATA_DIR / 'youtube_quarterly_engagement.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("✅ AGGREGATION COMPLETE")
        print("="*80)
        print(f"📁 Output: {output_file}")
        print(f"📊 Observations: {len(df):,}")
        print(f"🏆 Sports: {df['sport'].nunique()}")
        print(f"📅 Quarters: {df['quarter'].nunique()}")
        
        # Summary statistics
        print("\n📈 ENGAGEMENT SUMMARY:")
        print(f"   Total comments: {df['total_comments'].sum():,}")
        print(f"   Avg comments/quarter/sport: {df['total_comments'].mean():.1f}")
        print(f"   Avg engagement intensity: {df['engagement_intensity'].mean():.2f}")
        print(f"   Median active videos: {df['active_videos'].median():.0f}")
        
        return df

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YouTube Aggregator')
    parser.add_argument('--weighted', action='store_true', help='Enable weighted aggregation based on channel views')
    parser.add_argument('--window', type=int, default=2, help='Active window in quarters (default: 2)')
    
    args = parser.parse_args()
    
    aggregator = YouTubeQuarterlyAggregator(window_quarters=args.window)
    df = aggregator.run(weighted=args.weighted)