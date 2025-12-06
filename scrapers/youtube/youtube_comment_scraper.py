"""
YouTube Comment Scraper - ÉTAPE 2
==================================
Scrape TOUS les commentaires pour les videos collectées.

Méthodologie:
- Input: data/raw/youtube_videos.csv
- Process: Récupère tous les comments via commentThreads API
- Output: data/raw/youtube_comments/[Sport]_comments.csv

API Quota: ~20,800 units (2.08 jours)
"""

import pandas as pd
import time
import json
import os
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, YOUTUBE_API_KEY,
    RATE_LIMIT_DELAY, MAX_RETRIES
)

class YouTubeCommentScraper:
    """Scraper pour obtenir tous les commentaires des videos"""
    
    def __init__(self):
        if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY_HERE':
            raise ValueError("⚠️  YouTube API key not configured! Edit src/config.py")
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.api_units_used = 0
        
        # Setup directories
        self.comments_dir = RAW_DATA_DIR / 'youtube_comments'
        self.comments_dir.mkdir(exist_ok=True)
        
        # Load videos
        videos_file = RAW_DATA_DIR / 'youtube_videos.csv'
        if not videos_file.exists():
            raise FileNotFoundError(f"❌ Videos file not found: {videos_file}")
        
        self.videos_df = pd.read_csv(videos_file)
        print(f"✅ Loaded {len(self.videos_df)} videos to process")
        
        # Load progress
        self.progress_file = self.comments_dir / '_scraping_progress.json'
        self.processed_videos = self._load_progress()
        
    def _load_progress(self):
        """Load list of already processed video IDs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed_videos', []))
        return set()
    
    def _save_progress(self):
        """Save progress to JSON"""
        data = {
            'completed_videos': list(self.processed_videos),
            'last_update': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f)
            
    def get_video_comments(self, video_id, sport):
        """
        Récupère tous les commentaires d'une video.
        Cost: 1 unit per 100 comments
        """
        comments = []
        next_page_token = None
        
        while True:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    textFormat='plainText',
                    pageToken=next_page_token
                )
                
                response = request.execute()
                self.api_units_used += 1
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'video_id': video_id,
                        'sport': sport,
                        'comment_id': item['id'],
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt'],
                        'updated_at': comment['updatedAt']
                    })
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                time.sleep(RATE_LIMIT_DELAY)
                
            except HttpError as e:
                if e.resp.status == 403 and 'quotaExceeded' in str(e):
                    raise e  # Propagate quota error
                elif e.resp.status == 404:
                    print(f"    ⚠️ Video not found (404): {video_id}")
                    break
                elif e.resp.status == 403 and 'commentsDisabled' in str(e):
                    print(f"    ⚠️ Comments disabled: {video_id}")
                    break
                else:
                    print(f"    ❌ Error fetching comments: {e}")
                    break
            except Exception as e:
                print(f"    ❌ Unexpected error: {e}")
                break
                
        return comments

    def run(self):
        """Run comment scraping for all videos"""
        print("\n" + "="*80)
        print("🚀 YOUTUBE COMMENT SCRAPER - ÉTAPE 2")
        print("="*80)
        
        videos_to_process = self.videos_df[~self.videos_df['video_id'].isin(self.processed_videos)].copy()
        print(f"📋 Videos remaining: {len(videos_to_process)} / {len(self.videos_df)}")
        
        if len(videos_to_process) == 0:
            print("✅ All videos already processed!")
            return

        # Calculate channel size (total views) to prioritize smaller channels
        print("\n📊 Calculating channel sizes to prioritize smaller channels...")
        channel_sizes = self.videos_df.groupby('sport')['views'].sum().reset_index()
        channel_sizes.columns = ['sport', 'total_channel_views']
        
        # Merge size info
        videos_to_process = videos_to_process.merge(channel_sizes, on='sport', how='left')
        
        # Sort by total_channel_views ASCENDING (Smallest first)
        # This maximizes the number of channels processed within quota
        videos_to_process = videos_to_process.sort_values('total_channel_views', ascending=True)
        
        print("   Order of processing (Smallest to Largest):")
        unique_sports = videos_to_process[['sport', 'total_channel_views']].drop_duplicates()
        for _, row in unique_sports.head(5).iterrows():
            print(f"   - {row['sport']}: {row['total_channel_views']:,} views")
        print("   ...")
        for _, row in unique_sports.tail(3).iterrows():
            print(f"   - {row['sport']}: {row['total_channel_views']:,} views")

        # Group by sport to manage files
        sports = videos_to_process['sport'].unique()
        
        try:
            for sport in sports:
                print(f"\nProcessing sport: {sport}")
                sport_videos = videos_to_process[videos_to_process['sport'] == sport]
                
                # Load existing sport comments if any
                output_file = self.comments_dir / f'{sport}_comments.csv'
                if output_file.exists():
                    sport_comments = pd.read_csv(output_file).to_dict('records')
                else:
                    sport_comments = []
                
                initial_count = len(sport_comments)
                new_comments_count = 0
                
                for idx, row in sport_videos.iterrows():
                    video_id = row['video_id']
                    print(f"  [{idx+1}/{len(self.videos_df)}] Fetching comments for {video_id}...")
                    
                    comments = self.get_video_comments(video_id, sport)
                    sport_comments.extend(comments)
                    new_comments_count += len(comments)
                    
                    self.processed_videos.add(video_id)
                    
                    # Save periodically
                    if len(self.processed_videos) % 10 == 0:
                        pd.DataFrame(sport_comments).to_csv(output_file, index=False)
                        self._save_progress()
                        
                    # Check quota
                    if self.api_units_used > 9500:
                        raise Exception("QUOTA_LIMIT_REACHED")
                        
                # Final save for sport
                if sport_comments:
                    pd.DataFrame(sport_comments).to_csv(output_file, index=False)
                    self._save_progress()
                    print(f"  ✅ Saved {len(sport_comments)} comments for {sport} (+{new_comments_count} new)")
                    
        except Exception as e:
            if "QUOTA_LIMIT_REACHED" in str(e):
                print("\n⚠️  QUOTA LIMIT REACHED (9500 units)")
                print("   Pausing execution. Resume tomorrow with --resume")
            else:
                print(f"\n❌ Error: {e}")
            self._save_progress()
            
        print("\n" + "="*80)
        print("✅ SESSION COMPLETE")
        print(f"📊 API units used: {self.api_units_used}")
        print(f"📝 Videos processed: {len(self.processed_videos)} / {len(self.videos_df)}")

if __name__ == '__main__':
    scraper = YouTubeCommentScraper()
    scraper.run()
