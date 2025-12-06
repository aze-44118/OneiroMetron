"""
YouTube Video Scraper - ÉTAPE 1
================================
Scrape les top 100 videos (les plus vues) pour chaque channel sélectionné.

Méthodologie:
- 1 channel par sport (top purity index)
- Top 100 videos les plus vues (capture 80-90% engagement)
- Données: video_id, published_at, views, likes, comments

API Quota: ~10,500 units (1.05 jours)
"""

import pandas as pd
import time
from datetime import datetime
from googleapiclient.discovery import build
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, YOUTUBE_API_KEY,
    RATE_LIMIT_DELAY, MAX_RETRIES
)

import argparse
import os

class YouTubeVideoScraper:
    """Scraper pour obtenir les top videos de chaque channel"""
    
    def __init__(self, target_ranking=None, sports_filter=None):
        if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY_HERE':
            raise ValueError("⚠️  YouTube API key not configured! Edit src/config.py")
        
        self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        self.api_units_used = 0
        self.target_ranking = target_ranking
        self.sports_filter = sports_filter
        
        # Load selected channels
        self.channels_file = RAW_DATA_DIR / 'youtube_channel.csv'
        if not self.channels_file.exists():
            raise FileNotFoundError(f"❌ Channel file not found: {self.channels_file}")
        
        self.channels_df = pd.read_csv(self.channels_file)
        
        # Initialize status column if not present
        if 'status' not in self.channels_df.columns:
            self.channels_df['status'] = 'PENDING'
        else:
            self.channels_df['status'] = self.channels_df['status'].fillna('PENDING')
            
        self.channels_df.to_csv(self.channels_file, index=False)
            
        print(f"✅ Loaded {len(self.channels_df)} channels")
        if self.target_ranking:
            print(f"   Targeting ranking: {self.target_ranking}")
        if self.sports_filter:
            print(f"   Targeting sports: {self.sports_filter}")

    def get_channel_videos(self, channel_id, channel_name, sport, max_videos=100):
        """
        Récupère toutes les videos d'un channel, triées par views.
        """
        print(f"  📹 Fetching videos for: {channel_name}")
        all_videos = []
        next_page_token = None
        videos_fetched = 0
        
        # PHASE 1: Get video IDs (expensive)
        while videos_fetched < max_videos:
            try:
                request = self.youtube.search().list(
                    part='id',
                    channelId=channel_id,
                    maxResults=min(50, max_videos - videos_fetched),
                    order='viewCount',  # Sort by most viewed
                    type='video',
                    pageToken=next_page_token
                )
                
                response = request.execute()
                self.api_units_used += 100
                
                video_ids = [item['id']['videoId'] for item in response['items']]
                
                if not video_ids:
                    break
                
                # PHASE 2: Get video details (cheap)
                video_request = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                )
                
                video_response = video_request.execute()
                self.api_units_used += 1
                
                for item in video_response['items']:
                    stats = item['statistics']
                    snippet = item['snippet']
                    
                    all_videos.append({
                        'sport': sport,
                        'channel_id': channel_id,
                        'channel_name': channel_name,
                        'video_id': item['id'],
                        'title': snippet['title'],
                        'published_at': snippet['publishedAt'],
                        'views': int(stats.get('viewCount', 0)),
                        'likes': int(stats.get('likeCount', 0)),
                        'comments': int(stats.get('commentCount', 0)),
                        'duration': item.get('contentDetails', {}).get('duration', 'Unknown')
                    })
                
                videos_fetched += len(video_ids)
                next_page_token = response.get('nextPageToken')
                
                if not next_page_token:
                    break
                
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                print(f"    ❌ Error fetching videos: {e}")
                break
        
        print(f"    ✅ Collected {len(all_videos)} videos")
        print(f"    📊 API units used: {self.api_units_used:,}")
        
        return all_videos
    
    def run(self):
        """Run video scraping for all channels"""
        print("\n" + "="*80)
        print("🚀 YOUTUBE VIDEO SCRAPER - ÉTAPE 1")
        print("="*80)
        print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        new_videos = []
        
        # Filter channels
        channels_to_process = self.channels_df.copy()
        
        if self.target_ranking:
            channels_to_process = channels_to_process[channels_to_process['ranking'] == self.target_ranking]
            
        if self.sports_filter:
            sports_list = [s.strip() for s in self.sports_filter.split(',')]
            channels_to_process = channels_to_process[channels_to_process['sport'].isin(sports_list)]
            
        # Filter for pending channels (within the selection)
        # We only process if status is NOT DONE, or if we are forcing (implied by running specific ranking)
        # Actually, let's respect status. If user wants to re-run, they should clear status or use force flag (not implemented yet).
        # But for new channels, status should be PENDING (or NaN filled to PENDING).
        
        pending_channels = channels_to_process[channels_to_process['status'] != 'DONE']
        
        print(f"📋 Channels to process: {len(pending_channels)} (out of {len(channels_to_process)} selected)")
        
        if len(pending_channels) == 0:
            print("✅ No pending channels to process.")
            return
            
        print(f"💰 Estimated API cost: ~{len(pending_channels) * 202:,} units")
        
        for idx, row in pending_channels.iterrows():
            sport = row['sport']
            channel_name = row['channel']
            channel_id = row['id']
            
            print(f"\n[{idx+1}/{len(self.channels_df)}] {sport} - {channel_name} (Rank {row.get('ranking', '?')})")
            
            try:
                videos = self.get_channel_videos(
                    channel_id=channel_id,
                    channel_name=channel_name,
                    sport=sport,
                    max_videos=100
                )
                
                if videos:
                    new_videos.extend(videos)
                    self.channels_df.at[idx, 'status'] = 'DONE'
                else:
                    print("    ⚠️ No videos found")
                    self.channels_df.at[idx, 'status'] = 'NO_VIDEOS'
                    
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                self.channels_df.at[idx, 'status'] = 'FAILED'
            
            # Save status immediately
            self.channels_df.to_csv(self.channels_file, index=False)
            
            # Safety check: stop if approaching quota
            if self.api_units_used > 9000:
                print(f"\n⚠️  Approaching quota limit ({self.api_units_used:,}/10,000)")
                print("⚠️  Stopping to preserve quota for tomorrow.")
                break
            
            time.sleep(1)  # Extra rate limiting
        
        # Save results (Append mode)
        output_file = RAW_DATA_DIR / 'youtube_videos.csv'
        
        if new_videos:
            new_df = pd.DataFrame(new_videos)
            new_df['published_at'] = pd.to_datetime(new_df['published_at'])
            new_df['date_collected'] = datetime.now().strftime('%Y-%m-%d')
            
            if output_file.exists():
                print(f"\n📥 Loading existing videos from {output_file}")
                existing_df = pd.read_csv(output_file)
                existing_df['published_at'] = pd.to_datetime(existing_df['published_at'])
                
                # Combine and remove duplicates based on video_id
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['video_id'], keep='last')
                
                print(f"   Merged {len(new_df)} new videos. Total: {len(combined_df)}")
                df = combined_df
            else:
                df = new_df
                print(f"   Created new file with {len(df)} videos")
            
            # Sort
            df = df.sort_values(['sport', 'views'], ascending=[True, False])
            df.to_csv(output_file, index=False)
            
            elapsed_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("✅ VIDEO SCRAPING COMPLETE")
            print("="*80)
            print(f"📁 Output: {output_file}")
            print(f"📊 Total videos: {len(df):,}")
            print(f"💰 API units used: {self.api_units_used:,} / 10,000")
            
            return df
        else:
            print("\n❌ No new videos collected")
            return pd.DataFrame()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YouTube Video Scraper')
    parser.add_argument('--ranking', type=int, help='Filter channels by ranking (e.g., 2)')
    parser.add_argument('--sports', type=str, help='Filter by sports (comma separated)')
    
    args = parser.parse_args()
    
    scraper = YouTubeVideoScraper(target_ranking=args.ranking, sports_filter=args.sports)
    scraper.run()