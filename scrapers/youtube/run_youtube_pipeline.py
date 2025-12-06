"""
YouTube Data Pipeline - MASTER SCRIPT
=====================================
Orchestre les 3 étapes du scraping YouTube:

ÉTAPE 1: Scrape videos (1.05 jours, 10.5K units)
ÉTAPE 2: Scrape comments (2.08 jours, 20.8K units)
ÉTAPE 3: Aggregate quarterly (instantané, 0 units)

Total: ~4 jours, ~32K API units

Usage:
  python run_youtube_pipeline.py --all          # Run toutes les étapes
  python run_youtube_pipeline.py --step 1       # Run étape 1 seulement
  python run_youtube_pipeline.py --step 2       # Run étape 2 seulement
  python run_youtube_pipeline.py --step 3       # Run étape 3 seulement
  python run_youtube_pipeline.py --resume       # Resume étape 2 après quota
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def run_step_1():
    """ÉTAPE 1: Scrape videos"""
    print("\n" + "="*80)
    print("▶️  LAUNCHING STEP 1: VIDEO SCRAPER")
    print("="*80)
    
    from youtube_video_scraper import YouTubeVideoScraper
    
    scraper = YouTubeVideoScraper()
    df = scraper.run()
    
    if len(df) > 0:
        print("\n✅ Step 1 complete. Ready for Step 2.")
        return True
    else:
        print("\n❌ Step 1 failed. Check errors above.")
        return False

def run_step_2():
    """ÉTAPE 2: Scrape comments"""
    print("\n" + "="*80)
    print("▶️  LAUNCHING STEP 2: COMMENT SCRAPER")
    print("="*80)
    print("⚠️  This step can take 48+ hours. Script will auto-pause when quota reached.")
    print("⚠️  You can resume by running: python run_youtube_pipeline.py --resume\n")
    
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return False
    
    from youtube_comment_scraper import YouTubeCommentScraper
    
    scraper = YouTubeCommentScraper()
    scraper.run()
    
    print("\n✅ Step 2 session complete.")
    print("💡 Check if all comments scraped. Resume tomorrow if needed.")
    return True

def run_step_3():
    """ÉTAPE 3: Aggregate quarterly"""
    print("\n" + "="*80)
    print("▶️  LAUNCHING STEP 3: QUARTERLY AGGREGATION")
    print("="*80)
    
    from youtube_aggregator import YouTubeQuarterlyAggregator
    
    aggregator = YouTubeQuarterlyAggregator(window_quarters=2)
    df = aggregator.run()
    
    if len(df) > 0:
        print("\n✅ Step 3 complete. Final dataset ready!")
        return True
    else:
        print("\n❌ Step 3 failed. Check errors above.")
        return False

def check_prerequisites():
    """Check if required files exist"""
    from config import RAW_DATA_DIR, YOUTUBE_API_KEY
    
    print("\n🔍 CHECKING PREREQUISITES...")
    
    # Check API key
    if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY_HERE':
        print("❌ YouTube API key not configured!")
        print("   → Edit src/config.py and add your API key")
        return False
    else:
        print("✅ API key configured")
    
    # Check channels file
    channels_file = RAW_DATA_DIR / 'youtube_channel.csv'
    if not channels_file.exists():
        print(f"❌ Channels file not found: {channels_file}")
        return False
    else:
        print(f"✅ Channels file found: {channels_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='YouTube Data Pipeline')
    parser.add_argument('--all', action='store_true', help='Run all 3 steps')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], help='Run specific step')
    parser.add_argument('--resume', action='store_true', help='Resume step 2 after quota')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎬 YOUTUBE DATA PIPELINE")
    print("="*80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return
    
    # Run requested steps
    if args.all:
        print("\n📋 Running FULL PIPELINE (all 3 steps)")
        print("⚠️  This will take ~4 days total")
        print("⚠️  Step 2 will auto-pause when quota reached\n")
        
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        
        success_1 = run_step_1()
        if not success_1:
            return
        
        success_2 = run_step_2()
        if not success_2:
            print("\n💡 Resume step 2 tomorrow: python run_youtube_pipeline.py --resume")
            return
        
        success_3 = run_step_3()
        
        if success_3:
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLETE!")
            print("="*80)
            print("📁 Final output: data/raw/youtube_quarterly_engagement.csv")
            print("💡 Ready to merge with Google Trends data for regression analysis")
    
    elif args.step == 1:
        run_step_1()
    
    elif args.step == 2 or args.resume:
        run_step_2()
    
    elif args.step == 3:
        run_step_3()
    
    else:
        print("❌ No action specified. Use --all, --step N, or --resume")
        parser.print_help()

if __name__ == '__main__':
    main()