"""
Master script to run all scrapers in sequence.
"""

import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add scrapers directory to path for local imports
sys.path.append(str(Path(__file__).parent))

from google_trends_scraper import GoogleTrendsScraper
from youtube_scraper import YouTubeScraper
from wikipedia_scraper import WikipediaScraper

# Define output paths
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'

def check_file_exists(filename):
    """Check if a data file already exists."""
    filepath = RAW_DATA_DIR / filename
    if filepath.exists():
        file_size = filepath.stat().st_size
        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        return True, file_size, mod_time
    return False, 0, None

def run_all_scrapers(skip_youtube=True, skip_google_trends=False, force_rescrape=False):
    """
    Run all scrapers in optimal order.
    
    Parameters:
    - skip_youtube: Skip YouTube scraping (requires manual channel IDs)
    - skip_google_trends: Skip Google Trends (takes longest)
    - force_rescrape: Force re-scraping even if files exist
    """
    
    print("\n" + "="*80)
    print("🚀 MASTER SCRAPER - RUNNING ALL DATA COLLECTION")
    print("="*80)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    results = {}
    
    # ========================================================================
    # 1. WIKIPEDIA (FASTEST - ~30 seconds)
    # ========================================================================
    print("\n" + "▶"*40)
    print("STEP 1/3: Wikipedia Pageviews")
    print("▶"*40)
    
    wiki_file = 'wikipedia_pageviews.csv'
    exists, size, mod_time = check_file_exists(wiki_file)
    
    if exists and not force_rescrape:
        print(f"✓ File already exists: {wiki_file}")
        print(f"  Size: {size:,} bytes")
        print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("  ⏭️  Skipping (use --force to re-scrape)")
        results['wikipedia'] = {'success': True, 'skipped': True, 'existing': True}
    else:
        try:
            wiki_scraper = WikipediaScraper()
            wiki_df = wiki_scraper.run()
            results['wikipedia'] = {
                'success': not wiki_df.empty,
                'rows': len(wiki_df) if not wiki_df.empty else 0
            }
        except Exception as e:
            print(f"❌ Wikipedia scraping failed: {e}")
            results['wikipedia'] = {'success': False, 'error': str(e)}
    
    # ========================================================================
    # 2. YOUTUBE (REQUIRES SETUP)
    # ========================================================================
    if not skip_youtube:
        print("\n" + "▶"*40)
        print("STEP 2/3: YouTube Channels")
        print("▶"*40)
        
        youtube_file = 'youtube_sports_data_quarterly.csv'
        exists, size, mod_time = check_file_exists(youtube_file)
        
        if exists and not force_rescrape:
            print(f"✓ File already exists: {youtube_file}")
            print(f"  Size: {size:,} bytes")
            print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("  ⏭️  Skipping (use --force to re-scrape)")
            results['youtube'] = {'success': True, 'skipped': True, 'existing': True}
        else:
            try:
                youtube_scraper = YouTubeScraper()
                youtube_df = youtube_scraper.run(search_for_ids=False)
                results['youtube'] = {
                    'success': not youtube_df.empty,
                    'rows': len(youtube_df) if not youtube_df.empty else 0
                }
            except Exception as e:
                print(f"❌ YouTube scraping failed: {e}")
                results['youtube'] = {'success': False, 'error': str(e)}
    else:
        print("\n⏭️  SKIPPING YouTube (requires manual channel ID setup)")
        results['youtube'] = {'success': False, 'skipped': True}
    
    # ========================================================================
    # 3. GOOGLE TRENDS (SLOWEST - ~20 minutes)
    # ========================================================================
    if not skip_google_trends:
        print("\n" + "▶"*40)
        print("STEP 3/3: Google Trends")
        print("▶"*40)
        
        equipment_file = 'google_trends_equipment.csv'
        behavioral_file = 'google_trends_behavioral.csv'
        
        eq_exists, eq_size, eq_time = check_file_exists(equipment_file)
        beh_exists, beh_size, beh_time = check_file_exists(behavioral_file)
        
        if eq_exists and beh_exists and not force_rescrape:
            print(f"✓ Files already exist:")
            print(f"  - {equipment_file}: {eq_size:,} bytes (modified: {eq_time.strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"  - {behavioral_file}: {beh_size:,} bytes (modified: {beh_time.strftime('%Y-%m-%d %H:%M:%S')})")
            print("  ⏭️  Skipping (use --force to re-scrape)")
            results['google_trends'] = {'success': True, 'skipped': True, 'existing': True}
        else:
            try:
                trends_scraper = GoogleTrendsScraper()
                equipment_df, behavioral_df = trends_scraper.run()
                results['google_trends'] = {
                    'success': True,
                    'equipment_rows': len(equipment_df) if not equipment_df.empty else 0,
                    'behavioral_rows': len(behavioral_df) if not behavioral_df.empty else 0
                }
            except Exception as e:
                print(f"❌ Google Trends scraping failed: {e}")
                results['google_trends'] = {'success': False, 'error': str(e)}
    else:
        print("\n⏭️  SKIPPING Google Trends (can be run separately)")
        results['google_trends'] = {'success': False, 'skipped': True}
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("📊 SCRAPING SUMMARY")
    print("="*80)
    
    for source, result in results.items():
        if result.get('existing'):
            status = "📁"
        elif result.get('success'):
            status = "✅"
        elif result.get('skipped'):
            status = "⏭️ "
        else:
            status = "❌"
            
        print(f"\n{status} {source.upper()}:")
        
        if result.get('existing'):
            print(f"   → Using existing file")
        elif result.get('success') and not result.get('skipped'):
            if 'rows' in result:
                print(f"   → {result['rows']} observations collected")
            elif 'equipment_rows' in result:
                print(f"   → Equipment: {result['equipment_rows']} observations")
                print(f"   → Behavioral: {result['behavioral_rows']} observations")
        elif result.get('skipped') and not result.get('existing'):
            print(f"   → Skipped")
        elif not result.get('success'):
            print(f"   → Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n⏱️  Total time: {overall_time/60:.1f} minutes")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 ALL SCRAPERS COMPLETE!")
    print("="*80)
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all data scrapers')
    parser.add_argument('--include-youtube', action='store_true',
                       help='Include YouTube scraping (requires channel IDs)')
    parser.add_argument('--include-google-trends', action='store_true',
                       help='Include Google Trends scraping (takes ~20 min)')
    parser.add_argument('--all', action='store_true',
                       help='Run ALL scrapers including slow ones')
    parser.add_argument('--force', action='store_true',
                       help='Force re-scraping even if files exist')
    
    args = parser.parse_args()
    
    skip_youtube = not (args.include_youtube or args.all)
    skip_trends = not (args.include_google_trends or args.all)
    
    print("\n🎮 CONFIGURATION:")
    print(f"   Wikipedia: ✅ Enabled")
    print(f"   YouTube: {'✅ Enabled' if not skip_youtube else '⏭️  Skipped'}")
    print(f"   Google Trends: {'✅ Enabled' if not skip_trends else '⏭️  Skipped'}")
    print(f"   Force re-scrape: {'✅ Yes' if args.force else '❌ No (will use existing files)'}")
    
    if skip_trends:
        print("\n💡 TIP: To include Google Trends, run:")
        print("   python run_all_scrapers.py --include-google-trends")
    
    if not args.force:
        print("\n💡 TIP: To force re-scraping existing files, run:")
        print("   python run_all_scrapers.py --force")
    
    input("\nPress ENTER to continue or Ctrl+C to cancel...")
    
    run_all_scrapers(
        skip_youtube=skip_youtube,
        skip_google_trends=skip_trends,
        force_rescrape=args.force
    )