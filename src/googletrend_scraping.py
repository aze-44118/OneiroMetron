"""
Google Trends Scraper v2.0 - SMART Training Keywords
=====================================================
REALISTIC keywords that people ACTUALLY search when considering a sport

Author: Arthus Azais de Vergeron
Date: November 27, 2024
Version: 2.0 (Smart Keywords)
"""

import pandas as pd
from pytrends.request import TrendReq
import time
from datetime import datetime
from calendar import monthrange

# ============================================================================
# SMART TRAINING KEYWORDS (10 per sport)
# ============================================================================

TRAINING_KEYWORDS_V2 = {
    'hiking': [
        'beginner hiking trails',
        'hiking for beginners',
        'how to start hiking',
        'hiking tips',
        'what to bring hiking',
        'best hiking trails near me',
        'easy hiking trails',
        'hiking gear for beginners',
        'day hike',
        'hiking safety tips'
    ],
    
    'trail running': [
        'trail running for beginners',
        'how to start trail running',
        'trail running tips',
        'road running vs trail running',
        'trail running near me',
        'trail running shoes',
        'beginner trail run',
        'ultra running training',
        'how to run trails',
        'trail running technique'
    ],
    
    'camping': [
        'camping for beginners',
        'how to camp',
        'what do I need for camping',
        'camping essentials',
        'camping checklist',
        'camping tips',
        'first time camping',
        'camping near me',
        'car camping vs backpacking',
        'camping gear list'
    ],
    
    'mountain biking': [
        'mountain biking for beginners',
        'how to start mountain biking',
        'mountain bike trails near me',
        'beginner mountain bike',
        'mountain biking tips',
        'how to ride a mountain bike',
        'mountain bike vs road bike',
        'mountain bike skills',
        'downhill mountain biking',
        'mountain bike lessons'
    ],
    
    'rock climbing': [
        'rock climbing for beginners',
        'how to start rock climbing',
        'rock climbing gym near me',
        'indoor climbing',
        'bouldering vs rock climbing',
        'rock climbing lessons',
        'climbing gym',
        'how to belay',
        'rock climbing gear',
        'beginner climbing'
    ],
    
    'snowboarding': [
        'snowboarding for beginners',
        'how to snowboard',
        'snowboard lessons',
        'learn to snowboard',
        'snowboarding tips',
        'skiing vs snowboarding',
        'snowboard lessons near me',
        'beginner snowboard',
        'snowboarding tutorial',
        'snowboard school'
    ],
    
    'surfing': [
        'surf lessons',
        'how to surf',
        'surfing for beginners',
        'learn to surf',
        'surf school',
        'surf lessons near me',
        'beginner surfboard',
        'surf camp',
        'surfing tutorial',
        'how to paddle surf'
    ],
    
    'wingfoiling': [
        'wingfoil',
        'wing foiling',
        'what is wing foiling',
        'wingsurf',
        'wing foil lessons',
        'how to wingfoil',
        'wingfoiling for beginners',
        'wing foil vs kitesurfing',
        'learn wing foiling',
        'wingfoil school'
    ],
    
    'paragliding': [
        'paragliding',
        'paragliding near me',
        'paragliding lessons',
        'learn to paraglide',
        'paragliding school',
        'paragliding course',
        'paragliding for beginners',
        'how to paraglide',
        'tandem paragliding',
        'paragliding license'
    ],
    
    'kitesurfing': [
        'kitesurfing',
        'kiteboarding',
        'kitesurf lessons',
        'learn to kitesurf',
        'kitesurfing for beginners',
        'kite school',
        'kiteboarding lessons',
        'how to kitesurf',
        'kitesurfing near me',
        'kitesurf course'
    ]
}

# Date range
START_DATE = '2015-01'
END_DATE = '2024-12'

# Output file
OUTPUT_FILE = 'training_trends_v2.csv'

# Rate limiting
DELAY_BETWEEN_REQUESTS = 2

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_timeframe(start_month, end_month):
    """Convert YYYY-MM strings into YYYY-MM-DD range"""
    start_date = f"{start_month}-01"
    end_year, end_month_num = map(int, end_month.split('-'))
    last_day = monthrange(end_year, end_month_num)[1]
    end_date = f"{end_year:04d}-{end_month_num:02d}-{last_day:02d}"
    return f"{start_date} {end_date}"

def init_pytrends():
    """Initialize pytrends request object"""
    print("Initializing Google Trends connection...")
    pytrend = TrendReq(hl='en-US', tz=360)
    return pytrend

def fetch_keyword_data(pytrend, keyword, timeframe):
    """Fetch Google Trends data for a single keyword"""
    try:
        pytrend.build_payload(
            kw_list=[keyword],
            timeframe=timeframe,
            geo='',  # Worldwide
            gprop=''  # Web search
        )
        
        data = pytrend.interest_over_time()
        
        if data.empty:
            print(f"  ⚠️  No data returned for '{keyword}'")
            return None
        
        # Clean data
        data = data.reset_index()
        data = data.rename(columns={
            'date': 'month',
            keyword: 'search_index'
        })
        
        data = data[['month', 'search_index']]
        data['month'] = data['month'].dt.strftime('%Y-%m')
        
        return data
        
    except Exception as e:
        print(f"  ✗ Error fetching '{keyword}': {e}")
        return None

# ============================================================================
# MAIN COLLECTION
# ============================================================================

def collect_all_data():
    """Main collection function - fetches all keywords"""
    pytrend = init_pytrends()
    timeframe = build_timeframe(START_DATE, END_DATE)
    
    all_data = []
    total_keywords = sum(len(keywords) for keywords in TRAINING_KEYWORDS_V2.values())
    current = 0
    
    print(f"\n🎯 COLLECTING SMART TRAINING KEYWORDS V2.0")
    print(f"Total keywords: {total_keywords} across {len(TRAINING_KEYWORDS_V2)} sports")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("="*70)
    
    for sport, keywords in TRAINING_KEYWORDS_V2.items():
        print(f"\n[{sport.upper()}] ({len(keywords)} keywords)")
        
        for keyword in keywords:
            current += 1
            print(f"  [{current}/{total_keywords}] Fetching '{keyword}'...", end=' ')
            
            data = fetch_keyword_data(pytrend, keyword, timeframe)
            
            if data is not None:
                data['search_term'] = keyword
                all_data.append(data)
                print(f"✓ ({len(data)} months)")
            else:
                print("✗ Failed")
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    if not all_data:
        print("\n✗ No data collected!")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    df = df[['search_term', 'month', 'search_index']]
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("GOOGLE TRENDS SCRAPER V2.0 - SMART TRAINING KEYWORDS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check pytrends
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("✗ ERROR: pytrends not installed!")
        print("\nInstall with:")
        print("  pip install pytrends")
        return
    
    # Collect data
    start_time = time.time()
    df = collect_all_data()
    elapsed = time.time() - start_time
    
    if df is None:
        print("\n✗ Data collection failed!")
        return
    
    # Save to CSV
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved {len(df)} rows to: {OUTPUT_FILE}")
    print(f"\nData summary:")
    print(f"  - Total rows: {len(df):,}")
    print(f"  - Unique terms: {df['search_term'].nunique()}")
    print(f"  - Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"  - Elapsed time: {elapsed/60:.1f} minutes")
    
    # Quality check
    print(f"\n📊 QUALITY METRICS:")
    avg_by_term = df.groupby('search_term')['search_index'].mean().round(1)
    print(f"  - Mean search index (all): {df['search_index'].mean():.1f}")
    print(f"  - Median search index (all): {df['search_index'].median():.1f}")
    print(f"  - Keywords with avg >20: {(avg_by_term > 20).sum()}/{len(avg_by_term)}")
    print(f"  - Keywords with avg >10: {(avg_by_term > 10).sum()}/{len(avg_by_term)}")
    print(f"  - Keywords with avg <5: {(avg_by_term < 5).sum()}/{len(avg_by_term)}")
    
    # Show top performers
    print(f"\n🏆 TOP 10 KEYWORDS:")
    top10 = avg_by_term.sort_values(ascending=False).head(10)
    for i, (term, avg) in enumerate(top10.items(), 1):
        print(f"  {i}. {term}: {avg:.1f}")
    
    # Show bottom performers
    print(f"\n⚠️ BOTTOM 10 KEYWORDS:")
    bottom10 = avg_by_term.sort_values().head(10)
    for i, (term, avg) in enumerate(bottom10.items(), 1):
        print(f"  {i}. {term}: {avg:.1f}")
    
    print("\n" + "="*70)
    print("✅ COLLECTION COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Next step: Replace old training_trends.csv with this file")
    print("   Then re-run empirical analysis!")

if __name__ == "__main__":
    main()