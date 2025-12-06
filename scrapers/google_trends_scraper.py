"""
Google Trends Scraper
Fetches equipment and behavioral keyword trends for all sports.
"""

import pandas as pd
import time
from datetime import datetime
from pytrends.request import TrendReq
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    KEYWORDS_FILE, RAW_DATA_DIR, GT_TIMEFRAME, GT_GEO,
    RATE_LIMIT_DELAY, MAX_RETRIES
)

class GoogleTrendsScraper:
    """Scraper for Google Trends data"""
    
    def __init__(self):
        self.pytrend = TrendReq(hl='en-US', tz=360)
        self.keywords_df = pd.read_csv(KEYWORDS_FILE)
        print(f"✅ Loaded {len(self.keywords_df)} keywords from master file")
    
    def fetch_trends_batch(self, keywords, max_retries=MAX_RETRIES):
        """Fetch trends data for a batch of keywords with retry logic"""
        for attempt in range(max_retries):
            try:
                self.pytrend.build_payload(keywords, timeframe=GT_TIMEFRAME, geo=GT_GEO)
                data = self.pytrend.interest_over_time()
                
                if not data.empty:
                    data = data.drop('isPartial', axis=1, errors='ignore')
                    return data
                
                return pd.DataFrame()
                
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    print(f"⚠️  Rate limited. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error fetching {keywords}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def fetch_keyword_series(self, keyword):
        """Fetch interest over time for a single keyword."""
        data = self.fetch_trends_batch([keyword])
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index().rename(columns={'date': 'date', keyword: 'index'})
        data = data[['date', 'index']]
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        return data
    
    def scrape_by_type(self, keyword_type):
        """Scrape all keywords of a specific type (equipment or behavioral)"""
        print(f"\n{'='*80}")
        print(f"SCRAPING: {keyword_type.upper()} KEYWORDS")
        print(f"{'='*80}\n")
        
        type_keywords = self.keywords_df[self.keywords_df['type'] == keyword_type]
        if type_keywords.empty:
            print(f"⚠️  No keywords found for type '{keyword_type}'")
            return pd.DataFrame()

        results = []
        total_entries = len(type_keywords)
        for idx, row in type_keywords.reset_index(drop=True).iterrows():
            sport = row['sport']
            keyword = row['keyword']
            print(f"[{idx + 1}/{total_entries}] {sport} → '{keyword}'")
            series = self.fetch_keyword_series(keyword)
            if series.empty:
                print(f"  ⚠️  No data for '{keyword}'")
            else:
                series['sport'] = sport
                series['keyword'] = keyword
                results.append(series[['sport', 'keyword', 'date', 'index']])
                print(f"  ✅ {len(series)} points")
            time.sleep(RATE_LIMIT_DELAY)
        
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
    
    def run(self):
        """Run the complete scraping process"""
        print("\n🚀 GOOGLE TRENDS SCRAPER STARTING")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        start_time = time.time()
        
        # Scrape equipment keywords
        equipment_df = self.scrape_by_type('equipment')
        if not equipment_df.empty:
            output_file = RAW_DATA_DIR / 'google_trends_equipment.csv'
            equipment_df.to_csv(output_file, index=False)
            print(f"\n✅ Equipment data saved: {output_file}")
            print(f"   → {len(equipment_df)} observations")
        else:
            print("\n⚠️  No equipment data collected")
        
        # Scrape behavioral keywords
        behavioral_df = self.scrape_by_type('behavioral')
        if not behavioral_df.empty:
            output_file = RAW_DATA_DIR / 'google_trends_behavioral.csv'
            behavioral_df.to_csv(output_file, index=False)
            print(f"\n✅ Behavioral data saved: {output_file}")
            print(f"   → {len(behavioral_df)} observations")
        else:
            print("\n⚠️  No behavioral data collected")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print("🎉 SCRAPING COMPLETE!")
        
        return equipment_df, behavioral_df

if __name__ == '__main__':
    scraper = GoogleTrendsScraper()
    scraper.run()