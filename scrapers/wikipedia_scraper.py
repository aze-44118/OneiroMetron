"""
Wikipedia Pageviews Scraper
Fetches monthly pageview statistics for sport-related Wikipedia articles.
"""

import pandas as pd
import requests
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    KEYWORDS_FILE, RAW_DATA_DIR,
    RATE_LIMIT_DELAY, TIMEOUT
)

class WikipediaScraper:
    """Scraper for Wikipedia pageviews"""
    
    def __init__(self):
        self.keywords_df = pd.read_csv(KEYWORDS_FILE)
        self.wiki_pages = self.keywords_df[self.keywords_df['type'] == 'wikipedia']
        
        if len(self.wiki_pages) == 0:
            print("⚠️  No Wikipedia pages found in keywords file")
        else:
            print(f"✅ Loaded {len(self.wiki_pages)} Wikipedia pages")
    
    def get_pageviews(self, article_name, start_date='20150101', end_date='20241231'):
        """Fetch monthly Wikipedia pageviews for an article"""
        
        # Clean article name (replace spaces with underscores)
        article = article_name.replace(' ', '_')
        
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
            f"per-article/en.wikipedia/all-access/all-agents/"
            f"{article}/monthly/{start_date}/{end_date}"
        )
        
        headers = {
            'User-Agent': 'Academic Research Bot - Outdoor Sports Study'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data:
                    pageviews = []
                    
                    for item in data['items']:
                        pageviews.append({
                            'sport': article_name,
                            'article': article,
                            'date': datetime.strptime(item['timestamp'], '%Y%m%d%H').strftime('%Y-%m-%d'),
                            'pageviews': item['views']
                        })
                    
                    df = pd.DataFrame(pageviews)
                    print(f"  ✅ {article_name}: {len(df)} months, avg {df['pageviews'].mean():.0f} views/month")
                    return df
                else:
                    print(f"  ⚠️  No data for {article_name}")
                    return pd.DataFrame()
            
            elif response.status_code == 404:
                print(f"  ❌ Article not found: {article_name}")
                return pd.DataFrame()
            
            else:
                print(f"  ❌ Error {response.status_code} for {article_name}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error fetching {article_name}: {e}")
            return pd.DataFrame()
    
    def run(self):
        """Run Wikipedia pageviews scraping"""
        print("\n🚀 WIKIPEDIA SCRAPER STARTING")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        start_time = time.time()
        all_data = []
        
        total_pages = len(self.wiki_pages)
        
        for idx, row in self.wiki_pages.iterrows():
            page_num = idx + 1
            article_name = row['keyword']
            sport = row['sport']
            
            print(f"[{page_num}/{total_pages}] Fetching: {article_name}")
            
            df = self.get_pageviews(article_name)
            
            if not df.empty:
                df['sport'] = sport  # Override with sport name
                all_data.append(df)
            
            # Rate limiting (be nice to Wikimedia!)
            time.sleep(1)
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by date
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values(['sport', 'date'])
            
            output_file = RAW_DATA_DIR / 'wikipedia_pageviews.csv'
            final_df.to_csv(output_file, index=False)
            
            elapsed_time = time.time() - start_time
            
            # Stats
            total_views = final_df['pageviews'].sum()
            avg_views_per_month = final_df.groupby('sport')['pageviews'].mean()
            top_sport = avg_views_per_month.idxmax()
            
            print(f"\n✅ Wikipedia data saved: {output_file}")
            print(f"   → {len(final_df)} monthly observations")
            print(f"   → {final_df['sport'].nunique()} sports covered")
            print(f"   → Total pageviews: {total_views:,}")
            print(f"   → Top sport: {top_sport} ({avg_views_per_month[top_sport]:.0f} avg/month)")
            print(f"   → Time: {elapsed_time:.1f} seconds")
            print("🎉 SCRAPING COMPLETE!")
            
            return final_df
        else:
            print("\n⚠️  No data collected")
            return pd.DataFrame()

if __name__ == '__main__':
    scraper = WikipediaScraper()
    scraper.run()