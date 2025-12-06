"""
Scrapers package for outdoor sports data collection.
"""

from .google_trends_scraper import GoogleTrendsScraper
from .youtube_scraper import YouTubeScraper
from .reddit_scraper import RedditScraper
from .wikipedia_scraper import WikipediaScraper

__all__ = [
    'GoogleTrendsScraper',
    'YouTubeScraper', 
    'RedditScraper',
    'WikipediaScraper'
]