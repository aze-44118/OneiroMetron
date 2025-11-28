"""
Configuration file for empirical analysis
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for empirical analysis"""
    
    # Directory structure
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    OUTPUTS_DIR = BASE_DIR / 'outputs'
    TABLES_DIR = OUTPUTS_DIR / 'tables'
    FIGURES_DIR = OUTPUTS_DIR / 'figures'
    
    # Create directories if they don't exist
    for dir_path in [PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Data files
    YOUTUBE_FILE = RAW_DATA_DIR / 'youtube_sports_data_quarterly.csv'
    GOOGLE_TRENDS_FILE = RAW_DATA_DIR / 'google_trend.csv'
    BARRIER_INDEX_FILE = RAW_DATA_DIR / 'barrer_index.csv'
    IMDB_FILE = RAW_DATA_DIR / 'imdb_data.csv'
    FRED_FILE = RAW_DATA_DIR / 'fred.csv'
    
    # Analysis parameters
    START_YEAR = 2015
    END_YEAR = 2024
    
    # Barrier categories
    BARRIER_LOW_THRESHOLD = 3.0
    BARRIER_HIGH_THRESHOLD = 6.0
    
    # Search term to sport mapping
    SEARCH_TERM_MAPPING = {
        'buy kite': 'kitesurfing',
        'kiteboarding equipment': 'kitesurfing',
        'kite surf gear': 'kitesurfing',
        'buy paraglider': 'paragliding',
        'paragliding equipment': 'paragliding',
        'paraglider for sale': 'paragliding',
        'climbing gear': 'rock climbing',
        'buy climbing harness': 'rock climbing',
        'climbing equipment': 'rock climbing',
        'buy surfboard': 'surfing',
        'surf gear': 'surfing',
        'surfboard for sale': 'surfing',
        'buy snowboard': 'snowboarding',
        'snowboard equipment': 'snowboarding',
        'snowboard gear': 'snowboarding',
        'buy mountain bike': 'mountain biking',
        'MTB gear': 'mountain biking',
        'mountain bike for sale': 'mountain biking',
        'trail running shoes': 'trail running',
        'buy running gear': 'trail running',
        'buy tent': 'camping',
        'camping equipment': 'camping',
        'camping gear': 'camping',
        'hiking boots': 'hiking',
        'hiking gear': 'hiking',
        'buy wing foil': 'wingfoiling',
        'wingfoil equipment': 'wingfoiling',
    }
    
    # Sport name standardization
    SPORT_NAME_MAPPING = {
        # YouTube names (lowercase)
        'climbing': 'rock climbing',
        # Barrier Index names (Title Case)
        'Rock Climbing': 'rock climbing',
        'Hiking': 'hiking',
        'Trail Running': 'trail running',
        'Camping': 'camping',
        'Mountain Biking': 'mountain biking',
        'Surfing': 'surfing',
        'Snowboarding': 'snowboarding',
        'Kitesurfing': 'kitesurfing',
        'Paragliding': 'paragliding',
        'Wingfoiling': 'wingfoiling',
        'BASE Jumping': 'base jumping',
        'Hang Gliding': 'hang gliding',
        'Skydiving': 'skydiving',
    }
    
    # Major documentaries (IMDb > 7.5 AND streaming/theatrical)
    MAJOR_DOC_RATING_THRESHOLD = 7.5
    
    # Regression parameters
    CLUSTER_VAR = 'sport'
    SIGNIFICANCE_LEVELS = {
        'p001': 0.01,
        'p01': 0.05,
        'p05': 0.10
    }
    
    # IV weak instrument threshold (Stock & Yogo, 2005)
    WEAK_INSTRUMENT_THRESHOLD = 10.0
    
    # Figure parameters
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    FIGURE_SIZE = (12, 8)