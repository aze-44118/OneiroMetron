"""Central configuration module for analysis pipeline and scrapers."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class _Settings:
    """Project-wide configuration values and file paths."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / 'data'
    RAW_DATA_DIR: Path = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'
    OUTPUTS_DIR: Path = PROJECT_ROOT / 'outputs'
    TABLES_DIR: Path = OUTPUTS_DIR / 'tables'
    FIGURES_DIR: Path = OUTPUTS_DIR / 'figures'

    # Source files
    KEYWORDS_FILE: Path = RAW_DATA_DIR / 'master_keywords.csv'
    YOUTUBE_FILE: Path = RAW_DATA_DIR / 'youtube_sports_data_quarterly.csv'
    GOOGLE_TRENDS_EQUIPMENT_FILE: Path = RAW_DATA_DIR / 'google_trends_equipment.csv'
    GOOGLE_TRENDS_BEHAVIORAL_FILE: Path = RAW_DATA_DIR / 'google_trends_behavioral.csv'
    BARRIER_INDEX_FILE: Path = RAW_DATA_DIR / 'barrier_index_results.csv'
    IMDB_FILE: Path = RAW_DATA_DIR / 'imdb_data.csv'
    WIKIPEDIA_FILE: Path = RAW_DATA_DIR / 'wikipedia_pageviews.csv'
    FRED_FILE: Path = RAW_DATA_DIR / 'fred.csv'

    # Analysis parameters
    START_YEAR: int = 2015
    END_YEAR: int = 2024
    BARRIER_LOW_THRESHOLD: float = 3.0
    BARRIER_HIGH_THRESHOLD: float = 6.0
    MAJOR_DOC_RATING_THRESHOLD: float = 7.5
    WEAK_INSTRUMENT_THRESHOLD: float = 10.0

    # Visualization settings
    FIGURE_DPI: int = 300

    # Mapping placeholders (to be customized as needed)
    SPORT_NAME_MAPPING: dict = None
    SEARCH_TERM_MAPPING: dict = None

    def __post_init__(self):
        # Ensure directories exist
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.OUTPUTS_DIR,
            self.TABLES_DIR,
            self.FIGURES_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Default mappings if none provided
        if self.SPORT_NAME_MAPPING is None:
            self.SPORT_NAME_MAPPING = {}
        if self.SEARCH_TERM_MAPPING is None:
            self.SEARCH_TERM_MAPPING = {}


Config = _Settings()

# Legacy aliases for modules that import these directly
KEYWORDS_FILE = Config.KEYWORDS_FILE
RAW_DATA_DIR = Config.RAW_DATA_DIR
PROCESSED_DATA_DIR = Config.PROCESSED_DATA_DIR
GT_TIMEFRAME = '2015-01-01 2024-12-31'
GT_GEO = 'US'


# Scraper-specific settings (used outside Config dataclass)
RATE_LIMIT_DELAY = 2
MAX_RETRIES = 3
TIMEOUT = 30
GT_TIMEFRAME = '2015-01-01 2024-12-31'
GT_GEO = 'US'

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'YOUR_YOUTUBE_API_KEY_HERE')

print("✅ Configuration module loaded")