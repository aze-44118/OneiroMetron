# YouTube Content Marketing & Barrier to Entry
## The Aspirational Dreamers Effect

**Author:** Henri Boissonnas  
**Institution:** John Cabot University  
**Course:** Empirical Methods (EC302)  
**Date:** December 2024  

---

## 📄 Abstract

This repository contains the complete replication package for the paper 
"YouTube Content Marketing and Barrier to Entry: The Aspirational Dreamers Effect."

**Key Finding:** YouTube content marketing effectiveness is negatively moderated 
by barrier to entry (β₃ = -0.43, p < 0.001). High-barrier sports create 
"aspirational dreamers" rather than converting viewers to participants.

---

## 📊 Repository Structure

```
├── src/                          # Main analysis pipeline
│   ├── 01_data_merging.py       # Merge raw data → master_panel.csv
│   ├── 02_eda.py                # Exploratory Data Analysis
│   └── 03_regressions.py        # Regression analysis (main results)
│
├── scrapers/                     # Data collection scripts
│   ├── barrier_index_calculator.py
│   ├── google_trends_scraper.py
│   └── youtube/                  # YouTube-specific scrapers
│
├── data/
│   ├── raw/                      # Raw scraped data
│   └── processed/                # master_panel.csv (final dataset)
│
└── analysis/
    ├── tables/                   # Regression results
    └── figures/                  # Main paper figures
```

---

## 🚀 Reproduction Instructions

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### Step 1: Configure API Keys (Optional)

If you want to re-scrape data:

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Note:** Pre-scraped data is included in `data/raw/`, so scraping is optional.

### Step 2: Run Analysis Pipeline

```bash
# Merge raw data into master panel
python src/01_data_merging.py

# Generate EDA outputs
python src/02_eda.py

# Run regressions (main results)
python src/03_regressions.py
```

### Expected Outputs

- **Tables:** `analysis/tables/regression_results.tex`
- **Main Figure:** `analysis/figures/figure_marginal_effects.png`
- **Summary:** `analysis/regression_results_summary.md`

---

## 📈 Key Results

**Hypothesis:** Barrier to entry negatively moderates YouTube content effectiveness.

**Finding:** β₃ = -0.43, p < 0.001 ✅ STRONGLY SUPPORTED

**Crossover Point:** Barrier ≈ 4.29
- Below 4.29: YouTube drives conversion
- Above 4.29: YouTube creates aspirational dreamers

---

## 📊 Data Documentation

Detailed documentation of the dataset structure and provenance.

### Data Tree
```
data/
├── master_keywords_data.csv        # Keywords base
├── raw/                            # Raw collected data
│   ├── barrier_index_results.csv   # Barrier index by sport
│   ├── fred.csv                    # Economic data (Unemployment, Gas)
│   ├── google_trends_behavioral.csv # "Interest/Learning" trends
│   ├── google_trends_equipment.csv  # "Purchase Intent" trends
│   ├── imdb_data.csv               # Documentary data (Free Solo, etc.)
│   ├── wikipedia_pageviews.csv     # Wikipedia pageviews
│   ├── youtube_channel.csv         # YouTube channels tracked
│   ├── youtube_videos.csv          # Metadata for ~3000 videos
│   ├── youtube_quarterly_engagement.csv # Quarterly aggregated engagement
│   └── youtube_comments/           # Raw comments folders
└── processed/
    └── master_panel.csv            # Final merged panel for analysis
```

### Detailed File Description

#### 1. YouTube Data (`data/raw/youtube_quarterly_engagement.csv`)
Main independent variable ($X_{it}$) aggregated at Sport-Quarter level.
- **Frequency**: Quarterly (2015-2024)
- **Key**: `sport`, `year`, `quarter`
- **Metric**: `engagement_intensity` = `total_comments / active_videos` (Weighted)

#### 2. Google Trends (`data/raw/google_trends_equipment.csv`)
Search indices for purchase intention (Dependent Variable $Y_{it}$).
- **Keywords**: e.g., "buy running shoes", "climbing gear sale"
- **Index**: 0-100 normalized

#### 3. Structural Data (`data/raw/barrier_index_results.csv`)
Composite index measuring sport accessibility.
- **Dimensions**: Cost, Learning Time, Geography, Injury Risk.
- **Score**: 1 (Easy) to 10 (Hard).

---

## 🛠 Scraper Pipeline

The project uses a **weighted multi-channel scraping strategy** to ensure data robustness.

### Architecture
1.  **Video Scraping** (`youtube_video_scraper.py`): Collects top 100 videos/channel.
2.  **Comment Scraping** (`youtube_comment_scraper.py`): Collects all comments with smart prioritization.
3.  **Aggregation** (`youtube_aggregator.py`): Aggregates to quarterly level using view-weighted averaging across channels to avoid bias from single large channels.

### Quality Validation
- **Coverage**: 24 Sports, 40 Quarters (2015-2024).
- **Correlation**: Low correlation between engagement and barrier index (r = -0.24), validation of independence.

---

## 📚 Citation

If you use this data or code, please cite:

```bibtex
@misc{boissonnas2024youtube,
  author = {Boissonnas, Henri},
  title = {YouTube Content Marketing and Barrier to Entry: The Aspirational Dreamers Effect},
  year = {2024},
  institution = {John Cabot University},
  url = {https://github.com/yourusername/repo}
}
```

---

## 📄 License

- **Code:** MIT License
- **Data:** CC BY 4.0

---

## 📧 Contact

Henri Boissonnas  
Email: henri.boissonnas@students.johncabot.edu  