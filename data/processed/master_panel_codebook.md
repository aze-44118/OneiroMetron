# Master Panel Dataset Codebook

**Generated:** 2025-12-02 18:08:53

**Observations:** 1,040

**Variables:** 94

**Date Range:** 2015-01-01 to 2024-12-31

---

## Variable Definitions

### Identifiers

- **sport** (object): 26 unique values
- **date** (datetime64[ns]): 40 unique values
- **quarter** (object): 40 unique values
- **year** (int32): 10 unique values
- **quarter_num** (int32): 4 unique values

### Dependent Variables

- **equipment_interest** (float64): 999 unique values
- **log_equipment_interest** (float64): 994 unique values

### Independent Variables

- **behavioral_interest** (float64): 989 unique values
- **log_behavioral_interest** (float64): 983 unique values
- **youtube_engagement** (float64): 370 unique values
- **log_youtube_engagement** (float64): 370 unique values
- **youtube_comments** (float64): 190 unique values
- **youtube_active_videos** (float64): 31 unique values

### Moderator

- **barrier_index** (float64): 23 unique values
- **barrier_category** (category): 3 unique values

### Interactions

- **youtube_x_barrier** (float64): 454 unique values
- **log_youtube_x_barrier** (float64): 455 unique values

### Controls

- **camping_dummy** (int64): 2 unique values
- **covid_period** (int64): 2 unique values
- **time_trend** (int64): 35 unique values

---

## Data Sources

- **Google Trends Equipment:** Monthly search indices (2015-2024)
- **Google Trends Behavioral:** Monthly search indices (2015-2024)
- **YouTube Engagement:** Quarterly comment-based metrics
- **YouTube Views:** Quarterly estimated views
- **Barrier Index:** Sport difficulty scores (constant)
- **FRED:** Economic indicators (quarterly)
- **Wikipedia:** Pageviews (quarterly)

---

## Notes

- All monetary values in USD
- Log transformations use ln(x + 0.1) to handle zeros
- Fixed effects dummies use first category as baseline
- Panel structure: Balanced (24 sports × 40 quarters)
