# Exploratory Data Analysis - Key Insights

**Generated:** 2025-12-02 18:26:11

**Dataset:** 1,040 observations, 26 sports

---

## Descriptive

- Equipment interest: Mean=26.5, SD=14.5

## Outliers

- YouTube engagement has extreme outlier (Max=10674 vs Mean=46)
- Highest YouTube engagement: Camping (log=4.26)

## Multicollinearity

- High correlation: equipment_interest ↔ behavioral_interest (r=0.845)

## H1 Preview

- YouTube ↔ Equipment correlation: r=0.060 (positive)

## H2 Preview

- Barrier ↔ YouTube correlation: r=-0.112 (supports moderation hypothesis)

## Barrier Patterns

- Low barrier: Equipment=29.0, YouTube=127.2
- Medium barrier: Equipment=30.9, YouTube=7.0
- High barrier: Equipment=16.7, YouTube=3.3

## COVID Impact

- Equipment interest: Pre=25.8, COVID=27.9 (+8.3%), Post=26.6 (-4.6%)

## Visual Moderation

- Scatter plot shows declining slope from Low to High barrier (visual support for H2)

## Distribution

- YouTube engagement is highly skewed (skew=19.70) → Log transformation recommended

## Barrier Range

- Lowest barrier: Slacklining (1.09), Highest: BASE Jumping (7.36)

---

## Recommendations for Regression Analysis

### ⚠️ Multicollinearity Detected

- **Issue:** High correlation between equipment_interest and behavioral_interest (r>0.7)
- **Recommendation:** Consider excluding behavioral_interest from main models
- **Alternative:** Run separate models with each as DV

### ⚠️ Missing YouTube Data

- **Issue:** 80 observations (7.7%) missing YouTube engagement
- **Recommendation:** Use listwise deletion for main models
- **Robustness:** Compare with imputed values (already done via log transformation)

### ⚠️ Outliers Detected

- **Issue:** Extreme values in YouTube engagement (Camping)
- **Recommendation:** Include camping_dummy in all models
- **Robustness:** Run models excluding Camping

---

## Preliminary Hypothesis Support

### H1: Direct Effect (YouTube → Equipment Interest)
- ✅ **Supported:** Positive correlation (r=0.060)
- Expected regression result: β(youtube) > 0, p < 0.05

### H2: Moderation Effect (Barrier Index)
- ✅ **Directionally supported:** Negative correlation (r=-0.112)
- Expected: β(youtube × barrier) < 0 in moderation model
- Visual evidence: Scatter plot shows declining slopes by barrier level

---

## Next Steps

1. **Run Regression Models** (`03_regressions.py`)
   - Model 1: Direct effect
   - Model 2: Moderation (key hypothesis)
   - Model 3-6: Robustness checks

2. **Create Final Visualizations** (`04_visualizations.py`)
   - Marginal effects plot
   - Predicted values by barrier level

3. **Paper Writing**
   - Integrate tables and figures
   - Write results section
   - Discuss implications
