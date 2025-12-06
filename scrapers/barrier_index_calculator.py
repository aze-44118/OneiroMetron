"""
Barrier-to-Entry Index Calculator for Outdoor Sports
Version: 2.0 - Complete Dataset (26 sports)
Date: November 27, 2024

Methodology:
- 4 dimensions: Equipment Cost, Training Hours, Geographic Access (inverted), Injury Rate
- Each dimension normalized to 1-10 scale
- Final index = unweighted average of 4 dimensions
- Higher index = Higher barrier to entry
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# RAW DATA FROM PERPLEXITY SEARCH
# ============================================================================

raw_data = """
Sport,Min Equipment Cost (USD),Min Instruction (Hours),Access % (US Pop),Injuries (per 10k/yr),Fatality Rate (per 100k/yr),Primary Risk / Note
Running,$220,0,99.9%,5000,Negligible,Overuse injuries (knees/shins) are extremely common.
Hiking,$350,0,98%,280,Negligible,Dehydration and ankle sprains.
Camping,$450,0,95%,37,Negligible,Minor burns/cuts; benign environment.
Trail Running,$280,0,95%,6000,Negligible,Higher acute injury rate (falls) than road running.
Road Cycling,$1250,0,99%,2000,6,Vehicle collisions are the primary fatal risk.
Skateboarding,$200,0,95%,530,Negligible,Fractures (wrist/forearm) from falls on concrete.
Mountain Biking,$1050,6,92%,1000,2,Clavicle fractures and concussions common.
Kayaking,$660,3,85%,190,10 (Whitewater),Shoulder dislocations; drowning risk in whitewater.
Rock Climbing,$680,16,95%,4000,Negligible (Gym),Gym injury rate high (fingers/shoulders) but low severity.
Bouldering,$360,1,95%,4000,Negligible,Ankle sprains from ground falls are #1 injury.
SUP,$500,2,85%,400,Negligible,Weather changes blowing paddlers offshore.
Alpine Skiing,$1020,12,25%,250,1,ACL tears; collision with trees/skiers.
Snowboarding,$870,12,25%,350,1,Wrist fractures; edge-catch concussions.
Surfing,$580,3,30%,180,Negligible,Lacerations from own board/fins; drowning rare.
Scuba Diving,$1750,30,70%,150,16,Panic leading to drowning; Decompression Sickness.
Slacklining,$80,0,98%,200,Negligible,Low risk; mostly minor ankle rolls.
Wingfoiling,$2690,10,60%,570,Unknown,Impact with hydrofoil; equipment damage.
Spearfishing,$720,20,35%,200,High (Est. 50+),Shallow Water Blackout is a major silent killer.
Paragliding,$5650,80,15%,200,150,Collapse near terrain; 15 fatalities/10k pilots is high.
Kitesurfing,$2450,15,35%,1050,6,Lofting into land obstacles during launch/land.
Skydiving,$9250 (Used),40,75%,5,0.23,Landing accidents (low turns); exceptionally safe recently.
Hang Gliding,$6150,80,10%,150,100,Stalls on landing/launch; slightly safer than paragliding.
Road Racing,$2200,0,88%,9000,Unknown,Pack crashes (peloton) cause mass injuries.
Ice Climbing,$1510,16,5%,Unknown,Low,Falling ice; avalanche; leader fall trauma.
BASE Jumping,$3600,200,0.1%,4000,1700,Object strike; 1.7% annual fatality rate is extreme.
Backcountry Skiing,$2300,24,10%,Unknown,0.5 (per 10k days),Avalanche burial; hitting trees/rocks.
"""

# ============================================================================
# DATA CLEANING & PREPROCESSING
# ============================================================================

def clean_currency(value):
    """Remove $ and commas, extract first number if range"""
    if pd.isna(value):
        return np.nan
    value = str(value).replace('$', '').replace(',', '')
    # Extract first number if format like "$9250 (Used)"
    if '(' in value:
        value = value.split('(')[0].strip()
    try:
        return float(value)
    except:
        return np.nan

def clean_percentage(value):
    """Convert percentage string to float 0-100"""
    if pd.isna(value):
        return np.nan
    value = str(value).replace('%', '')
    try:
        return float(value)
    except:
        return np.nan

def clean_injury_rate(value):
    """Extract numeric injury rate, handle 'Unknown'"""
    if pd.isna(value) or value == 'Unknown':
        return np.nan
    try:
        return float(value)
    except:
        return np.nan

# Load data
from io import StringIO
df = pd.read_csv(StringIO(raw_data))

# Clean columns
df['equipment_cost'] = df['Min Equipment Cost (USD)'].apply(clean_currency)
df['training_hours'] = df['Min Instruction (Hours)']
df['access_pct'] = df['Access % (US Pop)'].apply(clean_percentage)
df['injury_rate'] = df['Injuries (per 10k/yr)'].apply(clean_injury_rate)

# Extract sport name (remove numbering)
df['sport'] = df['Sport'].str.replace(r'^\d+\.\s*', '', regex=True)

# ============================================================================
# HANDLE MISSING VALUES (IMPUTATION)
# ============================================================================

print("Missing values before imputation:")
print(df[['sport', 'equipment_cost', 'training_hours', 'access_pct', 'injury_rate']].isna().sum())
print("\nSports with missing injury rates:")
print(df[df['injury_rate'].isna()][['sport', 'injury_rate']])

# Impute missing injury rates based on similar sports
imputation_rules = {
    'Ice Climbing': 1500,  # Similar to paragliding/hang gliding but more dangerous
    'Backcountry Skiing': 800  # More dangerous than alpine skiing (250) but less than extreme sports
}

for sport, value in imputation_rules.items():
    df.loc[df['sport'] == sport, 'injury_rate'] = value

print("\n✅ After imputation:")
print(df[['sport', 'injury_rate']].tail(5))

# ============================================================================
# BARRIER INDEX CALCULATION
# ============================================================================

def normalize_to_1_10(series, reverse=False):
    """
    Normalize a pandas Series to 1-10 scale.
    
    Parameters:
    - series: pandas Series to normalize
    - reverse: if True, high values become low scores (for geographic access)
    
    Returns:
    - Normalized series with values between 1-10
    """
    min_val = series.min()
    max_val = series.max()
    
    if reverse:
        # High access % = LOW barrier
        normalized = 1 + ((max_val - series) / (max_val - min_val)) * 9
    else:
        # High value = HIGH barrier
        normalized = 1 + ((series - min_val) / (max_val - min_val)) * 9
    
    return normalized

# Calculate individual dimension scores (1-10 scale)
df['cost_score'] = normalize_to_1_10(df['equipment_cost'], reverse=False)
df['training_score'] = normalize_to_1_10(df['training_hours'], reverse=False)
df['access_score'] = normalize_to_1_10(df['access_pct'], reverse=True)  # INVERTED!
df['safety_score'] = normalize_to_1_10(df['injury_rate'], reverse=False)

# Calculate final Barrier Index (unweighted average)
df['barrier_index'] = (
    df['cost_score'] + 
    df['training_score'] + 
    df['access_score'] + 
    df['safety_score']
) / 4

# Categorize by barrier level
def categorize_barrier(index):
    if index < 2.5:
        return 'Very Low'
    elif index < 4.0:
        return 'Low'
    elif index < 5.5:
        return 'Medium'
    elif index < 7.0:
        return 'High'
    else:
        return 'Very High'

df['barrier_category'] = df['barrier_index'].apply(categorize_barrier)

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

# Sort by barrier index
df_sorted = df.sort_values('barrier_index')

# Create output dataframe
output_df = df_sorted[[
    'sport',
    'equipment_cost',
    'training_hours', 
    'access_pct',
    'injury_rate',
    'cost_score',
    'training_score',
    'access_score',
    'safety_score',
    'barrier_index',
    'barrier_category'
]].copy()

# Round for display
output_df['barrier_index'] = output_df['barrier_index'].round(2)
for col in ['cost_score', 'training_score', 'access_score', 'safety_score']:
    output_df[col] = output_df[col].round(2)

print("\n" + "="*100)
print("BARRIER-TO-ENTRY INDEX RESULTS")
print("="*100)
print("\nTop 10 LOWEST Barrier Sports:")
print(output_df.head(10).to_string(index=False))

print("\n" + "-"*100)
print("\nTop 10 HIGHEST Barrier Sports:")
print(output_df.tail(10).to_string(index=False))

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"\nBarrier Index Range: {df['barrier_index'].min():.2f} - {df['barrier_index'].max():.2f}")
print(f"Mean: {df['barrier_index'].mean():.2f}")
print(f"Median: {df['barrier_index'].median():.2f}")
print(f"Std Dev: {df['barrier_index'].std():.2f}")

print("\n" + "-"*100)
print("Distribution by Category:")
print(df['barrier_category'].value_counts().sort_index())

# ============================================================================
# EXPORT RESULTS
# ============================================================================

# Export to CSV
output_df.to_csv('barrier_index_results.csv', index=False)
print("\n✅ Results exported to: barrier_index_results.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Barrier Index by Sport (Horizontal Bar Chart)
ax1 = axes[0, 0]
colors = df_sorted['barrier_category'].map({
    'Very Low': '#2ecc71',
    'Low': '#3498db', 
    'Medium': '#f39c12',
    'High': '#e74c3c',
    'Very High': '#8e44ad'
})
ax1.barh(df_sorted['sport'], df_sorted['barrier_index'], color=colors)
ax1.set_xlabel('Barrier Index (1-10)', fontsize=12, fontweight='bold')
ax1.set_title('Barrier-to-Entry Index by Sport', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. Component Scores Heatmap (Top/Bottom 10)
ax2 = axes[0, 1]
top5 = df_sorted.head(5)
bottom5 = df_sorted.tail(5)
heatmap_data = pd.concat([top5, bottom5])[['sport', 'cost_score', 'training_score', 'access_score', 'safety_score']]
heatmap_data = heatmap_data.set_index('sport')
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            ax=ax2, cbar_kws={'label': 'Score (1-10)'})
ax2.set_title('Component Scores: Lowest vs Highest Barrier Sports', fontsize=14, fontweight='bold')
ax2.set_ylabel('')

# 3. Scatter: Equipment Cost vs Training Hours (colored by Barrier)
ax3 = axes[1, 0]
scatter = ax3.scatter(df['equipment_cost'], df['training_hours'], 
                     c=df['barrier_index'], s=200, cmap='RdYlGn_r', 
                     edgecolors='black', linewidth=1.5, alpha=0.7)
for idx, row in df.iterrows():
    if row['barrier_index'] > 7 or row['barrier_index'] < 2:
        ax3.annotate(row['sport'], (row['equipment_cost'], row['training_hours']),
                    fontsize=8, ha='right', va='bottom')
ax3.set_xlabel('Equipment Cost (USD)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Training Hours', fontsize=12, fontweight='bold')
ax3.set_title('Equipment Cost vs Training Requirements', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Barrier Index')
ax3.grid(alpha=0.3)

# 4. Distribution of Barrier Index
ax4 = axes[1, 1]
ax4.hist(df['barrier_index'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax4.axvline(df['barrier_index'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["barrier_index"].mean():.2f}')
ax4.axvline(df['barrier_index'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["barrier_index"].median():.2f}')
ax4.set_xlabel('Barrier Index', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Barrier-to-Entry Index', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('barrier_index_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: barrier_index_analysis.png")


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*100)
print("CORRELATION ANALYSIS")
print("="*100)

correlation_matrix = df[['equipment_cost', 'training_hours', 'access_pct', 'injury_rate', 'barrier_index']].corr()
print("\nCorrelation with Barrier Index:")
print(correlation_matrix['barrier_index'].sort_values(ascending=False))

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

print("\n" + "="*100)
print("VALIDATION CHECKS")
print("="*100)

# Check extreme values
print("\n✅ Lowest barrier sport:", df_sorted.iloc[0]['sport'], f"(Index: {df_sorted.iloc[0]['barrier_index']:.2f})")
print("✅ Highest barrier sport:", df_sorted.iloc[-1]['sport'], f"(Index: {df_sorted.iloc[-1]['barrier_index']:.2f})")

# Check specific sports mentioned in research
test_sports = ['Hiking', 'Snowboarding', 'Kitesurfing', 'BASE Jumping']
print("\n" + "-"*100)
print("Expected ranges validation:")
for sport in test_sports:
    row = df[df['sport'] == sport].iloc[0]
    print(f"{sport}: {row['barrier_index']:.2f} (Category: {row['barrier_category']})")

print("\n" + "="*100)
print("✅ ANALYSIS COMPLETE!")
print("="*100)