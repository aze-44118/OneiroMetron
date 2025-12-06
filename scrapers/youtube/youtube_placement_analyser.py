"""
Company Name Extraction (OpenAI API)
=========================================
Extracts the sponsoring company name from video descriptions for videos 
identified as having product placement.

Input: 
- data/processed/product_placement_flags.csv (existing flags)
- data/raw/youtube_videos.csv (source descriptions)

Output: 
- data/processed/product_placement_flags.csv (updated with 'company' column)
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path
from openai import OpenAI
import time
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Add project root to path for config
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import YOUTUBE_API_KEY

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize YouTube client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Cost tracking
COST_PER_1K_INPUT_TOKENS = 0.00015  # $0.150 per 1M tokens (gpt-4o-mini)
COST_PER_1K_OUTPUT_TOKENS = 0.0006  # $0.600 per 1M tokens
MAX_BUDGET = 2.00  # Budget for this specific task
total_cost = 0.0

def get_video_description(video_id):
    """Fetch video description from YouTube API."""
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        if response['items']:
            return response['items'][0]['snippet']['description']
    except Exception as e:
        print(f"⚠️ Failed to fetch description for {video_id}: {e}")
    return ""

def extract_company_openai(video_id, title, description, channel_name):
    """
    Use GPT-4o-mini to extract the sponsoring company name.
    """
    global total_cost
    
    # If description is missing/empty, try to fetch it
    if not description or pd.isna(description):
        print(f"   📥 Fetching description for {video_id}...")
        description = get_video_description(video_id)
        if not description:
            print(f"   ⚠️ No description available for {video_id}, skipping extraction.")
            return {'company': None, 'confidence': 0, 'cost': 0}
    
    # Check budget
    if total_cost >= MAX_BUDGET:
        raise Exception(f"⚠️ Budget limit reached (${total_cost:.2f})")
    
    # Construct prompt
    prompt = f"""Video: "{title}"
Channel: {channel_name}
Desc: {description[:1000]}

This video contains product placement or sponsorship. 
Identify the PAYING COMPANY/BRAND.
Reply ONLY JSON:
{{"company": "Company Name" (or null if unclear), "confidence": 0-100}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract sponsoring company names. Reply ONLY with JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50,
            timeout=10
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Clean markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        
        # Track cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        cost = (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS + 
                output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)
        total_cost += cost
        
        return {
            'company': result.get('company'),
            'confidence': result.get('confidence', 0),
            'cost': cost
        }
        
    except Exception as e:
        print(f"❌ Error for {video_id}: {e}")
        return {'company': None, 'confidence': 0, 'cost': 0}

def main():
    print("="*80)
    print("🔍 COMPANY NAME EXTRACTION (OpenAI GPT-4o-mini)")
    print("="*80)
    
    # Paths
    flags_path = Path('data/processed/product_placement_flags.csv')
    videos_path = Path('data/raw/youtube_videos.csv')
    
    if not flags_path.exists() or not videos_path.exists():
        print("❌ Missing input files")
        return
        
    # Load data
    print("📂 Loading data...")
    df_flags = pd.read_csv(flags_path)
    df_videos = pd.read_csv(videos_path)
    
    # Merge to get titles (and descriptions if they existed)
    # Handle missing description column
    cols_to_use = ['video_id', 'title']
    if 'description' in df_videos.columns:
        cols_to_use.append('description')
        
    df_merged = df_flags.merge(
        df_videos[cols_to_use], 
        on='video_id', 
        how='left'
    )
    
    # Ensure description column exists in merged df
    if 'description' not in df_merged.columns:
        df_merged['description'] = None
    
    # Filter for videos with placement that don't have a company yet (or force update)
    # If 'company' column doesn't exist, create it
    if 'company' not in df_flags.columns:
        df_flags['company'] = None
        
    # Identify rows to process: has_placement == 1 AND (company is null OR empty)
    # Ensure we catch both NaN and empty strings
    to_process_mask = (df_flags['has_placement'] == 1) & (
        df_flags['company'].isna() | 
        (df_flags['company'] == '') | 
        (df_flags['company'].astype(str).str.strip() == '')
    )
    videos_to_process = df_merged[to_process_mask].copy()
    
    print(f"📊 Found {len(videos_to_process)} sponsored videos needing company extraction")
    
    if len(videos_to_process) == 0:
        print("✅ All sponsored videos already have company names.")
        return

    # Batch fetch missing descriptions
    missing_desc_mask = videos_to_process['description'].isna() | (videos_to_process['description'] == '')
    missing_desc_ids = videos_to_process.loc[missing_desc_mask, 'video_id'].tolist()
    
    if missing_desc_ids:
        print(f"📥 Batch fetching descriptions for {len(missing_desc_ids)} videos...")
        
        # Process in batches of 50
        batch_size = 50
        fetched_count = 0
        
        for i in range(0, len(missing_desc_ids), batch_size):
            batch_ids = missing_desc_ids[i:i+batch_size]
            try:
                request = youtube.videos().list(
                    part="snippet",
                    id=','.join(batch_ids)
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    vid = item['id']
                    desc = item['snippet']['description']
                    # Update in videos_to_process
                    videos_to_process.loc[videos_to_process['video_id'] == vid, 'description'] = desc
                    
                fetched_count += len(batch_ids)
                print(f"   Fetched {fetched_count}/{len(missing_desc_ids)} descriptions...", end='\r')
                
            except Exception as e:
                print(f"\n⚠️ Error fetching batch: {e}")
                
        print("\n✅ Finished fetching descriptions")

    # Process
    print(f"💰 Budget: ${MAX_BUDGET:.2f}")
    
    try:
        for idx, row in tqdm(videos_to_process.iterrows(), total=len(videos_to_process)):
            # Skip if still no description (deleted video?)
            if pd.isna(row['description']) or not row['description']:
                continue
                
            res = extract_company_openai(
                row['video_id'],
                row.get('title', ''),
                row['description'],
                row.get('channel_name', '')
            )
            
            # Update main dataframe immediately (in memory)
            mask = df_flags['video_id'] == row['video_id']
            df_flags.loc[mask, 'company'] = res['company']
            
            # Rate limiting
            time.sleep(0.1)
            
            # Check budget
            if total_cost >= MAX_BUDGET:
                print("\n⚠️ Budget limit reached!")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        
    # Save results
    print(f"\n💾 Saving updates to {flags_path}...")
    df_flags.to_csv(flags_path, index=False)
    
    print(f"\n💰 Total spent: ${total_cost:.4f}")
    print("✅ Done")

if __name__ == '__main__':
    main()
