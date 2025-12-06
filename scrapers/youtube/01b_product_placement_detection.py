"""
Product Placement Detection (OpenAI API)
=========================================
Analyzes video descriptions to detect sponsored content using GPT-4o-mini.

Cost optimization:
- Uses gpt-4o-mini ($0.150/1M input tokens, $0.600/1M output tokens)
- Estimated cost: ~3,000 videos × ~300 tokens = ~$0.50 total
- Built-in budget protection: Stops if approaching limit

Input: data/raw/youtube_videos.csv
Output: data/processed/product_placement_flags.csv
"""

import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cost tracking
COST_PER_1K_INPUT_TOKENS = 0.00015  # $0.150 per 1M tokens
COST_PER_1K_OUTPUT_TOKENS = 0.0006  # $0.600 per 1M tokens
MAX_BUDGET = 4.00  # Leave $0.44 buffer
total_cost = 0.0


def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4


def detect_placement_openai(video_id, title, description, channel_name):
    """
    Use GPT-4o-mini to detect product placement.
    
    Returns:
    --------
    dict with detection results
    """
    global total_cost
    
    # Check budget
    if total_cost >= MAX_BUDGET:
        raise Exception(f"⚠️ Budget limit reached (${total_cost:.2f})")
    
    # Construct prompt (ULTRA CONCISE to minimize tokens)
    prompt = f"""Video: "{title}"
Channel: {channel_name}
Desc: {description[:500]}

Has product placement/sponsorship? Reply ONLY JSON:
{{"placement": true/false, "type": "sponsored/affiliate/organic", "confidence": 0-100}}"""
    
    try:
        # API call with minimal settings
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model
            messages=[
                {"role": "system", "content": "You detect product placement in YouTube videos. Reply ONLY with JSON, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Deterministic
            max_tokens=50,  # Minimal output (just JSON)
            timeout=10
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON (handle markdown code blocks)
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
            'video_id': video_id,
            'has_placement': 1 if result.get('placement', False) else 0,
            'placement_type': result.get('type', 'unknown'),
            'confidence': result.get('confidence', 50),
            'cost': cost,
            'tokens_used': input_tokens + output_tokens
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error for {video_id}: {result_text}")
        return {
            'video_id': video_id,
            'has_placement': 0,
            'placement_type': 'error',
            'confidence': 0,
            'cost': 0,
            'tokens_used': 0
        }
    
    except Exception as e:
        print(f"❌ Error for {video_id}: {e}")
        return {
            'video_id': video_id,
            'has_placement': 0,
            'placement_type': 'error',
            'confidence': 0,
            'cost': 0,
            'tokens_used': 0
        }


def process_batch(videos_df, batch_size=100, checkpoint_every=50):
    """
    Process videos in batches with checkpointing.
    
    Parameters:
    -----------
    videos_df : DataFrame
        Videos to process
    batch_size : int
        Process this many videos before stopping (cost control)
    checkpoint_every : int
        Save progress every N videos
    """
    global total_cost
    
    results = []
    checkpoint_file = Path('data/processed/placement_checkpoint.json')
    
    # Load checkpoint if exists
    if checkpoint_file.exists():
        print(f"📂 Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            results = checkpoint['results']
            total_cost = checkpoint['total_cost']
            start_idx = checkpoint['last_idx'] + 1
            print(f"   Resuming from video {start_idx} (${total_cost:.3f} spent)")
    else:
        start_idx = 0
    
    # Process videos
    print(f"\n🎬 Processing {len(videos_df)} videos (starting at {start_idx})...")
    print(f"💰 Budget: ${MAX_BUDGET:.2f} | Current: ${total_cost:.3f}\n")
    
    try:
        for idx in tqdm(range(start_idx, min(len(videos_df), start_idx + batch_size)), 
                       desc="Analyzing videos"):
            
            video = videos_df.iloc[idx]
            
            # Detect placement
            result = detect_placement_openai(
                video['video_id'],
                video.get('title', ''),
                video.get('description', ''),
                video.get('channel_name', '')
            )
            
            result['sport'] = video['sport']
            result['channel_name'] = video.get('channel_name', '')
            results.append(result)
            
            # Checkpoint
            if (idx + 1) % checkpoint_every == 0:
                checkpoint = {
                    'results': results,
                    'total_cost': total_cost,
                    'last_idx': idx
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                
                print(f"\n💾 Checkpoint saved at video {idx+1} (${total_cost:.3f})")
            
            # Budget check
            if total_cost >= MAX_BUDGET:
                print(f"\n⚠️ Budget limit reached at video {idx+1}")
                print(f"💰 Total spent: ${total_cost:.3f}")
                break
            
            # Rate limiting (conservative)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Saving checkpoint...")
        checkpoint = {
            'results': results,
            'total_cost': total_cost,
            'last_idx': idx
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        print(f"💾 Progress saved. Run again to resume.")
        raise
    
    return results


def main():
    print("="*80)
    print("🔍 PRODUCT PLACEMENT DETECTION (OpenAI GPT-4o-mini)")
    print("="*80)
    
    # Load videos
    videos_file = Path('data/raw/youtube_videos.csv')
    if not videos_file.exists():
        print(f"❌ File not found: {videos_file}")
        return
    
    df = pd.read_csv(videos_file)
    print(f"\n📹 Loaded {len(df)} videos")
    
    # Estimate cost
    avg_tokens_per_video = 300  # Conservative estimate
    estimated_cost = (len(df) * avg_tokens_per_video / 1000 * 
                     (COST_PER_1K_INPUT_TOKENS + COST_PER_1K_OUTPUT_TOKENS))
    
    print(f"\n💰 COST ESTIMATE:")
    print(f"   Total videos: {len(df)}")
    print(f"   Est. tokens/video: ~{avg_tokens_per_video}")
    print(f"   Est. total cost: ${estimated_cost:.2f}")
    print(f"   Your budget: ${MAX_BUDGET:.2f}")
    
    if estimated_cost > MAX_BUDGET:
        max_videos = int(MAX_BUDGET / estimated_cost * len(df))
        print(f"\n⚠️ Budget insufficient for all videos!")
        print(f"   Can process ~{max_videos} videos with ${MAX_BUDGET:.2f}")
        
        response = input(f"\nProcess first {max_videos} videos? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled by user")
            return
        
        df = df.head(max_videos)
    else:
        print(f"\n✅ Budget sufficient")
    
    # Process
    results = process_batch(df, batch_size=len(df))
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    total_sponsored = df_results['has_placement'].sum()
    pct_sponsored = total_sponsored / len(df_results) * 100
    
    print(f"\n📈 Overall:")
    print(f"   Videos analyzed: {len(df_results)}")
    print(f"   With product placement: {total_sponsored} ({pct_sponsored:.1f}%)")
    print(f"   Organic content: {len(df_results) - total_sponsored} ({100-pct_sponsored:.1f}%)")
    
    print(f"\n💰 Cost:")
    print(f"   Total spent: ${total_cost:.3f}")
    print(f"   Avg cost/video: ${total_cost/len(df_results):.5f}")
    print(f"   Total tokens: {df_results['tokens_used'].sum():,}")
    
    print(f"\n📊 By Placement Type:")
    type_counts = df_results['placement_type'].value_counts()
    for ptype, count in type_counts.items():
        print(f"   {ptype:15s}: {count:4d} ({count/len(df_results)*100:5.1f}%)")
    
    print(f"\n📊 By Sport (Top 10 by % Sponsored):")
    sport_summary = df_results.groupby('sport').agg({
        'has_placement': ['sum', 'count', 'mean']
    })
    sport_summary.columns = ['Sponsored', 'Total', 'Pct']
    sport_summary['Pct'] = (sport_summary['Pct'] * 100).round(1)
    sport_summary = sport_summary.sort_values('Pct', ascending=False)
    print(sport_summary.head(10).to_string())
    
    # Save results
    output_file = Path('data/processed/product_placement_flags.csv')
    df_results[['video_id', 'sport', 'channel_name', 'has_placement', 
                'placement_type', 'confidence']].to_csv(output_file, index=False)
    
    print(f"\n✅ Saved: {output_file}")
    
    # Clean up checkpoint
    checkpoint_file = Path('data/processed/placement_checkpoint.json')
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"🗑️ Cleaned up checkpoint file")
    
    print("\n" + "="*80)
    print("✅ DETECTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect product placement in YouTube videos')
    parser.add_argument('--max-budget', type=float, default=4.0,
                       help='Maximum budget in USD (default: 4.0)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Process only N videos (for testing)')
    
    args = parser.parse_args()
    
    MAX_BUDGET = args.max_budget
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print("   export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    main()