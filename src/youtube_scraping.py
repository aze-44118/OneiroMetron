"""Collect quarterly YouTube statistics for outdoor sports channels.

The script uses the YouTube Data API v3 to pull channel-level metadata,
enumerate every uploaded video (2015 Q1 - 2024 Q4), and aggregate current view
counts at the quarterly level. Results are saved to
``youtube_sports_data_quarterly.csv``.

Environment:
    - Requires a valid API key exposed via the ``GOOGLE_API_KEY`` environment
      variable or stored inside a local ``.env`` file as ``GOOGLE_API_KEY=<key>``.

Usage:
    $ python youtube_scraping.py

Notes:
    - The YouTube API does not expose historical subscriber figures; the script
      repeats the latest subscriber count for each quarter.
    - Video-level ``viewCount`` values are point-in-time snapshots (current
      totals when the script runs).
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
OUTPUT_CSV = "youtube_sports_data_quarterly.csv"
ENV_PATH = ".env"
ENV_VAR = "GOOGLE_API_KEY"
DATE_RANGE_START = dt.date(2015, 1, 1)
DATE_RANGE_END = dt.date(2024, 12, 31)
MAX_RESULTS_PER_PAGE = 50
MAX_VIDEO_IDS_PER_CALL = 50
MAX_RETRIES = 5
BACKOFF_SECONDS = 2

SPORT_CHANNELS: Dict[str, List[str]] = {
    "kitesurfing": [
        "CORE Kiteboarding",
        "North Kiteboarding",
        "Duotone Kiteboarding",
        "Ozone Kites",
        "Cabrinha Kites",
    ],
    "paragliding": [
        "Paragliding365",
        "Tucker Gott",
        "Super Fly Inc",
        "Flybubble Paragliding",
        "Greg Hamerton",
    ],
    "climbing": [
        "EpicTV",
        "Magnus Midtbø",
        "Adam Ondra",
        "Mani the Monkey",
        "Wide Boyz",
    ],
    "surfing": [
        "World Surf League",
        "Surfline",
        "Stab Magazine",
        "SURFER Magazine",
        "Koa Smith",
    ],
    "snowboarding": [
        "Snowboarder Magazine",
        "TransWorld SNOWboarding",
        "Red Gerard",
        "Shaun White",
    ],
    "mountain biking": [
        "GMBN",
        "Pinkbike",
        "Seth's Bike Hacks",
        "BKXC",
        "Skills with Phil",
    ],
    "trail running": [
        "The Ginger Runner",
        "iRunFar",
        "Billy Yang Films",
        "Sage Canaday",
    ],
    "camping": [
        "REI",
        "Outdoor Gear Review",
        "Darwin onthetrail",
        "Homemade Wanderlust",
    ],
    "hiking": [
        "Kraig Adams",
        "Homemade Wanderlust",
        "Eric Hanson",
    ],
    "wingfoiling": [
        "Slingshot Sports",
        "Duotone Wingfoil",
        "Takoon Kiteboarding",
        "Appletree Surfboards",
    ],
}


@dataclass
class ChannelContext:
    channel_id: str
    title: str
    uploads_playlist_id: str
    join_date: dt.date
    subscriber_count: Optional[int]
    video_count: Optional[int]
    total_view_count: Optional[int]


@dataclass
class VideoRecord:
    video_id: str
    published_at: dt.datetime


def load_api_key(env_var: str = ENV_VAR, env_path: str = ENV_PATH) -> str:
    """Load the API key from the environment or a .env file."""

    api_key = os.getenv(env_var)
    if api_key:
        return api_key

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == env_var:
                    return value.strip()

    raise RuntimeError(
        f"Missing API key. Set {env_var} in the environment or {env_path}."
    )


def parse_rfc3339(timestamp: str) -> dt.datetime:
    """Convert an RFC3339 timestamp string into a timezone-aware datetime."""

    if timestamp.endswith("Z"):
        timestamp = timestamp.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(timestamp)


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield successive chunks from an iterable."""

    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def call_youtube_api(
    resource: str,
    params: Dict[str, str],
    api_key: str,
    session: requests.Session,
) -> Dict:
    """Call a YouTube Data API resource with retry/backoff."""

    url = f"{YOUTUBE_API_BASE}/{resource}"
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(url, params={**params, "key": api_key}, timeout=30)
        if response.status_code == 429 or 500 <= response.status_code < 600:
            wait_time = BACKOFF_SECONDS ** attempt
            print(
                f"Rate/server error ({response.status_code}) on {resource}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)
            continue
        if response.ok:
            return response.json()

        error_message = response.text
        raise RuntimeError(
            f"YouTube API error {response.status_code} for {resource}: {error_message}"
        )

    raise RuntimeError(f"Exceeded retry budget while calling {resource}.")


def resolve_channel_context(
    channel_name: str,
    api_key: str,
    session: requests.Session,
    cache: Dict[str, ChannelContext],
) -> Optional[ChannelContext]:
    """Find the channel ID and metadata for the requested channel."""

    cache_key = channel_name.strip().lower()
    if cache_key in cache:
        return cache[cache_key]

    search_data = call_youtube_api(
        "search",
        {
            "part": "snippet",
            "q": channel_name,
            "type": "channel",
            "maxResults": "10",
        },
        api_key,
        session,
    )

    items = search_data.get("items", [])
    if not items:
        print(f"[WARN] Could not find channel for '{channel_name}'. Skipping.")
        return None

    matched_item = None
    requested_lower = channel_name.lower()
    for item in items:
        title = item["snippet"].get("channelTitle", "").lower()
        if title == requested_lower:
            matched_item = item
            break
    if matched_item is None:
        matched_item = items[0]
        print(
            f"[INFO] Using best match '{matched_item['snippet'].get('channelTitle')}' "
            f"for '{channel_name}'."
        )

    channel_id = matched_item["id"].get("channelId")
    if not channel_id:
        print(f"[WARN] Search result missing channelId for '{channel_name}'.")
        return None

    channel_data = call_youtube_api(
        "channels",
        {
            "part": "snippet,statistics,contentDetails",
            "id": channel_id,
            "maxResults": "1",
        },
        api_key,
        session,
    )

    channel_items = channel_data.get("items", [])
    if not channel_items:
        print(f"[WARN] No channel details returned for '{channel_name}'.")
        return None

    channel_info = channel_items[0]
    snippet = channel_info.get("snippet", {})
    statistics = channel_info.get("statistics", {})
    content_details = channel_info.get("contentDetails", {})

    uploads_playlist_id = (
        content_details.get("relatedPlaylists", {}).get("uploads")
    )
    if not uploads_playlist_id:
        print(f"[WARN] No uploads playlist found for '{channel_name}'.")
        return None

    join_date = parse_rfc3339(snippet.get("publishedAt")).date()
    subscriber_count = (
        int(statistics["subscriberCount"]) if "subscriberCount" in statistics else None
    )
    video_count = int(statistics["videoCount"]) if "videoCount" in statistics else None
    total_view_count = (
        int(statistics["viewCount"]) if "viewCount" in statistics else None
    )

    context = ChannelContext(
        channel_id=channel_id,
        title=snippet.get("title", channel_name),
        uploads_playlist_id=uploads_playlist_id,
        join_date=join_date,
        subscriber_count=subscriber_count,
        video_count=video_count,
        total_view_count=total_view_count,
    )

    cache[cache_key] = context
    return context


def fetch_channel_videos(
    uploads_playlist_id: str,
    api_key: str,
    session: requests.Session,
    start_date: dt.date,
    end_date: dt.date,
) -> List[VideoRecord]:
    """Fetch all videos within the target date range for a channel."""

    videos: List[VideoRecord] = []
    page_token: Optional[str] = None
    reached_earliest_needed = False

    while True:
        params = {
            "part": "contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": str(MAX_RESULTS_PER_PAGE),
        }
        if page_token:
            params["pageToken"] = page_token

        data = call_youtube_api("playlistItems", params, api_key, session)
        for item in data.get("items", []):
            content_details = item.get("contentDetails", {})
            video_id = content_details.get("videoId")
            published_at_raw = content_details.get("videoPublishedAt") or content_details.get(
                "publishedAt"
            )

            if not video_id or not published_at_raw:
                continue

            published_at = parse_rfc3339(published_at_raw)
            published_date = published_at.date()

            if published_date > end_date:
                continue
            if published_date < start_date:
                reached_earliest_needed = True
                continue

            videos.append(VideoRecord(video_id=video_id, published_at=published_at))

        page_token = data.get("nextPageToken")
        if not page_token or reached_earliest_needed:
            break

    return videos


def fetch_video_view_counts(
    video_ids: List[str],
    api_key: str,
    session: requests.Session,
) -> Dict[str, int]:
    """Fetch the current view count for each video ID."""

    views: Dict[str, int] = {}
    for chunk in chunked(video_ids, MAX_VIDEO_IDS_PER_CALL):
        params = {
            "part": "statistics",
            "id": ",".join(chunk),
            "maxResults": str(len(chunk)),
        }
        data = call_youtube_api("videos", params, api_key, session)
        for item in data.get("items", []):
            stats = item.get("statistics", {})
            video_id = item.get("id")
            if not video_id:
                continue
            views[video_id] = int(stats.get("viewCount", 0))
        time.sleep(0.1)

    return views


def quarter_key(date_value: dt.date) -> Tuple[int, str]:
    """Return the (year, 'Q#') tuple for a given date."""

    quarter_index = ((date_value.month - 1) // 3) + 1
    return date_value.year, f"Q{quarter_index}"


def quarter_schedule(
    start_date: dt.date, end_date: dt.date
) -> List[Tuple[int, str, dt.date]]:
    """Produce ordered quarterly buckets between two dates (inclusive)."""

    if start_date > end_date:
        return []

    normalized_start_month = ((start_date.month - 1) // 3) * 3 + 1
    current = dt.date(start_date.year, normalized_start_month, 1)
    schedule: List[Tuple[int, str, dt.date]] = []

    while current <= end_date:
        year, quarter_label = quarter_key(current)
        schedule.append((year, quarter_label, current))

        if current.month >= 10:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 3, 1)

    return schedule


def aggregate_quarterly_rows(
    channel_name: str,
    sport: str,
    subscriber_count: Optional[int],
    join_date: dt.date,
    videos: List[VideoRecord],
    video_views: Dict[str, int],
    global_start: dt.date,
    global_end: dt.date,
) -> List[Dict[str, object]]:
    """Aggregate per-video view counts into quarterly CSV rows."""

    if join_date > global_end:
        return []

    channel_start = max(global_start, join_date)
    schedule = quarter_schedule(channel_start, global_end)
    quarter_stats: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(
        lambda: {"views": 0.0, "videos": 0}
    )

    for video in videos:
        video_date = video.published_at.date()
        if video_date < channel_start or video_date > global_end:
            continue
        key = quarter_key(video_date)
        quarter_stats[key]["views"] += video_views.get(video.video_id, 0)
        quarter_stats[key]["videos"] += 1

    rows: List[Dict[str, object]] = []
    for year, quarter_label, _ in schedule:
        stats = quarter_stats[(year, quarter_label)]
        videos_uploaded = stats["videos"]
        total_views = int(stats["views"])
        avg_views = round(total_views / videos_uploaded, 2) if videos_uploaded else 0

        rows.append(
            {
                "channel_name": channel_name,
                "sport": sport.lower(),
                "year": year,
                "quarter": quarter_label,
                "total_views_in_quarter": total_views,
                "subscribers_end_quarter": subscriber_count
                if subscriber_count is not None
                else "",
                "videos_uploaded_in_quarter": videos_uploaded,
                "avg_views_per_video": avg_views,
            }
        )

    return rows


def process_channel(
    channel_name: str,
    sport: str,
    api_key: str,
    session: requests.Session,
    cache: Dict[str, ChannelContext],
) -> List[Dict[str, object]]:
    """Fetch and aggregate quarterly metrics for one channel."""

    context = resolve_channel_context(channel_name, api_key, session, cache)
    if not context:
        return []

    start_date = DATE_RANGE_START
    end_date = DATE_RANGE_END
    videos = fetch_channel_videos(
        context.uploads_playlist_id, api_key, session, start_date, end_date
    )
    video_views = (
        fetch_video_view_counts([video.video_id for video in videos], api_key, session)
        if videos
        else {}
    )

    print(
        f"[INFO] {channel_name} ({sport}): {len(videos)} videos in range | "
        f"Subscribers: {context.subscriber_count or 'N/A'}"
    )

    return aggregate_quarterly_rows(
        channel_name=context.title,
        sport=sport,
        subscriber_count=context.subscriber_count,
        join_date=context.join_date,
        videos=videos,
        video_views=video_views,
        global_start=start_date,
        global_end=end_date,
    )


def write_csv(rows: List[Dict[str, object]], output_path: str = OUTPUT_CSV) -> None:
    """Write aggregated rows to disk."""

    fieldnames = [
        "channel_name",
        "sport",
        "year",
        "quarter",
        "total_views_in_quarter",
        "subscribers_end_quarter",
        "videos_uploaded_in_quarter",
        "avg_views_per_video",
    ]

    rows = sorted(rows, key=lambda r: (r["channel_name"], r["year"], r["quarter"]))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote {len(rows)} rows to {output_path}.")


def main() -> None:
    api_key = load_api_key()
    session = requests.Session()
    cache: Dict[str, ChannelContext] = {}
    all_rows: List[Dict[str, object]] = []

    for sport, channels in SPORT_CHANNELS.items():
        for channel_name in channels:
            try:
                channel_rows = process_channel(
                    channel_name=channel_name,
                    sport=sport,
                    api_key=api_key,
                    session=session,
                    cache=cache,
                )
                all_rows.extend(channel_rows)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed for {channel_name} ({sport}): {exc}")
                continue

    if all_rows:
        write_csv(all_rows)
    else:
        print("[WARN] No data collected. Check API key or channel list.")


if __name__ == "__main__":
    main()

