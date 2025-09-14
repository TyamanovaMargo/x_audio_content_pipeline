#!/usr/bin/env python3

"""
Metadata-based Video Voice Filter: quick voice-content detection by keywords and video metadata.
Requires: yt-dlp, pandas
"""

import argparse
import json
import pandas as pd
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

try:
    import yt_dlp
    LIBS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing yt-dlp library: {e}")
    LIBS_AVAILABLE = False

class MetadataVoiceDetector:
    """Simple voice-content detection by metadata/keywords."""
    def __init__(self, config_path: str = "config.json"):
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.voice_keywords = self.config.get('voice_keywords', [
            'podcast', 'interview', 'talk', 'speech', 'conversation', 'discussion',
            'commentary', 'review', 'tutorial', 'explanation', 'vlog', 'story',
            'news', 'debate', 'presentation', 'lecture', 'monologue', 'dialogue',
            'аудио', 'подкаст', 'интервью', 'разговор', 'рассказ', 'лекция'
        ])
        self.non_voice_keywords = self.config.get('non_voice_keywords', [
            'music', 'song', 'instrumental', 'beat', 'remix', 'cover', 'concert',
            'performance', 'dance', 'dj set', 'mix', 'compilation', 'ambient',
            'музыка', 'песня', 'инструментал', 'концерт', 'танец'
        ])
        self.voice_categories = self.config.get('voice_categories', [
            'News & Politics', 'Education', 'Science & Technology', 'People & Blogs',
            'Entertainment', 'Gaming', 'Comedy', 'Sports', 'Film & Animation'
        ])
        self.logger.info("✅ MetadataVoiceDetector initialized")

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # 🔧 FIXED: Renamed method back to original name
    def detect_audio_content(self, links: List[Dict]) -> List[Dict]:
        if not LIBS_AVAILABLE:
            self.logger.error("❌ yt-dlp library not available")
            return []
        results = []
        for link in links:
            url = link.get("url")
            username = link.get("username", "unknown")
            platform = link.get("platform_type", "unknown")
            self.logger.info(f"🔍 Analyzing metadata for {username} ({platform})")
            try:
                metadata_result = self._analyze_video_metadata(url, username)
                results.append({
                    "username": username,
                    "url": url,
                    "platform_type": platform,
                    **metadata_result
                })
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {url}: {e}")
                results.append({
                    "username": username,
                    "url": url,
                    "platform_type": platform,
                    "voice_detected": False,
                    "voice_probability": 0.0,
                    "detection_method": "error",
                    "error": str(e)
                })
        return results


    def _analyze_video_metadata(self, url: str, username: str) -> Dict:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            'writeinfojson': False,
            'socket_timeout': 30,
            'retries': 2
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '').lower()
            description = info.get('description', '').lower()
            tags = [tag.lower() for tag in info.get('tags', [])] if info.get('tags', []) else []
            category = info.get('category', '')
            duration = info.get('duration', 0)
            uploader = info.get('uploader', '').lower()
        return self._calculate_voice_probability(
            title, description, tags, category, duration, uploader, username
        )

    def _calculate_voice_probability(self, title, description, tags, category, duration, uploader, username):
        score = 0.0
        factors = []
        methods = []
        # Keywords in title
        title_score = self._keyword_score(title)
        if title_score:
            score += title_score * 0.4
            factors.append(f"title({title_score:.2f})")
            methods.append("title_keywords")
        # Keywords in description
        desc_score = self._keyword_score(description)
        if desc_score:
            score += desc_score * 0.2
            factors.append(f"description({desc_score:.2f})")
            methods.append("desc_keywords")
        # Tags
        tag_score = self._keyword_score(' '.join(tags))
        if tag_score:
            score += tag_score * 0.2
            factors.append(f"tags({tag_score:.2f})")
            methods.append("tags_keywords")
        # Category heuristic
        if category in self.voice_categories:
            score += 0.15
            factors.append(f"category({category})")
            methods.append("category")
        # Duration
        if duration > 50:
            score += 0.1
            factors.append('duration')
            methods.append('duration')
        voice_detected = score >= 0.32
        confidence = "high" if score > 0.7 else "medium" if score > 0.5 else "low"
        self.logger.info(f"Result ({username}): Score={score:.2f} confidence={confidence} factors={factors}")
        return {
            "voice_detected": voice_detected,
            "voice_probability": min(1.0, score),
            "audio_confidence": confidence,
            "detection_method": "+".join(methods),
            "confidence_factors": factors,
            "metadata_available": True
        }

    def _keyword_score(self, text: str):
        if not text:
            return 0.0
        voice_count = sum(kw in text for kw in self.voice_keywords)
        non_voice_count = sum(kw in text for kw in self.non_voice_keywords)
        if non_voice_count > voice_count:
            return 0.0
        elif voice_count > 0:
            return min(0.6, 0.3 + voice_count * 0.1)
        else:
            return 0.05

class ChannelVideoFetcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_latest_video(self, channel_url: str, platform: str) -> Optional[str]:
        try:
            if platform == 'youtube':
                return self._get_youtube_latest(channel_url)
            elif platform == 'twitch':
                return self._get_twitch_latest(channel_url)
            elif platform == 'tiktok':
                return self._get_tiktok_latest(channel_url)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error fetching from {channel_url}: {e}")
            return None

    def _get_youtube_latest(self, channel_url: str) -> Optional[str]:
        if not channel_url.endswith('/videos'):
            if '/@' in channel_url:
                channel_url = channel_url.split('?')[0].rstrip('/') + '/videos'
            elif '/channel/' in channel_url or '/c/' in channel_url:
                channel_url = channel_url.rstrip('/') + '/videos'
        ydl_opts = {'quiet': True, 'extract_flat': True, 'playlistend': 1}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(channel_url, download=False)
            if playlist_info and 'entries' in playlist_info:
                first_entry = next(iter(playlist_info['entries']), None)
                if first_entry and first_entry.get('id'):
                    return f"https://www.youtube.com/watch?v={first_entry['id']}"
        return None

    def _get_twitch_latest(self, channel_url: str) -> Optional[str]:
        if '/videos' not in channel_url:
            channel_url = channel_url.rstrip('/') + '/videos'
        ydl_opts = {'quiet': True, 'extract_flat': True, 'playlistend': 1}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(channel_url, download=False)
            if playlist_info and 'entries' in playlist_info:
                first_entry = next(iter(playlist_info['entries']), None)
                if first_entry and first_entry.get('url'):
                    return first_entry['url']
        return None

    def _get_tiktok_latest(self, channel_url: str) -> Optional[str]:
        ydl_opts = {'quiet': True, 'extract_flat': True, 'playlistend': 1}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(channel_url, download=False)
            if playlist_info and 'entries' in playlist_info:
                first_entry = next(iter(playlist_info['entries']), None)
                if first_entry and first_entry.get('url'):
                    return first_entry['url']
        return None

class SimpleVideoVoiceFilter:
    def __init__(self, config_path: str = "config.json"):
        self._setup_logging()
        self.voice_detector = MetadataVoiceDetector(config_path)
        self.video_fetcher = ChannelVideoFetcher()
        self.min_voice_probability = 0.32

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_channels_from_csv(self, input_file: str) -> List[Dict]:
        try:
            df = pd.read_csv(input_file)
            channels = []
            for _, row in df.iterrows():
                if pd.isna(row.get("url")) or pd.isna(row.get("platform_type")):
                    continue
                platform = str(row["platform_type"]).lower()
                if platform not in ["youtube", "twitch", "tiktok"]:
                    continue
                url = str(row["url"])
                url_type = str(row.get("url_type", "unknown")).lower()
                channels.append({
                    "original_url": url,
                    "platform": platform,
                    "username": str(row.get("username", "unknown")),
                    "title": str(row.get("title", "unknown")),
                    "url_type": url_type,
                    "is_direct_video": url_type == "video"
                })
            return channels
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return []

    def process_channels(self, channels: List[Dict]) -> List[Dict]:
        print(f"\n🎬 Processing {len(channels)} items with metadata analysis...")
        voice_items = []
        for i, ch in enumerate(channels, 1):
            print(f"\n📺 [{i}/{len(channels)}] {ch['username']} ({ch['platform'].upper()})")
            if ch['is_direct_video']:
                video_url = ch['original_url']
                print(f"🎥 Direct video: {ch['title'][:50]}...")
            else:
                print("🔍 Fetching latest video from channel...")
                video_url = self.video_fetcher.get_latest_video(
                    ch['original_url'],
                    ch['platform']
                )
                if not video_url:
                    print("❌ No video found or channel inaccessible")
                    continue
                print(f"📹 Found latest video: {video_url[:60]}...")
            detection_data = [{
                "url": video_url,
                "platform_type": ch["platform"],
                "username": ch["username"]
            }]
            results = self.voice_detector.detect_voice_by_metadata(detection_data)
            if results and self._meets_voice_criteria(results[0]):
                result = results[0]
                voice_item = {
                    **ch,
                    "video_url": video_url,
                    "voice_probability": result.get("voice_probability", 0.0),
                    "audio_confidence": result.get("audio_confidence", "low"),
                    "detection_method": result.get("detection_method", "unknown"),
                    "confidence_factors": result.get("confidence_factors", []),
                    "timestamp": datetime.now().isoformat()
                }
                voice_items.append(voice_item)
                print(f"✅ VOICE DETECTED! Probability: {result.get('voice_probability', 0):.3f}")
                print(f"   Factors: {', '.join(result.get('confidence_factors', []))}")
            else:
                print("❌ No voice detected or criteria not met")
            time.sleep(0.5)
        print(f"\n📊 RESULTS: {len(voice_items)}/{len(channels)} channels have voice content")
        return voice_items

    def _meets_voice_criteria(self, result: Dict) -> bool:
        if result.get("detection_method") == "error":
            return False
        voice_prob = result.get("voice_probability", 0.0)
        confidence = result.get("audio_confidence", "very_low")
        return (voice_prob >= self.min_voice_probability and confidence in ["high", "medium", "low"])

    def save_results(self, voice_items: List[Dict], output_file: str):
        try:
            if voice_items:
                df = pd.DataFrame(voice_items)
                df.to_csv(output_file, index=False)
                print(f"✅ Results saved to {output_file}")
            else:
                print("❌ No voice items found")
        except Exception as e:
            self.logger.error(f"Error saving: {e}")

def main():
    if not LIBS_AVAILABLE:
        print("❌ Install required package: pip install yt-dlp pandas")
        return
    parser = argparse.ArgumentParser(description="Filter channels by voice content using metadata")
    parser.add_argument("input_file", help="Input CSV file with channels")
    parser.add_argument("-o", "--output", default="voice_results_metadata.csv", help="Output CSV file")
    parser.add_argument("-c", "--config", default="config.json", help="Configuration file")
    args = parser.parse_args()
    filter_tool = SimpleVideoVoiceFilter(args.config)
    channels = filter_tool.load_channels_from_csv(args.input_file)
    if not channels:
        print("❌ No valid channels loaded")
        return
    voice_items = filter_tool.process_channels(channels)
    filter_tool.save_results(voice_items, args.output)

if __name__ == "__main__":
    main()
