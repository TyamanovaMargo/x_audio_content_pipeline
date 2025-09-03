import os
import subprocess
import time
import json
import pandas as pd
import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_DURATION = 30      # seconds
MAX_DURATION = 3600    # 1 hour in seconds
CHUNK_DURATION = 1800  # 30 minutes in seconds

class AudioDownloader:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_links(self, filepath: str) -> List[Dict]:
        df = pd.read_csv(filepath)
        return df.to_dict('records')

    def download_audio_for_all(self, links: List[Dict]):
        for info in links:
            username = info.get('username', 'user')
            profile_name = info.get('profile_name', '')
            url = info.get('url', '')
            logger.info(f"Processing user: {username}, profile: {profile_name}, url: {url}")
            self.process_link(url, username)

    def process_link(self, url: str, username: str):
        platform = self.determine_platform(url)
        if platform == 'twitch':
            latest_video_url, duration = self.get_latest_twitch_vod_url_and_duration(url)
        else:
            latest_video_url, duration = self.get_latest_video_url_and_duration(url)

        if not latest_video_url:
            logger.warning(f"Could not get latest video url for {username} on {platform}")
            return

        logger.info(f"Latest video duration: {duration} seconds")
        if duration < MIN_DURATION:
            logger.warning(f"Video too short ({duration}s), skipping")
            return

        download_duration = min(duration, MAX_DURATION)
        filename = self.sanitize_filename(f"{username}_audio_full.mp3")
        filepath = os.path.join(self.output_dir, filename)

        success = self.download_audio_chunk(latest_video_url, filepath, 0, download_duration)
        if success:
            logger.info(f"Downloaded full audio: {filepath}")
            return

        if duration > CHUNK_DURATION:
            logger.info(f"Full download failed or incomplete, trying chunked download (2x30min)")
            for i in range(2):
                start = i * CHUNK_DURATION
                current_duration = min(CHUNK_DURATION, duration - start)
                chunk_filename = self.sanitize_filename(f"{username}_audio_part{i+1}.mp3")
                chunk_filepath = os.path.join(self.output_dir, chunk_filename)
                success_part = self.download_audio_chunk(latest_video_url, chunk_filepath, start, current_duration)
                if success_part:
                    logger.info(f"Downloaded chunk {i+1}: {chunk_filepath}")
                else:
                    logger.warning(f"Failed to download chunk {i+1}")

    def determine_platform(self, url: str) -> str:
        if 'twitch.tv' in url:
            return 'twitch'
        elif 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'tiktok.com' in url:
            return 'tiktok'
        else:
            return 'unknown'

    def get_latest_video_url_and_duration(self, channel_url: str):
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-playlist',
            '--dump-json',
            '--playlist-items', '1',
            channel_url
        ]
        try:
            output = subprocess.check_output(cmd, text=True)
            info = json.loads(output.strip().split('\n')[0])
            video_url = info.get('webpage_url') or info.get('url')
            duration = info.get('duration', 0)
            return video_url, duration
        except Exception as e:
            logger.error(f"Failed to fetch latest video info: {e}")
            return None, 0

    def get_latest_twitch_vod_url_and_duration(self, channel_url: str):
        videos_url = channel_url.rstrip('/') + '/videos'
        cmd = [
            'yt-dlp', 
            '--dump-json', 
            '--playlist-items', '1', 
            '--quiet', 
            '--no-warnings', 
            videos_url
        ]
        try:
            output = subprocess.check_output(cmd, text=True)
            info = json.loads(output.strip().split('\n')[0]) if output else None
            if info:
                video_url = info.get('webpage_url') or info.get('url')
                duration = info.get('duration', 0)
                return video_url, duration
            return None, 0
        except Exception as e:
            logger.error(f"Failed to fetch latest Twitch VOD info: {e}")
            return None, 0

    def download_audio_chunk(self, video_url: str, output_path: str, start_sec: int, duration_sec: int) -> bool:
        try:
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '192',
                '--output', output_path,
                '--postprocessor-args', f'ffmpeg:-ss {start_sec} -t {duration_sec}',
                '--no-playlist',
                video_url,
                '--quiet'
            ]
            logger.info(f"Downloading audio segment: start={start_sec}s, duration={duration_sec}s")
            subprocess.check_call(cmd)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
                return True
            else:
                logger.warning(f"Downloaded file too small or missing: {output_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to download audio chunk: {e}")
            return False

    def sanitize_filename(self, filename: str) -> str:
        filename = re.sub(r'[^\w\s-]', '', filename).strip().lower()
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python audio_downloader.py <input_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    downloader = AudioDownloader("output_audio2")
    links = downloader.load_links(input_csv)
    downloader.download_audio_for_all(links)

