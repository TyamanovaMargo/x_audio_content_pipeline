#!/usr/bin/env python3
"""
step6_voice_sample_extractor.py - Voice Sample Extraction (MP3 Audio Downloader)

Based on the working test.py implementation with pipeline integration.
"""

import os
import subprocess
import time
import json
import pandas as pd
import re
import logging
import argparse
import sys
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MIN_DURATION = 30  # seconds
MAX_DURATION = 3600  # 1 hour in seconds
CHUNK_DURATION = 1800  # 30 minutes in seconds


class AudioDownloader:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_links(self, filepath: str) -> List[Dict]:
        df = pd.read_csv(filepath)
        return df.to_dict('records')

    def download_audio_for_all(self, links: List[Dict]) -> List[Dict]:
        """Download audio for all links and return results for pipeline integration"""
        results = []
        
        for info in links:
            username = info.get('username', 'user')
            profile_name = info.get('profile_name', '')
            url = info.get('url', '')
            
            logger.info(f"Processing user: {username}, profile: {profile_name}, url: {url}")
            
            result = self.process_link(url, username)
            if result:
                # Add original info to result
                result.update({
                    'original_username': username,
                    'original_profile_name': profile_name,
                    'original_url': url
                })
                results.append(result)
                
        return results

    def process_link(self, url: str, username: str) -> Dict:
        """Process a single link - based on working test.py logic"""
        platform = self.determine_platform(url)
        
        if platform == 'twitch':
            latest_video_url, duration = self.get_latest_twitch_vod_url_and_duration(url)
        else:
            latest_video_url, duration = self.get_latest_video_url_and_duration(url)

        if not latest_video_url:
            logger.warning(f"Could not get latest video url for {username} on {platform}")
            return None

        logger.info(f"Latest video duration: {duration} seconds")

        if duration < MIN_DURATION:
            logger.warning(f"Video too short ({duration}s), skipping")
            return None

        download_duration = min(duration, MAX_DURATION)
        filename = self.sanitize_filename(f"{username}_audio_full.mp3")
        filepath = os.path.join(self.output_dir, filename)

        success = self.download_audio_chunk(latest_video_url, filepath, 0, download_duration)

        if success:
            logger.info(f"Downloaded full audio: {filepath}")
            return {
                'username': username,
                'platform': platform,
                'filepath': filepath,
                'filename': filename,
                'duration': duration,
                'download_duration': download_duration,
                'success': True,
                'chunks': 1,
                'file_size_bytes': os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }

        # Try chunked download if full download failed and video is long enough
        if duration > CHUNK_DURATION:
            logger.info(f"Full download failed or incomplete, trying chunked download (2x30min)")
            chunk_results = []
            
            for i in range(2):
                start = i * CHUNK_DURATION
                current_duration = min(CHUNK_DURATION, duration - start)
                chunk_filename = self.sanitize_filename(f"{username}_audio_part{i+1}.mp3")
                chunk_filepath = os.path.join(self.output_dir, chunk_filename)
                
                success_part = self.download_audio_chunk(latest_video_url, chunk_filepath, start, current_duration)
                
                if success_part:
                    logger.info(f"Downloaded chunk {i+1}: {chunk_filepath}")
                    chunk_results.append({
                        'chunk_number': i+1,
                        'filepath': chunk_filepath,
                        'filename': chunk_filename,
                        'start': start,
                        'duration': current_duration,
                        'file_size_bytes': os.path.getsize(chunk_filepath) if os.path.exists(chunk_filepath) else 0
                    })
                else:
                    logger.warning(f"Failed to download chunk {i+1}")
            
            if chunk_results:
                return {
                    'username': username,
                    'platform': platform,
                    'duration': duration,
                    'success': True,
                    'chunks': len(chunk_results),
                    'chunk_results': chunk_results
                }
        
        return None

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


def save_results(results: List[Dict], output_file: str):
    """Save processing results to CSV file"""
    if not results:
        logger.warning("No results to save")
        return
        
    # Flatten results for CSV output
    flattened_results = []
    for result in results:
        base_result = {
            'username': result.get('username'),
            'platform': result.get('platform'),
            'duration': result.get('duration'),
            'success': result.get('success'),
            'chunks': result.get('chunks', 1),
            'original_username': result.get('original_username'),
            'original_profile_name': result.get('original_profile_name'),
            'original_url': result.get('original_url')
        }
        
        if 'chunk_results' in result:
            # Multiple chunks
            for chunk in result['chunk_results']:
                chunk_result = base_result.copy()
                chunk_result.update({
                    'filepath': chunk['filepath'],
                    'filename': chunk['filename'],
                    'chunk_number': chunk['chunk_number'],
                    'start_time': chunk['start'],
                    'chunk_duration': chunk['duration'],
                    'file_size_bytes': chunk.get('file_size_bytes', 0)
                })
                flattened_results.append(chunk_result)
        else:
            # Single file
            base_result.update({
                'filepath': result.get('filepath'),
                'filename': result.get('filename'),
                'chunk_number': 1,
                'start_time': 0,
                'chunk_duration': result.get('download_duration'),
                'file_size_bytes': result.get('file_size_bytes', 0)
            })
            flattened_results.append(base_result)
    
    df = pd.DataFrame(flattened_results)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")


def main():
    """Main function for pipeline step execution"""
    parser = argparse.ArgumentParser(
        description="Step 6: Voice Sample Extraction (MP3 Audio Downloader)"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Input CSV file with URLs and metadata"
    )
    parser.add_argument(
        "--output-dir", 
        default="step6_voice_samples", 
        help="Directory to save downloaded audio samples"
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV file for results (default: auto-generated in output-dir)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Generate output CSV filename if not provided
    if not args.output_csv:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output_csv = os.path.join(args.output_dir, f"{input_basename}_step6_results.csv")
    
    logger.info("=" * 50)
    logger.info("STEP 6: Voice Sample Extraction - STARTED")
    logger.info("=" * 50)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Output CSV: {args.output_csv}")
    
    try:
        # Initialize downloader
        downloader = AudioDownloader(args.output_dir)
        
        # Load input data
        logger.info("Loading input data...")
        links = downloader.load_links(args.input)
        logger.info(f"Loaded {len(links)} links for processing")
        
        # Process all links
        logger.info("Starting audio download process...")
        results = downloader.download_audio_for_all(links)
        
        # Save results
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        save_results(results, args.output_csv)
        
        # Summary
        successful_results = [r for r in results if r.get('success')]
        total_chunks = sum(r.get('chunks', 0) for r in successful_results)
        
        logger.info("=" * 50)
        logger.info("STEP 6: Voice Sample Extraction - COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total links processed: {len(links)}")
        logger.info(f"Successful downloads: {len(successful_results)}")
        logger.info(f"Total audio files created: {total_chunks}")
        logger.info(f"Audio samples directory: {args.output_dir}")
        logger.info(f"Results CSV: {args.output_csv}")
        
        if successful_results:
            logger.info("✅ Step 6 completed successfully")
        else:
            logger.warning("⚠️ Step 6 completed but no audio files were downloaded")
            
    except Exception as e:
        logger.error(f"❌ Step 6 failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
