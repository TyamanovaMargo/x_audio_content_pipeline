import os
import subprocess
import time
import json
import pandas as pd
import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)

class VoiceSampleExtractor:
    
    MIN_DURATION = 30  # seconds
    MAX_DURATION = 3600  # 1 hour in seconds
    CHUNK_DURATION = 600  # 10 minutes in seconds (изменено с 1800 на 600)
    
    def __init__(self, output_dir: str = "output_audio2", min_duration=30, max_duration=3600, chunk_duration=600):
        self.output_dir = output_dir
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.chunk_duration = chunk_duration
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        print(f"🎵 AudioDownloader initialized:")
        print(f"📁 Output directory: {output_dir}")
        print(f"⏱️ Duration range: {min_duration}s - {max_duration}s")
        print(f"📦 Chunk size: {chunk_duration}s ({chunk_duration//60} minutes)")
        print(f"📺 Supported platforms: YouTube, Twitch, TikTok")

    def load_links(self, filepath: str) -> List[Dict]:
        """Load links from CSV file"""
        try:
            df = pd.read_csv(filepath)
            links = df.to_dict('records')
            print(f"📋 Loaded {len(links)} links from {filepath}")
            return links
        except Exception as e:
            self.logger.error(f"Failed to load links from {filepath}: {e}")
            return []

    def download_audio_for_all(self, links: List[Dict]) -> List[Dict]:
        """Download audio for all links with comprehensive tracking"""
        if not links:
            print("📝 No links to process")
            return []

        print(f"🎵 Starting audio download for {len(links)} links...")
        print(f"📁 Files will be saved to: {self.output_dir}")
        print(f"📦 Strategy: {self.chunk_duration//60}-minute chunks, max {self.max_duration//60} minutes total")

        processed_results = []

        for i, info in enumerate(links, 1):
            username = info.get('username', 'user')
            profile_name = info.get('profile_name', '')
            source_name = info.get('source_name', '')  # Extract source_name from link data
            url = info.get('url', '')
            
            if not url:
                print(f"⚠️ Skipping entry {i} - no URL provided")
                continue

            print(f"🎵 [{i}/{len(links)}] Processing user: {username}, profile: {profile_name}, url: {url}")
            
            # Process using existing logic with source_name
            result = self.process_link(url, username, source_name)
            
            if result:
                # Update info with results
                info.update({
                    'download_success': True,
                    'main_file': result.get('main_file'),
                    'download_status': 'success',
                    'audio_duration': result.get('duration'),
                    'processed_username': username,
                    'audio_filename': os.path.basename(result.get('main_file', '')),
                    'platform_source': result.get('platform'),
                    'download_method': result.get('method'),
                    'chunks_downloaded': result.get('chunks', 0),
                    'file_size': result.get('file_size', 0)
                })
                processed_results.append(info)
                print(f"✅ Successfully processed @{username}")
            else:
                info.update({
                    'download_success': False,
                    'download_status': 'failed',
                    'processed_username': username,
                    'platform_source': self.determine_platform(url)
                })
                print(f"❌ Failed to process @{username}")
                
            time.sleep(2)  # Rate limiting

        self._print_download_summary(processed_results, len(links))
        return processed_results

    def process_link(self, url: str, username: str, source_name: str = None) -> Optional[Dict]:
        """Process single link with existing logic"""
        platform = self.determine_platform(url)
        
        if platform == 'twitch':
            latest_video_url, duration = self.get_latest_twitch_vod_url_and_duration(url)
        else:
            latest_video_url, duration = self.get_latest_video_url_and_duration(url)

        if not latest_video_url:
            self.logger.warning(f"Could not get latest video url for {username} on {platform}")
            return None

        self.logger.info(f"Latest video duration: {duration} seconds")

        if duration < self.MIN_DURATION:
            self.logger.warning(f"Video too short ({duration}s), skipping")
            return None

        download_duration = min(duration, self.MAX_DURATION)

        # Generate filename with source_name if provided
        if source_name:
            base_filename = f"{username}_{source_name}_audio_full.mp3"
        else:
            base_filename = f"{username}_audio_full.mp3"
        
        filename = self.sanitize_filename(base_filename)
        filepath = os.path.join(self.output_dir, filename)

        success = self.download_audio_chunk(latest_video_url, filepath, 0, download_duration)

        if success:
            self.logger.info(f"Downloaded full audio: {filepath}")
            file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            return {
                'main_file': filepath,
                'duration': download_duration,
                'platform': platform,
                'method': 'full',
                'chunks': 1,
                'file_size': file_size
            }

        # If full download failed and video is long, try chunked download with 10-minute chunks
        if duration > self.CHUNK_DURATION:
            self.logger.info(f"Full download failed or incomplete, trying chunked download ({self.chunk_duration//60}min chunks)")
            
            # Calculate number of chunks needed (max 6 chunks for 1 hour)
            max_chunks = min(6, (self.MAX_DURATION // self.CHUNK_DURATION))
            chunks_downloaded = 0
            chunk_files = []
            
            for i in range(max_chunks):
                start = i * self.CHUNK_DURATION
                current_duration = min(self.CHUNK_DURATION, duration - start, self.MAX_DURATION - start)
                
                if current_duration <= 0:
                    break
                    
                # Generate chunk filename with source_name if provided
                if source_name:
                    chunk_base_filename = f"{username}_{source_name}_audio_part{i+1}.mp3"
                else:
                    chunk_base_filename = f"{username}_audio_part{i+1}.mp3"
                
                chunk_filename = self.sanitize_filename(chunk_base_filename)
                chunk_filepath = os.path.join(self.output_dir, chunk_filename)
                
                success_part = self.download_audio_chunk(latest_video_url, chunk_filepath, start, current_duration)
                
                if success_part:
                    self.logger.info(f"Downloaded chunk {i+1}: {chunk_filepath}")
                    chunks_downloaded += 1
                    chunk_files.append(chunk_filepath)
                else:
                    self.logger.warning(f"Failed to download chunk {i+1}")
                    
            if chunks_downloaded > 0:
                self.logger.info(f"Successfully downloaded {chunks_downloaded} chunks for {username}")
                # Return first chunk as main file
                main_file = chunk_files[0]
                file_size = os.path.getsize(main_file) if os.path.exists(main_file) else 0
                total_duration = chunks_downloaded * self.CHUNK_DURATION
                
                return {
                    'main_file': main_file,
                    'duration': min(total_duration, download_duration),
                    'platform': platform,
                    'method': 'chunked',
                    'chunks': chunks_downloaded,
                    'file_size': file_size,
                    'chunk_files': chunk_files
                }
            else:
                self.logger.warning(f"No chunks could be downloaded for {username}")
        
        return None

    def determine_platform(self, url: str) -> str:
        """Determine platform from URL"""
        if 'twitch.tv' in url:
            return 'twitch'
        elif 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'tiktok.com' in url:
            return 'tiktok'
        else:
            return 'unknown'

    def get_latest_video_url_and_duration(self, channel_url: str):
        """Get latest video URL and duration"""
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
            self.logger.error(f"Failed to fetch latest video info: {e}")
            return None, 0

    def get_latest_twitch_vod_url_and_duration(self, channel_url: str):
        """Get latest Twitch VOD URL and duration"""
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
            self.logger.error(f"Failed to fetch latest Twitch VOD info: {e}")
            return None, 0

    def download_audio_chunk(self, video_url: str, output_path: str, start_sec: int, duration_sec: int) -> bool:
        """Download audio chunk with improved error handling"""
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

            self.logger.info(f"Downloading audio segment: start={start_sec}s, duration={duration_sec}s")
            subprocess.check_call(cmd)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
                return True
            else:
                self.logger.warning(f"Downloaded file too small or missing: {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to download audio chunk: {e}")
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Clean filename for filesystem compatibility"""
        filename = re.sub(r'[^\w\s-]', '', filename).strip().lower()
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename

    def _print_download_summary(self, processed_results: List[Dict], total_links: int):
        """Print comprehensive download summary"""
        successful = len(processed_results)
        failed = total_links - successful

        print(f"\n🎵 AUDIO DOWNLOAD COMPLETED!")
        print("=" * 60)
        print(f"📊 Total links processed: {total_links}")
        print(f"✅ Successful downloads: {successful}")
        print(f"❌ Failed downloads: {failed}")
        print(f"📈 Success rate: {(successful / total_links * 100):.1f}%")
        print(f"📁 Files saved in: {self.output_dir}")

        if processed_results:
            # Method breakdown
            methods = {}
            platforms = {}
            for result in processed_results:
                method = result.get('download_method', 'unknown')
                platform = result.get('platform_source', 'unknown')
                methods[method] = methods.get(method, 0) + 1
                platforms[platform] = platforms.get(platform, 0) + 1

            print(f"\n📦 DOWNLOAD METHOD BREAKDOWN:")
            for method, count in methods.items():
                print(f"  {method}: {count} files")
                
            print(f"\n🔗 PLATFORM BREAKDOWN:")
            for platform, count in platforms.items():
                print(f"  {platform}: {count} files")
                
            # Show chunk statistics
            chunked_downloads = [r for r in processed_results if r.get('download_method') == 'chunked']
            if chunked_downloads:
                total_chunks = sum(r.get('chunks_downloaded', 0) for r in chunked_downloads)
                print(f"\n📦 CHUNK STATISTICS:")
                print(f"  Chunked downloads: {len(chunked_downloads)}")
                print(f"  Total chunks: {total_chunks}")
                print(f"  Average chunks per video: {total_chunks/len(chunked_downloads):.1f}")

    def generate_download_report(self, processed_results: List[Dict], output_file: str = None) -> str:
        """Generate comprehensive download report"""
        if not output_file:
            output_file = os.path.join(self.output_dir, "audio_download_report.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎵 AUDIO DOWNLOAD REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files downloaded: {len(processed_results)}\n")
            f.write(f"Chunk size: {self.chunk_duration} seconds ({self.chunk_duration//60} minutes)\n")
            f.write(f"Maximum duration: {self.max_duration} seconds ({self.max_duration//60} minutes)\n")
            f.write(f"Audio quality: 192 kbps\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            if processed_results:
                f.write("📋 DETAILED DOWNLOAD LIST:\n")
                f.write("-" * 40 + "\n")
                
                for i, result in enumerate(processed_results, 1):
                    f.write(f"{i:2d}. {result.get('audio_filename', 'N/A')}\n")
                    f.write(f"    User: @{result.get('processed_username', 'unknown')}\n")
                    f.write(f"    Platform: {result.get('platform_source', 'unknown')}\n")
                    f.write(f"    Duration: {result.get('audio_duration', 0)} seconds\n")
                    f.write(f"    Method: {result.get('download_method', 'unknown')}\n")
                    f.write(f"    Chunks: {result.get('chunks_downloaded', 1)}\n")
                    f.write(f"    File size: {result.get('file_size', 0)//1000}KB\n\n")

        print(f"📄 Audio download report saved: {output_file}")
        return output_file

    def clean_temp_files(self):
        """Clean temporary files if needed"""
        pass


# This class is designed to be imported and used by main_pipeline.py
# No standalone execution needed