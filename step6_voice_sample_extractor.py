import os
import subprocess
import time
import json
import pandas as pd
import re
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class VoiceSampleExtractor:
    def __init__(self, output_dir="voice_samples", max_duration_hours=1, quality="192", min_duration=30, max_duration=3600):
        self.output_dir = output_dir
        self.max_duration_hours = max_duration_hours
        self.max_duration_seconds = max_duration_hours * 3600
        self.quality = quality
        self.min_duration = min_duration  # MIN_DURATION from test.py
        self.max_duration = max_duration  # MAX_DURATION from test.py
        self.chunk_duration = 1800        # CHUNK_DURATION from test.py
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def extract_voice_samples(self, confirmed_voice_links: List[Dict]) -> List[Dict]:
        """Main method - exact logic from test.py download_audio_for_all"""
        if not confirmed_voice_links:
            self.logger.info("🔍 No confirmed voice links to extract samples from")
            return []

        self.logger.info(f"🎤 Starting audio extraction for {len(confirmed_voice_links)} links...")
        self.logger.info(f"📁 Files will be saved to: {self.output_dir}")
        self.logger.info(f"⏱️ Maximum duration: {self.max_duration_hours} hour(s)")

        extracted_samples = []

        for i, link_data in enumerate(confirmed_voice_links, 1):
            url = link_data.get('url', '')
            username = link_data.get('username', f'user{i}')
            
            if not url:
                self.logger.warning(f"⚠️ Skipping entry {i} - no URL provided")
                continue

            self.logger.info(f"🎤 [{i}/{len(confirmed_voice_links)}] Processing user: {username}, url: {url}")
            
            # Call test.py process_link logic
            success = self.process_link(url, username, link_data)
            
            if success:
                extracted_samples.append(link_data)

        self.logger.info(f"\n🎤 Audio extraction completed!")
        self.logger.info(f"📊 Total links processed: {len(confirmed_voice_links)}")
        self.logger.info(f"✅ Successful extractions: {len(extracted_samples)}")
        self.logger.info(f"❌ Failed extractions: {len(confirmed_voice_links) - len(extracted_samples)}")

        return extracted_samples

    def process_link(self, url: str, username: str, link_data: Dict) -> bool:
        """Exact logic from test.py process_link method"""
        platform = self.determine_platform(url)

        if platform == 'twitch':
            latest_video_url, duration = self.get_latest_twitch_vod_url_and_duration(url)
        else:
            latest_video_url, duration = self.get_latest_video_url_and_duration(url)

        if not latest_video_url:
            self.logger.warning(f"Could not get latest video url for {username} on {platform}")
            return False

        self.logger.info(f"Latest video duration: {duration} seconds")

        if duration < self.min_duration:
            self.logger.warning(f"Video too short ({duration}s), skipping")
            return False

        download_duration = min(duration, self.max_duration)

        filename = self.sanitize_filename(f"{username}_audio_full.mp3")
        filepath = os.path.join(self.output_dir, filename)

        success = self.download_audio_chunk(latest_video_url, filepath, 0, download_duration)

        if success:
            self.logger.info(f"Downloaded full audio: {filepath}")
            # Update link_data for pipeline compatibility
            link_data.update({
                'sample_extracted': True,
                'sample_file': filepath,
                'extraction_status': 'success',
                'download_duration': download_duration,
                'original_duration': duration,
                'sample_quality': self.quality,
                'processed_username': username,
                'sample_filename': filename,
                'platform_source': platform,
                'original_username': username
            })
            return True

        if duration > self.chunk_duration:
            self.logger.info(f"Full download failed or incomplete, trying chunked download (2x30min)")
            chunk_success = False
            
            for i in range(2):
                start = i * self.chunk_duration
                current_duration = min(self.chunk_duration, duration - start)
                
                chunk_filename = self.sanitize_filename(f"{username}_audio_part{i+1}.mp3")
                chunk_filepath = os.path.join(self.output_dir, chunk_filename)
                
                success_part = self.download_audio_chunk(latest_video_url, chunk_filepath, start, current_duration)
                
                if success_part:
                    self.logger.info(f"Downloaded chunk {i+1}: {chunk_filepath}")
                    chunk_success = True
                else:
                    self.logger.warning(f"Failed to download chunk {i+1}")
            
            if chunk_success:
                # Update link_data for pipeline compatibility
                link_data.update({
                    'sample_extracted': True,
                    'sample_file': filepath,
                    'extraction_status': 'success_chunked',
                    'download_duration': download_duration,
                    'original_duration': duration,
                    'sample_quality': self.quality,
                    'processed_username': username,
                    'sample_filename': filename,
                    'platform_source': platform,
                    'original_username': username
                })
                return True

        return False

    def determine_platform(self, url: str) -> str:
        """Exact logic from test.py"""
        if 'twitch.tv' in url:
            return 'twitch'
        elif 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'tiktok.com' in url:
            return 'tiktok'
        else:
            return 'unknown'

    def get_latest_video_url_and_duration(self, channel_url: str):
        """Exact logic from test.py"""
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
        """Exact logic from test.py"""
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
        """Exact logic from test.py"""
        try:
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', self.quality,
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
        """Exact logic from test.py"""
        filename = re.sub(r'[^\w\s-]', '', filename).strip().lower()
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename

    # Additional methods for pipeline compatibility
    def generate_samples_report(self, extracted_samples: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive report of extracted voice samples"""
        if not output_file:
            output_file = os.path.join(self.output_dir, "voice_samples_report.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("🎤 VOICE SAMPLES EXTRACTION REPORT (test.py logic)\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total samples extracted: {len(extracted_samples)}\n")
                f.write(f"Max duration per file: {self.max_duration_hours} hour(s)\n")
                f.write(f"Audio quality: {self.quality} kbps\n")
                f.write(f"Output directory: {self.output_dir}\n\n")
                
                if extracted_samples:
                    for i, sample in enumerate(extracted_samples, 1):
                        nickname = sample.get('processed_username', 'unknown')
                        platform = sample.get('platform_source', 'unknown')
                        filename = sample.get('sample_filename', 'N/A')
                        f.write(f"{i:2d}. {filename}\n")
                        f.write(f"    User: @{nickname}\n")
                        f.write(f"    Platform: {platform}\n")
                        f.write(f"    Status: {sample.get('extraction_status', 'unknown')}\n\n")
                else:
                    f.write("No samples were extracted.\n")
            
            self.logger.info(f"📄 Report saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return ""

    def clean_temp_files(self):
        """Clean temporary .ytdl files and other artifacts"""
        if not os.path.exists(self.output_dir):
            return
        
        cleaned_count = 0
        for filename in os.listdir(self.output_dir):
            if filename.endswith(('.ytdl', '.part', '.temp')):
                filepath = os.path.join(self.output_dir, filename)
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                    self.logger.info(f"🗑️ Cleaned: {filename}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not clean {filename}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned {cleaned_count} temporary files")

# Standalone execution for testing (like test.py)
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python step6_voice_sample_extractor.py <confirmed_voice_links.csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    
    # Create extractor using test.py logic
    extractor = VoiceSampleExtractor("voice_samples_output")
    
    # Load links like test.py load_links
    df = pd.read_csv(input_csv)
    confirmed_voice_links = df.to_dict('records')
    
    # Extract audio using test.py logic
    results = extractor.extract_voice_samples(confirmed_voice_links)
    
    # Generate report
    if results:
        report_file = extractor.generate_samples_report(results)
        print(f"Extracted {len(results)} audio samples successfully using test.py logic.")
        print(f"Report saved: {report_file}")
    else:
        print("No audio samples were extracted.")
