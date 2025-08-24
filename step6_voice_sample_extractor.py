import os
import subprocess
import requests
import pandas as pd
from typing import List, Dict
from urllib.parse import urlparse
import time
import logging
import json
import re

class VoiceSampleExtractor:
    def __init__(self, output_dir="voice_samples", sample_duration=120, quality="192", 
                 min_duration=30, max_duration=3600):
        self.output_dir = output_dir
        self.sample_duration = sample_duration  # seconds
        self.quality = quality  # kbps
        self.min_duration = min_duration  # 30 seconds minimum
        self.max_duration = max_duration  # 1 hour maximum
        os.makedirs(output_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_voice_samples(self, confirmed_voice_links: List[Dict]) -> List[Dict]:
        """Extract voice samples from confirmed voice content with duration filtering"""
        if not confirmed_voice_links:
            print("🔍 No confirmed voice links to extract samples from")
            return []

        print(f"🎤 Starting voice sample extraction for {len(confirmed_voice_links)} links...")
        print(f"📁 Samples will be saved to: {self.output_dir}")
        print(f"⏱️ Sample duration: {self.sample_duration} seconds")
        print(f"🎯 Duration filter: {self.min_duration}s - {self.max_duration}s")
        print(f"📝 Filename format: username_source_timestamp.mp3")

        extracted_samples = []
        duration_filtered = 0

        for i, link_data in enumerate(confirmed_voice_links, 1):
            url = link_data.get('url', '')
            username = self._extract_best_username(link_data, url)
            platform = link_data.get('platform_type', 'unknown')

            if not url:
                print(f" ⚠️ Skipping entry {i} - no URL provided")
                continue

            print(f"🎤 [{i}/{len(confirmed_voice_links)}] Processing @{username} ({platform})")

            # Check duration before extraction
            duration_info = self._check_video_duration(url)
            if duration_info and not duration_info['valid']:
                print(f" ⏰ Skipped: {duration_info['reason']} ({duration_info['duration']}s)")
                duration_filtered += 1
                continue

            # Generate filename
            safe_username = self._sanitize_filename(username)
            safe_platform = platform.lower() if platform else 'unknown'
            timestamp = int(time.time())
            filename = f"{safe_username}_{safe_platform}_{timestamp}"

            # Extract sample
            extraction_result = self._extract_audio_sample(url, filename, platform, safe_username)

            # Add results
            link_data.update({
                'sample_extracted': extraction_result['success'],
                'sample_file': extraction_result.get('file_path'),
                'extraction_status': extraction_result['status'],
                'sample_duration': self.sample_duration,
                'sample_quality': self.quality,
                'processed_username': safe_username,
                'sample_filename': filename + '.mp3',
                'platform_source': safe_platform,
                'original_username': username
            })

            if extraction_result['success']:
                extracted_samples.append(link_data)
                print(f" ✅ Sample saved: {safe_username}_{safe_platform}_{timestamp}.mp3")
            else:
                print(f" ❌ Failed: {extraction_result['status']}")

            time.sleep(2)

        print(f"\n🎤 Voice sample extraction completed!")
        print(f"📊 Total links processed: {len(confirmed_voice_links)}")
        print(f"⏰ Duration filtered: {duration_filtered}")
        print(f"✅ Successful extractions: {len(extracted_samples)}")
        print(f"❌ Failed extractions: {len(confirmed_voice_links) - len(extracted_samples) - duration_filtered}")

        return extracted_samples

    def _check_video_duration(self, url: str) -> Dict:
        """Check video duration before processing"""
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--quiet',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                info = json.loads(result.stdout)
                duration = info.get('duration', 0)
                title = info.get('title', 'Unknown')
                
                is_valid = self.min_duration <= duration <= self.max_duration
                
                return {
                    'duration': duration,
                    'title': title,
                    'valid': is_valid,
                    'reason': 'valid' if is_valid else 
                             'too_short' if duration < self.min_duration else 'too_long'
                }
        except Exception as e:
            return {
                'duration': 0,
                'title': 'Unknown',
                'valid': False,
                'reason': f'check_failed: {e}'
            }

    def _extract_audio_sample(self, url: str, filename: str, platform: str, nickname: str) -> Dict:
        output_path = os.path.join(self.output_dir, f"{filename}.mp3")

        try:
            if platform == 'youtube':
                return self._extract_youtube_sample(url, output_path, nickname)
            elif platform == 'twitch':
                return self._extract_twitch_sample(url, output_path, nickname)
            elif platform == 'tiktok':
                return self._extract_tiktok_sample(url, output_path, nickname)
            else:
                return {'success': False, 'status': f'unsupported_platform: {platform}'}
        except Exception as e:
            return {'success': False, 'status': f'extraction_error_for_{nickname}: {str(e)[:100]}'}

    def _extract_youtube_sample(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio sample from YouTube with duration constraints"""
        quality_options = [
            (self.quality, 240),
            ("128", 180),
            ("96", 120)
        ]

        for quality, timeout in quality_options:
            try:
                print(f" 🎧 Trying YouTube {quality} kbps (timeout: {timeout}s)")
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'mp3',
                    '--audio-quality', quality,
                    '--match-filter', f'duration > {self.min_duration} & duration < {self.max_duration}',
                    '--postprocessor-args', f'ffmpeg:-t {min(self.sample_duration, 120)}',
                    '--output', output_path.replace('.mp3', '.%(ext)s'),
                    '--no-playlist',
                    '--quiet',
                    '--no-warnings',
                    '--ignore-errors',
                    '--fragment-retries', '3',
                    '--retries', '3',
                    url
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

                if result.returncode == 0 and os.path.exists(output_path):
                    return {
                        'success': True,
                        'file_path': output_path,
                        'status': f'youtube_success_{nickname}_quality_{quality}'
                    }
                else:
                    print(f" ⚠️ Quality {quality} failed, trying next...")

            except subprocess.TimeoutExpired:
                print(f" ⏰ Timeout at {quality} kbps, trying lower quality...")
                continue
            except FileNotFoundError:
                return {'success': False, 'status': 'yt-dlp_not_installed'}
            except Exception as e:
                print(f" ❌ Error at {quality}: {str(e)[:50]}")
                continue

        return {'success': False, 'status': f'youtube_failed_all_qualities_{nickname}'}

    def _extract_tiktok_sample(self, url: str, output_path: str, nickname: str) -> dict:
        """Extract TikTok sample with duration filtering"""
        quality_options = [
            (self.quality, 180),
            ("128", 120),
            ("96", 90),
            ("64", 60)
        ]

        for quality, timeout in quality_options:
            try:
                print(f" 🎧 Trying TikTok {quality} kbps (timeout: {timeout}s)")
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'mp3',
                    '--audio-quality', quality,
                    '--match-filter', f'duration > {self.min_duration} & duration < {self.max_duration}',
                    '--postprocessor-args', f'ffmpeg:-t {min(self.sample_duration, 120)}',
                    '--output', output_path.replace('.mp3', '.%(ext)s'),
                    '--quiet',
                    '--no-warnings',
                    '--ignore-errors',
                    '--fragment-retries', '3',
                    '--retries', '3',
                    '--max-filesize', '50M',
                    url
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

                if result.returncode == 0 and os.path.exists(output_path):
                    return {
                        'success': True,
                        'file_path': output_path,
                        'status': f'tiktok_success_{nickname}_quality_{quality}'
                    }
                else:
                    print(f" ⚠️ Quality {quality} failed, trying next...")

            except subprocess.TimeoutExpired:
                print(f" ⏰ Timeout at {quality} kbps, trying lower quality...")
                continue
            except FileNotFoundError:
                return {'success': False, 'status': 'yt-dlp_not_installed'}
            except Exception as e:
                print(f" ❌ Error at {quality}: {str(e)[:50]}")
                continue

        return {'success': False, 'status': f'tiktok_failed_all_qualities_{nickname}'}

    def _extract_twitch_sample(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio sample from Twitch with duration constraints"""
        if '/videos/' not in url and '/clip/' not in url:
            if url.endswith('/videos'):
                return self._try_get_recent_twitch_vod(url, output_path, nickname)
            else:
                videos_url = url.rstrip('/') + '/videos'
                return self._try_get_recent_twitch_vod(videos_url, output_path, nickname)

        return self._extract_direct_twitch_content(url, output_path, nickname)

    def _extract_direct_twitch_content(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio from direct Twitch VOD with duration filtering"""
        quality_options = [
            (self.quality, 360),
            ("128", 300),
            ("96", 240),
            ("64", 180)
        ]

        for quality, timeout in quality_options:
            try:
                print(f" 🎧 Trying Twitch {quality} kbps (timeout: {timeout}s)")
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'mp3',
                    '--audio-quality', quality,
                    '--match-filter', f'duration > {self.min_duration} & duration < {self.max_duration}',
                    '--postprocessor-args', f'ffmpeg:-t {min(self.sample_duration, 120)}',
                    '--output', output_path.replace('.mp3', '.%(ext)s'),
                    '--quiet',
                    '--no-warnings',
                    '--ignore-errors',
                    '--fragment-retries', '3',
                    '--retries', '3',
                    '--max-filesize', '100M',
                    url
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

                if result.returncode == 0 and os.path.exists(output_path):
                    return {
                        'success': True,
                        'file_path': output_path,
                        'status': f'twitch_success_{nickname}_quality_{quality}'
                    }
                else:
                    print(f" ⚠️ Quality {quality} failed, trying next...")

            except subprocess.TimeoutExpired:
                print(f" ⏰ Timeout at {quality} kbps, trying lower quality...")
                continue
            except FileNotFoundError:
                return {'success': False, 'status': 'yt-dlp_not_installed'}
            except Exception as e:
                print(f" ❌ Error at {quality}: {str(e)[:50]}")
                continue

        return {'success': False, 'status': f'twitch_failed_all_qualities_{nickname}'}

    # Include all your existing helper methods here:
    # _extract_best_username, _sanitize_filename, _extract_username_from_url, etc.
    # (I'm keeping them the same as your original code for brevity)

    def _extract_best_username(self, link_data: Dict, url: str) -> str:
        """Extract username with URL parsing priority and descriptive text filtering"""
        # Your existing implementation
        username_from_url = self._extract_username_from_url(url)
        if username_from_url and len(username_from_url) > 2:
            return username_from_url

        username_fields = ['username', 'screen_name', 'user_name', 'handle', 'account_name']
        for field in username_fields:
            value = link_data.get(field)
            if value and not self._is_empty_value(value):
                username = str(value).strip()
                if username and not self._is_descriptive_text(username):
                    return username

        if url:
            url_hash = abs(hash(url)) % 10000
            return f"user_{url_hash}"

        return f"user_{int(time.time()) % 10000}"

    def _is_empty_value(self, value) -> bool:
        """Check if a value is considered empty"""
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, float) and (pd.isna(value) or value != value):
            return True
        str_val = str(value).lower().strip()
        return str_val in ['nan', '', 'none', 'null', 'undefined']

    def _is_descriptive_text(self, text: str) -> bool:
        """Check if text is descriptive rather than a real username"""
        if not text or len(text) > 30:
            return True
        text_lower = text.lower()
        descriptive_words = [
            'check', 'pinned', 'moved', 'see', 'bio', 'link', 'description',
            'follow', 'subscribe', 'contact', 'info', 'about', 'moved to',
            'see pinned', 'check bio', 'dm for', 'business', 'inquiries'
        ]
        word_count = sum(1 for w in descriptive_words if w in text_lower)
        space_count = text.count(' ')
        comma_count = text.count(',')
        return word_count >= 1 or space_count >= 2 or comma_count >= 1

    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from URL"""
        if not url:
            return None
        try:
            url = url.strip()
            if 'youtube.com' in url or 'youtu.be' in url:
                return self._extract_youtube_username(url)
            elif 'twitch.tv' in url:
                return self._extract_twitch_username(url)
            elif 'tiktok.com' in url:
                return self._extract_tiktok_username(url)
            else:
                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts:
                    return path_parts[-1][:20]
        except Exception:
            pass
        return None

    def _extract_youtube_username(self, url: str) -> str:
        """Extract username from YouTube URL"""
        patterns = [
            r'/channel/([^/?]+)',
            r'/user/([^/?]+)',
            r'/c/([^/?]+)',
            r'/@([^/?]+)',
            r'/watch\?v=([^&]+)',
            r'youtu\.be/([^/?]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)[:20]
                if username and not username.startswith('UC'):
                    return username
                elif username.startswith('UC'):
                    return f"yt_{username[-8:]}"
        return None

    def _extract_twitch_username(self, url: str) -> str:
        """Extract username from Twitch URL"""
        patterns = [
            r'twitch\.tv/([^/?]+)',
            r'twitch\.tv/([^/]+)/videos',
            r'clips\.twitch\.tv/([^/?]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)
                if username.lower() not in ['videos', 'clips', 'collections', 'following', 'directory']:
                    return username[:20]
        return None

    def _extract_tiktok_username(self, url: str) -> str:
        """Extract username from TikTok URL"""
        patterns = [
            r'tiktok\.com/@([^/?]+)',
            r'vm\.tiktok\.com/([^/?]+)',
            r'm\.tiktok\.com/@([^/?]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)
                if username and not username.startswith('video'):
                    return username[:20]
        return None

    def _sanitize_filename(self, filename) -> str:
        """Clean filename with emoji and special character handling"""
        if not filename or self._is_empty_value(filename):
            return f"user_{int(time.time()) % 10000}"

        filename = str(filename).strip()
        filename = re.sub(r'[^\w\s-]', '', filename, flags=re.UNICODE)
        filename = filename.lower()
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r'[-]+', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')
        filename = re.sub(r'[^a-zA-Z0-9_]', '', filename)

        if len(filename) > 20:
            filename = filename[:20]

        if not filename or len(filename) < 2:
            return f"user_{int(time.time()) % 10000}"

        return filename

    # Add your existing report generation methods here...

