import os
import subprocess
import time
import json
import re
from typing import List, Dict
from urllib.parse import urlparse
import pandas as pd
import logging
from tqdm import tqdm  


class VoiceSampleExtractor:
    def __init__(self, output_dir="voice_samples", max_duration_hours=1, quality="192", min_duration=30, max_duration=3600):
        self.output_dir = output_dir
        self.max_duration_hours = max_duration_hours
        self.max_duration_seconds = max_duration_hours * 3600
        self.quality = quality
        self.min_duration = min_duration
        self.max_duration = max_duration

        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_voice_samples(self, confirmed_voice_links: List[Dict]) -> List[Dict]:
        """Extract full audio files (up to 1 hour) from confirmed voice content"""
        if not confirmed_voice_links:
            self.logger.info("🔍 No confirmed voice links to extract samples from")
            return []

        # Separate channel URLs from direct video URLs
        channel_links = [link for link in confirmed_voice_links if link.get('url_type') == 'channel']
        video_links = [link for link in confirmed_voice_links if link.get('url_type') != 'channel']

        self.logger.info(f"🎤 Starting full audio extraction for {len(confirmed_voice_links)} links...")
        self.logger.info(f"📁 Files will be saved to: {self.output_dir}")
        self.logger.info(f"⏱️ Maximum duration: {self.max_duration_hours} hour(s)")
        self.logger.info(f"🎯 Will extract full audio (up to {self.max_duration_seconds} seconds)")
        self.logger.info(f"🎤 Processing {len(video_links)} direct video links and {len(channel_links)} channel links...")

        extracted_samples = []
        duration_filtered = 0

        # Process direct video links
        for i, link_data in enumerate(video_links, 1):
            url = link_data.get('url', '')
            username = self._extract_best_username(link_data, url)
            platform = link_data.get('platform_type', 'unknown')

            if not url:
                self.logger.warning(f"⚠️ Skipping entry {i} - no URL provided")
                continue

            self.logger.info(f"🎤 [{i}/{len(video_links)}] Processing @{username} ({platform})")
            self.logger.info(f" URL: {url[:60]}...")

            if platform == 'youtube':
                self.logger.info("🔴 YouTube extraction will be used")
            elif platform == 'twitch':
                self.logger.info("🟣 Twitch extraction will be used")
            elif platform == 'tiktok':
                self.logger.info("⚫ TikTok extraction will be used")
            else:
                self.logger.info(f"❓ Unknown platform: {platform}")

            # Check duration before extraction
            duration_info = self._check_video_duration(url)
            if duration_info and not duration_info['valid']:
                self.logger.info(f"⏰ Skipped: {duration_info['reason']} ({duration_info['duration']}s)")
                duration_filtered += 1
                continue

            # Generate filename
            safe_username = self._sanitize_filename(username)
            safe_platform = platform.lower() if platform else 'unknown'
            timestamp = int(time.time())

            # Determine download duration
            video_duration = duration_info.get('duration', 0) if duration_info else 0
            download_duration = min(video_duration, self.max_duration_seconds) if video_duration > 0 else self.max_duration_seconds

            filename = f"{safe_username}_{safe_platform}_{timestamp}_full"

            # Extract full audio (up to 1 hour)
            extraction_result = self._extract_full_audio(url, filename, platform, safe_username, download_duration)

            # Add results
            link_data.update({
                'sample_extracted': extraction_result['success'],
                'sample_file': extraction_result.get('file_path'),
                'extraction_status': extraction_result['status'],
                'download_duration': download_duration,
                'original_duration': video_duration,
                'sample_quality': self.quality,
                'processed_username': safe_username,
                'sample_filename': filename + '.mp3',
                'platform_source': safe_platform,
                'original_username': username
            })

            if extraction_result['success']:
                extracted_samples.append(link_data)
                self.logger.info(f"✅ Full audio saved: {safe_username}_{safe_platform}_{timestamp}_full.mp3")
                self.logger.info(f"Duration: {download_duration}s from original {video_duration}s")
            else:
                self.logger.info(f"❌ Failed: {extraction_result['status']}")

            time.sleep(2)

        # Process channel links - get recent videos
        for i, link_data in enumerate(channel_links, 1):
            channel_url = link_data.get('url', '')
            username = self._extract_best_username(link_data, channel_url)
            platform = link_data.get('platform_type', 'unknown')

            self.logger.info(f"📺 [{i}/{len(channel_links)}] Channel: @{username} ({platform})")
            self.logger.info(f"Channel URL: {channel_url[:60]}...")

            if platform == 'twitch':
                self.logger.info("🟣 Processing Twitch channel")
                videos_url = channel_url.rstrip('/') + '/videos'
                safe_username = self._sanitize_filename(username)
                timestamp = int(time.time())
                filename = f"{safe_username}_{platform}_{timestamp}_full"
                extraction_result = self._try_get_recent_twitch_vod_full(
                    videos_url, os.path.join(self.output_dir, f"{filename}.mp3"), safe_username, self.max_duration_seconds
                )
                if extraction_result['success']:
                    result_data = link_data.copy()
                    result_data.update({
                        'sample_extracted': True,
                        'sample_file': extraction_result.get('file_path'),
                        'channel_url': channel_url,
                        'download_duration': self.max_duration_seconds,
                        'original_duration': 'unknown',
                        'processed_username': safe_username,
                        'sample_filename': filename + '.mp3',
                        'platform_source': platform
                    })
                    extracted_samples.append(result_data)
                    self.logger.info("✅ Extracted from Twitch channel")
                else:
                    self.logger.info(f"❌ Failed Twitch extraction: {extraction_result['status']}")

            elif platform == 'youtube':
                self.logger.info("🔴 Processing YouTube channel")
                try:
                    from step4_audio_filter import AudioContentFilter
                    filter_obj = AudioContentFilter(self.min_duration, self.max_duration)
                    recent_videos = filter_obj.get_recent_videos_from_channel(channel_url, max_videos=3)

                    if recent_videos:
                        for video_info in recent_videos:
                            self.logger.info(f"🎬 Processing: {video_info['title'][:40]}...")
                            safe_username = self._sanitize_filename(username)
                            timestamp = int(time.time())
                            video_duration = video_info.get('duration', 0)
                            download_duration = min(video_duration, self.max_duration_seconds) if video_duration > 0 else self.max_duration_seconds
                            filename = f"{safe_username}_{platform}_{timestamp}_full"
                            extraction_result = self._extract_full_audio(
                                video_info['url'], filename, platform, safe_username, download_duration
                            )
                            if extraction_result['success']:
                                result_data = link_data.copy()
                                result_data.update({
                                    'sample_extracted': True,
                                    'sample_file': extraction_result.get('file_path'),
                                    'video_url': video_info['url'],
                                    'video_title': video_info['title'],
                                    'duration': video_info['duration'],
                                    'download_duration': download_duration,
                                    'original_duration': video_duration,
                                    'processed_username': safe_username,
                                    'sample_filename': filename + '.mp3',
                                    'platform_source': platform
                                })
                                extracted_samples.append(result_data)
                                self.logger.info(f"✅ Extracted from: {video_info['title'][:30]}...")
                                self.logger.info(f"Duration: {download_duration}s from original {video_duration}s")
                                break # Take first successful extraction per channel
                            else:
                                self.logger.info(f"❌ Failed: {extraction_result['status']}")
                    else:
                        self.logger.info("❌ No recent videos found for YouTube channel")
                except ImportError:
                    self.logger.warning("step4_audio_filter module not found, skipping YouTube channel processing")

            elif platform == 'tiktok':
                self.logger.info("⚫ Processing TikTok channel")
                safe_username = self._sanitize_filename(username)
                timestamp = int(time.time())
                filename = f"{safe_username}_{platform}_{timestamp}_full"
                extraction_result = self._extract_tiktok_channel_videos(
                    channel_url, filename, safe_username
                )
                if extraction_result['success']:
                    result_data = link_data.copy()
                    result_data.update({
                        'sample_extracted': True,
                        'sample_file': extraction_result.get('file_path'),
                        'channel_url': channel_url,
                        'download_duration': extraction_result.get('duration', 0),
                        'original_duration': extraction_result.get('original_duration', 0),
                        'processed_username': safe_username,
                        'sample_filename': filename + '.mp3',
                        'platform_source': platform
                    })
                    extracted_samples.append(result_data)
                    self.logger.info("✅ Extracted from TikTok channel")
                else:
                    self.logger.info(f"❌ Failed TikTok extraction: {extraction_result['status']}")
            else:
                self.logger.info(f"❓ Unknown platform '{platform}' - skipping")
                continue

            time.sleep(2) # Pause between channels

        self.logger.info(f"\n🎤 Full audio extraction completed!")
        self.logger.info(f"📊 Total links processed: {len(confirmed_voice_links)}")
        self.logger.info(f"⏰ Duration filtered: {duration_filtered}")
        self.logger.info(f"✅ Successful extractions: {len(extracted_samples)}")
        self.logger.info(f"❌ Failed extractions: {len(confirmed_voice_links) - len(extracted_samples) - duration_filtered}")

        return extracted_samples

    def _extract_full_audio(self, url: str, filename: str, platform: str, nickname: str, max_duration: int) -> Dict:
        """Extract full audio file (up to specified duration)"""
        output_path = os.path.join(self.output_dir, f"{filename}.mp3")
        try:
            if platform == 'youtube':
                return self._extract_youtube_full(url, output_path, nickname, max_duration)
            elif platform == 'twitch':
                return self._extract_twitch_full(url, output_path, nickname, max_duration)
            elif platform == 'tiktok':
                return self._extract_tiktok_full(url, output_path, nickname, max_duration)
            else:
                return {'success': False, 'status': f'unsupported_platform: {platform}'}
        except Exception as e:
            return {'success': False, 'status': f'extraction_error_for_{nickname}: {str(e)[:100]}'}

    def _extract_youtube_full(self, url: str, output_path: str, nickname: str, max_duration: int) -> Dict:
        """Extract full audio from YouTube (up to max_duration seconds) with chunk fallback"""
        # Try full download first
        success = self._download_audio_chunk(url, output_path, 0, max_duration)
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 100000:
            return {
                'success': True,
                'file_path': output_path,
                'file_size': os.path.getsize(output_path),
                'status': f'youtube_success_full_{nickname}_duration_{max_duration}s'
            }
        # If failed, try 6 chunks of 10 minutes
        chunk_duration = 600 # 10 minutes
        chunk_files = []
        for i in range(6):
            start = i * chunk_duration
            if start >= max_duration:
                break
            current_duration = min(chunk_duration, max_duration - start)
            chunk_file = os.path.join(self.output_dir, f"{nickname}_youtube_{int(time.time())}_part{i+1}.mp3")
            if self._download_audio_chunk(url, chunk_file, start, current_duration):
                chunk_files.append(chunk_file)
        if chunk_files:
            return {
                'success': True,
                'file_path': output_path,
                'chunks': chunk_files,
                'status': f'youtube_success_chunks_{nickname}'
            }
        return {'success': False, 'status': f'youtube_failed_all_methods_{nickname}'}

    def _extract_twitch_full(self, url: str, output_path: str, nickname: str, max_duration: int) -> Dict:
        """Extract full audio from Twitch with improved channel handling"""
        self.logger.info(f"🟣 Processing Twitch URL: {url[:50]}...")
        # Check Twitch URL type
        if '/videos/' in url or '/clip/' in url:
            self.logger.info("🎬 Direct Twitch VOD/Clip detected")
            return self._extract_direct_twitch_content_full(url, output_path, nickname, max_duration)
        elif url.endswith('/videos'):
            self.logger.info("📺 Twitch videos page detected")
            return self._try_get_recent_twitch_vod_full(url, output_path, nickname, max_duration)
        else:
            self.logger.info("📺 Twitch channel detected, searching for recent videos")
            videos_url = url.rstrip('/') + '/videos'
            return self._try_get_recent_twitch_vod_full(videos_url, output_path, nickname, max_duration)

    def _extract_direct_twitch_content_full(self, url: str, output_path: str, nickname: str, max_duration: int) -> Dict:
        """Extract full audio from direct Twitch VOD with enhanced stability"""
        # Try full download first
        success = self._download_audio_chunk(url, output_path, 0, max_duration)
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
            return {
                'success': True,
                'file_path': output_path,
                'file_size': os.path.getsize(output_path),
                'status': f'twitch_success_full_{nickname}_duration_{max_duration}s'
            }
        # If failed, try 6 chunks of 10 minutes
        chunk_duration = 600 # 10 minutes
        chunk_files = []
        for i in range(6):
            start = i * chunk_duration
            if start >= max_duration:
                break
            current_duration = min(chunk_duration, max_duration - start)
            chunk_file = os.path.join(self.output_dir, f"{nickname}_twitch_{int(time.time())}_part{i+1}.mp3")
            if self._download_audio_chunk(url, chunk_file, start, current_duration):
                chunk_files.append(chunk_file)
        if chunk_files:
            return {
                'success': True,
                'file_path': output_path,
                'chunks': chunk_files,
                'status': f'twitch_success_chunks_{nickname}'
            }
        return {'success': False, 'status': f'twitch_failed_all_methods_{nickname}'}

    def _try_get_recent_twitch_vod_full(self, videos_url: str, output_path: str, nickname: str, max_duration: int) -> Dict:
        """Try to get recent VOD from Twitch channel videos page"""
        try:
            self.logger.info(f"🔍 Searching recent VODs for @{nickname}...")
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--playlist-end', '1', # Get only the most recent VOD
                '--quiet',
                '--no-warnings',
                '--ignore-errors',
                '--socket-timeout', '20',
                videos_url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)

            if result.returncode == 0 and result.stdout.strip():
                lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
                if lines:
                    try:
                        vod_info = json.loads(lines[0])
                        vod_url = vod_info.get('webpage_url') or vod_info.get('url')
                        vod_title = vod_info.get('title', 'Unknown Title')[:30]
                        vod_duration = vod_info.get('duration', 0)

                        if vod_url:
                            self.logger.info(f"🎬 Found recent VOD: {vod_title}... (duration: {vod_duration}s)")
                            if vod_duration > self.max_duration_seconds:
                                self.logger.info(f"⏰ VOD too long ({vod_duration}s), limiting to {self.max_duration_seconds}s")
                                actual_max_duration = self.max_duration_seconds
                            else:
                                actual_max_duration = min(max_duration, vod_duration) if vod_duration > 0 else max_duration
                            return self._extract_direct_twitch_content_full(vod_url, output_path, nickname, actual_max_duration)
                        else:
                            self.logger.warning(f"⚠️ No valid URL found in VOD info for @{nickname}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"⚠️ Failed to parse VOD info for @{nickname}: {e}")
                else:
                    self.logger.warning(f"⚠️ No VOD data found for @{nickname}")
            else:
                self.logger.warning(f"⚠️ yt-dlp failed for @{nickname}: {result.stderr[:100] if result.stderr else 'unknown error'}")

            return {'success': False, 'status': f'no_recent_vods_found_for_{nickname}'}
        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ Timeout searching VODs for @{nickname}")
            return {'success': False, 'status': f'twitch_vod_search_timeout_for_{nickname}'}
        except Exception as e:
            self.logger.error(f"❌ Exception searching VODs for @{nickname}: {str(e)}")
            return {'success': False, 'status': f'twitch_vod_search_failed_for_{nickname}: {str(e)[:100]}'}

    def _extract_tiktok_full(self, url: str, output_path: str, nickname: str, max_duration: int) -> Dict:
        """Extract full TikTok audio with improved error handling"""
        # TikTok videos are usually short, try direct download
        success = self._download_audio_chunk(url, output_path, 0, max_duration)
        if success and os.path.exists(output_path) and os.path.getsize(output_path) > 50000:
            return {
                'success': True,
                'file_path': output_path,
                'file_size': os.path.getsize(output_path),
                'status': f'tiktok_success_{nickname}_duration_{max_duration}s'
            }
        return {'success': False, 'status': f'tiktok_failed_{nickname}'}

    def _extract_tiktok_channel_videos(self, channel_url: str, filename: str, nickname: str) -> Dict:
        """Extract audio from TikTok channel - get recent videos"""
        output_path = os.path.join(self.output_dir, f"{filename}.mp3")
        success = self._download_audio_chunk(channel_url, output_path, 0, self.max_duration_seconds)
        if success:
            return {
                'success': True,
                'file_path': output_path,
                'duration': self.max_duration_seconds,
                'status': f'tiktok_channel_success_{nickname}'
            }
        return {'success': False, 'status': f'tiktok_channel_failed_{nickname}'}

    def _download_audio_chunk(self, url: str, output_path: str, start: int, duration: int) -> bool:
        """Download a chunk of audio from start with specified duration"""
        timeout = min(900, max(300, duration * 2))
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', self.quality,
            '--postprocessor-args', f"ffmpeg:-ss {start} -t {duration}",
            '--output', output_path,
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--ignore-errors',
            '--retries', '3',
            '--socket-timeout', '60',
            url
        ]
        if 'tiktok.com' in url:
            cmd.extend([
                '--extractor-args', 'tiktok:api_hostname=api22-normal-c-useast2a.tiktokv.com',
                '--user-agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)',
                '--sleep-interval', '2',  # Пауза между запросами
                '--retries', '5'          # Больше попыток (заменит предыдущий --retries)
    ])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return file_size > 50000  # Accept if file size > 50KB
            else:
                if result.stderr:
                    self.logger.warning(f"⚠️ yt-dlp error: {result.stderr[:100]}...")
                return False
        except subprocess.TimeoutExpired:
            self.logger.warning(f"⏰ Download timeout after {timeout}s")
            return False
        except Exception as e:
            self.logger.error(f"❌ Exception during download: {e}")
            return False

    def _check_video_duration(self, url: str) -> Dict:
        """Check video duration before processing"""
        
        # Пропустить проверку для профилей и каналов
        skip_patterns = [
            'tiktok.com/@',           # TikTok профили  
            'youtube.com/@',          # YouTube каналы @username
            '/channel/',              # YouTube каналы /channel/
            '/user/',                 # YouTube каналы /user/
            '/c/',                    # YouTube каналы /c/
            'twitch.tv/' + '/' not in url.split('twitch.tv/')[-1]  # Twitch каналы
        ]
        
        if any(pattern in url for pattern in skip_patterns[:5]):
            return {
                'duration': 0,
                'title': 'Profile/Channel',
                'valid': True,  # ← Разрешить обработку
                'reason': 'profile_channel_skip'
            }
        
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--quiet',
                '--no-warnings',
                '--socket-timeout', '60',  # ← Увеличить таймаут
                url
            ]
            timeout = 90  # ← Увеличить общий таймаут
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
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
            self.logger.error(f"Error checking duration: {e}")

        return {
            'duration': 0,
            'title': 'Unknown',
            'valid': False,
            'reason': 'check_failed'
        }





    def _extract_best_username(self, link_data: Dict, url: str) -> str:
        """Extract username with URL parsing priority and descriptive text filtering"""
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

    def generate_samples_report(self, extracted_samples: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive report of extracted voice samples"""
        if not output_file:
            output_file = os.path.join(self.output_dir, "voice_samples_report.txt")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("🎤 VOICE SAMPLES EXTRACTION REPORT\n")
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

    def get_extraction_summary(self, extracted_samples: List[Dict]) -> Dict:
        """Get summary statistics of extracted samples"""
        if not extracted_samples:
            return {
                'total_samples': 0,
                'platforms': {},
                'unique_users': 0,
                'max_duration_hours': self.max_duration_hours,
                'success_rate': 0
            }

        platforms = {}
        nicknames = set()
        total_duration = 0

        for sample in extracted_samples:
            platform = sample.get('platform_source', 'unknown')
            nickname = sample.get('processed_username', 'unknown')

            platforms[platform] = platforms.get(platform, 0) + 1
            nicknames.add(nickname)

            duration = sample.get('download_duration', 0)
            if isinstance(duration, (int, float)):
                total_duration += duration

        return {
            'total_samples': len(extracted_samples),
            'platforms': platforms,
            'unique_users': len(nicknames),
            'total_duration_hours': round(total_duration / 3600, 2),
            'max_duration_hours': self.max_duration_hours,
            'audio_quality': self.quality,
            'output_directory': self.output_dir
        }

    def list_extracted_files(self) -> List[str]:
        """List all extracted MP3 files in the output directory"""
        if not os.path.exists(self.output_dir):
            return []

        mp3_files = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.mp3'):
                filepath = os.path.join(self.output_dir, filename)
                mp3_files.append(filepath)

        return sorted(mp3_files)

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
