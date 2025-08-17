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
    def __init__(self, output_dir="voice_samples", sample_duration=120, quality="192"):
        self.output_dir = output_dir
        self.sample_duration = sample_duration  # seconds
        self.quality = quality  # kbps
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_voice_samples(self, confirmed_voice_links: List[Dict]) -> List[Dict]:
        """Extract voice samples from confirmed voice content with nickname and source in filename"""
        
        if not confirmed_voice_links:
            print("🔍 No confirmed voice links to extract samples from")
            return []

        print(f"🎤 Starting voice sample extraction for {len(confirmed_voice_links)} links...")
        print(f"📁 Samples will be saved to: {self.output_dir}")
        print(f"⏱️ Sample duration: {self.sample_duration} seconds")
        print(f"📝 Filename format: username_source_timestamp.mp3")
        
        # DEBUG: Print data structure for first entry
        if confirmed_voice_links:
            print(f"\n🔍 DEBUG: Available data fields: {list(confirmed_voice_links[0].keys())}")
        
        extracted_samples = []
        
        for i, link_data in enumerate(confirmed_voice_links, 1):
            url = link_data.get('url', '')
            
            # Extract best username with improved logic
            username = self._extract_best_username(link_data, url)
            platform = link_data.get('platform_type', 'unknown')
            voice_type = link_data.get('voice_type', 'unknown')
            
            if not url:
                print(f"  ⚠️ Skipping entry {i} - no URL provided")
                continue
            
            print(f"🎤 [{i}/{len(confirmed_voice_links)}] Extracting from @{username} ({platform})")
            
            # Generate filename with nickname and source
            safe_username = self._sanitize_filename(username)
            safe_platform = platform.lower() if platform else 'unknown'
            timestamp = int(time.time())
            
            # Format: username_source_timestamp.mp3
            filename = f"{safe_username}_{safe_platform}_{timestamp}"
            
            extraction_result = self._extract_audio_sample(url, filename, platform, safe_username)
            
            # Add extraction results to link data
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
                print(f"  ✅ Sample saved: {safe_username}_{safe_platform}_{timestamp}.mp3")
            else:
                print(f"  ❌ Failed: {extraction_result['status']}")
            
            time.sleep(2)  # Increased delay to prevent rate limiting
        
        successful_extractions = len(extracted_samples)
        
        print(f"\n🎤 Voice sample extraction completed!")
        print(f"📊 Total links processed: {len(confirmed_voice_links)}")
        print(f"✅ Successful extractions: {successful_extractions}")
        print(f"❌ Failed extractions: {len(confirmed_voice_links) - successful_extractions}")
        print(f"📁 Samples saved in: {self.output_dir}")
        
        if successful_extractions > 0:
            print(f"\n🎵 Sample Files Generated:")
            for sample in extracted_samples:
                filename = sample.get('sample_filename', 'N/A')
                username = sample.get('processed_username', 'unknown')
                platform = sample.get('platform_source', 'unknown')
                print(f"  📄 {filename} (@{username} from {platform})")
        
        return extracted_samples

    def _extract_best_username(self, link_data: Dict, url: str) -> str:
        """Extract username with URL parsing priority and descriptive text filtering"""
        
        # PRIORITY 1: Extract from URL (most reliable)
        username_from_url = self._extract_username_from_url(url)
        if username_from_url and len(username_from_url) > 2:
            print(f"  🔗 Extracted from URL: {username_from_url}")
            return username_from_url
        
        # PRIORITY 2: Real username fields (non-descriptive text)
        username_fields = ['username', 'screen_name', 'user_name', 'handle', 'account_name']
        
        for field in username_fields:
            value = link_data.get(field)
            if value and not self._is_empty_value(value):
                username = str(value).strip()
                if username and not self._is_descriptive_text(username):
                    print(f"  📝 Found real username in {field}: {username}")
                    return username
        
        # PRIORITY 3: Only if profile_name is not descriptive
        profile_fields = ['profile_name', 'display_name', 'name']
        for field in profile_fields:
            value = link_data.get(field)
            if value and not self._is_empty_value(value):
                username = str(value).strip()
                if username and not self._is_descriptive_text(username):
                    print(f"  📝 Clean name from {field}: {username}")
                    return username
                else:
                    print(f"  ⚠️ Skipping descriptive text: {username[:40]}...")
        
        # PRIORITY 4: Generate ID based on URL hash
        if url:
            url_hash = abs(hash(url)) % 10000
            unique_id = f"user_{url_hash}"
            print(f"  🆔 Generated URL-based ID: {unique_id}")
            return unique_id
        
        # Final fallback
        fallback_id = f"user_{int(time.time()) % 10000}"
        print(f"  ⚠️ Using fallback ID: {fallback_id}")
        return fallback_id

    def _is_empty_value(self, value) -> bool:
        """Check if a value is considered empty"""
        if value is None:
            return True
        if pd.isna(value):
            return True
        if isinstance(value, float) and (pd.isna(value) or value != value):  # NaN check
            return True
        str_val = str(value).lower().strip()
        return str_val in ['nan', '', 'none', 'null', 'undefined']

    def _is_descriptive_text(self, text: str) -> bool:
        """Check if text is descriptive rather than a real username"""
        if not text or len(text) > 30:  # Too long for username
            return True
            
        text_lower = text.lower()
        descriptive_words = [
            'check', 'pinned', 'moved', 'see', 'bio', 'link', 'description',
            'follow', 'subscribe', 'contact', 'info', 'about', 'moved to',
            'see pinned', 'check bio', 'dm for', 'business', 'inquiries'
        ]
        
        # Count descriptive words and structural indicators
        word_count = sum(1 for w in descriptive_words if w in text_lower)
        space_count = text.count(' ')
        comma_count = text.count(',')
        
        # It's descriptive if contains descriptive words OR too many spaces/punctuation
        return word_count >= 1 or space_count >= 2 or comma_count >= 1

    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from YouTube or Twitch URL"""
        
        if not url:
            return None
            
        try:
            # Clean URL
            url = url.strip()
            
            if 'youtube.com' in url or 'youtu.be' in url:
                return self._extract_youtube_username(url)
            elif 'twitch.tv' in url:
                return self._extract_twitch_username(url)
            else:
                # Try generic URL parsing
                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts:
                    return path_parts[-1][:20]
        
        except Exception as e:
            print(f"  ⚠️ Error extracting from URL: {e}")
            
        return None

    def _extract_youtube_username(self, url: str) -> str:
        """Extract username from YouTube URL"""
        
        patterns = [
            r'/channel/([^/?]+)',
            r'/user/([^/?]+)', 
            r'/c/([^/?]+)',
            r'/@([^/?]+)',
            r'/watch\?v=([^&]+)',  # Video ID as fallback
            r'youtu\.be/([^/?]+)'  # Short URL
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                username = match.group(1)[:20]
                if username and not username.startswith('UC'):  # Skip channel IDs
                    return username
                elif username.startswith('UC'):
                    return f"yt_{username[-8:]}"  # Use last 8 chars of channel ID
        
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
                # Skip common non-username paths
                if username.lower() not in ['videos', 'clips', 'collections', 'following', 'directory']:
                    return username[:20]
        
        return None

    def _sanitize_filename(self, filename) -> str:
        """Clean filename with emoji and special character handling"""
        
        if not filename or self._is_empty_value(filename):
            return f"user_{int(time.time()) % 10000}"
        
        filename = str(filename).strip()
        
        # Remove emojis and special Unicode characters
        filename = re.sub(r'[^\w\s-]', '', filename, flags=re.UNICODE)
        
        # Clean and normalize
        filename = filename.lower()
        filename = re.sub(r'\s+', '_', filename)  # Spaces to underscores
        filename = re.sub(r'[-]+', '_', filename)  # Hyphens to underscores
        filename = re.sub(r'_+', '_', filename)   # Multiple underscores to single
        filename = filename.strip('_')
        
        # Remove any remaining non-alphanumeric characters except underscore
        filename = re.sub(r'[^a-zA-Z0-9_]', '', filename)
        
        # Limit length 
        if len(filename) > 20:
            filename = filename[:20]
        
        # Final validation
        if not filename or len(filename) < 2:
            return f"user_{int(time.time()) % 10000}"
        
        return filename

    def _extract_audio_sample(self, url: str, filename: str, platform: str, nickname: str) -> Dict:
        """Extract audio sample from URL with nickname and source in filename"""
        
        output_path = os.path.join(self.output_dir, f"{filename}.mp3")
        
        try:
            if platform == 'youtube':
                return self._extract_youtube_sample(url, output_path, nickname)
            elif platform == 'twitch':
                return self._extract_twitch_sample(url, output_path, nickname)
            else:
                return {
                    'success': False,
                    'status': f'unsupported_platform: {platform}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'extraction_error_for_{nickname}: {str(e)[:100]}'
            }

    def _extract_youtube_sample(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio sample from YouTube using yt-dlp with improved error handling"""
        
        # Try multiple quality levels
        quality_options = [
            (self.quality, 240),  # Original quality, 4 min timeout
            ("128", 180),         # Lower quality, 3 min timeout  
            ("96", 120)           # Lowest quality, 2 min timeout
        ]
        
        for quality, timeout in quality_options:
            try:
                print(f"  🎧 Trying YouTube {quality} kbps (timeout: {timeout}s)")
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'mp3',
                    '--audio-quality', quality,
                    '--postprocessor-args', f'ffmpeg:-t {self.sample_duration}',
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
                    print(f"  ⚠️ Quality {quality} failed, trying next...")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout at {quality} kbps, trying lower quality...")
                continue
            except FileNotFoundError:
                return {
                    'success': False,
                    'status': 'yt-dlp_not_installed'
                }
            except Exception as e:
                print(f"  ❌ Error at {quality}: {str(e)[:50]}")
                continue
        
        return {
            'success': False,
            'status': f'youtube_failed_all_qualities_{nickname}'
        }

    def _extract_twitch_sample(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio sample from Twitch with improved VOD handling"""
        
        # Check if it's a channel URL (not a specific video)
        if '/videos/' not in url and '/clip/' not in url:
            # It's a plain channel URL, try to find recent VODs
            if url.endswith('/videos'):
                return self._try_get_recent_twitch_vod(url, output_path, nickname)
            else:
                # Add /videos to get channel videos
                videos_url = url.rstrip('/') + '/videos'
                return self._try_get_recent_twitch_vod(videos_url, output_path, nickname)
        
        # It's a direct VOD or clip URL
        return self._extract_direct_twitch_content(url, output_path, nickname)

    def _try_get_recent_twitch_vod(self, videos_url: str, output_path: str, nickname: str) -> Dict:
        """Try to get recent VOD from Twitch channel videos page with improved error handling"""
        
        try:
            print(f"  🔍 Searching recent VODs for @{nickname}...")
            
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--playlist-end', '1',  # Get only the most recent VOD
                '--quiet',
                '--no-warnings',
                '--ignore-errors',
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
                        
                        if vod_url:
                            print(f"  🎬 Found recent VOD: {vod_title}...")
                            return self._extract_direct_twitch_content(vod_url, output_path, nickname)
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Failed to parse VOD info: {e}")
            
            return {
                'success': False,
                'status': f'no_recent_vods_found_for_{nickname}'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': f'twitch_vod_search_timeout_for_{nickname}'
            }
        except Exception as e:
            return {
                'success': False,
                'status': f'twitch_vod_search_failed_for_{nickname}: {str(e)[:100]}'
            }

    def _extract_direct_twitch_content(self, url: str, output_path: str, nickname: str) -> Dict:
        """Extract audio from direct Twitch VOD with retries and quality fallback"""
        
        # Try multiple qualities with progressively shorter timeouts
        quality_options = [
            (self.quality, 360),  # Original quality, 6 min timeout
            ("128", 300),         # Lower quality, 5 min timeout  
            ("96", 240),          # Even lower, 4 min timeout
            ("64", 180)           # Lowest quality, 3 min timeout
        ]
        
        for quality, timeout in quality_options:
            try:
                print(f"  🎧 Trying Twitch {quality} kbps (timeout: {timeout}s)")
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'mp3',
                    '--audio-quality', quality,
                    '--postprocessor-args', f'ffmpeg:-t {self.sample_duration}',
                    '--output', output_path.replace('.mp3', '.%(ext)s'),
                    '--quiet',
                    '--no-warnings',
                    '--ignore-errors',
                    '--fragment-retries', '3',        # Retry fragments
                    '--retries', '3',                 # Retry downloads
                    '--max-filesize', '100M',         # Limit file size
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
                    print(f"  ⚠️ Quality {quality} failed, trying next...")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout at {quality} kbps, trying lower quality...")
                continue
            except FileNotFoundError:
                return {
                    'success': False,
                    'status': 'yt-dlp_not_installed'
                }
            except Exception as e:
                print(f"  ❌ Error at {quality}: {str(e)[:50]}")
                continue
        
        return {
            'success': False,
            'status': f'twitch_failed_all_qualities_{nickname}'
        }

    def generate_samples_report(self, extracted_samples: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive report of extracted voice samples"""
        
        if not output_file:
            output_file = os.path.join(self.output_dir, "voice_samples_report.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎤 VOICE SAMPLES EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples extracted: {len(extracted_samples)}\n")
            f.write(f"Sample duration: {self.sample_duration} seconds\n")
            f.write(f"Audio quality: {self.quality} kbps\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Group by platform and other metrics
            platforms = {}
            voice_types = {}
            nicknames = []
            
            for sample in extracted_samples:
                platform = sample.get('platform_source', 'unknown')
                voice_type = sample.get('voice_type', 'unknown')
                nickname = sample.get('processed_username', 'unknown')
                
                platforms[platform] = platforms.get(platform, 0) + 1
                voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
                nicknames.append(nickname)
            
            f.write("📊 BREAKDOWN BY PLATFORM:\n")
            for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {platform}: {count} samples\n")
            
            f.write("\n📊 BREAKDOWN BY VOICE TYPE:\n")
            for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {voice_type}: {count} samples\n")
            
            unique_nicknames = list(set(nicknames))
            f.write(f"\n👥 UNIQUE USERS EXTRACTED: {len(unique_nicknames)}\n")
            f.write(f"📝 Users: {', '.join(sorted(unique_nicknames))}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("📋 DETAILED SAMPLE LIST\n")
            f.write("=" * 60 + "\n\n")
            
            for i, sample in enumerate(extracted_samples, 1):
                nickname = sample.get('processed_username', 'unknown')
                platform = sample.get('platform_source', 'unknown')
                voice_type = sample.get('voice_type', 'unknown')
                sample_filename = sample.get('sample_filename', 'N/A')
                confidence = sample.get('voice_confidence', 'unknown')
                original_username = sample.get('original_username', 'N/A')
                
                f.write(f"{i:2d}. {sample_filename}\n")
                f.write(f"    👤 Processed User: @{nickname}\n")
                f.write(f"    📝 Original Username: {original_username}\n")
                f.write(f"    🔗 Source: {platform}\n")
                f.write(f"    🎙️ Voice Type: {voice_type}\n")
                f.write(f"    📊 Confidence: {confidence}\n")
                f.write(f"    🌐 URL: {sample.get('url', 'N/A')[:70]}...\n")
                f.write(f"    📁 File: {sample.get('sample_file', 'N/A')}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("🎵 USAGE INSTRUCTIONS:\n")
            f.write("- All samples are in MP3 format\n")
            f.write("- Filename format: username_source_timestamp.mp3\n")
            f.write("- Ready for voice analysis or machine learning\n")
            f.write("- Check extraction_status for any failed extractions\n")
        
        print(f"📄 Comprehensive voice samples report saved: {output_file}")
        return output_file

    def get_extraction_summary(self, extracted_samples: List[Dict]) -> Dict:
        """Get summary statistics of extracted samples"""
        
        if not extracted_samples:
            return {
                'total_samples': 0,
                'platforms': {},
                'voice_types': {},
                'unique_users': 0,
                'success_rate': 0
            }
        
        platforms = {}
        voice_types = {}
        nicknames = set()
        
        for sample in extracted_samples:
            platform = sample.get('platform_source', 'unknown')
            voice_type = sample.get('voice_type', 'unknown')
            nickname = sample.get('processed_username', 'unknown')
            
            platforms[platform] = platforms.get(platform, 0) + 1
            voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
            nicknames.add(nickname)
        
        return {
            'total_samples': len(extracted_samples),
            'platforms': platforms,
            'voice_types': voice_types,
            'unique_users': len(nicknames),
            'sample_duration': self.sample_duration,
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
                    print(f"  🗑️ Cleaned: {filename}")
                except Exception as e:
                    print(f"  ⚠️ Could not clean {filename}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} temporary files")

# Command line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Sample Extractor with Enhanced Username Detection and Timeout Handling")
    parser.add_argument("--input", help="Input CSV file with confirmed voice links")
    parser.add_argument("--output-dir", default="voice_samples", help="Output directory")
    parser.add_argument("--duration", type=int, default=30, help="Sample duration in seconds")
    parser.add_argument("--quality", default="192", help="Audio quality in kbps")
    parser.add_argument("--list-files", action="store_true", help="List extracted MP3 files")
    parser.add_argument("--clean-temp", action="store_true", help="Clean temporary files")
    parser.add_argument("--debug-csv", help="Debug CSV structure")
    
    args = parser.parse_args()
    
    # Debug CSV structure if requested
    if args.debug_csv:
        if os.path.exists(args.debug_csv):
            try:
                df = pd.read_csv(args.debug_csv)
                print(f"🔍 CSV Structure Debug:")
                print(f"  📊 Rows: {len(df)}, Columns: {len(df.columns)}")
                print(f"  📋 Columns: {list(df.columns)}")
                print(f"  📝 First row sample:")
                for col in df.columns:
                    val = df.iloc[0][col]
                    print(f"    {col}: {repr(val)} ({type(val)})")
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print(f"❌ CSV file not found: {args.debug_csv}")
        exit(0)
    
    extractor = VoiceSampleExtractor(
        output_dir=args.output_dir,
        sample_duration=args.duration,
        quality=args.quality
    )
    
    if args.clean_temp:
        extractor.clean_temp_files()
    
    if args.list_files:
        files = extractor.list_extracted_files()
        print(f"📁 Found {len(files)} MP3 files in {args.output_dir}:")
        for file in files:
            print(f"  🎵 {os.path.basename(file)}")
    
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            exit(1)
        
        try:
            df = pd.read_csv(args.input)
            confirmed_voice = df.to_dict('records')
            
            print(f"📥 Loaded {len(confirmed_voice)} confirmed voice links")
            
            extracted_samples = extractor.extract_voice_samples(confirmed_voice)
            
            if extracted_samples:
                report_file = extractor.generate_samples_report(extracted_samples)
                summary = extractor.get_extraction_summary(extracted_samples)
                
                print(f"\n📊 EXTRACTION SUMMARY:")
                print(f"  🎵 Total samples: {summary['total_samples']}")
                print(f"  👥 Unique users: {summary['unique_users']}")
                print(f"  📁 Output directory: {summary['output_directory']}")
                
                # Clean temporary files after extraction
                extractor.clean_temp_files()
                
            else:
                print("❌ No samples were successfully extracted")
                print("💡 Check your internet connection and yt-dlp installation")
                
        except Exception as e:
            print(f"❌ Error processing input file: {e}")
            import traceback
            traceback.print_exc()
