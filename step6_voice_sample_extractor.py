import os
import subprocess
import requests
from typing import List, Dict
from urllib.parse import urlparse
import time
import logging

class VoiceSampleExtractor:
    def __init__(self, output_dir="voice_samples", sample_duration=30, quality="192"):
        self.output_dir = output_dir
        self.sample_duration = sample_duration  # seconds
        self.quality = quality  # kbps
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_voice_samples(self, confirmed_voice_links: List[Dict]) -> List[Dict]:
        """Extract voice samples from confirmed voice content"""
        
        if not confirmed_voice_links:
            print("🔍 No confirmed voice links to extract samples from")
            return []

        print(f"🎤 Starting voice sample extraction for {len(confirmed_voice_links)} links...")
        print(f"📁 Samples will be saved to: {self.output_dir}")
        print(f"⏱️ Sample duration: {self.sample_duration} seconds")
        
        extracted_samples = []
        
        for i, link_data in enumerate(confirmed_voice_links, 1):
            url = link_data.get('url', '')
            username = link_data.get('username', 'unknown')
            platform = link_data.get('platform_type', 'unknown')
            voice_type = link_data.get('voice_type', 'unknown')
            
            if not url:
                continue
                
            print(f"🎤 [{i}/{len(confirmed_voice_links)}] Extracting from {username} ({platform}): {voice_type}")
            
            # Generate safe filename
            safe_username = self._sanitize_filename(username)
            filename = f"{safe_username}_{platform}_{voice_type}_{int(time.time())}"
            
            extraction_result = self._extract_audio_sample(url, filename, platform)
            
            # Add extraction results to link data
            link_data.update({
                'sample_extracted': extraction_result['success'],
                'sample_file': extraction_result.get('file_path'),
                'extraction_status': extraction_result['status'],
                'sample_duration': self.sample_duration,
                'sample_quality': self.quality
            })
            
            if extraction_result['success']:
                extracted_samples.append(link_data)
                print(f"  ✅ Sample saved: {extraction_result['file_path']}")
            else:
                print(f"  ❌ Failed: {extraction_result['status']}")
            
            time.sleep(1)  # Rate limiting
        
        successful_extractions = len(extracted_samples)
        
        print(f"\n🎤 Voice sample extraction completed!")
        print(f"📊 Total links processed: {len(confirmed_voice_links)}")
        print(f"✅ Successful extractions: {successful_extractions}")
        print(f"❌ Failed extractions: {len(confirmed_voice_links) - successful_extractions}")
        print(f"📁 Samples saved in: {self.output_dir}")
        
        return extracted_samples

    def _extract_audio_sample(self, url: str, filename: str, platform: str) -> Dict:
        """Extract audio sample from URL"""
        
        output_path = os.path.join(self.output_dir, f"{filename}.mp3")
        
        try:
            if platform == 'youtube':
                return self._extract_youtube_sample(url, output_path)
            elif platform == 'twitch':
                return self._extract_twitch_sample(url, output_path)
            else:
                return {
                    'success': False,
                    'status': f'unsupported_platform: {platform}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'extraction_error: {str(e)}'
            }

    def _extract_youtube_sample(self, url: str, output_path: str) -> Dict:
        """Extract audio sample from YouTube using yt-dlp"""
        
        try:
            # Using yt-dlp (modern youtube-dl fork)
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', self.quality,
                '--postprocessor-args', f'-t {self.sample_duration}',  # Limit duration
                '--output', output_path.replace('.mp3', '.%(ext)s'),
                '--no-playlist',
                '--quiet',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return {
                    'success': True,
                    'file_path': output_path,
                    'status': 'youtube_extraction_successful'
                }
            else:
                # Fallback: try pytube method
                return self._extract_youtube_pytube(url, output_path)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': 'youtube_extraction_timeout'
            }
        except FileNotFoundError:
            # yt-dlp not installed, try pytube
            return self._extract_youtube_pytube(url, output_path)
        except Exception as e:
            return {
                'success': False,
                'status': f'youtube_extraction_failed: {str(e)}'
            }

    def _extract_youtube_pytube(self, url: str, output_path: str) -> Dict:
        """Extract audio using pytube (fallback method)"""
        
        try:
            from pytube import YouTube
            from pydub import AudioSegment
            
            yt = YouTube(url)
            
            # Get audio stream
            audio_stream = yt.streams.filter(only_audio=True).first()
            if not audio_stream:
                return {
                    'success': False,
                    'status': 'youtube_no_audio_stream'
                }
            
            # Download to temporary file
            temp_file = audio_stream.download(output_path=self.output_dir, filename="temp_audio")
            
            # Load and trim audio
            audio = AudioSegment.from_file(temp_file)
            
            # Trim to sample duration
            if len(audio) > self.sample_duration * 1000:  # pydub uses milliseconds
                audio = audio[:self.sample_duration * 1000]
            
            # Export as MP3
            audio.export(output_path, format="mp3", bitrate=f"{self.quality}k")
            
            # Clean up temp file
            os.remove(temp_file)
            
            return {
                'success': True,
                'file_path': output_path,
                'status': 'youtube_pytube_successful'
            }
            
        except ImportError:
            return {
                'success': False,
                'status': 'youtube_pytube_not_installed'
            }
        except Exception as e:
            return {
                'success': False,
                'status': f'youtube_pytube_failed: {str(e)}'
            }

    def _extract_twitch_sample(self, url: str, output_path: str) -> Dict:
        """Extract audio sample from Twitch"""
        
        try:
            # Method 1: Try yt-dlp for Twitch clips/VODs
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', self.quality,
                '--postprocessor-args', f'-t {self.sample_duration}',
                '--output', output_path.replace('.mp3', '.%(ext)s'),
                '--quiet',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return {
                    'success': True,
                    'file_path': output_path,
                    'status': 'twitch_extraction_successful'
                }
            else:
                # Method 2: Try alternative approach for live streams
                return self._extract_twitch_alternative(url, output_path)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'status': 'twitch_extraction_timeout'
            }
        except FileNotFoundError:
            return self._extract_twitch_alternative(url, output_path)
        except Exception as e:
            return {
                'success': False,
                'status': f'twitch_extraction_failed: {str(e)}'
            }

    def _extract_twitch_alternative(self, url: str, output_path: str) -> Dict:
        """Alternative method for Twitch audio extraction"""
        
        try:
            # For live streams, we might need to record for sample_duration
            # This is a simplified approach - you might need more sophisticated methods
            
            if '/clip/' in url:
                # For clips, try direct m3u8 extraction
                return self._extract_twitch_clip_direct(url, output_path)
            else:
                # For live streams, indicate limitation
                return {
                    'success': False,
                    'status': 'twitch_live_stream_not_supported'
                }
                
        except Exception as e:
            return {
                'success': False,
                'status': f'twitch_alternative_failed: {str(e)}'
            }

    def _extract_twitch_clip_direct(self, url: str, output_path: str) -> Dict:
        """Extract audio from Twitch clip using direct method"""
        
        try:
            # This would require Twitch API or scraping to get direct media URLs
            # For now, return a placeholder
            return {
                'success': False,
                'status': 'twitch_clip_extraction_requires_api'
            }
            
        except Exception as e:
            return {
                'success': False,
                'status': f'twitch_clip_extraction_failed: {str(e)}'
            }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe saving"""
        
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        return filename.strip()

    def generate_samples_report(self, extracted_samples: List[Dict], output_file: str = None) -> str:
        """Generate a report of extracted voice samples"""
        
        if not output_file:
            output_file = os.path.join(self.output_dir, "voice_samples_report.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎤 VOICE SAMPLES EXTRACTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples: {len(extracted_samples)}\n")
            f.write(f"Sample duration: {self.sample_duration} seconds\n")
            f.write(f"Audio quality: {self.quality} kbps\n\n")
            
            # Group by platform
            platforms = {}
            voice_types = {}
            
            for sample in extracted_samples:
                platform = sample.get('platform_type', 'unknown')
                voice_type = sample.get('voice_type', 'unknown')
                
                platforms[platform] = platforms.get(platform, 0) + 1
                voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
            
            f.write("📊 BREAKDOWN BY PLATFORM:\n")
            for platform, count in platforms.items():
                f.write(f"  {platform}: {count}\n")
            
            f.write("\n📊 BREAKDOWN BY VOICE TYPE:\n")
            for voice_type, count in voice_types.items():
                f.write(f"  {voice_type}: {count}\n")
            
            f.write("\n📋 DETAILED SAMPLE LIST:\n")
            f.write("-" * 50 + "\n")
            
            for i, sample in enumerate(extracted_samples, 1):
                username = sample.get('username', 'unknown')
                platform = sample.get('platform_type', 'unknown')
                voice_type = sample.get('voice_type', 'unknown')
                sample_file = sample.get('sample_file', 'N/A')
                
                f.write(f"{i:2d}. @{username} ({platform})\n")
                f.write(f"    Voice Type: {voice_type}\n")
                f.write(f"    Sample File: {sample_file}\n")
                f.write(f"    Original URL: {sample.get('url', 'N/A')[:80]}...\n\n")
        
        print(f"📄 Voice samples report saved: {output_file}")
        return output_file
