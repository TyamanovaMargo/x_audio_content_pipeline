import requests
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse
import time
import subprocess
import os
import tempfile
import logging
import random
import pandas as pd  # ADDED - Fix for pd.isna() NameError

# Try to import advanced dependencies with fallbacks
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("⚠️ Warning: yt-dlp not available. Using basic detection only.")

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ Warning: PyAnnote not available. Using basic detection only.")

class AudioContentDetector:
    """Enhanced audio detector without Twitch API dependency and improved false positive handling"""
    
    def __init__(self, timeout=10, min_duration=30, max_duration=3600, huggingface_token=None):
        self.timeout = timeout
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.huggingface_token = huggingface_token
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize VAD pipeline if available
        self.vad_pipeline = None
        if PYANNOTE_AVAILABLE and huggingface_token:
            try:
                self.vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=huggingface_token
                )
                print("✅ PyAnnote VAD model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load PyAnnote VAD: {e}")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a placeholder - ADDED"""
        if not url or not isinstance(url, str):
            return False
        
        # Check for placeholder strings
        invalid_placeholders = [
            'tiktok_default',
            'youtube_default', 
            'twitch_default',
            'unknown',
            'default'
        ]
        
        if url.lower() in invalid_placeholders:
            return False
        
        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return False
            
        return True

    def detect_audio_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Enhanced audio content detection with reduced false positives"""
        if not audio_links:
            print("🔍 No audio links to detect")
            return []
        
        print(f"🎵 Starting enhanced audio detection for {len(audio_links)} links...")
        print(f"⏰ Duration filter: {self.min_duration}s - {self.max_duration}s")
        print(f"🧠 VAD available: {'Yes' if self.vad_pipeline else 'No'}")
        print(f"📥 yt-dlp available: {'Yes' if YT_DLP_AVAILABLE else 'No'}")
        print(f"🎯 Strategy: Smart VAD to avoid subtitle false positives")
        
        audio_detected_links = []
        
        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')
            platform = link_data.get('platform_type', 'unknown')
            username = link_data.get('username', 'unknown')
            
            # FIXED - Handle NaN/float values for platform and username
            if not isinstance(platform, str) or pd.isna(platform):
                platform = 'unknown'
            if not isinstance(username, str) or pd.isna(username):
                username = 'unknown'
            
            # ADDED - URL validation before processing
            if not url or not self._is_valid_url(url):
                print(f"\n🔍 [{i}/{len(audio_links)}] {platform.upper()} - @{username}")
                print(f"❌ Invalid URL: {url}")
                link_data.update({
                    'has_audio': False,
                    'audio_confidence': 'low',
                    'audio_type': 'invalid_url',
                    'detection_status': 'invalid_url_skipped'
                })
                continue
            
            print(f"\n🔍 [{i}/{len(audio_links)}] {platform.upper()} - @{username}")
            print(f"🔗 URL: {url[:80]}...")
            
            try:
                # Check duration first if not already checked
                if 'duration' not in link_data or 'valid' not in link_data:
                    from step4_audio_filter import AudioContentFilter
                    filter_obj = AudioContentFilter(self.min_duration, self.max_duration)
                    duration_info = filter_obj._check_video_duration(url)
                    
                    if duration_info and not duration_info['valid']:
                        print(f"⏰ Skipped: {duration_info['reason']} ({duration_info['duration']}s)")
                        continue
                    elif duration_info:
                        link_data.update(duration_info)
                
                # Enhanced detection with false positive handling
                result = self._comprehensive_audio_detection_enhanced(url, platform, link_data)
                
                # Update link data with results
                link_data.update({
                    'has_audio': result['has_audio'],
                    'audio_confidence': result['confidence'],
                    'audio_type': result.get('audio_type'),
                    'detection_status': result['status']
                })
                
                if result['has_audio']:
                    audio_detected_links.append(link_data)
                    status_emoji = "✅" if result['confidence'] == 'high' else "⚠️"
                    print(f"{status_emoji} Audio detected: {result['confidence']} confidence - {result['audio_type']}")
                else:
                    print(f"❌ No voice: {result['status']}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")
                link_data.update({
                    'has_audio': False,
                    'audio_confidence': 'low',
                    'audio_type': 'error',
                    'detection_status': f'processing_error: {str(e)}'
                })
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"\n✅ Enhanced audio detection completed: {len(audio_detected_links)}/{len(audio_links)} valid")
        return audio_detected_links

    def _comprehensive_audio_detection_enhanced(self, url: str, platform: str, link_data: Dict) -> Dict:
        """Enhanced detection with smart VAD to avoid false positives"""
        
        # Level 1: Fast HTML heuristics
        print("   📄 Level 1: HTML heuristics...")
        html_result = self._fast_html_detection(url, platform)
        
        # Check for captions/subtitles (potential false positive source)
        has_captions = self._check_for_captions(url, platform)
        html_result['has_captions'] = has_captions
        
        # Level 2: ALWAYS run VAD for YouTube videos with captions to verify actual speech
        if (platform == 'youtube' and has_captions) or html_result['confidence'] in ['low', 'medium']:
            if self.vad_pipeline and YT_DLP_AVAILABLE:
                print("   🎤 Level 2: VAD verification (caption check)...")
                vad_result = self._smart_vad_analysis(url, html_result, link_data)
                return vad_result
            else:
                # If no VAD available but captions detected, be more cautious
                if has_captions:
                    print("   ⚠️ Captions detected but no VAD - lowering confidence")
                    html_result['confidence'] = 'medium'
                    html_result['status'] += '; captions_present_no_vad'
        
        # Level 3: Enhanced platform-specific checks
        if platform == 'youtube':
            enhanced_result = self._enhanced_youtube_detection(url, html_result)
        elif platform == 'twitch':
            enhanced_result = self._enhanced_twitch_detection(url, html_result)
        elif platform == 'tiktok':
            enhanced_result = self._enhanced_tiktok_detection(url, html_result)
        else:
            enhanced_result = html_result
        
        return enhanced_result

    def _check_for_captions(self, url: str, platform: str) -> bool:
        """Check if video has captions/subtitles"""
        if platform != 'youtube':
            return False
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # Look for subtitle indicators
            caption_indicators = [
                'captiontrack',
                'captions',
                'subtitle',
                'closed captions',
                'cc',
                '"captions":',
                'captionsmetadata'
            ]
            
            return any(indicator in content for indicator in caption_indicators)
        except:
            return False

    def _smart_vad_analysis(self, url: str, current_result: Dict, link_data: Dict) -> Dict:
        """Smart VAD analysis that handles caption false positives"""
        print("   📥 Downloading strategic audio segment for VAD...")
        
        audio_path = self._download_strategic_audio_segment(url, link_data, link_data.get('platform_type', 'unknown'))
        if not audio_path:
            current_result['status'] += '; vad_download_failed'
            # If captions but no VAD possible, be conservative
            if current_result.get('has_captions', False):
                current_result['confidence'] = 'medium'
                current_result['status'] += '; captions_but_no_vad_verification'
            return current_result
        
        try:
            print("   🧠 Running smart VAD analysis...")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Run VAD
            vad_result = self.vad_pipeline({
                'waveform': waveform,
                'sample_rate': sample_rate
            })
            
            # Calculate speech ratio
            total_duration = len(waveform[0]) / sample_rate
            speech_duration = sum(
                segment.duration for segment in vad_result.get_timeline()
            )
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
            
            print(f"   📊 Speech ratio: {speech_ratio:.2%}")
            
            # Smart decision making based on captions + VAD
            has_captions = current_result.get('has_captions', False)
            
            if has_captions and speech_ratio < 0.05:  # 5% threshold
                # CASE: Captions + very low speech = likely music with subtitles
                return {
                    'has_audio': False,
                    'confidence': 'low',
                    'audio_type': 'music_with_subtitles',
                    'status': f'vad_captions_false_positive: {speech_ratio:.1%} speech, has_captions'
                }
            
            elif has_captions and speech_ratio < 0.15:  # 15% threshold
                # CASE: Captions + low speech = uncertain
                return {
                    'has_audio': False,
                    'confidence': 'low',
                    'audio_type': 'mixed_uncertain',
                    'status': f'vad_captions_low_speech: {speech_ratio:.1%} speech, has_captions'
                }
            
            elif speech_ratio > 0.3:  # 30% threshold
                # CASE: High speech ratio = confirmed voice
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'speech',
                    'status': f'vad_confirmed_speech: {speech_ratio:.1%} speech'
                }
            
            elif speech_ratio > 0.1:  # 10% threshold
                # CASE: Medium speech ratio
                return {
                    'has_audio': True,
                    'confidence': 'medium',
                    'audio_type': 'mixed',
                    'status': f'vad_some_speech: {speech_ratio:.1%} speech'
                }
            
            else:
                # CASE: Very low speech
                return {
                    'has_audio': False,
                    'confidence': 'low',
                    'audio_type': 'music_or_silence',
                    'status': f'vad_no_speech: {speech_ratio:.1%} speech'
                }
                
        except Exception as e:
            current_result['status'] += f'; vad_error: {str(e)}'
            return current_result
        finally:
            # Always cleanup
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    def _fast_html_detection(self, url: str, platform: str) -> Dict:
        """Fast HTML-based detection"""
        if platform == 'youtube':
            return self._detect_youtube_audio(url)
        elif platform == 'twitch':
            return self._detect_twitch_audio(url)
        elif platform == 'tiktok':
            return self._detect_tiktok_audio(url)
        else:
            return {
                'has_audio': True,
                'confidence': 'medium',
                'audio_type': 'unknown',
                'status': 'unknown_platform_assumed_audio'
            }

    def _detect_youtube_audio(self, url: str) -> Dict:
        """Enhanced YouTube audio detection"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # YouTube audio indicators
            strong_audio_indicators = [
                '"hasaudio":true',
                '"audiotrack"',
                'itag.*?audio',
                'audio/.*?webm',
                'audio/.*?mp4'
            ]
            
            found_indicators = []
            for pattern in strong_audio_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    found_indicators.append(pattern)
            
            # Additional checks
            has_video_element = '<video' in content
            has_audio_mention = 'audio' in content
            
            # Content type classification
            content_type = self._classify_youtube_content(content)
            
            # Decision logic
            if len(found_indicators) >= 2:
                confidence = 'high'
                has_audio = True
            elif len(found_indicators) >= 1 or (has_video_element and has_audio_mention):
                confidence = 'medium'
                has_audio = True
            elif has_video_element:  # YouTube videos usually have audio
                confidence = 'medium'
                has_audio = True
            else:
                confidence = 'low'
                has_audio = False
            
            return {
                'has_audio': has_audio,
                'confidence': confidence,
                'audio_type': content_type,
                'status': f'youtube_audio: {len(found_indicators)} indicators, {content_type}'
            }
            
        except Exception as e:
            return {
                'has_audio': True,  # Assume YouTube has audio by default
                'confidence': 'medium',
                'audio_type': 'youtube_default',
                'status': f'youtube_error_default_true: {str(e)}'
            }

    def _detect_twitch_audio(self, url: str) -> Dict:
        """Enhanced Twitch audio detection without API"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # Determine Twitch content type from HTML content
            stream_type = self._classify_twitch_content_from_html(content)
            
            # Twitch almost always has audio
            if stream_type == 'just_chatting':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'live_talk',
                    'status': 'twitch_just_chatting_detected_html'
                }
            
            elif stream_type == 'talk_show':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'talk_show',
                    'status': 'twitch_talk_show_detected_html'
                }
            
            elif stream_type == 'music_category':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'music',
                    'status': 'twitch_music_category_detected_html'
                }
            
            elif stream_type == 'gaming_with_commentary':
                return {
                    'has_audio': True,
                    'confidence': 'medium',
                    'audio_type': 'gaming_commentary',
                    'status': 'twitch_gaming_with_voice_detected_html'
                }
            
            else:  # Any other Twitch content
                return {
                    'has_audio': True,
                    'confidence': 'medium',
                    'audio_type': 'twitch_stream',
                    'status': 'twitch_general_stream_html'
                }
                
        except Exception as e:
            return {
                'has_audio': True,  # Twitch almost always has audio
                'confidence': 'medium',
                'audio_type': 'twitch_default',
                'status': f'twitch_error_default_true: {str(e)}'
            }

    def _detect_tiktok_audio(self, url: str) -> dict:
        """TikTok videos almost always contain audio track (music/voice)"""
        try:
            return {
                'has_audio': True,
                'confidence': 'high',
                'audio_type': 'tiktok_default',
                'status': 'tiktok_audio_assumed'
            }
        except Exception as e:
            return {
                'has_audio': True,
                'confidence': 'medium',
                'audio_type': 'tiktok_default',
                'status': f'tiktok_error_assumed_audio: {str(e)}'
            }

    def _classify_youtube_content(self, content: str) -> str:
        """Classify YouTube content type for better voice detection"""
        if any(keyword in content for keyword in [
            'podcast', 'interview', 'talk', 'discussion', 'conversation'
        ]):
            return 'speech_content'
        elif any(keyword in content for keyword in [
            'tutorial', 'lecture', 'explanation', 'review', 'analysis'
        ]):
            return 'educational_content'
        elif any(keyword in content for keyword in [
            'music', 'song', 'album', 'artist', 'band', 'mv', 'official video'
        ]):
            return 'music_content'
        elif any(keyword in content for keyword in [
            'gameplay', 'gaming', 'game', 'let\'s play', 'walkthrough'
        ]):
            return 'gaming_content'
        else:
            return 'mixed_content'

    def _classify_twitch_content_from_html(self, content: str) -> str:
        """Classify Twitch stream type from HTML (no API)"""
        if 'just chatting' in content or 'justchatting' in content:
            return 'just_chatting'
        elif any(keyword in content for keyword in [
            'talk show', 'podcast', 'interview', 'discussion'
        ]):
            return 'talk_show'
        elif any(keyword in content for keyword in [
            'music & performing arts', 'dj', 'music', 'radio'
        ]):
            return 'music_category'
        elif any(keyword in content for keyword in [
            'gaming', 'gameplay', 'playing'
        ]) and any(keyword in content for keyword in [
            'commentary', 'talking', 'chat'
        ]):
            return 'gaming_with_commentary'
        else:
            return 'general_stream'

    def _enhanced_youtube_detection(self, url: str, html_result: Dict) -> Dict:
        """Enhanced YouTube detection without API"""
        return html_result

    def _enhanced_twitch_detection(self, url: str, html_result: Dict) -> Dict:
        """Enhanced Twitch detection without API"""
        return html_result

    def _enhanced_tiktok_detection(self, url: str, html_result: Dict) -> Dict:
        """Enhanced TikTok detection without API"""
        return html_result

    def _calculate_optimal_start_time(self, link_data: Dict, platform: str) -> int:
        """Calculate optimal start time for audio segment sampling"""
        duration = link_data.get('duration', 300)
        
        if duration <= 60:
            return max(5, duration // 4)
        elif duration <= 300:
            return random.randint(30, min(60, duration - 30))
        elif duration <= 900:
            return random.randint(60, min(180, duration - 60))
        else:
            segments = [duration // 4, duration // 2, (duration * 3) // 4]
            return random.choice(segments)

    def _download_strategic_audio_segment(self, url: str, link_data: Dict, platform: str) -> Optional[str]:
        """Download strategic audio segment (NOT from beginning)"""
        if not YT_DLP_AVAILABLE:
            return None
        
        start_time = self._calculate_optimal_start_time(link_data, platform)
        duration = 15
        
        print(f"   📍 Sampling from {start_time}s-{start_time + duration}s")
        
        temp_file = tempfile.mktemp(suffix='.wav')
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'extractaudio': True,
                'audioformat': 'wav',
                'outtmpl': temp_file,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'download_sections': [{
                    'start_time': start_time,
                    'end_time': start_time + duration
                }],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            possible_paths = [
                temp_file,
                temp_file + '.wav',
                temp_file.replace('.wav', '') + '.wav'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if path != temp_file:
                        os.rename(path, temp_file)
                    return temp_file
            
            return None
            
        except Exception as e:
            self.logger.error(f"Strategic segment download failed: {e}")
            return None
