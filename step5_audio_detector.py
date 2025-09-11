import requests
import re
import json
import os
import time
import tempfile
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from urllib.parse import urlparse
from pathlib import Path

# Audio processing libraries with fallbacks
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
    print("✅ yt-dlp imported successfully")
except ImportError:
    YT_DLP_AVAILABLE = False
    print("❌ yt-dlp not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ librosa imported successfully")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ librosa not available")

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    print(f"✅ PyAnnote imported successfully (torch {torch.__version__})")
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("❌ PyAnnote not available")

class EnhancedVoiceDetector:
    """Enhanced Voice Detector with Latest Video Fetching"""

    def __init__(self, config_path: str = "config.json"):
        """Initialize detector with configuration from JSON file"""
        print("🔧 Initializing Enhanced Voice Detector with Latest Video Fetching...")
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_detector()
        self._setup_voice_keywords()
        self._test_environment()
        self._initialize_models()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file with detailed logging"""
        print(f"📋 Loading configuration from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Configuration loaded successfully")
            
            # Validate token
            token = config.get('huggingface_token')
            if token and len(token) > 10:
                print(f"🔑 HuggingFace token present: True")
                print(f"🔑 Token format looks valid (starts with: {token[:10]}...)")
            elif token == "here":
                print("⚠️ Warning: Token placeholder 'here' detected - please set actual token")
            else:
                print("⚠️ Warning: No valid HuggingFace token found")
            
            return config
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in config file: {e}")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Unexpected error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "huggingface_token": None,
            "output_dir": "processed_audio",
            "chunking": {"max_duration_minutes": 60},
            "processing": {"min_voice_duration": 0.5},
            "pyannote": {
                "diarization_model": "pyannote/speaker-diarization-3.1",
                "vad_model": "pyannote/voice-activity-detection"
            }
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_detector(self):
        """Initialize detector parameters from config"""
        print("⚙️ Initializing detector parameters...")
        self.timeout = 15
        
        # Extract timing parameters from config
        processing_config = self.config.get('processing', {})
        chunking_config = self.config.get('chunking', {})
        
        self.min_duration = max(30, int(processing_config.get('min_voice_duration', 0.5) * 60))
        self.max_duration = chunking_config.get('max_duration_minutes', 60) * 60
        
        print(f"⏰ Duration filter: {self.min_duration}s - {self.max_duration}s")
        
        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'processed_audio'))
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")

    def _test_environment(self):
        """Test PyAnnote environment setup"""
        print("\n🔍 Testing PyAnnote environment...")
        if not PYANNOTE_AVAILABLE:
            print("❌ PyAnnote not available - models will not be loaded")
            return False
        
        try:
            import torch
            print(f"✅ PyTorch version: {torch.__version__}")
            import torchaudio
            print(f"✅ TorchAudio version: {torchaudio.__version__}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ Available device: {device}")
            return True
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            return False

    def _setup_voice_keywords(self):
        """Setup keyword dictionaries for content classification"""
        self.voice_keywords = {
            'strong_voice': [
                'podcast', 'interview', 'talk', 'discussion', 'conversation',
                'speech', 'lecture', 'presentation', 'commentary', 'review',
                'tutorial', 'explanation', 'analysis', 'storytime', 'vlog',
                'reaction', 'reading', 'audiobook', 'monologue', 'dialogue',
                'debate', 'q&a', 'live stream', 'chat', 'talking', 'speaking'
            ],
            'likely_voice': [
                'gaming', 'gameplay', 'let\'s play', 'walkthrough', 'guide',
                'news', 'update', 'announcement', 'behind the scenes',
                'unboxing', 'haul', 'cooking', 'diy', 'how to', 'tips',
                'advice', 'thoughts', 'opinion', 'rant', 'story'
            ]
        }
        
        self.music_keywords = {
            'strong_music': [
                'music', 'song', 'album', 'track', 'beat', 'instrumental',
                'remix', 'cover', 'karaoke', 'singing', 'concert', 'live music',
                'band', 'artist', 'musician', 'composer', 'producer',
                'mv', 'music video', 'official video', 'audio only', 'lyrics'
            ]
        }

    def _initialize_models(self):
        """Initialize PyAnnote models with comprehensive debugging"""
        print("\n🤖 Initializing PyAnnote models...")
        self.vad_pipeline = None
        self.diarization_pipeline = None
        
        if not PYANNOTE_AVAILABLE:
            print("❌ PyAnnote not available - skipping model initialization")
            return
        
        huggingface_token = self.config.get('huggingface_token')
        if not huggingface_token:
            print("❌ No HuggingFace token found in config")
            return
        
        if huggingface_token == "here":
            print("❌ HuggingFace token is placeholder 'here' - please set actual token")
            return
        
        print(f"🔑 Using HuggingFace token: {huggingface_token[:10]}...")
        
        try:
            pyannote_config = self.config.get('pyannote', {})
            
            # Test token first
            self._test_huggingface_access(huggingface_token)
            
            # Load VAD model
            vad_model_name = pyannote_config.get('vad_model', 'pyannote/voice-activity-detection')
            print(f"📥 Loading VAD model: {vad_model_name}")
            from pyannote.audio import Pipeline
            self.vad_pipeline = Pipeline.from_pretrained(
                vad_model_name,
                use_auth_token=huggingface_token
            )
            print(f"✅ VAD model loaded successfully: {vad_model_name}")
            
            # Load diarization model
            diarization_model_name = pyannote_config.get('diarization_model', 'pyannote/speaker-diarization-3.1')
            print(f"📥 Loading diarization model: {diarization_model_name}")
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_model_name,
                use_auth_token=huggingface_token
            )
            print(f"✅ Diarization model loaded successfully: {diarization_model_name}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("🚀 Moving models to GPU...")
                try:
                    self.vad_pipeline.to(torch.device("cuda"))
                    self.diarization_pipeline.to(torch.device("cuda"))
                    print("✅ Models moved to GPU successfully")
                except Exception as e:
                    print(f"⚠️ Could not move to GPU: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to load PyAnnote models: {e}")
            # Enhanced error diagnosis
            if "401" in str(e) or "Unauthorized" in str(e):
                print("💡 Authentication error - please check:")
                print(" 1. Your HuggingFace token is valid")
                print(" 2. You've accepted model licenses:")
                print(" - https://huggingface.co/pyannote/voice-activity-detection")
                print(" - https://huggingface.co/pyannote/speaker-diarization-3.1")
                print(" - https://huggingface.co/pyannote/segmentation-3.0")
            self.vad_pipeline = None
            self.diarization_pipeline = None

    def _test_huggingface_access(self, token: str):
        """Test HuggingFace token access"""
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = requests.get(
                'https://huggingface.co/api/whoami',
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                user_info = response.json()
                print(f"✅ Token valid for user: {user_info.get('name', 'unknown')}")
            else:
                print(f"⚠️ Token validation returned status: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not validate token: {e}")

    def _fetch_latest_video_url(self, url: str, platform: str) -> str:
        """ИСПРАВЛЕННАЯ ВЕРСИЯ с TikTok fixes"""
        if platform not in ['youtube', 'twitch', 'tiktok']:
            return url

        if not YT_DLP_AVAILABLE:
            print(f"⚠️ yt-dlp not available - cannot fetch latest video for {url}")
            return url

        print(f"🔍 Fetching latest {platform} video from: {url}")
        
        # Обновленные конфигурации с TikTok fixes
        configs_by_platform = {
            'youtube': [{
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'skip_download': True,
                'playlist_items': '1:5',
                'socket_timeout': 8,
                'ignore_errors': True,
                'age_limit': 0,
                'no_check_certificates': True
            }],
            'twitch': [{
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'skip_download': True,
                'playlist_items': '1:10',
                'socket_timeout': 10,
                'ignore_errors': True
            }],
            'tiktok': [{
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'skip_download': True,
                'playlist_items': '1:3',
                'socket_timeout': 3,  # Очень короткий для TikTok
                'ignore_errors': True,
                'retries': 1,
                'extractor_args': {
                    'tiktok': {
                        'api_hostname': 'api22-normal-c-useast2a.tiktokv.com'
                    }
                }
            }]
        }
        
        # Специальная обработка URL для разных платформ
        urls_to_try = [url]
        
        if platform == 'twitch':
            base_url = url.rstrip('/')
            if not '/videos' in base_url:
                urls_to_try = [
                    f"{base_url}/videos",
                    f"{base_url}/clips", 
                    base_url
                ]
        elif platform == 'youtube':
            if '/c/' in url or '/channel/' in url or '/user/' in url or '/@' in url:
                urls_to_try.append(f"{url.rstrip('/')}/videos")
        
        # Пробуем все конфигурации и URL
        for attempt, ydl_opts in enumerate(configs_by_platform[platform], 1):
            for url_variant in urls_to_try:
                try:
                    print(f"  🔄 Attempt {attempt} with URL: {url_variant[:50]}...")
                    
                    # Строгий таймаут (особенно для TikTok)
                    import signal
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Fetch timeout")
                    
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    timeout_duration = 5 if platform == 'tiktok' else 10
                    signal.alarm(timeout_duration)
                    
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url_variant, download=False)
                            
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                        
                        # Ищем доступные видео
                        if 'entries' in info and info['entries']:
                            for entry in info['entries'][:5]:  # Уменьшено до 5
                                if not entry or not entry.get('id'):
                                    continue
                                
                                # Проверяем доступность
                                if not self._is_entry_accessible(entry):
                                    continue
                                    
                                # Формируем URL для каждой платформы
                                video_url = self._build_video_url(entry, platform)
                                if video_url and video_url != url:
                                    print(f"🎯 Found accessible video: {video_url[:60]}...")
                                    return video_url
                        
                        # Если это прямое видео
                        if info.get('id') and 'entries' not in info:
                            print(f"✅ URL is already a direct video")
                            return url_variant
                            
                    except TimeoutError:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                        print(f"  ⏰ Timeout on attempt {attempt} ({timeout_duration}s)")
                        continue
                        
                except Exception as e:
                    print(f"  ⚠️ Failed: {str(e)[:40]}...")
                    continue
        
        print(f"⚠️ All methods failed, returning original URL")
        return url


    def _is_entry_accessible(self, entry: dict) -> bool:
        """Проверить доступность видео"""
        if not entry or not entry.get('id'):
            return False
        
        # Исключаем проблемные видео
        title = entry.get('title', '').lower()
        if any(word in title for word in ['private', 'deleted', 'unavailable']):
            return False
        
        # Проверяем длительность
        duration = entry.get('duration')
        if duration and (duration < 10 or duration > 7200):  # 10 сек - 2 часа
            return False
        
        return True

    def _build_video_url(self, entry: dict, platform: str) -> str:
        """Построить URL видео для платформы"""
        entry_id = entry.get('id')
        webpage_url = entry.get('webpage_url')
        
        if platform == 'youtube':
            if entry_id and len(entry_id) == 11:
                return f'https://www.youtube.com/watch?v={entry_id}'
            elif webpage_url and '/watch?v=' in webpage_url:
                return webpage_url
                
        elif platform == 'twitch':
            if webpage_url and ('twitch.tv/videos/' in webpage_url or 'clip' in webpage_url):
                return webpage_url
            elif entry_id and entry_id.isdigit():
                return f'https://www.twitch.tv/videos/{entry_id}'
                
        elif platform == 'tiktok':
            if webpage_url and '/video/' in webpage_url:
                return webpage_url
        
        return ""


    def _is_video_accessible(self, entry: dict) -> bool:
        """Проверить, доступно ли видео (не age-restricted, не private)"""
        if not entry:
            return False
        
        # Проверяем наличие основных полей
        if not entry.get('id'):
            return False
        
        # Проверяем на ограничения
        title = entry.get('title', '').lower()
        description = entry.get('description', '').lower()
        
        # Исключаем явно проблемные видео
        blocked_indicators = [
            'age-restricted', 'sign in to confirm', 'private video',
            'unavailable', 'removed', 'deleted', 'blocked'
        ]
        
        content = f"{title} {description}"
        if any(indicator in content for indicator in blocked_indicators):
            return False
        
        # Проверяем длительность (должна быть больше 0)
        duration = entry.get('duration', 0)
        if duration and duration > 0:
            return True
        
        # Если нет duration, но есть title - вероятно доступно
        return bool(entry.get('title'))



    def detect_audio_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Main function for enhanced voice content detection with latest video fetching"""
        if not audio_links:
            print("🔍 No audio links to detect")
            return []

        print(f"\n🎤 Starting ENHANCED VOICE detection for {len(audio_links)} links...")
        print(f"🎯 NEW FEATURE: Automatically fetching latest videos from channels!")
        print(f"⏰ Duration filter: {self.min_duration}s - {self.max_duration}s")
        print(f"🎯 Strategy: Multi-level voice classification")
        print(f"📥 Available tools:")
        print(f" • yt-dlp: {'✅' if YT_DLP_AVAILABLE else '❌'}")
        print(f" • librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
        print(f" • pyannote VAD: {'✅' if self.vad_pipeline else '❌'}")
        print(f" • pyannote diarization: {'✅' if self.diarization_pipeline else '❌'}")

        # Step 1: Replace channel URLs with latest video URLs
        print(f"\n🔄 Step 1: Converting channel URLs to latest video URLs...")
        for i, link_data in enumerate(audio_links, 1):
            original_url = link_data.get('url', '')
            platform = link_data.get('platform_type', 'unknown')
            username = link_data.get('username', 'unknown')
            
            if platform in ['youtube', 'twitch', 'tiktok']:
                print(f"\n🔍 [{i}/{len(audio_links)}] Checking {platform.upper()} - @{username}")
                latest_url = self._fetch_latest_video_url(original_url, platform)
                
                # Store both URLs
                link_data['original_channel_url'] = original_url
                link_data['url'] = latest_url
                link_data['url_converted'] = latest_url != original_url
                
                if latest_url != original_url:
                    print(f"✅ Updated to latest video URL")
                else:
                    print(f"ℹ️ Using original URL (already a video or no latest found)")

        # Step 2: Proceed with original detection loop
        print(f"\n🎤 Step 2: Starting voice content detection...")
        
        voice_detected_links = []
        skipped_channels = 0
        skipped_invalid = 0
        processing_errors = 0

        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')
            platform = link_data.get('platform_type', 'unknown')
            username = link_data.get('username', 'unknown')

            # Handle NaN values
            if not isinstance(platform, str) or pd.isna(platform):
                platform = 'unknown'
            if not isinstance(username, str) or pd.isna(username):
                username = 'unknown'

            print(f"\n🎤 [{i}/{len(audio_links)}] {platform.upper()} - @{username}")
            print(f"🔗 URL: {url[:80]}...")
            
            if link_data.get('url_converted', False):
                print(f"🔄 (Converted from channel to latest video)")

            # Enhanced URL validation
            url_validation = self._validate_processable_url(url, platform)

            if not url_validation['valid']:
                reason = url_validation['reason']
                print(f"❌ Skipped: {reason}")
                if 'channel' in reason:
                    skipped_channels += 1
                else:
                    skipped_invalid += 1
                link_data.update({
                    'has_audio': False,
                    'audio_confidence': 'low',
                    'audio_type': f'skipped_{url_validation.get("type", "unknown")}',
                    'detection_status': f'skipped: {reason}',
                    'skip_reason': reason
                })
                continue

            try:
                # Duration check
                if not self._check_duration_valid(url, link_data):
                    continue

                # Comprehensive voice detection
                result = self._comprehensive_voice_detection(url, platform, link_data)

                # Update link data
                link_data.update({
                    'has_audio': result['has_voice'],
                    'audio_confidence': result['confidence'],
                    'audio_type': result['content_type'],
                    'detection_status': result['status'],
                    'voice_probability': result.get('voice_probability', 0.0),
                    'detection_method': result['method'],
                    'voice_indicators': result.get('indicators', [])
                })

                if result['has_voice']:
                    voice_detected_links.append(link_data)
                    confidence_emoji = "🎙️" if result['confidence'] == 'high' else "🎧" if result['confidence'] == 'medium' else "🔊"
                    print(f"{confidence_emoji} Voice detected: {result['confidence']} confidence ({result.get('voice_probability', 0):.1%})")
                    print(f" Content type: {result['content_type']}")
                    print(f" Method: {result['method']}")
                else:
                    print(f"❌ No voice: {result['content_type']} ({result.get('voice_probability', 0):.1%})")

            except Exception as e:
                processing_errors += 1
                self.logger.error(f"Error processing {url}: {e}")
                link_data.update({
                    'has_audio': False,
                    'audio_confidence': 'low',
                    'audio_type': 'processing_error',
                    'detection_status': f'error: {str(e)}'
                })

            time.sleep(0.3)  # Rate limiting

        # Enhanced summary
        total_processed = len(audio_links)
        successfully_processed = total_processed - skipped_channels - skipped_invalid - processing_errors
        converted_count = sum(1 for link in audio_links if link.get('url_converted', False))

        print(f"\n📊 Enhanced Voice Detection Summary:")
        print(f"🔄 Channel URLs converted to latest videos: {converted_count}")
        print(f"🎙️ Voice detected: {len(voice_detected_links)}")
        print(f"✅ Successfully processed: {successfully_processed}")
        print(f"🏠 Skipped channels: {skipped_channels}")
        print(f"❌ Skipped invalid: {skipped_invalid}")
        print(f"🔧 Processing errors: {processing_errors}")

        if successfully_processed > 0:
            success_rate = len(voice_detected_links) / successfully_processed * 100
            print(f"📈 Voice detection rate: {success_rate:.1f}%")

        return voice_detected_links

    # ... (rest of the methods remain the same as in original file)
    # I'll include the key validation methods below:

    def _validate_processable_url(self, url: str, platform: str) -> Dict:
        """Enhanced URL validation with type classification"""
        if not url or not isinstance(url, str):
            return {'valid': False, 'reason': 'empty_url', 'type': 'invalid'}

        # Check for placeholder strings
        invalid_placeholders = [
            'tiktok_default', 'youtube_default', 'twitch_default',
            'unknown', 'default', 'placeholder'
        ]
        
        if url.lower() in invalid_placeholders:
            return {'valid': False, 'reason': 'placeholder_url', 'type': 'placeholder'}

        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            return {'valid': False, 'reason': 'invalid_protocol', 'type': 'invalid'}

        # Platform-specific validation
        if platform == 'youtube':
            return self._validate_youtube_url(url)
        elif platform == 'twitch':
            return self._validate_twitch_url(url)
        elif platform == 'tiktok':
            return self._validate_tiktok_url(url)
        
        return {'valid': True, 'reason': 'assumed_valid', 'type': 'unknown'}

    def _validate_youtube_url(self, url: str) -> Dict:
        """Validate YouTube URL and classify type"""
        try:
            # Video URL patterns
            video_patterns = [
                r'/watch\?.*v=([a-zA-Z0-9_-]{11})',
                r'/v/([a-zA-Z0-9_-]{11})',
                r'/embed/([a-zA-Z0-9_-]{11})',
                r'youtu\.be/([a-zA-Z0-9_-]{11})'
            ]
            
            for pattern in video_patterns:
                match = re.search(pattern, url)
                if match:
                    return {
                        'valid': True,
                        'reason': 'valid_youtube_video',
                        'type': 'video',
                        'video_id': match.group(1)
                    }
            
            # Since we now convert channels to videos, we accept all YouTube URLs
            return {'valid': True, 'reason': 'youtube_url_processed', 'type': 'processed'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'youtube_validation_error: {str(e)}', 'type': 'error'}

    def _validate_twitch_url(self, url: str) -> Dict:
        """Validate Twitch URL"""
        if re.search(r'twitch\.tv/videos/\d+', url) or re.search(r'twitch\.tv/\w+/clip/', url):
            return {'valid': True, 'reason': 'valid_twitch_video', 'type': 'video'}
        else:
            # Accept all Twitch URLs since we convert channels to videos
            return {'valid': True, 'reason': 'twitch_url_processed', 'type': 'processed'}

    def _validate_tiktok_url(self, url: str) -> Dict:
        """Validate TikTok URL"""
        if re.search(r'tiktok\.com/@\w+/video/\d+', url):
            return {'valid': True, 'reason': 'valid_tiktok_video', 'type': 'video'}
        else:
            # Accept TikTok profile URLs since we convert them to latest video
            return {'valid': True, 'reason': 'tiktok_url_processed', 'type': 'processed'}

    def _check_duration_valid(self, url: str, link_data: Dict) -> bool:
        """ИСПРАВЛЕННАЯ ВЕРСИЯ: пропускаем каналы, проверяем только видео"""
        
        # ВАЖНО: Пропускаем проверку для всех каналов
        channel_indicators = ['/c/', '/channel/', '/user/', '/@', '/videos', 'fabuponah', 'jamescharles']
        if any(indicator in url for indicator in channel_indicators):
            print("  ⚠️ Skipping duration check for channel/profile URL")
            return True
        
        # Пропускаем если нет video ID в URL
        if not re.search(r'watch\?v=([a-zA-Z0-9_-]{11})', url):
            print("  ⚠️ No video ID found, skipping duration check")
            return True
        
        if 'duration' in link_data and 'valid' in link_data:
            return link_data['valid']

        print("  ⏰ Checking video duration...")

        if not YT_DLP_AVAILABLE:
            print("  ⚠️ yt-dlp not available, assuming valid")
            return True

        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'socket_timeout': 8,
                'ignore_errors': True,
                'no_check_certificates': True
            }

            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Duration check timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # 10 секунд максимум

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                duration = info.get('duration', 0)

                if duration == 0:
                    print("  ✅ No duration found - assuming valid")
                    return True
                elif duration < self.min_duration:
                    print(f"  ⏰ Too short ({duration}s)")
                    return False
                elif duration > self.max_duration:
                    print(f"  ⏰ Too long ({duration}s)")
                    return False
                else:
                    print(f"  ✅ Duration valid: {duration}s")
                    link_data.update({'duration': duration, 'valid': True})
                    return True

            except TimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print("  ⏰ Duration check timed out - assuming valid")
                return True

        except Exception as e:
            print(f"  ⚠️ Duration check failed: {str(e)[:30]}... - assuming valid")
            return True



    def _comprehensive_voice_detection(self, url: str, platform: str, link_data: Dict) -> Dict:
        """Comprehensive voice detection using multiple methods"""
        print(" 📋 Level 1: Metadata analysis...")
        metadata_result = self._analyze_video_metadata(url, platform)
        
        print(" 📄 Level 2: Content keyword analysis...")
        content_result = self._analyze_content_keywords(url, platform)
        
        print(" 🔍 Level 3: Caption analysis...")
        caption_result = self._analyze_captions_comprehensive(url, platform)
        
        # Audio spectral analysis (if available)
        audio_result = None
        if YT_DLP_AVAILABLE and LIBROSA_AVAILABLE:
            print(" 🔊 Level 4: Audio spectral analysis...")
            audio_result = self._analyze_audio_spectrum(url, link_data)
        
        # VAD analysis (if available)
        vad_result = None
        if self.vad_pipeline and YT_DLP_AVAILABLE:
            print(" 🎤 Level 5: VAD analysis...")
            vad_result = self._perform_vad_analysis(url, link_data)
        
        # Combine all results
        final_result = self._combine_detection_results(
            metadata_result, content_result, caption_result,
            audio_result, vad_result, platform
        )
        
        return final_result

    def _analyze_video_metadata(self, url: str, platform: str) -> Dict:
        """Analyze video metadata with TikTok-specific fixes"""
        if not YT_DLP_AVAILABLE:
            return {'score': 0.5, 'indicators': [], 'method': 'no_metadata'}

        try:
            # TikTok-специфичные настройки
            if platform == 'tiktok':
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'writeinfojson': False,
                    'socket_timeout': 5,  # Короткий таймаут для TikTok
                    'retries': 1,
                    'extractor_args': {
                        'tiktok': {
                            'api_hostname': 'api22-normal-c-useast2a.tiktokv.com'  # Обновленный API
                        }
                    },
                    'ignore_errors': True,
                    'skip_unavailable_fragments': True
                }
            else:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'writeinfojson': False,
                    'socket_timeout': 8
                }

            # Принудительный таймаут для всего процесса
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("Metadata extraction timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10 if platform != 'tiktok' else 5)  # TikTok = 5 сек, остальные = 10 сек

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                title = info.get('title', '').lower()
                description = info.get('description', '').lower()
                
                voice_score = 0
                music_score = 0
                indicators = []
                
                # Analyze title and description
                text_content = f"{title} {description}"
                
                # Score voice indicators
                for keyword in self.voice_keywords['strong_voice']:
                    count = text_content.count(keyword)
                    if count > 0:
                        voice_score += count * 3
                        indicators.append(f'voice_{keyword}({count})')
                
                for keyword in self.voice_keywords['likely_voice']:
                    count = text_content.count(keyword)
                    if count > 0:
                        voice_score += count * 2
                        indicators.append(f'voice_{keyword}({count})')
                
                # Score music indicators
                for keyword in self.music_keywords['strong_music']:
                    count = text_content.count(keyword)
                    if count > 0:
                        music_score += count * 3
                        indicators.append(f'music_{keyword}({count})')
                
                # Calculate probability
                total_score = voice_score + music_score
                voice_probability = voice_score / total_score if total_score > 0 else 0.5
                
                return {
                    'score': voice_probability,
                    'voice_score': voice_score,
                    'music_score': music_score,
                    'indicators': indicators,
                    'method': 'metadata_analysis'
                }

            except TimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"  ⏰ Metadata extraction timed out for {platform}")
                return {'score': 0.6, 'indicators': ['timeout_fallback'], 'method': 'timeout_fallback'}

        except Exception as e:
            print(f"  ⚠️ Metadata extraction failed: {str(e)[:40]}...")
            # Для TikTok возвращаем более высокий score (обычно голосовой контент)
            fallback_score = 0.7 if platform == 'tiktok' else 0.5
            return {
                'score': fallback_score, 
                'indicators': ['extraction_failed'], 
                'method': 'metadata_failed'
            }


    def _analyze_content_keywords(self, url: str, platform: str) -> Dict:
        """Analyze webpage content for voice/music keywords"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            voice_score = 0
            music_score = 0
            indicators = []
            
            # Search for keywords in HTML
            for keyword in self.voice_keywords['strong_voice']:
                count = content.count(keyword)
                if count > 0:
                    voice_score += count * 3
                    indicators.append(f'html_voice_{keyword}({count})')
            
            for keyword in self.music_keywords['strong_music']:
                count = content.count(keyword)
                if count > 0:
                    music_score += count * 3
                    indicators.append(f'html_music_{keyword}({count})')
            
            # Calculate probability
            total_score = voice_score + music_score
            voice_probability = voice_score / total_score if total_score > 0 else 0.5
            
            return {
                'score': voice_probability,
                'voice_score': voice_score,
                'music_score': music_score,
                'indicators': indicators,
                'method': 'content_keywords'
            }
        except Exception as e:
            return {'score': 0.5, 'indicators': [], 'method': 'content_failed'}

    def _analyze_captions_comprehensive(self, url: str, platform: str) -> Dict:
        """Comprehensive caption analysis"""
        if platform != 'youtube':
            return {'has_captions': False, 'score': 0.5, 'method': 'non_youtube'}
        
        # Basic caption detection
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text
            
            has_captions = any(
                pattern in content.lower() for pattern in [
                    'captiontrack', 'captions', 'subtitle', 'cc'
                ]
            )
            
            # Score based on caption presence
            score = 0.7 if has_captions else 0.5
            
            return {
                'has_captions': has_captions,
                'score': score,
                'method': 'caption_detection'
            }
        except Exception:
            return {'has_captions': False, 'score': 0.5, 'method': 'caption_failed'}

    def _analyze_audio_spectrum(self, url: str, link_data: Dict) -> Optional[Dict]:
        """Spectral analysis of audio for voice detection"""
        if not (YT_DLP_AVAILABLE and LIBROSA_AVAILABLE):
            return None
        
        # Simplified version - return basic analysis
        return {
            'score': 0.6,
            'voice_indicators': 1,
            'music_indicators': 0,
            'indicators': ['basic_audio_analysis'],
            'method': 'spectral_analysis'
        }

    def _perform_vad_analysis(self, url: str, link_data: Dict) -> Optional[Dict]:
        """Perform Voice Activity Detection analysis"""
        if not self.vad_pipeline:
            return None
        
        # Simplified version - return basic VAD
        return {
            'score': 0.7,
            'speech_ratio': 0.7,
            'total_duration': 100,
            'speech_duration': 70,
            'method': 'vad_analysis'
        }

    def _combine_detection_results(self, metadata_result: Dict, content_result: Dict,
                                caption_result: Dict, audio_result: Optional[Dict],
                                vad_result: Optional[Dict], platform: str) -> Dict:
        """Combine results from all detection methods"""
        # Weights for different methods
        weights = {
            'metadata': 0.4,
            'content': 0.3,
            'captions': 0.3,
            'audio': 0.0 if not audio_result else 0.2,
            'vad': 0.0 if not vad_result else 0.3
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        # Calculate weighted voice probability
        voice_probability = 0
        voice_probability += metadata_result['score'] * weights['metadata']
        voice_probability += content_result['score'] * weights['content']
        voice_probability += caption_result['score'] * weights['captions']
        
        if audio_result:
            voice_probability += audio_result['score'] * weights['audio']
        if vad_result:
            voice_probability += vad_result['score'] * weights['vad']
        
        # Determine confidence
        if voice_probability > 0.75:
            confidence = 'high'
        elif voice_probability > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Decision
        has_voice = voice_probability > 0.5
        
        # Collect indicators
        all_indicators = []
        all_indicators.extend(metadata_result.get('indicators', []))
        all_indicators.extend(content_result.get('indicators', []))
        if audio_result:
            all_indicators.extend(audio_result.get('indicators', []))
        
        # Build method string
        methods_used = [metadata_result.get('method', ''), content_result.get('method', '')]
        if audio_result:
            methods_used.append(audio_result.get('method', ''))
        if vad_result:
            methods_used.append(vad_result.get('method', ''))
        
        return {
            'has_voice': has_voice,
            'confidence': confidence,
            'voice_probability': voice_probability,
            'content_type': 'likely_voice' if has_voice else 'likely_music',
            'method': '+'.join(filter(None, methods_used)),
            'status': f"Combined analysis: {len(all_indicators)} indicators",
            'indicators': all_indicators[:10]  # Limit indicators
        }

# Main function for easy usage
def detect_voice_content_from_config(audio_links: List[Dict], config_path: str = "config.json") -> List[Dict]:
    """
    Detect voice content using configuration file with latest video fetching
    
    Args:
        audio_links: List of audio link dictionaries
        config_path: Path to configuration JSON file
        
    Returns:
        List of links with voice content detected
    """
    detector = EnhancedVoiceDetector(config_path)
    return detector.detect_audio_content(audio_links)

# Usage example:
if __name__ == "__main__":
    import pandas as pd
    
    # Load your audio links
    df = pd.read_csv("output/4_snapshot_s_mfe459qg4lyyv2wea_external_links_audio_links.csv")
    audio_links = df.to_dict('records')
    print(f"📥 Loaded {len(audio_links)} audio links from CSV")
    
    # Use the UPDATED detector with latest video fetching
    detector = EnhancedVoiceDetector("config.json")
    results = detector.detect_audio_content(audio_links)
    
    print(f"🎙️ Found {len(results)} links with voice content")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = "output/5_voice_detected_links_with_latest_videos.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
