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
    """Fixed Enhanced Voice Detector with proper config loading"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize detector with configuration from JSON file"""
        print("🔧 Initializing Enhanced Voice Detector...")
        
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
                print("   1. Your HuggingFace token is valid")
                print("   2. You've accepted model licenses:")
                print("      - https://huggingface.co/pyannote/voice-activity-detection")
                print("      - https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("      - https://huggingface.co/pyannote/segmentation-3.0")
            
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
    
    def detect_audio_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Main function for enhanced voice content detection"""
        if not audio_links:
            print("🔍 No audio links to detect")
            return []
        
        print(f"\n🎤 Starting ENHANCED VOICE detection for {len(audio_links)} links...")
        print(f"⏰ Duration filter: {self.min_duration}s - {self.max_duration}s")
        print(f"🎯 Strategy: Multi-level voice classification")
        print(f"📥 Available tools:")
        print(f"   • yt-dlp: {'✅' if YT_DLP_AVAILABLE else '❌'}")
        print(f"   • librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
        print(f"   • pyannote VAD: {'✅' if self.vad_pipeline else '❌'}")
        print(f"   • pyannote diarization: {'✅' if self.diarization_pipeline else '❌'}")
        
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
                    print(f"   Content type: {result['content_type']}")
                    print(f"   Method: {result['method']}")
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
        
        print(f"\n📊 Enhanced Voice Detection Summary:")
        print(f"🎙️ Voice detected: {len(voice_detected_links)}")
        print(f"✅ Successfully processed: {successfully_processed}")
        print(f"🏠 Skipped channels: {skipped_channels}")
        print(f"❌ Skipped invalid: {skipped_invalid}")
        print(f"🔧 Processing errors: {processing_errors}")
        
        if successfully_processed > 0:
            success_rate = len(voice_detected_links) / successfully_processed * 100
            print(f"📈 Voice detection rate: {success_rate:.1f}%")
        
        return voice_detected_links
    
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
            
            # Channel URL patterns
            channel_patterns = [r'/c/', r'/channel/', r'/user/', r'/@']
            if any(re.search(pattern, url) for pattern in channel_patterns):
                return {
                    'valid': False,
                    'reason': 'youtube_channel_not_processable',
                    'type': 'channel'
                }
            
            return {'valid': False, 'reason': 'unrecognized_youtube_pattern', 'type': 'unknown'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'youtube_validation_error: {str(e)}', 'type': 'error'}
    
    def _validate_twitch_url(self, url: str) -> Dict:
        """Validate Twitch URL"""
        if re.search(r'twitch\.tv/videos/\d+', url) or re.search(r'twitch\.tv/\w+/clip/', url):
            return {'valid': True, 'reason': 'valid_twitch_video', 'type': 'video'}
        elif re.search(r'twitch\.tv/\w+$', url):
            return {'valid': False, 'reason': 'twitch_channel_not_processable', 'type': 'channel'}
        else:
            return {'valid': True, 'reason': 'assumed_valid_twitch', 'type': 'unknown'}
    
    def _validate_tiktok_url(self, url: str) -> Dict:
        """Validate TikTok URL"""
        if re.search(r'tiktok\.com/@\w+/video/\d+', url):
            return {'valid': True, 'reason': 'valid_tiktok_video', 'type': 'video'}
        else:
            return {'valid': False, 'reason': 'invalid_tiktok_pattern', 'type': 'invalid'}
    
    def _check_duration_valid(self, url: str, link_data: Dict) -> bool:
        """Check if video duration is valid"""
        if 'duration' in link_data and 'valid' in link_data:
            return link_data['valid']
        
        print("   ⏰ Checking video duration...")
        
        if not YT_DLP_AVAILABLE:
            print("   ⚠️ yt-dlp not available, skipping duration check")
            return True
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'socket_timeout': self.timeout
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                
                if duration == 0:
                    print(f"   ⏰ Skipped: No duration found")
                    return False
                elif duration < self.min_duration:
                    print(f"   ⏰ Skipped: Too short ({duration}s < {self.min_duration}s)")
                    return False
                elif duration > self.max_duration:
                    print(f"   ⏰ Skipped: Too long ({duration}s > {self.max_duration}s)")
                    return False
                else:
                    print(f"   ✅ Duration valid: {duration}s")
                    link_data.update({
                        'duration': duration,
                        'valid': True,
                        'reason': 'duration_valid'
                    })
                    return True
                    
        except Exception as e:
            print(f"   ⚠️ Duration check failed: {e}")
            return True  # Continue processing if duration check fails
    
    def _comprehensive_voice_detection(self, url: str, platform: str, link_data: Dict) -> Dict:
        """Comprehensive voice detection using multiple methods"""
        
        print("   📋 Level 1: Metadata analysis...")
        metadata_result = self._analyze_video_metadata(url, platform)
        
        print("   📄 Level 2: Content keyword analysis...")
        content_result = self._analyze_content_keywords(url, platform)
        
        print("   🔍 Level 3: Caption analysis...")
        caption_result = self._analyze_captions_comprehensive(url, platform)
        
        # Audio spectral analysis (if available)
        audio_result = None
        if YT_DLP_AVAILABLE and LIBROSA_AVAILABLE:
            print("   🔊 Level 4: Audio spectral analysis...")
            audio_result = self._analyze_audio_spectrum(url, link_data)
        
        # VAD analysis (if available)
        vad_result = None
        if self.vad_pipeline and YT_DLP_AVAILABLE:
            print("   🎤 Level 5: VAD analysis...")
            vad_result = self._perform_vad_analysis(url, link_data)
        
        # Combine all results
        final_result = self._combine_detection_results(
            metadata_result, content_result, caption_result,
            audio_result, vad_result, platform
        )
        
        return final_result
    
    def _analyze_video_metadata(self, url: str, platform: str) -> Dict:
        """Analyze video metadata for voice indicators"""
        if not YT_DLP_AVAILABLE:
            return {'score': 0.5, 'indicators': [], 'method': 'no_metadata'}
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writeinfojson': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
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
                
        except Exception as e:
            return {'score': 0.5, 'indicators': [], 'method': 'metadata_failed'}
    
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
        
        # Download audio sample
        audio_path = self._download_audio_sample(url, duration=10)
        if not audio_path:
            return None
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=10, sr=22050)
            
            voice_indicators = 0
            music_indicators = 0
            indicators = []
            
            # Spectral centroid (speech: 1-4kHz)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            mean_centroid = np.mean(spectral_centroids)
            
            if 1000 < mean_centroid < 4000:
                voice_indicators += 2
                indicators.append(f'speech_centroid_{mean_centroid:.0f}Hz')
            elif mean_centroid > 5000:
                music_indicators += 1
                indicators.append(f'high_centroid_{mean_centroid:.0f}Hz')
            
            # Calculate probability
            total_indicators = voice_indicators + music_indicators
            voice_probability = voice_indicators / (voice_indicators + music_indicators) if total_indicators > 0 else 0.5
            
            return {
                'score': voice_probability,
                'voice_indicators': voice_indicators,
                'music_indicators': music_indicators,
                'indicators': indicators,
                'method': 'spectral_analysis'
            }
            
        except Exception as e:
            return None
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    def _perform_vad_analysis(self, url: str, link_data: Dict) -> Optional[Dict]:
        """Perform Voice Activity Detection analysis"""
        if not self.vad_pipeline:
            return None
        
        # Download strategic audio segment
        audio_path = self._download_strategic_audio_segment(url, link_data)
        if not audio_path:
            return None
        
        try:
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
            
            print(f"     📊 VAD Speech ratio: {speech_ratio:.2%}")
            
            return {
                'score': speech_ratio,
                'speech_ratio': speech_ratio,
                'total_duration': total_duration,
                'speech_duration': speech_duration,
                'method': 'vad_analysis'
            }
            
        except Exception as e:
            print(f"     ❌ VAD analysis failed: {e}")
            return None
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
    
    def _download_audio_sample(self, url: str, duration: int = 10) -> Optional[str]:
        """Download short audio sample for analysis"""
        if not YT_DLP_AVAILABLE:
            return None
        
        temp_file = tempfile.mktemp(suffix='.wav')
        start_time = random.randint(30, 120)
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'outtmpl': temp_file,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'postprocessor_args': [
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-ar', '22050',
                    '-ac', '1'
                ]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if file exists
            possible_paths = [temp_file, temp_file + '.wav']
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            
            return None
            
        except Exception:
            return None
    
    def _download_strategic_audio_segment(self, url: str, link_data: Dict) -> Optional[str]:
        """Download strategic audio segment for VAD analysis"""
        if not YT_DLP_AVAILABLE:
            return None
        
        duration = link_data.get('duration', 300)
        start_time = self._calculate_optimal_start_time(duration)
        segment_duration = 15
        
        print(f"     📍 Sampling from {start_time}s-{start_time + segment_duration}s")
        
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
                    'end_time': start_time + segment_duration
                }],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            possible_paths = [temp_file, temp_file + '.wav']
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            
            return None
            
        except Exception as e:
            return None
    
    def _calculate_optimal_start_time(self, duration: int) -> int:
        """Calculate optimal start time for audio sampling"""
        if duration <= 60:
            return max(5, duration // 4)
        elif duration <= 300:
            return random.randint(30, min(60, duration - 30))
        elif duration <= 900:
            return random.randint(60, min(180, duration - 60))
        else:
            segments = [duration // 4, duration // 2, (duration * 3) // 4]
            return random.choice(segments)
    
    def _combine_detection_results(self, metadata_result: Dict, content_result: Dict,
                                 caption_result: Dict, audio_result: Optional[Dict],
                                 vad_result: Optional[Dict], platform: str) -> Dict:
        """Combine results from all detection methods"""
        
        # Weights for different methods
        weights = {
            'metadata': 0.25,
            'content': 0.25,
            'captions': 0.15,
            'audio': 0.20 if audio_result else 0.0,
            'vad': 0.35 if vad_result else 0.0
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
        
        # Determine content type and confidence
        content_type = self._classify_content_type(metadata_result, content_result, voice_probability)
        
        # Smart confidence calculation
        if vad_result and vad_result['score'] > 0.6:
            confidence = 'high'
        elif voice_probability > 0.75:
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
            'content_type': content_type,
            'method': '+'.join(filter(None, methods_used)),
            'status': f"Combined analysis: {len(all_indicators)} indicators",
            'indicators': all_indicators[:10]  # Limit indicators
        }
    
    def _classify_content_type(self, metadata_result: Dict, content_result: Dict, voice_probability: float) -> str:
        """Classify content type based on analysis"""
        
        # Combine indicators
        all_indicators = []
        all_indicators.extend(metadata_result.get('indicators', []))
        all_indicators.extend(content_result.get('indicators', []))
        
        indicator_text = ' '.join(all_indicators).lower()
        
        # Classification logic
        if any(word in indicator_text for word in ['podcast', 'interview', 'talk', 'discussion']):
            return 'podcast_interview'
        elif any(word in indicator_text for word in ['tutorial', 'explanation', 'lecture']):
            return 'educational_voice'
        elif any(word in indicator_text for word in ['gaming', 'gameplay', 'commentary']):
            return 'gaming_commentary'
        elif any(word in indicator_text for word in ['music', 'song', 'album', 'artist']):
            return 'music_content'
        elif voice_probability > 0.6:
            return 'likely_voice'
        elif voice_probability < 0.4:
            return 'likely_music'
        else:
            return 'mixed_content'


# Main function for easy usage
def detect_voice_content_from_config(audio_links: List[Dict], config_path: str = "config.json") -> List[Dict]:
    """
    Detect voice content using configuration file
    
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
    
    # Use the FIXED detector with your actual config.json
    detector = EnhancedVoiceDetector("config.json")  # This will properly load your token
    results = detector.detect_audio_content(audio_links)
    
    print(f"🎙️ Found {len(results)} links with voice content")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = "output/5_voice_detected_links.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
