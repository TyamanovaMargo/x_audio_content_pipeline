import requests
import re
import json
import os
import time
import tempfile
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
import multiprocessing
import concurrent.futures
import hashlib
from dataclasses import dataclass

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

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_AVAILABLE = True
    print("✅ Silero VAD imported successfully")
except ImportError:
    SILERO_AVAILABLE = False
    print("❌ Silero VAD not available")


class SmartAudioExtractor:
    """Smart audio extraction from middle portions of videos"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def extract_smart_audio_sample(self, url: str, sample_duration: int = 30) -> Optional[str]:
        """
        Extract audio sample from the most representative part of the video
        
        Args:
            url: Video URL
            sample_duration: Duration of audio sample in seconds
            
        Returns:
            Path to extracted audio file or None
        """
        if not YT_DLP_AVAILABLE:
            self.logger.warning("yt-dlp not available for audio extraction")
            return None
            
        try:
            # First, get video metadata to determine optimal extraction points
            video_duration = self._get_video_duration(url)
            extraction_points = self._calculate_extraction_points(video_duration, sample_duration)
            
            self.logger.info(f"Video duration: {video_duration}s, extraction points: {extraction_points}")
            
            # Try extracting from different points until successful
            for i, (offset, duration) in enumerate(extraction_points):
                self.logger.info(f"Trying extraction point {i+1}/{len(extraction_points)}: {offset}s-{offset+duration}s")
                
                audio_path = self._extract_audio_segment(url, offset, duration)
                if audio_path and os.path.exists(audio_path):
                    # Verify the extracted audio is valid
                    if self._validate_audio_file(audio_path):
                        self.logger.info(f"Successfully extracted audio from {offset}s-{offset+duration}s")
                        return audio_path
                    else:
                        # Clean up invalid file
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                        
            self.logger.warning("All extraction attempts failed")
            return None
            
        except Exception as e:
            self.logger.error(f"Smart audio extraction failed for {url}: {e}")
            return None
    
    def _get_video_duration(self, url: str) -> int:
        """Get video duration in seconds"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 10,
                'retries': 1,
                'ignore_errors': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                return int(duration) if duration else 300  # Default to 5 minutes if unknown
                
        except Exception as e:
            self.logger.warning(f"Could not get video duration: {e}")
            return 300  # Default fallback
    
    def _calculate_extraction_points(self, video_duration: int, sample_duration: int = 20) -> List[Tuple[int, int]]:
        """Simplified extraction points - avoid complex strategies that can hang"""
        if video_duration <= sample_duration:
            return [(0, video_duration)]

        # ✅ ПРОСТАЯ стратегия: только середина видео
        middle_point = video_duration // 2
        offset = max(0, middle_point - sample_duration // 2)

        # ✅ Fallback: начало после пропуска интро
        fallback_offset = min(30, video_duration // 4)

        return [
            (offset, sample_duration),           # Середина
            (fallback_offset, sample_duration)   # Fallback
        ]

    
    def _extract_audio_segment(self, url: str, offset: int, duration: int) -> Optional[str]:
        """Extract specific audio segment"""
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            file_hash = hashlib.md5(f"{url}_{offset}_{duration}".encode()).hexdigest()[:10]
            output_path = os.path.join(temp_dir, f"audio_segment_{file_hash}.wav")
            
            # Remove existing file
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # yt-dlp options for segment extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_path.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 15,
                'retries': 1,
                # Extract specific segment using ffmpeg
                'external_downloader_args': {
                    'ffmpeg_i': ['-ss', str(offset), '-t', str(duration)]
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Check if file was created (might have different extension)
            if os.path.exists(output_path):
                return output_path
            else:
                # Try alternative extensions
                for ext in ['.wav', '.m4a', '.webm', '.mp3', '.ogg']:
                    alt_path = output_path.replace('.wav', ext)
                    if os.path.exists(alt_path):
                        return alt_path
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Audio segment extraction failed: {e}")
            return None
    
    def _validate_audio_file(self, audio_path: str) -> bool:
        """Validate that extracted audio file is usable"""
        try:
            if not os.path.exists(audio_path):
                return False
                
            # Check file size (should be at least 100KB for a meaningful sample)
            file_size = os.path.getsize(audio_path)
            if file_size < 100000:  # 100KB
                self.logger.warning(f"Audio file too small: {file_size} bytes")
                return False
            
            # Try to load with librosa if available
            if LIBROSA_AVAILABLE:
                try:
                    y, sr = librosa.load(audio_path, duration=5)  # Load first 5 seconds as test
                    if len(y) < sr:  # Less than 1 second of audio
                        self.logger.warning("Audio file contains insufficient audio data")
                        return False
                except Exception as e:
                    self.logger.warning(f"Audio file validation failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation error: {e}")
            return False


def extract_metadata_isolated(url, platform, result_queue, voice_keywords, music_keywords):
    """Isolated metadata extraction in separate process"""
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 3,
            'retries': 0,
            'ignore_errors': True
        }
        
        if platform == 'tiktok':
            ydl_opts['extractor_args'] = {
                'tiktok': {'api_hostname': 'api22-normal-c-useast2a.tiktokv.com'}
            }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        # Analyze metadata
        title = info.get('title', '').lower()
        description = info.get('description', '').lower()
        text_content = f"{title} {description}"
        
        voice_score = 0
        music_score = 0
        indicators = []
        
        # Score voice indicators
        for keyword in voice_keywords['strong_voice']:
            count = text_content.count(keyword)
            if count > 0:
                voice_score += count * 3
                indicators.append(f'voice_{keyword}({count})')
        
        for keyword in voice_keywords['likely_voice']:
            count = text_content.count(keyword)
            if count > 0:
                voice_score += count * 2
                indicators.append(f'voice_{keyword}({count})')
        
        for keyword in music_keywords['strong_music']:
            count = text_content.count(keyword)
            if count > 0:
                music_score += count * 3
                indicators.append(f'music_{keyword}({count})')
        
        total_score = voice_score + music_score
        score = voice_score / total_score if total_score > 0 else 0.5
        
        result_queue.put({
            'score': score,
            'voice_score': voice_score,
            'music_score': music_score,
            'indicators': indicators,
            'method': 'isolated_metadata'
        })
        
    except Exception as e:
        fallback_score = 0.7 if platform == 'tiktok' else 0.5
        result_queue.put({
            'score': fallback_score,
            'indicators': ['process_fallback'],
            'method': 'process_failed'
        })


class SileroVADAnalyzer:
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.vad_model = None
        
        if SILERO_AVAILABLE:
            try:
                self.vad_model = load_silero_vad()
                self.logger.info("Silero VAD model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Silero VAD: {e}")
    
    def analyze(self, audio_path: str) -> Dict:
        """Analyze audio file for voice activity"""
        if not self.vad_model or not SILERO_AVAILABLE:
            return {
                'voice_detected': False,
                'confidence': 'low',
                'voice_probability': 0.5,
                'method': 'silero_unavailable',
                'indicators': ['model_not_loaded']
            }
        
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            
            # Ensure mono and correct sample rate
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                sr = 16000
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(wav.squeeze(), self.vad_model, sampling_rate=sr)
            
            # Calculate speech ratio
            total_speech = sum((end - start) / sr for start, end in speech_timestamps)
            total_duration = wav.shape[1] / sr
            speech_ratio = total_speech / total_duration if total_duration > 0 else 0
            
            # Determine voice presence
            voice_detected = speech_ratio > 0.2  # 20% speech threshold
            confidence = 'high' if speech_ratio > 0.5 else 'medium' if speech_ratio > 0.2 else 'low'
            
            return {
                'voice_detected': voice_detected,
                'confidence': confidence,
                'voice_probability': speech_ratio,
                'method': 'silero_vad',
                'indicators': [f'speech_segments:{len(speech_timestamps)}', f'speech_ratio:{speech_ratio:.3f}'],
                'metadata': {'speech_ratio': speech_ratio, 'segments': len(speech_timestamps)}
            }
            
        except Exception as e:
            self.logger.error(f"Silero VAD analysis failed: {e}")
            return {
                'voice_detected': False,
                'confidence': 'low',
                'voice_probability': 0.5,
                'method': 'silero_error',
                'indicators': ['analysis_failed']
            }


class EnhancedVoiceDetector:
    """Enhanced Voice Detector with Smart Middle-Segment Audio Analysis"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize detector with configuration from JSON file"""
        print("🔧 Initializing Enhanced Voice Detector with Smart Audio Analysis...")
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_detector()
        self._setup_voice_keywords()
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Configuration loaded successfully")
            return config
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            "audio_analysis_enabled": True,
            "audio_sample_duration": 30,
            "voice_threshold": 0.6,
            "smart_extraction": True,
            "extraction_strategies": ["middle_third", "multiple_points", "skip_intro"]
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_detector(self):
        """Initialize detector parameters"""
        print("⚙️ Initializing detector parameters...")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.audio_analysis_enabled = self.config.get('audio_analysis_enabled', True)
        self.audio_sample_duration = self.config.get('audio_sample_duration', 30)
        self.voice_threshold = self.config.get('voice_threshold', 0.6)
        self.smart_extraction = self.config.get('smart_extraction', True)
        
        print(f"🎵 Smart audio extraction: {'✅ Enabled' if self.smart_extraction else '❌ Disabled'}")
        print(f"⏱️ Sample duration: {self.audio_sample_duration}s")
    
    def _setup_voice_keywords(self):
        """Setup keyword dictionaries"""
        self.voice_keywords = {
            'strong_voice': [
                'podcast', 'interview', 'talk', 'discussion', 'conversation',
                'speech', 'lecture', 'presentation', 'commentary', 'review',
                'tutorial', 'explanation', 'analysis', 'storytime', 'vlog',
                'reaction', 'reading', 'audiobook', 'monologue', 'dialogue',
                'debate', 'q&a', 'live stream', 'chat', 'talking', 'speaking'
            ],
            'likely_voice': [
                'gaming', 'gameplay', "let's play", 'walkthrough', 'guide',
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
    
    def _initialize_components(self):
        """Initialize audio analysis components"""
        print("🎛️ Initializing components...")
        
        # Smart audio extractor
        self.audio_extractor = SmartAudioExtractor(self.logger)
        
        # Audio analyzers
        self.audio_analyzers = []
        if SILERO_AVAILABLE:
            self.audio_analyzers.append(SileroVADAnalyzer(self.logger))
            print("✅ Silero VAD analyzer enabled")
        
        print(f"🔧 Initialized {len(self.audio_analyzers)} audio analyzers")
    
    def _analyze_video_metadata(self, url: str, platform: str) -> Dict:
        """Analyze video metadata with process isolation"""
        if not YT_DLP_AVAILABLE:
            return {'score': 0.5, 'indicators': [], 'method': 'no_yt_dlp'}
        
        # Skip metadata for channels 
        if platform == 'youtube':
            channel_patterns = ['/channel/', '/c/', '/user/', '/@', '/videos']
            if any(pattern in url for pattern in channel_patterns):
                return {'score': 0.6, 'indicators': ['channel_skipped'], 'method': 'channel_skipped'}
        
        # Use multiprocessing for isolation
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=extract_metadata_isolated,
            args=(url, platform, result_queue, self.voice_keywords, self.music_keywords)
        )
        
        print("  🔄 Extracting metadata...")
        process.start()
        process.join(timeout=5)
        
        if process.is_alive():
            print("  ⏰ Terminating hung process")
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
            return {'score': 0.6, 'indicators': ['process_timeout'], 'method': 'timeout'}
        
        try:
            return result_queue.get_nowait()
        except:
            return {'score': 0.5, 'indicators': ['no_result'], 'method': 'no_result'}
    
    def _analyze_content_keywords(self, url: str, platform: str) -> Dict:
        """Analyze webpage content for keywords"""
        try:
            response = self.session.get(url, timeout=8)
            content = response.text.lower()
            voice_score = 0
            music_score = 0
            indicators = []
            
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
            
            total_score = voice_score + music_score
            score = voice_score / total_score if total_score > 0 else 0.5
            
            return {
                'score': score,
                'voice_score': voice_score,
                'music_score': music_score,
                'indicators': indicators,
                'method': 'content_keywords'
            }
            
        except Exception as e:
            return {'score': 0.5, 'indicators': [], 'method': 'content_failed'}
    
    def _analyze_smart_audio_content(self, url: str) -> Optional[Dict]:
        """Analyze audio content using smart middle-segment extraction"""
        if not self.audio_analysis_enabled or not self.audio_analyzers:
            return None
            
        print("  🎵 Smart audio extraction from middle segments...")
        
        # Extract smart audio sample
        audio_path = self.audio_extractor.extract_smart_audio_sample(
            url, 
            sample_duration=self.audio_sample_duration
        )
        
        if not audio_path:
            self.logger.warning("Failed to extract smart audio sample")
            return None
            
        try:
            # Run audio analyzers
            results = []
            for analyzer in self.audio_analyzers:
                try:
                    result = analyzer.analyze(audio_path)
                    results.append(result)
                    print(f"    📊 {result['method']}: {result['voice_probability']:.3f}")
                except Exception as e:
                    self.logger.error(f"Analyzer failed: {e}")
            
            # Combine audio analysis results
            if results:
                return self._combine_audio_results(results)
            else:
                return None
                
        finally:
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
    
    def _combine_audio_results(self, results: List[Dict]) -> Dict:
        """Combine multiple audio analysis results with lower thresholds"""
        if not results:
            return {'voice_detected': False, 'voice_probability': 0.5, 'confidence': 'low'}
        
        # Average the probabilities
        total_prob = sum(result['voice_probability'] for result in results)
        avg_prob = total_prob / len(results)
        
        print(f"    📊 Audio analysis average: {avg_prob:.3f}")
        
        # ✅ ИСПРАВЛЕНИЕ: Снижен порог с 0.25 до 0.15
        voice_detected = avg_prob > 0.15  # Было: 0.25
        
        # ✅ ИСПРАВЛЕНИЕ: Более мягкие пороги confidence
        confidence = 'high' if avg_prob > 0.5 else 'medium' if avg_prob > 0.15 else 'low'  # Было: 0.6 и 0.25
        
        all_indicators = []
        methods = []
        for result in results:
            all_indicators.extend(result.get('indicators', []))
            methods.append(result['method'])
        
        return {
            'voice_detected': voice_detected,
            'voice_probability': avg_prob,
            'confidence': confidence,
            'method': '+'.join(methods),
            'indicators': all_indicators[:10]
        }

    
    def _comprehensive_voice_detection(self, url: str, platform: str, link_data: Dict) -> Dict:
        """Comprehensive voice detection with improved audio triggering"""
        print("  📋 Level 1: Metadata analysis...")
        metadata_result = self._analyze_video_metadata(url, platform)
        
        print("  📄 Level 2: Content keyword analysis...")
        content_result = self._analyze_content_keywords(url, platform)
        
        # Calculate initial probability
        metadata_score = metadata_result.get('score', 0.5)
        content_score = content_result.get('score', 0.5)
        initial_probability = (metadata_score * 0.6 + content_score * 0.4)
        
        print(f"  📊 Metadata: {metadata_score:.3f}, Content: {content_score:.3f}")
        print(f"  📊 Initial combined: {initial_probability:.3f}")
        
        
        # ✅ ИСПРАВЛЕНИЕ: Принудительный аудио анализ для всех видео
        audio_result = None
        force_audio = self.config.get('force_audio_analysis', False)
        
        if self.audio_analysis_enabled and (force_audio or (0.15 <= initial_probability <= 0.85)):
            if force_audio:
                print("  🎵 Level 3: Forced audio analysis for all videos...")
            else:
                print("  🎵 Level 3: Smart audio analysis (expanded trigger)...")
                
            audio_result = self._analyze_smart_audio_content(url)
        
        # Combine all results
        return self._combine_all_results(metadata_result, content_result, audio_result)

    
    def _combine_all_results(self, metadata_result: Dict, content_result: Dict, audio_result: Optional[Dict]) -> Dict:
        """Combine all analysis results with improved thresholds"""
        
        # Более агрессивные веса для аудио
        if audio_result:
            metadata_weight = 0.2    # Снижено с 0.25
            content_weight = 0.2     # Снижено с 0.25  
            audio_weight = 0.6       # Увеличено с 0.5
        else:
            metadata_weight = 0.6
            content_weight = 0.4
            audio_weight = 0.0
        
        # Calculate final probability
        final_probability = (
            metadata_result.get('score', 0.5) * metadata_weight +
            content_result.get('score', 0.5) * content_weight
        )
        
        if audio_result:
            final_probability += audio_result['voice_probability'] * audio_weight
        
        print(f"  📊 Final probability: {final_probability:.3f}")
        
        # ✅ ИСПРАВЛЕНИЕ: Снижен порог с 0.6 до 0.5
        voice_detected = final_probability >= 0.5  # Было: self.voice_threshold (0.6)
        
        # ✅ ИСПРАВЛЕНИЕ: Более мягкие пороги confidence
        if final_probability > 0.75:
            confidence = 'high'
        elif final_probability > 0.5:   # Было: 0.6
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Collect indicators
        all_indicators = []
        all_indicators.extend(metadata_result.get('indicators', []))
        all_indicators.extend(content_result.get('indicators', []))
        if audio_result:
            all_indicators.extend(audio_result.get('indicators', []))
        
        # Build method string
        methods = [metadata_result.get('method', ''), content_result.get('method', '')]
        if audio_result:
            methods.append(audio_result['method'])
        
        return {
            'has_voice': voice_detected,
            'confidence': confidence,
            'voice_probability': final_probability,
            'content_type': 'likely_voice' if voice_detected else 'likely_music',
            'method': '+'.join(filter(None, methods)),
            'status': f'Enhanced analysis: audio={"yes" if audio_result else "no"}, threshold=50%',
            'indicators': all_indicators[:15]
        }

    
    def detect_audio_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Main detection function with smart audio analysis"""
        if not audio_links:
            print("🔍 No audio links to detect")
            return []

        print(f"\n🎤 Starting SMART VOICE detection for {len(audio_links)} links...")
        print(f"🎯 Smart extraction: {'✅ Enabled' if self.smart_extraction else '❌ Disabled'}")
        print(f"🔧 Available analyzers: {len(self.audio_analyzers)}")

        voice_detected_links = []
        processing_errors = 0
        audio_analyzed_count = 0

        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')  # ✅ Правильный отступ
            platform = link_data.get('platform_type', 'unknown')
            username = link_data.get('username', 'unknown')
            
            # Handle NaN values
            if not isinstance(platform, str) or pd.isna(platform):
                platform = 'unknown'
            if not isinstance(username, str) or pd.isna(username):
                username = 'unknown'
            
            print(f"\n🎤 [{i}/{len(audio_links)}] {platform.upper()} - @{username}")
            print(f"🔗 URL: {url[:80]}...")
            
            if not url or not url.startswith('http'):
                print("❌ Invalid URL")
                continue
            
            try:
                # ✅ ЕДИНСТВЕННЫЙ вызов comprehensive_voice_detection
                result = self._comprehensive_voice_detection(url, platform, link_data)
                
                # ✅ Подробное логирование результата
                print(f"📊 Final result:")
                print(f"   Voice detected: {result['has_voice']}")
                print(f"   Confidence: {result['confidence']}")  
                print(f"   Probability: {result.get('voice_probability', 0):.3f}")
                print(f"   Method: {result['method']}")
                print(f"   Status: {result['status']}")
                
                # Track audio analysis usage
                if 'silero' in result['method'] or 'spectral' in result['method']:
                    audio_analyzed_count += 1
                    print("🔊 Audio analysis was performed")
                
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
                
                # ✅ ЕДИНСТВЕННОЕ добавление в результаты
                if result['has_voice']:
                    voice_detected_links.append(link_data)
                    confidence_emoji = "🎙️" if result['confidence'] == 'high' else "🎧" if result['confidence'] == 'medium' else "🔊"
                    print(f"{confidence_emoji} ✅ VOICE DETECTED")
                else:
                    print(f"❌ NO VOICE DETECTED")
                    
            except Exception as e:
                processing_errors += 1
                self.logger.error(f"Error processing {url}: {e}")
                link_data.update({
                    'has_audio': False,
                    'audio_confidence': 'low',
                    'audio_type': 'processing_error',
                    'detection_status': f'error: {str(e)}'
                })
            
            time.sleep(0.2)  # Rate limiting

        # Summary
        total_processed = len(audio_links)
        print(f"\n📊 Smart Voice Detection Summary:")
        print(f"🎙️ Voice detected: {len(voice_detected_links)}")
        print(f"🎵 Smart audio analyzed: {audio_analyzed_count}")
        print(f"✅ Successfully processed: {total_processed - processing_errors}")
        print(f"🔧 Processing errors: {processing_errors}")

        if total_processed - processing_errors > 0:
            success_rate = len(voice_detected_links) / (total_processed - processing_errors) * 100
            print(f"📈 Voice detection rate: {success_rate:.1f}%")

        return voice_detected_links
