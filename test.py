import os
import shutil
import argparse
import logging
import json
from glob import glob
from pathlib import Path
from pydub import AudioSegment
import torch
import whisper
from pyannote.audio import Pipeline
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# NeMo imports for Parakeet
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("⚠️ NeMo not installed. Install with: pip install nemo_toolkit[asr]")

class ParakeetWrapper:
    """
    Wrapper for NeMo Parakeet-TDT-0.6B-v3 model with pipeline-like API
    """
    def __init__(self, device: str = "auto"):
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo not available. Install with: pip install nemo_toolkit[asr]")
        
        print("🔄 Loading Parakeet model via NeMo...")
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )
            
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.device = device
            print("✅ Parakeet model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load Parakeet: {e}")
            raise

    def __call__(self, input_dict):
        """
        Transcribe audio
        input_dict = {"array": np.ndarray, "sampling_rate": 16000}
        Returns: {"text": str}
        """
        if not isinstance(input_dict, dict):
            raise TypeError("Expected dict with 'array' and 'sampling_rate'")
        
        wav = input_dict["array"]
        sr = input_dict["sampling_rate"]
        
        if sr != 16000:
            raise ValueError("Model expects 16kHz audio")
        
        # Ensure proper format for NeMo
        if len(wav.shape) > 1:
            wav = wav.squeeze()
        
        try:
            # NeMo expects list of numpy arrays
            transcriptions = self.model.transcribe([wav], batch_size=1)
            text = transcriptions[0] if transcriptions else ""
            return {"text": text}
        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": ""}

class SpeechOnlyDetector:
    """
    Advanced speech detector that filters out singing, music, and noise
    Saves only files containing pure speech/conversation
    """
    
    def __init__(self, 
                 output_dir: str,
                 config_path: str = "config.json",
                 threshold: float = 0.6,
                 min_duration: float = 5.0,
                 verbose: bool = True):
        
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.min_duration = min_duration
        self.verbose = verbose
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Override with config values if available
        if self.config:
            self.threshold = self.config.get('threshold', self.threshold)
            self.min_duration = self.config.get('min_duration', self.min_duration)
            self.verbose = self.config.get('verbose', self.verbose)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Setup logging
        self._setup_logging()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'speech_detected': 0,
            'singing_detected': 0,
            'music_detected': 0,
            'too_short': 0,
            'errors': 0
        }
    
    def _load_config(self, config_path: str) -> Optional[Dict]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Config load error: {e}")
        return None
    
    def _initialize_models(self):
        """Initialize all AI models"""
        # Get Hugging Face token
        hf_token = None
        if self.config:
            hf_token = self.config.get('huggingface_token')
        
        # Initialize Whisper
        print("🔄 Loading Whisper model...")
        whisper_size = "base"
        if self.config and 'whisper' in self.config:
            whisper_size = self.config['whisper'].get('model_size', 'base')
        
        self.whisper_model = whisper.load_model(whisper_size)
        print("✅ Whisper model loaded!")
        
        # Initialize Pyannote VAD
        print("🔄 Loading Pyannote VAD...")
        try:
            vad_model = "pyannote/voice-activity-detection"
            if self.config and 'pyannote' in self.config:
                vad_model = self.config['pyannote'].get('vad_model', vad_model)
            
            if hf_token:
                self.vad_pipeline = Pipeline.from_pretrained(vad_model, use_auth_token=hf_token)
            else:
                self.vad_pipeline = Pipeline.from_pretrained(vad_model)
            print("✅ Pyannote VAD loaded!")
        except Exception as e:
            print(f"⚠️ Pyannote VAD failed: {e}")
            self.vad_pipeline = None
        
        # Initialize Parakeet
        try:
            device = "auto"
            if self.config:
                device = self.config.get('device', 'auto')
            self.parakeet_pipeline = ParakeetWrapper(device=device)
        except Exception as e:
            print(f"⚠️ Parakeet failed to load: {e}")
            self.parakeet_pipeline = None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "speech_detection.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _audio_to_numpy(self, audio_path: str) -> tuple:
        """Load audio and convert to numpy array"""
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Convert to numpy
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2**15)  # Normalize to [-1, 1]
            
            duration = len(audio) / 1000.0
            return samples, duration, True
        except Exception as e:
            if self.verbose:
                print(f"❌ Audio load error: {e}")
            return None, 0, False
    
    def _detect_voice_activity(self, audio_path: str, samples: np.ndarray, duration: float) -> Dict:
        """Detect voice activity using Pyannote VAD"""
        if not self.vad_pipeline:
            # Fallback: assume reasonable speech ratio
            return {
                'speech_duration': duration * 0.7,
                'speech_ratio': 0.7,
                'speech_segments': [(0, duration)]
            }
        
        try:
            # Convert to tensor for Pyannote
            waveform = torch.from_numpy(samples).unsqueeze(0)
            vad_input = {'waveform': waveform, 'sample_rate': 16000}
            
            vad_result = self.vad_pipeline(vad_input)
            speech_segments = vad_result.get_timeline().support()
            speech_duration = sum(segment.duration for segment in speech_segments)
            speech_ratio = speech_duration / duration if duration > 0 else 0
            
            return {
                'speech_duration': speech_duration,
                'speech_ratio': speech_ratio,
                'speech_segments': [(seg.start, seg.end) for seg in speech_segments]
            }
        except Exception as e:
            if self.verbose:
                print(f"⚠️ VAD error: {e}")
            return {
                'speech_duration': duration * 0.5,
                'speech_ratio': 0.5,
                'speech_segments': [(0, duration)]
            }
    
    def _transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            text = result.get('text', '').strip()
            return {
                'transcription': text,
                'word_count': len(text.split()) if text else 0,
                'success': True
            }
        except Exception as e:
            if self.verbose:
                print(f"❌ Whisper error: {e}")
            return {
                'transcription': '',
                'word_count': 0,
                'success': False
            }
    
    def _detect_singing_parakeet(self, samples: np.ndarray, duration: float) -> Dict:
        """Detect singing using Parakeet + heuristics"""
        result = {
            'is_singing': False,
            'singing_score': 0.0,
            'parakeet_transcription': '',
            'details': {}
        }
        
        if not self.parakeet_pipeline:
            return result
        
        try:
            # Transcribe with Parakeet
            input_data = {"array": samples, "sampling_rate": 16000}
            parakeet_result = self.parakeet_pipeline(input_data)
            transcription = parakeet_result.get('text', '').strip()
            result['parakeet_transcription'] = transcription
            
            if not transcription:
                return result
            
            # Analyze transcription for singing patterns
            words = transcription.lower().split()
            word_count = len(words)
            unique_words = len(set(words)) if words else 0
            repetition_ratio = unique_words / word_count if word_count > 0 else 0
            words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
            
            # Singing detection heuristics
            singing_score = 0.0
            
            # Unusual speaking rate (too slow/fast)
            if words_per_minute < 40 or words_per_minute > 220:
                singing_score += 0.4
            
            # High repetition (common in songs)
            if repetition_ratio < 0.25:
                singing_score += 0.35
            
            # Too few words for duration (instrumental parts)
            if word_count < 8 and duration > 15:
                singing_score += 0.25
            
            # Common singing patterns
            singing_patterns = ['la la', 'na na', 'oh oh', 'hey hey', 'yeah yeah', 'da da']
            if any(pattern in transcription.lower() for pattern in singing_patterns):
                singing_score += 0.3
            
            result['singing_score'] = min(singing_score, 1.0)
            result['is_singing'] = singing_score > 0.5
            result['details'] = {
                'words_per_minute': words_per_minute,
                'repetition_ratio': repetition_ratio,
                'word_count': word_count
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Parakeet singing detection error: {e}")
        
        return result
    
    def _detect_music_heuristics(self, whisper_text: str, parakeet_text: str, 
                                vad_info: Dict, duration: float) -> Dict:
        """Detect music using various heuristics"""
        music_score = 0.0
        reasons = []
        
        # Text-based indicators
        words = whisper_text.lower().split() if whisper_text else []
        word_count = len(words)
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
        
        # Very low speech ratio
        speech_ratio = vad_info.get('speech_ratio', 0)
        if speech_ratio < 0.25:
            music_score += 0.3
            reasons.append("low_speech_ratio")
        
        # Unusual word rate
        if words_per_minute < 30 or words_per_minute > 250:
            music_score += 0.25
            reasons.append("unusual_word_rate")
        
        # Inconsistency between transcription models
        if whisper_text and parakeet_text:
            whisper_words = set(whisper_text.lower().split())
            parakeet_words = set(parakeet_text.lower().split())
            if len(whisper_words.intersection(parakeet_words)) < len(whisper_words) * 0.3:
                music_score += 0.2
                reasons.append("transcription_inconsistency")
        
        # Music-related keywords
        music_keywords = ['music', 'song', 'beat', 'melody', 'instrumental', 'track']
        if any(keyword in whisper_text.lower() for keyword in music_keywords):
            music_score += 0.15
            reasons.append("music_keywords")
        
        return {
            'is_music': music_score > 0.4,
            'music_score': music_score,
            'reasons': reasons
        }
    
    def analyze_audio_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Comprehensive analysis of a single audio file
        Returns detailed analysis results
        """
        file_path = Path(file_path)
        
        result = {
            'file_path': str(file_path),
            'filename': file_path.name,
            'contains_speech': False,
            'is_singing': False,
            'is_music': False,
            'speech_score': 0.0,
            'duration': 0.0,
            'transcription': '',
            'word_count': 0,
            'error': None,
            'details': {}
        }
        
        try:
            if self.verbose:
                print(f"\n🎵 Analyzing: {file_path.name}")
            
            # Check file size
            if file_path.stat().st_size < 1000:
                result['error'] = 'File too small (< 1KB)'
                self.stats['errors'] += 1
                return result
            
            # Load audio
            samples, duration, success = self._audio_to_numpy(str(file_path))
            if not success:
                result['error'] = 'Failed to load audio'
                self.stats['errors'] += 1
                return result
            
            result['duration'] = duration
            
            # Check minimum duration
            if duration < self.min_duration:
                result['error'] = f'Too short: {duration:.1f}s < {self.min_duration}s'
                self.stats['too_short'] += 1
                return result
            
            # Voice Activity Detection
            vad_info = self._detect_voice_activity(str(file_path), samples, duration)
            speech_duration = vad_info['speech_duration']
            
            if speech_duration < self.min_duration:
                result['error'] = f'Insufficient speech: {speech_duration:.1f}s'
                self.stats['too_short'] += 1
                return result
            
            # Transcription with Whisper
            whisper_result = self._transcribe_audio(str(file_path))
            if not whisper_result['success'] or whisper_result['word_count'] < 3:
                result['error'] = 'No meaningful speech detected'
                self.stats['errors'] += 1
                return result
            
            result['transcription'] = whisper_result['transcription']
            result['word_count'] = whisper_result['word_count']
            
            # Singing detection with Parakeet
            singing_result = self._detect_singing_parakeet(samples, duration)
            result['is_singing'] = singing_result['is_singing']
            
            if result['is_singing']:
                result['error'] = f'Singing detected (score: {singing_result["singing_score"]:.2f})'
                self.stats['singing_detected'] += 1
                if self.verbose:
                    print(f"  🎤 Singing detected!")
                return result
            
            # Music detection
            music_result = self._detect_music_heuristics(
                whisper_result['transcription'],
                singing_result['parakeet_transcription'],
                vad_info,
                duration
            )
            result['is_music'] = music_result['is_music']
            
            if result['is_music']:
                result['error'] = f'Music detected (score: {music_result["music_score"]:.2f})'
                self.stats['music_detected'] += 1
                if self.verbose:
                    print(f"  🎶 Music detected!")
                return result
            
            # Calculate speech quality score
            speech_score = 0.5  # Base score
            
            # Word rate quality
            words_per_minute = (whisper_result['word_count'] / duration) * 60
            if 60 <= words_per_minute <= 180:
                speech_score += 0.2
            
            # Speech ratio quality
            if vad_info['speech_ratio'] > 0.6:
                speech_score += 0.2
            
            # Word count quality
            if whisper_result['word_count'] > 15:
                speech_score += 0.1
            
            result['speech_score'] = min(speech_score, 1.0)
            result['details'] = {
                'words_per_minute': words_per_minute,
                'speech_ratio': vad_info['speech_ratio'],
                'speech_duration': speech_duration,
                'vad_segments': len(vad_info['speech_segments'])
            }
            
            # Final decision
            if result['speech_score'] >= self.threshold and whisper_result['word_count'] >= 5:
                result['contains_speech'] = True
                self.stats['speech_detected'] += 1
                if self.verbose:
                    print(f"  ✅ Pure speech detected! Score: {result['speech_score']:.2f}")
                    print(f"     Words: {whisper_result['word_count']}, "
                          f"Rate: {words_per_minute:.0f} wpm, "
                          f"Duration: {speech_duration:.1f}s")
            else:
                result['error'] = f'Low speech quality (score: {result["speech_score"]:.2f})'
                if self.verbose:
                    print(f"  ❌ Low quality speech")
        
        except Exception as e:
            result['error'] = f'Analysis failed: {str(e)}'
            self.stats['errors'] += 1
            if self.verbose:
                print(f"  ❌ Error: {e}")
        
        return result
    
    def process_directory(self, source_dir: Union[str, Path]) -> List[Dict]:
        """
        Process all audio files in a directory
        Saves speech-only files to output directory
        """
        source_dir = Path(source_dir)
        
        # Find all audio files
        audio_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(source_dir.glob(ext))
        
        if not audio_files:
            print(f"❌ No audio files found in {source_dir}")
            return []
        
        print(f"\n🎤 Found {len(audio_files)} audio files in {source_dir}")
        print("=" * 70)
        
        results = []
        copied_files = 0
        
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n📁 Processing ({i}/{len(audio_files)}): {file_path.name}")
            
            # Analyze file
            result = self.analyze_audio_file(file_path)
            results.append(result)
            self.stats['total_processed'] += 1
            
            # Copy speech-only files
            if result['contains_speech'] and not result['error']:
                output_path = self.output_dir / file_path.name
                try:
                    shutil.copy2(file_path, output_path)
                    result['output_path'] = str(output_path)
                    copied_files += 1
                    if self.verbose:
                        print(f"     💾 Saved to: {output_path.name}")
                except Exception as e:
                    if self.verbose:
                        print(f"     ❌ Copy failed: {e}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("🎯 PROCESSING COMPLETE!")
        print(f"📊 Total files processed: {self.stats['total_processed']}")
        print(f"✅ Speech-only files saved: {self.stats['speech_detected']}")
        print(f"🎤 Singing detected: {self.stats['singing_detected']}")
        print(f"🎶 Music detected: {self.stats['music_detected']}")
        print(f"⏱️ Too short: {self.stats['too_short']}")
        print(f"❌ Errors: {self.stats['errors']}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Save detailed CSV report
        self._save_csv_report(results)
        
        return results
    
    def _save_csv_report(self, results: List[Dict]):
        """Save detailed analysis report to CSV"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "speech_analysis_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"📄 Detailed report saved: {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Speech-Only Audio Detector - Filters out singing, music, and noise'
    )
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory with audio files')
    parser.add_argument('--dest', type=str, required=True,
                        help='Destination directory for speech-only files')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Configuration file path')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Speech quality threshold (0.0-1.0)')
    parser.add_argument('--min_duration', type=float, default=5.0,
                        help='Minimum duration in seconds')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SpeechOnlyDetector(
        output_dir=args.dest,
        config_path=args.config,
        threshold=args.threshold,
        min_duration=args.min_duration,
        verbose=args.verbose
    )
    
    # Process directory
    results = detector.process_directory(args.source)
    
    # Final summary
    speech_count = sum(1 for r in results if r['contains_speech'])
    print(f"\n🏁 FINAL RESULT: {speech_count}/{len(results)} files contain pure speech")
    print(f"   These files have been saved to: {args.dest}")

if __name__ == '__main__':
    main()
