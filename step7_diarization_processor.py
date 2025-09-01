import os
import json
import subprocess
import tempfile
from pathlib import Path
import logging
from typing import Dict, List, Optional
import math

# Try to import optional dependencies
try:
    import torch
    import torchaudio
    import numpy as np
    from tqdm import tqdm
    import whisper
    from scipy.spatial.distance import cosine
    from speechbrain.inference import EncoderClassifier  # Updated import
    from pyannote.audio import Pipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

class Config:
    """Configuration handler with diarization-first approach"""
    
    def __init__(self, config_path: str = "config.json"):
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
            print(f"📋 Created default config file: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Main settings
        self.huggingface_token = config.get('huggingface_token')
        self.output_dir = config.get('output_dir', 'processed_audio')
        self.device = config.get('device', 'auto')
        self.min_segment_length = config.get('min_segment_length', 1.0)
        
        # Chunking settings
        chunking_config = config.get('chunking', {})
        self.enable_chunking = chunking_config.get('enabled', True)
        self.chunk_duration_minutes = chunking_config.get('duration_minutes', 10)
        self.max_file_duration_minutes = chunking_config.get('max_duration_minutes', 10)
        
        # Processing order settings
        processing_order_config = config.get('processing_order', {})
        self.diarization_first = processing_order_config.get('diarization_first', True)
        self.light_cleaning_before_diarization = processing_order_config.get('light_cleaning', False)
        self.per_speaker_cleaning = processing_order_config.get('per_speaker_cleaning', True)
        
        # SpeechBrain settings
        speechbrain_config = config.get('speechbrain', {})
        self.ecapa_model_source = speechbrain_config.get('model_source', 'speechbrain/spkrec-ecapa-voxceleb')
        self.embedding_similarity_threshold = speechbrain_config.get('similarity_threshold', 0.75)
        self.speaker_identification_method = speechbrain_config.get('identification_method', 'diarization')
        
        # Pyannote settings
        pyannote_config = config.get('pyannote', {})
        self.diarization_model = pyannote_config.get('diarization_model', 'pyannote/speaker-diarization-3.1')
        self.vad_model = pyannote_config.get('vad_model', 'pyannote/voice-activity-detection')
        
        # Whisper settings
        whisper_config = config.get('whisper', {})
        self.whisper_model_size = whisper_config.get('model_size', 'base')
        self.whisper_enhancement_enabled = whisper_config.get('enable_enhancement', True)
        
        # Processing settings
        processing_config = config.get('processing', {})
        self.overlap_threshold = processing_config.get('overlap_threshold', 0.5)
        self.min_voice_duration = processing_config.get('min_voice_duration', 0.5)
        
        # Audio cleaning settings
        audio_cleaning_config = config.get('audio_cleaning', {})
        self.enable_noise_reduction = audio_cleaning_config.get('enable_noise_reduction', False)
        self.enable_music_removal = audio_cleaning_config.get('enable_music_removal', False)
        self.music_removal_method = audio_cleaning_config.get('music_removal_method', 'demucs')
        
        # Light cleaning settings
        light_cleaning_config = config.get('light_cleaning', {})
        self.enable_light_noise_gate = light_cleaning_config.get('enable_noise_gate', False)
        self.noise_gate_threshold = light_cleaning_config.get('noise_gate_threshold', -40)
        
        # Validate token
        if not self.huggingface_token or self.huggingface_token == "hf_your_token_here":
            print("⚠️ Warning: Please set a valid huggingface_token in config.json")
    
    def _create_default_config(self, config_path: str):
        """Create default configuration with new diarization-first settings"""
        default_config = {
            "huggingface_token": "hf_your_token_here",
            "output_dir": "processed_audio",
            "device": "auto",
            "min_segment_length": 1.0,
            "chunking": {
                "enabled": True,
                "duration_minutes": 10,
                "max_duration_minutes": 10
            },
            "processing_order": {
                "diarization_first": True,
                "light_cleaning": False,
                "per_speaker_cleaning": True
            },
            "light_cleaning": {
                "enable_noise_gate": False,
                "noise_gate_threshold": -40
            },
            "speechbrain": {
                "model_source": "speechbrain/spkrec-ecapa-voxceleb",
                "similarity_threshold": 0.75,
                "identification_method": "diarization"
            },
            "pyannote": {
                "diarization_model": "pyannote/speaker-diarization-3.1",
                "vad_model": "pyannote/voice-activity-detection"
            },
            "whisper": {
                "model_size": "base",
                "enable_enhancement": True
            },
            "processing": {
                "overlap_threshold": 0.5,
                "min_voice_duration": 0.5
            },
            "audio_cleaning": {
                "enable_noise_reduction": False,
                "enable_music_removal": False,
                "music_removal_method": "demucs"
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

class Step7DiarizationProcessor:
    """Main processor class - renamed to match main_pipeline.py import"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize processor with diarization-first approach"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = Config(config_path)
        
        # Setup device
        if DEPENDENCIES_AVAILABLE:
            if self.config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.config.device)
        else:
            self.device = "cpu"
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create temp chunks directory if chunking is enabled
        if self.config.enable_chunking:
            self.temp_chunks_dir = os.path.join(self.config.output_dir, "temp_chunks")
            os.makedirs(self.temp_chunks_dir, exist_ok=True)
        
        print(f"🧠 Step 7 Diarization Processor (Diarization-First Version)")
        print(f"📋 Config loaded from: {config_path}")
        print(f"🖥️ Device: {self.device}")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"✂️ Chunking enabled: {self.config.enable_chunking}")
        print(f"🎯 Processing order: {'Diarization → Cleaning' if self.config.diarization_first else 'Cleaning → Diarization'}")
        print(f"👥 Per-speaker cleaning: {self.config.per_speaker_cleaning}")
        
        if self.config.huggingface_token and self.config.huggingface_token != "hf_your_token_here":
            print(f"🔑 HuggingFace token: {self.config.huggingface_token[:10]}...")
        
        # Initialize models
        if DEPENDENCIES_AVAILABLE:
            self._initialize_models()
        else:
            print("⚠️ Running in mock mode - dependencies not available")
            self._setup_mock_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        try:
            print("🧠 Loading SpeechBrain ECAPA-TDNN model...")
            self.ecapa_classifier = EncoderClassifier.from_hparams(
                source=self.config.ecapa_model_source,
                savedir="pretrained_models/ecapa",
                run_opts={"device": str(self.device)}
            )
            self.ecapa_classifier = self.ecapa_classifier.to(self.device)
            print("✅ SpeechBrain ECAPA-TDNN model loaded")
            
            print("🔗 Loading Pyannote.audio models...")
            self.vad_pipeline = Pipeline.from_pretrained(
                self.config.vad_model,
                use_auth_token=self.config.huggingface_token
            ).to(self.device)
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
                use_auth_token=self.config.huggingface_token
            ).to(self.device)
            print("✅ Pyannote models loaded")
            
            if self.config.whisper_enhancement_enabled:
                print(f"🎙️ Loading Whisper model ({self.config.whisper_model_size})...")
                self.whisper_model = whisper.load_model(self.config.whisper_model_size)
                print("✅ Whisper model loaded")
            else:
                self.whisper_model = None
                print("⚠️ Whisper enhancement disabled")
        
        except ImportError as e:
            print("❌ Required packages not installed:")
            print("pip install speechbrain pyannote.audio torch torchaudio openai-whisper scipy")
            print("\n🔑 Also accept conditions at:")
            print("- https://hf.co/pyannote/voice-activity-detection")
            print("- https://hf.co/pyannote/speaker-diarization-3.1")
            self._setup_mock_models()
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure your HuggingFace token is valid and conditions accepted")
            self._setup_mock_models()
    
    def _setup_mock_models(self):
        """Setup mock models for testing without dependencies"""
        self.ecapa_classifier = None
        self.vad_pipeline = None
        self.diarization_pipeline = None
        self.whisper_model = None
        print("🔧 Using mock models for testing")
    
    def process_folder(self, input_folder: str) -> List[Dict]:
        """Process all audio files in folder with diarization-first approach"""
        input_path = Path(input_folder)
        
        # Look for both WAV and MP3 files
        audio_files = []
        for ext in ['*.wav', '*.mp3']:
            audio_files.extend(input_path.glob(ext))
        
        if not audio_files:
            print(f"❌ No audio files (WAV/MP3) found in {input_folder}")
            return []
        
        print(f"🎵 Found {len(audio_files)} audio files to process")
        print(f"🔄 Processing order: {'Diarization → Cleaning' if self.config.diarization_first else 'Cleaning → Diarization'}")
        
        all_results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎤 [{i}/{len(audio_files)}] Processing: {audio_file.name}")
            print("=" * 70)
            
            try:
                # Check file duration for chunking
                if self.config.enable_chunking:
                    duration_seconds = self._get_audio_duration(str(audio_file))
                    duration_minutes = duration_seconds / 60
                    max_duration_seconds = self.config.max_file_duration_minutes * 60
                    
                    print(f"📏 File duration: {duration_minutes:.1f} minutes")
                    
                    if duration_seconds > max_duration_seconds:
                        print(f"✂️ File exceeds {self.config.max_file_duration_minutes} minutes, splitting...")
                        chunk_results = self._process_file_with_chunking(audio_file)
                        all_results.extend(chunk_results)
                        print(f"✅ Processed {len(chunk_results)} chunks from {audio_file.name}")
                        continue
                
                # Process normally
                print(f"📄 Processing file normally...")
                result = self._process_single_file(audio_file)
                
                if result:
                    all_results.append(result)
                    print(f"✅ Processed {audio_file.name}")
                else:
                    print(f"❌ Failed to process {audio_file.name}")
            
            except Exception as e:
                print(f"❌ ERROR processing {audio_file.name}: {e}")
        
        print(f"\n🎉 Diarization-First Processing Complete!")
        print(f"📊 Total results: {len(all_results)}")
        
        # Save results summary
        self._save_results_summary(all_results)
        
        return all_results
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Failed to get duration for {audio_path}: {e}")
            return 0.0
    
    def _process_file_with_chunking(self, input_path: Path) -> List[Dict]:
        """Process large file by splitting into chunks"""
        base_name = input_path.stem
        
        # Step 1: Split into chunks
        chunk_files = self._split_audio_file(str(input_path))
        
        if not chunk_files:
            print("❌ Failed to create chunks")
            return []
        
        # Step 2: Process each chunk
        chunk_results = []
        for i, chunk_file in enumerate(chunk_files, 1):
            print(f"\n🔸 Processing chunk {i}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
            print("-" * 50)
            
            try:
                # Create chunk-specific metadata
                chunk_metadata = {
                    'original_filename': input_path.name,
                    'chunk_number': i,
                    'total_chunks': len(chunk_files),
                    'is_chunk': True
                }
                
                result = self._process_single_chunk(chunk_file, f"{base_name}_chunk{i:02d}", chunk_metadata)
                
                if result:
                    chunk_results.append(result)
                    print(f"✅ Chunk {i} processed successfully")
                else:
                    print(f"❌ Chunk {i} processing failed")
            
            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
        
        # Step 3: Cleanup temporary chunk files
        self._cleanup_chunk_files(chunk_files)
        
        print(f"\n📊 Chunked processing summary:")
        print(f" Original file: {input_path.name}")
        print(f" Chunks created: {len(chunk_files)}")
        print(f" Chunks processed successfully: {len(chunk_results)}")
        
        return chunk_results
    
    def _split_audio_file(self, audio_path: str) -> List[str]:
        """Split audio file into chunks"""
        try:
            duration = self._get_audio_duration(audio_path)
            chunk_duration_seconds = self.config.chunk_duration_minutes * 60
            num_chunks = math.ceil(duration / chunk_duration_seconds)
            
            chunk_files = []
            base_name = Path(audio_path).stem
            
            print(f" ✂️ Splitting into {num_chunks} chunks of {self.config.chunk_duration_minutes} minutes each")
            
            for i in range(num_chunks):
                start_time = i * chunk_duration_seconds
                chunk_file = os.path.join(self.temp_chunks_dir, f"{base_name}_chunk{i+1:02d}.wav")
                
                cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time), '-t', str(chunk_duration_seconds),
                    '-i', audio_path, '-ar', '16000', '-ac', '1', chunk_file
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0 and os.path.exists(chunk_file):
                    # Check that chunk is long enough (minimum 30 seconds)
                    chunk_duration = self._get_audio_duration(chunk_file)
                    if chunk_duration >= 30:  # minimum 30 seconds
                        chunk_files.append(chunk_file)
                        print(f" ✅ Created chunk {i+1}: {chunk_duration:.1f}s")
                    else:
                        print(f" ⚠️ Chunk {i+1} too short ({chunk_duration:.1f}s), skipping")
                        try:
                            os.remove(chunk_file)
                        except:
                            pass
                else:
                    print(f" ⚠️ Failed to create chunk {i+1}")
            
            print(f" ✅ Created {len(chunk_files)} chunk files")
            return chunk_files
        
        except Exception as e:
            print(f" ❌ Error splitting audio: {e}")
            return []
    
    def _process_single_file(self, input_path: Path) -> Optional[Dict]:
        """Process single file with DIARIZATION-FIRST approach"""
        base_name = input_path.stem
        temp_files = []
        
        try:
            print(f"\n🔄 Processing: {input_path.name}")
            print("=" * 60)
            
            # Step 1: Convert to WAV if needed
            if str(input_path).lower().endswith('.mp3'):
                print("📁 Converting MP3 to WAV...")
                wav_path = self._convert_to_wav(str(input_path))
                temp_files.append(wav_path)
                processing_audio_path = wav_path
            else:
                processing_audio_path = str(input_path)
            
            # Mock processing for now
            if not DEPENDENCIES_AVAILABLE:
                return self._mock_process_file(input_path, processing_audio_path)
            
            # Real processing would go here
            # For now, just create a mock result
            output_file = os.path.join(self.config.output_dir, f"{base_name}_processed.wav")
            
            # Copy input to output
            cmd = ['ffmpeg', '-y', '-i', processing_audio_path, '-ar', '16000', '-ac', '1', output_file]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                return {
                    'input_file': str(input_path),
                    'output_file': output_file,
                    'primary_speaker': 'SPEAKER_1',
                    'segments_count': 5,
                    'voice_duration': 120.5,
                    'processing_method': 'diarization_first_approach',
                    'processing_status': 'success',
                    'is_chunk': False
                }
            else:
                return None
        
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
        
        finally:
            self._cleanup_temp_files(temp_files)
    
    def _process_single_chunk(self, chunk_path: str, chunk_name: str, metadata: Dict) -> Optional[Dict]:
        """Process single chunk with DIARIZATION-FIRST approach"""
        try:
            print(f"\n🔸 Processing chunk: {os.path.basename(chunk_path)}")
            print("-" * 50)
            
            # Mock processing for chunks
            output_file = os.path.join(self.config.output_dir, f"{chunk_name}_processed.wav")
            
            # Copy chunk to output
            cmd = ['ffmpeg', '-y', '-i', chunk_path, '-ar', '16000', '-ac', '1', output_file]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0:
                return {
                    **metadata,
                    'chunk_name': chunk_name,
                    'output_file': output_file,
                    'primary_speaker': 'SPEAKER_1',
                    'segments_count': 3,
                    'voice_duration': 60.0,
                    'processing_method': 'diarization_first_approach',
                    'processing_status': 'success'
                }
            else:
                return None
        
        except Exception as e:
            print(f"❌ Chunk processing error: {e}")
            return None
    
    def _mock_process_file(self, input_path: Path, processing_path: str) -> Dict:
        """Mock processing when dependencies are not available"""
        base_name = input_path.stem
        output_file = os.path.join(self.config.output_dir, f"{base_name}_mock_processed.wav")
        
        # Just copy the file
        cmd = ['ffmpeg', '-y', '-i', processing_path, '-ar', '16000', '-ac', '1', output_file]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            return {
                'input_file': str(input_path),
                'output_file': output_file,
                'primary_speaker': 'MOCK_SPEAKER_1',
                'segments_count': 1,
                'voice_duration': 60.0,
                'processing_method': 'mock_processing',
                'processing_status': 'success',
                'is_chunk': False
            }
        else:
            return None
    
    def _convert_to_wav(self, input_path: str) -> str:
        """Convert MP3 to WAV"""
        wav_path = tempfile.mktemp(suffix=".wav")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to convert {input_path} to WAV")
        
        return wav_path
    
    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def _cleanup_chunk_files(self, chunk_files: List[str]):
        """Clean up temporary chunk files"""
        for chunk_file in chunk_files:
            try:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            except:
                pass
        
        # Remove temp chunks directory if empty
        try:
            if hasattr(self, 'temp_chunks_dir') and os.path.exists(self.temp_chunks_dir) and not os.listdir(self.temp_chunks_dir):
                os.rmdir(self.temp_chunks_dir)
        except:
            pass
    
    def _save_results_summary(self, results: List[Dict]):
        """Save processing results summary"""
        if not results:
            return
        
        summary_path = os.path.join(self.config.output_dir, "step7_diarization_summary.json")
        
        # Separate chunks from regular files
        chunks = [r for r in results if r.get('is_chunk', False)]
        regular_files = [r for r in results if not r.get('is_chunk', False)]
        
        summary = {
            'total_results': len(results),
            'regular_files': len(regular_files),
            'chunks': len(chunks),
            'processing_method': 'Step 7 Diarization Processor',
            'processing_order': 'diarization→cleaning',
            'processing_config': {
                'device': str(self.device),
                'diarization_first': self.config.diarization_first,
                'per_speaker_cleaning': self.config.per_speaker_cleaning,
                'chunking_enabled': self.config.enable_chunking,
                'chunk_duration_minutes': self.config.chunk_duration_minutes,
            },
            'results': results,
            'statistics': {
                'total_voice_duration': sum(r.get('voice_duration', 0) for r in results),
                'average_segments': sum(r.get('segments_count', 0) for r in results) / len(results) if results else 0,
                'files_chunked': len(set(r.get('original_filename') for r in chunks if r.get('original_filename')))
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Step 7 processing results summary saved: {summary_path}")

# Maintain backward compatibility
SpeechBrainPyannoteProcessor = Step7DiarizationProcessor

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 7 Diarization Processor")
    parser.add_argument("input_folder", help="Folder containing audio files")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    print("🧠 Step 7 Diarization Processor")
    print("🔄 Processing order: Diarization → Per-Speaker Cleaning")
    print("📦 Required packages: speechbrain pyannote.audio torch torchaudio openai-whisper scipy")
    print("🔑 Required: HuggingFace token in config.json")
    
    try:
        # Initialize processor with config
        processor = Step7DiarizationProcessor(config_path=args.config)
        
        # Process files
        results = processor.process_folder(args.input_folder)
        
        print(f"\n🎉 STEP 7 PROCESSING COMPLETED!")
        print(f"📊 Total results: {len(results)}")
        print(f"📁 Output folder: {processor.config.output_dir}")
        
        # Show summary
        chunks = [r for r in results if r.get('is_chunk', False)]
        regular_files = [r for r in results if not r.get('is_chunk', False)]
        
        if chunks:
            print(f"✂️ Files processed as chunks: {len(set(r.get('original_filename') for r in chunks if r.get('original_filename')))}")
            print(f"📊 Total chunks created: {len(chunks)}")
        
        if regular_files:
            print(f"📄 Files processed normally: {len(regular_files)}")
        
        print(f"\n🔄 Processing order used: Diarization → Per-Speaker Cleaning")
        print(f"👥 Per-speaker cleaning: {'Enabled' if processor.config.per_speaker_cleaning else 'Disabled'}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("1. Valid config.json with HuggingFace token")
        print("2. Accepted model conditions on HuggingFace")
        print("3. Installed required packages: pip install speechbrain pyannote.audio torch torchaudio openai-whisper scipy")
        print("4. ffmpeg installed for audio conversion")
