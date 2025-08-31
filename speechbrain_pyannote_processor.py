import os
import json
import subprocess
import tempfile
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import whisper
from typing import Dict, List, Optional
import math
from scipy.spatial.distance import cosine
from speechbrain.pretrained import EncoderClassifier
from pyannote.audio import Pipeline

class Config:
    """Configuration handler with all fixes applied"""
    
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
        
        # Chunking settings (FIXED)
        chunking_config = config.get('chunking', {})
        self.enable_chunking = chunking_config.get('enabled', True)
        self.chunk_duration_minutes = chunking_config.get('duration_minutes', 10)
        self.max_file_duration_minutes = chunking_config.get('max_duration_minutes', 10)
        
        # SpeechBrain ECAPA-TDNN settings (FIXED)
        speechbrain_config = config.get('speechbrain', {})
        self.ecapa_model_source = speechbrain_config.get('model_source', 'speechbrain/spkrec-ecapa-voxceleb')
        self.embedding_similarity_threshold = speechbrain_config.get('similarity_threshold', 0.75)  # INCREASED
        self.speaker_identification_method = speechbrain_config.get('identification_method', 'diarization')  # CHANGED
        
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
        
        # Audio cleaning settings (NEW)
        audio_cleaning_config = config.get('audio_cleaning', {})
        self.enable_noise_reduction = audio_cleaning_config.get('enable_noise_reduction', False)
        self.enable_music_removal = audio_cleaning_config.get('enable_music_removal', False)
        self.music_removal_method = audio_cleaning_config.get('music_removal_method', 'demucs')  # 'demucs' or 'spleeter'
        
        # Validate token
        if not self.huggingface_token or self.huggingface_token == "hf_your_token_here":
            raise ValueError("❌ Please set a valid huggingface_token in config.json")
    
    def _create_default_config(self, config_path: str):
        """Create default configuration file with all fixes"""
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

class SpeechBrainPyannoteProcessor:
    def __init__(self, config_path: str = "config.json"):
        """Initialize processor with all fixes applied"""
        # Load configuration
        self.config = Config(config_path)
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create temp chunks directory if chunking is enabled
        if self.config.enable_chunking:
            self.temp_chunks_dir = os.path.join(self.config.output_dir, "temp_chunks")
            os.makedirs(self.temp_chunks_dir, exist_ok=True)
        
        print(f"🧠 SpeechBrain + Pyannote Audio Processor (FIXED VERSION)")
        print(f"📋 Config loaded from: {config_path}")
        print(f"🖥️ Device: {self.device}")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"✂️ Chunking enabled: {self.config.enable_chunking}")
        print(f"🎯 Speaker method: {self.config.speaker_identification_method}")
        print(f"🎵 Music removal: {self.config.enable_music_removal}")
        print(f"🔇 Noise reduction: {self.config.enable_noise_reduction}")
        print(f"🔑 HuggingFace token: {self.config.huggingface_token[:10]}...")
        
        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models with error handling"""
        try:
            print("🧠 Loading SpeechBrain ECAPA-TDNN model...")
            
            # Load SpeechBrain ECAPA-TDNN for speaker embeddings
            self.ecapa_classifier = EncoderClassifier.from_hparams(
                source=self.config.ecapa_model_source,
                savedir="pretrained_models/ecapa"
            )
            
            # Move to device
            self.ecapa_classifier = self.ecapa_classifier.to(self.device)
            print("✅ SpeechBrain ECAPA-TDNN model loaded")
            
            print("🔗 Loading Pyannote.audio models...")
            
            # Load Pyannote Voice Activity Detection
            self.vad_pipeline = Pipeline.from_pretrained(
                self.config.vad_model,
                use_auth_token=self.config.huggingface_token
            ).to(self.device)
            
            # Load Pyannote Speaker Diarization
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
                use_auth_token=self.config.huggingface_token
            ).to(self.device)
            
            print("✅ Pyannote models loaded")
            
            # Load Whisper if enhancement is enabled
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
            raise e
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure your HuggingFace token is valid and conditions accepted")
            raise e

    def process_folder(self, input_folder: str) -> List[Dict]:
        """Process all MP3 files in folder with automatic chunking"""
        input_path = Path(input_folder)
        mp3_files = list(input_path.glob("*.mp3"))
        
        if not mp3_files:
            print(f"❌ No MP3 files found in {input_folder}")
            return []
        
        print(f"🎵 Found {len(mp3_files)} MP3 files to process")
        
        all_results = []
        
        for i, mp3_file in enumerate(mp3_files, 1):
            print(f"\n🎤 [{i}/{len(mp3_files)}] Processing: {mp3_file.name}")
            print("=" * 70)
            
            try:
                # Check file duration if chunking is enabled
                if self.config.enable_chunking:
                    duration_seconds = self._get_audio_duration(str(mp3_file))
                    duration_minutes = duration_seconds / 60
                    max_duration_seconds = self.config.max_file_duration_minutes * 60
                    
                    print(f"📏 File duration: {duration_minutes:.1f} minutes")
                    
                    if duration_seconds > max_duration_seconds:
                        print(f"✂️ File exceeds {self.config.max_file_duration_minutes} minutes, splitting into chunks...")
                        chunk_results = self._process_file_with_chunking(mp3_file)
                        all_results.extend(chunk_results)
                        print(f"✅ Processed {len(chunk_results)} chunks from {mp3_file.name}")
                        continue
                
                # Process normally
                print(f"📄 Processing file normally...")
                result = self._process_single_file(mp3_file)
                if result:
                    all_results.append(result)
                    print(f"✅ Processed {mp3_file.name}")
                else:
                    print(f"❌ Failed to process {mp3_file.name}")
                        
            except Exception as e:
                print(f"❌ ERROR processing {mp3_file.name}: {e}")
        
        print(f"\n🎉 SpeechBrain + Pyannote Processing complete!")
        print(f"📊 Total results: {len(all_results)}")
        
        # Save results summary
        self._save_results_summary(all_results)
        
        return all_results

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0

    def _split_audio_file(self, audio_path: str) -> List[str]:
        """Split audio file into chunks"""
        try:
            duration = self._get_audio_duration(audio_path)
            chunk_duration_seconds = self.config.chunk_duration_minutes * 60
            
            num_chunks = math.ceil(duration / chunk_duration_seconds)
            chunk_files = []
            
            base_name = Path(audio_path).stem
            
            print(f"   ✂️ Splitting into {num_chunks} chunks of {self.config.chunk_duration_minutes} minutes each")
            
            pbar = tqdm(range(num_chunks), desc="   📊 Creating chunks", unit="chunk")
            
            for i in pbar:
                start_time = i * chunk_duration_seconds
                chunk_file = os.path.join(self.temp_chunks_dir, f"{base_name}_chunk{i+1:02d}.wav")
                
                pbar.set_description(f"   📊 Creating chunk {i+1}/{num_chunks}")
                
                cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time), '-t', str(chunk_duration_seconds),
                    '-i', audio_path, '-c', 'copy', chunk_file
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0 and os.path.exists(chunk_file):
                    chunk_files.append(chunk_file)
                    pbar.set_postfix({"Created": f"{len(chunk_files)}/{num_chunks}"})
                else:
                    print(f"   ⚠️ Failed to create chunk {i+1}")
            
            pbar.close()
            print(f"   ✅ Created {len(chunk_files)} chunk files")
            return chunk_files
            
        except Exception as e:
            print(f"   ❌ Error splitting audio: {e}")
            return []

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
        print(f"   Original file: {input_path.name}")
        print(f"   Chunks created: {len(chunk_files)}")
        print(f"   Chunks processed successfully: {len(chunk_results)}")
        
        return chunk_results

    def _process_single_chunk(self, chunk_path: str, chunk_name: str, metadata: Dict) -> Optional[Dict]:
        """Process a single audio chunk with all fixes applied"""
        temp_files = []
        
        try:
            # Step 1: Audio cleaning (if enabled)
            clean_audio_path = chunk_path
            if self.config.enable_music_removal:
                print("🎵 Removing background music...")
                clean_audio_path = self._remove_background_music(chunk_path)
                if clean_audio_path != chunk_path:
                    temp_files.append(clean_audio_path)
            
            if self.config.enable_noise_reduction:
                print("🔇 Reducing background noise...")
                clean_audio_path = self._reduce_noise(clean_audio_path)
                if clean_audio_path not in temp_files and clean_audio_path != chunk_path:
                    temp_files.append(clean_audio_path)
            
            # Step 2: Voice Activity Detection
            print("🧠 Voice Activity Detection...")
            vad_segments = self._detect_voice_activity_pyannote(clean_audio_path)
            
            if not vad_segments:
                print("❌ No voice activity detected in chunk")
                return None
            
            print(f"✅ Found {len(vad_segments)} voice segments")
            
            # Step 3: Speaker Analysis (FIXED METHOD)
            if self.config.speaker_identification_method == "embeddings":
                print("🎯 Speaker identification using ECAPA-TDNN embeddings...")
                speaker_info = self._identify_speakers_with_ecapa(clean_audio_path, vad_segments)
            else:
                print("👥 Speaker diarization using Pyannote...")
                speaker_info = self._speaker_diarization_pyannote(clean_audio_path)
            
            if not speaker_info or not speaker_info.get('primary_speaker'):
                print("❌ No speakers identified in chunk")
                return None
            
            primary_speaker = speaker_info['primary_speaker']
            print(f"🎯 Primary speaker: {primary_speaker}")
            
            # Step 4: Extract primary speaker segments
            print("🔍 Extracting primary speaker segments...")
            if self.config.speaker_identification_method == "embeddings":
                filtered_segments = speaker_info['primary_segments']
            else:
                filtered_segments = self._filter_primary_speaker_segments(
                    vad_segments, speaker_info['speaker_segments'], primary_speaker
                )
            
            if not filtered_segments:
                print("❌ No segments found for primary speaker")
                return None
            
            print(f"✅ Found {len(filtered_segments)} segments for primary speaker")
            
            # Step 5: Extract audio
            print("✂️ Extracting primary speaker audio...")
            extracted_path = self._extract_speaker_audio(clean_audio_path, filtered_segments, chunk_name)
            temp_files.append(extracted_path)
            
            if not extracted_path:
                print("❌ Failed to extract speaker audio")
                return None
            
            # Step 6: Enhance with Whisper (if enabled) - FIXED FILTERS
            final_path = extracted_path
            if self.config.whisper_enhancement_enabled and self.whisper_model:
                print("🎙️ Enhancing with Whisper...")
                final_path = self._enhance_with_whisper(extracted_path, chunk_name)
            else:
                # Just copy to final location
                final_path = os.path.join(self.config.output_dir, f"{chunk_name}_voice.wav")
                subprocess.run(["ffmpeg", "-y", "-i", extracted_path, final_path], capture_output=True)
            
            # Calculate statistics
            total_voice_duration = sum(seg['duration'] for seg in filtered_segments)
            
            return {
                **metadata,
                'chunk_name': chunk_name,
                'output_file': final_path,
                'primary_speaker': primary_speaker,
                'segments_count': len(filtered_segments),
                'voice_duration': total_voice_duration,
                'speaker_info': speaker_info,
                'processing_method': f"speechbrain_ecapa_{self.config.speaker_identification_method}",
                'processing_status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Chunk processing error: {e}")
            return None
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_files)

    def _process_single_file(self, input_path: Path) -> Optional[Dict]:
        """Process single file with all fixes applied"""
        base_name = input_path.stem
        temp_files = []
        
        try:
            # Convert to WAV
            print("🔄 Converting to WAV...")
            wav_path = self._convert_to_wav(str(input_path))
            temp_files.append(wav_path)
            
            # Audio cleaning (if enabled)
            clean_audio_path = wav_path
            if self.config.enable_music_removal:
                print("🎵 Removing background music...")
                clean_audio_path = self._remove_background_music(wav_path)
                if clean_audio_path != wav_path:
                    temp_files.append(clean_audio_path)
            
            if self.config.enable_noise_reduction:
                print("🔇 Reducing background noise...")
                clean_audio_path = self._reduce_noise(clean_audio_path)
                if clean_audio_path not in temp_files and clean_audio_path != wav_path:
                    temp_files.append(clean_audio_path)
            
            # Voice Activity Detection
            print("🧠 Voice Activity Detection...")
            vad_segments = self._detect_voice_activity_pyannote(clean_audio_path)
            
            if not vad_segments:
                print("❌ No voice activity detected")
                return None
            
            print(f"✅ Found {len(vad_segments)} voice segments")
            
            # Speaker Analysis (FIXED METHOD)
            if self.config.speaker_identification_method == "embeddings":
                print("🎯 Speaker identification using ECAPA-TDNN embeddings...")
                speaker_info = self._identify_speakers_with_ecapa(clean_audio_path, vad_segments)
            else:
                print("👥 Speaker diarization using Pyannote...")
                speaker_info = self._speaker_diarization_pyannote(clean_audio_path)
            
            if not speaker_info or not speaker_info.get('primary_speaker'):
                print("❌ No speakers identified")
                return None
            
            primary_speaker = speaker_info['primary_speaker']
            print(f"🎯 Primary speaker: {primary_speaker}")
            
            # Extract primary speaker segments
            print("🔍 Extracting primary speaker segments...")
            if self.config.speaker_identification_method == "embeddings":
                filtered_segments = speaker_info['primary_segments']
            else:
                filtered_segments = self._filter_primary_speaker_segments(
                    vad_segments, speaker_info['speaker_segments'], primary_speaker
                )
            
            if not filtered_segments:
                print("❌ No segments found for primary speaker")
                return None
            
            print(f"✅ Found {len(filtered_segments)} segments for primary speaker")
            
            # Extract audio
            print("✂️ Extracting primary speaker audio...")
            extracted_path = self._extract_speaker_audio(clean_audio_path, filtered_segments, base_name)
            temp_files.append(extracted_path)
            
            if not extracted_path:
                print("❌ Failed to extract speaker audio")
                return None
            
            # Enhance with Whisper (FIXED FILTERS)
            final_path = extracted_path
            if self.config.whisper_enhancement_enabled and self.whisper_model:
                print("🎙️ Enhancing with Whisper...")
                final_path = self._enhance_with_whisper(extracted_path, base_name)
            else:
                final_path = os.path.join(self.config.output_dir, f"{base_name}_voice.wav")
                subprocess.run(["ffmpeg", "-y", "-i", extracted_path, final_path], capture_output=True)
            
            # Calculate statistics
            original_size = input_path.stat().st_size
            final_size = Path(final_path).stat().st_size
            compression = (1 - final_size/original_size) * 100
            total_voice_duration = sum(seg['duration'] for seg in filtered_segments)
            
            print(f"📊 File size: {original_size//1024}KB → {final_size//1024}KB ({compression:.1f}% reduction)")
            print(f"📊 Voice duration: {total_voice_duration:.1f}s")
            
            return {
                'input_file': str(input_path),
                'output_file': final_path,
                'primary_speaker': primary_speaker,
                'segments_count': len(filtered_segments),
                'voice_duration': total_voice_duration,
                'compression_ratio': compression,
                'speaker_info': speaker_info,
                'processing_method': f"speechbrain_ecapa_{self.config.speaker_identification_method}",
                'processing_status': 'success',
                'is_chunk': False
            }
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
            
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(temp_files)

    def _remove_background_music(self, input_path: str) -> str:
        """Remove background music using Demucs or Spleeter"""
        try:
            if self.config.music_removal_method == "demucs":
                return self._separate_vocals_demucs(input_path)
            elif self.config.music_removal_method == "spleeter":
                return self._separate_vocals_spleeter(input_path)
            else:
                print(f"   ⚠️ Unknown music removal method: {self.config.music_removal_method}")
                return input_path
        except Exception as e:
            print(f"   ⚠️ Music removal error: {e}")
            return input_path

    def _separate_vocals_demucs(self, input_path: str) -> str:
        """Separate vocals from background music using Demucs"""
        
            
        try:
            print(f"   🎵 Starting Demucs separation (this may take 5-15 minutes)...")
            print(f"   ⏳ Processing {self._get_audio_duration(input_path)/60:.1f} minutes of audio...")
            import tempfile
            
            # Create temporary output directory
            temp_dir = tempfile.mkdtemp()
            vocals_path = tempfile.mktemp(suffix="_vocals.wav")
            
            # Use Demucs to separate vocals from music
            cmd = [
                "python", "-m", "demucs.separate", 
                "--two-stems=vocals",  # Only extract vocals
                "-n", "htdemucs",      # Use HT-Demucs model
                "-o", temp_dir,
                input_path
            ]
            
            print(f"   🎵 Running Demucs vocal separation...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find the vocals file (Demucs creates subfolder structure)
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                demucs_output = os.path.join(temp_dir, "htdemucs", base_name, "vocals.wav")
                
                if os.path.exists(demucs_output):
                    # Copy to our temp location
                    subprocess.run(["cp", demucs_output, vocals_path], capture_output=True)
                    print(f"   ✅ Vocals extracted successfully")
                    return vocals_path
            
            print(f"   ⚠️ Demucs separation failed, using original audio")
            return input_path
            
        except Exception as e:
            print(f"   ⚠️ Demucs error: {e}")
            return input_path

    def _separate_vocals_spleeter(self, input_path: str) -> str:
        """Separate vocals using Spleeter"""
        try:
            import tempfile
            
            temp_dir = tempfile.mkdtemp()
            vocals_path = tempfile.mktemp(suffix="_vocals.wav")
            
            # Run Spleeter to separate vocals
            cmd = [
                "python", "-m", "spleeter", "separate",
                "-p", "spleeter:2stems-16kHz",  # 2 stems: vocals + accompaniment
                "-o", temp_dir,
                input_path
            ]
            
            print(f"   🎵 Running Spleeter vocal separation...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find vocals file
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                spleeter_vocals = os.path.join(temp_dir, base_name, "vocals.wav")
                
                if os.path.exists(spleeter_vocals):
                    subprocess.run(["cp", spleeter_vocals, vocals_path], capture_output=True)
                    print(f"   ✅ Vocals extracted successfully")
                    return vocals_path
            
            print(f"   ⚠️ Spleeter separation failed, using original audio")
            return input_path
            
        except Exception as e:
            print(f"   ⚠️ Spleeter error: {e}")
            return input_path

    def _reduce_noise(self, input_path: str) -> str:
        """Reduce background noise"""
        try:
            import noisereduce as nr
            import librosa
            import soundfile as sf
            import tempfile
            
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            # Reduce noise
            reduced_noise = nr.reduce_noise(y=y, sr=sr)
            
            # Save cleaned audio
            clean_path = tempfile.mktemp(suffix="_clean.wav")
            sf.write(clean_path, reduced_noise, sr)
            
            print(f"   🔇 Background noise reduced")
            return clean_path
            
        except ImportError:
            print(f"   ⚠️ noisereduce not installed, skipping noise reduction")
            print(f"   💡 Install with: pip install noisereduce librosa soundfile")
            return input_path
        except Exception as e:
            print(f"   ⚠️ Noise reduction failed: {e}")
            return input_path

    def _extract_speaker_embeddings(self, audio_path: str, segment: Dict = None) -> np.ndarray:
        """Extract speaker embeddings using SpeechBrain ECAPA-TDNN"""
        try:
            # Load audio
            signal, fs = torchaudio.load(audio_path)
            
            # Extract segment if specified
            if segment:
                start_sample = int(segment['start'] * fs)
                end_sample = int(segment['end'] * fs)
                signal = signal[:, start_sample:end_sample]
            
            # Move to device
            signal = signal.to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.ecapa_classifier.encode_batch(signal)
            
            # Convert to numpy
            return embeddings.squeeze().detach().cpu().numpy()
            
        except Exception as e:
            print(f"⚠️ Embedding extraction error: {e}")
            return None

    def _identify_speakers_with_ecapa(self, wav_path: str, vad_segments: List[Dict]) -> Dict:
        """Identify speakers using ECAPA-TDNN embeddings with IMPROVED THRESHOLD"""
        try:
            speaker_embeddings = {}
            segment_speakers = []
            
            print(f"   🧠 Extracting ECAPA-TDNN embeddings for {len(vad_segments)} segments...")
            print(f"   🎯 Similarity threshold: {self.config.embedding_similarity_threshold}")
            
            # Extract embeddings for each segment
            pbar = tqdm(vad_segments, desc="   🔍 ECAPA Embeddings", unit="segment")
            
            for i, segment in enumerate(pbar):
                embedding = self._extract_speaker_embeddings(wav_path, segment)
                
                if embedding is not None:
                    # Compare with existing speakers
                    assigned_speaker = None
                    min_distance = float('inf')
                    
                    for speaker_id, speaker_emb in speaker_embeddings.items():
                        # Calculate cosine distance
                        distance = cosine(embedding, speaker_emb)
                        
                        if distance < min_distance:
                            min_distance = distance
                            # FIXED: Use improved threshold
                            if distance < self.config.embedding_similarity_threshold:
                                assigned_speaker = speaker_id
                    
                    # Assign to existing speaker or create new one
                    if assigned_speaker is None:
                        assigned_speaker = f"SPEAKER_{len(speaker_embeddings):02d}"
                        speaker_embeddings[assigned_speaker] = embedding
                    
                    segment_speakers.append({
                        **segment,
                        'speaker': assigned_speaker,
                        'embedding_distance': min_distance
                    })
                    
                    pbar.set_postfix({"Speakers": len(speaker_embeddings)})
            
            pbar.close()
            
            # Find primary speaker (most speaking time)
            speaker_times = {}
            for seg in segment_speakers:
                speaker = seg['speaker']
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += seg['duration']
            
            if not speaker_times:
                return {'primary_speaker': None, 'speaker_embeddings': {}, 'primary_segments': []}
            
            primary_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            
            # Get primary speaker segments
            primary_segments = [seg for seg in segment_speakers if seg['speaker'] == primary_speaker]
            
            print(f"   📊 ECAPA-TDNN Speaker Analysis (FIXED):")
            print(f"   📊 Total speakers detected: {len(speaker_embeddings)} (should be much fewer now)")
            for speaker, time_sec in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)[:5]:  # Show top 5
                marker = "👑" if speaker == primary_speaker else "  "
                print(f"     {marker} {speaker}: {time_sec:.1f}s ({time_sec/60:.1f}min)")
            
            if len(speaker_embeddings) > 5:
                print(f"     ... and {len(speaker_embeddings) - 5} more speakers")
            
            return {
                'primary_speaker': primary_speaker,
                'speaker_embeddings': speaker_embeddings,
                'speaker_times': speaker_times,
                'primary_segments': primary_segments,
                'all_segments': segment_speakers,
                'method': 'ecapa_embeddings_fixed'
            }
            
        except Exception as e:
            print(f"   ⚠️ ECAPA speaker identification error: {e}")
            return {'primary_speaker': None, 'speaker_embeddings': {}, 'primary_segments': []}

    def _detect_voice_activity_pyannote(self, wav_path: str) -> List[Dict]:
        """Detect voice activity using Pyannote VAD"""
        try:
            vad_output = self.vad_pipeline(wav_path)
            
            segments = []
            for segment in vad_output.get_timeline().support():
                if segment.duration >= self.config.min_voice_duration:
                    segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'duration': segment.duration
                    })
            
            return segments
            
        except Exception as e:
            print(f"⚠️ Pyannote VAD error: {e}")
            return []

    def _speaker_diarization_pyannote(self, wav_path: str) -> Dict:
        """Perform speaker diarization using Pyannote (PREFERRED METHOD)"""
        try:
            diarization_output = self.diarization_pipeline(wav_path)
            
            segments = []
            speaker_times = {}
            
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.duration,
                    'speaker': speaker
                })
                
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += turn.duration
            
            if not speaker_times:
                return {'primary_speaker': None, 'speaker_segments': []}
            
            primary_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            
            print(f"   📊 Pyannote Diarization (RECOMMENDED):")
            print(f"   📊 Total speakers detected: {len(speaker_times)}")
            for speaker, time_sec in speaker_times.items():
                marker = "👑" if speaker == primary_speaker else "  "
                print(f"     {marker} {speaker}: {time_sec:.1f}s ({time_sec/60:.1f}min)")
            
            return {
                'primary_speaker': primary_speaker,
                'speaker_segments': segments,
                'speaker_times': speaker_times,
                'method': 'pyannote_diarization'
            }
            
        except Exception as e:
            print(f"⚠️ Pyannote diarization error: {e}")
            return {'primary_speaker': None, 'speaker_segments': []}

    def _filter_primary_speaker_segments(self, vad_segments: List[Dict], 
                                       speaker_segments: List[Dict], 
                                       primary_speaker: str) -> List[Dict]:
        """Filter segments for primary speaker only"""
        filtered = []
        
        # Get primary speaker intervals
        primary_intervals = [
            (seg['start'], seg['end']) 
            for seg in speaker_segments 
            if seg['speaker'] == primary_speaker
        ]
        
        # Find overlaps with VAD segments
        for vad_seg in vad_segments:
            for speaker_start, speaker_end in primary_intervals:
                # Check interval overlap
                overlap_start = max(vad_seg['start'], speaker_start)
                overlap_end = min(vad_seg['end'], speaker_end)
                
                if overlap_start < overlap_end:  # There is overlap
                    overlap_duration = overlap_end - overlap_start
                    
                    # If significant overlap
                    if overlap_duration > vad_seg['duration'] * self.config.overlap_threshold:
                        filtered.append({
                            'start': overlap_start,
                            'end': overlap_end,
                            'duration': overlap_duration
                        })
                        break
        
        return filtered

    def _convert_to_wav(self, input_path: str) -> str:
        """Convert MP3 to WAV"""
        wav_path = tempfile.mktemp(suffix=".wav")
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"Failed to convert {input_path} to WAV")
        return wav_path

    def _extract_speaker_audio(self, wav_path: str, segments: List[Dict], base_name: str) -> str:
        """Extract audio segments for primary speaker"""
        if not segments:
            return None
        
        try:
            # Create segment files
            segment_files = []
            
            for i, segment in enumerate(segments):
                seg_file = tempfile.mktemp(suffix=f"_seg_{i}.wav")
                
                cmd = [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-ss", str(segment['start']),
                    "-t", str(segment['duration']),
                    "-c", "copy", seg_file
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    segment_files.append(seg_file)
            
            if not segment_files:
                return None
            
            # Concatenate segments
            output_path = tempfile.mktemp(suffix="_extracted.wav")
            
            # Create file list for FFmpeg
            list_file = tempfile.mktemp(suffix=".txt")
            with open(list_file, 'w') as f:
                for seg_file in segment_files:
                    f.write(f"file '{seg_file}'\n")
            
            # Concatenate
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file, "-c", "copy", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            # Cleanup
            for seg_file in segment_files:
                try:
                    os.remove(seg_file)
                except:
                    pass
            
            try:
                os.remove(list_file)
            except:
                pass
            
            return output_path if result.returncode == 0 else None
                
        except Exception as e:
            print(f"⚠️ Audio extraction error: {e}")
            return None

    def _enhance_with_whisper(self, input_path: str, base_name: str) -> str:
        """Enhanced audio using Whisper analysis with FIXED FILTERS"""
        try:
            if not self.whisper_model:
                # Just copy file if Whisper is disabled
                enhanced_path = os.path.join(self.config.output_dir, f"{base_name}_voice.wav")
                cmd = ["ffmpeg", "-y", "-i", input_path, enhanced_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(enhanced_path):
                    return enhanced_path
                else:
                    print(f"   ⚠️ Basic copy failed, returning original: {result.stderr}")
                    return input_path
            
            # Run Whisper transcription
            result = self.whisper_model.transcribe(input_path)
            
            # Create enhanced file path
            enhanced_path = os.path.join(self.config.output_dir, f"{base_name}_enhanced.wav")
            
            # Apply audio enhancement with FIXED FILTER
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", "volume=1.2,dynaudnorm",  # ✅ FIXED: Simple, working filter
                enhanced_path
            ]
            
            print(f"   🎬 Running FFmpeg enhancement with FIXED filter...")
            ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if FFmpeg succeeded
            if ffmpeg_result.returncode != 0:
                print(f"   ⚠️ FFmpeg enhancement failed, trying basic volume: {ffmpeg_result.stderr}")
                # Fallback: just volume boost
                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-af", "volume=1.2",  # Even simpler fallback
                    enhanced_path
                ]
                ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if ffmpeg_result.returncode != 0:
                    print(f"   ⚠️ Even basic enhancement failed, copying original")
                    # Final fallback: just copy
                    subprocess.run(["ffmpeg", "-y", "-i", input_path, enhanced_path], capture_output=True)
            
            # Verify the enhanced file exists
            if not os.path.exists(enhanced_path):
                print(f"   ⚠️ Enhanced file missing, creating copy")
                subprocess.run(["ffmpeg", "-y", "-i", input_path, enhanced_path], capture_output=True)
            
            # Save transcription
            transcript_path = enhanced_path.replace('.wav', '_transcript.txt')
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"Language: {result['language']}\n")
                f.write(f"Processing: SpeechBrain ECAPA-TDNN + Pyannote (FIXED VERSION)\n")
                f.write("Audio Cleaning Applied: ")
                if self.config.enable_music_removal:
                    f.write(f"Music removal ({self.config.music_removal_method}) ")
                if self.config.enable_noise_reduction:
                    f.write("Noise reduction ")
                f.write("\n")
                f.write("=" * 50 + "\n")
                f.write(result['text'])
            
            print(f"   📝 Transcript saved: {transcript_path}")
            print(f"   🌍 Language detected: {result['language']}")
            print(f"   ✅ Enhanced file created: {enhanced_path}")
            
            return enhanced_path
            
        except Exception as e:
            print(f"   ⚠️ Whisper enhancement error: {e}")
            # Fallback: copy input file
            fallback_path = os.path.join(self.config.output_dir, f"{base_name}_voice.wav")
            cmd = ["ffmpeg", "-y", "-i", input_path, fallback_path]
            subprocess.run(cmd, capture_output=True)
            return fallback_path

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
        
        summary_path = os.path.join(self.config.output_dir, "speechbrain_pyannote_complete_summary.json")
        
        # Separate chunks from regular files
        chunks = [r for r in results if r.get('is_chunk', False)]
        regular_files = [r for r in results if not r.get('is_chunk', False)]
        
        summary = {
            'total_results': len(results),
            'regular_files': len(regular_files),
            'chunks': len(chunks),
            'processing_method': 'SpeechBrain ECAPA-TDNN + Pyannote (Complete Fixed Version)',
            'fixes_applied': [
                'Increased embedding similarity threshold to 0.75',
                'Changed default method to pyannote diarization',
                'Fixed FFmpeg enhancement filters',
                'Added automatic chunking for long files',
                'Improved speaker grouping logic',
                'Added optional music removal (Demucs/Spleeter)',
                'Added optional noise reduction'
            ],
            'processing_config': {
                'device': str(self.device),
                'chunking_enabled': self.config.enable_chunking,
                'chunk_duration_minutes': self.config.chunk_duration_minutes,
                'speechbrain_model': self.config.ecapa_model_source,
                'speaker_identification': self.config.speaker_identification_method,
                'similarity_threshold': self.config.embedding_similarity_threshold,
                'pyannote_models': {
                    'vad': self.config.vad_model,
                    'diarization': self.config.diarization_model
                },
                'whisper_model': self.config.whisper_model_size,
                'audio_cleaning': {
                    'music_removal_enabled': self.config.enable_music_removal,
                    'music_removal_method': self.config.music_removal_method,
                    'noise_reduction_enabled': self.config.enable_noise_reduction
                }
            },
            'results': results,
            'statistics': {
                'total_voice_duration': sum(r['voice_duration'] for r in results),
                'average_segments': sum(r['segments_count'] for r in results) / len(results) if results else 0,
                'files_chunked': len(set(r.get('original_filename') for r in chunks if r.get('original_filename')))
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Complete processing results summary saved: {summary_path}")

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SpeechBrain ECAPA-TDNN + Pyannote Audio Processor (Complete Fixed Version)")
    parser.add_argument("input_folder", help="Folder containing MP3 files")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    print("🧠 SpeechBrain ECAPA-TDNN + Pyannote Audio Processor (COMPLETE FIXED VERSION)")
    print("📦 Required packages: speechbrain pyannote.audio torch torchaudio openai-whisper scipy")
    print("📦 Optional packages: demucs spleeter noisereduce librosa soundfile")
    print("🔑 Required: HuggingFace token in config.json")
    print("🔧 Fixes: Improved speaker grouping, Fixed FFmpeg filters, Auto-chunking, Audio cleaning")
    
    try:
        # Initialize processor with config
        processor = SpeechBrainPyannoteProcessor(config_path=args.config)
        
        # Process files
        results = processor.process_folder(args.input_folder)
        
        print(f"\n🎉 COMPLETE SPEECHBRAIN + PYANNOTE PROCESSING COMPLETED!")
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
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("1. Valid config.json with HuggingFace token")
        print("2. Accepted model conditions on HuggingFace")
        print("3. Installed all required packages")
        print("4. Optional: Install demucs/spleeter for music removal")
        print("5. Optional: Install noisereduce for noise reduction")
