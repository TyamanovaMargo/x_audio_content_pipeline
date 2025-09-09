import os
import subprocess
import time
import tempfile
from typing import List, Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm
from pyannote.audio import Pipeline
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
import whisper
import re

class OverlapDetector:
    def __init__(self, output_dir="clean_chunks", chunk_duration_minutes=6,
                 overlap_threshold=0.3, noise_reduction=True, vad_threshold=0.05, 
                 max_workers=4, huggingface_token=None):
        """
        Enhanced overlapping voices detector with Voice Activity Detection (VAD) and audio cleaning
        
        Args:
            output_dir: Directory for clean chunks
            chunk_duration_minutes: Chunk duration in minutes (default 6)
            overlap_threshold: Overlap threshold (0.3 = 30% overlap time = remove chunk)
            noise_reduction: Enable light audio cleaning before processing (default True)
            vad_threshold: Voice activity threshold (0.05 = 5% minimum voice activity)
            max_workers: Maximum parallel workers for processing (default 4)
            huggingface_token: HuggingFace token for pyannote
        """
        self.output_dir = output_dir
        self.chunk_duration_minutes = chunk_duration_minutes
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.overlap_threshold = overlap_threshold
        self.noise_reduction = noise_reduction
        self.vad_threshold = vad_threshold
        self.max_workers = max_workers
        self.temp_chunks_dir = os.path.join(output_dir, "temp_chunks")
        self.cleaned_chunks_dir = os.path.join(output_dir, "cleaned_chunks")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_chunks_dir, exist_ok=True)
        os.makedirs(self.cleaned_chunks_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize pyannote pipelines
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diarization_pipeline = None
        self.vad_pipeline = None
        
        if huggingface_token:
            try:
                # Load speaker diarization pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=huggingface_token
                ).to(self.device)
                self.logger.info("✅ Pyannote speaker diarization model loaded")
                
                # Load VAD pipeline
                self.vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=huggingface_token
                ).to(self.device)
                self.logger.info("✅ Pyannote VAD model loaded")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load pyannote models: {e}")
                self.diarization_pipeline = None
                self.vad_pipeline = None
        else:
            self.logger.warning("⚠️ No HuggingFace token provided, using fallback detection methods")

    def process_extracted_samples(self, extracted_samples: List[Dict]) -> List[Dict]:
        """
        Process extracted samples with enhanced chunking, cleaning, and dual filtering
        
        Workflow:
        1. Split files >5min into 6min chunks
        2. Apply light audio cleaning
        3. Voice activity detection (discard chunks without voice)
        4. Overlap detection (discard chunks with overlapping voices)
        5. Keep only clean single-speaker voice chunks
        """
        if not extracted_samples:
            self.logger.info("🔍 No extracted samples to process")
            return []

        self.logger.info(f"🔍 STAGE 6.5: Enhanced Audio Processing Pipeline")
        self.logger.info(f"📊 Processing {len(extracted_samples)} extracted samples")
        self.logger.info(f"✂️ Chunk duration: {self.chunk_duration_minutes} minutes")
        self.logger.info(f"🧹 Audio cleaning: {'Enabled' if self.noise_reduction else 'Disabled'}")
        self.logger.info(f"🗣️ VAD threshold: {self.vad_threshold * 100}% (minimum voice activity)")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}% (maximum overlap)")

        # Statistics tracking
        stats = {
            'total_files': len(extracted_samples),
            'total_chunks_created': 0,
            'voice_positive_chunks': 0,
            'overlap_rejected_chunks': 0,
            'final_clean_chunks': 0
        }

        clean_chunks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i, sample in enumerate(extracted_samples, 1):
                sample_file = sample.get('sample_file')
                username = sample.get('processed_username', 'unknown')
                platform = sample.get('platform_source', 'unknown')

                if not sample_file or not os.path.exists(sample_file):
                    self.logger.warning(f"⚠️ [{i}/{len(extracted_samples)}] Sample file not found: {sample_file}")
                    continue

                self.logger.info(f"🎤 [{i}/{len(extracted_samples)}] Processing @{username} ({platform})")

                # Check file duration
                duration = self._get_audio_duration(sample_file)
                duration_minutes = duration / 60
                self.logger.info(f"📏 File duration: {duration_minutes:.1f} minutes")

                # Process based on duration (changed threshold from 10 to 5 minutes)
                if duration_minutes <= 5:
                    # Small file - process entire file
                    self.logger.info("📄 File ≤5min, processing whole file")
                    stats['total_chunks_created'] += 1
                    
                    future = executor.submit(self._process_single_file, sample_file, username, platform, sample, stats)
                    futures.append(future)
                else:
                    # Large file - split into chunks
                    self.logger.info(f"✂️ File >5min, splitting into {self.chunk_duration_minutes}min chunks")
                    chunks = self._split_audio_into_chunks(sample_file, username, platform)
                    stats['total_chunks_created'] += len(chunks)

                    # Process each chunk
                    for chunk_idx, chunk_file in enumerate(chunks, 1):
                        self.logger.info(f"🔍 Processing chunk {chunk_idx}/{len(chunks)}")
                        
                        future = executor.submit(self._process_chunk_file, chunk_file, username, platform, chunk_idx, len(chunks), sample, stats)
                        futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        clean_chunks.append(result)
                        stats['final_clean_chunks'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error processing sample: {e}")

        # Cleanup temporary directories
        self._cleanup_temp_files()

        # Final statistics
        self._log_processing_summary(stats)

        return clean_chunks

    def _process_single_file(self, sample_file: str, username: str, platform: str, sample: Dict, stats: Dict) -> Optional[Dict]:
        """Process a single file through the complete pipeline"""
        try:
            # Step 1: Audio cleaning
            cleaned_file = self._apply_audio_cleaning(sample_file, username, platform, 1) if self.noise_reduction else sample_file
            if not cleaned_file:
                self.logger.warning("⚠️ Audio cleaning failed, skipping file")
                return None

            # Step 2: Voice activity detection
            self.logger.info("🗣️ Checking voice activity...")
            voice_result = self._detect_voice_activity(cleaned_file)
            voice_pct = voice_result.get('voice_percentage', 0)
            
            if not voice_result.get('has_voice', False) or voice_pct < self.vad_threshold * 100:
                self.logger.info(f"❌ File REJECTED - No sufficient voice activity ({voice_pct:.1f}% < {self.vad_threshold * 100}%)")
                self._cleanup_file(cleaned_file, sample_file)
                return None
            
            stats['voice_positive_chunks'] += 1
            self.logger.info(f"✅ Voice activity detected ({voice_pct:.1f}%)")

            # Step 3: Overlap detection
            self.logger.info("🔍 Checking for overlapping voices...")
            overlap_result = self._detect_overlapping_voices(cleaned_file)
            overlap_pct = overlap_result.get('overlap_percentage', 0)
            
            if overlap_result.get('has_overlap', False):
                self.logger.info(f"❌ File REJECTED - Overlapping voices detected ({overlap_pct:.1f}% > {self.overlap_threshold * 100}%)")
                stats['overlap_rejected_chunks'] += 1
                self._cleanup_file(cleaned_file, sample_file)
                return None

            # Step 4: Save clean file
            final_file = self._save_clean_chunk(cleaned_file, username, platform, 1)
            if not final_file:
                self.logger.warning("⚠️ Failed to save clean file")
                return None

            # Create result data
            chunk_data = sample.copy()
            chunk_data.update({
                'clean_chunk_file': final_file,
                'chunk_number': 1,
                'total_chunks': 1,
                'original_duration': self._get_audio_duration(sample_file),
                'has_overlap': False,
                'overlap_percentage': overlap_pct,
                'voice_percentage': voice_pct,
                'speakers_detected': overlap_result.get('speakers_count', 1),
                'processing_method': 'single_file_pipeline'
            })
            
            self.logger.info(f"✅ File ACCEPTED - Clean single-speaker voice content")
            return chunk_data

        except Exception as e:
            self.logger.error(f"❌ Error processing single file: {e}")
            return None

    def _process_chunk_file(self, chunk_file: str, username: str, platform: str, chunk_idx: int, total_chunks: int, sample: Dict, stats: Dict) -> Optional[Dict]:
        """Process a chunk file through the complete pipeline"""
        try:
            # Step 1: Audio cleaning
            cleaned_file = self._apply_audio_cleaning(chunk_file, username, platform, chunk_idx) if self.noise_reduction else chunk_file
            if not cleaned_file:
                self.logger.warning(f"⚠️ Audio cleaning failed for chunk {chunk_idx}, skipping")
                self._cleanup_file(chunk_file)
                return None

            # Step 2: Voice activity detection
            self.logger.info(f"🗣️ Checking voice activity in chunk {chunk_idx}...")
            voice_result = self._detect_voice_activity(cleaned_file)
            voice_pct = voice_result.get('voice_percentage', 0)
            
            if not voice_result.get('has_voice', False) or voice_pct < self.vad_threshold * 100:
                self.logger.info(f"❌ Chunk {chunk_idx} REJECTED - No sufficient voice activity ({voice_pct:.1f}% < {self.vad_threshold * 100}%)")
                self._cleanup_file(cleaned_file, chunk_file)
                return None
            
            stats['voice_positive_chunks'] += 1
            self.logger.info(f"✅ Chunk {chunk_idx} has voice activity ({voice_pct:.1f}%)")

            # Step 3: Overlap detection
            self.logger.info(f"🔍 Checking for overlapping voices in chunk {chunk_idx}...")
            overlap_result = self._detect_overlapping_voices(cleaned_file)
            overlap_pct = overlap_result.get('overlap_percentage', 0)
            
            if overlap_result.get('has_overlap', False):
                self.logger.info(f"❌ Chunk {chunk_idx} REJECTED - Overlapping voices detected ({overlap_pct:.1f}% > {self.overlap_threshold * 100}%)")
                stats['overlap_rejected_chunks'] += 1
                self._cleanup_file(cleaned_file, chunk_file)
                return None

            # Step 4: Save clean chunk
            final_file = self._save_clean_chunk(cleaned_file, username, platform, chunk_idx)
            if not final_file:
                self.logger.warning(f"⚠️ Failed to save clean chunk {chunk_idx}")
                return None

            # Create result data
            chunk_data = sample.copy()
            chunk_data.update({
                'clean_chunk_file': final_file,
                'chunk_number': chunk_idx,
                'total_chunks': total_chunks,
                'original_duration': self._get_audio_duration(chunk_file),
                'chunk_duration': self.chunk_duration_seconds,
                'has_overlap': False,
                'overlap_percentage': overlap_pct,
                'voice_percentage': voice_pct,
                'speakers_detected': overlap_result.get('speakers_count', 1),
                'processing_method': 'chunk_pipeline'
            })
            
            self.logger.info(f"✅ Chunk {chunk_idx} ACCEPTED - Clean single-speaker voice content")
            return chunk_data

        except Exception as e:
            self.logger.error(f"❌ Error processing chunk {chunk_idx}: {e}")
            return None

    def _apply_audio_cleaning(self, audio_path: str, username: str, platform: str, chunk_idx: int) -> Optional[str]:
        """Apply light audio cleaning to improve voice clarity"""
        try:
            timestamp = int(time.time())
            cleaned_filename = f"{username}_{platform}_{timestamp}_chunk{chunk_idx:02d}_cleaned.wav"
            cleaned_path = os.path.join(self.cleaned_chunks_dir, cleaned_filename)
            
            self.logger.info(f"🧹 Applying audio cleaning to chunk {chunk_idx}...")
            
            # FFmpeg command for light audio cleaning
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                # Normalize audio levels
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
                # Reduce noise (light highpass filter to remove low-frequency noise)
                '-af', 'highpass=f=80',
                # Remove silence at beginning and end
                '-af', 'silenceremove=start_periods=1:start_silence=0.1:start_threshold=0.02:detection=peak',
                # Standard format
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-c:a', 'pcm_s16le',  # PCM 16-bit
                cleaned_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(cleaned_path):
                # Verify the cleaned file has reasonable duration
                cleaned_duration = self._get_audio_duration(cleaned_path)
                if cleaned_duration > 1.0:  # At least 1 second
                    self.logger.info(f"✅ Audio cleaning completed for chunk {chunk_idx}")
                    return cleaned_path
                else:
                    self.logger.warning(f"⚠️ Cleaned audio too short ({cleaned_duration:.1f}s)")
                    try:
                        os.remove(cleaned_path)
                    except:
                        pass
                    return None
            else:
                self.logger.warning(f"⚠️ Audio cleaning failed for chunk {chunk_idx}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in audio cleaning: {e}")
            return None

    def _save_clean_chunk(self, source_file: str, username: str, platform: str, chunk_idx: int) -> Optional[str]:
        """Save the final clean chunk to output directory"""
        try:
            timestamp = int(time.time())
            if chunk_idx == 1:
                final_filename = f"{username}_{platform}_{timestamp}_clean.wav"
            else:
                final_filename = f"{username}_{platform}_{timestamp}_chunk{chunk_idx:02d}_clean.wav"
            
            final_path = os.path.join(self.output_dir, final_filename)
            
            # Copy to final location with standard format
            cmd = [
                'ffmpeg', '-y', '-i', source_file,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-c:a', 'pcm_s16le',  # PCM 16-bit
                final_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(final_path):
                return final_path
            else:
                self.logger.warning(f"⚠️ Failed to save final clean chunk {chunk_idx}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving clean chunk: {e}")
            return None

    def _cleanup_file(self, *files):
        """Clean up temporary files"""
        for file_path in files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.debug(f"Cleanup warning for {file_path}: {e}")

    def _cleanup_temp_files(self):
        """Clean up all temporary directories"""
        try:
            # Clean temp chunks directory
            if os.path.exists(self.temp_chunks_dir):
                for file in os.listdir(self.temp_chunks_dir):
                    file_path = os.path.join(self.temp_chunks_dir, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass
                try:
                    os.rmdir(self.temp_chunks_dir)
                except:
                    pass
            
            # Clean cleaned chunks directory
            if os.path.exists(self.cleaned_chunks_dir):
                for file in os.listdir(self.cleaned_chunks_dir):
                    file_path = os.path.join(self.cleaned_chunks_dir, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass
                try:
                    os.rmdir(self.cleaned_chunks_dir)
                except:
                    pass
                    
        except Exception as e:
            self.logger.debug(f"Cleanup warning: {e}")

    def _log_processing_summary(self, stats: Dict):
        """Log comprehensive processing summary"""
        self.logger.info(f"\n🎯 ENHANCED AUDIO PROCESSING SUMMARY:")
        self.logger.info(f"📊 Original files: {stats['total_files']}")
        self.logger.info(f"✂️ Total chunks created: {stats['total_chunks_created']}")
        self.logger.info(f"🗣️ Voice-positive chunks: {stats['voice_positive_chunks']}")
        self.logger.info(f"❌ Overlap-rejected chunks: {stats['overlap_rejected_chunks']}")
        self.logger.info(f"✅ Final clean chunks: {stats['final_clean_chunks']}")
        
        if stats['total_chunks_created'] > 0:
            voice_rate = (stats['voice_positive_chunks'] / stats['total_chunks_created']) * 100
            final_rate = (stats['final_clean_chunks'] / stats['total_chunks_created']) * 100
            self.logger.info(f"📈 Voice detection rate: {voice_rate:.1f}%")
            self.logger.info(f"📈 Final acceptance rate: {final_rate:.1f}%")
        
        self.logger.info(f"📁 Clean chunks directory: {self.output_dir}")
        self.logger.info(f"🧹 Audio cleaning: {'Applied' if self.noise_reduction else 'Skipped'}")
        self.logger.info(f"🗣️ VAD threshold: {self.vad_threshold * 100}%")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}%")

    def process_audio_directory(self, voice_samples_dir: str) -> List[Dict]:
        """
        Process audio directory containing voice samples (MP3/WAV files)
        This is the main entry point called from the pipeline
        """
        if not os.path.exists(voice_samples_dir):
            self.logger.error(f"❌ Voice samples directory not found: {voice_samples_dir}")
            return []

        self.logger.info(f"🎵 Processing voice samples directory: {voice_samples_dir}")
        
        # Find all audio files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
            audio_files.extend(Path(voice_samples_dir).glob(ext))
            audio_files.extend(Path(voice_samples_dir).glob(ext.upper()))

        if not audio_files:
            self.logger.warning(f"⚠️ No audio files found in {voice_samples_dir}")
            return []

        self.logger.info(f"📁 Found {len(audio_files)} audio files to process")

        # Convert file paths to extracted samples format
        extracted_samples = []
        for audio_file in audio_files:
            filename = audio_file.stem
            # Extract username and platform from filename if possible
            parts = filename.split('_')
            username = parts[0] if parts else 'unknown'
            platform = parts[1] if len(parts) > 1 else 'unknown'
            
            sample_data = {
                'sample_file': str(audio_file),
                'processed_username': username,
                'platform_source': platform,
                'original_filename': filename
            }
            extracted_samples.append(sample_data)

        # Process using existing pipeline
        return self.process_extracted_samples(extracted_samples)

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                return duration
            else:
                self.logger.warning(f"⚠️ Could not get duration for {audio_path}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"❌ Error getting audio duration: {e}")
            return 0.0

    def _split_audio_into_chunks(self, audio_path: str, username: str, platform: str) -> List[str]:
        """Split audio file into chunks of specified duration"""
        try:
            chunks = []
            duration = self._get_audio_duration(audio_path)
            
            if duration <= 0:
                self.logger.warning(f"⚠️ Invalid duration for {audio_path}")
                return []

            num_chunks = int(np.ceil(duration / self.chunk_duration_seconds))
            self.logger.info(f"✂️ Splitting into {num_chunks} chunks of {self.chunk_duration_minutes}min each")

            timestamp = int(time.time())
            
            for i in range(num_chunks):
                start_time = i * self.chunk_duration_seconds
                chunk_filename = f"{username}_{platform}_{timestamp}_chunk{i+1:02d}.wav"
                chunk_path = os.path.join(self.temp_chunks_dir, chunk_filename)
                
                # Create chunk using ffmpeg
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(self.chunk_duration_seconds),
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',      # Mono
                    '-c:a', 'pcm_s16le',  # PCM 16-bit
                    chunk_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(chunk_path):
                    # Verify chunk has reasonable duration
                    chunk_duration = self._get_audio_duration(chunk_path)
                    if chunk_duration > 1.0:  # At least 1 second
                        chunks.append(chunk_path)
                        self.logger.info(f"✅ Created chunk {i+1}/{num_chunks}: {chunk_duration:.1f}s")
                    else:
                        self.logger.warning(f"⚠️ Chunk {i+1} too short ({chunk_duration:.1f}s), skipping")
                        try:
                            os.remove(chunk_path)
                        except:
                            pass
                else:
                    self.logger.warning(f"⚠️ Failed to create chunk {i+1}/{num_chunks}")

            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error splitting audio: {e}")
            return []

    def _detect_voice_activity(self, audio_path: str) -> Dict:
        """
        Detect voice activity using pyannote VAD or fallback method
        Returns dict with voice_percentage and has_voice status
        """
        try:
            duration = self._get_audio_duration(audio_path)
            if duration <= 0:
                return {'voice_percentage': 0.0, 'has_voice': False, 'method': 'error'}

            # Try pyannote VAD first
            if self.vad_pipeline:
                try:
                    vad = self.vad_pipeline(audio_path)
                    voice_regions = vad.get_timeline()
                    voice_duration = voice_regions.duration() if voice_regions else 0
                    voice_percentage = (voice_duration / duration) * 100 if duration > 0 else 0
                    has_voice = voice_percentage > (self.vad_threshold * 100)
                    
                    return {
                        'voice_percentage': voice_percentage,
                        'has_voice': has_voice,
                        'method': 'pyannote_vad',
                        'voice_duration': voice_duration,
                        'total_duration': duration
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Pyannote VAD failed: {e}, using fallback")

            # Fallback: Enhanced Voice vs Music detection using librosa
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                
                # Calculate multiple features for voice detection
                frame_length = int(0.025 * sr)  # 25ms frames
                hop_length = int(0.010 * sr)    # 10ms hop
                
                # 1. RMS energy
                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                
                # 2. Zero-crossing rate (voice has moderate ZCR, music often higher)
                zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
                
                # 3. Spectral centroid (voice typically 1-4kHz, music more varied)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
                
                # 4. MFCCs (voice has characteristic patterns)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
                
                # 5. Spectral rolloff (voice has specific rolloff patterns)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
                
                # 6. Fundamental frequency detection (human voice range)
                try:
                    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
                    valid_f0_frames = ~np.isnan(f0)
                    f0_percentage = np.sum(valid_f0_frames) / len(f0) * 100 if len(f0) > 0 else 0
                except:
                    f0_percentage = 0
                
                # Voice detection criteria - much stricter
                energy_threshold = np.percentile(rms, 40)  # Higher energy threshold
                
                # Voice-specific frequency range (human speech is typically 300-3400 Hz)
                voice_centroid_frames = (spectral_centroids >= 300) & (spectral_centroids <= 3400)
                voice_centroid_percentage = np.sum(voice_centroid_frames) / len(voice_centroid_frames) * 100
                
                # ZCR should be moderate for voice (not too high like music)
                moderate_zcr_frames = zcr < np.percentile(zcr, 60)
                moderate_zcr_percentage = np.sum(moderate_zcr_frames) / len(moderate_zcr_frames) * 100
                
                # Energy-based voice frames
                energy_voice_frames = rms > energy_threshold
                energy_voice_percentage = np.sum(energy_voice_frames) / len(energy_voice_frames) * 100
                
                # MFCC consistency check (voice has more consistent patterns)
                mfcc_variance = np.var(mfccs, axis=1)
                mfcc_consistency_score = 1 / (1 + np.mean(mfcc_variance))
                
                # Combine all criteria for strict voice detection
                # All conditions must be met for voice detection
                voice_criteria = {
                    'energy_voice': energy_voice_percentage > 30,  # At least 30% energy activity
                    'voice_frequency': voice_centroid_percentage > 60,  # 60% in voice frequency range
                    'moderate_zcr': moderate_zcr_percentage > 50,  # 50% moderate ZCR
                    'fundamental_freq': f0_percentage > 15,  # At least 15% valid F0
                    'mfcc_consistent': mfcc_consistency_score > 0.3  # Consistent MFCC patterns
                }
                
                # Count how many criteria are met
                criteria_met = sum(voice_criteria.values())
                min_criteria_required = 4  # At least 4 out of 5 criteria must be met
                
                # Final voice percentage is the minimum of all percentages
                voice_percentage = min(
                    energy_voice_percentage,
                    voice_centroid_percentage,
                    moderate_zcr_percentage,
                    max(f0_percentage * 4, 20)  # Scale F0 percentage
                ) if criteria_met >= min_criteria_required else 0
                
                has_voice = voice_percentage > (self.vad_threshold * 100) and criteria_met >= min_criteria_required
                
                # Log detailed analysis for debugging
                self.logger.info(f"🔍 Voice analysis: energy={energy_voice_percentage:.1f}%, "
                               f"freq_range={voice_centroid_percentage:.1f}%, "
                               f"zcr={moderate_zcr_percentage:.1f}%, "
                               f"f0={f0_percentage:.1f}%, "
                               f"criteria_met={criteria_met}/{len(voice_criteria)}")
                
                return {
                    'voice_percentage': voice_percentage,
                    'has_voice': has_voice,
                    'method': 'enhanced_multi_feature_analysis',
                    'criteria_met': criteria_met,
                    'min_required': min_criteria_required,
                    'energy_voice_pct': energy_voice_percentage,
                    'voice_freq_pct': voice_centroid_percentage,
                    'moderate_zcr_pct': moderate_zcr_percentage,
                    'f0_pct': f0_percentage,
                    'mfcc_consistency': mfcc_consistency_score
                }
                
            except Exception as e:
                self.logger.error(f"❌ Fallback VAD failed: {e}")
                return {'voice_percentage': 0.0, 'has_voice': False, 'method': 'error'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in voice activity detection: {e}")
            return {'voice_percentage': 0.0, 'has_voice': False, 'method': 'error'}

    def _detect_overlapping_voices(self, audio_path: str) -> Dict:
        """
        Detect overlapping voices using pyannote speaker diarization or fallback method
        Returns dict with overlap info and speaker count
        """
        try:
            duration = self._get_audio_duration(audio_path)
            if duration <= 0:
                return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 0, 'method': 'error'}

            # Try pyannote diarization first
            if self.diarization_pipeline:
                try:
                    diarization = self.diarization_pipeline(audio_path)
                    
                    # Get overlap regions
                    overlaps = diarization.get_overlap()
                    overlap_duration = overlaps.duration() if overlaps else 0
                    overlap_percentage = (overlap_duration / duration) * 100 if duration > 0 else 0
                    
                    # Count unique speakers
                    speakers = set()
                    for segment, _, speaker in diarization.itertracks(yield_label=True):
                        speakers.add(speaker)
                    speakers_count = len(speakers)
                    
                    has_overlap = overlap_percentage > (self.overlap_threshold * 100)
                    
                    return {
                        'has_overlap': has_overlap,
                        'overlap_percentage': overlap_percentage,
                        'overlap_duration': overlap_duration,
                        'speakers_count': speakers_count,
                        'total_duration': duration,
                        'method': 'pyannote_diarization'
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Pyannote diarization failed: {e}, using fallback")

            # Fallback: Simple energy-based overlap detection
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                
                # Calculate spectral features that might indicate multiple speakers
                frame_length = int(0.025 * sr)  # 25ms frames
                hop_length = int(0.010 * sr)    # 10ms hop
                
                # Spectral centroid and rolloff (different speakers have different spectral characteristics)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
                
                # High variance in spectral features might indicate multiple speakers
                centroid_var = np.var(spectral_centroids)
                rolloff_var = np.var(spectral_rolloff)
                
                # Thresholds based on typical single vs multi-speaker variance
                centroid_threshold = np.percentile(spectral_centroids, 85)
                rolloff_threshold = np.percentile(spectral_rolloff, 85)
                
                # Estimate overlap based on spectral variance (rough heuristic)
                overlap_score = min(100, (centroid_var / 1000000 + rolloff_var / 1000000) * 50)
                has_overlap = overlap_score > (self.overlap_threshold * 100)
                
                # Rough speaker count estimation (very basic)
                speakers_count = 1 if overlap_score < 20 else 2
                
                return {
                    'has_overlap': has_overlap,
                    'overlap_percentage': overlap_score,
                    'speakers_count': speakers_count,
                    'method': 'spectral_variance_fallback',
                    'centroid_variance': centroid_var,
                    'rolloff_variance': rolloff_var
                }
                
            except Exception as e:
                self.logger.error(f"❌ Fallback overlap detection failed: {e}")
                return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 1, 'method': 'error'}
                
        except Exception as e:
            self.logger.error(f"❌ Error in overlap detection: {e}")
            return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 1, 'method': 'error'}
