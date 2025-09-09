import os
import subprocess
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC, AutoTokenizer, Wav2Vec2Processor
import librosa
import re

class W2VBertOverlapDetector:
    def __init__(
        self,
        output_dir: str,
        chunk_duration_minutes: int = 6,
        overlap_threshold: float = 0.3,
        vad_threshold: float = 0.15,
        noise_reduction: bool = True,
        max_workers: int = 2,
        model_name: str = "facebook/wav2vec2-base-960h",  # Fixed: Use working model
        huggingface_token: str = None,
    ):
        """
        Initialize the Wav2Vec2 overlap detector with proper authentication.
        
        Args:
            output_dir: Directory for output files
            chunk_duration_minutes: Duration of audio chunks in minutes
            overlap_threshold: Threshold for overlap detection (0.0-1.0)
            vad_threshold: Voice activity detection threshold
            noise_reduction: Whether to apply noise reduction
            max_workers: Maximum number of worker threads
            model_name: Wav2Vec2 model name from HuggingFace
            huggingface_token: HuggingFace authentication token
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_duration_minutes = chunk_duration_minutes
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.overlap_threshold = overlap_threshold
        self.vad_threshold = vad_threshold
        self.noise_reduction = noise_reduction
        self.max_workers = max_workers
        self.model_name = model_name
        self.huggingface_token = huggingface_token

        self.temp_chunks_dir = os.path.join(output_dir, "temp_chunks")
        self.cleaned_chunks_dir = os.path.join(output_dir, "cleaned_chunks")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_chunks_dir, exist_ok=True)
        os.makedirs(self.cleaned_chunks_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self._initialize_models()

        self.logger.info(f"✅ W2VBertOverlapDetector initialized")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"🎵 Chunk duration: {chunk_duration_minutes} minutes")
        self.logger.info(f"🎯 Overlap threshold: {overlap_threshold}")
        self.logger.info(f"🤖 Using model: {self.model_name}")

    def _initialize_models(self):
        """Initialize Wav2Vec2 models with fallback options."""
        
        # Initialize all attributes first to avoid AttributeError
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.transcription_model = None
        self.diarization_pipeline = None
        self.w2vbert_available = False

        # List of working models to try (in order of preference)
        models_to_try = [
            self.model_name,
            "facebook/wav2vec2-base-960h",
            "facebook/wav2vec2-base",
            "microsoft/wav2vec2-base-960h-lv60-self"
        ]

        for model_name in models_to_try:
            try:
                self.logger.info(f"🤖 Attempting to load model: {model_name}")
                
                # Use Wav2Vec2Processor and Wav2Vec2ForCTC (correct classes)
                if self.huggingface_token:
                    self.processor = Wav2Vec2Processor.from_pretrained(
                        model_name,
                        use_auth_token=self.huggingface_token
                    )
                    self.model = Wav2Vec2ForCTC.from_pretrained(
                        model_name,
                        use_auth_token=self.huggingface_token
                    ).to(self.device)
                else:
                    self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                    self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
                
                # Wav2Vec2 doesn't need a separate tokenizer - processor handles it
                self.tokenizer = self.processor.tokenizer
                
                self.w2vbert_available = True
                self.model_name = model_name
                self.logger.info(f"✅ Successfully loaded model: {model_name}")
                break
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        if not self.w2vbert_available:
            self.logger.error("❌ No Wav2Vec2 models could be loaded. Using fallback processing.")

        # Try to load Whisper as additional fallback
        try:
            import whisper
            self.logger.info("🤖 Loading Whisper model as additional fallback...")
            self.transcription_model = whisper.load_model("base")
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load Whisper model: {e}")

        # Try to initialize diarization pipeline (optional)
        try:
            if self.huggingface_token:
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.huggingface_token
                )
                self.logger.info("✅ Pyannote diarization pipeline loaded")
        except Exception as e:
            self.logger.info(f"ℹ️ Diarization pipeline not available: {e}")
            self.diarization_pipeline = None

    def _detect_voice_activity_fallback(self, audio_path: str) -> Dict:
        """
        Fallback voice activity detection using librosa when Wav2Vec2 is not available
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            if duration <= 0:
                return {'voice_percentage': 0.0, 'has_voice': False, 'transcription': '', 'method': 'error'}

            # Basic voice activity detection using energy and spectral features
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop

            # RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            
            # Voice activity analysis
            energy_threshold = np.percentile(rms, 40)
            voice_centroid_frames = (spectral_centroids >= 300) & (spectral_centroids <= 3400)
            voice_centroid_percentage = np.sum(voice_centroid_frames) / len(voice_centroid_frames) * 100
            
            moderate_zcr_frames = zcr < np.percentile(zcr, 60)
            moderate_zcr_percentage = np.sum(moderate_zcr_frames) / len(moderate_zcr_frames) * 100
            
            energy_voice_frames = rms > energy_threshold
            energy_voice_percentage = np.sum(energy_voice_frames) / len(energy_voice_frames) * 100
            
            # Simple heuristic for voice percentage
            voice_percentage = min(energy_voice_percentage, voice_centroid_percentage, moderate_zcr_percentage)
            
            # Basic transcription (placeholder)
            transcription = f"[Fallback processing - {duration:.1f}s audio detected]"
            
            has_voice = voice_percentage > (self.vad_threshold * 100)
            
            self.logger.info(f"🔍 Fallback Voice analysis: energy={energy_voice_percentage:.1f}%, "
                            f"freq_range={voice_centroid_percentage:.1f}%, voice_pct={voice_percentage:.1f}%")
            
            return {
                'voice_percentage': voice_percentage,
                'has_voice': has_voice,
                'transcription': transcription,
                'word_count': len(transcription.split()),
                'char_count': len(transcription),
                'avg_confidence': 0.5,  # Default confidence
                'method': 'librosa_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback VAD failed: {e}")
            return {'voice_percentage': 0.0, 'has_voice': False, 'transcription': '', 'method': 'error'}

    def _detect_voice_activity_w2vbert(self, audio_path: str) -> Dict:
        """
        Detect voice activity using Wav2Vec2 transcription with fallback
        """
        # Check if Wav2Vec2 is available
        if not self.w2vbert_available or self.processor is None or self.model is None:
            self.logger.warning("⚠️ Wav2Vec2 not available, using fallback method")
            return self._detect_voice_activity_fallback(audio_path)
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            if duration <= 0:
                return {'voice_percentage': 0.0, 'has_voice': False, 'transcription': '', 'method': 'error'}

            # Process with Wav2Vec2
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Get predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Clean transcription
            transcription = transcription.lower().strip()
            transcription = re.sub(r'[<pad>|<s>|</s>|<unk>]', '', transcription)  # Clean special tokens
            transcription = re.sub(r'[|||]', '', transcription)
            transcription = re.sub(r'\s+', ' ', transcription).strip()

            # Voice activity analysis based on transcription
            word_count = len(transcription.split()) if transcription else 0
            char_count = len(transcription.replace(' ', '')) if transcription else 0

            # Analyze logits confidence
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            avg_confidence = torch.mean(max_probs).item()

            # Voice activity criteria
            min_words = max(1, duration * 0.5)
            min_chars = max(3, duration * 2)
            min_confidence = 0.3

            # Calculate voice percentage
            word_score = min(100, (word_count / min_words) * 100) if min_words > 0 else 0
            char_score = min(100, (char_count / min_chars) * 100) if min_chars > 0 else 0
            confidence_score = avg_confidence * 100

            voice_percentage = (word_score * 0.4 + char_score * 0.3 + confidence_score * 0.3)

            has_voice = (
                word_count >= min_words and 
                char_count >= min_chars and 
                avg_confidence >= min_confidence and
                voice_percentage > (self.vad_threshold * 100)
            )

            self.logger.info(f"🔍 Wav2Vec2 Voice analysis: words={word_count}, chars={char_count}, "
                            f"confidence={avg_confidence:.3f}, voice_pct={voice_percentage:.1f}%")

            return {
                'voice_percentage': voice_percentage,
                'has_voice': has_voice,
                'transcription': transcription,
                'word_count': word_count,
                'char_count': char_count,
                'avg_confidence': avg_confidence,
                'method': 'wav2vec2_transcription'
            }

        except Exception as e:
            self.logger.error(f"❌ Wav2Vec2 VAD failed: {e}")
            return self._detect_voice_activity_fallback(audio_path)

    def _detect_overlapping_voices_w2vbert(self, audio_path: str) -> Dict:
        """
        Detect overlapping voices using Wav2Vec2 embeddings and transcription analysis
        Returns dict with overlap info and speaker count
        """
        # Fallback if Wav2Vec2 not available
        if not self.w2vbert_available or self.processor is None or self.model is None:
            self.logger.warning("⚠️ Wav2Vec2 not available for overlap detection, using basic method")
            return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 1, 'method': 'fallback'}

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            if duration <= 0:
                return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 0, 'method': 'error'}

            # Split audio into overlapping windows for temporal analysis
            window_size = int(2 * sr)  # 2-second windows
            hop_size = int(1 * sr)     # 1-second hop
            
            window_confidences = []
            window_transcriptions = []
            
            # Analyze each window
            for i in range(0, len(audio) - window_size, hop_size):
                window_audio = audio[i:i + window_size]
                
                try:
                    # Process window with Wav2Vec2
                    inputs = self.processor(window_audio, sampling_rate=sr, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits

                    # Calculate confidence variance (high variance may indicate overlap)
                    probs = torch.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    confidence_variance = torch.var(max_probs).item()
                    avg_confidence = torch.mean(max_probs).item()
                    
                    window_confidences.append({
                        'confidence_variance': confidence_variance,
                        'avg_confidence': avg_confidence,
                        'timestamp': i / sr
                    })
                    
                    # Get transcription for this window
                    predicted_ids = torch.argmax(logits, dim=-1)
                    window_transcription = self.processor.batch_decode(predicted_ids)[0]
                    window_transcription = re.sub(r'[<pad>|<s>|</s>|<unk>]', '', window_transcription)
                    window_transcription = re.sub(r'[|||]', '', window_transcription).strip()
                    window_transcriptions.append(window_transcription)
                    
                except Exception as e:
                    self.logger.debug(f"Window processing error: {e}")
                    continue
            
            if not window_confidences:
                return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 1, 'method': 'wav2vec2_error'}

            # Analyze overlap indicators
            confidences = [w['confidence_variance'] for w in window_confidences]
            avg_confidences = [w['avg_confidence'] for w in window_confidences]
            
            # High variance in confidence may indicate overlapping speakers
            overall_confidence_variance = np.var(confidences) if confidences else 0
            avg_confidence_variance = np.mean(confidences) if confidences else 0
            
            # Transcription-based overlap detection
            full_transcription = ' '.join(window_transcriptions)
            
            # Look for indicators of multiple speakers in transcription
            overlap_indicators = [
                '[inaudible]', '[overlapping]', '[crosstalk]',
                'speaker 1', 'speaker 2', 'person 1', 'person 2'
            ]
            
            transcription_overlap_score = sum(
                full_transcription.lower().count(indicator) for indicator in overlap_indicators
            )
            
            # Calculate overlap percentage based on confidence variance and transcription
            variance_threshold = 0.01  # Threshold for "high" variance
            confidence_overlap_score = min(100, (avg_confidence_variance / variance_threshold) * 100)
            
            # Combine scores
            overlap_percentage = min(100, confidence_overlap_score + (transcription_overlap_score * 20))
            
            # Determine if overlap exists
            has_overlap = overlap_percentage > (self.overlap_threshold * 100)

            # Estimate speaker count (basic heuristic)
            speakers_count = 1
            if has_overlap:
                speakers_count = 2 + min(2, transcription_overlap_score)  # Max 4 speakers estimated

            self.logger.info(f"🔍 Wav2Vec2 Overlap analysis: conf_var={avg_confidence_variance:.4f}, "
                           f"transcription_indicators={transcription_overlap_score}, "
                           f"overlap_pct={overlap_percentage:.1f}%")

            return {
                'has_overlap': has_overlap,
                'overlap_percentage': overlap_percentage,
                'speakers_count': speakers_count,
                'confidence_variance': avg_confidence_variance,
                'transcription_overlap_score': transcription_overlap_score,
                'full_transcription': full_transcription,
                'method': 'wav2vec2_analysis'
            }

        except Exception as e:
            self.logger.error(f"❌ Wav2Vec2 overlap detection failed: {e}")
            return {'has_overlap': False, 'overlap_percentage': 0.0, 'speakers_count': 1, 'method': 'error'}

    def process_extracted_samples(self, extracted_samples: List[Dict]) -> List[Dict]:
        """
        Process extracted samples with Wav2Vec2 based chunking, cleaning, and dual filtering
        """
        if not extracted_samples:
            self.logger.info("🔍 No extracted samples to process")
            return []

        self.logger.info(f"🔍 STAGE 6.5: Wav2Vec2 Audio Processing Pipeline")
        self.logger.info(f"📊 Processing {len(extracted_samples)} extracted samples")
        self.logger.info(f"✂️ Chunk duration: {self.chunk_duration_minutes} minutes")
        self.logger.info(f"🧹 Audio cleaning: {'Enabled' if self.noise_reduction else 'Disabled'}")
        self.logger.info(f"🗣️ VAD threshold: {self.vad_threshold * 100}% (minimum voice activity)")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}% (maximum overlap)")

        stats = {
            "total_files": len(extracted_samples),
            "total_chunks_created": 0,
            "voice_positive_chunks": 0,
            "overlap_rejected_chunks": 0,
            "final_clean_chunks": 0,
        }

        clean_chunks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, sample in enumerate(extracted_samples, 1):
                sample_file = sample.get("sample_file")
                username = sample.get("processed_username", "unknown")
                platform = sample.get("platform_source", "unknown")

                if not sample_file or not os.path.exists(sample_file):
                    self.logger.warning(f"⚠️ [{i}/{len(extracted_samples)}] Sample file not found: {sample_file}")
                    continue

                self.logger.info(f"🎤 [{i}/{len(extracted_samples)}] Processing @{username} ({platform})")

                duration = self._get_audio_duration(sample_file)
                duration_minutes = duration / 60
                self.logger.info(f"📏 File duration: {duration_minutes:.1f} minutes")

                if duration_minutes <= 5:
                    self.logger.info("📄 File ≤5min, processing whole file")
                    stats["total_chunks_created"] += 1
                    future = executor.submit(self._process_single_file, sample_file, username, platform, sample, stats)
                    futures.append(future)
                else:
                    self.logger.info(f"✂️ File >5min, splitting into {self.chunk_duration_minutes}min chunks")
                    chunks = self._split_audio_into_chunks(sample_file, username, platform)
                    stats["total_chunks_created"] += len(chunks)

                    for chunk_idx, chunk_file in enumerate(chunks, 1):
                        self.logger.info(f"🔍 Processing chunk {chunk_idx}/{len(chunks)}")
                        future = executor.submit(
                            self._process_chunk_file, chunk_file, username, platform, chunk_idx, len(chunks), sample, stats
                        )
                        futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        clean_chunks.append(result)
                        stats["final_clean_chunks"] += 1
                except Exception as e:
                    self.logger.error(f"❌ Error processing sample: {e}")

        self._cleanup_temp_files()
        self._log_processing_summary(stats)
        return clean_chunks

    def _process_single_file(
        self, sample_file: str, username: str, platform: str, sample: Dict, stats: Dict
    ) -> Optional[Dict]:
        """Process a single file through the Wav2Vec2 pipeline"""
        try:
            cleaned_file = self._apply_audio_cleaning(sample_file, username, platform, 1) if self.noise_reduction else sample_file
            if not cleaned_file:
                self.logger.warning("⚠️ Audio cleaning failed, skipping file")
                return None

            # Step 1: Wav2Vec2 Transcription-based Voice Activity Detection
            self.logger.info("🗣️ Checking voice activity with Wav2Vec2...")
            voice_result = self._detect_voice_activity_w2vbert(cleaned_file)
            voice_pct = voice_result.get("voice_percentage", 0)

            if not voice_result.get("has_voice", False) or voice_pct < self.vad_threshold * 100:
                self.logger.info(f"❌ File REJECTED - No sufficient voice activity ({voice_pct:.1f}% < {self.vad_threshold * 100}%)")
                self._cleanup_file(cleaned_file, sample_file)
                return None

            stats["voice_positive_chunks"] += 1
            self.logger.info(f"✅ Voice activity detected ({voice_pct:.1f}%)")

            # Step 2: Wav2Vec2 Transcription-based Overlap Detection
            self.logger.info("🔍 Checking for overlapping voices with Wav2Vec2...")
            overlap_result = self._detect_overlapping_voices_w2vbert(cleaned_file)
            overlap_pct = overlap_result.get("overlap_percentage", 0)

            if overlap_result.get("has_overlap", False):
                self.logger.info(f"❌ File REJECTED - Overlapping voices detected ({overlap_pct:.1f}% > {self.overlap_threshold * 100}%)")
                stats["overlap_rejected_chunks"] += 1
                self._cleanup_file(cleaned_file, sample_file)
                return None

            final_file = self._save_clean_chunk(cleaned_file, username, platform, 1)
            if not final_file:
                self.logger.warning("⚠️ Failed to save clean file")
                return None

            chunk_data = sample.copy()
            chunk_data.update({
                "clean_chunk_file": final_file,
                "chunk_number": 1,
                "total_chunks": 1,
                "original_duration": self._get_audio_duration(sample_file),
                "has_overlap": False,
                "overlap_percentage": overlap_pct,
                "voice_percentage": voice_pct,
                "transcription": voice_result.get("transcription", ""),
                "speakers_detected": overlap_result.get("speakers_count", 1),
                "processing_method": "wav2vec2_single_file_pipeline"
            })

            self.logger.info(f"✅ File ACCEPTED - Clean single-speaker voice content")
            self.logger.info(f"📝 Transcription: {voice_result.get('transcription', '')[:100]}...")
            return chunk_data

        except Exception as e:
            self.logger.error(f"❌ Error processing single file: {e}")
            return None

    def _process_chunk_file(
        self,
        chunk_file: str,
        username: str,
        platform: str,
        chunk_idx: int,
        total_chunks: int,
        sample: Dict,
        stats: Dict,
    ) -> Optional[Dict]:
        """Process a chunk file through the Wav2Vec2 pipeline"""
        try:
            cleaned_file = self._apply_audio_cleaning(chunk_file, username, platform, chunk_idx) if self.noise_reduction else chunk_file
            if not cleaned_file:
                self.logger.warning(f"⚠️ Audio cleaning failed for chunk {chunk_idx}, skipping")
                self._cleanup_file(chunk_file)
                return None

            self.logger.info(f"🗣️ Checking voice activity in chunk {chunk_idx} with Wav2Vec2...")
            voice_result = self._detect_voice_activity_w2vbert(cleaned_file)
            voice_pct = voice_result.get("voice_percentage", 0)

            if not voice_result.get("has_voice", False) or voice_pct < self.vad_threshold * 100:
                self.logger.info(f"❌ Chunk {chunk_idx} REJECTED - No sufficient voice activity ({voice_pct:.1f}% < {self.vad_threshold * 100}%)")
                self._cleanup_file(cleaned_file, chunk_file)
                return None

            stats["voice_positive_chunks"] += 1
            self.logger.info(f"✅ Chunk {chunk_idx} has voice activity ({voice_pct:.1f}%)")

            self.logger.info(f"🔍 Checking for overlapping voices in chunk {chunk_idx} with Wav2Vec2...")
            overlap_result = self._detect_overlapping_voices_w2vbert(cleaned_file)
            overlap_pct = overlap_result.get("overlap_percentage", 0)

            if overlap_result.get("has_overlap", False):
                self.logger.info(f"❌ Chunk {chunk_idx} REJECTED - Overlapping voices detected ({overlap_pct:.1f}% > {self.overlap_threshold * 100}%)")
                stats["overlap_rejected_chunks"] += 1
                self._cleanup_file(cleaned_file, chunk_file)
                return None

            final_file = self._save_clean_chunk(cleaned_file, username, platform, chunk_idx)
            if not final_file:
                self.logger.warning(f"⚠️ Failed to save clean chunk {chunk_idx}")
                return None

            chunk_data = sample.copy()
            chunk_data.update({
                "clean_chunk_file": final_file,
                "chunk_number": chunk_idx,
                "total_chunks": total_chunks,
                "original_duration": self._get_audio_duration(chunk_file),
                "chunk_duration": self.chunk_duration_seconds,
                "has_overlap": False,
                "overlap_percentage": overlap_pct,
                "voice_percentage": voice_pct,
                "transcription": voice_result.get("transcription", ""),
                "speakers_detected": overlap_result.get("speakers_count", 1),
                "processing_method": "wav2vec2_chunk_pipeline"
            })

            self.logger.info(f"✅ Chunk {chunk_idx} ACCEPTED - Clean single-speaker voice content")
            self.logger.info(f"📝 Transcription: {voice_result.get('transcription', '')[:50]}...")
            return chunk_data

        except Exception as e:
            self.logger.error(f"❌ Error processing chunk {chunk_idx}: {e}")
            return None

    # Keep all the existing utility methods unchanged
    def _apply_audio_cleaning(self, audio_path: str, username: str, platform: str, chunk_idx: int) -> Optional[str]:
        """Apply light audio cleaning to improve voice clarity"""
        try:
            timestamp = int(time.time())
            cleaned_filename = f"{username}_{platform}_{timestamp}_chunk{chunk_idx:02d}_cleaned.wav"
            cleaned_path = os.path.join(self.cleaned_chunks_dir, cleaned_filename)
            self.logger.info(f"🧹 Applying audio cleaning to chunk {chunk_idx}...")

            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80,silenceremove=start_periods=1:start_silence=0.1:start_threshold=0.02:detection=peak",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", cleaned_path,
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0 and os.path.exists(cleaned_path):
                cleaned_duration = self._get_audio_duration(cleaned_path)
                if cleaned_duration > 1.0:
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

            cmd = ["ffmpeg", "-y", "-i", source_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", final_path]

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
            for temp_dir in [self.temp_chunks_dir, self.cleaned_chunks_dir]:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    try:
                        os.rmdir(temp_dir)
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"Cleanup warning: {e}")

    def _log_processing_summary(self, stats: Dict):
        """Log comprehensive processing summary"""
        self.logger.info(f"\n🎯 WAV2VEC2 AUDIO PROCESSING SUMMARY:")
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
        self.logger.info(f"🤖 Model: {self.model_name}")
        self.logger.info(f"🧹 Audio cleaning: {'Applied' if self.noise_reduction else 'Skipped'}")
        self.logger.info(f"🗣️ VAD threshold: {self.vad_threshold * 100}%")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}%")

    def process_audio_directory(self, voice_samples_dir: str) -> List[Dict]:
        """Process audio directory containing voice samples"""
        if not os.path.exists(voice_samples_dir):
            self.logger.error(f"❌ Voice samples directory not found: {voice_samples_dir}")
            return []

        self.logger.info(f"🎵 Processing voice samples directory: {voice_samples_dir}")

        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
            audio_files.extend(Path(voice_samples_dir).glob(ext))
            audio_files.extend(Path(voice_samples_dir).glob(ext.upper()))

        if not audio_files:
            self.logger.warning(f"⚠️ No audio files found in {voice_samples_dir}")
            return []

        self.logger.info(f"📁 Found {len(audio_files)} audio files to process")

        extracted_samples = []
        for audio_file in audio_files:
            filename = audio_file.stem
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

        return self.process_extracted_samples(extracted_samples)

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe with fallback handling"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                duration_str = result.stdout.strip()
                
                # Handle 'N/A' response from ffprobe
                if duration_str.lower() == 'n/a' or not duration_str:
                    self.logger.warning(f"⚠️ ffprobe returned N/A for {audio_path}, trying fallback")
                    # Try fallback with librosa
                    try:
                        audio, sr = librosa.load(audio_path, sr=None)
                        duration = len(audio) / sr if sr > 0 else 0.0
                        return duration
                    except:
                        return 0.0
                
                duration = float(duration_str)
                return duration
            else:
                self.logger.warning(f"⚠️ Could not get duration for {audio_path}")
                return 0.0
                
        except ValueError as e:
            self.logger.error(f"❌ Error parsing duration for {audio_path}: {e}")
            # Try fallback method
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                duration = len(audio) / sr if sr > 0 else 0.0
                return duration
            except:
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

                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ss', str(start_time), '-t', str(self.chunk_duration_seconds),
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', chunk_path
                ]

                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode == 0 and os.path.exists(chunk_path):
                    chunk_duration = self._get_audio_duration(chunk_path)
                    if chunk_duration > 1.0:
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


# Example usage
if __name__ == "__main__":
    # Initialize the Wav2Vec2 overlap detector
    detector = W2VBertOverlapDetector(  # Class name kept for compatibility
        output_dir="wav2vec2_clean_chunks",
        chunk_duration_minutes=6,
        overlap_threshold=0.3,
        vad_threshold=0.05,
        noise_reduction=True,
        max_workers=2,  # Reduce for memory management
        model_name="facebook/wav2vec2-base-960h",  # Fixed: Use working model
        huggingface_token=None  # Add your HuggingFace token here if needed
    )
    
    # Process audio directory
    clean_chunks = detector.process_audio_directory("voice_samples")
    
    print(f"✅ Processing complete! Found {len(clean_chunks)} clean chunks.")
    for chunk in clean_chunks:
        print(f"📁 {chunk['clean_chunk_file']}")
        print(f"📝 Transcription: {chunk.get('transcription', '')[:100]}...")
        print("---")
