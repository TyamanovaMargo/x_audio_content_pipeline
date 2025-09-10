import os
import time
import subprocess
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import whisper
import pandas as pd
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyannoteWhisperProcessor:
    def __init__(
        self,
        output_dir: str,
        chunk_duration_minutes: int = 5,
        vad_threshold: float = 0.3,
        cleanup: bool = True,
        max_workers: int = 1,
        model_size: str = "base",
        huggingface_token: Optional[str] = None
    ):
        """
        Enhanced Audio Processor using Pyannote for VAD/Diarization and Whisper for Transcription
        
        Args:
            output_dir: Directory for output files
            chunk_duration_minutes: Duration of audio chunks in minutes (default: 5)
            vad_threshold: Voice activity detection threshold (default: 0.3 - stricter)
            cleanup: Whether to clean up temporary files (default: True)
            max_workers: Maximum worker threads (default: 1 for Whisper compatibility)
            model_size: Whisper model size (tiny, base, small, medium, large)
            huggingface_token: HuggingFace token for Pyannote models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.chunks_dir = self.output_dir / "chunks"
        self.cleaned_dir = self.output_dir / "cleaned"
        self.segments_dir = self.output_dir / "segments"
        
        # Create directories
        for directory in [self.chunks_dir, self.cleaned_dir, self.segments_dir]:
            directory.mkdir(exist_ok=True)
        
        self.chunk_duration_minutes = chunk_duration_minutes
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.vad_threshold = vad_threshold
        self.cleanup = cleanup
        self.max_workers = max_workers
        self.model_size = model_size
        
        logger.info(f"✅ Enhanced Audio Processor initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⏱️ Chunk duration: {chunk_duration_minutes} minutes")
        logger.info(f"🎯 VAD threshold: {vad_threshold} (stricter)")
        logger.info(f"🧹 Cleanup enabled: {cleanup}")
        
        # Initialize Pyannote pipelines
        self._initialize_pyannote(huggingface_token)
        
        # Initialize Whisper
        self._initialize_whisper()
        
        logger.info(f"🚀 Ready to process audio files")

    def _initialize_pyannote(self, token: Optional[str]):
        """Initialize Pyannote pipelines for VAD and diarization."""
        try:
            logger.info("🤖 Loading Pyannote models...")
            
            # Speaker diarization pipeline (includes VAD)
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            
            # Voice Activity Detection pipeline
            self.vad_pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=token
            )
            
            # Overlap detection pipeline
            self.overlap_pipeline = Pipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=token
            )
            
            self.pyannote_available = True
            logger.info("✅ Pyannote models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Pyannote models: {e}")
            logger.warning("💡 Make sure you have access to Pyannote models and provide a valid HuggingFace token")
            self.pyannote_available = False

    def _initialize_whisper(self):
        """Initialize Whisper model for transcription only."""
        try:
            logger.info(f"🤖 Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            self.whisper_available = True
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            self.whisper_available = False

    def split_audio_into_chunks(self, audio_path: str) -> List[str]:
        """Split audio file into 5-minute chunks."""
        logger.info(f"✂️ Splitting audio into {self.chunk_duration_minutes}-minute chunks")
        
        duration = self._get_audio_duration(audio_path)
        if duration <= 0:
            logger.warning(f"⚠️ Invalid duration for {audio_path}")
            return []

        num_chunks = int(np.ceil(duration / self.chunk_duration_seconds))
        logger.info(f"📊 Creating {num_chunks} chunks from {duration:.1f}s audio")

        chunks = []
        timestamp = int(time.time())
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration_seconds
            chunk_filename = f"chunk_{timestamp}_{i+1:03d}.wav"
            chunk_path = self.chunks_dir / chunk_filename
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time), '-t', str(self.chunk_duration_seconds),
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(chunk_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0 and chunk_path.exists():
                chunk_duration = self._get_audio_duration(str(chunk_path))
                if chunk_duration > 1.0:
                    chunks.append(str(chunk_path))
                    logger.info(f"✅ Created chunk {i+1}/{num_chunks}: {chunk_duration:.1f}s")
                else:
                    logger.warning(f"⚠️ Chunk {i+1} too short ({chunk_duration:.1f}s), skipping")
                    chunk_path.unlink(missing_ok=True)
            else:
                logger.warning(f"⚠️ Failed to create chunk {i+1}/{num_chunks}")
        
        return chunks

    def apply_enhanced_cleaning(self, audio_path: str) -> Optional[str]:
        """Apply enhanced audio cleaning with noise reduction."""
        timestamp = int(time.time())
        cleaned_filename = f"{Path(audio_path).stem}_cleaned_{timestamp}.wav"
        cleaned_path = self.cleaned_dir / cleaned_filename
        
        logger.info(f"🧹 Applying enhanced audio cleaning...")
        
        # Enhanced cleaning with afftdn noise reduction
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-af", 
            "afftdn=nf=-25," +  # Noise reduction
            "loudnorm=I=-16:TP=-1.5:LRA=11," +  # Loudness normalization
            "highpass=f=80," +  # High-pass filter
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=0.02:detection=peak",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(cleaned_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and cleaned_path.exists():
            cleaned_duration = self._get_audio_duration(str(cleaned_path))
            if cleaned_duration > 1.0:
                logger.info(f"✅ Audio cleaning completed: {cleaned_duration:.1f}s")
                return str(cleaned_path)
            else:
                logger.warning(f"⚠️ Cleaned audio too short ({cleaned_duration:.1f}s)")
                cleaned_path.unlink(missing_ok=True)
                return None
        else:
            logger.warning(f"⚠️ Audio cleaning failed")
            return None

    def process_with_pyannote(self, audio_path: str) -> List[Dict]:
        """Process audio with Pyannote for VAD, diarization, and overlap detection."""
        if not self.pyannote_available:
            logger.error("❌ Pyannote models not available")
            return []
        
        logger.info(f"🎤 Processing with Pyannote: {audio_path}")
        
        # Run diarization (includes VAD)
        diarization = self.diarization_pipeline(audio_path)
        
        # Run overlap detection
        overlap_detection = self.overlap_pipeline(audio_path)
        
        # Filter segments
        clean_segments = []
        segment_id = 1
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Check duration and voice activity
            if turn.duration < 1.0:  # Skip segments shorter than 1 second
                continue
                
            # Check for overlaps in this segment
            overlap_timeline = overlap_detection.get_timeline().crop(turn)
            overlap_ratio = overlap_timeline.duration() / turn.duration if turn.duration > 0 else 1.0
            
            # Apply stricter VAD threshold and overlap filtering
            voice_ratio = diarization.get_timeline().crop(turn).duration() / turn.duration if turn.duration > 0 else 0.0
            
            if voice_ratio >= self.vad_threshold and overlap_ratio < 0.1:  # Less than 10% overlap
                clean_segments.append({
                    'segment_id': segment_id,
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.duration,
                    'speaker': speaker,
                    'voice_ratio': voice_ratio,
                    'overlap_ratio': overlap_ratio
                })
                segment_id += 1
        
        logger.info(f"🔍 Found {len(clean_segments)} clean segments after Pyannote filtering")
        logger.info(f"📊 VAD threshold: {self.vad_threshold}, Max overlap: 10%")
        
        return clean_segments

    def extract_and_save_segment(self, audio_path: str, segment: Dict) -> Optional[str]:
        """Extract and save individual audio segment."""
        try:
            # Load audio segment
            y, sr = librosa.load(
                audio_path, 
                sr=16000, 
                offset=segment['start'], 
                duration=segment['duration']
            )
            
            # Create filename
            segment_filename = (
                f"segment_{segment['segment_id']:04d}_"
                f"{segment['speaker']}_"
                f"{segment['start']:.2f}_{segment['end']:.2f}.wav"
            )
            segment_path = self.segments_dir / segment_filename
            
            # Save segment
            sf.write(str(segment_path), y, sr)
            
            logger.info(f"💾 Saved segment {segment['segment_id']}: {segment['duration']:.1f}s")
            return str(segment_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving segment {segment['segment_id']}: {e}")
            return None

    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe audio segment using Whisper."""
        if not self.whisper_available:
            return "[Whisper not available]"
        
        try:
            # Use fp16=False to avoid CPU warning
            result = self.whisper_model.transcribe(audio_path, fp16=False)
            transcription = result["text"].strip()
            logger.info(f"📝 Transcribed: {len(transcription)} characters")
            return transcription
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return "[Transcription failed]"

    def process_audio_file(self, input_path: str) -> List[Dict]:
        """Process a complete audio file through the enhanced pipeline."""
        logger.info(f"🎵 Starting enhanced processing: {input_path}")
        
        # Step 1: Split into chunks
        chunks = self.split_audio_into_chunks(input_path)
        if not chunks:
            logger.error("❌ No chunks created")
            return []
        
        all_results = []
        
        # Step 2: Process each chunk
        for chunk_idx, chunk_path in enumerate(chunks, 1):
            logger.info(f"🔄 Processing chunk {chunk_idx}/{len(chunks)}")
            
            # Step 3: Apply enhanced cleaning
            cleaned_path = self.apply_enhanced_cleaning(chunk_path)
            if not cleaned_path:
                logger.warning(f"⚠️ Skipping chunk {chunk_idx} - cleaning failed")
                continue
            
            # Step 4: Pyannote processing
            segments = self.process_with_pyannote(cleaned_path)
            if not segments:
                logger.warning(f"⚠️ No clean segments in chunk {chunk_idx}")
                continue
            
            # Step 5: Extract segments and transcribe
            for segment in segments:
                segment_path = self.extract_and_save_segment(cleaned_path, segment)
                if segment_path:
                    transcription = self.transcribe_with_whisper(segment_path)
                    
                    result = {
                        'chunk_id': chunk_idx,
                        'segment_id': segment['segment_id'],
                        'start_time': float(segment['start']),
                        'end_time': float(segment['end']),
                        'duration': float(segment['duration']),
                        'speaker_label': segment['speaker'],
                        'voice_ratio': float(segment['voice_ratio']),
                        'overlap_ratio': float(segment['overlap_ratio']),
                        'transcription': transcription,
                        'audio_file': segment_path,
                        'processing_method': 'pyannote_whisper'
                    }
                    
                    all_results.append(result)
                    
                    logger.info(f"✅ Completed segment {segment['segment_id']}: "
                              f"{segment['duration']:.1f}s, Speaker: {segment['speaker']}")
        
        # Step 6: Cleanup if requested
        if self.cleanup:
            self._cleanup_temp_files()
        
        # Step 7: Save results
        self._save_results(all_results)
        
        logger.info(f"🎉 Processing complete! Generated {len(all_results)} transcribed segments")
        return all_results

    def _save_results(self, results: List[Dict]):
        """Save results to CSV file."""
        if results:
            df = pd.DataFrame(results)
            csv_path = self.output_dir / "transcription_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"📊 Results saved to: {csv_path}")
            
            # Save summary
            summary = {
                'total_segments': len(results),
                'total_duration': sum(r['duration'] for r in results),
                'unique_speakers': len(set(r['speaker_label'] for r in results)),
                'avg_voice_ratio': np.mean([r['voice_ratio'] for r in results]),
                'avg_overlap_ratio': np.mean([r['overlap_ratio'] for r in results])
            }
            
            summary_path = self.output_dir / "processing_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("🎤 ENHANCED AUDIO PROCESSING SUMMARY\n")
                f.write("=" * 40 + "\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"📋 Summary saved to: {summary_path}")

    def _cleanup_temp_files(self):
        """Clean up temporary chunk and cleaned directories."""
        logger.info("🧹 Cleaning up temporary files...")
        try:
            shutil.rmtree(self.chunks_dir, ignore_errors=True)
            shutil.rmtree(self.cleaned_dir, ignore_errors=True)
            logger.info("✅ Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe with fallback."""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                   '-of', 'csv=p=0', audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                duration_str = result.stdout.strip()
                if duration_str.lower() != 'n/a' and duration_str:
                    return float(duration_str)
            
            # Fallback to librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr if sr > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Duration detection error: {e}")
            return 0.0


# Convenience function for easy usage
def process_long_audio(
    input_path: str,
    output_dir: str,
    cleanup: bool = True,
    vad_threshold: float = 0.3,
    huggingface_token: Optional[str] = None
) -> List[Dict]:
    """
    Process long audio file with enhanced Pyannote + Whisper pipeline.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory for output files
        cleanup: Whether to clean up temporary files
        vad_threshold: Voice activity detection threshold (0.3 = stricter)
        huggingface_token: HuggingFace token for Pyannote models
    
    Returns:
        List of dictionaries containing transcription results
    """
    processor = PyannoteWhisperProcessor(
        output_dir=output_dir,
        cleanup=cleanup,
        vad_threshold=vad_threshold,
        huggingface_token=huggingface_token
    )
    
    return processor.process_audio_file(input_path)
