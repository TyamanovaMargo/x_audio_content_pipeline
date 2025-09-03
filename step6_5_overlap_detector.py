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

class OverlapDetector:
    def __init__(self, output_dir="clean_chunks", chunk_duration_minutes=5, 
                 overlap_threshold=0.3, huggingface_token=None):
        """
        Overlapping voices detector with audio chunking
        
        Args:
            output_dir: Directory for clean chunks
            chunk_duration_minutes: Chunk duration in minutes (default 5)
            overlap_threshold: Overlap threshold (0.3 = 30% overlap time = remove)
            huggingface_token: HuggingFace token for pyannote
        """
        self.output_dir = output_dir
        self.chunk_duration_minutes = chunk_duration_minutes
        self.chunk_duration_seconds = chunk_duration_minutes * 60
        self.overlap_threshold = overlap_threshold
        self.temp_chunks_dir = os.path.join(output_dir, "temp_chunks")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_chunks_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize pyannote for speaker diarization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if huggingface_token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=huggingface_token
                ).to(self.device)
                self.logger.info("✅ Pyannote speaker diarization model loaded")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load pyannote model: {e}")
                self.diarization_pipeline = None
        else:
            self.logger.warning("⚠️ No HuggingFace token provided, overlap detection will be limited")
            self.diarization_pipeline = None

    def process_extracted_samples(self, extracted_samples: List[Dict]) -> List[Dict]:
        """
        Process extracted samples with chunking and overlap filtering
        
        Rules:
        - Max duration: 1 hour, Min duration: 30 seconds
        - Files > 10 minutes: split into chunks
        - Check each file/chunk for overlapping voices
        - Remove chunks with overlapping voices
        """
        if not extracted_samples:
            self.logger.info("🔍 No extracted samples to process")
            return []

        self.logger.info(f"🔍 STAGE 6.5: Audio Chunking and Overlap Detection")
        self.logger.info(f"📊 Processing {len(extracted_samples)} extracted samples")
        self.logger.info(f"✂️ Chunk duration: {self.chunk_duration_minutes} minutes")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}%")

        clean_chunks = []
        total_chunks_created = 0
        total_chunks_kept = 0

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

            # If file is ≤10 minutes - check entire file for overlaps
            if duration_minutes <= 10:
                self.logger.info("📄 File ≤10min, checking whole file for overlaps")
                
                overlap_result = self._detect_overlapping_voices(sample_file)
                
                if not overlap_result['has_overlap']:
                    # Copy file as single "chunk"
                    clean_file = self._copy_to_clean_chunks(sample_file, username, platform, 0)
                    if clean_file:
                        chunk_data = sample.copy()
                        chunk_data.update({
                            'clean_chunk_file': clean_file,
                            'chunk_number': 1,
                            'total_chunks': 1,
                            'original_duration': duration,
                            'has_overlap': False,
                            'overlap_percentage': overlap_result.get('overlap_percentage', 0),
                            'speakers_detected': overlap_result.get('speakers_count', 0)
                        })
                        clean_chunks.append(chunk_data)
                        total_chunks_kept += 1
                        self.logger.info("✅ File clean - kept as single chunk")
                    
                    total_chunks_created += 1
                else:
                    self.logger.info(f"❌ File has {overlap_result.get('overlap_percentage', 0):.1f}% overlap - rejected")
            
            else:
                # File >10 minutes - split into chunks
                self.logger.info(f"✂️ File >10min, splitting into {self.chunk_duration_minutes}min chunks")
                
                chunks = self._split_audio_into_chunks(sample_file, username, platform)
                total_chunks_created += len(chunks)
                
                # Check each chunk for overlaps
                for chunk_idx, chunk_file in enumerate(chunks, 1):
                    self.logger.info(f"🔍 Checking chunk {chunk_idx}/{len(chunks)}")
                    
                    overlap_result = self._detect_overlapping_voices(chunk_file)
                    
                    if not overlap_result['has_overlap']:
                        # Move clean chunk to final directory
                        clean_file = self._move_to_clean_chunks(chunk_file, username, platform, chunk_idx)
                        if clean_file:
                            chunk_data = sample.copy()
                            chunk_data.update({
                                'clean_chunk_file': clean_file,
                                'chunk_number': chunk_idx,
                                'total_chunks': len(chunks),
                                'original_duration': duration,
                                'chunk_duration': self.chunk_duration_seconds,
                                'has_overlap': False,
                                'overlap_percentage': overlap_result.get('overlap_percentage', 0),
                                'speakers_detected': overlap_result.get('speakers_count', 0)
                            })
                            clean_chunks.append(chunk_data)
                            total_chunks_kept += 1
                            self.logger.info(f"✅ Chunk {chunk_idx} clean - kept")
                    else:
                        self.logger.info(f"❌ Chunk {chunk_idx} has {overlap_result.get('overlap_percentage', 0):.1f}% overlap - rejected")
                        # Remove chunk with overlap
                        try:
                            os.remove(chunk_file)
                        except:
                            pass

        # Cleanup temporary directory
        self._cleanup_temp_chunks()

        # Final statistics
        self.logger.info(f"\n🎯 OVERLAP DETECTION SUMMARY:")
        self.logger.info(f"📊 Original samples: {len(extracted_samples)}")
        self.logger.info(f"✂️ Total chunks created: {total_chunks_created}")
        self.logger.info(f"✅ Clean chunks kept: {total_chunks_kept}")
        self.logger.info(f"❌ Overlapping chunks removed: {total_chunks_created - total_chunks_kept}")
        self.logger.info(f"📈 Clean chunk rate: {(total_chunks_kept / total_chunks_created * 100):.1f}%" if total_chunks_created > 0 else "0%")
        self.logger.info(f"📁 Clean chunks directory: {self.output_dir}")

        return clean_chunks

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get duration for {audio_path}: {e}")
            return 0.0

    def _split_audio_into_chunks(self, audio_path: str, username: str, platform: str) -> List[str]:
        """Split audio into N-minute chunks"""
        try:
            duration = self._get_audio_duration(audio_path)
            num_chunks = int(duration // self.chunk_duration_seconds) + (1 if duration % self.chunk_duration_seconds > 30 else 0)
            
            chunk_files = []
            base_name = f"{username}_{platform}_{int(time.time())}"
            
            self.logger.info(f"✂️ Splitting into {num_chunks} chunks of {self.chunk_duration_minutes}min each")
            
            for i in range(num_chunks):
                start_time = i * self.chunk_duration_seconds
                chunk_file = os.path.join(self.temp_chunks_dir, f"{base_name}_chunk{i+1:02d}.wav")
                
                cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time), '-t', str(self.chunk_duration_seconds),
                    '-i', audio_path, '-c', 'copy', chunk_file
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                
                if result.returncode == 0 and os.path.exists(chunk_file):
                    # Check that chunk is long enough (minimum 30 seconds)
                    chunk_duration = self._get_audio_duration(chunk_file)
                    if chunk_duration >= 30:  # minimum 30 seconds
                        chunk_files.append(chunk_file)
                        self.logger.debug(f"✅ Created chunk {i+1}: {chunk_duration:.1f}s")
                    else:
                        self.logger.debug(f"⚠️ Chunk {i+1} too short ({chunk_duration:.1f}s), skipping")
                        try:
                            os.remove(chunk_file)
                        except:
                            pass
                else:
                    self.logger.warning(f"⚠️ Failed to create chunk {i+1}")
            
            return chunk_files
            
        except Exception as e:
            self.logger.error(f"❌ Error splitting audio: {e}")
            return []

    def _detect_overlapping_voices(self, audio_path: str) -> Dict:
        """
        Detect overlapping voices in audio chunk
        """
        try:
            if not self.diarization_pipeline:
                # Fallback: simple energy-based detection
                return self._fallback_overlap_detection(audio_path)
            
            # Use pyannote speaker diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Analyze speaker time intervals
            speaker_intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_intervals.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            # Find overlaps
            total_duration = self._get_audio_duration(audio_path)
            overlap_duration = 0
            
            for i, interval1 in enumerate(speaker_intervals):
                for interval2 in speaker_intervals[i+1:]:
                    if interval1['speaker'] != interval2['speaker']:
                        # Check interval overlap
                        overlap_start = max(interval1['start'], interval2['start'])
                        overlap_end = min(interval1['end'], interval2['end'])
                        
                        if overlap_start < overlap_end:
                            overlap_duration += overlap_end - overlap_start
            
            overlap_percentage = (overlap_duration / total_duration) * 100 if total_duration > 0 else 0
            has_overlap = overlap_percentage > (self.overlap_threshold * 100)
            
            return {
                'has_overlap': has_overlap,
                'overlap_percentage': overlap_percentage,
                'overlap_duration': overlap_duration,
                'total_duration': total_duration,
                'speakers_count': len(set(interval['speaker'] for interval in speaker_intervals)),
                'method': 'pyannote_diarization'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Pyannote overlap detection failed: {e}, using fallback")
            return self._fallback_overlap_detection(audio_path)

    def _fallback_overlap_detection(self, audio_path: str) -> Dict:
        """Fallback overlap detection based on signal energy"""
        try:
            # Simple detection based on RMS energy analysis
            cmd = [
                'ffmpeg', '-i', audio_path, '-af', 'astats=metadata=1:reset=1:length=0.5',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Analyze output for high energy peaks (possible overlaps)
            high_energy_count = result.stderr.count('RMS level') if result.stderr else 0
            
            # Simple heuristic: if many high-energy segments
            estimated_overlap = min(high_energy_count / 10, 50)  # Rough estimation
            has_overlap = estimated_overlap > (self.overlap_threshold * 100)
            
            return {
                'has_overlap': has_overlap,
                'overlap_percentage': estimated_overlap,
                'speakers_count': 1,  # Unknown without diarization
                'method': 'energy_fallback'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Fallback detection failed: {e}")
            # In case of error, assume file is clean
            return {
                'has_overlap': False,
                'overlap_percentage': 0,
                'speakers_count': 1,
                'method': 'error_assume_clean'
            }

    def _copy_to_clean_chunks(self, source_file: str, username: str, platform: str, chunk_idx: int) -> str:
        """Copy file to clean chunks directory"""
        try:
            timestamp = int(time.time())
            clean_filename = f"{username}_{platform}_{timestamp}_clean.wav"
            clean_path = os.path.join(self.output_dir, clean_filename)
            
            cmd = ['ffmpeg', '-y', '-i', source_file, '-c', 'copy', clean_path]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(clean_path):
                return clean_path
            else:
                self.logger.warning("⚠️ Failed to copy clean file")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error copying clean file: {e}")
            return None

    def _move_to_clean_chunks(self, chunk_file: str, username: str, platform: str, chunk_idx: int) -> str:
        """Move clean chunk to final directory"""
        try:
            timestamp = int(time.time())
            clean_filename = f"{username}_{platform}_{timestamp}_chunk{chunk_idx:02d}_clean.wav"
            clean_path = os.path.join(self.output_dir, clean_filename)
            
            cmd = ['ffmpeg', '-y', '-i', chunk_file, '-c', 'copy', clean_path]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and os.path.exists(clean_path):
                # Remove original temporary file
                try:
                    os.remove(chunk_file)
                except:
                    pass
                return clean_path
            else:
                self.logger.warning(f"⚠️ Failed to move chunk {chunk_idx}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error moving chunk: {e}")
            return None

    def _cleanup_temp_chunks(self):
        """Clean up temporary files"""
        try:
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
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning: {e}")

    def generate_report(self, clean_chunks: List[Dict], output_file: str = None) -> str:
        """Generate clean chunks report"""
        if not output_file:
            output_file = os.path.join(self.output_dir, "clean_chunks_report.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("🔍 CLEAN CHUNKS OVERLAP DETECTION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total clean chunks: {len(clean_chunks)}\n")
                f.write(f"Chunk duration: {self.chunk_duration_minutes} minutes\n")
                f.write(f"Overlap threshold: {self.overlap_threshold * 100}%\n")
                f.write(f"Clean chunks directory: {self.output_dir}\n\n")
                
                if clean_chunks:
                    for i, chunk in enumerate(clean_chunks, 1):
                        username = chunk.get('processed_username', 'unknown')
                        platform = chunk.get('platform_source', 'unknown')
                        chunk_num = chunk.get('chunk_number', 1)
                        total_chunks = chunk.get('total_chunks', 1)
                        overlap_pct = chunk.get('overlap_percentage', 0)
                        speakers = chunk.get('speakers_detected', 0)
                        
                        f.write(f"{i:2d}. {os.path.basename(chunk.get('clean_chunk_file', 'N/A'))}\n")
                        f.write(f"    User: @{username} ({platform})\n")
                        f.write(f"    Chunk: {chunk_num}/{total_chunks}\n")
                        f.write(f"    Overlap: {overlap_pct:.1f}%\n")
                        f.write(f"    Speakers: {speakers}\n\n")
                else:
                    f.write("No clean chunks found.\n")
            
            self.logger.info(f"📄 Report saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            return ""
    def process_audio_directory(self, audio_dir: str) -> List[Dict]:
        """
        Process audio files directly from directory
        
        Args:
            audio_dir: Directory containing audio files from Stage 6
        
        Returns:
            List of clean chunk dictionaries
        """
        if not os.path.exists(audio_dir):
            self.logger.error(f"❌ Audio directory not found: {audio_dir}")
            return []
        
        # Find all audio files in directory
        audio_extensions = ['.mp3', '.wav', '.m4a', '.aac']
        audio_files = []
        
        for file in os.listdir(audio_dir):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                full_path = os.path.join(audio_dir, file)
                if os.path.isfile(full_path):
                    audio_files.append(full_path)
        
        if not audio_files:
            self.logger.info("🔍 No audio files found in directory")
            return []
        
        self.logger.info(f"🔍 STAGE 6.5: Audio Chunking and Overlap Detection")
        self.logger.info(f"📊 Processing {len(audio_files)} audio files from directory")
        self.logger.info(f"📁 Source directory: {audio_dir}")
        self.logger.info(f"✂️ Chunk duration: {self.chunk_duration_minutes} minutes")
        self.logger.info(f"🎯 Overlap threshold: {self.overlap_threshold * 100}%")
        
        # Convert audio files to extracted_samples format
        extracted_samples = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            # Extract username and platform from filename if possible
            # Assume format: username_platform_timestamp.mp3
            parts = filename.split('_')
            username = parts[0] if len(parts) > 0 else 'unknown'
            platform = parts[1] if len(parts) > 1 else 'unknown'
            
            sample_data = {
                'sample_file': audio_file,
                'sample_filename': filename,
                'processed_username': username,
                'platform_source': platform,
                'file_size_bytes': os.path.getsize(audio_file)
            }
            extracted_samples.append(sample_data)
        
        # Use existing process_extracted_samples method
        return self.process_extracted_samples(extracted_samples)
