import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile

from config import Config

class SpeakerOneExtractor:
    """
    Stage 9: Speaker Diarization and Speaker 1 Extraction

    Uses pyannote.audio for speaker diarization and extracting segments of only the first speaker.
    Supports batch processing of .wav and .mp3 files.
    """

    def __init__(self, output_dir=None, huggingface_token=None, target_sample_rate=None, min_segment_duration=None):
        # Use config values as default
        self.output_dir = output_dir or getattr(Config, 'SPEAKER_ANALYSIS_DIR', 'speaker_analysis')
        self.huggingface_token = huggingface_token or getattr(Config, 'HUGGINGFACE_TOKEN', None)
        self.target_sample_rate = target_sample_rate or getattr(Config, 'TARGET_SAMPLE_RATE', 16000)
        self.min_segment_duration = min_segment_duration or getattr(Config, 'MIN_SEGMENT_DURATION', 1.0)

        # Create output directories
        self.speaker1_dir = os.path.join(self.output_dir, "speaker1_only")
        self.analysis_dir = os.path.join(self.output_dir, "diarization_results")
        os.makedirs(self.speaker1_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        print("🎤 Speaker 1 Extractor initialized")
        print(f"📁 Speaker 1 output: {self.speaker1_dir}")
        print(f"📊 Analysis results: {self.analysis_dir}")

        # Initialize pyannote pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize pyannote.audio pipeline for diarization"""
        try:
            from pyannote.audio import Pipeline
            
            if self.huggingface_token:
                print("🔐 Loading model with HuggingFace token...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.huggingface_token
                )
            else:
                print("🔓 Trying to load model without token...")
                try:
                    self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                except Exception:
                    print("❌ HuggingFace token required to access the model")
                    print("💡 Add HUGGINGFACE_TOKEN to config.py")
                    raise

            print("✅ pyannote.audio pipeline successfully loaded")

        except ImportError:
            print("❌ pyannote.audio not installed")
            print("💡 Install: pip install pyannote.audio")
            raise
        except Exception as e:
            print(f"❌ Error initializing diarization pipeline: {e}")
            raise

    def process_audio_directory(self, input_dir: str) -> List[Dict]:
        """Process all audio files in a directory"""
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            return []

        # Find audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.m4a', '.flac']:
            audio_files.extend(Path(input_dir).glob(f"*{ext}"))

        if not audio_files:
            print(f"❌ No audio files found in: {input_dir}")
            return []

        print(f"🎵 Found {len(audio_files)} audio files for speaker extraction")
        print("🎯 Processing strategy:")
        print(" 1. Speaker diarization (identify all speakers)")
        print(" 2. Analyze number of speakers")
        print(" 3. Identify Speaker 1 (first speaker)")
        print(" 4. Extract Speaker 1 segments")
        print(" 5. Reconstruct and export audio")

        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎤 [{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            result = self._process_single_audio_file(str(audio_file))
            if result:
                results.append(result)
            
            time.sleep(0.5)  # Brief pause between files

        print(f"\n🎤 SPEAKER EXTRACTION COMPLETE!")
        print(f"📊 Total files processed: {len(audio_files)}")
        print(f"✅ Speaker 1 files created: {len(results)}")

        return results

    def _process_single_audio_file(self, audio_file: str) -> Optional[Dict]:
        """Process a single audio file for speaker extraction"""
        try:
            filename = os.path.basename(audio_file)
            base_name = os.path.splitext(filename)[0]

            print(f" 📊 Step 1: Running speaker diarization...")

            # Perform speaker diarization
            diarization_result = self._perform_diarization(audio_file)
            if not diarization_result:
                print(f" ❌ Diarization failed")
                return None

            speakers = diarization_result['speakers']
            segments = diarization_result['segments']
            num_speakers = len(speakers)

            print(f" ✅ Found {num_speakers} speaker(s): {list(speakers)}")

            # Generate output filename
            timestamp = int(time.time())
            output_filename = f"{base_name}_speaker1_{timestamp}.wav"
            output_path = os.path.join(self.speaker1_dir, output_filename)

            if num_speakers == 1:
                # Single speaker - convert and save entire file
                print(" 👤 Single speaker detected - saving entire audio")
                success = self._save_entire_audio(audio_file, output_path)

                if success:
                    duration = self._get_audio_duration(output_path)
                    result = {
                        'original_file': audio_file,
                        'output_file': output_path,
                        'filename': output_filename,
                        'num_speakers': num_speakers,
                        'speaker1_duration': duration,
                        'extraction_method': 'full_audio_single_speaker',
                        'segments_count': 1,
                        'processing_status': 'success'
                    }
                    print(f" ✅ Single speaker file saved: {output_filename} ({duration:.1f}s)")
                    return result

            else:
                # Multiple speakers - extract Speaker 1 segments
                print(" 👥 Multiple speakers detected - extracting Speaker 1")
                speaker1_segments = self._get_speaker1_segments(segments, speakers)

                if not speaker1_segments:
                    print(" ❌ No segments found for Speaker 1")
                    return None

                print(f" ✂️ Found {len(speaker1_segments)} segments for Speaker 1")

                # Extract and concatenate Speaker 1 segments
                success, total_duration = self._extract_speaker1_audio(
                    audio_file, speaker1_segments, output_path
                )

                if success:
                    result = {
                        'original_file': audio_file,
                        'output_file': output_path,
                        'filename': output_filename,
                        'num_speakers': num_speakers,
                        'speaker1_duration': total_duration,
                        'extraction_method': 'speaker_diarization',
                        'segments_count': len(speaker1_segments),
                        'processing_status': 'success',
                        'all_speakers': list(speakers)
                    }
                    print(f" ✅ Speaker 1 extracted: {output_filename} ({total_duration:.1f}s from {len(speaker1_segments)} segments)")
                    return result
                else:
                    print(" ❌ Failed to extract Speaker 1 audio")
                    return None

        except Exception as e:
            print(f" ❌ Processing error: {str(e)[:100]}")
            return None

    def _perform_diarization(self, audio_file: str) -> Optional[Dict]:
        """Perform speaker diarization on audio file"""
        try:
            # Apply diarization pipeline
            diarization = self.pipeline(audio_file)

            # Extract speaker information
            speakers = set()
            segments = []

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.end - turn.start,
                    'speaker': speaker
                })

            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])

            return {
                'speakers': speakers,
                'segments': segments,
                'total_segments': len(segments)
            }

        except Exception as e:
            print(f" ⚠️ Diarization error: {str(e)[:100]}")
            return None

    def _get_speaker1_segments(self, segments: List[Dict], speakers: set) -> List[Dict]:
        """Identify Speaker 1 as the earliest speaker and get their segments"""
        if not segments:
            return []

        # Find first speaker (earliest start time)
        first_segment = min(segments, key=lambda x: x['start'])
        speaker1_label = first_segment['speaker']

        # Get all segments for Speaker 1
        speaker1_segments = [
            seg for seg in segments
            if seg['speaker'] == speaker1_label and seg['duration'] >= self.min_segment_duration
        ]

        print(f" 🎯 Speaker 1 identified as: {speaker1_label}")
        return speaker1_segments

    def _extract_speaker1_audio(self, input_file: str, segments: List[Dict], output_path: str) -> Tuple[bool, float]:
        """Extract and concatenate Speaker 1 segments using ffmpeg"""
        try:
            # Create temporary files for segments
            temp_dir = tempfile.mkdtemp()
            segment_files = []
            total_duration = 0

            for i, segment in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.wav")

                # Extract segment using ffmpeg
                cmd = [
                    'ffmpeg', '-i', input_file,
                    '-ss', str(segment['start']),
                    '-t', str(segment['duration']),
                    '-ar', str(self.target_sample_rate),
                    '-ac', '1',  # Mono
                    '-c:a', 'pcm_s16le',
                    '-y', segment_file
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and os.path.exists(segment_file):
                    segment_files.append(segment_file)
                    total_duration += segment['duration']

            if not segment_files:
                return False, 0

            # Create concat file for ffmpeg
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")

            # Concatenate segments
            concat_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-ar', str(self.target_sample_rate),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-y', output_path
            ]

            result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=60)

            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir)

            if result.returncode == 0 and os.path.exists(output_path):
                return True, total_duration
            else:
                print(f" ⚠️ Concatenation failed: {result.stderr[:100]}")
                return False, 0

        except Exception as e:
            print(f" ⚠️ Extraction error: {str(e)[:100]}")
            return False, 0

    def _save_entire_audio(self, input_file: str, output_path: str) -> bool:
        """Save entire audio file (single speaker case)"""
        try:
            cmd = [
                'ffmpeg', '-i', input_file,
                '-ar', str(self.target_sample_rate),
                '-ac', '1',  # Mono
                '-c:a', 'pcm_s16le',
                '-y', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0 and os.path.exists(output_path)

        except Exception:
            return False

    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio duration in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0

    def save_results(self, results: List[Dict]) -> str:
        """Save processing results to CSV"""
        if not results:
            print("❌ No results to save")
            return ""

        import pandas as pd
        timestamp = int(time.time())
        results_file = os.path.join(self.analysis_dir, f"speaker1_extraction_results_{timestamp}.csv")

        # Flatten results for CSV
        flattened_results = []
        for result in results:
            flattened_results.append({
                'original_filename': os.path.basename(result.get('original_file', '')),
                'output_filename': result.get('filename', ''),
                'num_speakers': result.get('num_speakers', 0),
                'speaker1_duration_sec': result.get('speaker1_duration', 0),
                'extraction_method': result.get('extraction_method', ''),
                'segments_count': result.get('segments_count', 0),
                'processing_status': result.get('processing_status', ''),
                'output_file_path': result.get('output_file', '')
            })

        df = pd.DataFrame(flattened_results)
        df.to_csv(results_file, index=False)

        print(f"📁 Results saved: {results_file}")
        return results_file

    def generate_report(self, results: List[Dict]) -> str:
        """Generate detailed processing report"""
        report_file = os.path.join(self.analysis_dir, "speaker1_extraction_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎤 SPEAKER 1 EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(results)}\n")
            f.write(f"Output directory: {self.speaker1_dir}\n\n")

            f.write("🎯 PROCESSING STRATEGY:\n")
            f.write("1. Speaker diarization - Identify all speakers in audio\n")
            f.write("2. Speaker count analysis - Single vs multiple speakers\n")
            f.write("3. Speaker 1 identification - First speaker to talk\n")
            f.write("4. Segment extraction - Extract only Speaker 1 segments\n")
            f.write("5. Audio reconstruction - Concatenate segments into final audio\n\n")

            # Statistics
            single_speaker_count = sum(1 for r in results if r.get('num_speakers', 0) == 1)
            multi_speaker_count = len(results) - single_speaker_count
            total_duration = sum(r.get('speaker1_duration', 0) for r in results)

            f.write("📊 PROCESSING STATISTICS:\n")
            f.write(f"Files with single speaker: {single_speaker_count}\n")
            f.write(f"Files with multiple speakers: {multi_speaker_count}\n")
            f.write(f"Total Speaker 1 duration: {total_duration:.1f} seconds\n")
            avg_duration = total_duration / len(results) if results else 0
            f.write(f"Average duration per file: {avg_duration:.1f}s\n\n")

            f.write("✅ PROCESSED FILES:\n")
            f.write("-" * 40 + "\n")

            for i, result in enumerate(results, 1):
                f.write(f"{i:2d}. {result.get('filename', 'N/A')}\n")
                f.write(f"    Original: {os.path.basename(result.get('original_file', ''))}\n")
                f.write(f"    Speakers: {result.get('num_speakers', 0)}\n")
                f.write(f"    Method: {result.get('extraction_method', 'N/A')}\n")
                f.write(f"    Duration: {result.get('speaker1_duration', 0):.1f}s\n")
                f.write(f"    Segments: {result.get('segments_count', 0)}\n\n")

        print(f"📄 Processing report saved: {report_file}")
        return report_file
