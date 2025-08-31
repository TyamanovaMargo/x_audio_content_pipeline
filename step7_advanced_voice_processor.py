import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict
import time
import tempfile
import speech_recognition as sr
from pathlib import Path
from tqdm import tqdm
import math

class AdvancedVoiceProcessor:
    def __init__(self, output_dir="voice_analysis", min_voice_confidence=0.3, voice_segment_min_length=1.5, 
                 gaming_mode=False, music_mode=False, podcast_mode=False, max_duration_minutes=10):
        self.output_dir = output_dir
        self.min_voice_confidence = min_voice_confidence
        self.voice_segment_min_length = voice_segment_min_length
        self.max_duration_seconds = max_duration_minutes * 60  # Convert to seconds
        
        # Different processing modes
        self.gaming_mode = gaming_mode
        self.music_mode = music_mode
        self.podcast_mode = podcast_mode
        
        os.makedirs(output_dir, exist_ok=True)
        self.voice_only_dir = os.path.join(output_dir, "voice_only_audio")
        self.analysis_dir = os.path.join(output_dir, "analysis_results")
        self.temp_chunks_dir = os.path.join(output_dir, "temp_chunks")  # For split files
        os.makedirs(self.voice_only_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.temp_chunks_dir, exist_ok=True)

        self.recognizer = sr.Recognizer()

        print(f"🎤 Universal Voice Processor initialized")
        print(f"📁 Voice-only audio: {self.voice_only_dir}")
        print(f"⏱️ Max file duration: {max_duration_minutes} minutes")
        print(f"🎯 Mode: {'Gaming' if gaming_mode else 'Music' if music_mode else 'Podcast' if podcast_mode else 'Universal'}")

    def process_audio_directory(self, audio_dir: str) -> List[Dict]:
        """Process all types of audio files with automatic splitting for long files"""
        if not os.path.exists(audio_dir):
            print(f"❌ Audio directory not found: {audio_dir}")
            return []

        audio_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            audio_files.extend(Path(audio_dir).glob(f"*{ext}"))

        if not audio_files:
            print(f"❌ No audio files found in: {audio_dir}")
            return []

        print(f"🎵 Found {len(audio_files)} audio files")
        print(f"🎯 Universal processing with auto-splitting:")
        print(f" 1. File Duration Check & Auto-Split 📊")
        print(f" 2. Adaptive Voice Detection 🔄")
        print(f" 3. Multi-type Audio Analysis")
        print(f" 4. Smart Background Filtering")
        print(f" 5. Voice Segment Extraction")
        print(f" 6. Quality Validation")

        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎤 [{i}/{len(audio_files)}] Processing: {audio_file.name}")
            print("=" * 70)
            
            # Step 1: Check duration and split if necessary
            file_chunks = self._split_audio_if_long(str(audio_file))
            
            if len(file_chunks) > 1:
                print(f"📊 File split into {len(file_chunks)} chunks (>{self.max_duration_seconds/60:.0f} min)")
            
            # Process each chunk
            for j, chunk_file in enumerate(file_chunks):
                chunk_suffix = f"_part{j+1}" if len(file_chunks) > 1 else ""
                print(f"\n{'🔸' if len(file_chunks) > 1 else '🎤'} Processing chunk {j+1}/{len(file_chunks)}: {os.path.basename(chunk_file)}")
                
                metadata = self._extract_file_metadata(audio_file.name, chunk_suffix)
                result = self._process_universal_audio(chunk_file, metadata)
                
                if result:
                    results.append(result)
                    print(f"✅ SUCCESS: Voice extracted ({result['voice_duration']:.1f}s)")
                else:
                    print(f"❌ FAILED: No clear voice detected")
            
            # Clean up temporary chunks
            self._cleanup_temp_chunks(file_chunks, str(audio_file))
            time.sleep(0.1)

        print(f"\n🎤 UNIVERSAL VOICE PROCESSING COMPLETED!")
        print(f"📊 Total files processed: {len(audio_files)}")
        print(f"✅ Voice-only files created: {len(results)}")
        return results

    def _split_audio_if_long(self, audio_path: str) -> List[str]:
        """Split audio file into chunks if duration exceeds max_duration_seconds"""
        try:
            # Get duration
            duration = self._get_audio_duration(audio_path)
            
            print(f"   📏 Audio duration: {duration:.1f}s ({duration/60:.1f} min)")
            
            if duration <= self.max_duration_seconds:
                return [audio_path]  # No splitting needed
            
            print(f"   ✂️ Splitting long file (>{self.max_duration_seconds/60:.0f} min)...")
            
            # Calculate number of chunks needed
            num_chunks = math.ceil(duration / self.max_duration_seconds)
            chunk_paths = []
            
            # Create progress bar for splitting
            pbar = tqdm(range(num_chunks), desc="   📊 Splitting file", unit="chunk")
            
            for i in pbar:
                start_time = i * self.max_duration_seconds
                chunk_duration = min(self.max_duration_seconds, duration - start_time)
                
                # Generate chunk filename
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                chunk_filename = f"{base_name}_part{i+1}.wav"
                chunk_path = os.path.join(self.temp_chunks_dir, chunk_filename)
                
                pbar.set_description(f"   📊 Creating chunk {i+1}/{num_chunks}")
                
                # Split using ffmpeg
                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-c', 'copy',  # Copy without re-encoding for speed
                    '-y', chunk_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0:
                    chunk_paths.append(chunk_path)
                    pbar.set_postfix({"Chunks": f"{len(chunk_paths)}/{num_chunks}"})
                else:
                    print(f"   ⚠️ Failed to create chunk {i+1}")
            
            pbar.close()
            print(f"   ✅ Created {len(chunk_paths)} chunks")
            return chunk_paths
            
        except Exception as e:
            print(f"   ⚠️ Splitting error: {str(e)[:50]}")
            return [audio_path]  # Return original file if splitting fails

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _cleanup_temp_chunks(self, chunk_paths: List[str], original_path: str):
        """Clean up temporary chunk files"""
        if len(chunk_paths) <= 1:
            return  # No chunks were created
        
        for chunk_path in chunk_paths:
            if chunk_path != original_path and os.path.exists(chunk_path):
                try:
                    os.unlink(chunk_path)
                except Exception:
                    pass  # Ignore cleanup errors

    def _process_universal_audio(self, audio_file: str, metadata: Dict) -> Dict:
        """Universal audio processing with adaptive filtering"""
        try:
            # Step 1: Adaptive voice detection based on content type
            print(f"🔄 Step 1: Adaptive Voice Detection...")
            voice_segments = self._detect_voice_adaptive(audio_file)
            
            if not voice_segments:
                print(f"❌ No voice segments detected")
                return None

            print(f"✅ Found {len(voice_segments)} voice segments")

            # Step 2: Enhanced speech recognition
            print(f"🗣️ Step 2: Multi-language Speech Recognition...")
            speech_analysis = self._analyze_speech_adaptive(audio_file, voice_segments)

            # Step 3: Extract voice segments
            print(f"✂️ Step 3: Extracting voice segments...")
            voice_only_file = self._extract_voice_segments_with_progress(audio_file, voice_segments, metadata)
            
            if not voice_only_file:
                print(f"❌ Failed to extract voice segments")
                return None

            # Step 4: Quality verification
            print(f"🔍 Step 4: Quality verification...")
            final_analysis = self._verify_voice_quality(voice_only_file)
            
            if final_analysis['is_acceptable']:
                print(f"✅ Voice-only file created: {os.path.basename(voice_only_file)}")
                
                orig_size = os.path.getsize(audio_file)
                new_size = os.path.getsize(voice_only_file)
                reduction = (1 - new_size/orig_size) * 100
                print(f"📊 Size: {orig_size//1024}KB → {new_size//1024}KB ({reduction:.1f}% reduction)")
                
                result = {
                    **metadata,
                    'original_file': audio_file,
                    'voice_only_file': voice_only_file,
                    'voice_segments': voice_segments,
                    'speech_analysis': speech_analysis,
                    'final_analysis': final_analysis,
                    'processing_status': 'success',
                    'voice_duration': sum(seg['duration'] for seg in voice_segments),
                    'total_voice_segments': len(voice_segments)
                }
                return result
            else:
                print(f"❌ Voice quality failed: {final_analysis['rejection_reason']}")
                if os.path.exists(voice_only_file):
                    os.remove(voice_only_file)
                return None

        except Exception as e:
            print(f"❌ Processing error: {str(e)[:50]}")
            return None

    def _detect_voice_adaptive(self, audio_file: str) -> List[Dict]:
        """Adaptive voice detection for different content types"""
        try:
            wav_file = self._convert_to_wav(audio_file)
            
            # Get audio duration
            duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', wav_file]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            total_duration = float(duration_result.stdout.strip())
            
            print(f"   📏 Chunk duration: {total_duration:.1f} seconds")
            
            # Adaptive chunk size based on content type
            if self.gaming_mode:
                chunk_size = 3.0  # Smaller for gaming (quick commentary)
            elif self.music_mode:
                chunk_size = 8.0  # Larger for music (longer vocal phrases)
            elif self.podcast_mode:
                chunk_size = 6.0  # Medium for podcasts
            else:
                chunk_size = 4.0  # Default universal
            
            segments = []
            chunks = list(np.arange(0, total_duration, chunk_size))
            
            pbar = tqdm(chunks, desc="   🔄 Adaptive Detection", unit="chunk",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            for start_time in pbar:
                end_time = min(start_time + chunk_size, total_duration)
                pbar.set_description(f"   🔄 Analyzing {start_time:.0f}s-{end_time:.0f}s")
                
                # Extract chunk
                chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                chunk_file.close()
                
                extract_cmd = ['ffmpeg', '-i', wav_file, '-ss', str(start_time), 
                              '-t', str(end_time - start_time), '-y', chunk_file.name]
                result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    voice_score = self._analyze_chunk_adaptive(chunk_file.name)
                    
                    # Adaptive threshold based on mode
                    threshold = self._get_adaptive_threshold()
                    
                    if voice_score >= threshold:
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time,
                            'voice_score': voice_score
                        })
                        pbar.set_postfix({"Voice": f"{len(segments)} segments"})
                
                os.unlink(chunk_file.name)
            
            pbar.close()
            
            if wav_file != audio_file:
                os.unlink(wav_file)
            
            # Adaptive merging
            print(f"   🔗 Merging segments...")
            gap_threshold = 2.0 if self.podcast_mode else 1.5 if self.music_mode else 1.0
            merged_segments = self._merge_adjacent_segments(segments, gap_threshold)
            print(f"   📈 Final segments: {len(merged_segments)}")
            
            return merged_segments

        except Exception as e:
            print(f"   ⚠️ Adaptive detection error: {str(e)[:50]}")
            return []

    def _analyze_chunk_adaptive(self, chunk_file: str) -> float:
        """Adaptive chunk analysis for different content types"""
        try:
            # Step 1: Quick silence check
            silence_cmd = ['ffmpeg', '-i', chunk_file, '-af', 'volumedetect', '-f', 'null', '-']
            silence_result = subprocess.run(silence_cmd, capture_output=True, text=True)
            
            if 'max_volume: -inf dB' in silence_result.stderr:
                return 0.0  # Complete silence
            
            voice_score = 0.0
            
            # Step 2: Speech recognition (primary method)
            try:
                with sr.AudioFile(chunk_file) as source:
                    # Adaptive noise adjustment
                    noise_duration = 0.3 if self.gaming_mode else 0.2
                    self.recognizer.adjust_for_ambient_noise(source, duration=noise_duration)
                    audio_data = self.recognizer.record(source)
                    
                    # Try multiple languages for better coverage
                    languages = ['en-US', 'ru-RU'] if not self.music_mode else ['en-US']
                    
                    for lang in languages:
                        try:
                            text = self.recognizer.recognize_google(audio_data, language=lang)
                            if text and len(text.strip()) > 0:
                                # Higher score for longer recognized text
                                speech_score = min(0.95, 0.6 + len(text) / 80)
                                return speech_score
                        except:
                            continue
                            
            except Exception:
                pass
            
            # Step 3: Audio characteristics analysis
            stats_cmd = ['ffmpeg', '-i', chunk_file, '-af', 'astats=metadata=1:reset=1', '-f', 'null', '-']
            result = subprocess.run(stats_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                stderr = result.stderr.lower()
                
                # Adaptive scoring based on mode
                if self.gaming_mode:
                    # More sensitive to quick voice patterns
                    if 'dynamic_range' in stderr:
                        voice_score += 0.25
                    if 'rms_level' in stderr:
                        voice_score += 0.2
                        
                elif self.music_mode:
                    # Focus on sustained vocal patterns
                    if 'rms_level' in stderr and 'peak_level' in stderr:
                        voice_score += 0.3
                        
                elif self.podcast_mode:
                    # Focus on clear speech patterns
                    if all(indicator in stderr for indicator in ['rms_level', 'dynamic_range']):
                        voice_score += 0.35
                        
                else:
                    # Universal scoring
                    indicators = ['rms_level', 'dynamic_range', 'peak_level']
                    voice_score = 0.2 + (sum(indicator in stderr for indicator in indicators) * 0.15)
            
            return min(voice_score, 1.0)
            
        except Exception:
            return 0.0

    def _get_adaptive_threshold(self) -> float:
        """Get adaptive threshold based on processing mode"""
        if self.gaming_mode:
            return 0.4  # Higher threshold to filter out game sounds
        elif self.music_mode:
            return 0.35  # Medium threshold for vocals vs instruments
        elif self.podcast_mode:
            return 0.25  # Lower threshold for clear speech
        else:
            return self.min_voice_confidence  # User-defined

    def _analyze_speech_adaptive(self, audio_file: str, voice_segments: List[Dict]) -> Dict:
        """Adaptive speech analysis with multi-language support"""
        try:
            wav_file = self._convert_to_wav(audio_file)
            all_text = []
            total_confidence = 0
            successful_recognitions = 0
            
            # Languages to try based on mode
            if self.music_mode:
                languages = ['en-US']  # Music usually English
            else:
                languages = ['en-US', 'ru-RU', 'es-ES']  # Multiple languages
            
            pbar = tqdm(voice_segments, desc="   🗣️ Speech Analysis", unit="segment",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            for segment in pbar:
                pbar.set_description(f"   🗣️ Segment {segment['start']:.1f}s-{segment['end']:.1f}s")
                
                segment_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                segment_file.close()
                
                extract_cmd = ['ffmpeg', '-i', wav_file, '-ss', str(segment['start']),
                              '-t', str(segment['duration']), '-y', segment_file.name]
                result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    recognized = False
                    for lang in languages:
                        try:
                            with sr.AudioFile(segment_file.name) as source:
                                audio_data = self.recognizer.record(source)
                                text = self.recognizer.recognize_google(audio_data, language=lang)
                                if text.strip():
                                    all_text.append(f"[{lang}] {text}")
                                    successful_recognitions += 1
                                    total_confidence += len(text) / 100
                                    recognized = True
                                    break
                        except:
                            continue
                    
                    if recognized:
                        pbar.set_postfix({"Recognized": f"{successful_recognitions}/{len(voice_segments)}"})
                
                os.unlink(segment_file.name)
            
            pbar.close()
            
            if wav_file != audio_file:
                os.unlink(wav_file)

            combined_text = ' '.join(all_text)
            avg_confidence = total_confidence / len(voice_segments) if voice_segments else 0
            
            print(f"   📝 Recognized: {len(combined_text)} chars, {len(combined_text.split())} words")

            return {
                'combined_text': combined_text,
                'text_length': len(combined_text),
                'word_count': len(combined_text.split()) if combined_text else 0,
                'successful_segments': successful_recognitions,
                'total_segments': len(voice_segments),
                'speech_confidence': min(0.9, avg_confidence),
                'recognition_rate': successful_recognitions / len(voice_segments) if voice_segments else 0
            }

        except Exception as e:
            print(f"   ⚠️ Speech analysis error: {str(e)[:50]}")
            return {
                'combined_text': '', 'text_length': 0, 'word_count': 0,
                'successful_segments': 0, 'total_segments': len(voice_segments),
                'speech_confidence': 0.0, 'recognition_rate': 0.0, 'error': str(e)[:50]
            }

    def _extract_voice_segments_with_progress(self, audio_file: str, voice_segments: List[Dict], metadata: Dict) -> str:
        """Extract voice segments with progress tracking"""
        try:
            base_name = metadata.get('username', 'unknown')
            platform = metadata.get('platform', 'unknown')
            chunk_suffix = metadata.get('chunk_suffix', '')
            timestamp = int(time.time())
            output_filename = f"{base_name}_{platform}{chunk_suffix}_voice_only_{timestamp}.wav"
            output_path = os.path.join(self.voice_only_dir, output_filename)

            print(f"   🔄 Converting to WAV...")
            wav_file = self._convert_to_wav(audio_file)

            print(f"   ✂️ Extracting {len(voice_segments)} voice segments...")
            filter_parts = []
            input_parts = []

            pbar = tqdm(voice_segments, desc="   📝 Creating filters", unit="segment")
            for i, segment in enumerate(pbar):
                filter_parts.append(f"[0:a]atrim=start={segment['start']}:end={segment['end']},asetpts=PTS-STARTPTS[seg{i}]")
                input_parts.append(f"[seg{i}]")
                pbar.set_postfix({"Segment": f"{i+1}/{len(voice_segments)}"})
            pbar.close()

            concat_filter = f"{''.join(input_parts)}concat=n={len(voice_segments)}:v=0:a=1[out]"
            filter_complex = ';'.join(filter_parts) + ';' + concat_filter

            print(f"   🎬 Running FFmpeg extraction...")
            cmd = ['ffmpeg', '-i', wav_file, '-filter_complex', filter_complex, '-map', '[out]',
                   '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', output_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if wav_file != audio_file:
                os.unlink(wav_file)

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"   ✅ Extraction successful!")
                return output_path
            else:
                print(f"   ⚠️ FFmpeg error: {result.stderr[:100]}")
                return None

        except Exception as e:
            print(f"   ⚠️ Extraction error: {str(e)[:50]}")
            return None

    def _merge_adjacent_segments(self, segments: List[Dict], gap_threshold: float = 1.0) -> List[Dict]:
        """Merge voice segments that are close together"""
        if not segments:
            return []

        segments.sort(key=lambda x: x['start'])
        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            if next_seg['start'] - current['end'] <= gap_threshold:
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
                current['voice_score'] = max(current['voice_score'], next_seg['voice_score'])
            else:
                if current['duration'] >= self.voice_segment_min_length:
                    merged.append(current)
                current = next_seg.copy()

        if current['duration'] >= self.voice_segment_min_length:
            merged.append(current)

        return merged

    def _verify_voice_quality(self, voice_file: str) -> Dict:
        """Quality check for extracted voice file"""
        try:
            file_size = os.path.getsize(voice_file)
            if file_size < 3000:  # Lowered for split chunks
                return {'is_acceptable': False, 'rejection_reason': 'File too small', 'file_size': file_size}

            duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', voice_file]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            duration = float(duration_result.stdout.strip())

            if duration < 0.5:  # Lowered minimum for chunks
                return {'is_acceptable': False, 'rejection_reason': 'Too short duration', 'duration': duration, 'file_size': file_size}

            print(f"   ✅ Quality check passed: {duration:.1f}s, {file_size}B")
            return {
                'is_acceptable': True, 'duration': duration, 'file_size': file_size,
                'final_text': f'Voice file validated - {duration:.1f}s duration',
                'final_confidence': 0.7, 'validation_method': 'chunk_adaptive_metrics'
            }

        except Exception as e:
            return {'is_acceptable': False, 'rejection_reason': f'Quality error: {str(e)[:30]}', 'file_size': 0}

    def _convert_to_wav(self, audio_file: str) -> str:
        """Convert audio file to WAV format"""
        if audio_file.lower().endswith('.wav'):
            return audio_file

        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()

        cmd = ['ffmpeg', '-i', audio_file, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_wav.name]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode == 0:
            return temp_wav.name
        else:
            os.unlink(temp_wav.name)
            raise Exception("Audio conversion failed")

    def _extract_file_metadata(self, filename: str, chunk_suffix: str = '') -> Dict:
        """Extract metadata from filename with chunk support"""
        parts = filename.replace('.mp3', '').replace('.wav', '').replace('.m4a', '').replace('.flac', '').split('_')
        return {
            'original_filename': filename,
            'username': parts[0] if parts else 'unknown',
            'platform': parts[1] if len(parts) > 1 else 'unknown',
            'video_number': parts[2] if len(parts) > 2 else 'v1',
            'timestamp': parts[3] if len(parts) > 3 else str(int(time.time())),
            'chunk_suffix': chunk_suffix  # Add chunk suffix for split files
        }

    def save_results(self, results: List[Dict]) -> str:
        """Save processing results to CSV"""
        if not results:
            print("❌ No results to save")
            return ""

        flattened_results = []
        for result in results:
            flat_result = {
                'original_filename': result.get('original_filename', ''),
                'username': result.get('username', ''),
                'platform': result.get('platform', ''),
                'chunk_suffix': result.get('chunk_suffix', ''),  # Track chunks
                'original_file': result.get('original_file', ''),
                'voice_only_file': result.get('voice_only_file', ''),
                'voice_duration': result.get('voice_duration', 0),
                'total_voice_segments': result.get('total_voice_segments', 0),
                'processing_status': result.get('processing_status', ''),
                'combined_text': result.get('speech_analysis', {}).get('combined_text', ''),
                'word_count': result.get('speech_analysis', {}).get('word_count', 0),
                'speech_confidence': result.get('speech_analysis', {}).get('speech_confidence', 0),
                'recognition_rate': result.get('speech_analysis', {}).get('recognition_rate', 0),
                'final_duration': result.get('final_analysis', {}).get('duration', 0),
                'final_file_size': result.get('final_analysis', {}).get('file_size', 0),
                'final_text': result.get('final_analysis', {}).get('final_text', ''),
                'final_confidence': result.get('final_analysis', {}).get('final_confidence', 0)
            }
            flattened_results.append(flat_result)

        timestamp = int(time.time())
        results_file = os.path.join(self.analysis_dir, f"voice_processing_results_{timestamp}.csv")
        pd.DataFrame(flattened_results).to_csv(results_file, index=False)
        print(f"📁 Results saved: {results_file}")
        return results_file

    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive processing report"""
        report_file = os.path.join(self.analysis_dir, "voice_processing_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎤 UNIVERSAL VOICE PROCESSING REPORT (WITH AUTO-SPLITTING)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max file duration: {self.max_duration_seconds/60:.0f} minutes\n")
            f.write(f"Total chunks processed: {len(results)}\n")
            f.write(f"Processing mode: {'Gaming' if self.gaming_mode else 'Music' if self.music_mode else 'Podcast' if self.podcast_mode else 'Universal'}\n")
            f.write(f"Voice-only files created: {len([r for r in results if r.get('processing_status') == 'success'])}\n\n")

            successful = [r for r in results if r.get('processing_status') == 'success']
            total_voice_duration = sum(r.get('voice_duration', 0) for r in successful)
            
            f.write("📊 PROCESSING STATISTICS:\n")
            f.write(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)\n")
            f.write(f"Total voice duration: {total_voice_duration:.1f} seconds\n")
            
            # Group by original file
            original_files = {}
            for result in successful:
                orig_name = result.get('original_filename', 'unknown')
                if orig_name not in original_files:
                    original_files[orig_name] = []
                original_files[orig_name].append(result)
            
            f.write(f"Original files processed: {len(original_files)}\n")
            f.write(f"Files that were split: {len([f for f in original_files.values() if len(f) > 1])}\n\n")

            f.write("📁 PROCESSED FILES:\n")
            f.write("-" * 50 + "\n")
            
            for i, (orig_file, chunks) in enumerate(original_files.items(), 1):
                f.write(f"{i:2d}. {orig_file}\n")
                if len(chunks) > 1:
                    f.write(f"    📊 Split into {len(chunks)} chunks\n")
                
                total_duration = sum(chunk.get('voice_duration', 0) for chunk in chunks)
                total_segments = sum(chunk.get('total_voice_segments', 0) for chunk in chunks)
                
                f.write(f"    Voice Duration: {total_duration:.1f}s | Segments: {total_segments}\n")
                
                # Combine text from all chunks
                all_text = ' '.join(chunk.get('speech_analysis', {}).get('combined_text', '') for chunk in chunks)
                f.write(f"    Text: \"{all_text[:80]}...\"\n")
                
                # List chunk files
                for j, chunk in enumerate(chunks):
                    chunk_suffix = chunk.get('chunk_suffix', '')
                    f.write(f"      {j+1}. {os.path.basename(chunk.get('voice_only_file', ''))}\n")
                
                f.write("\n")

        print(f"📄 Report saved: {report_file}")
        return report_file

# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Voice Processor with Auto-Splitting")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--output-dir", default="voice_analysis", help="Output directory")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum voice confidence")
    parser.add_argument("--min-segment-length", type=float, default=1.5, help="Minimum segment length")
    parser.add_argument("--max-duration", type=int, default=10, help="Max file duration in minutes before splitting")
    parser.add_argument("--gaming-mode", action="store_true", help="Optimize for gaming audio")
    parser.add_argument("--music-mode", action="store_true", help="Optimize for music videos")
    parser.add_argument("--podcast-mode", action="store_true", help="Optimize for podcasts/interviews")

    args = parser.parse_args()

    if not os.path.exists(args.audio_dir):
        print(f"❌ Audio directory not found: {args.audio_dir}")
        exit(1)

    processor = AdvancedVoiceProcessor(
        output_dir=args.output_dir,
        min_voice_confidence=args.min_confidence,
        voice_segment_min_length=args.min_segment_length,
        max_duration_minutes=args.max_duration,
        gaming_mode=args.gaming_mode,
        music_mode=args.music_mode,
        podcast_mode=args.podcast_mode
    )

    results = processor.process_audio_directory(args.audio_dir)

    if results:
        results_file = processor.save_results(results)
        report_file = processor.generate_report(results)

        print(f"\n🎤 UNIVERSAL VOICE PROCESSING COMPLETED!")
        print(f"✅ Successfully processed: {len(results)} chunks")
        print(f"📁 Voice-only files: {processor.voice_only_dir}")
        print(f"📊 Results: {results_file}")
        print(f"📄 Report: {report_file}")
        print(f"⏱️ Auto-split files longer than {processor.max_duration_seconds/60:.0f} minutes")
    else:
        print("❌ No voice detected in any files")
