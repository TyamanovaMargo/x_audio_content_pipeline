import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict
import time
import tempfile
import speech_recognition as sr
from pathlib import Path

class AdvancedVoiceProcessor:
    def __init__(self, output_dir="voice_analysis", min_voice_confidence=0.6, voice_segment_min_length=1.0, default_language='en-US'):
        self.default_language = default_language
        self.output_dir = output_dir
        self.min_voice_confidence = min_voice_confidence
        self.voice_segment_min_length = voice_segment_min_length
        
        os.makedirs(output_dir, exist_ok=True)
        self.voice_only_dir = os.path.join(output_dir, "voice_only_audio")
        self.analysis_dir = os.path.join(output_dir, "analysis_results")
        os.makedirs(self.voice_only_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        self.recognizer = sr.Recognizer()
        
        print(f"🎤 Advanced Voice Processor initialized")
        print(f"📁 Voice-only output: {self.voice_only_dir}")
        print(f"📊 Analysis results: {self.analysis_dir}")
        print(f"⚙️ Settings: min_confidence={min_voice_confidence}, min_length={voice_segment_min_length}s")

    def process_audio_directory(self, audio_dir: str) -> List[Dict]:
        """Process all audio files to extract voice-only segments"""
        if not os.path.exists(audio_dir):
            print(f"❌ Directory not found: {audio_dir}")
            return []

        # Find all audio files
        audio_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            audio_files.extend(Path(audio_dir).glob(f"*{ext}"))

        if not audio_files:
            print(f"❌ No audio files found in: {audio_dir}")
            return []

        print(f"🎵 Found {len(audio_files)} audio files to process")
        print(f"🎯 Strategy: Multi-language voice detection + noise/music removal")

        results = []
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎤 [{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            metadata = self._extract_metadata(audio_file.name)
            language = self._detect_language(metadata.get('username', ''))
            metadata['language'] = language
            
            print(f"  🌍 Detected language: {language}")
            
            result = self._process_single_file(str(audio_file), metadata)
            if result:
                results.append(result)
                print(f"  ✅ Success: Voice extracted")
            else:
                print(f"  ❌ No voice detected")

        print(f"\n🎉 Processing complete!")
        print(f"📊 Files processed: {len(audio_files)}")
        print(f"✅ Voice-only files created: {len(results)}")
        return results

    def _process_single_file(self, file_path: str, metadata: Dict) -> Dict:
        """Process a single audio file"""
        try:
            language = metadata['language']
            
            # Step 1: Detect voice segments
            voice_segments = self._detect_voice_segments(file_path, language)
            if not voice_segments:
                return None

            print(f"    🔊 Found {len(voice_segments)} voice segments")

            # Step 2: Extract voice-only audio
            output_file = self._extract_voice_segments(file_path, voice_segments, metadata)
            if not output_file:
                return None

            # Step 3: Verify quality
            if not self._verify_output_quality(output_file):
                os.remove(output_file)
                return None

            # Step 4: Analyze extracted speech
            speech_info = self._analyze_extracted_speech(output_file, language)

            return {
                **metadata,
                'original_file': file_path,
                'voice_only_file': output_file,
                'voice_segments': voice_segments,
                'speech_info': speech_info,
                'processing_status': 'success',
                'voice_duration': sum(seg['duration'] for seg in voice_segments),
                'total_segments': len(voice_segments)
            }

        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            return None

    def _detect_voice_segments(self, file_path: str, language: str) -> List[Dict]:
        """Detect segments containing human voice"""
        wav_file = self._convert_to_wav(file_path)
        
        try:
            # Get duration
            duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', wav_file]
            result = subprocess.run(duration_cmd, capture_output=True, text=True)
            total_duration = float(result.stdout.strip())
            
            print(f"    📏 Duration: {total_duration:.1f}s")
            
            # Use larger chunks for faster processing
            chunk_size = 6.0  # 6-second chunks for efficiency
            segments = []
            
            for start in np.arange(0, total_duration, chunk_size):
                end = min(start + chunk_size, total_duration)
                
                # Extract chunk
                chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                chunk_file.close()
                
                extract_cmd = ['ffmpeg', '-i', wav_file, '-ss', str(start), '-t', str(end-start), '-y', chunk_file.name]
                subprocess.run(extract_cmd, capture_output=True)
                
                # Analyze chunk for voice
                voice_score = self._analyze_chunk_for_voice(chunk_file.name, language)
                
                if voice_score >= 0.5:  # Voice detected
                    segments.append({
                        'start': start,
                        'end': end,
                        'duration': end - start,
                        'voice_score': voice_score
                    })
                    print(f"      ✅ Voice: {start:.1f}s-{end:.1f}s (score={voice_score:.2f})")
                
                os.unlink(chunk_file.name)
            
            # Merge adjacent segments
            merged = self._merge_adjacent_segments(segments)
            
            # Filter by minimum length and confidence
            filtered = [seg for seg in merged if 
                       seg['duration'] >= self.voice_segment_min_length and 
                       seg['voice_score'] >= self.min_voice_confidence]
            
            return filtered
            
        finally:
            if wav_file != file_path:
                os.unlink(wav_file)

    def _analyze_chunk_for_voice(self, chunk_file: str, language: str) -> float:
        """Analyze chunk for human voice - OPTIMIZED VERSION"""
        try:
            # Fast silence detection first
            silence_cmd = ['ffmpeg', '-i', chunk_file, '-af', 'volumedetect', '-f', 'null', '-']
            result = subprocess.run(silence_cmd, capture_output=True, text=True)
            
            # Skip speech recognition if silence
            if 'max_volume: -inf dB' in result.stderr:
                return 0.0
            
            # Try speech recognition on chunks with audio
            try:
                with sr.AudioFile(chunk_file) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio_data = self.recognizer.record(source)
                    
                    # Multi-language support
                    try:
                        text = self.recognizer.recognize_google(audio_data, language=language)
                    except:
                        # Fallback to English if primary language fails
                        text = self.recognizer.recognize_google(audio_data, language='en-US')
                    
                    if text and len(text.strip()) > 0:
                        # Voice detected - score based on text length
                        score = min(0.9, 0.6 + len(text) / 100)
                        return score
                        
            except sr.UnknownValueError:
                # Audio exists but no speech recognized
                pass
            except Exception:
                pass
            
            # Check if audio has voice-like characteristics
            stats_cmd = ['ffmpeg', '-i', chunk_file, '-af', 'astats=metadata=1', '-f', 'null', '-']
            result = subprocess.run(stats_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                stderr = result.stderr.lower()
                # Look for audio activity indicators
                if any(indicator in stderr for indicator in ['rms_level', 'dynamic_range']):
                    return 0.3  # Some audio but no speech
            
            return 0.0
            
        except Exception:
            return 0.0

    def _merge_adjacent_segments(self, segments: List[Dict], gap_threshold: float = 1.0) -> List[Dict]:
        """Merge adjacent voice segments"""
        if not segments:
            return []
        
        segments.sort(key=lambda x: x['start'])
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            if seg['start'] - current['end'] <= gap_threshold:
                current['end'] = seg['end']
                current['duration'] = current['end'] - current['start']
                current['voice_score'] = max(current['voice_score'], seg['voice_score'])
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        return merged

    def _extract_voice_segments(self, file_path: str, segments: List[Dict], metadata: Dict) -> str:
        """Extract and concatenate voice segments"""
        try:
            # Generate output filename
            username = metadata.get('username', 'unknown')
            platform = metadata.get('platform', 'unknown')
            timestamp = int(time.time())
            
            output_file = os.path.join(
                self.voice_only_dir, 
                f"{username}_{platform}_voice_only_{timestamp}.wav"
            )
            
            wav_file = self._convert_to_wav(file_path)
            
            # Create ffmpeg filter to extract and concatenate segments
            filter_parts = []
            input_parts = []
            
            for i, seg in enumerate(segments):
                filter_parts.append(f"[0:a]atrim=start={seg['start']}:end={seg['end']},asetpts=PTS-STARTPTS[seg{i}]")
                input_parts.append(f"[seg{i}]")
            
            # Concatenate all segments
            concat_filter = f"{''.join(input_parts)}concat=n={len(segments)}:v=0:a=1[out]"
            filter_complex = ';'.join(filter_parts) + ';' + concat_filter
            
            # Run ffmpeg
            cmd = [
                'ffmpeg', '-i', wav_file,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            if wav_file != file_path:
                os.unlink(wav_file)
            
            if result.returncode == 0 and os.path.exists(output_file):
                # Show compression info
                orig_size = os.path.getsize(file_path)
                new_size = os.path.getsize(output_file)
                compression = (1 - new_size/orig_size) * 100
                print(f"      📊 Size: {orig_size//1024}KB → {new_size//1024}KB ({compression:.1f}% reduction)")
                return output_file
            
            return None
            
        except Exception as e:
            print(f"      ❌ Extraction failed: {e}")
            return None

    def _verify_output_quality(self, file_path: str) -> bool:
        """Verify the extracted voice file quality"""
        try:
            size = os.path.getsize(file_path)
            if size < 1000:  # Too small
                return False
            
            # Get duration
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            
            return duration >= 0.5  # At least 0.5 seconds
            
        except:
            return False

    def _analyze_extracted_speech(self, file_path: str, language: str) -> Dict:
        """Analyze the final extracted speech"""
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=language)
                
                return {
                    'extracted_text': text,
                    'word_count': len(text.split()) if text else 0,
                    'language_detected': language,
                    'confidence': min(0.9, len(text) / 100) if text else 0
                }
        except:
            return {
                'extracted_text': '',
                'word_count': 0,
                'language_detected': language,
                'confidence': 0
            }

    def _detect_language(self, username: str) -> str:
        """Detect language from username"""
        username_lower = username.lower()
        
        # Russian
        if any(c in username_lower for c in 'йцукенгшщзхъфывапролджэячсмитьбю'):
            return 'ru-RU'
        # Spanish
        elif any(c in username_lower for c in 'ñáéíóúü'):
            return 'es-ES'
        # German
        elif any(c in username_lower for c in 'äöüß'):
            return 'de-DE'
        # French
        elif any(c in username_lower for c in 'àâäçéèêëïîôùûüÿñæœ'):
            return 'fr-FR'
        else:
            return self.default_language

    def _convert_to_wav(self, file_path: str) -> str:
        """Convert audio to WAV format"""
        if file_path.lower().endswith('.wav'):
            return file_path
        
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        cmd = ['ffmpeg', '-i', file_path, '-ac', '1', '-ar', '16000', '-y', temp_wav.name]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0:
            return temp_wav.name
        else:
            os.unlink(temp_wav.name)
            raise Exception("Audio conversion failed")

    def _extract_metadata(self, filename: str) -> Dict:
        """Extract metadata from filename"""
        parts = filename.replace('.mp3', '').replace('.wav', '').replace('.m4a', '').split('_')
        
        return {
            'original_filename': filename,
            'username': parts[0] if parts else 'unknown',
            'platform': parts[1] if len(parts) > 1 else 'unknown',
            'video_number': parts[2] if len(parts) > 2 else 'v1',
            'timestamp': parts[3] if len(parts) > 3 else str(int(time.time()))
        }

    def save_results(self, results: List[Dict]) -> str:
        """Save processing results to CSV"""
        if not results:
            return ""
        
        data = []
        for result in results:
            data.append({
                'original_filename': result.get('original_filename'),
                'username': result.get('username'),
                'platform': result.get('platform'),
                'language': result.get('language'),
                'voice_duration': result.get('voice_duration', 0),
                'total_segments': result.get('total_segments', 0),
                'extracted_text': result.get('speech_info', {}).get('extracted_text', ''),
                'word_count': result.get('speech_info', {}).get('word_count', 0),
                'voice_only_file': result.get('voice_only_file')
            })
        
        csv_file = os.path.join(self.analysis_dir, f"voice_extraction_results_{int(time.time())}.csv")
        pd.DataFrame(data).to_csv(csv_file, index=False)
        print(f"📄 Results saved: {csv_file}")
        return csv_file

    def generate_report(self, results: List[Dict]) -> str:
        """Generate processing report"""
        report_file = os.path.join(self.analysis_dir, "voice_extraction_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎤 VOICE EXTRACTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Files processed: {len(results)}\n")
            
            total_duration = sum(r.get('voice_duration', 0) for r in results)
            f.write(f"Total voice extracted: {total_duration:.1f} seconds\n\n")
            
            # Language breakdown
            languages = {}
            for r in results:
                lang = r.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            
            f.write("🌍 LANGUAGES DETECTED:\n")
            for lang, count in languages.items():
                f.write(f"  {lang}: {count} files\n")
            
            f.write("\n📁 EXTRACTED FILES:\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result.get('original_filename')}\n")
                f.write(f"   User: {result.get('username')} | Platform: {result.get('platform')}\n")
                f.write(f"   Language: {result.get('language')} | Duration: {result.get('voice_duration', 0):.1f}s\n")
                f.write(f"   Text: {result.get('speech_info', {}).get('extracted_text', '')[:100]}...\n")
                f.write(f"   Output: {os.path.basename(result.get('voice_only_file', ''))}\n\n")
        
        print(f"📋 Report saved: {report_file}")
        return report_file

# Command line usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract voice-only segments from audio files")
    parser.add_argument("audio_dir", help="Directory containing MP3/audio files")
    parser.add_argument("--output-dir", default="voice_analysis", help="Output directory")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum voice confidence (0.0-1.0)")
    parser.add_argument("--min-length", type=float, default=1.0, help="Minimum segment length in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_dir):
        print(f"❌ Directory not found: {args.audio_dir}")
        exit(1)
    
    # Process files
    processor = AdvancedVoiceProcessor(
        output_dir=args.output_dir,
        min_voice_confidence=args.min_confidence,
        voice_segment_min_length=args.min_length
    )
    
    results = processor.process_audio_directory(args.audio_dir)
    
    if results:
        csv_file = processor.save_results(results)
        report_file = processor.generate_report(results)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"✅ Voice extracted from {len(results)} files")
        print(f"📁 Voice-only files: {processor.voice_only_dir}")
        print(f"📄 Results: {csv_file}")
        print(f"📋 Report: {report_file}")
    else:
        print("❌ No voice detected in any files")
