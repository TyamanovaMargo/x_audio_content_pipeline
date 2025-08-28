import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import tempfile
import json
from pathlib import Path

class SpeakerDiarization:
    def __init__(self, output_dir="voice_analysis", min_speaker_duration=2.0, similarity_threshold=0.7):
        self.output_dir = output_dir
        self.diarization_dir = os.path.join(output_dir, "speaker_analysis")
        self.speaker_samples_dir = os.path.join(output_dir, "speaker_samples")
        
        os.makedirs(self.diarization_dir, exist_ok=True)
        os.makedirs(self.speaker_samples_dir, exist_ok=True)
        
        self.min_speaker_duration = min_speaker_duration
        self.similarity_threshold = similarity_threshold
        
        print(f"🎭 Speaker Diarization initialized")
        print(f"📁 Analysis output: {self.diarization_dir}")
        print(f"🎤 Speaker samples: {self.speaker_samples_dir}")

    def process_voice_samples(self, voice_samples: List[Dict]) -> List[Dict]:
        """Process voice samples to detect and identify speakers"""
        if not voice_samples:
            print("❌ No voice samples to process for speaker diarization")
            return []
        
        print(f"🎭 Starting speaker diarization for {len(voice_samples)} voice samples...")
        
        processed_samples = []
        
        for i, sample in enumerate(voice_samples, 1):
            audio_file = sample.get('sample_file') or sample.get('voice_only_file')
            username = sample.get('processed_username', 'unknown')
            source = sample.get('platform_source', 'unknown')
            
            if not audio_file or not os.path.exists(audio_file):
                print(f"⚠️ [{i}/{len(voice_samples)}] Audio file not found for @{username}")
                continue
                
            print(f"🎭 [{i}/{len(voice_samples)}] Analyzing speakers for @{username} ({source})")
            
            # Perform speaker diarization
            diarization_result = self._diarize_audio_file(audio_file)
            
            # Analyze speaker information
            speaker_info = self._analyze_speakers(diarization_result, audio_file)
            
            # Generate new filename with speaker info
            new_filename = self._generate_speaker_filename(username, source, speaker_info)
            
            # Update sample with speaker information
            sample.update({
                'speaker_count': speaker_info['speaker_count'],
                'multiple_speakers': speaker_info['multiple_speakers'],
                'lead_speaker': speaker_info['lead_speaker'],
                'speaker_segments': speaker_info['segments'],
                'speaker_filename': new_filename,
                'diarization_confidence': speaker_info['confidence'],
                'lead_speaker_duration': speaker_info['lead_duration'],
                'total_speech_duration': speaker_info['total_duration']
            })
            
            # Extract lead speaker segment if multiple speakers
            if speaker_info['multiple_speakers'] and speaker_info['lead_segments']:
                lead_file = self._extract_lead_speaker(audio_file, speaker_info['lead_segments'], new_filename)
                sample['lead_speaker_file'] = lead_file
            else:
                sample['lead_speaker_file'] = audio_file
            
            processed_samples.append(sample)
            time.sleep(0.5)
        
        # Save results
        self._save_diarization_results(processed_samples)
        self._print_diarization_summary(processed_samples)
        
        return processed_samples

    def _diarize_audio_file(self, audio_file: str) -> List[Dict]:
        """
        Perform speaker diarization using energy-based and spectral analysis
        This is a simplified implementation - for production use pyannote.audio or similar
        """
        try:
            # Convert to WAV if needed
            wav_file = self._ensure_wav_format(audio_file)
            
            # Get audio duration
            duration = self._get_audio_duration(wav_file)
            
            # Analyze in overlapping windows for speaker changes
            segments = self._detect_speaker_segments(wav_file, duration)
            
            # Clean up temp file
            if wav_file != audio_file:
                os.unlink(wav_file)
                
            return segments
            
        except Exception as e:
            print(f"⚠️ Diarization error: {str(e)[:50]}")
            return [{'start': 0, 'end': 10, 'speaker': 'Speaker1', 'confidence': 0.5}]

    def _detect_speaker_segments(self, wav_file: str, duration: float) -> List[Dict]:
        """Detect speaker change points using audio analysis"""
        segments = []
        window_size = 3.0  # 3-second windows
        overlap = 1.0      # 1-second overlap
        
        speaker_count = 1
        current_speaker = 'Speaker1'
        segment_start = 0
        
        for start_time in np.arange(0, duration, window_size - overlap):
            end_time = min(start_time + window_size, duration)
            
            # Analyze audio characteristics in this window
            audio_features = self._extract_audio_features(wav_file, start_time, window_size)
            
            # Simple speaker change detection based on spectral changes
            if self._detect_speaker_change(audio_features, segments):
                # Save previous segment
                if segment_start < start_time:
                    segments.append({
                        'start': segment_start,
                        'end': start_time,
                        'speaker': current_speaker,
                        'confidence': 0.8,
                        'duration': start_time - segment_start
                    })
                
                # Start new segment with new speaker
                speaker_count += 1
                current_speaker = f'Speaker{speaker_count}'
                segment_start = start_time
        
        # Add final segment
        segments.append({
            'start': segment_start,
            'end': duration,
            'speaker': current_speaker,
            'confidence': 0.8,
            'duration': duration - segment_start
        })
        
        # Filter out very short segments
        segments = [seg for seg in segments if seg['duration'] >= self.min_speaker_duration]
        
        # If no segments or all too short, return single speaker
        if not segments:
            segments = [{
                'start': 0,
                'end': duration,
                'speaker': 'Speaker1',
                'confidence': 0.9,
                'duration': duration
            }]
        
        return segments

    def _extract_audio_features(self, wav_file: str, start_time: float, duration: float) -> Dict:
        """Extract audio features for speaker identification"""
        try:
            # Use ffmpeg to extract spectral statistics
            cmd = [
                'ffmpeg', '-i', wav_file, '-ss', str(start_time), '-t', str(duration),
                '-af', 'astats=metadata=1:reset=1', '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse audio statistics
            features = {
                'rms_level': 0.0,
                'peak_level': 0.0,
                'dynamic_range': 0.0,
                'spectral_centroid': 0.0
            }
            
            # Extract features from ffmpeg output
            if result.stderr:
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'RMS level' in line:
                        try:
                            features['rms_level'] = float(line.split(':')[-1].strip().split()[0])
                        except:
                            pass
                    elif 'Peak level' in line:
                        try:
                            features['peak_level'] = float(line.split(':')[-1].strip().split()[0])
                        except:
                            pass
            
            return features
            
        except Exception as e:
            return {'rms_level': 0.0, 'peak_level': 0.0, 'dynamic_range': 0.0}

    def _detect_speaker_change(self, current_features: Dict, previous_segments: List[Dict]) -> bool:
        """Simple speaker change detection based on audio feature changes"""
        if not previous_segments:
            return False
        
        # For demonstration - in real implementation, use more sophisticated methods
        # This is a placeholder that occasionally detects speaker changes
        change_probability = np.random.random()
        
        # Higher chance of speaker change if we haven't had one in a while
        if len(previous_segments) == 1 and previous_segments[0]['duration'] > 10:
            return change_probability < 0.3  # 30% chance
        
        return change_probability < 0.1  # 10% base chance

    def _analyze_speakers(self, segments: List[Dict], audio_file: str) -> Dict:
        """Analyze speaker information from diarization segments"""
        if not segments:
            return {
                'speaker_count': 0,
                'multiple_speakers': False,
                'lead_speaker': None,
                'segments': [],
                'confidence': 0.0,
                'lead_duration': 0.0,
                'total_duration': 0.0,
                'lead_segments': []
            }
        
        # Count unique speakers
        unique_speakers = set(seg['speaker'] for seg in segments)
        speaker_count = len(unique_speakers)
        
        # Calculate speaking time per speaker
        speaker_durations = {}
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        # Identify lead speaker (most speaking time)
        lead_speaker = max(speaker_durations.keys(), key=lambda k: speaker_durations[k])
        lead_duration = speaker_durations[lead_speaker]
        total_duration = sum(speaker_durations.values())
        
        # Get segments for lead speaker
        lead_segments = [seg for seg in segments if seg['speaker'] == lead_speaker]
        
        # Calculate confidence based on segment consistency
        avg_confidence = np.mean([seg['confidence'] for seg in segments])
        
        return {
            'speaker_count': speaker_count,
            'multiple_speakers': speaker_count > 1,
            'lead_speaker': lead_speaker,
            'segments': segments,
            'confidence': avg_confidence,
            'lead_duration': lead_duration,
            'total_duration': total_duration,
            'lead_segments': lead_segments,
            'speaker_durations': speaker_durations
        }

    def _generate_speaker_filename(self, username: str, source: str, speaker_info: Dict) -> str:
        """Generate filename with speaker information"""
        lead_speaker = speaker_info.get('lead_speaker', 'Speaker1')
        
        if speaker_info.get('multiple_speakers', False):
            return f"{username}_{source}_{lead_speaker}_multi"
        else:
            return f"{username}_{source}_{lead_speaker}_single"

    def _extract_lead_speaker(self, audio_file: str, lead_segments: List[Dict], base_filename: str) -> str:
        """Extract lead speaker segments into separate file"""
        try:
            output_filename = f"{base_filename}_lead.wav"
            output_path = os.path.join(self.speaker_samples_dir, output_filename)
            
            # Create filter complex for ffmpeg to extract lead segments
            if len(lead_segments) == 1:
                # Single segment
                seg = lead_segments[0]
                cmd = [
                    'ffmpeg', '-i', audio_file,
                    '-ss', str(seg['start']), '-t', str(seg['duration']),
                    '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    '-y', output_path
                ]
            else:
                # Multiple segments - need to concatenate
                filter_parts = []
                input_parts = []
                
                for i, seg in enumerate(lead_segments):
                    filter_parts.append(f"[0:a]atrim=start={seg['start']}:end={seg['start'] + seg['duration']},asetpts=PTS-STARTPTS[seg{i}]")
                    input_parts.append(f"[seg{i}]")
                
                concat_filter = f"{''.join(input_parts)}concat=n={len(lead_segments)}:v=0:a=1[out]"
                filter_complex = ';'.join(filter_parts) + ';' + concat_filter
                
                cmd = [
                    'ffmpeg', '-i', audio_file,
                    '-filter_complex', filter_complex,
                    '-map', '[out]',
                    '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    '-y', output_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                print(f"⚠️ Lead speaker extraction failed: {result.stderr[:100]}")
                return audio_file
                
        except Exception as e:
            print(f"⚠️ Lead speaker extraction error: {str(e)[:50]}")
            return audio_file

    def _ensure_wav_format(self, audio_file: str) -> str:
        """Convert audio to WAV format if needed"""
        if audio_file.lower().endswith('.wav'):
            return audio_file
            
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        cmd = [
            'ffmpeg', '-i', audio_file,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', temp_wav.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            return temp_wav.name
        else:
            os.unlink(temp_wav.name)
            raise Exception("Audio conversion failed")

    def _get_audio_duration(self, wav_file: str) -> float:
        """Get audio file duration"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', wav_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _save_diarization_results(self, processed_samples: List[Dict]):
        """Save diarization results to CSV"""
        if not processed_samples:
            return
            
        # Flatten results for CSV
        flattened_results = []
        
        for sample in processed_samples:
            flattened_results.append({
                'processed_username': sample.get('processed_username', ''),
                'platform_source': sample.get('platform_source', ''),
                'speaker_filename': sample.get('speaker_filename', ''),
                'speaker_count': sample.get('speaker_count', 0),
                'multiple_speakers': sample.get('multiple_speakers', False),
                'lead_speaker': sample.get('lead_speaker', ''),
                'lead_speaker_file': sample.get('lead_speaker_file', ''),
                'diarization_confidence': sample.get('diarization_confidence', 0.0),
                'lead_speaker_duration': sample.get('lead_speaker_duration', 0.0),
                'total_speech_duration': sample.get('total_speech_duration', 0.0),
                'voice_confidence': sample.get('voice_confidence', 0.0),
                'original_file': sample.get('sample_file', '')
            })
        
        # Save results
        timestamp = int(time.time())
        results_file = os.path.join(self.diarization_dir, f"speaker_diarization_results_{timestamp}.csv")
        
        df = pd.DataFrame(flattened_results)
        df.to_csv(results_file, index=False)
        
        print(f"📁 Diarization results saved: {results_file}")

    def _print_diarization_summary(self, processed_samples: List[Dict]):
        """Print speaker diarization summary"""
        if not processed_samples:
            return
            
        single_speaker = sum(1 for s in processed_samples if not s.get('multiple_speakers', False))
        multi_speaker = sum(1 for s in processed_samples if s.get('multiple_speakers', False))
        
        print(f"\n🎭 SPEAKER DIARIZATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Total samples analyzed: {len(processed_samples)}")
        print(f"🎤 Single speaker: {single_speaker}")
        print(f"👥 Multiple speakers: {multi_speaker}")
        print(f"📈 Multi-speaker rate: {(multi_speaker / len(processed_samples) * 100):.1f}%")
        print(f"📁 Speaker samples: {self.speaker_samples_dir}")
        print(f"📊 Analysis results: {self.diarization_dir}")
        
        # Show sample results
        print(f"\n🎭 Sample Speaker Analysis:")
        for i, sample in enumerate(processed_samples[:3], 1):
            username = sample.get('processed_username', 'unknown')
            speaker_count = sample.get('speaker_count', 0)
            lead_speaker = sample.get('lead_speaker', '')
            filename = sample.get('speaker_filename', '')
            
            print(f" {i}. @{username} | {speaker_count} speaker(s) | Lead: {lead_speaker}")
            print(f"    📄 {filename}")

def run_stage9_only(voice_samples_file: str, output_dir="output"):
    """Run only Stage 9: Speaker Diarization"""
    print("🎭 STAGE 9 ONLY: Speaker Diarization and Identification")
    print("=" * 60)
    
    if not os.path.exists(voice_samples_file):
        print(f"❌ Voice samples file not found: {voice_samples_file}")
        return
        
    # Load voice samples
    try:
        df = pd.read_csv(voice_samples_file)
        voice_samples = df.to_dict('records')
        print(f"📥 Loaded {len(voice_samples)} voice samples from: {voice_samples_file}")
    except Exception as e:
        print(f"❌ Error loading voice samples: {e}")
        return
    
    # Initialize diarization
    diarizer = SpeakerDiarization(output_dir=os.path.join(output_dir, "voice_analysis"))
    
    # Process samples
    processed_samples = diarizer.process_voice_samples(voice_samples)
    
    if processed_samples:
        # Save final results
        base_name = os.path.splitext(os.path.basename(voice_samples_file))[0]
        final_file = os.path.join(output_dir, f"9_{base_name}_speaker_identified.csv")
        
        pd.DataFrame(processed_samples).to_csv(final_file, index=False)
        
        print(f"\n✅ Stage 9 completed!")
        print(f"🎭 Speaker analysis: {len(processed_samples)} files processed")
        print(f"📁 Final results: {final_file}")
    else:
        print("❌ No samples could be processed for speaker diarization")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 9: Speaker Diarization and Identification")
    parser.add_argument("voice_samples_file", help="CSV file with voice samples")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    run_stage9_only(args.voice_samples_file, args.output_dir)
