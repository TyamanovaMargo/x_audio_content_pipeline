import os
import shutil
import argparse
import logging
from glob import glob
from pydub import AudioSegment
import torch
import whisper
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition  # Fixed import
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class AdvancedVoiceDetector:
    """Advanced voice detection using Whisper, Pyannote VAD, and SpeechBrain"""

    def __init__(self, output_dir: str, threshold: float = 0.5, min_duration: float = 5.0, 
                 huggingface_token: Optional[str] = None, verbose: bool = False):
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_duration = min_duration
        self.huggingface_token = huggingface_token
        self.verbose = verbose

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models
        print("🔄 Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")

        print("🔄 Loading Pyannote VAD pipeline...")
        # Initialize Pyannote pipeline for VAD
        if huggingface_token:
            self.pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=huggingface_token
            )
        else:
            self.pyannote_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

        print("🔄 Loading SpeechBrain model...")
        # Load SpeechBrain pretrained speaker recognition model
        self.speechbrain_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('voice_detection.log'),
                logging.StreamHandler() if verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        print("✅ All models loaded successfully!")

    def load_audio(self, file_path: str) -> Optional[AudioSegment]:
        """Load and preprocess audio file with better error handling"""
        try:
            # Check file size first
            file_size = os.path.getsize(file_path)
            if file_size < 1000:  # Less than 1KB, likely corrupted
                self.logger.error(f"File too small, likely corrupted: {file_path}")
                return None
                
            audio = AudioSegment.from_file(file_path)
            
            # Check if audio has content
            if len(audio) < 1000:  # Less than 1 second
                self.logger.error(f"Audio too short: {file_path}")
                return None
                
            audio = audio.set_channels(1)  # mono
            audio = audio.set_frame_rate(16000)  # 16 kHz
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def audio_to_tensor(self, audio: AudioSegment) -> torch.Tensor:
        """Convert AudioSegment to proper tensor format for pyannote"""
        try:
            # Convert to numpy array first
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            if audio.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0
            
            # Convert to tensor and add batch dimension
            waveform = torch.tensor(samples).unsqueeze(0)
            return waveform
            
        except Exception as e:
            self.logger.error(f"Error converting audio to tensor: {e}")
            raise

    def detect_voice_in_file(self, file_path: str) -> Dict:
        """Detect voice in a single audio file with improved speech vs music detection"""
        result = {
            'input_file': file_path,
            'voice_detected': False,
            'voice_score': 0.0,
            'speech_duration': 0.0,
            'transcription': '',
            'error': None,
            'output_file': None,
            'music_detected': False,
            'speech_to_music_ratio': 0.0
        }

        try:
            if self.verbose:
                print(f"🎵 Processing: {os.path.basename(file_path)}")

            # Load audio
            audio = self.load_audio(file_path)
            if audio is None:
                result['error'] = 'Failed to load audio'
                return result

            waveform = self.audio_to_tensor(audio)

            # Use pyannote VAD pipeline
            vad_input = {
                'waveform': waveform,
                'sample_rate': 16000
            }
            vad_result = self.pyannote_pipeline(vad_input)
            speech_segments = vad_result.get_timeline().support()
            speech_duration = sum(segment.duration for segment in speech_segments)
            result['speech_duration'] = speech_duration

            total_duration = len(waveform[0]) / 16000
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0

            if speech_duration < self.min_duration:
                result['error'] = f'Insufficient speech duration: {speech_duration:.2f}s (min: {self.min_duration}s)'
                if self.verbose:
                    print(f" ❌ Too short: {speech_duration:.2f}s")
                return result

            # Use Whisper for transcription
            try:
                whisper_result = self.whisper_model.transcribe(file_path)
                transcript = whisper_result.get('text', '').strip()
                result['transcription'] = transcript

                word_count = len(transcript.split()) if transcript else 0

                if word_count < 5:
                    result['error'] = 'No meaningful speech detected by Whisper'
                    if self.verbose:
                        print(f" ❌ Too few words: {word_count}")
                    return result

                words_per_minute = (word_count / (speech_duration / 60)) if speech_duration > 0 else 0

                # Music detection heuristics
                music_indicators = 0
                speech_indicators = 0

                if words_per_minute < 50:
                    music_indicators += 1
                elif words_per_minute > 200:
                    music_indicators += 1
                else:
                    speech_indicators += 1

                words = transcript.lower().split()
                if len(words) > 0:
                    unique_words = len(set(words))
                    repetition_ratio = unique_words / len(words)
                    if repetition_ratio < 0.3:
                        music_indicators += 1
                    else:
                        speech_indicators += 1

                if speech_ratio < 0.3:
                    music_indicators += 1
                else:
                    speech_indicators += 1

                result['music_detected'] = music_indicators > speech_indicators
                result['speech_to_music_ratio'] = speech_indicators / (music_indicators + speech_indicators)

                if result['music_detected']:
                    result['error'] = f'Music detected (speech ratio: {speech_ratio:.2f}, words/min: {words_per_minute:.1f})'
                    if self.verbose:
                        print(f" 🎵 Music detected: {result['error']}")
                    return result

            except Exception as e:
                result['error'] = f'Whisper transcription failed: {str(e)}'
                if self.verbose:
                    print(f" ❌ Whisper failed: {e}")
                return result

            # Use SpeechBrain speaker recognition model to get voice score
            try:
                verification_result = self.speechbrain_model.verify_files(file_path, file_path)
                # Handle result if it's a tuple or a tensor
                if isinstance(verification_result, tuple):
                    score = verification_result[0]
                    if hasattr(score, 'item'):
                        score = score.item()
                elif hasattr(verification_result, 'item'):
                    score = verification_result.item()
                else:
                    score = float(verification_result)

                result['voice_score'] = score

            except Exception as e:
                self.logger.warning(f"SpeechBrain failed for {file_path}, using enhanced scoring: {e}")
                base_score = 0.5
                if 60 <= words_per_minute <= 180:
                    base_score += 0.2
                if speech_ratio > 0.5:
                    base_score += 0.2
                if word_count > 20:
                    base_score += 0.1
                result['voice_score'] = min(base_score, 1.0)

            # Enhanced voice detection criteria
            voice_criteria_met = (
                result['voice_score'] >= self.threshold and
                speech_duration >= self.min_duration and
                not result['music_detected'] and
                word_count >= 5 and
                speech_ratio >= 0.3
            )

            if voice_criteria_met:
                result['voice_detected'] = True
                filename = os.path.basename(file_path)
                output_path = os.path.join(self.output_dir, filename)
                shutil.copy(file_path, output_path)
                result['output_file'] = output_path

                if self.verbose:
                    print(f" ✅ Voice detected! Score: {result['voice_score']:.3f}, Duration: {speech_duration:.2f}s")
                    print(f" Words: {word_count}, Rate: {words_per_minute:.1f} wpm")
                    print(f" Transcript: {transcript[:100]}...")

            else:
                if self.verbose:
                    reasons = []
                    if result['voice_score'] < self.threshold:
                        reasons.append(f"low score: {result['voice_score']:.3f}")
                    if result['music_detected']:
                        reasons.append("music detected")
                    if word_count < 5:
                        reasons.append(f"few words: {word_count}")
                    if speech_ratio < 0.3:
                        reasons.append(f"low speech ratio: {speech_ratio:.2f}")
                    print(f" ❌ No voice: {', '.join(reasons)}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error processing {file_path}: {e}")
            if self.verbose:
                print(f" ❌ Processing error: {e}")

        return result

    def process_audio_file(self, file_path: str) -> List[Dict]:
        """Process a single audio file - compatibility method for pipeline"""
        result = self.detect_voice_in_file(file_path)
        return [result]  # Return as list for compatibility

    def process_audio_directory(self, source_dir: str) -> List[Dict]:
        """Process all MP3 files in a directory with progress tracking"""
        files = glob(os.path.join(source_dir, "*.mp3"))
        results = []
        total_files = len(files)
        processed_files = 0
        passed_files = 0
        failed_files = 0

        print(f"\n🎤 Found {total_files} MP3 files in {source_dir}")
        print("=" * 60)

        for file_path in files:
            processed_files += 1
            print(f"\n📁 Processing ({processed_files}/{total_files}): {os.path.basename(file_path)}")
            
            result = self.detect_voice_in_file(file_path)
            results.append(result)
            
            if result['voice_detected']:
                passed_files += 1
            elif result['error']:
                failed_files += 1

        print("\n" + "=" * 60)
        print(f"✅ Processing Complete!")
        print(f"📊 Total processed: {processed_files}")
        print(f"🎤 Voice detected: {passed_files}")
        print(f"❌ Failed/Errors: {failed_files}")
        print(f"📁 Output directory: {self.output_dir}")

        self.logger.info(f"Processed {processed_files} files. {passed_files} files contained voices and were saved.")
        return results

    def save_results_csv(self, results: List[Dict], output_file: str = "voice_detection_results.csv"):
        """Save detailed results to CSV"""
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, output_file)
        df.to_csv(csv_path, index=False)
        print(f"📄 Results saved to: {csv_path}")
        return csv_path


def main():
    parser = argparse.ArgumentParser(description='Advanced Voice Detection in MP3 files')
    parser.add_argument('--source', type=str, required=True, help='Source directory path')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory path')  
    parser.add_argument('--threshold', type=float, default=0.5, help='Voice detection score threshold (0.0-1.0)')
    parser.add_argument('--min_duration', type=float, default=5.0, help='Minimum speech duration in seconds')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save_csv', action='store_true', help='Save results to CSV file')
    
    args = parser.parse_args()

    # Initialize detector
    detector = AdvancedVoiceDetector(
        output_dir=args.dest,
        threshold=args.threshold,
        min_duration=args.min_duration,
        huggingface_token=None,  # Set your Hugging Face token if needed
        verbose=args.verbose
    )

    # Process audio directory
    results = detector.process_audio_directory(args.source)

    # Save CSV if requested
    if args.save_csv:
        detector.save_results_csv(results)

    # Final summary
    voice_detected_count = sum(1 for result in results if result['voice_detected'])
    print(f"\n🎯 Final Result: {voice_detected_count}/{len(results)} files contained voices and were saved to {args.dest}")


if __name__ == '__main__':
    main()
