#!/usr/bin/env python3
"""
Audio Speech Filter Script - Complete Rewrite
Uses Hugging Face Audio Spectrogram Transformer to classify MP3 files
and keep only those containing speech content.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Hugging Face transformers
try:
    from transformers import pipeline, AutoProcessor, AutoModelForAudioClassification
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
except ImportError:
    print("Error: transformers library not installed.")
    print("Please install with: pip install transformers torch torchaudio")
    sys.exit(1)

class AudioSpeechClassifier:
    """Audio classifier using Hugging Face AST model."""
    
    def __init__(self, device: str = None):
        """Initialize the AST classifier."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load AST model fine-tuned on AudioSet
        self.model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        print(f"Loading model: {self.model_name}")
        
        try:
            # Load model and processor
            self.processor = ASTFeatureExtractor.from_pretrained(self.model_name)
            self.model = ASTForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get the label list for AudioSet classes
            self.labels = self.model.config.id2label
            print(f"Model loaded successfully with {len(self.labels)} classes")
            
            # Find speech-related class indices
            self.speech_classes = self._find_speech_classes()
            self.music_classes = self._find_music_classes()
            self.singing_classes = self._find_singing_classes()
            
            print(f"Found {len(self.speech_classes)} speech classes")
            print(f"Found {len(self.music_classes)} music classes") 
            print(f"Found {len(self.singing_classes)} singing classes")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def _find_speech_classes(self) -> List[int]:
        """Find class indices related to speech."""
        speech_keywords = [
            'speech', 'speaking', 'conversation', 'narration', 'monologue',
            'male speech', 'female speech', 'child speech', 'whisper',
            'human voice', 'voice', 'talk', 'lecture'
        ]
        
        speech_indices = []
        for idx, label in self.labels.items():
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in speech_keywords):
                speech_indices.append(idx)
                print(f"  Speech class {idx}: {label}")
        
        return speech_indices
    
    def _find_music_classes(self) -> List[int]:
        """Find class indices related to music."""
        music_keywords = [
            'music', 'musical', 'instrument', 'piano', 'guitar', 'drum',
            'orchestra', 'symphony', 'melody', 'rhythm', 'beat'
        ]
        
        music_indices = []
        for idx, label in self.labels.items():
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in music_keywords) and 'speech' not in label_lower:
                music_indices.append(idx)
        
        return music_indices
    
    def _find_singing_classes(self) -> List[int]:
        """Find class indices related to singing."""
        singing_keywords = [
            'singing', 'vocal', 'song', 'choir', 'opera', 'chant', 'hum'
        ]
        
        singing_indices = []
        for idx, label in self.labels.items():
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in singing_keywords):
                singing_indices.append(idx)
                print(f"  Singing class {idx}: {label}")
        
        return singing_indices
    
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Preprocess audio for AST model."""
        # Convert to numpy for the processor
        audio_np = audio.numpy()
        
        # Convert to mono if stereo
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            audio_np = np.mean(audio_np, axis=0)
        
        # Ensure 1D array
        if len(audio_np.shape) > 1:
            audio_np = audio_np.flatten()
        
        return audio_np
    
    def classify_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Classify a single audio chunk using AST."""
        try:
            # Process audio with the AST processor
            inputs = self.processor(
                audio_chunk, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Calculate category scores
            speech_score = np.mean([probabilities[idx] for idx in self.speech_classes]) if self.speech_classes else 0.0
            music_score = np.mean([probabilities[idx] for idx in self.music_classes]) if self.music_classes else 0.0
            singing_score = np.mean([probabilities[idx] for idx in self.singing_classes]) if self.singing_classes else 0.0
            
            # Get top predictions for debugging
            top_indices = np.argsort(probabilities)[-5:][::-1]
            top_predictions = [(self.labels[idx], probabilities[idx]) for idx in top_indices]
            
            return {
                'speech': float(speech_score),
                'music': float(music_score),
                'singing': float(singing_score),
                'other': float(1.0 - speech_score - music_score - singing_score),
                'top_predictions': top_predictions
            }
            
        except Exception as e:
            print(f"Error in chunk classification: {e}")
            return {
                'speech': 0.0,
                'music': 0.0,
                'singing': 0.0,
                'other': 1.0,
                'top_predictions': []
            }

class AudioSpeechFilter:
    """Main class for filtering audio files based on speech content."""
    
    def __init__(self, input_dir: str, output_dir: str = "speech_only_output",
                 chunk_duration: float = 10.0, speech_threshold: float = 0.1):
        """Initialize the audio speech filter."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_duration = chunk_duration
        self.speech_threshold = speech_threshold
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize classifier
        self.classifier = AudioSpeechClassifier()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'speech_files': 0,
            'music_files': 0,
            'other_files': 0,
            'error_files': 0
        }
    
    def load_audio_file(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load an audio file."""
        try:
            # Load audio at 16kHz to match AST model requirements
            audio, sr = torchaudio.load(str(file_path))
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
                sr = 16000
            
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None, None
    
    def split_audio_into_chunks(self, audio: torch.Tensor, sr: int) -> List[np.ndarray]:
        """Split audio into chunks for processing."""
        chunk_samples = int(self.chunk_duration * sr)
        chunks = []
        
        # Convert to numpy and preprocess
        audio_np = self.classifier.preprocess_audio(audio, sr)
        
        # Split into chunks
        for i in range(0, len(audio_np), chunk_samples):
            chunk = audio_np[i:i + chunk_samples]
            # Only process chunks that are at least 5 seconds long
            if len(chunk) >= sr * 5:
                chunks.append(chunk)
        
        return chunks
    
    def classify_file(self, file_path: Path) -> Dict[str, any]:
        """Classify an entire audio file."""
        print(f"\nProcessing: {file_path.name}")
        
        # Load audio
        audio, sr = self.load_audio_file(file_path)
        if audio is None:
            return {'category': 'error', 'confidence': 0.0, 'chunks_analyzed': 0}
        
        # Split into chunks
        chunks = self.split_audio_into_chunks(audio, sr)
        if not chunks:
            print("  No valid chunks found")
            return {'category': 'error', 'confidence': 0.0, 'chunks_analyzed': 0}
        
        print(f"  Analyzing {len(chunks)} chunks...")
        
        # Classify each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            result = self.classifier.classify_audio_chunk(chunk, sr)
            chunk_results.append(result)
            
            # Print top prediction for first few chunks
            if i < 3 and result['top_predictions']:
                top_pred = result['top_predictions'][0]
                print(f"    Chunk {i+1}: {top_pred[0]} ({top_pred[1]:.3f})")
        
        # Aggregate results
        avg_scores = {
            'speech': np.mean([r['speech'] for r in chunk_results]),
            'music': np.mean([r['music'] for r in chunk_results]),
            'singing': np.mean([r['singing'] for r in chunk_results]),
            'other': np.mean([r['other'] for r in chunk_results])
        }
        
        # Calculate speech statistics
        speech_chunks = sum(1 for r in chunk_results if r['speech'] > self.speech_threshold)
        speech_ratio = speech_chunks / len(chunk_results)
        
        # Print detailed results
        print(f"  Average scores:")
        print(f"    Speech: {avg_scores['speech']:.3f}")
        print(f"    Music: {avg_scores['music']:.3f}")
        print(f"    Singing: {avg_scores['singing']:.3f}")
        print(f"    Other: {avg_scores['other']:.3f}")
        print(f"  Speech chunks: {speech_chunks}/{len(chunk_results)} ({speech_ratio:.2f})")
        
        # Decision logic
        if avg_scores['speech'] > self.speech_threshold and avg_scores['speech'] >= max(avg_scores['music'], avg_scores['singing']):
            category = 'speech'
            confidence = avg_scores['speech']
        elif avg_scores['music'] > avg_scores['singing'] and avg_scores['music'] > 0.1:
            category = 'music'
            confidence = avg_scores['music']
        elif avg_scores['singing'] > 0.1:
            category = 'music'  # Treat singing as music for filtering purposes
            confidence = avg_scores['singing']
        else:
            category = 'other'
            confidence = avg_scores['other']
        
        return {
            'category': category,
            'confidence': confidence,
            'chunks_analyzed': len(chunks),
            'speech_ratio': speech_ratio,
            'avg_scores': avg_scores
        }
    
    def process_files(self):
        """Process all MP3 files in the input directory."""
        # Find all MP3 files
        mp3_files = list(self.input_dir.glob("*.mp3"))
        if not mp3_files:
            print(f"No MP3 files found in {self.input_dir}")
            return
        
        print(f"Found {len(mp3_files)} MP3 files to process")
        self.stats['total_files'] = len(mp3_files)
        
        # Process each file
        for file_path in mp3_files:
            try:
                result = self.classify_file(file_path)
                
                if result['category'] == 'speech':
                    # Copy file to output directory
                    output_path = self.output_dir / file_path.name
                    shutil.copy2(file_path, output_path)
                    self.stats['speech_files'] += 1
                    print(f"  ✓ SAVED: Contains speech (confidence: {result['confidence']:.3f})")
                    
                elif result['category'] == 'music':
                    self.stats['music_files'] += 1
                    print(f"  ✗ SKIPPED: Contains music/singing (confidence: {result['confidence']:.3f})")
                    
                elif result['category'] == 'error':
                    self.stats['error_files'] += 1
                    print(f"  ⚠ ERROR: Could not process file")
                    
                else:
                    self.stats['other_files'] += 1
                    print(f"  ✗ SKIPPED: Other content (confidence: {result['confidence']:.3f})")
                    
            except Exception as e:
                print(f"  ⚠ ERROR processing {file_path.name}: {e}")
                self.stats['error_files'] += 1
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Speech files saved: {self.stats['speech_files']}")
        print(f"Music files skipped: {self.stats['music_files']}")
        print(f"Other files skipped: {self.stats['other_files']}")
        print(f"Error files: {self.stats['error_files']}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Files saved to output: {len(list(self.output_dir.glob('*.mp3')))}")
        
        if self.stats['speech_files'] > 0:
            print(f"\n✓ Success! Found {self.stats['speech_files']} files containing speech.")
        else:
            print(f"\n⚠ No speech files detected. You may need to adjust the speech threshold.")
            print(f"  Current threshold: {self.speech_threshold}")
            print(f"  Try running with: --speech_threshold 0.05")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Filter MP3 files to keep only those containing speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_filter.py /path/to/audio/folder
  python speech_filter.py /path/to/audio/folder --output_dir speech_files
  python speech_filter.py /path/to/audio/folder --speech_threshold 0.05
        """
    )
    
    parser.add_argument("input_dir", help="Directory containing MP3 files")
    parser.add_argument("--output_dir", default="speech_only_output",
                       help="Output directory for speech files (default: speech_only_output)")
    parser.add_argument("--chunk_duration", type=float, default=10.0,
                       help="Duration of audio chunks in seconds (default: 10.0)")
    parser.add_argument("--speech_threshold", type=float, default=0.1,
                       help="Minimum speech score threshold (default: 0.1)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    print("="*60)
    print("AUDIO SPEECH FILTER")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Speech threshold: {args.speech_threshold}")
    print("="*60)
    
    # Create and run filter
    filter_system = AudioSpeechFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        speech_threshold=args.speech_threshold
    )
    
    print("\nStarting audio classification...")
    filter_system.process_files()
    filter_system.print_summary()

if __name__ == "__main__":
    main()
