#!/usr/bin/env python3
"""
Audio Labels Analyzer - Shows ALL AudioSet labels for audio files
Uses Hugging Face AST model to display complete classification results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Hugging Face transformers
try:
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
except ImportError:
    print("Error: transformers library not installed.")
    print("Please install with: pip install transformers torch torchaudio")
    sys.exit(1)

class AudioLabelsAnalyzer:
    """Complete audio analyzer showing all AudioSet labels."""
    
    def __init__(self, device: str = None, top_k: int = 20):
        """Initialize the analyzer."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_k = top_k
        print(f"Using device: {self.device}")
        
        # Load AST model
        self.model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        print(f"Loading model: {self.model_name}")
        
        try:
            self.processor = ASTFeatureExtractor.from_pretrained(self.model_name)
            self.model = ASTForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get all AudioSet labels
            self.labels = self.model.config.id2label
            print(f"Model loaded with {len(self.labels)} AudioSet classes")
            
            # Show all available labels
            self.show_all_labels()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def show_all_labels(self):
        """Display all 527 AudioSet labels."""
        print("\n" + "="*80)
        print("ALL AUDIOSET LABELS (527 classes):")
        print("="*80)
        
        # Group labels by category
        categories = {
            'Human sounds': [],
            'Animal sounds': [], 
            'Music': [],
            'Musical instruments': [],
            'Natural sounds': [],
            'Sounds of things': [],
            'Channel, environment and background': []
        }
        
        for idx, label in self.labels.items():
            label_lower = label.lower()
            
            # Categorize labels
            if any(word in label_lower for word in ['speech', 'voice', 'human', 'laugh', 'cry', 'conversation', 'babbl']):
                categories['Human sounds'].append((idx, label))
            elif any(word in label_lower for word in ['animal', 'bird', 'dog', 'cat', 'insect', 'frog', 'roar', 'bark']):
                categories['Animal sounds'].append((idx, label))
            elif any(word in label_lower for word in ['music', 'song', 'singing', 'vocal', 'choir', 'opera']):
                categories['Music'].append((idx, label))
            elif any(word in label_lower for word in ['piano', 'guitar', 'drum', 'violin', 'trumpet', 'instrument', 'flute']):
                categories['Musical instruments'].append((idx, label))
            elif any(word in label_lower for word in ['wind', 'rain', 'water', 'ocean', 'thunder', 'fire']):
                categories['Natural sounds'].append((idx, label))
            elif any(word in label_lower for word in ['vehicle', 'engine', 'door', 'bell', 'alarm', 'phone', 'computer']):
                categories['Sounds of things'].append((idx, label))
            else:
                categories['Channel, environment and background'].append((idx, label))
        
        # Print categorized labels
        for category, items in categories.items():
            if items:
                print(f"\n{category} ({len(items)} labels):")
                print("-" * 40)
                for idx, label in sorted(items):
                    print(f"  {idx:3d}: {label}")
        
        print(f"\nTotal labels: {len(self.labels)}")
        print("="*80)
    
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> np.ndarray:
        """Preprocess audio for AST model."""
        audio_np = audio.numpy()
        
        if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
            audio_np = np.mean(audio_np, axis=0)
        
        if len(audio_np.shape) > 1:
            audio_np = audio_np.flatten()
        
        return audio_np
    
    def analyze_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """Get complete classification results for an audio chunk."""
        try:
            inputs = self.processor(
                audio_chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities for ALL classes
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-self.top_k:][::-1]
            
            # Create complete results
            all_predictions = []
            for i, prob in enumerate(probabilities):
                all_predictions.append({
                    'class_id': i,
                    'label': self.labels[i],
                    'probability': float(prob)
                })
            
            top_predictions = [
                {
                    'class_id': int(idx),
                    'label': self.labels[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            return {
                'all_predictions': all_predictions,
                'top_predictions': top_predictions,
                'max_probability': float(np.max(probabilities)),
                'avg_probability': float(np.mean(probabilities))
            }
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return {
                'all_predictions': [],
                'top_predictions': [],
                'max_probability': 0.0,
                'avg_probability': 0.0
            }
    
    def analyze_file(self, file_path: Path, chunk_duration: float = 10.0) -> Dict:
        """Analyze complete audio file with all labels."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {file_path.name}")
        print('='*60)
        
        try:
            # Load and resample audio
            audio, sr = torchaudio.load(str(file_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
                sr = 16000
            
            # Preprocess
            audio_np = self.preprocess_audio(audio, sr)
            
            # Split into chunks
            chunk_samples = int(chunk_duration * sr)
            chunks = []
            for i in range(0, len(audio_np), chunk_samples):
                chunk = audio_np[i:i + chunk_samples]
                if len(chunk) >= sr * 5:  # At least 5 seconds
                    chunks.append(chunk)
            
            if not chunks:
                print("No valid chunks found!")
                return {'error': 'No valid chunks'}
            
            print(f"Analyzing {len(chunks)} chunks...")
            
            # Analyze each chunk
            chunk_results = []
            all_class_scores = np.zeros(len(self.labels))
            
            for i, chunk in enumerate(chunks):
                result = self.analyze_audio_chunk(chunk)
                chunk_results.append(result)
                
                # Accumulate scores for averaging
                for pred in result['all_predictions']:
                    all_class_scores[pred['class_id']] += pred['probability']
                
                # Show top 5 for this chunk
                print(f"\nChunk {i+1} (time: {i*chunk_duration:.1f}-{(i+1)*chunk_duration:.1f}s):")
                for j, pred in enumerate(result['top_predictions'][:5]):
                    print(f"  {j+1}. {pred['label']}: {pred['probability']:.4f}")
            
            # Average scores across all chunks
            avg_class_scores = all_class_scores / len(chunks)
            
            # Get overall top predictions
            top_indices = np.argsort(avg_class_scores)[-self.top_k:][::-1]
            overall_top_predictions = [
                {
                    'class_id': int(idx),
                    'label': self.labels[idx],
                    'avg_probability': float(avg_class_scores[idx])
                }
                for idx in top_indices
            ]
            
            # Show overall results
            print(f"\n{'='*60}")
            print("OVERALL FILE ANALYSIS:")
            print('='*60)
            print(f"File: {file_path.name}")
            print(f"Duration: ~{len(chunks) * chunk_duration:.1f} seconds")
            print(f"Chunks analyzed: {len(chunks)}")
            
            print(f"\nTop {self.top_k} predictions (averaged across all chunks):")
            for i, pred in enumerate(overall_top_predictions):
                print(f"  {i+1:2d}. {pred['label']:50s}: {pred['avg_probability']:.4f}")
            
            # Create complete file analysis
            file_analysis = {
                'filename': file_path.name,
                'analysis_time': datetime.now().isoformat(),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'duration_seconds': len(chunks) * chunk_duration,
                'chunks_analyzed': len(chunks),
                'overall_top_predictions': overall_top_predictions,
                'all_class_averages': [
                    {
                        'class_id': i,
                        'label': self.labels[i],
                        'avg_probability': float(avg_class_scores[i])
                    }
                    for i in range(len(self.labels))
                ],
                'chunk_details': [
                    {
                        'chunk_id': i + 1,
                        'start_time': i * chunk_duration,
                        'end_time': (i + 1) * chunk_duration,
                        'top_predictions': result['top_predictions']
                    }
                    for i, result in enumerate(chunk_results)
                ]
            }
            
            return file_analysis
            
        except Exception as e:
            print(f"Error analyzing file: {e}")
            return {'error': str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze audio files and show ALL AudioSet label predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_path", help="Path to MP3 file or directory")
    parser.add_argument("--output_file", default="audio_analysis_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--top_k", type=int, default=20,
                       help="Number of top predictions to show (default: 20)")
    parser.add_argument("--chunk_duration", type=float, default=10.0,
                       help="Duration of audio chunks in seconds")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AudioLabelsAnalyzer(top_k=args.top_k)
    
    # Find files to analyze
    input_path = Path(args.input_path)
    if input_path.is_file():
        files_to_analyze = [input_path]
    elif input_path.is_dir():
        files_to_analyze = list(input_path.glob("*.mp3"))
        if not files_to_analyze:
            print(f"No MP3 files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {args.input_path} not found")
        sys.exit(1)
    
    print(f"\nFound {len(files_to_analyze)} file(s) to analyze")
    
    # Analyze all files
    all_results = {}
    
    for file_path in files_to_analyze:
        result = analyzer.analyze_file(file_path, args.chunk_duration)
        all_results[file_path.name] = result
    
    # Save results to JSON
    output_data = {
        'analysis_summary': {
            'total_files': len(files_to_analyze),
            'analysis_time': datetime.now().isoformat(),
            'model_used': analyzer.model_name,
            'parameters': {
                'top_k': args.top_k,
                'chunk_duration': args.chunk_duration
            }
        },
        'audioset_labels': {
            str(idx): label for idx, label in analyzer.labels.items()
        },
        'file_results': all_results
    }
    
    output_file = Path(args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print('='*60)
    print(f"Results saved to: {output_file}")
    print(f"Files analyzed: {len(files_to_analyze)}")
    print(f"Total AudioSet classes: {len(analyzer.labels)}")
    
    # Show summary of most common predictions
    if all_results:
        print(f"\nMost common predictions across all files:")
        all_class_counts = {}
        
        for file_result in all_results.values():
            if 'overall_top_predictions' in file_result:
                for pred in file_result['overall_top_predictions'][:5]:
                    label = pred['label']
                    if label not in all_class_counts:
                        all_class_counts[label] = 0
                    all_class_counts[label] += 1
        
        sorted_counts = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_counts[:10]:
            print(f"  {label}: {count} files")

if __name__ == "__main__":
    main()
