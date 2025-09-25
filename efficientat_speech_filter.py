#!/usr/bin/env python3
"""
Voice Model Audio Filter - Saves files to separate folders by quality
Filters audio specifically for voice model training purposes
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

try:
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
except ImportError:
    print("Error: transformers library not installed.")
    print("Please install with: pip install transformers torch torchaudio")
    sys.exit(1)

class VoiceModelFilter:
    """Filter audio files for voice model training with separate folders."""
    
    def __init__(self, input_dir: str, output_base_dir: str = "voice_model_data"):
        """Initialize the voice model filter."""
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create output folders
        self.folders = {
            'excellent': self.output_base_dir / "01_excellent_for_voice",
            'good': self.output_base_dir / "02_good_for_voice", 
            'acceptable': self.output_base_dir / "03_acceptable_with_caution",
            'poor_quality': self.output_base_dir / "04_poor_quality",
            'background_music': self.output_base_dir / "05_has_background_music",
            'singing_vocal': self.output_base_dir / "06_singing_vocal_music",
            'emotional': self.output_base_dir / "07_emotional_speech",
            'conversation': self.output_base_dir / "08_conversation_multiple_speakers",
            'other': self.output_base_dir / "09_other_content"
        }
        
        # Create all folders
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifier
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.processor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label
        
        print(f"Model loaded with {len(self.labels)} classes")
        
        # Define label categories
        self.define_label_categories()
        
        # Statistics
        self.stats = {folder: 0 for folder in self.folders.keys()}
        self.metadata = {}
    
    def define_label_categories(self):
        """Define which labels belong to which categories."""
        
        # Excellent for voice model training
        self.excellent_labels = [
            'speech', 'narration', 'monologue', 'male speech', 'female speech'
        ]
        
        # Good but need verification
        self.good_labels = [
            'child speech', 'human voice'
        ]
        
        # Acceptable with caution
        self.acceptable_labels = [
            'babbling', 'whispering'
        ]
        
        # Poor quality indicators
        self.poor_quality_labels = [
            'inside, large room', 'inside, small room', 'reverberation',
            'echo', 'distortion', 'telephone'
        ]
        
        # Background music indicators
        self.background_music_labels = [
            'music', 'musical instrument', 'background music'
        ]
        
        # Singing/vocal music
        self.singing_labels = [
            'singing', 'vocal music', 'choir', 'opera', 'song'
        ]
        
        # Emotional speech (avoid for consistent voice models)
        self.emotional_labels = [
            'laughter', 'crying', 'screaming', 'shouting', 'giggle', 
            'chuckle', 'sobbing', 'whoop', 'bellow', 'shout'
        ]
        
        # Conversation/multiple speakers
        self.conversation_labels = [
            'conversation', 'hubbub', 'speech noise', 'speech babble'
        ]
    
    def load_and_preprocess_audio(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file."""
        try:
            audio, sr = torchaudio.load(str(file_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
                sr = 16000
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None, None
    
    def classify_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """Classify audio chunk and return detailed results."""
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
            
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get top 10 predictions
            top_indices = np.argsort(probabilities)[-10:][::-1]
            top_predictions = [
                {
                    'label': self.labels[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            return {
                'top_predictions': top_predictions,
                'all_probabilities': probabilities
            }
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return {'top_predictions': [], 'all_probabilities': np.zeros(527)}
    
    def analyze_file_for_voice_model(self, file_path: Path, chunk_duration: float = 10.0) -> Dict:
        """Analyze file and determine best folder for voice model training."""
        print(f"\nAnalyzing: {file_path.name}")
        
        audio, sr = self.load_and_preprocess_audio(file_path)
        if audio is None:
            return {'category': 'other', 'reason': 'Failed to load audio'}
        
        # Convert to numpy
        audio_np = audio.numpy().flatten()
        
        # Split into chunks
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        for i in range(0, len(audio_np), chunk_samples):
            chunk = audio_np[i:i + chunk_samples]
            if len(chunk) >= sr * 5:  # At least 5 seconds
                chunks.append(chunk)
        
        if not chunks:
            return {'category': 'other', 'reason': 'No valid chunks'}
        
        print(f"  Analyzing {len(chunks)} chunks...")
        
        # Analyze each chunk
        chunk_results = []
        for chunk in chunks:
            result = self.classify_chunk(chunk)
            chunk_results.append(result)
        
        # Aggregate results and make decision
        return self.categorize_for_voice_model(chunk_results, file_path.name)
    
    def categorize_for_voice_model(self, chunk_results: List[Dict], filename: str) -> Dict:
        """Categorize audio based on chunk analysis for voice model training."""
        
        # Analyze labels across all chunks
        all_top_labels = []
        speech_confidence_scores = []
        
        for result in chunk_results:
            if result['top_predictions']:
                top_labels = [pred['label'].lower() for pred in result['top_predictions'][:3]]
                all_top_labels.extend(top_labels)
                
                # Calculate speech confidence
                speech_score = 0
                for pred in result['top_predictions'][:5]:
                    if any(speech_word in pred['label'].lower() 
                          for speech_word in ['speech', 'voice', 'speaking', 'narration']):
                        speech_score += pred['probability']
                speech_confidence_scores.append(speech_score)
        
        avg_speech_confidence = np.mean(speech_confidence_scores) if speech_confidence_scores else 0
        
        # Decision logic for voice model categorization
        
        # Check for excellent labels (ideal for voice training)
        excellent_count = sum(1 for label in all_top_labels 
                            if any(excellent in label for excellent in self.excellent_labels))
        
        # Check for problematic content
        emotional_count = sum(1 for label in all_top_labels 
                            if any(emotional in label for emotional in self.emotional_labels))
        
        singing_count = sum(1 for label in all_top_labels 
                          if any(singing in label for singing in self.singing_labels))
        
        music_count = sum(1 for label in all_top_labels 
                        if any(music in label for music in self.background_music_labels))
        
        conversation_count = sum(1 for label in all_top_labels 
                               if any(conv in label for conv in self.conversation_labels))
        
        poor_quality_count = sum(1 for label in all_top_labels 
                               if any(poor in label for poor in self.poor_quality_labels))
        
        # Make categorization decision
        total_chunks = len(chunk_results)
        
        print(f"  Speech confidence: {avg_speech_confidence:.3f}")
        print(f"  Excellent labels: {excellent_count}/{total_chunks*3}")
        print(f"  Emotional: {emotional_count}, Singing: {singing_count}, Music: {music_count}")
        
        # EXCELLENT: High speech confidence, mostly excellent labels, no interference
        if (avg_speech_confidence > 0.6 and 
            excellent_count > total_chunks * 1.5 and  # At least 50% excellent
            emotional_count == 0 and singing_count == 0 and music_count <= 1):
            
            return {
                'category': 'excellent',
                'confidence': avg_speech_confidence,
                'reason': f'High-quality speech, no interference (confidence: {avg_speech_confidence:.3f})'
            }
        
        # GOOD: Good speech confidence, minimal interference
        elif (avg_speech_confidence > 0.4 and 
              excellent_count > total_chunks and
              emotional_count <= 1 and singing_count == 0 and music_count <= 2):
            
            return {
                'category': 'good',
                'confidence': avg_speech_confidence,
                'reason': f'Good speech quality, minimal interference (confidence: {avg_speech_confidence:.3f})'
            }
        
        # BACKGROUND MUSIC: Speech present but with background music
        elif (avg_speech_confidence > 0.3 and music_count > 2 and singing_count <= 1):
            return {
                'category': 'background_music',
                'confidence': avg_speech_confidence,
                'reason': f'Speech with background music (music count: {music_count})'
            }
        
        # SINGING/VOCAL: Primarily singing content
        elif singing_count > total_chunks:
            return {
                'category': 'singing_vocal',
                'confidence': singing_count / (total_chunks * 3),
                'reason': f'Singing/vocal music detected (count: {singing_count})'
            }
        
        # EMOTIONAL: Contains emotional speech
        elif emotional_count > 2:
            return {
                'category': 'emotional',
                'confidence': avg_speech_confidence,
                'reason': f'Emotional speech detected (count: {emotional_count})'
            }
        
        # CONVERSATION: Multiple speakers
        elif conversation_count > 1:
            return {
                'category': 'conversation',
                'confidence': avg_speech_confidence,
                'reason': f'Conversation/multiple speakers (count: {conversation_count})'
            }
        
        # POOR QUALITY: Audio quality issues
        elif poor_quality_count > 1:
            return {
                'category': 'poor_quality',
                'confidence': avg_speech_confidence,
                'reason': f'Poor audio quality (count: {poor_quality_count})'
            }
        
        # ACCEPTABLE: Some speech but with cautions
        elif avg_speech_confidence > 0.2:
            return {
                'category': 'acceptable',
                'confidence': avg_speech_confidence,
                'reason': f'Acceptable speech with cautions (confidence: {avg_speech_confidence:.3f})'
            }
        
        # OTHER: Everything else
        else:
            return {
                'category': 'other',
                'confidence': avg_speech_confidence,
                'reason': f'No significant speech content (confidence: {avg_speech_confidence:.3f})'
            }
    
    def process_files(self):
        """Process all MP3 files and organize into folders."""
        mp3_files = list(self.input_dir.glob("*.mp3"))
        if not mp3_files:
            print(f"No MP3 files found in {self.input_dir}")
            return
        
        print(f"Found {len(mp3_files)} MP3 files to process")
        
        for file_path in tqdm(mp3_files, desc="Processing files"):
            try:
                result = self.analyze_file_for_voice_model(file_path)
                
                # Copy file to appropriate folder
                category = result['category']
                target_folder = self.folders[category]
                target_path = target_folder / file_path.name
                
                shutil.copy2(file_path, target_path)
                self.stats[category] += 1
                
                # Store metadata
                self.metadata[file_path.name] = {
                    'category': category,
                    'reason': result['reason'],
                    'confidence': result.get('confidence', 0),
                    'target_folder': str(target_folder),
                    'processing_time': datetime.now().isoformat()
                }
                
                print(f"  → {category.upper()}: {result['reason']}")
                
            except Exception as e:
                print(f"  ⚠ ERROR: {e}")
                # Copy to other folder if error
                shutil.copy2(file_path, self.folders['other'] / file_path.name)
                self.stats['other'] += 1
        
        self.save_metadata()
        self.print_summary()
    
    def save_metadata(self):
        """Save detailed metadata about the categorization."""
        metadata_file = self.output_base_dir / "voice_model_categorization.json"
        
        summary_data = {
            'processing_summary': {
                'total_files': sum(self.stats.values()),
                'folder_distribution': self.stats,
                'processing_time': datetime.now().isoformat()
            },
            'folder_descriptions': {
                'excellent': 'Perfect for voice model training - clean speech, single speaker',
                'good': 'Good quality speech with minimal interference',
                'acceptable': 'Acceptable but may need manual review',
                'poor_quality': 'Audio quality issues (reverb, distortion)',
                'background_music': 'Speech with background music - may still be usable',
                'singing_vocal': 'Singing/vocal music - not suitable for speech models',
                'emotional': 'Emotional speech - may affect voice consistency',
                'conversation': 'Multiple speakers - extract individual voices',
                'other': 'Non-speech content or processing errors'
            },
            'file_details': self.metadata
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nMetadata saved to: {metadata_file}")
    
    def print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*80}")
        print("VOICE MODEL AUDIO CATEGORIZATION SUMMARY")
        print('='*80)
        
        for category, count in self.stats.items():
            folder_name = self.folders[category].name
            print(f"{folder_name:40s}: {count:3d} files")
        
        print(f"\nTotal files processed: {sum(self.stats.values())}")
        print(f"Output directory: {self.output_base_dir}")
        
        # Recommendations
        excellent_count = self.stats['excellent']
        good_count = self.stats['good']
        
        print(f"\n📊 RECOMMENDATIONS:")
        print(f"✅ START WITH: {excellent_count} excellent files")
        print(f"✅ THEN ADD: {good_count} good files")
        print(f"⚠️  REVIEW MANUALLY: {self.stats['acceptable']} acceptable files")
        print(f"🎵 BACKGROUND MUSIC: {self.stats['background_music']} files (may be usable)")

def main():
    parser = argparse.ArgumentParser(description="Organize audio files for voice model training")
    parser.add_argument("input_dir", help="Directory containing MP3 files")
    parser.add_argument("--output_dir", default="voice_model_data",
                       help="Base output directory (default: voice_model_data)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    print("🎙️ VOICE MODEL AUDIO ORGANIZER")
    print("="*50)
    print("This script will organize your audio files into folders")
    print("optimized for voice model training quality.")
    print("="*50)
    
    filter_system = VoiceModelFilter(args.input_dir, args.output_dir)
    filter_system.process_files()

if __name__ == "__main__":
    main()
