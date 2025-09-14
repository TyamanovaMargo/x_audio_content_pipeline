#!/usr/bin/env python3
"""
step7_final_merger.py - Final Results Merger
Combines original input (username, mbti) with Stage 6 voice detection results and audio files
"""

import os
import pandas as pd
import argparse
import sys
from typing import Dict, List
from pathlib import Path
import re

class FinalResultsMerger:
    """Merges original input data with Stage 6 voice detection results"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        
    def extract_username_from_filename(self, filename: str) -> str:
        """Extract username from audio filename (before _audio_part)"""
        if not filename:
            return ""
        
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Extract username (everything before _audio_part)
        if '_audio_part' in name_without_ext:
            return name_without_ext.split('_audio_part')[0]
        else:
            # Fallback: everything before first underscore
            parts = name_without_ext.split('_')
            return parts[0] if parts else ""
    
    def get_stage6_results(self) -> pd.DataFrame:
        """Get Stage 6 voice detection results"""
        # Look for Stage 6 results file in voice_samples directory
        voice_samples_dir = os.path.join(self.output_dir, "voice_samples")
        stage6_files = []
        
        # Check both locations for Stage 6 results
        for search_dir in [self.output_dir, voice_samples_dir]:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.startswith('6_voice_detection_results') and file.endswith('.csv'):
                        stage6_files.append(os.path.join(search_dir, file))
        
        if not stage6_files:
            print("❌ No Stage 6 results found")
            return pd.DataFrame()
        
        # Use the most recent Stage 6 file
        stage6_file = sorted(stage6_files)[-1]
        
        print(f"📊 Loading Stage 6 results from: {os.path.basename(stage6_file)}")
        return pd.read_csv(stage6_file)
    
    def get_processed_audio_files(self) -> List[str]:
        """Get list of successfully processed audio files from Step 6 output directory"""
        # Step 6 saves only files with detected voice to its output directory
        processed_dir = os.path.join(self.output_dir, "voice_samples", "stage6_processed")
        
        if not os.path.exists(processed_dir):
            print("❌ Step 6 output directory not found")
            return []
        
        audio_files = []
        for file in os.listdir(processed_dir):
            if file.endswith(('.mp3', '.wav')):
                audio_files.append(file)
        
        print(f"🎵 Found {len(audio_files)} Step 6 processed audio files (voice detected)")
        return audio_files
    
    def merge_results(self, input_file: str, output_file: str = None) -> str:
        """
        Merge original input with Stage 6 results
        
        Args:
            input_file: Original CSV with username, mbti columns
            output_file: Output CSV path (optional)
        
        Returns:
            Path to merged results file
        """
        
        # Load original input data
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"📥 Loading original input from: {input_file}")
        original_df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['username', 'mbti']
        missing_cols = [col for col in required_cols if col not in original_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"📊 Original data: {len(original_df)} users")
        
        # Get Stage 6 results
        stage6_df = self.get_stage6_results()
        processed_files = self.get_processed_audio_files()
        
        # Create results list
        results = []
        
        for _, row in original_df.iterrows():
            username = row['username'].lower()  # Normalize for matching
            mbti = row['mbti']
            
            # Find matching audio files for this username
            user_files = []
            for audio_file in processed_files:
                file_username = self.extract_username_from_filename(audio_file).lower()
                if file_username == username:
                    user_files.append(audio_file)
            
            # Find Stage 6 analysis results for this user
            voice_detected = len(user_files) > 0  # Default: if has audio files, assume voice detected
            voice_score = 1.0 if voice_detected else 0.0
            transcription = "Audio files found - Stage 6 analysis not available"
            word_count = 0
            
            if not stage6_df.empty:
                # Reset defaults if we have Stage 6 results
                voice_detected = False
                voice_score = 0.0
                transcription = ""
                
                # Look for results matching this username's files
                for audio_file in user_files:
                    file_matches = stage6_df[stage6_df['file_path'].str.contains(audio_file, na=False)]
                    if not file_matches.empty:
                        result = file_matches.iloc[0]
                        if result.get('voice_detected', False):
                            voice_detected = True
                            voice_score = max(voice_score, result.get('voice_score', 0.0))
                            if result.get('transcription'):
                                transcription = result.get('transcription', '')
                                word_count = result.get('word_count', 0)
            
            # Create result entry
            result_entry = {
                'username': row['username'],  # Keep original case
                'mbti': mbti,
                'voice_detected': voice_detected,
                'audio_files_count': len(user_files),
                'audio_files': '; '.join(user_files) if user_files else '',
                'voice_score': voice_score,
                'word_count': word_count,
                'transcription_sample': transcription[:200] + '...' if len(transcription) > 200 else transcription
            }
            
            results.append(result_entry)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"7_final_results_{timestamp}.csv")
        
        # Save results
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✅ Final Results Merger Completed!")
        print(f"📊 Total users: {len(results_df)}")
        print(f"🎵 Users with voice detected: {len(results_df[results_df['voice_detected']])}")
        print(f"🎤 Users with audio files: {len(results_df[results_df['audio_files_count'] > 0])}")
        print(f"📁 Results saved to: {output_file}")
        
        # Show sample results
        print(f"\n📋 Sample Results:")
        for i, row in results_df.head(5).iterrows():
            status = "✅" if row['voice_detected'] else "❌"
            print(f"  {status} {row['username']} ({row['mbti']}) - {row['audio_files_count']} files")
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Merge original input with Stage 6 voice detection results")
    parser.add_argument("input_file", help="Original CSV file with username and mbti columns")
    parser.add_argument("--output", help="Output CSV file path (optional)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    try:
        merger = FinalResultsMerger(args.output_dir)
        result_file = merger.merge_results(args.input_file, args.output)
        print(f"\n🎉 Success! Final results available at: {result_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
