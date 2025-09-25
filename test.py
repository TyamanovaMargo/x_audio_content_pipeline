#!/usr/bin/env python3
"""
Robust Conservative Voice Detector - Fixed Harmonic Analysis + Additional Features
Uses multiple independent methods to detect music even when HPSS fails
"""

import os
import shutil
import argparse
import logging
import warnings
from glob import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Audio processing
from pydub import AudioSegment
import librosa
import soundfile as sf
import torch
import whisper
from pyannote.audio import Pipeline
from scipy import signal
from scipy.stats import entropy

# Silence warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voice_detection.log')
    ]
)
logger = logging.getLogger(__name__)


class RobustConservativeMusicDetector:
    """Robust conservative detector with multiple fallback methods"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        logger.info("🎵 Initialized robust conservative music detector")
    
    def extract_robust_features(self, audio: np.ndarray) -> Dict:
        """Extract features with multiple fallback methods"""
        
        if len(audio) == 0:
            return self._get_default_features()
        
        features = {}
        
        try:
            # Method 1: Multiple HPSS approaches
            features.update(self._extract_multiple_hpss_features(audio))
            
            # Method 2: Direct spectral features (HPSS-independent)
            features.update(self._extract_direct_spectral_features(audio))
            
            # Method 3: Pitch-based features (independent)
            features.update(self._extract_pitch_features(audio))
            
            # Method 4: Rhythm features (independent)
            features.update(self._extract_rhythm_features(audio))
            
            # Method 5: Frequency distribution (independent)
            features.update(self._extract_frequency_features(audio))
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting robust features: {e}")
            return self._get_default_features()
    
    def _extract_multiple_hpss_features(self, audio: np.ndarray) -> Dict:
        """Try multiple HPSS settings and take the most reliable result"""
        features = {}
        
        # Try different margin values
        margins = [4.0, 8.0, 12.0]
        hpss_results = []
        
        for margin in margins:
            try:
                harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
                harmonic_energy = np.sum(harmonic ** 2)
                total_energy = np.sum(audio ** 2)
                
                if total_energy > 0:
                    harmonic_ratio = harmonic_energy / total_energy
                    hpss_results.append({
                        'margin': margin,
                        'harmonic_ratio': harmonic_ratio,
                        'harmonic_energy': harmonic_energy,
                        'total_energy': total_energy
                    })
            except Exception as e:
                continue
        
        if hpss_results:
            # Choose the result with highest harmonic ratio (most sensitive to music)
            best_result = max(hpss_results, key=lambda x: x['harmonic_ratio'])
            features['harmonic_ratio'] = float(best_result['harmonic_ratio'])
            features['best_hpss_margin'] = float(best_result['margin'])
            
            # Also store the range of results
            ratios = [r['harmonic_ratio'] for r in hpss_results]
            features['harmonic_ratio_max'] = float(max(ratios))
            features['harmonic_ratio_min'] = float(min(ratios))
            features['harmonic_ratio_std'] = float(np.std(ratios))
        else:
            features.update({
                'harmonic_ratio': 0.0, 'best_hpss_margin': 8.0,
                'harmonic_ratio_max': 0.0, 'harmonic_ratio_min': 0.0, 'harmonic_ratio_std': 0.0
            })
        
        # Additional harmonic analysis
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            if chroma.size > 0:
                chroma_mean = np.mean(chroma, axis=1)
                features['chroma_max'] = float(np.max(chroma_mean))
                features['tonal_strength'] = float(np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-10))
            else:
                features['chroma_max'] = 0.0
                features['tonal_strength'] = 1.0
        except:
            features['chroma_max'] = 0.0
            features['tonal_strength'] = 1.0
        
        return features
    
    def _extract_direct_spectral_features(self, audio: np.ndarray) -> Dict:
        """Direct spectral analysis without HPSS dependency"""
        features = {}
        
        try:
            # Spectral contrast - key music indicator
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            features['spectral_contrast_std'] = float(np.std(contrast))
            features['spectral_contrast_max'] = float(np.max(contrast))
            
            # Spectral centroid analysis
            centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(centroids))
            features['spectral_centroid_std'] = float(np.std(centroids))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(rolloff))
            features['spectral_rolloff_std'] = float(np.std(rolloff))
            
            # Spectral flatness (tonality indicator)
            flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['spectral_flatness_mean'] = float(np.mean(flatness))
            
            # Bandwidth analysis
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
            
        except Exception as e:
            features.update({
                'spectral_contrast_mean': 0.0, 'spectral_contrast_std': 0.0, 'spectral_contrast_max': 0.0,
                'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
                'spectral_flatness_mean': 0.0, 'spectral_bandwidth_mean': 0.0
            })
        
        return features
    
    def _extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """Pitch analysis independent of HPSS"""
        features = {}
        
        try:
            # Use pyin for robust pitch detection
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=self.sr, threshold=0.1
            )
            
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 10:
                features['f0_mean'] = float(np.mean(f0_clean))
                features['f0_std'] = float(np.std(f0_clean))
                features['f0_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                features['voiced_ratio'] = float(np.sum(~np.isnan(f0)) / len(f0))
                
                # Pitch stability
                if features['f0_std'] > 0 and features['f0_mean'] > 0:
                    features['pitch_stability'] = float(1.0 / (1.0 + features['f0_std'] / features['f0_mean']))
                else:
                    features['pitch_stability'] = 1.0
                
                # Pitch contour smoothness (music tends to be smoother)
                if len(f0_clean) > 20:
                    f0_diff = np.abs(np.diff(f0_clean))
                    features['pitch_smoothness'] = float(1.0 / (1.0 + np.mean(f0_diff)))
                else:
                    features['pitch_smoothness'] = 0.0
            else:
                features.update({
                    'f0_mean': 0.0, 'f0_std': 0.0, 'f0_range': 0.0,
                    'voiced_ratio': 0.0, 'pitch_stability': 0.0, 'pitch_smoothness': 0.0
                })
        except:
            features.update({
                'f0_mean': 0.0, 'f0_std': 0.0, 'f0_range': 0.0,
                'voiced_ratio': 0.0, 'pitch_stability': 0.0, 'pitch_smoothness': 0.0
            })
        
        return features
    
    def _extract_rhythm_features(self, audio: np.ndarray) -> Dict:
        """Rhythm analysis independent of other methods"""
        features = {}
        
        try:
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr, trim=True)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            if len(beats) > 5:
                beat_times = librosa.frames_to_time(beats, sr=self.sr)
                beat_intervals = np.diff(beat_times)
                if len(beat_intervals) > 0 and np.mean(beat_intervals) > 0:
                    cv = np.std(beat_intervals) / np.mean(beat_intervals)
                    features['beat_regularity'] = float(1.0 / (1.0 + cv))
                else:
                    features['beat_regularity'] = 0.0
                
                # Beat strength
                features['beat_density'] = float(len(beats) / (len(audio) / self.sr))
            else:
                features['beat_regularity'] = 0.0
                features['beat_density'] = 0.0
            
            # Onset analysis
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr, threshold=0.7)
            features['onset_density'] = float(len(onset_frames) / (len(audio) / self.sr))
            
        except:
            features.update({
                'tempo': 0.0, 'beat_count': 0, 'beat_regularity': 0.0,
                'beat_density': 0.0, 'onset_density': 0.0
            })
        
        return features
    
    def _extract_frequency_features(self, audio: np.ndarray) -> Dict:
        """Frequency distribution analysis"""
        features = {}
        
        try:
            # Compute power spectrum
            S = np.abs(librosa.stft(audio)) ** 2
            freqs = librosa.fft_frequencies(sr=self.sr)
            power_spectrum = np.mean(S, axis=1)
            
            # Frequency band analysis
            total_power = np.sum(power_spectrum)
            
            if total_power > 0:
                # Speech-focused bands
                speech_band = (freqs >= 300) & (freqs <= 3400)
                features['speech_band_ratio'] = float(np.sum(power_spectrum[speech_band]) / total_power)
                
                # Music-focused bands
                low_band = (freqs >= 80) & (freqs < 300)   # Bass instruments
                mid_band = (freqs >= 1000) & (freqs < 4000)  # Melody range
                high_band = (freqs > 4000) & (freqs <= 8000)  # Harmonics
                
                features['music_low_ratio'] = float(np.sum(power_spectrum[low_band]) / total_power)
                features['music_mid_ratio'] = float(np.sum(power_spectrum[mid_band]) / total_power)
                features['music_high_ratio'] = float(np.sum(power_spectrum[high_band]) / total_power)
                
                # Overall frequency distribution
                features['frequency_entropy'] = float(entropy(power_spectrum + 1e-10))
            else:
                features.update({
                    'speech_band_ratio': 0.0, 'music_low_ratio': 0.0,
                    'music_mid_ratio': 0.0, 'music_high_ratio': 0.0, 'frequency_entropy': 0.0
                })
                
        except:
            features.update({
                'speech_band_ratio': 0.0, 'music_low_ratio': 0.0,
                'music_mid_ratio': 0.0, 'music_high_ratio': 0.0, 'frequency_entropy': 0.0
            })
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Default features"""
        return {
            # HPSS features
            'harmonic_ratio': 0.0, 'best_hpss_margin': 8.0,
            'harmonic_ratio_max': 0.0, 'harmonic_ratio_min': 0.0, 'harmonic_ratio_std': 0.0,
            'chroma_max': 0.0, 'tonal_strength': 1.0,
            
            # Spectral features  
            'spectral_contrast_mean': 0.0, 'spectral_contrast_std': 0.0, 'spectral_contrast_max': 0.0,
            'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
            'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
            'spectral_flatness_mean': 0.0, 'spectral_bandwidth_mean': 0.0,
            
            # Pitch features
            'f0_mean': 0.0, 'f0_std': 0.0, 'f0_range': 0.0,
            'voiced_ratio': 0.0, 'pitch_stability': 0.0, 'pitch_smoothness': 0.0,
            
            # Rhythm features
            'tempo': 0.0, 'beat_count': 0, 'beat_regularity': 0.0,
            'beat_density': 0.0, 'onset_density': 0.0,
            
            # Frequency features
            'speech_band_ratio': 0.0, 'music_low_ratio': 0.0,
            'music_mid_ratio': 0.0, 'music_high_ratio': 0.0, 'frequency_entropy': 0.0
        }
    
    def classify_robust_conservative(self, features: Dict) -> Dict:
        """
        Robust conservative classification using multiple evidence sources
        """
        
        music_evidence = []
        music_score = 0.0
        
        # Evidence 1: HIGH spectral contrast (clear music production)
        spectral_contrast = features.get('spectral_contrast_mean', 0)
        spectral_contrast_max = features.get('spectral_contrast_max', 0)
        
        if spectral_contrast > 30.0 or spectral_contrast_max > 35.0:
            music_evidence.append(f"Very high spectral contrast ({spectral_contrast:.1f})")
            music_score += 3.0
        elif spectral_contrast > 25.0:
            music_evidence.append(f"High spectral contrast ({spectral_contrast:.1f})")
            music_score += 2.0
        
        # Evidence 2: HIGH harmonic ratio (using best HPSS result)
        harmonic_ratio_max = features.get('harmonic_ratio_max', 0)
        harmonic_ratio = features.get('harmonic_ratio', 0)
        
        # Use the maximum harmonic ratio from multiple HPSS attempts
        best_harmonic = max(harmonic_ratio_max, harmonic_ratio)
        
        if best_harmonic > 0.85:
            music_evidence.append(f"Very high harmonic content ({best_harmonic:.2f})")
            music_score += 3.0
        elif best_harmonic > 0.7:
            music_evidence.append(f"High harmonic content ({best_harmonic:.2f})")
            music_score += 2.0
        elif best_harmonic > 0.5:
            music_score += 1.0
        
        # Evidence 3: Strong tonal structure
        tonal_strength = features.get('tonal_strength', 1.0)
        chroma_max = features.get('chroma_max', 0)
        
        if tonal_strength > 4.0 and chroma_max > 0.4:
            music_evidence.append(f"Strong tonal structure ({tonal_strength:.1f}, {chroma_max:.2f})")
            music_score += 2.5
        elif tonal_strength > 3.0:
            music_score += 1.0
        
        # Evidence 4: Musical rhythm pattern
        beat_regularity = features.get('beat_regularity', 0)
        tempo = features.get('tempo', 0)
        
        if beat_regularity > 0.9 and 80 <= tempo <= 180:
            music_evidence.append(f"Strong musical rhythm ({beat_regularity:.2f}, {tempo:.0f} BPM)")
            music_score += 2.0
        elif beat_regularity > 0.8 and tempo > 0:
            music_score += 1.0
        
        # Evidence 5: Musical frequency distribution
        music_low = features.get('music_low_ratio', 0)
        music_high = features.get('music_high_ratio', 0) 
        speech_band = features.get('speech_band_ratio', 0)
        
        if speech_band < 0.4 and (music_low > 0.2 or music_high > 0.15):
            music_evidence.append(f"Musical frequency distribution (speech: {speech_band:.2f})")
            music_score += 1.5
        
        # Evidence 6: Very stable/smooth pitch (sustained musical notes)
        pitch_stability = features.get('pitch_stability', 0)
        pitch_smoothness = features.get('pitch_smoothness', 0)
        voiced_ratio = features.get('voiced_ratio', 0)
        
        if pitch_stability > 0.85 and pitch_smoothness > 0.7 and voiced_ratio > 0.7:
            music_evidence.append(f"Very stable musical pitch ({pitch_stability:.2f})")
            music_score += 2.0
        elif pitch_stability > 0.8 and voiced_ratio > 0.6:
            music_score += 1.0
        
        # CONSERVATIVE DECISION LOGIC
        is_music = False
        confidence = 0.0
        
        # Primary: Very high score with multiple evidence
        if music_score >= 6.0 and len(music_evidence) >= 2:
            is_music = True
            confidence = min(music_score / 10.0, 1.0)
        
        # Secondary: Extreme single indicator
        elif best_harmonic > 0.90 or spectral_contrast > 35.0:
            is_music = True  
            confidence = 0.85
            if best_harmonic > 0.90:
                music_evidence.append(f"Extreme harmonic ratio ({best_harmonic:.2f})")
            
        # Tertiary: High harmonic + spectral together
        elif best_harmonic > 0.8 and spectral_contrast > 25.0:
            is_music = True
            confidence = 0.75
            music_evidence.append(f"Combined harmonic+spectral evidence")
        
        reasoning = " | ".join(music_evidence[:3]) if music_evidence else "All indicators suggest speech"
        
        return {
            'is_music': is_music,
            'confidence': float(confidence),
            'music_score': float(music_score),
            'evidence_count': len(music_evidence),
            'reasoning': reasoning,
            'debug_features': {
                'spectral_contrast': spectral_contrast,
                'harmonic_ratio_best': best_harmonic,
                'harmonic_ratio_original': harmonic_ratio,
                'tonal_strength': tonal_strength,
                'beat_regularity': beat_regularity,
                'tempo': tempo,
                'speech_band_ratio': speech_band,
                'pitch_stability': pitch_stability
            }
        }


# Остальной код класса детектора остается таким же, только заменяем:
# self.music_detector = RobustConservativeMusicDetector()
# conservative_features = self.music_detector.extract_robust_features(audio_sample)
# robust_classification = self.music_detector.classify_robust_conservative(conservative_features)

def main():
    parser = argparse.ArgumentParser(
        description="Robust Conservative Voice Detector - Fixed HPSS + Fallback Methods"
    )
    parser.add_argument("--source", "-s", required=True)
    parser.add_argument("--dest", "-d", required=True)  
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--min-duration", "-md", type=float, default=5.0)
    parser.add_argument("--huggingface-token", "-hf", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save-csv", "-csv", action="store_true")
    
    args = parser.parse_args()
    
    print("🎤 Robust Conservative Voice Detector")
    print("🔧 Fixed HPSS parameters + Multiple fallback methods")
    print("🛡️ Ultra-conservative music detection")
    print("📊 Shows both original and max harmonic ratios")

if __name__ == "__main__":
    main()
