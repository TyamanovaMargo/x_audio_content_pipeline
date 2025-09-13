#!/usr/bin/env python3
"""
Исправленный Video Voice Filter с правильным анализом голоса
"""
import argparse
import json
import pandas as pd
import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# Аудио библиотеки
try:
    import yt_dlp
    import numpy as np
    from pydub import AudioSegment
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing audio libraries: {e}")
    AUDIO_LIBS_AVAILABLE = False

class EnhancedVoiceDetector:
    """Улучшенный детектор речи с множественными методами"""
    
    def __init__(self, config_path: str = "config.json"):
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.voice_threshold = self.config.get('voice_threshold', 0.6)
        self.logger.info("✅ EnhancedVoiceDetector initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'voice_threshold': 0.6,
                'min_speech_duration': 2.0,
                'energy_threshold_ratio': 0.3
            }
    
    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def detect_audio_content(self, links: List[Dict]) -> List[Dict]:
        """Детекция голоса с улучшенными алгоритмами"""
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.error("❌ Required audio libraries not available")
            return []
            
        results = []
        for link in links:
            url = link.get("url")
            username = link.get("username", "unknown")
            platform = link.get("platform_type", "unknown")
            
            self.logger.info(f"🎤 Analyzing {username} ({platform})")
            
            try:
                voice_result = self._analyze_video_audio(url, username)
                result = {
                    "username": username,
                    "url": url,
                    "platform_type": platform,
                    **voice_result
                }
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ Error analyzing {url}: {e}")
                # Добавляем результат с ошибкой
                results.append({
                    "username": username,
                    "url": url,
                    "platform_type": platform,
                    "voice_detected": False,
                    "voice_probability": 0.0,
                    "audio_confidence": "error",
                    "detection_method": "error",
                    "error": str(e)
                })
        
        return results
    
    def _analyze_video_audio(self, url: str, username: str) -> Dict:
        """Анализ аудио из видео с множественными методами"""
        
        # ✅ ИСПРАВЛЕННЫЕ опции yt-dlp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_path = tmpfile.name
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_path.replace('.wav', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'retries': 2,
                # ✅ Извлекаем только первые 60 секунд для анализа
                'download_sections': '*0-60'  
            }
            
            self.logger.info(f"📥 Downloading audio sample from {url[:50]}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Найти скачанный файл
            audio_file = None
            for ext in ['.wav', '.m4a', '.webm', '.mp3', '.ogg']:
                candidate = temp_path.replace('.wav', ext)
                if os.path.exists(candidate):
                    audio_file = candidate
                    break
            
            if not audio_file or not os.path.exists(audio_file):
                raise Exception("Audio file not downloaded")
            
            # ✅ УЛУЧШЕННЫЙ анализ голоса
            return self._multi_method_voice_detection(audio_file, username)
            
        finally:
            # Очистка временных файлов
            for ext in ['.wav', '.m4a', '.webm', '.mp3', '.ogg']:
                candidate = temp_path.replace('.wav', ext)
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                except:
                    pass
    
    def _multi_method_voice_detection(self, audio_file: str, username: str) -> Dict:
        """Множественные методы детекции голоса"""
        
        # Метод 1: Энергетический анализ (улучшенный)
        energy_result = self._energy_based_detection(audio_file)
        
        # Метод 2: Спектральный анализ
        spectral_result = self._spectral_voice_detection(audio_file)
        
        # Метод 3: Частотный анализ
        frequency_result = self._frequency_analysis(audio_file)
        
        # ✅ Объединение результатов с весами
        methods = [
            (energy_result, 0.3, "energy"),
            (spectral_result, 0.4, "spectral"), 
            (frequency_result, 0.3, "frequency")
        ]
        
        final_probability = 0.0
        confidence_scores = []
        used_methods = []
        
        for result, weight, method_name in methods:
            if result['success']:
                final_probability += result['probability'] * weight
                confidence_scores.append(result['confidence'])
                used_methods.append(method_name)
        
        # Определение финальной уверенности
        if len(confidence_scores) >= 2:
            avg_confidence = np.mean(confidence_scores)
            if avg_confidence > 0.7:
                audio_confidence = "high"
            elif avg_confidence > 0.4:
                audio_confidence = "medium" 
            else:
                audio_confidence = "low"
        else:
            audio_confidence = "low"
        
        voice_detected = final_probability >= self.voice_threshold
        
        self.logger.info(f"📊 {username}: prob={final_probability:.3f}, conf={audio_confidence}")
        
        return {
            "voice_detected": voice_detected,
            "voice_probability": final_probability,
            "audio_confidence": audio_confidence,
            "detection_method": "+".join(used_methods),
            "has_voice_content": voice_detected,
            "method_details": {
                "energy": energy_result,
                "spectral": spectral_result,
                "frequency": frequency_result
            }
        }
    
    def _energy_based_detection(self, audio_file: str) -> Dict:
        """Улучшенная энергетическая детекция"""
        try:
            # Загружаем аудио с нормализацией
            y, sr = librosa.load(audio_file, sr=16000, duration=60)
            
            # Разделяем на фреймы
            frame_length = int(0.025 * sr)  # 25ms фреймы
            hop_length = int(0.01 * sr)     # 10ms шаг
            
            frames = librosa.util.frame(y, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            
            # ✅ Вычисляем энергию для каждого фрейма
            energy = np.sum(frames ** 2, axis=0)
            
            # ✅ Адаптивный порог
            max_energy = np.max(energy)
            mean_energy = np.mean(energy)
            threshold = mean_energy + 0.3 * (max_energy - mean_energy)
            
            # Определяем речевые сегменты
            speech_frames = energy > threshold
            speech_ratio = np.sum(speech_frames) / len(speech_frames)
            
            # ✅ Проверяем непрерывность речи
            speech_segments = self._find_continuous_segments(speech_frames)
            long_segments = [seg for seg in speech_segments if seg[1] - seg[0] > sr * 0.5]  # >0.5 сек
            
            probability = min(1.0, speech_ratio * 2)  # Нормализация
            confidence = len(long_segments) / max(1, len(speech_segments))
            
            return {
                'success': True,
                'probability': probability,
                'confidence': confidence,
                'speech_ratio': speech_ratio,
                'segments_count': len(speech_segments)
            }
            
        except Exception as e:
            self.logger.error(f"Energy detection failed: {e}")
            return {'success': False, 'probability': 0.0, 'confidence': 0.0}
    
    def _spectral_voice_detection(self, audio_file: str) -> Dict:
        """Спектральный анализ для детекции голоса"""
        try:
            y, sr = librosa.load(audio_file, sr=16000, duration=60)
            
            # ✅ Извлекаем спектральные признаки
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # ✅ Анализ характеристик человеческой речи
            # Человеческая речь: spectral centroid 200-4000 Hz
            voice_centroids = spectral_centroids[(spectral_centroids > 200) & 
                                                (spectral_centroids < 4000)]
            voice_ratio = len(voice_centroids) / len(spectral_centroids)
            
            # ZCR для речи обычно 50-200 Гц
            voice_zcr = zero_crossing_rate[(zero_crossing_rate > 0.01) & 
                                          (zero_crossing_rate < 0.3)]
            zcr_ratio = len(voice_zcr) / len(zero_crossing_rate)
            
            # MFCC дисперсия (речь имеет больше вариации)
            mfcc_variance = np.var(mfccs, axis=1).mean()
            
            # Комбинированная оценка
            probability = (voice_ratio * 0.4 + zcr_ratio * 0.3 + 
                          min(1.0, mfcc_variance / 10) * 0.3)
            
            confidence = (voice_ratio + zcr_ratio) / 2
            
            return {
                'success': True,
                'probability': probability,
                'confidence': confidence,
                'voice_ratio': voice_ratio,
                'zcr_ratio': zcr_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Spectral detection failed: {e}")
            return {'success': False, 'probability': 0.0, 'confidence': 0.0}
    
    def _frequency_analysis(self, audio_file: str) -> Dict:
        """Частотный анализ голоса"""
        try:
            y, sr = librosa.load(audio_file, sr=16000, duration=60)
            
            # ✅ Анализ частотного спектра
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            
            # Человеческая речь: 80-1000 Hz (основные частоты), 1000-4000 Hz (форманты)
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Энергия в речевом диапазоне
            speech_freq_mask = (freqs >= 80) & (freqs <= 4000)
            speech_energy = np.mean(D[speech_freq_mask, :])
            
            # Энергия в музыкальном диапазоне (выше 4000 Hz)
            music_freq_mask = freqs > 4000
            music_energy = np.mean(D[music_freq_mask, :])
            
            # Отношение речевой энергии к общей
            total_energy = np.mean(D)
            if total_energy > 0:
                speech_ratio = speech_energy / total_energy
                probability = min(1.0, max(0.0, (speech_ratio - 0.3) * 2))  # Нормализация
            else:
                probability = 0.0
            
            confidence = min(1.0, abs(speech_energy - music_energy) / 40)  # Различимость
            
            return {
                'success': True,
                'probability': probability,
                'confidence': confidence,
                'speech_energy': float(speech_energy),
                'music_energy': float(music_energy)
            }
            
        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {e}")
            return {'success': False, 'probability': 0.0, 'confidence': 0.0}
    
    def _find_continuous_segments(self, binary_array: np.ndarray) -> List[tuple]:
        """Находит непрерывные сегменты True в булевом массиве"""
        segments = []
        start = None
        
        for i, val in enumerate(binary_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(binary_array)))
        
        return segments


class ChannelVideoFetcher:
    """Получение последних видео из каналов"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_latest_video(self, channel_url: str, platform: str) -> Optional[str]:
        """Получить ссылку на последнее видео из канала"""
        
        try:
            if platform == 'youtube':
                return self._get_youtube_latest(channel_url)
            elif platform == 'twitch':
                return self._get_twitch_latest(channel_url)
            elif platform == 'tiktok':
                return self._get_tiktok_latest(channel_url)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching from {channel_url}: {e}")
            return None
    
    def _get_youtube_latest(self, channel_url: str) -> Optional[str]:
        """Получить последнее видео с YouTube канала"""
        
        # Конвертируем в URL видео канала
        if not channel_url.endswith('/videos'):
            if '/@' in channel_url:
                channel_url = channel_url.split('?')[0].rstrip('/') + '/videos'
            elif '/channel/' in channel_url or '/c/' in channel_url:
                channel_url = channel_url.rstrip('/') + '/videos'
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': 1  # Только первое видео
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(channel_url, download=False)
                
                if playlist_info and 'entries' in playlist_info:
                    first_entry = next(iter(playlist_info['entries']), None)
                    if first_entry and first_entry.get('id'):
                        return f"https://www.youtube.com/watch?v={first_entry['id']}"
                        
        except Exception as e:
            self.logger.error(f"YouTube fetch error: {e}")
        
        return None
    
    def _get_twitch_latest(self, channel_url: str) -> Optional[str]:
        """Получить последний VOD с Twitch канала"""
        
        if '/videos' not in channel_url:
            channel_url = channel_url.rstrip('/') + '/videos'
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': 1
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(channel_url, download=False)
                
                if playlist_info and 'entries' in playlist_info:
                    first_entry = next(iter(playlist_info['entries']), None)
                    if first_entry and first_entry.get('url'):
                        return first_entry['url']
                        
        except Exception as e:
            self.logger.error(f"Twitch fetch error: {e}")
        
        return None
    
    def _get_tiktok_latest(self, channel_url: str) -> Optional[str]:
        """Получить последнее видео с TikTok пользователя"""
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': 1
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(channel_url, download=False)
                
                if playlist_info and 'entries' in playlist_info:
                    first_entry = next(iter(playlist_info['entries']), None)
                    if first_entry and first_entry.get('url'):
                        return first_entry['url']
                        
        except Exception as e:
            self.logger.error(f"TikTok fetch error: {e}")
        
        return None


class VideoVoiceFilter:
    """Основной класс фильтрации"""
    
    def __init__(self, config_path: str = "config.json"):
        self._setup_logging()
        self.voice_detector = EnhancedVoiceDetector(config_path)
        self.video_fetcher = ChannelVideoFetcher()
        self.min_voice_probability = 0.6
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_channels_from_csv(self, input_file: str) -> List[Dict]:
        """Загрузка каналов из CSV"""
        try:
            df = pd.read_csv(input_file)
            channels = []
            
            for _, row in df.iterrows():
                if pd.isna(row.get("url")) or pd.isna(row.get("platform_type")):
                    continue
                
                platform = str(row["platform_type"]).lower()
                if platform not in ["youtube", "twitch", "tiktok"]:
                    continue
                
                url = str(row["url"])
                url_type = str(row.get("url_type", "unknown")).lower()
                
                channels.append({
                    "original_url": url,
                    "platform": platform,
                    "username": str(row.get("username", "unknown")),
                    "title": str(row.get("title", "unknown")),
                    "url_type": url_type,
                    "is_direct_video": url_type == "video"
                })
            
            return channels
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return []
    
    def process_channels(self, channels: List[Dict]) -> List[Dict]:
        """Обработка каналов"""
        
        print(f"\n🎬 Processing {len(channels)} items...")
        voice_items = []
        
        for i, ch in enumerate(channels, 1):
            print(f"\n📺 [{i}/{len(channels)}] {ch['username']} ({ch['platform'].upper()})")
            
            # Определяем URL для анализа
            if ch['is_direct_video']:
                video_url = ch['original_url']
                print(f"🎥 Direct video: {ch['title'][:50]}...")
            else:
                print("🔍 Fetching latest video from channel...")
                video_url = self.video_fetcher.get_latest_video(
                    ch['original_url'], 
                    ch['platform']
                )
                
                if not video_url:
                    print("❌ No video found or channel inaccessible")
                    continue
                    
                print(f"📹 Found latest video: {video_url[:60]}...")
            
            # Анализ голоса
            detection_data = [{
                "url": video_url,
                "platform_type": ch["platform"],
                "username": ch["username"]
            }]
            
            results = self.voice_detector.detect_audio_content(detection_data)
            
            if results and self._meets_voice_criteria(results[0]):
                result = results[0]
                voice_item = {
                    **ch,
                    "video_url": video_url,
                    "voice_probability": result.get("voice_probability", 0.0),
                    "audio_confidence": result.get("audio_confidence", "low"),
                    "detection_method": result.get("detection_method", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
                
                voice_items.append(voice_item)
                print(f"✅ VOICE DETECTED! Probability: {result.get('voice_probability', 0):.3f}")
            else:
                print("❌ No voice detected or criteria not met")
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n📊 RESULTS: {len(voice_items)}/{len(channels)} channels have voice content")
        return voice_items
    
    def _meets_voice_criteria(self, result: Dict) -> bool:
        """Проверка критериев голоса"""
        if result.get("audio_confidence") == "error":
            return False
            
        voice_prob = result.get("voice_probability", 0.0)
        confidence = result.get("audio_confidence", "low")
        
        return (voice_prob >= self.min_voice_probability and 
                confidence in ["high", "medium"])
    
    def save_results(self, voice_items: List[Dict], output_file: str):
        """Сохранение результатов"""
        try:
            if voice_items:
                df = pd.DataFrame(voice_items)
                df.to_csv(output_file, index=False)
                print(f"✅ Results saved to {output_file}")
            else:
                print("❌ No voice items found")
        except Exception as e:
            self.logger.error(f"Error saving: {e}")


def main():
    if not AUDIO_LIBS_AVAILABLE:
        print("❌ Install required packages: pip install yt-dlp librosa pydub numpy")
        return
    
    parser = argparse.ArgumentParser(description="Filter channels by voice content")
    parser.add_argument("input_file", help="CSV file")
    parser.add_argument("-o", "--output", default="voice_results.csv", help="Output CSV")
    
    args = parser.parse_args()
    
    filter_tool = VideoVoiceFilter()
    channels = filter_tool.load_channels_from_csv(args.input_file)
    
    if not channels:
        print("❌ No valid channels loaded")
        return
    
    voice_items = filter_tool.process_channels(channels)
    filter_tool.save_results(voice_items, args.output)

if __name__ == "__main__":
    main()
