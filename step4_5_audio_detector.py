import requests
import re
from typing import List, Dict
from urllib.parse import urlparse
import time

class AudioContentDetector:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AudioBot/1.0)'
        })

    def detect_audio_content(self, audio_links: List[Dict]) -> List[Dict]:
        if not audio_links:
            print("🔍 No audio links to detect")
            return []
        print(f"🎵 Starting YouTube, Twitch & TikTok audio detection for {len(audio_links)} links...")
        audio_detected_links = []
        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')
            platform = link_data.get('platform_type', 'unknown')
            if not url:
                continue
            print(f"🔍 [{i}/{len(audio_links)}] {platform.upper()} check: {url[:50]}...")
            if platform == 'youtube':
                audio_result = self._detect_youtube_audio(url)
            elif platform == 'twitch':
                audio_result = self._detect_twitch_audio(url)
            elif platform == 'tiktok':
                audio_result = self._detect_tiktok_audio(url)
            else:
                continue
            link_data.update({
                'has_audio': audio_result['has_audio'],
                'audio_confidence': audio_result['confidence'],
                'audio_type': audio_result.get('audio_type'),
                'detection_status': audio_result['status']
            })
            if audio_result['has_audio']:
                audio_detected_links.append(link_data)
            time.sleep(0.5)
        return audio_detected_links

    def _detect_tiktok_audio(self, url: str) -> dict:
        # TikTok videos почти всегда содержат аудио-трек (музыка/голос)
        try:
            # Можно ещё добавить простейший GET и проверку по содержимому, если хочется подробности
            return {
                'has_audio': True,
                'confidence': 'high',
                'audio_type': 'tiktok_default',
                'status': 'tiktok_audio_assumed'
            }
        except Exception as e:
            return {
                'has_audio': True,
                'confidence': 'medium',
                'audio_type': 'tiktok_default',
                'status': f'tiktok_error_assumed_audio: {str(e)}'
            }
            
    def _detect_youtube_audio(self, url: str) -> Dict:
        """Enhanced YouTube audio detection"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # YouTube аудио-индикаторы
            strong_audio_indicators = [
                '"hasaudio":true',
                '"audiotrack"',
                'itag.*?audio',
                'audio/.*?webm',
                'audio/.*?mp4'
            ]
            
            found_indicators = []
            for pattern in strong_audio_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    found_indicators.append(pattern)
            
            # Дополнительные проверки
            has_video_element = '<video' in content or 'video' in content
            has_audio_mention = 'audio' in content and ('track' in content or 'stream' in content)
            
            # Определяем тип контента (важно для голосовой проверки)
            content_type = self._classify_youtube_content(content)
            
            # Логика принятия решения
            if len(found_indicators) >= 2:
                confidence = 'high'
                has_audio = True
            elif len(found_indicators) >= 1 or (has_video_element and has_audio_mention):
                confidence = 'medium'
                has_audio = True
            elif has_video_element:  # YouTube видео обычно имеют аудио
                confidence = 'medium'
                has_audio = True
            else:
                confidence = 'low'
                has_audio = False
                
            return {
                'has_audio': has_audio,
                'confidence': confidence,
                'audio_type': content_type,
                'status': f'youtube_audio: {len(found_indicators)} indicators, {content_type}'
            }
            
        except Exception as e:
            return {
                'has_audio': True,  # Предполагаем наличие аудио в YouTube по умолчанию
                'confidence': 'medium',
                'audio_type': 'youtube_default',
                'status': f'youtube_error_default_true: {str(e)}'
            }

    def _detect_twitch_audio(self, url: str) -> Dict:
        """Enhanced Twitch audio detection"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # Определяем тип Twitch контента
            stream_type = self._classify_twitch_content(url, content)
            
            # Twitch почти всегда имеет аудио
            if stream_type == 'just_chatting':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'live_talk',
                    'status': 'twitch_just_chatting_high_voice_probability'
                }
            elif stream_type == 'talk_show':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'talk_show',
                    'status': 'twitch_talk_show'
                }
            elif stream_type == 'gaming_with_commentary':
                return {
                    'has_audio': True,
                    'confidence': 'high',
                    'audio_type': 'gaming_commentary',
                    'status': 'twitch_gaming_with_voice'
                }
            else:  # Любой другой Twitch контент
                return {
                    'has_audio': True,
                    'confidence': 'medium',
                    'audio_type': 'twitch_stream',
                    'status': 'twitch_default_stream'
                }
                
        except Exception as e:
            return {
                'has_audio': True,  # Twitch почти всегда имеет аудио
                'confidence': 'high',
                'audio_type': 'twitch_default',
                'status': f'twitch_error_default_true: {str(e)}'
            }

    def _classify_youtube_content(self, content: str) -> str:
        """Classify YouTube content type for better voice detection"""
        
        # Высокая вероятность голосового контента
        if any(keyword in content for keyword in [
            'podcast', 'interview', 'talk', 'discussion', 'conversation'
        ]):
            return 'speech_content'
        
        # Образовательный контент (часто голосовой)
        elif any(keyword in content for keyword in [
            'tutorial', 'lecture', 'explanation', 'review', 'analysis'
        ]):
            return 'educational_content'
        
        # Музыкальный контент
        elif any(keyword in content for keyword in [
            'music', 'song', 'album', 'artist', 'band', 'mv', 'official video'
        ]):
            return 'music_content'
        
        # Игровой контент
        elif any(keyword in content for keyword in [
            'gameplay', 'gaming', 'game', 'let\'s play', 'walkthrough'
        ]):
            return 'gaming_content'
        
        else:
            return 'mixed_content'

    def _classify_twitch_content(self, url: str, content: str) -> str:
        """Classify Twitch stream type"""
        
        # Just Chatting - высокая вероятность голоса
        if 'just chatting' in content or 'justchatting' in content:
            return 'just_chatting'
        
        # Talk shows и подкасты
        elif any(keyword in content for keyword in [
            'talk show', 'podcast', 'interview', 'discussion'
        ]):
            return 'talk_show'
        
        # Gaming с комментариями
        elif any(keyword in content for keyword in [
            'gaming', 'gameplay', 'playing'
        ]) and any(keyword in content for keyword in [
            'commentary', 'talking', 'chat'
        ]):
            return 'gaming_with_commentary'
        
        else:
            return 'general_stream'
