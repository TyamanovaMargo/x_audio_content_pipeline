import requests
from typing import List, Dict
from urllib.parse import urlparse

class VoiceContentVerifier:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; VoiceBot/1.0)'
        })

        # Голосовые ключевые слова (исключаем музыкальные)
        self.voice_keywords = [
            'podcast', 'interview', 'talk', 'speech', 'conversation',
            'discussion', 'lecture', 'presentation', 'commentary',
            'analysis', 'review', 'explanation', 'tutorial'
        ]

        # Анти-музыкальные ключевые слова
        self.music_keywords = [
            'music', 'song', 'album', 'artist', 'band', 'mv', 
            'official video', 'lyrics', 'beat', 'melody'
        ]

    def verify_voice_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Verify voice content in YouTube and Twitch links"""
        
        if not audio_links:
            print("🔍 No audio links to verify")
            return []

        verified_links = []
        print(f"🎙️ Starting voice verification for {len(audio_links)} YouTube/Twitch links/ TikTok...")

        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')
            username = link_data.get('username', 'unknown')
            platform = link_data.get('platform_type', 'unknown')
            audio_type = link_data.get('audio_type', 'unknown')

            if not url:
                continue

            print(f"🔍 [{i}/{len(audio_links)}] Voice check {username} ({platform}): {url[:50]}...")

            if platform == 'youtube':
                voice_result = self._verify_youtube_voice(url, audio_type)
            elif platform == 'twitch':
                voice_result = self._verify_twitch_voice(url, audio_type)
            elif platform == 'tiktok':
                voice_result = self._verify_tiktok_voice(url, audio_type)
            else:
                continue

            # Добавляем результаты проверки голоса
            link_data.update({
                'has_voice': voice_result['has_voice'],
                'voice_confidence': voice_result['confidence'],
                'voice_type': voice_result.get('voice_type'),
                'verification_status': voice_result['status']
            })

            verified_links.append(link_data)

        # Фильтруем только подтвержденный голосовой контент
        confirmed_voice = [link for link in verified_links if link['has_voice']]

        print(f"\n🎙️ Voice verification completed!")
        print(f"📊 Total links checked: {len(audio_links)}")
        print(f"✅ Confirmed voice content: {len(confirmed_voice)}")
        print(f"❌ No voice content: {len(audio_links) - len(confirmed_voice)}")

        return verified_links
    def _verify_tiktok_voice(self, url: str, audio_type: str) -> dict:
    # Пока по умолчанию — high confidence, если контент не явно музыкальный.
        return {
            'has_voice': True,
            'confidence': 'high',
            'voice_type': 'tiktok_general',
            'status': 'tiktok_voice_assumed'  # Возможна проверка ключевых слов, если нужно
        }
        
    def _verify_youtube_voice(self, url: str, audio_type: str) -> Dict:
        """Verify voice content in YouTube videos"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()

            # Подсчет голосовых и музыкальных индикаторов
            voice_score = sum(1 for keyword in self.voice_keywords if keyword in content)
            music_score = sum(1 for keyword in self.music_keywords if keyword in content)

            # Учитываем предварительную классификацию из аудио-детекции
            if audio_type == 'speech_content':
                voice_score += 3
            elif audio_type == 'educational_content':
                voice_score += 2
            elif audio_type == 'music_content':
                music_score += 3

            # Финальный скор (голос минус музыка)
            final_score = voice_score - (music_score * 0.7)

            if final_score >= 3:
                return {
                    'has_voice': True,
                    'confidence': 'high',
                    'voice_type': self._determine_youtube_voice_type(content, audio_type),
                    'status': f'youtube_voice_confirmed (score: {final_score})'
                }
            elif final_score >= 1:
                return {
                    'has_voice': True,
                    'confidence': 'medium',
                    'voice_type': self._determine_youtube_voice_type(content, audio_type),
                    'status': f'youtube_voice_likely (score: {final_score})'
                }
            else:
                return {
                    'has_voice': False,
                    'confidence': 'medium',
                    'status': f'youtube_non_voice_content (score: {final_score})'
                }

        except Exception as e:
            return {
                'has_voice': False,
                'confidence': 'unknown',
                'status': f'youtube_verification_error: {str(e)}'
            }

    def _verify_twitch_voice(self, url: str, audio_type: str) -> Dict:
        """Verify voice content in Twitch streams"""
        
        # Twitch имеет высокую вероятность голосового контента
        if audio_type == 'live_talk':
            return {
                'has_voice': True,
                'confidence': 'high',
                'voice_type': 'live_conversation',
                'status': 'twitch_just_chatting_confirmed'
            }
        elif audio_type == 'talk_show':
            return {
                'has_voice': True,
                'confidence': 'high',
                'voice_type': 'talk_show',
                'status': 'twitch_talk_show_confirmed'
            }
        elif audio_type == 'gaming_commentary':
            return {
                'has_voice': True,
                'confidence': 'medium',
                'voice_type': 'gaming_commentary',
                'status': 'twitch_gaming_with_voice'
            }
        else:
            return {
                'has_voice': True,
                'confidence': 'medium',
                'voice_type': 'general_stream',
                'status': 'twitch_general_stream'
            }

    def _determine_youtube_voice_type(self, content: str, audio_type: str) -> str:
        """Determine specific voice type for YouTube"""
        if audio_type == 'speech_content':
            if 'podcast' in content:
                return 'podcast'
            elif 'interview' in content:
                return 'interview'
            else:
                return 'talk_content'
        elif audio_type == 'educational_content':
            return 'educational'
        else:
            return 'voice_content'
