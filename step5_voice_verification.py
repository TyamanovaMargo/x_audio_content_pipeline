# step5_voice_verification.py
import requests
import pandas as pd
from typing import List, Dict
from urllib.parse import urlparse

class VoiceContentVerifier:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; VoiceBot/1.0)'
        })
        
        # Voice-specific keywords instead of music
        self.voice_keywords = [
            'podcast', 'interview', 'talk', 'speech', 'conversation',
            'discussion', 'lecture', 'presentation', 'webinar',
            'audiobook', 'storytelling', 'radio', 'show', 'episode',
            'host', 'guest', 'speaking', 'voice', 'audio content',
            'commentary', 'analysis', 'debate', 'panel', 'dialogue'
        ]
        
        # High-confidence voice platforms
        self.voice_platforms = [
            'anchor.fm', 'podcasts.apple.com', 'open.spotify.com/show',
            'open.spotify.com/episode', 'castbox.fm', 'overcast.fm',
            'pocketcasts.com', 'stitcher.com', 'podbean.com',
            'buzzsprout.com', 'libsyn.com', 'audioboom.com'
        ]

    def verify_voice_content(self, audio_links: List[Dict]) -> List[Dict]:
        """Verify if links contain voice/speech content"""
        if not audio_links:
            print("🔍 No audio links to verify")
            return []
            
        verified_links = []
        print(f"🎙️ Starting voice verification for {len(audio_links)} links...")
        
        for i, link_data in enumerate(audio_links, 1):
            url = link_data.get('url', '')
            username = link_data.get('username', 'unknown')
            
            if not url:
                continue
            
            print(f"🔍 [{i}/{len(audio_links)}] Checking {username}: {url[:50]}...")
            
            verification_result = self._verify_voice_link(url)
            
            # Add verification data
            link_data.update({
                'has_voice': verification_result['has_voice'],
                'voice_confidence': verification_result['confidence'],
                'content_type': verification_result['content_type'],
                'voice_type': verification_result.get('voice_type'),
                'verification_status': verification_result['status']
            })
            
            verified_links.append(link_data)
        
        # Filter only confirmed voice content
        confirmed_voice = [link for link in verified_links if link['has_voice']]
        
        print(f"\n🎙️ Voice verification completed!")
        print(f"📊 Total links checked: {len(audio_links)}")
        print(f"✅ Confirmed voice content: {len(confirmed_voice)}")
        print(f"❌ No voice content: {len(audio_links) - len(confirmed_voice)}")
        
        return verified_links

    def _verify_voice_link(self, url: str) -> Dict:
        """Verify a single URL for voice content"""
        try:
            domain = urlparse(url).netloc.lower()
            
            # High confidence voice platforms
            for platform in self.voice_platforms:
                if platform in url.lower():
                    voice_type = self._detect_voice_type_from_platform(url, platform)
                    return {
                        'has_voice': True,
                        'confidence': 'high',
                        'content_type': 'voice/platform',
                        'voice_type': voice_type,
                        'status': f'voice_platform_verified: {platform}'
                    }
            
            # Platform-specific voice detection
            if 'youtube.com' in domain or 'youtu.be' in domain:
                return self._check_youtube_for_voice(url)
            elif 'twitch.tv' in domain:
                return self._check_twitch_for_voice(url)
            elif 'instagram.com' in domain:
                return self._check_instagram_for_voice(url)
            else:
                return self._check_generic_for_voice(url)
                
        except Exception as e:
            return {
                'has_voice': False,
                'confidence': 'unknown',
                'content_type': 'error',
                'status': f'verification_failed: {str(e)}'
            }

    def _check_youtube_for_voice(self, url: str) -> Dict:
        """Check YouTube specifically for voice/speech content"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            # Voice-specific scoring
            voice_score = sum(1 for keyword in self.voice_keywords if keyword in content)
            
            # Anti-music indicators
            music_keywords = ['music', 'song', 'album', 'artist', 'band', 'beat', 'melody']
            music_score = sum(1 for keyword in music_keywords if keyword in content)
            
            # Adjust score: favor voice, penalize music
            final_score = voice_score - (music_score * 0.5)
            
            if final_score >= 3:
                voice_type = self._detect_voice_type_from_content(content)
                return {
                    'has_voice': True,
                    'confidence': 'high' if final_score >= 5 else 'medium',
                    'content_type': 'video/youtube',
                    'voice_type': voice_type,
                    'status': f'youtube_voice_detected (score: {final_score})'
                }
            else:
                return {
                    'has_voice': False,
                    'confidence': 'medium',
                    'content_type': 'video/youtube',
                    'status': f'youtube_non_voice (score: {final_score})'
                }
                
        except Exception as e:
            return {
                'has_voice': False,
                'confidence': 'unknown',
                'status': f'youtube_check_failed: {str(e)}'
            }

    def _check_twitch_for_voice(self, url: str) -> Dict:
        """Check Twitch for voice content"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            voice_categories = [
                'just chatting', 'talk shows', 'podcasts', 'asmr',
                'commentary', 'interview', 'discussion'
            ]
            
            category_found = any(category in content for category in voice_categories)
            voice_score = sum(1 for keyword in self.voice_keywords if keyword in content)
            
            if category_found or voice_score >= 2:
                return {
                    'has_voice': True,
                    'confidence': 'high',
                    'content_type': 'stream/twitch',
                    'voice_type': 'live_talk' if category_found else 'voice_stream',
                    'status': 'twitch_voice_stream_detected'
                }
            else:
                return {
                    'has_voice': False,
                    'confidence': 'medium',
                    'content_type': 'stream/twitch',
                    'status': 'twitch_gaming_stream'
                }
                
        except Exception as e:
            return {'has_voice': False, 'confidence': 'unknown', 'status': f'twitch_check_failed: {str(e)}'}

    def _check_instagram_for_voice(self, url: str) -> Dict:
        """Check Instagram for voice content creators"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content = response.text.lower()
            
            voice_indicators = [
                'podcaster', 'host', 'speaker', 'interviewer', 'storyteller',
                'voice coach', 'narrator', 'commentator', 'radio host'
            ]
            
            found_indicators = [indicator for indicator in voice_indicators if indicator in content]
            voice_score = sum(1 for keyword in self.voice_keywords if keyword in content)
            
            if found_indicators or voice_score >= 3:
                return {
                    'has_voice': True,
                    'confidence': 'medium',
                    'content_type': 'social/instagram',
                    'voice_type': 'voice_creator',
                    'status': f'instagram_voice_creator: {found_indicators[:2]}'
                }
            else:
                return {
                    'has_voice': False,
                    'confidence': 'medium',
                    'content_type': 'social/instagram',
                    'status': 'instagram_non_voice'
                }
                
        except Exception as e:
            return {'has_voice': False, 'confidence': 'unknown', 'status': f'instagram_check_failed: {str(e)}'}

    def _check_generic_for_voice(self, url: str) -> Dict:
        """Check generic URLs for voice content"""
        try:
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            
            # Direct audio files
            if 'audio/' in content_type:
                return {
                    'has_voice': True,
                    'confidence': 'medium',
                    'content_type': content_type,
                    'voice_type': 'audio_file',
                    'status': 'direct_audio_file'
                }
            
            # Check webpage content
            if 'text/html' in content_type:
                response = self.session.get(url, timeout=self.timeout)
                content = response.text.lower()
                
                voice_score = sum(1 for keyword in self.voice_keywords if keyword in content)
                
                if voice_score >= 2:
                    return {
                        'has_voice': True,
                        'confidence': 'medium',
                        'content_type': 'text/html',
                        'voice_type': 'voice_webpage',
                        'status': f'voice_content_detected (score: {voice_score})'
                    }
            
            return {
                'has_voice': False,
                'confidence': 'low',
                'content_type': content_type,
                'status': 'no_voice_indicators'
            }
            
        except Exception as e:
            return {'has_voice': False, 'confidence': 'unknown', 'status': f'generic_check_failed: {str(e)}'}

    def _detect_voice_type_from_content(self, content: str) -> str:
        """Detect specific type of voice content"""
        if 'podcast' in content:
            return 'podcast'
        elif 'interview' in content:
            return 'interview'
        elif 'lecture' in content or 'presentation' in content:
            return 'educational'
        elif 'audiobook' in content or 'storytelling' in content:
            return 'narrative'
        elif 'radio' in content or 'show' in content:
            return 'radio_show'
        elif 'commentary' in content or 'analysis' in content:
            return 'commentary'
        else:
            return 'voice_content'

    def _detect_voice_type_from_platform(self, url: str, platform: str) -> str:
        """Detect voice type based on platform"""
        if 'podcasts.apple.com' in platform or 'anchor.fm' in platform:
            return 'podcast'
        elif 'show' in url:
            return 'podcast_show'
        elif 'episode' in url:
            return 'podcast_episode'
        else:
            return 'voice_platform'
