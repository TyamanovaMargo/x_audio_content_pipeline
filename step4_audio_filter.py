from urllib.parse import urlparse
from typing import List, Dict
import subprocess
import json
import re

class AudioContentFilter:
    def __init__(self, min_duration=30, max_duration=3600):
        self.min_duration = min_duration  # 30 seconds
        self.max_duration = max_duration  # 1 hour (3600 seconds)
        
        self.AUDIO_PLATFORMS = {
            'youtube.com': 'youtube',
            'youtu.be': 'youtube',
            'twitch.tv': 'twitch',
            'm.twitch.tv': 'twitch',
            'www.youtube.com': 'youtube',
            'www.twitch.tv': 'twitch',
            'tiktok.com': 'tiktok',
            'www.tiktok.com': 'tiktok',
            'vm.tiktok.com': 'tiktok',
            'm.tiktok.com': 'tiktok'
        }

    def filter_audio_links(self, links: List[Dict]) -> List[Dict]:
        """Filter links for YouTube, Twitch, and TikTok with intelligent handling
        Now processes original url, youtube_url, and twitch_url columns"""
        results = []
        platform_stats = {
            'youtube': 0, 
            'twitch': 0, 
            'tiktok': 0, 
            'filtered_out': 0,
            'channel_urls': 0,
            'duration_filtered': 0
        }
        
        for link in links:
            urls_to_check = []
            
            # Collect all URLs to check from different columns
            if 'url' in link and link['url']:
                urls_to_check.append(('original', link['url']))
            if 'youtube_url' in link and link['youtube_url']:
                urls_to_check.append(('youtube', link['youtube_url']))
            if 'twitch_url' in link and link['twitch_url']:
                urls_to_check.append(('twitch', link['twitch_url']))
                
            found_platform = None
            
            # Process each URL
            for url_source, url in urls_to_check:
                domain = urlparse(url).netloc.lower()
                
                for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                    if plat_domain in domain:
                        url_type = self._classify_url_type(url, plat_name)
                        
                        if url_type == 'channel':
                            # For channel URLs, skip duration check and mark for later processing
                            link_copy = link.copy()
                            link_copy['url'] = url  # Use the found audio platform URL
                            link_copy['platform_type'] = plat_name
                            link_copy['url_type'] = 'channel'
                            link_copy['url_source'] = url_source
                            link_copy['requires_video_selection'] = True
                            results.append(link_copy)
                            platform_stats['channel_urls'] += 1
                            found_platform = plat_name
                            print(f"📺 Channel URL detected: {plat_name} - will select recent videos")
                            
                        elif url_type == 'video':
                            # For direct video URLs, check duration
                            duration_info = self._check_video_duration(url)
                            
                            if duration_info and duration_info['valid']:
                                link_copy = link.copy()
                                link_copy['url'] = url  # Use the found audio platform URL
                                link_copy['platform_type'] = plat_name
                                link_copy['url_type'] = 'video'
                                link_copy['url_source'] = url_source
                                link_copy.update(duration_info)
                                results.append(link_copy)
                                platform_stats[plat_name] += 1
                                found_platform = plat_name
                            elif duration_info:
                                platform_stats['duration_filtered'] += 1
                                print(f"⏰ Duration filtered: {duration_info['duration']}s not in range")
                            else:
                                # Duration check failed, but still include (will be handled later)
                                link_copy = link.copy()
                                link_copy['url'] = url  # Use the found audio platform URL
                                link_copy['platform_type'] = plat_name
                                link_copy['url_type'] = 'video'
                                link_copy['url_source'] = url_source
                                results.append(link_copy)
                                platform_stats[plat_name] += 1
                                found_platform = plat_name
                        
                        break
                
                if found_platform:
                    break
            
            if not found_platform:
                platform_stats['filtered_out'] += 1

        print("🎯 Filtered results:")
        print(f" ✅ YouTube: {platform_stats['youtube']}")
        print(f" ✅ Twitch: {platform_stats['twitch']}")
        print(f" ✅ TikTok: {platform_stats['tiktok']}")
        print(f" 📺 Channel URLs: {platform_stats['channel_urls']}")
        print(f" ⏰ Duration filtered: {platform_stats['duration_filtered']}")
        print(f" ❌ Other platforms: {platform_stats['filtered_out']}")
        
        return results

    def _classify_url_type(self, url: str, platform: str) -> str:
        """Classify if URL is a channel, video, or playlist"""
        url_lower = url.lower()
        
        if platform == 'youtube':
            # Channel patterns
            channel_patterns = [
                r'youtube\.com/@[^/?]+/?$',
                r'youtube\.com/c/[^/?]+/?$',
                r'youtube\.com/channel/[^/?]+/?$',
                r'youtube\.com/user/[^/?]+/?$'
            ]
            
            # Video patterns
            video_patterns = [
                r'youtube\.com/watch\?v=',
                r'youtu\.be/[^/?]+'
            ]
            
            for pattern in channel_patterns:
                if re.search(pattern, url_lower):
                    return 'channel'
            
            for pattern in video_patterns:
                if re.search(pattern, url_lower):
                    return 'video'
                    
        elif platform == 'twitch':
            # Channel patterns
            if re.match(r'.*twitch\.tv/[^/?]+/?$', url_lower):
                return 'channel'
            # Video/VOD patterns
            elif '/videos/' in url_lower or '/clip/' in url_lower:
                return 'video'
            else:
                return 'channel'  # Default for Twitch
                
        elif platform == 'tiktok':
            # TikTok is usually individual videos
            if re.search(r'tiktok\.com/@[^/]+/video/', url_lower):
                return 'video'
            elif re.search(r'tiktok\.com/@[^/?]+/?$', url_lower):
                return 'channel'
            else:
                return 'video'  # Default assumption for TikTok
        
        return 'video'  # Default assumption

    def _check_video_duration(self, url: str) -> Dict:
        """Check video duration with improved timeout handling"""
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--quiet',
                '--no-warnings',
                '--socket-timeout', '15',  # Add socket timeout
                url
            ]
            
            # Reduced timeout for individual videos
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0 and result.stdout.strip():
                info = json.loads(result.stdout)
                duration = info.get('duration', 0)
                title = info.get('title', 'Unknown')[:50]
                
                is_valid = self.min_duration <= duration <= self.max_duration
                
                return {
                    'duration': duration,
                    'title': title,
                    'valid': is_valid,
                    'reason': 'valid' if is_valid else 
                             'too_short' if duration < self.min_duration else 'too_long'
                }
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout checking duration for: {url[:50]}...")
            return None
        except Exception as e:
            print(f"⚠️ Duration check error: {str(e)[:50]}")
            return None
        
        return None

    def get_recent_videos_from_channel(self, channel_url: str, max_videos: int = 5) -> List[Dict]:
        """Get recent videos from a channel URL"""
        try:
            print(f"🔍 Getting recent videos from channel: {channel_url[:50]}...")
            
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--quiet',
                '--no-warnings',
                '--playlist-end', str(max_videos),
                '--socket-timeout', '20',
                channel_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                videos = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            video_info = json.loads(line)
                            duration = video_info.get('duration', 0)
                            
                            # Check duration constraint
                            if self.min_duration <= duration <= self.max_duration:
                                videos.append({
                                    'url': video_info.get('webpage_url', ''),
                                    'title': video_info.get('title', 'Unknown'),
                                    'duration': duration,
                                    'valid': True,
                                    'reason': 'valid'
                                })
                        except json.JSONDecodeError:
                            continue
                
                print(f"✅ Found {len(videos)} valid videos from channel")
                return videos[:max_videos]  # Limit results
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout getting videos from channel")
        except Exception as e:
            print(f"❌ Error getting channel videos: {str(e)[:50]}")
            
        return []

