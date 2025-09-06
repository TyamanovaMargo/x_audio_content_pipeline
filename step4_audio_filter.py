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
        """Filter links for YouTube, Twitch, and TikTok with intelligent handling"""
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
            domain = urlparse(link['url']).netloc.lower()
            found_platform = None
            
            for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                if plat_domain in domain:
                    url_type = self._classify_url_type(link['url'], plat_name)
                    
                    if url_type == 'channel':
                        # For channel URLs, skip duration check and mark for later processing
                        link['platform_type'] = plat_name
                        link['url_type'] = 'channel'
                        link['requires_video_selection'] = True
                        results.append(link)
                        platform_stats['channel_urls'] += 1
                        found_platform = plat_name
                        print(f"📺 Channel URL detected: {plat_name} - will select recent videos")
                        
                    elif url_type == 'video':
                        # For direct video URLs, check duration
                        duration_info = self._check_video_duration(link['url'])
                        
                        if duration_info and duration_info['valid']:
                            link['platform_type'] = plat_name
                            link['url_type'] = 'video'
                            link.update(duration_info)
                            results.append(link)
                            platform_stats[plat_name] += 1
                            found_platform = plat_name
                        elif duration_info:
                            platform_stats['duration_filtered'] += 1
                            print(f"⏰ Duration filtered: {duration_info['duration']}s not in range")
                        else:
                            # Duration check failed, but still include (will be handled later)
                            link['platform_type'] = plat_name
                            link['url_type'] = 'video'
                            results.append(link)
                            platform_stats[plat_name] += 1
                            found_platform = plat_name
                    
                    break
            
            if not found_platform:
                platform_stats['filtered_out'] += 1

        print(f"🎯 Filtered results:")
        print(f" ✅ YouTube: {platform_stats['youtube']}")
        print(f" ✅ Twitch: {platform_stats['twitch']}")
        print(f" ✅ TikTok: {platform_stats['tiktok']}")
        print(f" 📺 Channel URLs: {platform_stats['channel_urls']}")
        print(f" ⏰ Duration filtered: {platform_stats['duration_filtered']}")
        print(f" ❌ Other platforms: {platform_stats['filtered_out']}")
        
        return results

    def filter_audio_links_enhanced(self, enhanced_links: List[Dict]) -> List[Dict]:
        """Enhanced filter that handles both original URLs and discovered YouTube/Twitch channels"""
        results = []
        platform_stats = {
            'youtube': 0, 
            'twitch': 0, 
            'tiktok': 0, 
            'filtered_out': 0,
            'channel_urls': 0,
            'duration_filtered': 0,
            'discovered_youtube': 0,
            'discovered_twitch': 0
        }
        
        for link_data in enhanced_links:
            found_urls = []
            
            # Check original URL from profile
            original_url = link_data.get('url', '')
            if original_url:
                domain = urlparse(original_url).netloc.lower()
                for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                    if plat_domain in domain:
                        found_urls.append({
                            'url': original_url,
                            'platform_type': plat_name,
                            'source': 'profile_link',
                            'score': 100  # Original links get highest score
                        })
                        break
            
            # Check discovered YouTube URL
            youtube_url = link_data.get('youtube_url', '')
            youtube_score = link_data.get('youtube_score', 0)
            if youtube_url and youtube_score > 0:
                found_urls.append({
                    'url': youtube_url,
                    'platform_type': 'youtube',
                    'source': 'discovered',
                    'score': youtube_score
                })
                platform_stats['discovered_youtube'] += 1
            
            # Check discovered Twitch URL
            twitch_url = link_data.get('twitch_url', '')
            twitch_score = link_data.get('twitch_score', 0)
            if twitch_url and twitch_score > 0:
                found_urls.append({
                    'url': twitch_url,
                    'platform_type': 'twitch',
                    'source': 'discovered',
                    'score': twitch_score
                })
                platform_stats['discovered_twitch'] += 1
            
            # Process each found URL
            for url_info in found_urls:
                url = url_info['url']
                platform = url_info['platform_type']
                source = url_info['source']
                score = url_info['score']
                
                # Classify URL type
                url_type = self._classify_url_type(url, platform)
                
                # Create result entry
                result_entry = link_data.copy()  # Keep all original data
                result_entry.update({
                    'filtered_url': url,  # The URL we're actually using
                    'platform_type': platform,
                    'url_type': url_type,
                    'discovery_source': source,
                    'discovery_score': score
                })
                
                if url_type == 'channel':
                    result_entry['requires_video_selection'] = True
                    platform_stats['channel_urls'] += 1
                    print(f"📺 {source.title()} {platform} channel: @{link_data.get('username', 'unknown')}")
                    
                elif url_type == 'video':
                    # Check duration for videos
                    duration_info = self._check_video_duration(url)
                    if duration_info and duration_info['valid']:
                        result_entry.update(duration_info)
                        print(f"🎬 {source.title()} {platform} video: @{link_data.get('username', 'unknown')} ({duration_info['duration']}s)")
                    elif duration_info:
                        platform_stats['duration_filtered'] += 1
                        print(f"⏰ Duration filtered: {duration_info['duration']}s not in range")
                        continue
                
                results.append(result_entry)
                platform_stats[platform] += 1
            
            # If no audio URLs found for this profile
            if not found_urls:
                platform_stats['filtered_out'] += 1
        
        print(f"\n🎯 Enhanced Filtering Results:")
        print(f" ✅ YouTube (original): {platform_stats['youtube'] - platform_stats['discovered_youtube']}")
        print(f" 🔍 YouTube (discovered): {platform_stats['discovered_youtube']}")
        print(f" ✅ Twitch (original): {platform_stats['twitch'] - platform_stats['discovered_twitch']}")
        print(f" 🔍 Twitch (discovered): {platform_stats['discovered_twitch']}")
        print(f" ✅ TikTok: {platform_stats['tiktok']}")
        print(f" 📺 Channel URLs: {platform_stats['channel_urls']}")
        print(f" ⏰ Duration filtered: {platform_stats['duration_filtered']}")
        print(f" ❌ No audio links: {platform_stats['filtered_out']}")
        print(f" 📈 Total audio links: {len(results)}")
        
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

