from urllib.parse import urlparse
from typing import List, Dict
from urllib.parse import urlparse
import subprocess
import json

class AudioContentFilter:
    # YouTube, Twitch, and TikTok platforms
    def __init__(self, min_duration=30, max_duration=3600):
        self.min_duration = min_duration  # 30 seconds
        self.max_duration = max_duration  # 1 hour (3600 seconds)
    AUDIO_PLATFORMS = {
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

    # def filter_audio_links(self, links: List[Dict]) -> List[Dict]:
    #     """Filter links for YouTube, Twitch, and TikTok only"""
    #     results = []
    #     platform_stats = {'youtube': 0, 'twitch': 0, 'tiktok': 0, 'filtered_out': 0}
        
    #     for link in links:
    #         domain = urlparse(link['url']).netloc.lower()
    #         found_platform = None
            
    #         for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
    #             if plat_domain in domain:
    #                 link['platform_type'] = plat_name
    #                 results.append(link)
    #                 platform_stats[plat_name] += 1
    #                 found_platform = plat_name
    #                 break
            
    #         if not found_platform:
    #             platform_stats['filtered_out'] += 1
        
    #     print(f"🎯 Filtered for YouTube, Twitch, and TikTok only:")
    #     print(f" ✅ YouTube: {platform_stats['youtube']}")
    #     print(f" ✅ Twitch: {platform_stats['twitch']}")
    #     print(f" ✅ TikTok: {platform_stats['tiktok']}")
    #     print(f" ❌ Other platforms filtered out: {platform_stats['filtered_out']}")
        
    #     return results
    def filter_audio_links(self, links: List[Dict]) -> List[Dict]:
        """Filter links for YouTube, Twitch, and TikTok with duration constraints"""
        results = []
        platform_stats = {
            'youtube': 0, 
            'twitch': 0, 
            'tiktok': 0, 
            'filtered_out': 0, 
            'duration_filtered': 0
        }
        
        for link in links:
            domain = urlparse(link['url']).netloc.lower()
            found_platform = None
            
            # Check platform
            for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                if plat_domain in domain:
                    # Check duration if available
                    duration_info = self._check_video_duration(link['url'])
                    
                    if duration_info and duration_info['valid']:
                        link['platform_type'] = plat_name
                        link.update(duration_info)  # Add duration info
                        results.append(link)
                        platform_stats[plat_name] += 1
                        found_platform = plat_name
                    elif duration_info:
                        platform_stats['duration_filtered'] += 1
                        print(f"⏰ Filtered out {plat_name} video: {duration_info['duration']}s not in range {self.min_duration}-{self.max_duration}s")
                    else:
                        # If duration check fails, still include (will be checked later)
                        link['platform_type'] = plat_name
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
        print(f" ⏰ Duration filtered: {platform_stats['duration_filtered']}")
        print(f" ❌ Other platforms: {platform_stats['filtered_out']}")
        
        return results

    def _check_video_duration(self, url: str) -> Dict:
        """Check video duration using yt-dlp"""
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--quiet',
                '--no-warnings',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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
        except Exception as e:
            print(f"⚠️ Duration check failed for {url[:50]}: {e}")
            return None
        
        return None
