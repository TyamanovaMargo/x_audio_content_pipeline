from urllib.parse import urlparse
from typing import List, Dict

class AudioContentFilter:
    # Только YouTube и Twitch
    AUDIO_PLATFORMS = {
        'youtube.com': 'youtube',
        'youtu.be': 'youtube',
        'twitch.tv': 'twitch',
        'm.twitch.tv': 'twitch',
        'www.youtube.com': 'youtube',
        'www.twitch.tv': 'twitch'
    }

    def filter_audio_links(self, links: List[Dict]) -> List[Dict]:
        """Filter links for YouTube and Twitch only"""
        results = []
        platform_stats = {'youtube': 0, 'twitch': 0, 'filtered_out': 0}
        
        for link in links:
            domain = urlparse(link['url']).netloc.lower()
            found_platform = None
            
            for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                if plat_domain in domain:
                    link['platform_type'] = plat_name
                    results.append(link)
                    platform_stats[plat_name] += 1
                    found_platform = plat_name
                    break
            
            if not found_platform:
                platform_stats['filtered_out'] += 1
        
        print(f"🎯 Filtered for YouTube and Twitch only:")
        print(f"  ✅ YouTube: {platform_stats['youtube']}")
        print(f"  ✅ Twitch: {platform_stats['twitch']}")
        print(f"  ❌ Other platforms filtered out: {platform_stats['filtered_out']}")
        
        return results
