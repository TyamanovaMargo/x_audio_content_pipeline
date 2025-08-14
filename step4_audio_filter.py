from urllib.parse import urlparse
from typing import List, Dict

class AudioContentFilter:
    AUDIO_PLATFORMS = {
        'youtube.com': 'youtube',
        'youtu.be': 'youtube',
        'twitch.tv': 'twitch',
        'tiktok.com': 'tiktok',
        'instagram.com': 'instagram',
        'spotify.com': 'spotify',
        'soundcloud.com': 'soundcloud',
        'apple.com/music': 'apple_music',
        'podcasts.apple.com': 'apple_podcasts',
        'anchor.fm': 'anchor'
    }

    def filter_audio_links(self, links: List[Dict]) -> List[Dict]:
        results = []
        for link in links:
            domain = urlparse(link['url']).netloc.lower()
            for plat_domain, plat_name in self.AUDIO_PLATFORMS.items():
                if plat_domain in domain:
                    link['platform_type'] = plat_name
                    results.append(link)
                    break
        return results
