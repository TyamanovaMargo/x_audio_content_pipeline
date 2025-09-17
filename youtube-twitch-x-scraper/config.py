#!/usr/bin/env python3
"""
Configuration file for YouTube/Twitch Scraper
"""

# File paths
DATA_FILE = "../output/1.csv"  # DYNAMIC PLACEHOLDER
PROXY_FILE = "proxy/Free_Proxy_List.csv"
OUTPUT_FILE = "../output/youtube_twitch_results_enhanced.csv"

# Matching thresholds
FUZZY_MATCH_THRESHOLD = 70
ENHANCED_MATCH_THRESHOLD = 80
WORD_MATCH_RATIO = 0.6
URL_CONTAINMENT_THRESHOLD_WITH_TITLE = 0.20
URL_CONTAINMENT_THRESHOLD_WITHOUT_TITLE = 0.35

# Search settings
MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES_PER_USER = 2
MAX_PROXY_ATTEMPTS = 3

# Rate limiting (seconds)
DELAY_BETWEEN_SEARCHES = (1, 3)  # Random range
DELAY_BETWEEN_QUERIES = (2, 4)   # Random range
DELAY_BETWEEN_USERS = (3, 6)     # Random range

# Platform settings
PLATFORMS = {
    'youtube': {
        'domain': 'youtube.com',
        'valid_paths': ['/channel/', '/c/', '/@', '/user/'],
        'search_site': 'site:youtube.com'
    },
    'twitch': {
        'domain': 'twitch.tv',
        'min_url_parts': 4,
        'search_site': 'site:twitch.tv'
    }
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'