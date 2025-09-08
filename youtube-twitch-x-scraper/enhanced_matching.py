#!/usr/bin/env python3
"""
Enhanced Name Matching Logic - Extracted from banana.py
This contains the sophisticated matching logic that works fantastically
"""

import re
import logging
import requests
from bs4 import BeautifulSoup
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class EnhancedMatcher:
    """Enhanced matching logic for social media profiles"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self):
        """Create requests session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    def enhanced_name_match(self, url: str, username: str, profile_name: str = None) -> bool:
        """Enhanced social media name matching with title verification"""
        try:
            # Use username as primary, fallback to profile_name
            business_name = username if username else profile_name
            if not business_name:
                return False
                
            business_lower = business_name.lower().strip()
            business_clean = re.sub(r'[^\w]', '', business_lower)  # For containment checks
            business_with_spaces = re.sub(r'[^\w\s]', '', business_lower)  # For word splitting
            
            # Get page title for additional verification
            page_title = self.get_page_title(url, business_name)
            title_matches = False
            
            if page_title:
                # Check if title contains business name - improved word-based matching
                title_clean = re.sub(r'[^\w\s]', '', page_title.lower())
                business_words = [word for word in business_with_spaces.split() if len(word) > 2]
                title_words = title_clean.split()
                
                # Check for exact business name match in title
                if business_clean in title_clean:
                    title_matches = True
                    logger.info(f"Exact title match found: '{business_name}' in page title '{page_title}'")
                # Check for word-based matching (at least 60% of business words in title)
                elif business_words and title_words:
                    word_matches = sum(1 for word in business_words if any(word in tword or tword in word for tword in title_words))
                    match_ratio = word_matches / len(business_words)
                    if match_ratio >= 0.6:
                        title_matches = True
                        logger.info(f"Word-based title match found: '{business_name}' matches page title '{page_title}' (ratio: {match_ratio:.2f})")
            
            # URL path analysis
            url_path = url.split('/')
            potential_usernames = []
            
            for part in url_path:
                if part and not part.startswith('?') and len(part) > 2:
                    clean_part = re.sub(r'[^\w]', '', part.lower())
                    if clean_part:
                        potential_usernames.append(clean_part)
            
            for url_username in potential_usernames:
                # Check both directions: username in business_clean OR business_clean in username
                # Also check for partial word matches
                username_match = False
                
                if url_username in business_clean:
                    # Username is contained in business name (e.g., "5brostransport" in "5brostransportservice")
                    threshold = 0.20 if title_matches else 0.25
                    if len(url_username) >= 3 and len(url_username) >= len(business_clean) * threshold:
                        username_match = True
                elif business_clean in url_username:
                    # Business name is contained in username (original logic)
                    threshold = 0.20 if title_matches else 0.35
                    if len(business_clean) >= 3 and len(business_clean) >= len(url_username) * threshold:
                        username_match = True
                else:
                    # Check for word-by-word matching for complex names
                    business_words = [word for word in business_with_spaces.split() if len(word) > 2]
                    username_words = url_username.replace('_', ' ').replace('-', ' ').split()
                    
                    if business_words and username_words:
                        # Method 1: Check if most significant words from business name appear in username
                        word_matches = sum(1 for word in business_words if any(word in uword or uword in word for uword in username_words))
                        match_ratio = word_matches / len(business_words)
                        
                        if match_ratio >= 0.6:  # At least 60% of business words should match
                            username_match = True
                        else:
                            # Method 2: Enhanced abbreviation matching
                            username_match = self._check_abbreviation_patterns(url_username, business_words)
                
                if username_match:
                    match_type = "enhanced" if title_matches else "standard"
                    logger.info(f"{match_type.capitalize()} match found: '{business_name}' matches URL username '{url_username}' (threshold check passed)")
                    return True
            
            # If title matches but URL doesn't meet threshold, still allow it
            if title_matches:
                logger.info(f"Title-based match: '{business_name}' found in page title, allowing URL")
                return True
            
            logger.debug(f"No match: '{business_name}' not found in URL usernames {potential_usernames} or page title")
            return False
            
        except Exception as e:
            logger.debug(f"Error in enhanced name matching: {str(e)}")
            return False

    def _check_abbreviation_patterns(self, username: str, business_words: list) -> bool:
        """Check for abbreviation patterns in usernames"""
        try:
            # Method 2: Enhanced abbreviation matching
            # Check simple patterns first (e.g., "laauto" = "la" + "auto")
            for word in business_words:
                if word in username:
                    # Found a business word in username, check if rest could be abbreviation
                    word_start = username.find(word)
                    
                    if word_start == 0:
                        # Word is at the beginning (e.g., "auto" in "autola")
                        remaining_username = username[len(word):]
                    else:
                        # Word is in the middle/end (e.g., "pizza" in "nypizza") 
                        remaining_username = username[:word_start]
                    
                    if len(remaining_username) >= 1:
                        # Check if remaining part could be abbreviation of other words
                        other_words = [w for w in business_words if w != word]
                        
                        # Try different abbreviation patterns
                        # Pattern 1: All other words abbreviated
                        pattern1 = ''.join(w[0] for w in other_words)
                        
                        # Pattern 2: Try subsets of words (for cases like "ny" from "new york" ignoring "company")
                        patterns = [pattern1]
                        
                        # Add patterns for subsets of words (consecutive words)
                        for i in range(len(other_words)):
                            for j in range(i + 1, len(other_words) + 1):
                                subset = other_words[i:j]
                                if len(subset) >= 2:  # Only for 2+ words
                                    subset_pattern = ''.join(w[0] for w in subset)
                                    patterns.append(subset_pattern)
                        
                        # Remove duplicates
                        patterns = list(set(patterns))
                        
                        for pattern in patterns:
                            if pattern and pattern == remaining_username:
                                logger.info(f"Abbreviation + word match: '{remaining_username}' + '{word}' = '{username}'")
                                return True
                        
                        if len(remaining_username) >= 1:
                            # Check if remaining matches any word exactly
                            for other_word in other_words:
                                if remaining_username == other_word:
                                    logger.info(f"Word combination match: '{remaining_username}' + '{word}' = '{username}'")
                                    return True
            
            # Method 3: Check reverse pattern (word + abbreviation) 
            for word in business_words:
                if username.startswith(word):
                    remaining = username[len(word):]
                    if len(remaining) >= 1:
                        other_words = [w for w in business_words if w != word]
                        patterns = [
                            ''.join(w[0] for w in other_words),
                            ''.join(w[:2] for w in other_words if len(w) >= 2),
                        ]
                        
                        # Add subset patterns
                        for i in range(len(other_words)):
                            for j in range(i + 1, len(other_words) + 1):
                                subset = other_words[i:j]
                                if len(subset) >= 1:
                                    subset_pattern = ''.join(w[0] for w in subset)
                                    patterns.append(subset_pattern)
                        
                        patterns = list(set(patterns))
                        
                        for pattern in patterns:
                            if pattern and pattern == remaining:
                                logger.info(f"Word + abbreviation match: '{word}' + '{pattern}' = '{username}'")
                                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error in abbreviation matching: {str(e)}")
            return False

    def get_page_title(self, url: str, business_name: str) -> Optional[str]:
        """Get page title from URL for enhanced matching"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text().strip()
                    logger.debug(f"Page title for {url}: {title}")
                    return title
            else:
                logger.debug(f"Failed to fetch page title for {url}: Status {response.status_code}")
        except Exception as e:
            logger.debug(f"Error fetching page title for {url}: {str(e)}")
        return None

    def calculate_match_score(self, username: str, profile_name: str, title: str, url: str) -> int:
        """Calculate enhanced match score between user data and search result"""
        try:
            business_name = username if username else profile_name
            if not business_name:
                return 0
            
            business_lower = business_name.lower().strip()
            business_clean = re.sub(r'[^\w]', '', business_lower)
            business_with_spaces = re.sub(r'[^\w\s]', '', business_lower)
            
            title_lower = title.lower().strip()
            title_clean = re.sub(r'[^\w]', '', title_lower)
            
            url_lower = url.lower().strip()
            url_clean = re.sub(r'[^\w]', '', url_lower)
            
            scores = []
            
            # Title matching scores
            if business_clean in title_clean:
                scores.append(95)  # High score for exact match
            else:
                # Word-based title matching
                business_words = [word for word in business_with_spaces.split() if len(word) > 2]
                title_words = title_lower.split()
                
                if business_words and title_words:
                    word_matches = sum(1 for word in business_words if any(word in tword or tword in word for tword in title_words))
                    match_ratio = word_matches / len(business_words)
                    scores.append(int(match_ratio * 90))  # Scale to 0-90
            
            # URL matching scores
            if business_clean in url_clean:
                scores.append(85)  # High score for URL match
            else:
                # Check URL path components
                url_path = url.split('/')
                for part in url_path:
                    if part and len(part) > 2:
                        part_clean = re.sub(r'[^\w]', '', part.lower())
                        if part_clean and business_clean in part_clean:
                            scores.append(75)
                            break
                        elif part_clean and part_clean in business_clean:
                            scores.append(60)
                            break
            
            return max(scores) if scores else 0
            
        except Exception as e:
            logger.debug(f"Error calculating match score: {str(e)}")
            return 0
