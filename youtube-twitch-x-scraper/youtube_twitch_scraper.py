#!/usr/bin/env python3
"""
YouTube and Twitch Channel Finder - Modular Version
Uses enhanced matching logic adapted from banana scraper
"""

import pandas as pd
import asyncio
import csv
import random
import re
import os
import logging
from urllib.parse import quote_plus
from enhanced_matching import EnhancedMatcher
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChannelMatcher(EnhancedMatcher):
    """Enhanced channel matching using sophisticated logic from banana.py"""
    
    def __init__(self, session):
        super().__init__()
        # Override parent session with the passed one
        self.session = session


class ProxyManager:
    """Manages proxy rotation and session configuration"""
    
    def __init__(self, proxy_file: str):
        self.proxy_file = proxy_file
        self.proxies = []
        self.current_index = 0
        self.failed_proxies = set()
        self.load_proxies()
    
    def load_proxies(self):
        """Load and shuffle proxy list"""
        try:
            proxy_df = pd.read_csv(self.proxy_file)
            for _, row in proxy_df.iterrows():
                proxy = {
                    'http': f"http://{row['ip']}:{row['port']}",
                    'https': f"http://{row['ip']}:{row['port']}"
                }
                self.proxies.append(proxy)
            
            random.shuffle(self.proxies)
            logger.info(f"Loaded {len(self.proxies)} proxies")
        except Exception as e:
            logger.error(f"Error loading proxies: {e}")
            self.proxies = []
    
    def get_next_proxy(self):
        """Get next working proxy in rotation"""
        if not self.proxies:
            return None
        
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            
            proxy_key = f"{proxy['http']}"
            if proxy_key not in self.failed_proxies:
                return proxy
            
            attempts += 1
        
        return None
    
    def mark_proxy_failed(self, proxy):
        """Mark a proxy as failed"""
        if proxy:
            proxy_key = f"{proxy['http']}"
            self.failed_proxies.add(proxy_key)
            logger.debug(f"Marked proxy as failed: {proxy_key}")


class SearchEngine:
    """Handles Google search with proxy rotation"""
    
    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
        self.session = requests.Session()
    
    async def search_with_crawl4ai(self, query: str, platform: str, max_results: int = 5):
        """Search for channels using crawl4ai with proxy rotation"""
        max_proxy_attempts = 3
        
        for attempt in range(max_proxy_attempts):
            proxy = self.proxy_manager.get_next_proxy()
            
            try:
                # Create search URL
                if platform == "youtube":
                    search_url = f"https://www.google.com/search?q={quote_plus(query + ' youtube channel')}"
                else:  # twitch
                    search_url = f"https://www.google.com/search?q={quote_plus(query + ' twitch')}"
                
                # Configure crawler
                crawler_config = {
                    'headless': True,
                    'verbose': False,
                }
                
                if proxy:
                    crawler_config['proxy'] = proxy['http']
                
                async with AsyncWebCrawler(**crawler_config) as crawler:
                    await asyncio.sleep(random.uniform(1, 3))
                    
                    result = await crawler.arun(
                        url=search_url,
                        word_count_threshold=10,
                        bypass_cache=True,
                        delay_before_return_html=2
                    )
                    
                    if result.success:
                        # Parse the HTML directly instead of using complex extraction strategy
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(result.html, 'html.parser')
                        
                        # DEBUG: Log what we got
                        logger.debug(f"Page title: {soup.title.string if soup.title else 'No title'}")
                        
                        results = []
                        
                        # Try multiple selectors for Google search results
                        search_result_selectors = [
                            'div.g',           # Standard Google result
                            'div[data-ved]',   # Alternative selector
                            '.yuRUbf',         # Another common selector
                            'div.tF2Cxc'      # Yet another selector
                        ]
                        
                        search_divs = []
                        for selector in search_result_selectors:
                            search_divs = soup.select(selector)
                            if search_divs:
                                logger.debug(f"Found {len(search_divs)} results with selector: {selector}")
                                break
                        
                        if not search_divs:
                            # DIRECT LINK EXTRACTION: Look for all YouTube/Twitch links in the page
                            logger.debug(f"Using direct link extraction for {platform}")
                            all_links = soup.find_all('a', href=True)
                            platform_links = []
                            
                            for link in all_links:
                                href = link.get('href', '')
                                text = link.get_text().strip()
                                
                                # Check if it's a platform URL
                                is_platform_url = False
                                if platform == 'youtube' and 'youtube.com' in href:
                                    # Filter out non-channel URLs
                                    if any(path in href for path in ['/watch?', '/shorts/', '/playlist?']):
                                        continue
                                    is_platform_url = True
                                elif platform == 'twitch' and 'twitch.tv' in href:
                                    is_platform_url = True
                                
                                if is_platform_url and text and len(text) > 3:
                                    # Clean Google redirect URLs
                                    clean_href = href
                                    if href.startswith('/url?q='):
                                        try:
                                            import urllib.parse
                                            clean_href = urllib.parse.unquote(href.split('/url?q=')[1].split('&')[0])
                                        except Exception:
                                            clean_href = href.split('/url?q=')[1].split('&')[0]
                                    
                                    platform_links.append({
                                        'title': text[:100],  # Limit title length
                                        'url': clean_href,
                                        'snippet': ''
                                    })
                            
                            logger.debug(f"Found {len(platform_links)} {platform} links via direct extraction")
                            
                            # Use these direct links
                            results = platform_links[:max_results]
                        else:
                            # Parse structured results
                            logger.debug(f"Parsing {len(search_divs)} structured results")
                            
                            for i, div in enumerate(search_divs[:max_results * 2]):  # Check more results
                                # Try multiple combinations of selectors
                                title_selectors = ['h3', '.LC20lb', '[role="heading"]', 'h3 span', '.DKV0Md', '.BNeawe.vvjwJb.AP7Wnd']
                                link_selectors = ['a[href]', 'a', '[href]']
                                
                                title_elem = None
                                link_elem = None
                                
                                for title_sel in title_selectors:
                                    title_elem = div.select_one(title_sel)
                                    if title_elem and title_elem.get_text().strip():
                                        break
                                
                                for link_sel in link_selectors:
                                    link_elem = div.select_one(link_sel)
                                    if link_elem and link_elem.get('href'):
                                        break
                                
                                if title_elem and link_elem:
                                    title = title_elem.get_text().strip()
                                    url = link_elem['href']
                                    
                                    # DEBUG: Show what we found
                                    logger.info(f"🔍 DEBUG: Found candidate {i+1}: '{title}' -> '{url}'")
                                    
                                    # Check if it's a platform URL
                                    is_platform_url = False
                                    if platform == 'youtube' and 'youtube.com' in url:
                                        is_platform_url = True
                                    elif platform == 'twitch' and 'twitch.tv' in url:
                                        is_platform_url = True
                                    
                                    if is_platform_url:
                                        # Clean Google redirect URLs
                                        if url.startswith('/url?q='):
                                            try:
                                                import urllib.parse
                                                url = urllib.parse.unquote(url.split('/url?q=')[1].split('&')[0])
                                            except Exception:
                                                url = url.split('/url?q=')[1].split('&')[0]
                                        
                                        snippet_elem = div.select_one('.VwiC3b') or div.select_one('.s') or div.select_one('.st')
                                        snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                                        
                                        logger.info(f"✅ Found {platform} result: '{title}' -> '{url}'")
                                        
                                        results.append({
                                            'title': title,
                                            'url': url,
                                            'snippet': snippet
                                        })
                                        
                                        if len(results) >= max_results:
                                            break
                                else:
                                    # DEBUG: Show why this div was skipped
                                    title_text = title_elem.get_text().strip() if title_elem else "No title"
                                    url_text = link_elem.get('href') if link_elem else "No URL"
                                    logger.debug(f"🔍 DEBUG: Skipped {i+1}: title='{title_text}' url='{url_text}'")
                                    if i < 5:  # Only show first few for debugging
                                        logger.debug(f"🔍 DEBUG: Div {i+1} HTML: {str(div)[:200]}...")
                        
                        if results:
                            logger.info(f"✅ Successfully extracted {len(results)} search results for {query}")
                            for i, result in enumerate(results):
                                logger.info(f"  Result {i+1}: {result['title']} -> {result['url']}")
                            return results
                        else:
                            logger.debug(f"No {platform} results found in structured parsing")
                            
                            # TEXT EXTRACTION FALLBACK: Look for URLs in the raw text content
                            logger.debug(f"Trying text extraction fallback for {query}")
                            text_content = result.extracted_content or result.html
                            
                            import re
                            fallback_results = []
                            
                            if platform == 'youtube':
                                # Look for YouTube URLs in the text
                                youtube_patterns = [
                                    r'https?://(?:www\.)?youtube\.com/(?:c/|channel/|user/|@)[\w\-\.]+',
                                    r'https?://(?:www\.)?youtube\.com/[\w\-\.]+',
                                    r'youtube\.com/(?:c/|channel/|user/|@)[\w\-\.]+',
                                ]
                                
                                for pattern in youtube_patterns:
                                    urls = re.findall(pattern, text_content, re.IGNORECASE)
                                    for url in urls:
                                        if not url.startswith('http'):
                                            url = 'https://' + url
                                        fallback_results.append({
                                            'title': f'YouTube Channel (from text)',
                                            'url': url,
                                            'snippet': ''
                                        })
                                        if len(fallback_results) >= max_results:
                                            break
                                    if len(fallback_results) >= max_results:
                                        break
                                        
                            elif platform == 'twitch':
                                # Look for Twitch URLs in the text
                                twitch_patterns = [
                                    r'https?://(?:www\.)?twitch\.tv/[\w\-\.]+',
                                    r'twitch\.tv/[\w\-\.]+',
                                ]
                                
                                for pattern in twitch_patterns:
                                    urls = re.findall(pattern, text_content, re.IGNORECASE)
                                    for url in urls:
                                        if not url.startswith('http'):
                                            url = 'https://' + url
                                        fallback_results.append({
                                            'title': f'Twitch Channel (from text)',
                                            'url': url,
                                            'snippet': ''
                                        })
                                        if len(fallback_results) >= max_results:
                                            break
                                    if len(fallback_results) >= max_results:
                                        break
                            
                            if fallback_results:
                                logger.debug(f"Text extraction found {len(fallback_results)} {platform} URLs")
                                return fallback_results
                            else:
                                logger.debug(f"No {platform} URLs found even with text extraction")
                                return []
                    else:
                        error_msg = getattr(result, 'error_message', 'Unknown error')
                        logger.warning(f"Crawl4AI failed: {error_msg}")
                        return []
                        
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed with proxy {proxy}: {e}")
                if proxy:
                    self.proxy_manager.mark_proxy_failed(proxy)
                await asyncio.sleep(2)
        
        logger.error(f"All search attempts failed for query: {query}")
        return []


class URLFilter:
    """Filters and validates URLs for specific platforms"""
    
    @staticmethod
    def filter_youtube_url(url: str) -> str:
        """Filter and clean YouTube URLs"""
        if not url:
            return None
        
        # Remove Google redirect
        if url.startswith('/url?q='):
            url = url.split('/url?q=')[1].split('&')[0]
        
        # Validate YouTube URLs
        if "youtube.com" in url:
            # Accept various YouTube URL formats
            valid_patterns = ['/channel/', '/c/', '/@', '/user/']
            if any(pattern in url for pattern in valid_patterns):
                return url
        
        return None
    
    @staticmethod
    def filter_twitch_url(url: str) -> str:
        """Filter and clean Twitch URLs"""
        if not url:
            return None
        
        # Remove Google redirect
        if url.startswith('/url?q='):
            url = url.split('/url?q=')[1].split('&')[0]
        
        # Validate Twitch URLs
        if "twitch.tv" in url and len(url.split('/')) >= 4:
            # Basic Twitch channel URL validation
            return url
        
        return None


class YouTubeTwitchScraper:
    """Main scraper class - modular and clean"""
    
    def __init__(self, data_file: str, proxy_file: str, output_file: str, max_workers: int = 1):
        self.data_file = data_file
        self.proxy_file = proxy_file  
        self.output_file = output_file
        self.max_workers = max(1, min(10, max_workers))  # Clamp between 1 and 10
        self.processed_users = set()
        self._lock = asyncio.Lock()  # For thread-safe operations
        
        # Initialize components
        self.proxy_manager = ProxyManager(proxy_file)
        self.search_engine = SearchEngine(self.proxy_manager)
        self.channel_matcher = ChannelMatcher(self.search_engine.session)
        
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results to resume from where left off"""
        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_csv(self.output_file)
                self.processed_users = set(existing_df['username'].tolist())
                logger.info(f"Found {len(self.processed_users)} already processed users")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
    
    def extract_name_from_url(self, url: str) -> str:
        """Extract potential name from profile URL"""
        try:
            # Common patterns in X/Twitter URLs
            if 'x.com' in url or 'twitter.com' in url:
                # Split by / and find the username part
                parts = url.split('/')
                for i, part in enumerate(parts):
                    if part in ['x.com', 'twitter.com'] and i + 1 < len(parts):
                        username_part = parts[i + 1]
                        # Remove query parameters
                        if '?' in username_part:
                            username_part = username_part.split('?')[0]
                        # Remove @ symbol
                        username_part = username_part.replace('@', '')
                        # Skip common paths that aren't usernames
                        if username_part not in ['home', 'explore', 'notifications', 'messages', 'bookmarks', 'lists', 'profile', 'more', 'compose', 'search', 'settings', 'help']:
                            return username_part.strip()
            else:
                # For other URLs, use the last path segment
                path = url.split('/')[-1] if '/' in url else url
                
                # Remove query parameters
                if '?' in path:
                    path = path.split('?')[0]
                
                # Remove common suffixes
                path = path.replace('@', '')
                
                # Clean up the extracted name
                cleaned_name = path.strip()
                
                if cleaned_name and len(cleaned_name) > 0:
                    return cleaned_name
            
        except Exception as e:
            logger.debug(f"Error extracting name from URL {url}: {e}")
        
        return ""
    
    async def find_channels_for_user(self, username: str, profile_name: str, url: str = ""):
        """Find YouTube and Twitch channels for a user"""
        results = {
            'youtube_url': None,
            'youtube_score': 0,
            'youtube_not_sure': 0,
            'twitch_url': None,
            'twitch_score': 0,
            'twitch_not_sure': 0
        }
        
        # Create search queries
        queries = []
        if username:
            queries.append(username)  # Removed quotes for better results
        if profile_name and profile_name != username:
            queries.append(profile_name)  # Removed quotes for better results
        if username and profile_name:
            queries.append(f'{username} {profile_name}')
        
        # Extract name from URL as fallback if provided
        url_extracted_name = ""
        if url:
            url_extracted_name = self.extract_name_from_url(url)
            if url_extracted_name and url_extracted_name not in [username, profile_name]:
                queries.append(url_extracted_name)  # Removed quotes for better results
        
        # Remove empty queries
        queries = [q for q in queries if q.strip()]
        queries = queries[:3]  # Limit queries to prevent overuse (increased from 2 to 3)
        
        # Search each platform
        for platform in ['youtube', 'twitch']:
            best_match = {'score': 0, 'title': None, 'url': None}
            fallback_url = None
            not_sure = 0  # Track if we used fallback (1) or enhanced matching (0)
            
            for query in queries:
                try:
                    search_results = await self.search_engine.search_with_crawl4ai(query, platform)
                    
                    if search_results:
                        # Store first URL for fallback (no filtering!)
                        if not fallback_url:
                            first_result = search_results[0]
                            fallback_url = first_result.get('url', '')
                        
                        for result in search_results:
                            title = result.get('title', '')
                            url = result.get('url', '')
                            
                            # Filter valid URLs for enhanced matching only
                            if platform == 'youtube':
                                clean_url = URLFilter.filter_youtube_url(url)
                            else:
                                clean_url = URLFilter.filter_twitch_url(url)
                            
                            if not clean_url:
                                continue
                            
                            # Enhanced matching
                            if self.channel_matcher.enhanced_name_match(clean_url, username, profile_name):
                                # Use enhanced matching score
                                match_score = self.channel_matcher.calculate_match_score(username, profile_name, title, clean_url)
                                
                                # Set minimum score for enhanced matches
                                if match_score < 50:
                                    match_score = 50
                                
                                best_match = {
                                    'score': match_score,
                                    'title': title,
                                    'url': clean_url
                                }
                                not_sure = 0  # Enhanced match found
                                
                                # Found a good match, stop searching more results for this query
                                break
                        
                        # Found a good match for this query, stop searching more results
                        if best_match['score'] >= 50:
                            break
                            
                except Exception as e:
                    logger.error(f"Search failed for {query} on {platform}: {e}")
                    continue
                
                # If we found a good match, stop searching other queries
                if best_match['score'] >= 50:
                    break
                
                # Rate limiting between queries
                await asyncio.sleep(random.uniform(2, 4))
            
            # Store best match or fallback
            if best_match['score'] >= 50:
                # Enhanced match found
                if platform == 'youtube':
                    results['youtube_url'] = best_match['url']
                    results['youtube_score'] = best_match['score']
                    results['youtube_not_sure'] = not_sure
                else:
                    results['twitch_url'] = best_match['url']
                    results['twitch_score'] = best_match['score']
                    results['twitch_not_sure'] = not_sure
            elif fallback_url:
                # Use fallback URL when enhanced matching fails
                if platform == 'youtube':
                    results['youtube_url'] = fallback_url
                    results['youtube_score'] = 30
                    results['youtube_not_sure'] = 1
                else:
                    results['twitch_url'] = fallback_url
                    results['twitch_score'] = 30
                    results['twitch_not_sure'] = 1
        
        return results

    async def process_single_user(self, user_data: dict, index: int, total: int):
        """Process a single user - thread-safe version"""
        username = user_data['username']
        profile_name = user_data['profile_name']
        url = user_data['url']
        followers = user_data['followers']
        
        # Check if already processed (thread-safe)
        async with self._lock:
            if username in self.processed_users:
                logger.info(f"Skipping {username} - already processed")
                return
            
        logger.info(f"Processing {index + 1}/{total}: {username}")
        
        try:
            # Find channels
            channels = await self.find_channels_for_user(username, profile_name, url)
            
            # Save result immediately (thread-safe)
            await self.save_result_async(username, profile_name, url, followers, channels)
            
            # Mark as processed (thread-safe)
            async with self._lock:
                self.processed_users.add(username)
                
        except Exception as e:
            logger.error(f"Error processing {username}: {e}")

    async def save_result_async(self, username: str, profile_name: str, url: str, followers: int, channels: dict):
        """Thread-safe async version of save_result"""
        async with self._lock:
            self.save_result(username, profile_name, url, followers, channels)
    
    def save_result(self, username: str, profile_name: str, url: str, followers: int, channels: dict):
        """Save result to CSV immediately"""
        try:
            file_exists = os.path.exists(self.output_file)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'username', 'profile_name', 'url', 'followers',
                    'youtube_url', 'youtube_score', 'youtube_not_sure',
                    'twitch_url', 'twitch_score', 'twitch_not_sure'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'username': username,
                    'profile_name': profile_name,
                    'url': url,
                    'followers': followers,
                    **channels
                })
            
            logger.info(f"✅ Saved result for {username} to {self.output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save result for {username}: {e}")
            # Try to create a backup file
            try:
                backup_file = f"backup_{self.output_file}"
                with open(backup_file, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = [
                        'username', 'profile_name', 'url', 'followers',
                        'youtube_url', 'youtube_score', 'youtube_not_sure',
                        'twitch_url', 'twitch_score', 'twitch_not_sure'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    if not os.path.exists(backup_file):
                        writer.writeheader()
                    
                    writer.writerow({
                        'username': username,
                        'profile_name': profile_name,
                        'url': url,
                        'followers': followers,
                        **channels
                    })
                logger.info(f"✅ Saved backup result for {username} to {backup_file}")
            except Exception as backup_e:
                logger.error(f"❌ Failed to save backup for {username}: {backup_e}")
    
    async def process_users(self):
        """Process all users from the data file with parallel processing"""
        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(df)} users from data file")
            logger.info(f"Using {self.max_workers} parallel workers")
            
            # Filter out already processed users
            unprocessed_data = []
            for index, row in df.iterrows():
                username = row['username']
                if username not in self.processed_users:
                    user_data = {
                        'username': username,
                        'profile_name': row['profile_name'],
                        'url': row['url'],
                        'followers': row['followers']
                    }
                    unprocessed_data.append((index, user_data))
            
            total_to_process = len(unprocessed_data)
            logger.info(f"{total_to_process} users remaining to process")
            
            if total_to_process == 0:
                logger.info("All users already processed!")
                return
            
            # Process users in batches with semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_with_semaphore(index_data_pair):
                async with semaphore:
                    index, user_data = index_data_pair
                    await self.process_single_user(user_data, index, len(df))
                    # Rate limiting between users (adjusted for parallel processing)
                    await asyncio.sleep(random.uniform(1, 3))
            
            # Create tasks for parallel processing
            tasks = [process_with_semaphore(item) for item in unprocessed_data]
            
            # Process tasks in batches to avoid overwhelming the system
            batch_size = self.max_workers * 2  # Process 2 batches worth at a time
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
                await asyncio.gather(*batch, return_exceptions=True)
                
                # Brief pause between batches
                if i + batch_size < len(tasks):
                    await asyncio.sleep(2)
            
            processed_count = len([u for u in self.processed_users if u in df['username'].values])
            logger.info(f"Processing completed. Total processed: {processed_count}")
            
        except Exception as e:
            logger.error(f"Error in process_users: {e}")


async def main():
    """Main async function"""
    data_file = "data/3_snapshot_s_mepo7m7c1bhrdvfkc6_external_links(without_YT_twitch).csv"
    proxy_file = "proxy/Free_Proxy_List.csv"
    output_file = "youtube_twitch_results_enhanced.csv"
    
    # Get worker count from user
    while True:
        try:
            worker_count = int(input("Enter number of parallel workers (1-10): "))
            if 1 <= worker_count <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")
    
    logger.info(f"Starting YouTube/Twitch scraper with {worker_count} parallel workers...")
    scraper = YouTubeTwitchScraper(data_file, proxy_file, output_file, worker_count)
    await scraper.process_users()

if __name__ == "__main__":
    asyncio.run(main())
