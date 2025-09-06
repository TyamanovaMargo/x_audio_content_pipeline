#!/usr/bin/env python3
"""
Stage 3.5: YouTube & Twitch Channel Discovery
Integrates the youtube-twitch-x-scraper functionality into the main pipeline
"""

import pandas as pd
import asyncio
import os
import logging
import sys
from typing import List, Dict

# Add the youtube-twitch-x-scraper directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'youtube-twitch-x-scraper'))

try:
    from youtube_twitch_scraper import YouTubeTwitchScraper, ProxyManager, SearchEngine, ChannelMatcher
    from enhanced_matching import EnhancedMatcher
except ImportError as e:
    print(f"⚠️ Warning: Could not import YouTube/Twitch scraper components: {e}")
    print("💡 Make sure the youtube-twitch-x-scraper folder contains all required files")

class ChannelDiscovery:
    """Stage 3.5: Discover YouTube and Twitch channels for profiles"""
    
    def __init__(self, output_dir="output", max_workers=3):
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Proxy file path (relative to scraper folder)
        self.proxy_file = os.path.join("youtube-twitch-x-scraper", "proxy", "Free_Proxy_List.csv")
        
        # Check if proxy file exists
        if not os.path.exists(self.proxy_file):
            self.logger.warning(f"⚠️ Proxy file not found: {self.proxy_file}")
            self.logger.info("💡 Channel discovery will work without proxies but may be slower")
            self.proxy_file = None
    
    def discover_channels_for_snapshot(self, snapshot_id: str, profiles_data: List[Dict]) -> str:
        """
        Discover YouTube/Twitch channels for profiles and save enhanced CSV
        
        Args:
            snapshot_id: The snapshot ID for naming output files
            profiles_data: List of profile dictionaries from Stage 3
            
        Returns:
            Path to the enhanced CSV file with discovered channels
        """
        
        print(f"\n🔍 STAGE 3.5: YouTube & Twitch Channel Discovery")
        print("-" * 60)
        print(f"📊 Processing {len(profiles_data)} profiles for channel discovery")
        print(f"🔄 Max workers: {self.max_workers}")
        print(f"🆔 Snapshot ID: {snapshot_id}")
        
        # Save input data as temporary CSV for the scraper
        temp_input_file = os.path.join(self.output_dir, f"temp_{snapshot_id}_input.csv")
        pd.DataFrame(profiles_data).to_csv(temp_input_file, index=False)
        
        # Output file path - match the scraper config output filename
        enhanced_output_file = os.path.join(self.output_dir, "youtube_twitch_results_enhanced.csv")
        
        try:
            # Run the channel discovery
            discovered_results = asyncio.run(self._run_channel_discovery(temp_input_file, enhanced_output_file))
            
            # Clean up temporary file
            if os.path.exists(temp_input_file):
                os.remove(temp_input_file)
            
            if discovered_results:
                print(f"✅ Stage 3.5 completed successfully!")
                print(f"📁 Enhanced CSV saved: {enhanced_output_file}")
                print(f"🎯 YouTube channels found: {discovered_results['youtube_found']}")
                print(f"🎯 Twitch channels found: {discovered_results['twitch_found']}")
                print(f"📈 Discovery rate: {discovered_results['discovery_rate']:.1f}%")
                
                return enhanced_output_file
            else:
                print(f"❌ Stage 3.5 failed - no results generated")
                return None
                
        except Exception as e:
            print(f"❌ Stage 3.5 error: {e}")
            # Clean up temporary file on error
            if os.path.exists(temp_input_file):
                os.remove(temp_input_file)
            return None
    
    async def _run_channel_discovery(self, input_file: str, output_file: str) -> Dict:
        """Run the actual channel discovery process"""
        
        try:
            # Initialize the scraper components
            if self.proxy_file and os.path.exists(self.proxy_file):
                proxy_manager = ProxyManager(self.proxy_file)
                print(f"📡 Proxy manager initialized with {len(proxy_manager.proxies)} proxies")
            else:
                proxy_manager = ProxyManager("")  # Empty proxy manager
                print("📡 Running without proxies")
            
            search_engine = SearchEngine(proxy_manager)
            
            # Create a simplified scraper for integration
            scraper = SimplifiedYouTubeTwitchScraper(
                data_file=input_file,
                output_file=output_file,
                max_workers=self.max_workers,
                search_engine=search_engine
            )
            
            # Run the discovery process
            results = await scraper.process_all_users()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Channel discovery failed: {e}")
            return None


class SimplifiedYouTubeTwitchScraper:
    """Simplified version of the YouTube/Twitch scraper for pipeline integration"""
    
    def __init__(self, data_file: str, output_file: str, max_workers: int = 3, search_engine=None):
        self.data_file = data_file
        self.output_file = output_file
        self.max_workers = max_workers
        self.search_engine = search_engine
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize channel matcher
        if search_engine:
            self.channel_matcher = ChannelMatcher(search_engine.session)
        else:
            self.channel_matcher = EnhancedMatcher()
    
    async def process_all_users(self) -> Dict:
        """Process all users and discover their channels"""
        
        # Load the input data
        try:
            df = pd.read_csv(self.data_file)
            users = df.to_dict('records')
            self.logger.info(f"📥 Loaded {len(users)} users from {self.data_file}")
        except Exception as e:
            self.logger.error(f"Failed to load input file: {e}")
            return None
        
        # Initialize result tracking
        results = []
        youtube_found = 0
        twitch_found = 0
        
        # Process users (simplified - no complex async batching for now)
        for i, user in enumerate(users, 1):
            username = user.get('username', '')
            profile_name = user.get('profile_name', '')
            
            print(f"🔍 [{i}/{len(users)}] Processing: {username}")
            
            try:
                # Find channels for this user
                channels = await self._find_channels_for_user(username, profile_name)
                
                # Add results to user data
                user_result = user.copy()
                user_result.update(channels)
                results.append(user_result)
                
                # Count discoveries
                if channels.get('youtube_url'):
                    youtube_found += 1
                if channels.get('twitch_url'):
                    twitch_found += 1
                    
            except Exception as e:
                self.logger.warning(f"Error processing {username}: {e}")
                # Add user without channels
                user_result = user.copy()
                user_result.update({
                    'youtube_url': '',
                    'youtube_score': 0,
                    'twitch_url': '',
                    'twitch_score': 0
                })
                results.append(user_result)
        
        # Save results
        try:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.output_file, index=False)
            self.logger.info(f"💾 Saved results to {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return None
        
        # Calculate statistics
        total_channels = youtube_found + twitch_found
        discovery_rate = (total_channels / len(users)) * 100 if users else 0
        
        return {
            'total_users': len(users),
            'youtube_found': youtube_found,
            'twitch_found': twitch_found,
            'total_channels': total_channels,
            'discovery_rate': discovery_rate
        }
    
    async def _find_channels_for_user(self, username: str, profile_name: str = "") -> Dict:
        """Find YouTube and Twitch channels for a single user"""
        
        result = {
            'youtube_url': '',
            'youtube_score': 0,
            'twitch_url': '',
            'twitch_score': 0
        }
        
        if not username:
            return result
        
        # Search queries
        queries = []
        if username:
            queries.append(username)
        if profile_name and profile_name != username:
            queries.append(profile_name)
        
        # Limit to prevent overuse
        queries = queries[:2]
        
        # Search each platform
        for platform in ['youtube', 'twitch']:
            best_match = {'score': 0, 'url': None}
            
            for query in queries:
                try:
                    if self.search_engine:
                        # Use search engine if available
                        search_results = await self.search_engine.search_with_crawl4ai(query, platform, max_results=3)
                    else:
                        # Fallback to simple search (placeholder)
                        search_results = []
                    
                    for search_result in search_results:
                        title = search_result.get('title', '')
                        url = search_result.get('url', '')
                        
                        if url and self.channel_matcher:
                            # Use enhanced matching to verify
                            is_match = self.channel_matcher.enhanced_name_match(url, username, profile_name)
                            if is_match:
                                score = 85 if query == username else 50
                                if score > best_match['score']:
                                    best_match = {'score': score, 'url': url}
                                    break
                    
                    # Add small delay between searches
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.debug(f"Search failed for {query} on {platform}: {e}")
                    continue
            
            # Store best match for this platform
            if best_match['url']:
                if platform == 'youtube':
                    result['youtube_url'] = best_match['url']
                    result['youtube_score'] = best_match['score']
                else:
                    result['twitch_url'] = best_match['url']
                    result['twitch_score'] = best_match['score']
        
        return result


# Main function for standalone testing
def main():
    """Test the channel discovery functionality"""
    
    # Example usage
    discovery = ChannelDiscovery(output_dir="output", max_workers=2)
    
    # Sample data (replace with actual Stage 3 output)
    sample_profiles = [
        {
            'username': 'kirstnicolexo',
            'profile_name': 'kirstie (taylor\'s version)',
            'url': 'http://brokenblame.tumblr.com/',
            'followers': 350,
            'bio': ''
        },
        {
            'username': 'JumpersJump',
            'profile_name': 'Jump Start Jumpers',
            'url': 'https://x.com/jumpersjump',
            'followers': 1,
            'bio': ''
        }
    ]
    
    # Run discovery
    result_file = discovery.discover_channels_for_snapshot("test_snapshot", sample_profiles)
    
    if result_file:
        print(f"✅ Test completed! Results saved to: {result_file}")
    else:
        print("❌ Test failed!")


if __name__ == "__main__":
    main()
