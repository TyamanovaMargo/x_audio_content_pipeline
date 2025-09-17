import asyncio
import csv
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import random
import time
from utils.io_utils import read_usernames, write_results_csv
from utils.username_utils import normalize_username

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionManager:
    """Enhanced session manager with better account switching and state tracking."""
    
    def __init__(self):
        self.primary_account = None
        self.backup_account = None
        self.current_context = None
        self.is_using_backup = False
        self.login_attempts = 0
        self.max_login_attempts = 2
        self.is_authenticated = False
        self.last_activity = datetime.now()
    
    def set_accounts(self, primary_login, primary_password, backup_login=None, backup_password=None):
        if primary_login and primary_password:
            self.primary_account = (primary_login, primary_password)
            logger.info(f"Primary account configured: {primary_login}")
        
        if backup_login and backup_password:
            self.backup_account = (backup_login, backup_password)
            logger.info(f"Backup account configured: {backup_login}")
    
    async def switch_to_backup(self, browser: Browser):
        """Switch to backup account with new context."""
        if not self.backup_account or self.is_using_backup:
            logger.warning("Backup account unavailable or already in use")
            return False
        
        logger.info("Switching to backup account...")
        self.is_using_backup = True
        self.login_attempts = 0
        self.is_authenticated = False
        
        if self.current_context:
            await self.current_context.close()
        
        # Create new context with different fingerprint
        self.current_context = await browser.new_context(
            user_agent=self.get_random_user_agent(),
            viewport={'width': random.randint(1200, 1400), 'height': random.randint(800, 1000)},
            locale=random.choice(['en-US', 'en-GB', 'en-CA']),
            timezone_id=random.choice(['America/New_York', 'Europe/London', 'America/Los_Angeles'])
        )
        
        logger.info("New context created for backup account")
        return True
    
    def get_random_user_agent(self) -> str:
        agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'
        ]
        return random.choice(agents)
    
    def should_switch_account(self) -> bool:
        return (self.login_attempts >= self.max_login_attempts and 
                self.backup_account and not self.is_using_backup)

class ProgressManager:
    """Enhanced progress manager with better error handling."""
    
    def __init__(self, filename='output/processed_accounts.json'):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        self.processed = set()
        self.failed = set()
        self.stats = {'total': 0, 'success': 0, 'failed': 0}
        self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.processed = set(data.get('processed', []))
                    self.failed = set(data.get('failed', []))
                    self.stats = data.get('stats', {'total': 0, 'success': 0, 'failed': 0})
                    logger.info(f"Progress loaded: {len(self.processed)} processed, {len(self.failed)} failed")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
    
    def save_progress(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump({
                    'processed': list(self.processed),
                    'failed': list(self.failed),
                    'stats': self.stats,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def is_processed(self, username: str) -> bool:
        return username in self.processed
    
    def mark_processed(self, username: str, success: bool = True):
        self.processed.add(username)
        if success:
            self.stats['success'] += 1
        else:
            self.stats['failed'] += 1
            self.failed.add(username)
        
        if len(self.processed) % 5 == 0:
            self.save_progress()

class AccountValidator:
    def __init__(self, max_concurrent=1, log_file="output/processed_accounts.json", 
                 login=None, password=None, backup_login=None, backup_password=None):
        self.max_concurrent = max_concurrent
        self.log_file = log_file
        self.progress_manager = ProgressManager(log_file)
        self.session_manager = SessionManager()
        
        # Set up authentication if provided
        if login and password:
            self.session_manager.set_accounts(login, password, backup_login, backup_password)
    
    async def human_like_behavior(self, page: Page, intensity: str = "normal"):
        """Enhanced human-like behavior with different intensity levels."""
        try:
            if intensity == "light":
                await page.mouse.move(random.randint(100, 300), random.randint(100, 300))
                await asyncio.sleep(random.uniform(0.5, 1.0))
            elif intensity == "normal":
                await page.mouse.move(random.randint(100, 500), random.randint(100, 400))
                await asyncio.sleep(random.uniform(1.0, 2.0))
                await page.mouse.move(random.randint(200, 600), random.randint(200, 500))
                await asyncio.sleep(random.uniform(0.5, 1.5))
                await page.evaluate('window.scrollBy(0, Math.floor(Math.random() * 200));')
                await asyncio.sleep(random.uniform(0.3, 1.0))
        except Exception as e:
            logger.warning(f"Error in human-like behavior: {e}")

    async def verify_account_by_following_click(self, page: Page, username: str) -> bool:
        """Verify account exists by attempting to click Following button."""
        try:
            logger.info(f"🔍 Verifying {username} by clicking Following button...")
            
            # Wait a bit for page to fully load
            await asyncio.sleep(random.uniform(2.0, 4.0))
            
            # Try different selectors for Following button/link
            following_selectors = [
                'a[href$="/following"]',
                'a[href*="/following"]',
                '[data-testid="UserProfileHeader_Items"] a[href*="following"]',
                'a:has-text("Following")',
                'div:has-text("Following")',
                'span:has-text("Following")'
            ]
            
            for selector in following_selectors:
                try:
                    following_element = await page.query_selector(selector)
                    if following_element:
                        # Check if element is clickable
                        is_clickable = await following_element.is_enabled()
                        if is_clickable:
                            logger.info(f"✅ Found clickable Following element: {selector}")
                            
                            # Perform the click
                            await following_element.click()
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                            
                            # Check if we navigated to following page or got some response
                            current_url = page.url
                            if '/following' in current_url or 'following' in current_url.lower():
                                logger.info(f"✅ Successfully clicked Following - navigated to: {current_url}")
                                return True
                            else:
                                # Even if URL didn't change, the click worked which means account exists
                                logger.info(f"✅ Following button clicked successfully (account verified)")
                                return True
                        else:
                            logger.info(f"Following element found but not clickable: {selector}")
                except Exception as e:
                    logger.debug(f"Error with selector {selector}: {e}")
                    continue
            
            logger.warning(f"⚠️ No clickable Following button found for {username}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying account by Following click: {e}")
            return False

    async def check_account_simple(self, page: Page, username: str) -> Dict:
        """Simplified account checking with better detection avoidance."""
        url = f'https://x.com/{username}'
        result = {
            "username": username,
            "status": "error",
            "last_activity": None,
            "follower_count": None,
            "following_count": None,
            "verification": "unknown"
        }
        
        try:
            logger.info(f"Checking account: {username}")
            
            # Navigate to profile
            response = await page.goto(url, timeout=30000)
            await asyncio.sleep(random.uniform(4.0, 8.0))
            await self.human_like_behavior(page, "light")
            
            # Check response status
            if response and response.status == 404:
                result['status'] = 'does_not_exist'
                logger.info(f"❌ Account {username} does not exist (404)")
                return result
            
            # Get page content for analysis
            content = await page.content()
            content_lower = content.lower()
            
            # Check for various account states
            if "this account doesn't exist" in content_lower or "page does not exist" in content_lower:
                result['status'] = 'does_not_exist'
                logger.info(f"❌ Account {username} does not exist")
                return result
            
            if "account suspended" in content_lower:
                result['status'] = 'suspended'
                logger.info(f"🚫 Account {username} is suspended")
                return result
            
            if any(phrase in content_lower for phrase in ["tweets are protected", "protected tweets"]):
                result['status'] = 'protected'
                logger.info(f"🔒 Account {username} is protected")
                return result
            
            # Look for profile indicators
            profile_indicators = [
                '[data-testid="UserName"]',
                '[data-testid="UserDescription"]',
                '[data-testid="UserAvatar"]',
                '[data-testid="UserProfileHeader"]'
            ]
            
            profile_found = False
            for indicator in profile_indicators:
                if await page.query_selector(indicator):
                    profile_found = True
                    logger.info(f"✅ Profile element found: {indicator}")
                    break
            
            if profile_found:
                # Try to extract basic info safely
                await self.extract_basic_info(page, result)
                
                # Additional verification by clicking Following button
                if await self.verify_account_by_following_click(page, username):
                    result['status'] = 'exists'
                    logger.info(f"✅ Account {username} exists")
                else:
                    result['status'] = 'requires_auth'
                    logger.info(f"🔑 Account {username} may require authentication")
            else:
                result['status'] = 'requires_auth'
                logger.info(f"🔑 Account {username} may require authentication")
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Error checking {username}: {e}")
            return result

    async def extract_basic_info(self, page: Page, result: Dict):
        """Safely extract basic account information."""
        try:
            # Extract follower/following counts
            count_selectors = [
                'a[href*="/followers"] span',
                'a[href*="/following"] span',
                '[data-testid="UserProfileHeader_Items"] a span'
            ]
            
            for selector in count_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if text and any(char.isdigit() for char in text):
                            # Simple number extraction
                            numbers = ''.join(filter(str.isdigit, text))
                            if numbers:
                                count = int(numbers)
                                parent = await element.evaluate('el => el.closest("a")')
                                if parent:
                                    href = await parent.get_attribute('href')
                                    if href and '/followers' in href:
                                        result['follower_count'] = count
                                    elif href and '/following' in href:
                                        result['following_count'] = count
                except:
                    continue
            
            # Check verification status
            verification_selectors = [
                '[data-testid="verificationBadge"]',
                'svg[aria-label="Verified account"]'
            ]
            
            for selector in verification_selectors:
                if await page.query_selector(selector):
                    result['verification'] = "verified"
                    break
            else:
                result['verification'] = "not_verified"
            
        except Exception as e:
            logger.warning(f"Error extracting basic info: {e}")

    async def worker(self, sem: asyncio.Semaphore, context: BrowserContext, username: str) -> Dict:
        """Worker function with enhanced rate limiting."""
        if self.progress_manager.is_processed(username):
            return {"username": username, "status": "skipped"}
        
        async with sem:
            # Enhanced delays for better stealth
            await asyncio.sleep(random.uniform(5.0, 12.0))
            
            page = await context.new_page()
            try:
                result = await self.check_account_simple(page, username)
                self.progress_manager.mark_processed(username, result['status'] != 'error')
                return result
            finally:
                await page.close()
                await asyncio.sleep(random.uniform(2.0, 5.0))

    def validate_accounts_from_file(self, input_file: str, output_file: str, 
                                  max_accounts: int = None, force_recheck: bool = False):
        """Main validation function that integrates with the pipeline."""
        return asyncio.run(self._async_validate_accounts_from_file(
            input_file, output_file, max_accounts, force_recheck
        ))

    async def _async_validate_accounts_from_file(self, input_file: str, output_file: str, 
                                               max_accounts: int = None, force_recheck: bool = False):
        """Async validation implementation."""
        logger.info(f"📖 Reading usernames from {input_file}")
        all_usernames = read_usernames(input_file)
        
        if max_accounts:
            all_usernames = all_usernames[:max_accounts]

        logger.info(f"📊 Total usernames in input: {len(all_usernames)}")
        logger.info(f"📋 Previously processed accounts: {len(self.progress_manager.processed)}")

        # Filter already processed if not forcing recheck
        if not force_recheck:
            usernames_to_check = [u for u in all_usernames if not self.progress_manager.is_processed(u)]
            logger.info(f"📝 Accounts to check: {len(usernames_to_check)}")
        else:
            usernames_to_check = all_usernames
            logger.info("🔄 Force recheck enabled - processing all accounts")

        if not usernames_to_check:
            logger.info("✅ All accounts already processed!")
            return []

        # Start Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # Create main context
            context = await browser.new_context(
                user_agent=self.session_manager.get_random_user_agent(),
                viewport={'width': random.randint(1280, 1400), 'height': random.randint(800, 1000)},
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            self.session_manager.current_context = context

            # Process accounts with conservative settings in batches
            sem = asyncio.Semaphore(1)  # Very conservative: only 1 concurrent request
            results = []
            batch_size = 20  # Process in batches of 20
            
            logger.info(f"🚀 Starting validation (concurrency: 1, batch size: {batch_size})")
            
            # Process in batches
            for batch_start in range(0, len(usernames_to_check), batch_size):
                batch_usernames = usernames_to_check[batch_start:batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(usernames_to_check) + batch_size - 1) // batch_size
                
                logger.info(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_usernames)} accounts)")
                
                # Create tasks for current batch
                batch_tasks = [asyncio.create_task(self.worker(sem, context, username)) 
                              for username in batch_usernames]
                
                # Process batch results
                for i, task in enumerate(asyncio.as_completed(batch_tasks), 1):
                    try:
                        result = await task
                        if result['status'] != 'skipped':
                            results.append(result)
                        
                        status_emoji = {
                            'exists': '✅',
                            'does_not_exist': '❌',
                            'suspended': '🚫',
                            'protected': '🔒',
                            'requires_auth': '🔑',
                            'error': '⚠️',
                            'skipped': '⏭️'
                        }.get(result['status'], '❓')
                        
                        extra_info = []
                        if result.get('following_count') is not None:
                            extra_info.append(f"Following: {result['following_count']}")
                        if result.get('follower_count') is not None:
                            extra_info.append(f"Followers: {result['follower_count']}")
                        if result.get('verification') == 'verified':
                            extra_info.append("✓")
                        
                        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
                        global_index = batch_start + i
                        
                        print(f"{status_emoji} [{global_index}/{len(usernames_to_check)}] {result['username']} -> {result['status']}{extra_str}")
                        
                    except Exception as e:
                        logger.error(f"Task error: {e}")
                        results.append({"username": "unknown", "status": "error"})
                
                # Save intermediate progress
                if batch_start % (batch_size * 5) == 0:  # Save every 5 batches
                    temp_output = output_file.replace('.csv', f'_checkpoint_{batch_num}.csv')
                    existing_accounts = [r for r in results if r.get('status') == 'exists']
                    if existing_accounts:
                        self._save_results(existing_accounts, temp_output)
                        logger.info(f"💾 Checkpoint save: {temp_output} ({len(existing_accounts)} existing accounts)")
                
                # Pause between batches (except for the last batch)
                if batch_start + batch_size < len(usernames_to_check):
                    pause_time = random.randint(120, 300)  # 2-5 minutes pause
                    logger.info(f"⏳ Pausing between batches: {pause_time} seconds...")
                    await asyncio.sleep(pause_time)
            
            # Save final results - only existing accounts
            self.progress_manager.save_progress()
            existing_accounts = [result for result in results if result.get('status') == 'exists']
            self._save_results(existing_accounts, output_file)
            
            logger.info(f"🎉 Validation complete! Results saved to {output_file}")
            logger.info(f"📊 Found {len(existing_accounts)} existing accounts out of {len(results)} total processed")
            await browser.close()
            
            return existing_accounts

    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to CSV file - only accounts with 'exists' status."""
        try:
            # Simplified fieldnames - only username and status
            fieldnames = ['username', 'status']
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Clean up data for CSV - only essential fields
                    row = {
                        'username': result.get('username', ''),
                        'status': result.get('status', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Results saved to {output_file} - {len(results)} existing accounts only")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _load_log(self):
        """Legacy method for compatibility."""
        return self.progress_manager.processed
