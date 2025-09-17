#!/usr/bin/env python3

import asyncio
import csv
import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import random
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('x_checker.log'),
        logging.StreamHandler()
    ]
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
    
    def __init__(self, filename='x_checker_progress.json'):
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
        
        if len(self.processed) % 5 == 0:  # Save more frequently
            self.save_progress()

async def human_like_behavior(page: Page, intensity: str = "normal"):
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
        elif intensity == "heavy":
            for _ in range(random.randint(2, 4)):
                await page.mouse.move(random.randint(100, 800), random.randint(100, 600))
                await asyncio.sleep(random.uniform(0.5, 1.0))
            await page.evaluate('window.scrollBy(0, Math.floor(Math.random() * 300));')
            await asyncio.sleep(random.uniform(2.0, 4.0))
    except Exception as e:
        logger.warning(f"Error in human-like behavior: {e}")

async def wait_for_element(page: Page, selectors: List[str], timeout: int = 5000, visible: bool = True):
    """Enhanced element waiting with multiple selectors."""
    for selector in selectors:
        try:
            element = await page.wait_for_selector(selector, timeout=timeout, state='visible' if visible else 'attached')
            if element:
                return element, selector
        except:
            continue
    return None, None

async def login_x(context: BrowserContext, headless: bool, login: str, password: str, session_manager: SessionManager) -> BrowserContext:
    """Completely rewritten login function with modern selectors and robust verification."""
    page = await context.new_page()
    
    try:
        account_type = "backup" if session_manager.is_using_backup else "primary"
        logger.info(f"Starting login process ({account_type} account)")
        
        # Navigate to login page
        await page.goto('https://x.com/i/flow/login', timeout=30000)
        await asyncio.sleep(random.uniform(3.0, 6.0))
        
        logger.info(f"Current URL: {page.url}")
        logger.info(f"Page title: {await page.title()}")
        
        # Step 1: Find and fill username field
        username_selectors = [
            'input[autocomplete="username"]',
            'input[name="text"]',
            'input[data-testid="ocfEnterTextTextInput"]',
            'input[type="text"]:not([style*="display: none"])',
            'input[placeholder*="username"]',
            'input[placeholder*="email"]'
        ]
        
        username_input, used_selector = await wait_for_element(page, username_selectors, timeout=10000)
        
        if not username_input:
            logger.error("Username input field not found")
            session_manager.login_attempts += 1
            if not headless:
                logger.info("Browser remains open for manual login")
                input("Please log in manually and press Enter to continue...")
            await page.close()
            return context
        
        logger.info(f"Username field found: {used_selector}")
        await username_input.clear()
        await username_input.fill(login)
        await human_like_behavior(page, "normal")
        
        # Step 2: Click Next button
        next_selectors = [
            'div[role="button"] span:text("Next")',
            'button:text("Next")',
            'div[role="button"]:has-text("Next")',
            '[data-testid="LoginForm_Login_Button"]',
            'div[role="button"]:has-text("Далее")',
            'button[type="button"]:has-text("Next")'
        ]
        
        next_button, next_selector = await wait_for_element(page, next_selectors, timeout=8000)
        if next_button:
            await next_button.click()
            logger.info(f"Next button clicked: {next_selector}")
        else:
            logger.warning("Next button not found, continuing...")
        
        await asyncio.sleep(random.uniform(3.0, 5.0))
        
        # Step 3: Handle potential username verification or challenges
        await handle_username_challenges(page, headless, session_manager)
        
        # Step 4: Find and fill password field
        password_selectors = [
            'input[autocomplete="current-password"]',
            'input[name="password"]',
            'input[type="password"]',
            'input[data-testid="ocfEnterTextTextInput"]:not([autocomplete="username"])',
            'input[placeholder*="password"]'
        ]
        
        password_input, pass_selector = await wait_for_element(page, password_selectors, timeout=15000)
        
        if not password_input:
            logger.error("Password input field not found")
            session_manager.login_attempts += 1
            if not headless:
                logger.info("Browser remains open for manual login")
                input("Please log in manually and press Enter to continue...")
            await page.close()
            return context
        
        logger.info(f"Password field found: {pass_selector}")
        await password_input.clear()
        await password_input.fill(password)
        await human_like_behavior(page, "normal")
        
        # Step 5: Click login button
        login_selectors = [
            'div[role="button"] span:text("Log in")',
            'button:text("Log in")',
            'div[role="button"]:has-text("Log in")',
            'div[role="button"]:has-text("Войти")',
            '[data-testid="LoginForm_Login_Button"]',
            'button[type="submit"]'
        ]
        
        login_button, login_selector = await wait_for_element(page, login_selectors, timeout=8000)
        if login_button:
            await login_button.click()
            logger.info(f"Login button clicked: {login_selector}")
        else:
            logger.warning("Login button not found, but continuing...")
        
        # Step 6: Wait for login processing
        logger.info("Waiting for login to process...")
        await asyncio.sleep(random.uniform(8.0, 12.0))
        
        # Step 7: Handle 2FA and other challenges
        if await handle_2fa_and_challenges(page, headless, session_manager):
            await page.close()
            return context
        
        # Step 8: Check for login errors
        if await check_login_errors(page, session_manager):
            await page.close()
            return context
        
        # Step 9: Verify successful login
        success = await verify_login_success(page, session_manager)
        
        if success:
            logger.info(f"✅ Successful login to X.com ({account_type} account)")
            session_manager.login_attempts = 0
            session_manager.is_authenticated = True
        else:
            logger.warning("Could not verify successful login, but continuing...")
            session_manager.login_attempts += 1
            logger.info(f"Final URL: {page.url}")
        
        await page.close()
        return context
        
    except Exception as e:
        logger.error(f"Critical error during login: {e}")
        session_manager.login_attempts += 1
        
        if not headless:
            logger.info("Browser remains open for debugging")
            input("Check browser state and press Enter...")
        
        try:
            await page.close()
        except:
            pass
        return context

async def handle_username_challenges(page: Page, headless: bool, session_manager: SessionManager):
    """Handle username verification challenges."""
    try:
        # Look for unusual activity warnings
        warning_selectors = [
            'text="There was unusual login activity"',
            'text="Help us verify that it\'s you"',
            'text="Unusual activity"'
        ]
        
        for selector in warning_selectors:
            if await page.query_selector(selector):
                logger.warning(f"Unusual activity warning detected: {selector}")
                if not headless:
                    input("Please handle the verification manually and press Enter...")
                    await asyncio.sleep(3)
                break
    except Exception as e:
        logger.warning(f"Error handling username challenges: {e}")

async def handle_2fa_and_challenges(page: Page, headless: bool, session_manager: SessionManager) -> bool:
    """Handle 2FA and other login challenges."""
    try:
        # Look for various challenge indicators
        challenge_selectors = [
            'input[name="challenge_response"]',
            'input[data-testid="ocfEnterTextTextInput"][placeholder*="code"]',
            'input[placeholder*="verification"]',
            'input[placeholder*="Code"]',
            'text="Enter your phone number"',
            'text="Check your email"'
        ]
        
        for selector in challenge_selectors:
            element = await page.query_selector(selector)
            if element:
                logger.info(f"Challenge detected: {selector}")
                if headless:
                    logger.error("Challenge requires manual input. Run with --no-headless")
                    session_manager.login_attempts += 1
                    return True
                else:
                    logger.info("Please complete the challenge manually in the browser")
                    input("Press Enter after completing the challenge...")
                    await asyncio.sleep(5)
                    break
        
        return False
        
    except Exception as e:
        logger.error(f"Error handling challenges: {e}")
        return False

async def check_login_errors(page: Page, session_manager: SessionManager) -> bool:
    """Check for login errors."""
    try:
        error_indicators = [
            "Sorry, we could not authenticate you",
            "Wrong password",
            "Your account has been locked",
            "Too many attempts",
            "Неправильное имя пользователя",
            "Неверный пароль"
        ]
        
        page_content = await page.content()
        
        for error_text in error_indicators:
            if error_text.lower() in page_content.lower():
                logger.error(f"Login error detected: {error_text}")
                session_manager.login_attempts += 1
                return True
        
        # Check for error elements
        error_selectors = [
            '[data-testid="LoginForm_Error"]',
            '[role="alert"]',
            '.error-message'
        ]
        
        for selector in error_selectors:
            error_element = await page.query_selector(selector)
            if error_element:
                error_text = await error_element.inner_text()
                logger.error(f"Login error: {error_text}")
                session_manager.login_attempts += 1
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking login errors: {e}")
        return False

async def verify_login_success(page: Page, session_manager: SessionManager) -> bool:
    """Verify successful login using multiple methods."""
    try:
        current_url = page.url
        logger.info(f"Verifying login success. Current URL: {current_url}")
        
        # Method 1: URL-based verification (most reliable)
        success_url_patterns = [
            'https://x.com/home',
            'https://x.com/',
            'https://twitter.com/home',
            'https://twitter.com/'
        ]
        
        for pattern in success_url_patterns:
            if current_url.startswith(pattern):
                logger.info("✅ Login verified by URL")
                return True
        
        # Method 2: Check if not on login pages
        if not any(keyword in current_url.lower() for keyword in ['login', 'flow', 'authenticate', 'sessions']):
            logger.info("✅ Login verified: no longer on login page")
            return True
        
        # Method 3: Look for authenticated elements
        success_selectors = [
            '[data-testid="AppTabBar_Home_Link"]',
            '[data-testid="SideNav_NewTweet_Button"]',
            '[data-testid="SideNav_AccountSwitcher_Button"]',
            '[aria-label="Home timeline"]',
            'nav[role="navigation"][aria-label="Primary"]',
            '[data-testid="primaryColumn"]',
            'aside[aria-label="Who to follow"]',
            'main[role="main"]'
        ]
        
        for selector in success_selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=3000)
                if element:
                    logger.info(f"✅ Login verified by element: {selector}")
                    return True
            except:
                continue
        
        # Method 4: Check page title
        page_title = await page.title()
        if any(title in page_title.lower() for title in ['home', 'x']):
            if 'login' not in page_title.lower():
                logger.info(f"✅ Login verified by title: {page_title}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error verifying login success: {e}")
        return False

async def check_account_simple(page: Page, username: str) -> Dict:
    """Simplified account checking with better detection avoidance."""
    url = f'https://x.com/{username}'
    result = {
        "username": username,
        "status": "error",
        "last_activity": None,
        "external_links": [],
        "follower_count": None,
        "following_count": None,
        "verification": "unknown"
    }
    
    try:
        logger.info(f"Checking account: {username}")
        
        # Navigate to profile
        response = await page.goto(url, timeout=30000)
        await asyncio.sleep(random.uniform(4.0, 8.0))
        await human_like_behavior(page, "light")
        
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
            await extract_basic_info(page, result)
            result['status'] = 'exists'
            logger.info(f"✅ Account {username} exists")
        else:
            result['status'] = 'requires_auth'
            logger.info(f"🔑 Account {username} may require authentication")
        
        return result
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"Error checking {username}: {e}")
        return result

async def extract_basic_info(page: Page, result: Dict):
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
        
        # Extract external links safely
        link_selectors = [
            '[data-testid="UserUrl"] a',
            '[data-testid="UserDescription"] a[href^="http"]'
        ]
        
        external_links = []
        for selector in link_selectors:
            try:
                links = await page.query_selector_all(selector)
                for link in links:
                    href = await link.get_attribute('href')
                    if href and not any(domain in href.lower() for domain in ['x.com', 'twitter.com', 't.co']):
                        external_links.append(href)
            except:
                continue
        
        result['external_links'] = list(set(external_links))  # Remove duplicates
        
    except Exception as e:
        logger.warning(f"Error extracting basic info: {e}")

async def worker(sem: asyncio.Semaphore, context: BrowserContext, username: str, progress_manager: ProgressManager) -> Dict:
    """Worker function with enhanced rate limiting."""
    if progress_manager.is_processed(username):
        return {"username": username, "status": "skipped"}
    
    async with sem:
        # Enhanced delays for better stealth
        await asyncio.sleep(random.uniform(5.0, 12.0))
        
        page = await context.new_page()
        try:
            result = await check_account_simple(page, username)
            progress_manager.mark_processed(username, result['status'] != 'error')
            return result
        finally:
            await page.close()
            await asyncio.sleep(random.uniform(2.0, 5.0))

def read_usernames_from_file(filepath: str) -> List[str]:
    """Enhanced file reading with better error handling."""
    usernames = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            if filepath.endswith('.csv'):
                reader = csv.DictReader(infile)
                
                if 'username' not in reader.fieldnames:
                    logger.error(f"CSV file missing 'username' column")
                    logger.info(f"Available columns: {reader.fieldnames}")
                    sys.exit(1)
                
                for row in reader:
                    username = row['username'].strip()
                    if username:
                        username = username.lstrip('@')  # Remove @ if present
                        usernames.append(username)
            else:
                for line in infile:
                    username = line.strip()
                    if username:
                        username = username.lstrip('@')
                        usernames.append(username)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_usernames = []
        for username in usernames:
            if username not in seen:
                seen.add(username)
                unique_usernames.append(username)
        
        logger.info(f"Loaded {len(unique_usernames)} unique usernames")
        return unique_usernames
        
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        sys.exit(1)

def save_results(results: List[Dict], output_file: str):
    """Enhanced results saving with better error handling."""
    try:
        fieldnames = ['username', 'status', 'last_activity', 'external_links', 
                     'following_count', 'follower_count', 'verification']
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Clean up data for CSV
                row = {
                    'username': result.get('username', ''),
                    'status': result.get('status', ''),
                    'last_activity': result.get('last_activity', ''),
                    'external_links': '; '.join(result.get('external_links', [])),
                    'following_count': result.get('following_count', ''),
                    'follower_count': result.get('follower_count', ''),
                    'verification': result.get('verification', '')
                }
                writer.writerow(row)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_statistics(results: List[Dict]):
    """Print detailed statistics."""
    stats = {}
    verified_count = 0
    total_links = 0
    
    for result in results:
        status = result.get('status', 'unknown')
        stats[status] = stats.get(status, 0) + 1
        
        if result.get('verification') == 'verified':
            verified_count += 1
        
        if result.get('external_links'):
            total_links += len(result['external_links'])
    
    print("\n📈 Final Statistics:")
    status_emojis = {
        'exists': '✅',
        'does_not_exist': '❌',
        'suspended': '🚫',
        'protected': '🔒',
        'requires_auth': '🔑',
        'error': '⚠️',
        'skipped': '⏭️'
    }
    
    for status, count in stats.items():
        emoji = status_emojis.get(status, '❓')
        print(f"  {emoji} {status}: {count}")
    
    print(f"\n🔍 Additional Statistics:")
    print(f"  ✅ Verified accounts: {verified_count}")
    print(f"  🔗 Total external links found: {total_links}")
    print(f"  📊 Success rate: {(stats.get('exists', 0) / max(len(results), 1) * 100):.1f}%")

async def main():
    """Enhanced main function with better flow control."""
    parser = argparse.ArgumentParser(description="X.com Account Status Checker - Enhanced Version")
    parser.add_argument('--input', '-i', required=True, help='Input file with usernames (CSV with "username" column or TXT)')
    parser.add_argument('--output', '-o', default='x_results.csv', help='Output CSV file (default: x_results.csv)')
    parser.add_argument('--login', type=str, help='Primary X login (username/email)')
    parser.add_argument('--password', type=str, help='Primary X password')
    parser.add_argument('--backup-login', type=str, help='Backup X login (username/email)')
    parser.add_argument('--backup-password', type=str, help='Backup X password')
    parser.add_argument('--max-concurrent', '-c', type=int, default=1, help='Max concurrent checks (default: 1)')
    parser.add_argument('--no-headless', action='store_true', help='Run browser visibly (for debugging/2FA)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize managers
    progress_manager = ProgressManager()
    session_manager = SessionManager()
    
    # Get credentials
    login_cred = args.login or os.getenv('X_LOGIN')
    password_cred = args.password or os.getenv('X_PASSWORD')
    backup_login = args.backup_login or os.getenv('X_BACKUP_LOGIN')
    backup_password = args.backup_password or os.getenv('X_BACKUP_PASSWORD')
    
    session_manager.set_accounts(login_cred, password_cred, backup_login, backup_password)
    
    if (login_cred and not password_cred) or (password_cred and not login_cred):
        logger.error("Both login and password must be provided together")
        sys.exit(1)
    
    # Check input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Read usernames
    usernames = read_usernames_from_file(args.input)
    
    # Filter already processed if resuming
    if args.resume:
        original_count = len(usernames)
        usernames = [u for u in usernames if not progress_manager.is_processed(u)]
        logger.info(f"Resuming: {original_count - len(usernames)} already processed")
        logger.info(f"Remaining to process: {len(usernames)} users")
    else:
        logger.info(f"Total users to check: {len(usernames)}")
    
    if not usernames:
        logger.info("✅ All users already processed!")
        return
    
    # Start Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.no_headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        # Create main context
        context = await browser.new_context(
            user_agent=session_manager.get_random_user_agent(),
            viewport={'width': random.randint(1280, 1400), 'height': random.randint(800, 1000)},
            locale='en-US',
            timezone_id='America/New_York'
        )
        
        session_manager.current_context = context
        
        # Attempt login if credentials provided
        if session_manager.primary_account:
            logger.info("🔐 Attempting login to primary account...")
            context = await login_x(context, not args.no_headless,
                                  session_manager.primary_account[0],
                                  session_manager.primary_account[1],
                                  session_manager)
            session_manager.current_context = context
        
        # Process in batches with conservative settings
        sem = asyncio.Semaphore(1)  # Very conservative: only 1 concurrent request
        batch_size = min(args.batch_size, 20)
        results = []
        processed_count = 0
        
        logger.info(f"🚀 Starting processing (concurrency: 1, batch size: {batch_size})")
        
        for batch_start in range(0, len(usernames), batch_size):
            batch_usernames = usernames[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            
            logger.info(f"\n📦 Processing batch {batch_num} ({len(batch_usernames)} accounts)")
            
            # Check if account switching is needed
            if session_manager.should_switch_account():
                logger.warning("Too many failed login attempts, switching to backup account...")
                if await session_manager.switch_to_backup(browser):
                    context = session_manager.current_context
                    context = await login_x(context, not args.no_headless,
                                          session_manager.backup_account[0],
                                          session_manager.backup_account[1],
                                          session_manager)
                    session_manager.current_context = context
                else:
                    logger.error("Failed to switch to backup account")
            
            # Create tasks for batch
            tasks = [asyncio.create_task(worker(sem, session_manager.current_context, username, progress_manager))
                    for username in batch_usernames]
            
            # Process batch results
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    result = await task
                    if result['status'] != 'skipped':
                        results.append(result)
                        processed_count += 1
                    
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
                    if result.get('external_links'):
                        extra_info.append(f"Links: {len(result['external_links'])}")
                    if result.get('verification') == 'verified':
                        extra_info.append("✓")
                    
                    extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
                    account_info = "🔄 Backup" if session_manager.is_using_backup else ""
                    
                    print(f"{status_emoji} [{processed_count}/{len(usernames)}] {result['username']} -> {result['status']}{extra_str} {account_info}")
                    
                except Exception as e:
                    logger.error(f"Task error: {e}")
                    results.append({"username": "unknown", "status": "error"})
            
            # Save intermediate results
            if results:
                temp_output = args.output.replace('.csv', f'_batch_{batch_num}.csv')
                save_results(results, temp_output)
                logger.info(f"💾 Intermediate save: {temp_output}")
            
            # Pause between batches (longer for better stealth)
            if batch_start + batch_size < len(usernames):
                pause_time = random.randint(120, 300)  # 2-5 minutes
                logger.info(f"⏳ Pausing between batches: {pause_time} seconds...")
                await asyncio.sleep(pause_time)
        
        # Final save and statistics
        progress_manager.save_progress()
        save_results(results, args.output)
        print_statistics(results)
        
        logger.info(f"🎉 Processing complete! Results saved to {args.output}")
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
