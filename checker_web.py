import time
import random
import logging
from typing import Dict, Optional
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError

class XAccountChecker:
    def __init__(self, headless: bool = True, timeout: int = 10000):
        self.headless = headless
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    def check_account_status(self, username: str, url: str, max_retries: int = 2) -> Dict[str, str]:
        """
        Check if X/Twitter account exists using Playwright.
        
        Returns:
            Dict with username, profile_url, and status
        """
        for attempt in range(max_retries + 1):
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=self.headless)
                    context = browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    )
                    page = context.new_page()
                    
                    # Navigate to profile URL
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                    
                    # Wait a moment for dynamic content to load
                    page.wait_for_timeout(2000)
                    
                    status = self._determine_account_status(page, username)
                    
                    browser.close()
                    
                    return {
                        'username': username,
                        'profile_url': url,
                        'status': status
                    }
                    
            except PlaywrightTimeoutError:
                self.logger.warning(f"Timeout for {username} (attempt {attempt + 1})")
                if attempt == max_retries:
                    return {
                        'username': username,
                        'profile_url': url,
                        'status': 'error_timeout'
                    }
                time.sleep(2)  # Brief pause before retry
                
            except Exception as e:
                self.logger.error(f"Error checking {username}: {str(e)}")
                if attempt == max_retries:
                    return {
                        'username': username,
                        'profile_url': url,
                        'status': 'error'
                    }
                time.sleep(2)
    
    # def _determine_account_status(self, page: Page, username: str) -> str:
    #     """
    #     Determine account status based on page content.
        
    #     Returns:
    #         'exists', 'does_not_exist', 'suspended', or 'private'
    #     """
    #     try:
    #         # Check for "This account doesn't exist" message
    #         not_exist_locator = page.get_by_text("This account doesn't exist", exact=False)
    #         try:
    #             not_exist_locator.wait_for(timeout=3000)
    #             return 'does_not_exist'
    #         except PlaywrightTimeoutError:
    #             pass
            
    #         # Check for "Account suspended" message
    #         suspended_locator = page.get_by_text("Account suspended", exact=False)
    #         try:
    #             suspended_locator.wait_for(timeout=3000)
    #             return 'suspended'
    #         except PlaywrightTimeoutError:
    #             pass
            
    #         # Check for "These Tweets are protected" (private account)
    #         protected_locator = page.get_by_text("These Tweets are protected", exact=False)
    #         try:
    #             protected_locator.wait_for(timeout=3000)
    #             return 'exists'  # Private accounts still exist
    #         except PlaywrightTimeoutError:
    #             pass
            
    #         # Check for profile indicators showing account exists
    #         # Look for the @username handle in the profile
    #         handle_locator = page.get_by_text(f"@{username}", exact=False)
    #         try:
    #             handle_locator.wait_for(timeout=5000)
                
    #             # Additional check: look for profile structure elements
    #             # Check if we can find a profile header or user info section
    #             profile_indicators = [
    #                 'div[data-testid="UserName"]',
    #                 'div[data-testid="UserDescription"]',
    #                 'div[data-testid="primaryColumn"]'
    #             ]
                
    #             for indicator in profile_indicators:
    #                 try:
    #                     page.wait_for_selector(indicator, timeout=2000)
    #                     return 'exists'
    #                 except PlaywrightTimeoutError:
    #                     continue
                
    #             return 'exists'  # Found @username, likely exists
                
    #         except PlaywrightTimeoutError:
    #             pass
            
    #         # If none of the above conditions are met, check page title
    #         title = page.title()
    #         if "not found" in title.lower() or "doesn't exist" in title.lower():
    #             return 'does_not_exist'
            
    #         # Default to unknown if we can't determine
    #         return 'unknown'
            
    #     except Exception as e:
    #         self.logger.error(f"Error determining status for {username}: {str(e)}")
    #         return 'error'
    def _determine_account_status(self, page: Page, username: str) -> str:
        try:
            print(f"\n🔍 DEBUGGING: {username}")
            print(f"📍 URL: {page.url}")
            
            # Wait for page load
            page.wait_for_timeout(5000)
            
            # Get page info
            page_title = page.title()
            print(f"📄 Page Title: '{page_title}'")
            
            # Take screenshot for visual inspection
            page.screenshot(path=f"debug_{username}.png")
            print(f"📸 Screenshot saved: debug_{username}.png")
            
            # Check what text is actually on the page
            page_text = page.inner_text('body')
            print(f"📝 Page contains key phrases:")
            print(f"   - 'doesn't exist': {'✅' if 'doesn\'t exist' in page_text.lower() else '❌'}")
            print(f"   - 'suspended': {'✅' if 'suspended' in page_text.lower() else '❌'}")
            print(f"   - '@{username}': {'✅' if f'@{username}' in page_text.lower() else '❌'}")
            
            # Your existing detection logic here...
            return 'exists'  # Temporary - replace with actual logic
            
        except Exception as e:
            print(f"❌ ERROR for {username}: {e}")
            return 'error'

    def add_random_delay(self, min_delay: float, max_delay: float):
        """Add randomized delay between requests."""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
