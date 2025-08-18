import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.checker_web import XAccountChecker
from utils.io_utils import read_usernames, write_results_csv
from utils.username_utils import normalize_username, build_profile_url

class AccountValidator:
    def __init__(self, max_concurrent=3, delay_min=1.5, delay_max=3.5, log_file="processed_accounts.json"):
        self.max_concurrent = max_concurrent
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.checker = XAccountChecker(headless=True)
        self.log_file = log_file
        self.processed_log = self._load_log()

    def validate_accounts_from_file(self, input_file: str, output_file: str, max_accounts: int = None, force_recheck: bool = False):
        """Validate accounts with skipping already processed ones"""
        print(f"📖 Reading usernames from {input_file}")
        
        all_usernames = read_usernames(input_file)
        if max_accounts:
            all_usernames = all_usernames[:max_accounts]
        
        print(f"📊 Total usernames in input: {len(all_usernames)}")
        print(f"📋 Previously processed accounts: {len(self.processed_log)}")
        
        # Filter already processed accounts
        if force_recheck:
            usernames_to_check = all_usernames
            print(f"🔄 Force recheck enabled - processing all {len(usernames_to_check)} usernames")
        else:
            usernames_to_check = []
            for username in all_usernames:
                normalized = normalize_username(username)
                if normalized and normalized not in self.processed_log:
                    usernames_to_check.append(username)
                elif normalized:
                    print(f"⏭️  Skipping {username} (already processed: {self.processed_log[normalized]['status']})")
            
            print(f"🆕 New usernames to check: {len(usernames_to_check)}")
            print(f"⏭️  Skipped usernames: {len(all_usernames) - len(usernames_to_check)}")
        
        if not usernames_to_check:
            print("✅ All usernames have already been processed!")
            print("💡 Use --force-recheck to reprocess all usernames")
            
            # Generate output file from existing log
            self._generate_output_from_log(output_file)
            return self._get_existing_accounts_from_log()
        
        # Check new accounts
        results = []
        completed = 0
        total = len(usernames_to_check)
        
        print(f"\n🚀 Starting validation of {total} new accounts...")
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_username = {
                executor.submit(self.check_one, username): username 
                for username in usernames_to_check
            }
            
            for future in as_completed(future_to_username):
                username = future_to_username[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        # Add to log
                        self._add_to_log(result['username'], result['status'], result['profile_url'])
                        completed += 1
                        
                        # Progress
                        self._print_progress(completed, total, username, result['status'])
                        
                        # Periodic log save (every 10 checks)
                        if completed % 10 == 0:
                            self._save_log()
                            
                except Exception as e:
                    print(f"❌ Error processing {username}: {e}")
        
        # Final log save
        self._save_log()
        
        # Combine new results with existing ones
        all_existing_accounts = self._get_all_existing_accounts(results)
        
        # Save results
        write_results_csv(all_existing_accounts, output_file)
        
        # Statistics
        self._print_summary(len(all_usernames), len(usernames_to_check), len(results), len(all_existing_accounts))
        
        return all_existing_accounts

    def check_one(self, username: str):
        """Check a single account"""
        normalized = normalize_username(username)
        if not normalized:
            return {'username': username, 'profile_url': '', 'status': 'invalid'}

        url = build_profile_url(normalized)
        result = self.checker.check_account_status(normalized, url)
        self.checker.add_random_delay(self.delay_min, self.delay_max)
        return result

    def _load_log(self):
        """Load processed accounts log"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                print(f"📚 Loaded {len(log_data)} processed accounts from {self.log_file}")
                return log_data
            except Exception as e:
                print(f"⚠️ Error loading log file {self.log_file}: {e}")
                return {}
        else:
            print(f"📄 No existing log file found, creating new one: {self.log_file}")
            return {}

    def _save_log(self):
        """Save processed accounts log"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving log file: {e}")

    def _add_to_log(self, username: str, status: str, profile_url: str = ""):
        """Add account to log"""
        normalized = normalize_username(username)
        if normalized:
            self.processed_log[normalized] = {
                'original_username': username,
                'status': status,
                'profile_url': profile_url,
                'checked_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

    def _generate_output_from_log(self, output_file: str):
        """Generate output file from existing log"""
        existing_accounts = []
        
        for username, info in self.processed_log.items():
            if info['status'] == 'exists':
                existing_accounts.append({
                    'username': username,
                    'profile_url': info['profile_url'],
                    'status': info['status']
                })
        
        write_results_csv(existing_accounts, output_file)
        print(f"📁 Generated output from log: {output_file} ({len(existing_accounts)} existing accounts)")

    def _get_existing_accounts_from_log(self):
        """Get existing accounts from log"""
        existing_accounts = []
        
        for username, info in self.processed_log.items():
            if info['status'] == 'exists':
                existing_accounts.append({
                    'username': username,
                    'profile_url': info['profile_url'],
                    'status': info['status']
                })
        
        return existing_accounts

    def _get_all_existing_accounts(self, new_results):
        """Combine new results with existing ones from log"""
        # New existing accounts
        new_existing = [r for r in new_results if r['status'] == 'exists']
        
        # Existing from log
        log_existing = self._get_existing_accounts_from_log()
        
        # Combine, avoiding duplicates
        all_existing = {}
        
        # First from log
        for acc in log_existing:
            all_existing[acc['username']] = acc
        
        # Then new ones (will overwrite if duplicates)
        for acc in new_existing:
            all_existing[acc['username']] = acc
        
        return list(all_existing.values())

    def _print_progress(self, completed: int, total: int, username: str, status: str):
        """Show progress"""
        percentage = (completed / total) * 100
        status_emoji = {
            'exists': '✅',
            'does_not_exist': '❌',
            'suspended': '🚫',
            'invalid': '⚠️',
            'error': '💥',
            'error_timeout': '⏰'
        }
        
        emoji = status_emoji.get(status, '❓')
        print(f"{emoji} {completed}/{total} ({percentage:.1f}%) | {username} -> {status}")

    def _print_summary(self, total_input: int, checked: int, new_results: int, total_existing: int):
        """Show final summary"""
        print(f"\n📈 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total input usernames: {total_input}")
        print(f"🆕 New usernames checked: {checked}")
        print(f"⏭️  Previously processed (skipped): {total_input - checked}")
        print(f"📋 Total in log now: {len(self.processed_log)}")
        print(f"✅ Total existing accounts: {total_existing}")
        print(f"📁 Log file: {self.log_file}")

    def get_log_stats(self):
        """Get log statistics"""
        stats = {}
        for info in self.processed_log.values():
            status = info['status']
            stats[status] = stats.get(status, 0) + 1
        return stats

    def clear_log(self):
        """Clear log (careful!)"""
        self.processed_log = {}
        self._save_log()
        print("🗑️ Log cleared!")

    def show_log_summary(self):
        """Show log summary"""
        stats = self.get_log_stats()
        print(f"\n📚 LOG SUMMARY ({len(self.processed_log)} total)")
        print("-" * 30)
        for status, count in stats.items():
            print(f"  {status}: {count}")

