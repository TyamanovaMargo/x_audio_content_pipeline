from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.io_utils import read_usernames, write_results_csv
from utils.username_utils import normalize_username

class AccountValidator:
    def __init__(self, max_concurrent=3, log_file="processed_accounts.json"):
        self.max_concurrent = max_concurrent
        self.log_file = log_file
        self.processed_log = self._load_log()

    def check_account_existence(self, username: str) -> str:
        url = f"https://x.com/{username}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                response = page.goto(url, timeout=10000)
                if response is None or response.status >= 400:
                    return 'does_not_exist'

                try:
                    page.wait_for_selector('div[data-testid="primaryColumn"]', timeout=8000)
                    return 'exists'
                except PlaywrightTimeoutError:
                    content = page.content().lower()
                    if 'account suspended' in content:
                        return 'suspended'
                    if 'account doesn’t exist' in content or 'try another' in content:
                        return 'does_not_exist'
                    return 'error'

            except Exception as e:
                print(f"Ошибка при проверке {username}: {e}")
                return 'error'
            finally:
                browser.close()

    def validate_accounts_from_file(self, input_file: str, output_file: str, max_accounts: int=None, force_recheck: bool=False):
        print(f"📖 Reading usernames from {input_file}")
        all_usernames = read_usernames(input_file)
        if max_accounts:
            all_usernames = all_usernames[:max_accounts]

        print(f"📊 Total usernames in input: {len(all_usernames)}")
        print(f"📋 Previously processed accounts: {len(self.processed_log)}")

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
                    print(f"⏭️ Skipping {username} (already processed: {self.processed_log[normalized]['status']})")

            print(f"🆕 New usernames to check: {len(usernames_to_check)}")
            print(f"⏭️ Skipped usernames: {len(all_usernames) - len(usernames_to_check)}")

        if not usernames_to_check:
            print("✅ All usernames have already been processed!")
            self._generate_output_from_log(output_file)
            return self._get_existing_accounts_from_log()

        results = []
        completed = 0
        total = len(usernames_to_check)

        print(f"\n🚀 Starting validation of {total} new accounts...")

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_username = {executor.submit(self.check_one, username): username for username in usernames_to_check}

            for future in as_completed(future_to_username):
                username = future_to_username[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self._add_to_log(result['username'], result['status'])
                    completed += 1
                    self._print_progress(completed, total, username, result['status'])
                    if completed % 10 == 0:
                        self._save_log()
                except Exception as e:
                    print(f"❌ Error processing {username}: {e}")

        self._save_log()

        all_existing_accounts = self._get_all_existing_accounts(results)

        write_results_csv(all_existing_accounts, output_file)

        self._print_summary(len(all_usernames), len(usernames_to_check), len(results), len(all_existing_accounts))

        return all_existing_accounts

    def check_one(self, username: str):
        normalized = normalize_username(username)
        if not normalized:
            return {'username': username, 'status': 'invalid'}
        status = self.check_account_existence(normalized)
        return {'username': normalized, 'status': status}

    def _load_log(self):
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
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving log file: {e}")

    def _add_to_log(self, username: str, status: str):
        self.processed_log[username] = {
            'username': username,
            'status': status,
            'checked_at': datetime.now().isoformat()
        }

    def _generate_output_from_log(self, output_file: str):
        accounts = [{'username': u, 'status': info['status']} for u, info in self.processed_log.items()]
        write_results_csv(accounts, output_file)
        print(f"📁 Generated output from log: {output_file} ({len(accounts)} accounts)")

    def _get_existing_accounts_from_log(self):
        return [{'username': u, 'status': info['status']} for u, info in self.processed_log.items()]

    def _get_all_existing_accounts(self, new_results):
        all_acc = {acc['username']: acc for acc in self._get_existing_accounts_from_log()}
        for acc in new_results:
            all_acc[acc['username']] = acc
        return list(all_acc.values())

    def _print_progress(self, completed, total, username, status):
        status_emoji = {
            'exists': '✅',
            'does_not_exist': '❌',
            'suspended': '🚫',
            'invalid': '⚠️',
            'error': '💥',
            'error_timeout': '⏰'
        }
        emoji = status_emoji.get(status, '❓')
        percentage = (completed / total) * 100
        print(f"{emoji} {completed}/{total} ({percentage:.1f}%) | {username} -> {status}")

    def _print_summary(self, total_input, checked, new_results, total_existing):
        print("\n📈 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total input usernames: {total_input}")
        print(f"🆕 New usernames checked: {checked}")
        print(f"⏭️ Previously processed (skipped): {total_input - checked}")
        print(f"📋 Total in log now: {len(self.processed_log)}")
        print(f"✅ Total accounts processed: {total_existing}")
