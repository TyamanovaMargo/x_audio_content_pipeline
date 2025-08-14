# import requests
# import os
# from dotenv import load_dotenv

# load_dotenv()
# api_token = os.getenv('BRIGHT_DATA_API_TOKEN')

# # Test the trigger endpoint (POST request)
# def test_trigger_endpoint():
#     url = "https://api.brightdata.com/datasets/v3/trigger"
#     headers = {
#         "Authorization": f"Bearer {api_token}",
#         "Content-Type": "application/json"
#     }
    
#     # Parameters for X.com dataset
#     params = {
#         "dataset_id": "gd_lwxmeb2u1cniijd7t4",  # X.com dataset ID
#         "include_errors": "true",
#         "type": "discover_new",
#         "discover_by": "user_name"
#     }
    
#     # Test data - simple profile to check
#     data = [{"user_name": "elonmusk"}]
    
#     try:
#         print("Testing trigger endpoint...")
#         response = requests.post(url, headers=headers, params=params, json=data)
#         print(f"Status: {response.status_code}")
        
#         if response.status_code == 200:
#             result = response.json()
#             snapshot_id = result.get('snapshot_id')
#             print(f"✅ API Token works!")
#             print(f"🆔 Snapshot ID: {snapshot_id}")
#             print(f"📊 Response: {result}")
#             return True
#         elif response.status_code == 403:
#             print("❌ API token authentication failed")
#             print(f"Error: {response.text}")
#             return False
#         elif response.status_code == 404:
#             print("❌ Endpoint not found")
#             print(f"Error: {response.text}")
#             return False
#         else:
#             print(f"❌ Unexpected error: {response.status_code}")
#             print(f"Error: {response.text}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Request failed: {e}")
#         return False

# # Run the test
# if __name__ == "__main__":
#     test_trigger_endpoint()




# import requests

# headers = {
#     "Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d"
# }

# response = requests.get(
#     "https://api.brightdata.com/datasets/v3/snapshot/s_me5jswdd1rafvdunir",
#     headers=headers
# )

# print(f"Status: {response.status_code}")
# print(f"Response: {response.json()}")

# """
# Comprehensive Bright Data API Test Script
# Tests authentication, snapshot creation, polling, and external links extraction
# Optimized for your X.com bio collection workflow
# """

# import os
# import time
# import requests
# import sys
# import json
# import re
# from datetime import datetime

# # Load environment variables
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     print("Warning: python-dotenv not installed, using system env only")

# # Configuration from your setup
# DATASET_ID = os.getenv("BRIGHT_DATA_DATASET_ID", "gd_lwxmeb2u1cniijd7t4")
# API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")
# BASE_URL = "https://api.brightdata.com/datasets/v3"
# HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}

# def print_test_header(title):
#     print(f"\n{'='*60}")
#     print(f" {title}")
#     print(f"{'='*60}")

# def test_step(step_num, description):
#     print(f"\n[TEST {step_num}] {description}")

# def pass_test(message):
#     print(f"✅ PASS: {message}")
#     return True

# def fail_test(message):
#     print(f"❌ FAIL: {message}")
#     return False

# def warn_test(message):
#     print(f"⚠️  WARN: {message}")

# class BrightDataAPITester:
#     def __init__(self):
#         self.snapshot_id = None
#         self.test_results = []
    
#     def test_environment(self):
#         """Test 1: Environment and credentials check"""
#         test_step(1, "Environment & Credentials Check")
        
#         if not API_TOKEN:
#             self.test_results.append(fail_test("BRIGHT_DATA_API_TOKEN not found in environment"))
#             return False
        
#         if len(API_TOKEN) < 40:
#             self.test_results.append(fail_test("API token seems too short - check if it's complete"))
#             return False
            
#         self.test_results.append(pass_test(f"API Token found (length: {len(API_TOKEN)} chars)"))
#         self.test_results.append(pass_test(f"Dataset ID: {DATASET_ID}"))
#         return True
    
#     def test_authentication(self):
#         """Test 2: API authentication"""
#         test_step(2, "Authentication Test")
        
#         try:
#             url = f"{BASE_URL}/snapshots"
#             response = requests.get(url, headers=HEADERS, timeout=30)
            
#             print(f"GET {url}")
#             print(f"Response Status: {response.status_code}")
            
#             if response.status_code == 401:
#                 self.test_results.append(fail_test("401 Unauthorized - API token invalid or expired"))
#                 return False
#             elif response.status_code == 403:
#                 self.test_results.append(fail_test("403 Forbidden - Token valid but no dataset access"))
#                 return False
#             elif response.status_code in [200, 404]:
#                 self.test_results.append(pass_test("Authentication successful"))
#                 return True
#             else:
#                 warn_test(f"Unexpected status {response.status_code}: {response.text[:200]}")
#                 return True
                
#         except requests.exceptions.RequestException as e:
#             self.test_results.append(fail_test(f"Network error: {e}"))
#             return False
    
#     def test_snapshot_creation(self):
#         """Test 3: Snapshot creation (bio-only like your workflow)"""
#         test_step(3, "Snapshot Creation Test (Bio Collection)")
        
#         try:
#             url = f"{BASE_URL}/trigger"
#             params = {
#                 "dataset_id": DATASET_ID,
#                 "include_errors": "true"
#             }
            
#             # Оптимальный профиль для тестирования (средний размер)
#             payload = [{
#                 "url": "https://x.com/vercel",  # ~500K подписчиков, быстрый
#                 "max_number_of_posts": 0      # Только био-информация
#             }]
            
#             print(f"POST {url}")
#             print(f"Params: {params}")
#             print(f"Payload: {json.dumps(payload, indent=2)}")
            
#             response = requests.post(url, headers=HEADERS, params=params, json=payload, timeout=30)
            
#             print(f"Response Status: {response.status_code}")
#             print(f"Response Body: {response.text[:300]}")
            
#             if response.status_code == 401:
#                 self.test_results.append(fail_test("401 during snapshot creation"))
#                 return False
#             elif response.status_code == 400:
#                 self.test_results.append(fail_test(f"400 Bad Request: {response.text}"))
#                 return False
            
#             response.raise_for_status()
            
#             data = response.json()
#             self.snapshot_id = data.get("snapshot_id")
            
#             if not self.snapshot_id:
#                 self.test_results.append(fail_test("No snapshot_id in response"))
#                 return False
                
#             self.test_results.append(pass_test(f"Snapshot created: {self.snapshot_id}"))
#             return True
            
#         except requests.exceptions.RequestException as e:
#             self.test_results.append(fail_test(f"Request error: {e}"))
#             return False
#         except json.JSONDecodeError:
#             self.test_results.append(fail_test(f"Invalid JSON response: {response.text}"))
#             return False
    
#     def test_snapshot_polling(self, max_wait=300):  # 5 минут для среднего профиля
#         """Test 4: Snapshot status polling with enhanced progress tracking"""
#         test_step(4, f"Snapshot Polling Test (max wait: {max_wait}s)")
        
#         if not self.snapshot_id:
#             self.test_results.append(fail_test("No snapshot_id to poll"))
#             return False
        
#         status_url = f"{BASE_URL}/snapshot/{self.snapshot_id}"
#         start_time = time.time()
#         last_status = None
#         none_count = 0
#         total_checks = 0
        
#         print(f"Polling: {status_url}")
#         print("📊 Progress tracking enabled - you'll see regular updates...")
#         print(f"⏰ Target completion time: 1-3 minutes for medium profile (~500K followers)")
#         print("-" * 60)
        
#         while time.time() - start_time < max_wait:
#             try:
#                 response = requests.get(status_url, headers=HEADERS, timeout=30)
#                 elapsed = int(time.time() - start_time)
#                 total_checks += 1
                
#                 # Progress bar calculation
#                 progress_percent = min(int((elapsed / max_wait) * 100), 100)
#                 progress_bar = "█" * (progress_percent // 5) + "░" * (20 - (progress_percent // 5))
                
#                 # Check if snapshot still exists
#                 if response.status_code == 404:
#                     print(f"\n❌ [{elapsed}s] Snapshot expired/cleaned up")
#                     self.test_results.append(fail_test(f"Snapshot expired after {elapsed}s"))
#                     return False
#                 elif response.status_code == 401:
#                     self.test_results.append(fail_test("401 during polling"))
#                     return False
#                 elif response.status_code == 202:
#                     print(f"🔄 [{elapsed}s] HTTP 202: Still processing | {progress_bar} {progress_percent}%")
#                 elif response.status_code != 200:
#                     print(f"⚠️  [{elapsed}s] HTTP {response.status_code}: {response.text[:100]}")
                
#                 try:
#                     data = response.json()
#                 except json.JSONDecodeError:
#                     print(f"❌ [{elapsed}s] Invalid JSON response - continuing...")
#                     time.sleep(15)
#                     continue
                
#                 current_status = data.get("status")
                
#                 # Enhanced None status tracking with progress
#                 if current_status is None:
#                     none_count += 1
#                     # Show progress indicators for None status
#                     if elapsed < 60:
#                         phase = "🔄 Initializing"
#                     elif elapsed < 120:
#                         phase = "⚙️  Processing"
#                     elif elapsed < 180:
#                         phase = "🔍 Analyzing"
#                     else:
#                         phase = "⏳ Finalizing"
                    
#                     print(f"{phase} [{elapsed}s] Status: None (#{none_count}) | {progress_bar} {progress_percent}%")
                    
#                     # Show milestone messages
#                     if elapsed in [60, 120, 180]:
#                         if elapsed == 60:
#                             print("   💡 1 minute elapsed - initialization phase normal")
#                         elif elapsed == 120:
#                             print("   💡 2 minutes elapsed - bio processing in progress")
#                         elif elapsed == 180:
#                             print("   💡 3 minutes elapsed - still processing, please wait")
                    
#                     # Check for additional status fields
#                     extra_info = []
#                     if 'message' in data:
#                         extra_info.append(f"Message: {data['message']}")
#                     if 'progress' in data:
#                         extra_info.append(f"Progress: {data['progress']}")
#                     if 'error' in data:
#                         print(f"❌ [{elapsed}s] ERROR: {data['error']}")
#                         self.test_results.append(fail_test(f"Snapshot error: {data['error']}"))
#                         return False
                    
#                     if extra_info:
#                         print(f"   ℹ️  {' | '.join(extra_info)}")
                        
#                 else:
#                     # Status changed from None - major progress!
#                     if current_status != last_status:
#                         print(f"✅ [{elapsed}s] Status: {current_status} | {progress_bar} {progress_percent}%")
#                         if last_status is None and current_status == "running":
#                             print("   🎉 Status changed from None to running - great progress!")
#                         last_status = current_status
#                         none_count = 0
                    
#                     # Check completion states
#                     if current_status == "completed":
#                         print(f"🎉 [{elapsed}s] COMPLETED! Total processing time: {elapsed}s")
#                         print(f"📊 Total status checks: {total_checks}")
#                         self.test_results.append(pass_test(f"Snapshot completed in {elapsed}s"))
#                         return True
#                     elif current_status in ["failed", "expired", "cancelled"]:
#                         print(f"❌ [{elapsed}s] Status: {current_status}")
#                         self.test_results.append(fail_test(f"Snapshot {current_status} after {elapsed}s"))
#                         return False
                
#                 # Smart sleep timing based on elapsed time
#                 if elapsed < 60:
#                     sleep_time = 10  # Check every 10s in first minute
#                 elif elapsed < 120:
#                     sleep_time = 15  # Every 15s in first 2 minutes  
#                 else:
#                     sleep_time = 20  # Every 20s after 2 minutes
                
#                 # Show next check time
#                 next_check = elapsed + sleep_time
#                 print(f"   ⏱️  Next check in {sleep_time}s (at {next_check}s total)")
#                 print("-" * 60)
                
#                 time.sleep(sleep_time)
                
#             except requests.exceptions.RequestException as e:
#                 elapsed = int(time.time() - start_time)
#                 print(f"❌ [{elapsed}s] Request error: {e}")
#                 print("   🔄 Retrying in 20 seconds...")
#                 time.sleep(20)
        
#         # Timeout reached
#         final_elapsed = int(time.time() - start_time)
#         print(f"\n⏰ Polling timeout after {final_elapsed}s")
#         print(f"📊 Total status checks performed: {total_checks}")
#         print(f"🔄 Last known status: {last_status or 'None'}")
        
#         if final_elapsed > 240:  # Если больше 4 минут для среднего профиля
#             warn_test(f"Profile took longer than expected ({final_elapsed}s) - may indicate API slowness")
#         else:
#             warn_test(f"Polling timeout after {max_wait}s - snapshot may still be processing")
#         return True  # Don't fail entire test suite
    
#     def test_data_download(self):
#         """Test 5: Data download and external links extraction"""
#         test_step(5, "Data Download & External Links Test")
        
#         if not self.snapshot_id:
#             self.test_results.append(fail_test("No snapshot_id for download"))
#             return False
        
#         try:
#             url = f"{BASE_URL}/snapshot/{self.snapshot_id}/download"
#             response = requests.get(url, headers=HEADERS, timeout=60)
            
#             print(f"GET {url}")
#             print(f"Status: {response.status_code}")
            
#             if response.status_code == 401:
#                 self.test_results.append(fail_test("401 during download"))
#                 return False
#             elif response.status_code != 200:
#                 self.test_results.append(fail_test(f"Download failed: {response.status_code}"))
#                 return False
            
#             try:
#                 data = response.json()
#                 print(f"Downloaded {len(data)} items")
                
#                 if not data:
#                     warn_test("No data returned - profile might be private or error occurred")
#                     return True
                
#                 # Test external links extraction (your main use case)
#                 profile = data[0]
#                 external_links = []
                
#                 # Check primary fields (from your optimized extraction)
#                 for field in ['external_link', 'website']:
#                     link = profile.get(field, '').strip()
#                     if link and link.startswith('http'):
#                         external_links.append({'type': 'primary', 'url': link})
#                         break
                
#                 # Check bio description if no primary links
#                 if not external_links:
#                     bio = profile.get('description', '')
#                     if bio and 'http' in bio:
#                         match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', bio)
#                         if match:
#                             clean_url = match.group().rstrip('.,;:!?)')
#                             external_links.append({'type': 'bio', 'url': clean_url})
                
#                 print(f"Profile: {profile.get('profile_name', 'N/A')}")
#                 print(f"Bio: {profile.get('description', 'N/A')[:100]}...")
#                 print(f"Website field: {profile.get('website', 'N/A')}")
#                 print(f"Followers: {profile.get('followers', 'N/A')}")
#                 print(f"External links found: {len(external_links)}")
                
#                 for link in external_links:
#                     print(f"  • {link['url']} ({link['type']})")
                
#                 if external_links:
#                     self.test_results.append(pass_test(f"External links extraction successful: {len(external_links)} links"))
#                 else:
#                     warn_test("No external links found - profile may not have links in bio")
                
#                 self.test_results.append(pass_test("Data download and processing successful"))
#                 return True
                
#             except json.JSONDecodeError:
#                 self.test_results.append(fail_test(f"Invalid JSON in download: {response.text[:200]}"))
#                 return False
                
#         except requests.exceptions.RequestException as e:
#             self.test_results.append(fail_test(f"Download request error: {e}"))
#             return False
    
#     def run_all_tests(self):
#         """Run complete API health check"""
#         print_test_header("BRIGHT DATA API HEALTH CHECK")
#         print("Testing your external links collection workflow...")
#         print("🎯 Optimized for bio-only collection with progress tracking")
        
#         start_time = time.time()
        
#         # Run all tests
#         tests = [
#             self.test_environment,
#             self.test_authentication, 
#             self.test_snapshot_creation,
#             self.test_snapshot_polling,
#             self.test_data_download
#         ]
        
#         for test in tests:
#             if not test():
#                 print(f"\n⚠️  Test {test.__name__} failed - stopping test suite")
#                 break  # Stop on critical failure
#             time.sleep(1)  # Brief pause between tests
        
#         # Results summary
#         elapsed = int(time.time() - start_time)
#         passed = sum(1 for result in self.test_results if result)
#         total = len(self.test_results)
        
#         print_test_header("TEST RESULTS SUMMARY")
#         print(f"⏱️  Total time: {elapsed}s")
#         print(f"📊 Tests passed: {passed}/{total}")
        
#         if passed == total:
#             print("🎉 ALL TESTS PASSED - Your API is working perfectly!")
#             print("✅ Ready for production external links collection")
#             print("💡 You can now use your main script with confidence")
#         elif passed >= total - 1:
#             print("⚠️  Most tests passed - minor issues detected")
#             print("✅ API is functional but may need timeout adjustments")
#         else:
#             print("❌ Multiple tests failed - check the issues above")
#             print("🔧 Fix authentication or configuration before production use")
            
#         if self.snapshot_id:
#             print(f"📝 Test snapshot ID: {self.snapshot_id}")
        
#         # Recommendations based on results
#         print("\n💡 RECOMMENDATIONS FOR YOUR MAIN SCRIPT:")
#         if passed >= total - 1:
#             print("✅ Use these settings in x_prepare_usernames.py:")
#             print("   • batch_size = 1-2 profiles at a time")
#             print("   • dynamic_timeout = 300-600 seconds") 
#             print("   • check_intervals = [15, 20, 30] seconds")
#             print("   • Conservative polling for reliable results")
        
#         return passed >= total - 1  # Allow one warning/timeout

# def main():
#     """Main test execution"""
#     print("🚀 Starting Bright Data API comprehensive test...")
#     print("This will validate your external links collection workflow")
    
#     tester = BrightDataAPITester()
#     success = tester.run_all_tests()
    
#     if success:
#         print("\n🎉 Test completed successfully! Your API is ready.")
#         sys.exit(0)
#     else:
#         print("\n❌ Test failed. Please fix issues before using main script.")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# import requests

# url = "https://api.brightdata.com/datasets/v3/progress/s_me5nysqd5q0rl0cgq"
# headers = {
# 	"Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d",
# }

# response = requests.get(url, headers=headers)
# print(response.json())
# from dotenv import load_dotenv
# load_dotenv()  # Загрузить .env файл

# import requests
# import json
# import os



# API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN")
# snapshot_id = "s_me5nysqd5q0rl0cgq"  # Замените на ID из ответа

# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# # Проверка статуса
# response = requests.get(
#     f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
#     headers=headers
# )

# print(f"Статус: {response.status_code}")
# if response.status_code == 200:
#     data = response.json()
#     print(f"Status: {data.get('status')}")
#     print(f"Records: {data.get('records', 0)}")
#     print(f"Errors: {data.get('errors', 0)}")
# import requests

# url = "https://api.brightdata.com/datasets/v3/trigger"
# headers = {
# 	"Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d",
# 	"Content-Type": "application/json",
# }
# params = {
# 	"dataset_id": "gd_lwxmeb2u1cniijd7t4",
# 	"include_errors": "true",
# }
# data = [
# 	{"url":"https://x.com/elonmusk","max_number_of_posts":2},

# ]

# response = requests.post(url, headers=headers, params=params, json=data)
# print(response.json())


# import requests

# # Your snapshot ID from the previous request
# snapshot_id = "s_me6u96nr16ue6iemb1"

# url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
# # headers = {
# #     "Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d"
# # }
# headers = {
# 	"Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d",
# 	"Content-Type": "application/json",
# }
# # Optional parameters for download format
# # params = {
# #     "format": "json",      # Options: json, ndjson, jsonl, csv
# #     "compress": False,     # Set to True for gzip compression
# # }
# params = {
# 	"dataset_id": "gd_lwxmeb2u1cniijd7t4",
# 	"include_errors": "true",
# 	"type": "discover_new",
# 	"discover_by": "user_name",
# }
import requests
import csv

# Read usernames from your CSV file, skipping header
usernames = []
csv_file_path = "/Users/margotiamanova/Desktop/PROJECTS/x_account_validation/debug_results.csv"

with open(csv_file_path, "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        if row and row[0].strip():  # Check if row exists and username is not empty
            usernames.append({"user_name": row[0].strip()})

print(f"Found {len(usernames)} usernames to process")

# Bright Data API configuration
url = "https://api.brightdata.com/datasets/v3/trigger"
headers = {
    "Authorization": "Bearer 357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d",
    "Content-Type": "application/json",
}
params = {
    "dataset_id": "gd_lwxmeb2u1cniijd7t4",
    "include_errors": "true",
    "type": "discover_new",
    "discover_by": "user_name",
}

# Trigger the scraping job
response = requests.post(url, headers=headers, params=params, json=usernames)

if response.status_code == 200:
    result = response.json()
    snapshot_id = result.get("snapshot_id")
    if snapshot_id:
        print(f"✅ New collection triggered! Snapshot ID: {snapshot_id}")
        print("Now you need to wait for completion and download the results.")
    else:
        print("❌ No snapshot ID received in response")
        print(f"Response: {result}")
else:
    print(f"❌ Error triggering collection: {response.status_code}")
    print(f"Response: {response.text}")



