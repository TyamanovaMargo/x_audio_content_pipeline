import requests
import time
import json
from typing import List, Dict

class BrightDataDownloader:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "User-Agent": "Python-Pipeline/1.0"
        }

    def wait_and_download_snapshot(self, snapshot_id: str, max_wait=600) -> List[Dict]:
        start = time.time()
        unknown_count = 0
        
        print(f"⏳ Waiting for snapshot {snapshot_id} to complete...")
        
        while (time.time() - start) < max_wait:
            status = self._check_status(snapshot_id)
            elapsed = int(time.time() - start)
            
            print(f"🔄 [{elapsed}s] Snapshot {snapshot_id} status: {status}")
            
            if status == "completed":
                print("✅ Snapshot ready! Downloading...")
                return self._download(snapshot_id)
            elif status == "unknown" or status == "error":
                unknown_count += 1
                print(f"⚠️ Status issue (attempt {unknown_count}/5)")
                
                # After 5 unknown statuses, try direct download
                if unknown_count >= 5:
                    print("🔄 Attempting direct download despite status issues...")
                    try:
                        data = self._download(snapshot_id)
                        if data and len(data) > 0:
                            print("✅ Direct download successful!")
                            return data
                        else:
                            print("❌ Direct download returned empty data")
                    except Exception as e:
                        print(f"❌ Direct download failed: {e}")
                    
                    unknown_count = 0  # Reset counter
                    
            elif status in ["failed", "expired", "cancelled"]:
                print(f"❌ Snapshot {status}")
                return []
            
            time.sleep(15)
        
        print(f"⏰ Timeout after {max_wait}s, attempting final download...")
        # Final attempt at download even if timeout
        try:
            return self._download(snapshot_id)
        except:
            return []

    def _check_status(self, snapshot_id: str) -> str:
        """Check snapshot status with enhanced debugging"""
        try:
            url = f"{self.base_url}/snapshot/{snapshot_id}"
            
            # Add debugging
            print(f"🔍 Checking URL: {url}")
            print(f"🔍 Headers: {self.headers}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            
            print(f"🔍 Response Status: {response.status_code}")
            print(f"🔍 Response Headers: {dict(response.headers)}")
            print(f"🔍 Raw Response (first 300 chars): {response.text[:300]}")
            
            if not response.ok:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return "error"
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"🔍 Parsed JSON: {data}")
                
                if isinstance(data, dict):
                    status = data.get("status", "unknown")
                    print(f"📊 Extracted status: {status}")
                    return status
                elif isinstance(data, list) and len(data) > 0:
                    # Sometimes returns array with status in first element
                    first_item = data[0]
                    if isinstance(first_item, dict):
                        status = first_item.get("status", "unknown")
                        print(f"📊 Extracted status from array: {status}")
                        return status
                
                print(f"⚠️ Unexpected JSON structure: {type(data)}")
                return "unknown"
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"❌ Response text: {response.text}")
                return "unknown"
                
        except requests.RequestException as e:
            print(f"❌ Request error: {e}")
            return "error"
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return "error"

    def _download(self, snapshot_id: str) -> List[Dict]:
        """Download snapshot data with enhanced error handling"""
        try:
            url = f"{self.base_url}/snapshot/{snapshot_id}"
            params = {"format": "json", "compress": "false"}
            
            print(f"📥 Downloading from: {url}")
            print(f"📥 Parameters: {params}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=120)
            
            print(f"📥 Download Response Status: {response.status_code}")
            print(f"📥 Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            print(f"📥 Content-Length: {response.headers.get('Content-Length', 'unknown')}")
            
            if not response.ok:
                print(f"❌ Download failed - HTTP {response.status_code}: {response.text[:200]}")
                return []
            
            # Parse response data
            data = self._safe_json_parse(response)
            
            if isinstance(data, list):
                print(f"📊 Downloaded {len(data)} profiles")
                return data
            elif isinstance(data, dict):
                # Sometimes wrapped in a data field
                if "data" in data and isinstance(data["data"], list):
                    print(f"📊 Downloaded {len(data['data'])} profiles from data field")
                    return data["data"]
                else:
                    print(f"📊 Downloaded 1 profile (single object)")
                    return [data]
            else:
                print(f"❌ Unexpected data format: {type(data)}")
                return []
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return []

    def _safe_json_parse(self, response):
        """Enhanced JSON parsing with NDJSON support"""
        try:
            # First try standard JSON parsing
            return response.json()
        except requests.exceptions.JSONDecodeError:
            # Handle NDJSON (newline-delimited JSON)
            text = response.text.strip()
            
            if not text:
                return []
            
            # Try parsing as NDJSON (multiple JSON objects separated by newlines)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            if len(lines) == 1:
                # Single line, try parsing as JSON
                try:
                    return json.loads(lines[0])
                except json.JSONDecodeError:
                    print(f"❌ Failed to parse single line JSON: {lines[0][:100]}...")
                    return []
            else:
                # Multiple lines, parse each as JSON
                parsed_objects = []
                for i, line in enumerate(lines):
                    try:
                        parsed_objects.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse line {i+1}: {line[:100]}...")
                        continue
                
                if parsed_objects:
                    return parsed_objects
                else:
                    print(f"❌ No valid JSON found in {len(lines)} lines")
                    return []

    def extract_external_links(self, profiles: List[Dict]) -> List[Dict]:
        """Extract external links from profiles with None-safe handling"""
        if not profiles:
            return []
            
        external_links = []
        
        for profile in profiles:
            if not profile:  # Skip if profile is None
                continue
                
            username = profile.get('username') or profile.get('screen_name', '')
            
            # Search for external links in various fields
            link_fields = ['external_link', 'url', 'website', 'profile_external_link', 'bio_link']
            
            found_link = None
            for field in link_fields:
                link = profile.get(field)
                
                # Safe None handling
                if link is None:
                    link = ''
                else:
                    link = str(link).strip()  # Convert to string and strip
                
                if link and link.startswith('http'):
                    found_link = link
                    break
            
            if found_link:
                # Safe handling for description field too
                description = profile.get('description')
                bio = description[:100] if description else ''
                
                external_links.append({
                    'username': username,
                    'profile_name': profile.get('profile_name', ''),
                    'url': found_link,
                    'followers': profile.get('followers', 0),
                    'bio': bio
                })
        
        print(f"🔗 Extracted {len(external_links)} external links from {len(profiles)} profiles")
        return external_links
