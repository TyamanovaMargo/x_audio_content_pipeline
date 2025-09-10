#!/usr/bin/env python3
"""
Step 3.5: YouTube & Twitch Channel Discovery
Runs the youtube-twitch-x-scraper after step 3 to find actual channels
"""

import os
import sys
import subprocess
import pandas as pd

class Step3_5_YouTubeTwitchRunner:
    def __init__(self, output_dir="output/"):
        self.output_dir = output_dir
        self.scraper_dir = os.path.join(os.path.dirname(__file__), "youtube-twitch-x-scraper")
        
    def run_scraper_for_snapshot(self, snapshot_id):
        """Run the YouTube-Twitch scraper for a specific snapshot"""
        
        # 1. Find the external links CSV from step 3
        external_links_file = os.path.join(self.output_dir, f"3_snapshot_{snapshot_id}_external_links.csv")
        
        if not os.path.exists(external_links_file):
            print(f"❌ External links file not found: {external_links_file}")
            return None
            
        print(f"📂 Found external links file: {external_links_file}")
        
        # 2. Update the scraper's config with the actual file path
        config_file = os.path.join(self.scraper_dir, "config.py")
        scraper_file = os.path.join(self.scraper_dir, "youtube_twitch_scraper.py")
        
        # 3. Replace the dynamic path in config
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Replace the dynamic placeholder with actual snapshot ID
        config_content = config_content.replace(
            "3_snapshot_DYNAMIC_external_links.csv", 
            f"3_snapshot_{snapshot_id}_external_links.csv"
        )
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"📝 Updated scraper config for snapshot {snapshot_id}")
        
        # 3.5. Verify config update was successful
        with open(config_file, 'r') as f:
            updated_content = f.read()
        
        if f"3_snapshot_{snapshot_id}_external_links.csv" in updated_content:
            print(f"✅ Config successfully updated with snapshot {snapshot_id}")
        else:
            print(f"⚠️ Warning: Config update may have failed")
        
        # 4. Run the scraper
        print(f"🚀 Running YouTube-Twitch scraper...")
        print(f"📂 Input file: {external_links_file}")
        print(f"📂 Expected output: {os.path.join(self.output_dir, 'youtube_twitch_results_enhanced.csv')}")
        
        try:
            # Change to scraper directory and run
            original_cwd = os.getcwd()
            os.chdir(self.scraper_dir)
            
            # Run the scraper with automatic input (1 worker) - NO CAPTURE, NO TIMEOUT
            result = subprocess.run([
                sys.executable, "youtube_twitch_scraper.py"
            ], input="1\n", text=True)  # Let it run and show all logs
            
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                output_file = os.path.join(self.output_dir, "youtube_twitch_results_enhanced.csv")
                if os.path.exists(output_file):
                    print("✅ Scraper completed successfully!")
                    print(f"📊 Results saved to: {output_file}")
                    
                    # Show basic stats about the results
                    try:
                        df = pd.read_csv(output_file)
                        print(f"📈 Enhanced {len(df)} records with YouTube/Twitch data")
                    except Exception as e:
                        print(f"⚠️ Could not read results file: {e}")
                    
                    return output_file
                else:
                    print("❌ Scraper completed but no output file found")
                    print(f"📂 Expected output file: {output_file}")
                    if result.stdout:
                        print(f"📝 Stdout: {result.stdout}")
                    return None
            else:
                print(f"❌ Scraper failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"📝 Error output: {result.stderr}")
                if result.stdout:
                    print(f"📝 Standard output: {result.stdout}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to run scraper: {e}")
            return None


def main():
    """Standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 3.5: Run YouTube-Twitch scraper")
    parser.add_argument("--snapshot-id", required=True, help="Snapshot ID to process")
    parser.add_argument("--output-dir", default="output/", help="Output directory")
    
    args = parser.parse_args()
    
    runner = Step3_5_YouTubeTwitchRunner(args.output_dir)
    result = runner.run_scraper_for_snapshot(args.snapshot_id)
    
    if result:
        print(f"✅ Step 3.5 completed: {result}")
    else:
        print("❌ Step 3.5 failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
