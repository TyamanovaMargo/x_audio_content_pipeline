import os
import pandas as pd
import argparse
import sys
from config import Config
from step1_validate_accounts import AccountValidator
from step2_bright_data_trigger import BrightDataTrigger
from step3_bright_data_download import BrightDataDownloader
from step4_audio_filter import AudioContentFilter
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("🚀 X/TWITTER AUDIO CONTENT PIPELINE")
    print("=" * 50)

    # Stage 1 with logging
    print("\n✅ STAGE 1: Account Validation with Persistent Logging")
    print("-" * 50)
    
    log_file = os.path.join(cfg.OUTPUT_DIR, "processed_accounts.json")
    validator = AccountValidator(
        max_concurrent=cfg.MAX_CONCURRENT_VALIDATIONS,
        delay_min=cfg.VALIDATION_DELAY_MIN, 
        delay_max=cfg.VALIDATION_DELAY_MAX,
        log_file=log_file
    )
    
    existing_accounts_file = os.path.join(cfg.OUTPUT_DIR, "1_existing_accounts.csv")
    valid_accounts = validator.validate_accounts_from_file(
        input_file, existing_accounts_file, force_recheck=force_recheck
    )

    if not valid_accounts:
        print("❌ No valid accounts found.")
        return

    # Stage 2 - Bright Data Snapshot Management with Duplicate Prevention
    print("\n🚀 STAGE 2: Bright Data Snapshot Management")
    print("-" * 50)

    trigger = BrightDataTrigger(cfg.BRIGHT_DATA_API_TOKEN, cfg.BRIGHT_DATA_DATASET_ID)
    usernames = [acc['username'] for acc in valid_accounts]

    # Check for existing snapshot before creating new one
    sm = SnapshotManager(cfg.OUTPUT_DIR)
    existing_snapshot = sm.get_reusable_snapshot(usernames)

    if existing_snapshot:
        print(f"🔄 Using existing snapshot: {existing_snapshot}")
        snapshot_id = existing_snapshot
    else:
        # Create new snapshot only if no suitable one exists
        print(f"🆕 Creating new snapshot for {len(usernames)} usernames")
        snapshot_id = trigger.create_snapshot_from_usernames(usernames)
        if not snapshot_id:
            print("❌ Failed to create snapshot")
            return
        sm.register_snapshot(snapshot_id, valid_accounts)

    print(f"✅ Stage 2 completed: Using snapshot {snapshot_id}")

    # Stage 3 - Data Download & Processing
    print("\n⬇️ STAGE 3: Data Download & External Link Extraction")
    print("-" * 50)
    
    downloader = BrightDataDownloader(cfg.BRIGHT_DATA_API_TOKEN)
    profiles = downloader.wait_and_download_snapshot(snapshot_id, cfg.MAX_SNAPSHOT_WAIT)
    
    if profiles:
        # Update snapshot status to completed
        sm.update_snapshot_status(snapshot_id, "completed", profiles)
        
        # Save profile data
        profiles_file = os.path.join(cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
        pd.DataFrame(profiles).to_csv(profiles_file, index=False)
        print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
        
        # Extract external links
        links = downloader.extract_external_links(profiles)
        
        if links:
            links_file = os.path.join(cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
            pd.DataFrame(links).to_csv(links_file, index=False)
            print(f"🔗 Saved {len(links)} external links to: {links_file}")

            # Stage 4 - Audio Content Filtering
            print("\n🎵 STAGE 4: Audio Content Filtering")
            print("-" * 50)
            
            audio_filter = AudioContentFilter()
            audio_links = audio_filter.filter_audio_links(links)
            
            if audio_links:
                audio_file = os.path.join(cfg.OUTPUT_DIR, f"4_snapshot_{snapshot_id}_audio_links.csv")
                pd.DataFrame(audio_links).to_csv(audio_file, index=False)
                print(f"🎵 Found {len(audio_links)} audio platform links!")
                print(f"📁 Saved to: {audio_file}")
                
                # Show summary of platforms found
                platform_counts = {}
                for link in audio_links:
                    platform = link.get('platform_type', 'unknown')
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                
                print("\n📊 Audio platforms breakdown:")
                for platform, count in platform_counts.items():
                    print(f"   {platform}: {count}")
                    
            else:
                print("🔍 No audio platform links found in the external links")
        else:
            print("🔗 No external links found in profiles")
            
        # Final summary
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📊 Total accounts processed: {len(valid_accounts)}")
        print(f"📥 Profiles downloaded: {len(profiles)}")
        print(f"🔗 External links found: {len(links) if links else 0}")
        print(f"🎵 Audio links found: {len(audio_links) if links and audio_links else 0}")
        print(f"🆔 Snapshot ID: {snapshot_id}")
        print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")
        
    else:
        print("❌ Failed to download snapshot data")
        sm.update_snapshot_status(snapshot_id, "failed")

def show_help():
    """Show detailed usage help"""
    help_text = """
🚀 X/TWITTER AUDIO CONTENT PIPELINE

This pipeline validates X/Twitter accounts, collects profile data through Bright Data API,
extracts external links, and filters for audio/video platforms.

USAGE:
  python main_pipeline.py --input <file> [options]

REQUIRED ARGUMENTS:
  --input FILE          Input CSV/TXT file with usernames

OPTIONAL ARGUMENTS:
  --force-recheck       Force recheck all accounts (ignore cache)
  --show-log           Show processed accounts summary
  --show-snapshots     Show all Bright Data snapshots
  --analyze-duplicates  Analyze duplicate snapshots
  --clear-log          Clear processed accounts cache
  --help-detailed      Show this detailed help

EXAMPLES:
  # Normal run (uses cache)
  python main_pipeline.py --input usernames.csv

  # Force recheck all accounts
  python main_pipeline.py --input usernames.csv --force-recheck

  # Show statistics
  python main_pipeline.py --show-log
  python main_pipeline.py --show-snapshots

PIPELINE STAGES:
  1. Account Validation - Check if Twitter accounts exist
  2. Snapshot Creation - Create Bright Data collection job
  3. Data Download - Download profile data and extract links
  4. Audio Filtering - Filter for audio/video platforms

OUTPUT FILES:
  - 1_existing_accounts.csv - Validated existing accounts
  - 2_snapshot_<id>_results.csv - Full profile data
  - 3_snapshot_<id>_external_links.csv - All external links
  - 4_snapshot_<id>_audio_links.csv - Audio platform links only
  - processed_accounts.json - Validation cache
  - snapshots/ - Snapshot metadata and tracking

SUPPORTED AUDIO PLATFORMS:
  - High: Spotify, SoundCloud, Apple Music, Podcasts, Anchor
  - Medium: YouTube, Twitch, TikTok
  - Low: Instagram, Discord, Kick
"""
    print(help_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X/Twitter Audio Content Pipeline with Logging and Duplicate Prevention",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=False, help="Input CSV/TXT file with usernames")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")
    parser.add_argument("--show-log", action="store_true", help="Show log summary and exit")
    parser.add_argument("--show-snapshots", action="store_true", help="Show snapshot summary and exit")
    parser.add_argument("--analyze-duplicates", action="store_true", help="Analyze duplicate snapshots")
    parser.add_argument("--clear-log", action="store_true", help="Clear processed accounts log")
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed usage help")
    
    args = parser.parse_args()
    
    # Handle help and info commands
    if args.help_detailed:
        show_help()
        sys.exit(0)
    
    if args.show_log:
        validator = AccountValidator()
        validator.show_log_summary()
        sys.exit(0)
    
    if args.show_snapshots:
        sm = SnapshotManager()
        sm.print_snapshot_summary()
        sys.exit(0)
        
    if args.analyze_duplicates:
        sm = SnapshotManager()
        sm.print_duplicate_analysis()
        sys.exit(0)
    
    if args.clear_log:
        validator = AccountValidator()
        validator.clear_log()
        sys.exit(0)
    
    # Validate required arguments
    if not args.input:
        print("❌ Error: --input argument is required for pipeline execution")
        print("💡 Use --help-detailed for usage examples")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Run main pipeline
    try:
        main(args.input, args.force_recheck)
    except KeyboardInterrupt:
        print("\n\n⏹️ Pipeline interrupted by user")
        print("💾 All progress has been saved and can be resumed")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("💡 Check your configuration and try again")
        sys.exit(1)
