import os
import pandas as pd
import argparse
import sys
from config import Config
from step1_validate_accounts import AccountValidator
from step2_bright_data_trigger import BrightDataTrigger
from step3_bright_data_download import BrightDataDownloader
from step4_audio_filter import AudioContentFilter
from step5_voice_verification import VoiceContentVerifier
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("🎙️ X/TWITTER VOICE CONTENT PIPELINE")
    print("=" * 60)

    # Stage 1: Account Validation with Persistent Logging
    print("\n✅ STAGE 1: Account Validation with Persistent Logging")
    print("-" * 60)
    
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

    # Stage 2: Bright Data Snapshot Management with Duplicate Prevention
    print("\n🚀 STAGE 2: Bright Data Snapshot Management")
    print("-" * 60)

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

    # Stage 3: Data Download & External Link Extraction
    print("\n⬇️ STAGE 3: Data Download & External Link Extraction")
    print("-" * 60)
    
    downloader = BrightDataDownloader(cfg.BRIGHT_DATA_API_TOKEN)
    profiles = downloader.wait_and_download_snapshot(snapshot_id, cfg.MAX_SNAPSHOT_WAIT)
    
    if not profiles:
        print("❌ Failed to download snapshot data")
        sm.update_snapshot_status(snapshot_id, "failed")
        return

    # Update snapshot status to completed
    sm.update_snapshot_status(snapshot_id, "completed", profiles)
    
    # Save profile data
    profiles_file = os.path.join(cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
    pd.DataFrame(profiles).to_csv(profiles_file, index=False)
    print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
    
    # Extract external links
    links = downloader.extract_external_links(profiles)
    
    if not links:
        print("🔗 No external links found in profiles")
        print("⚠️ Pipeline completed but no links to process further")
        return

    links_file = os.path.join(cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
    pd.DataFrame(links).to_csv(links_file, index=False)
    print(f"🔗 Saved {len(links)} external links to: {links_file}")

    # Stage 4: Audio Content Filtering
    print("\n🎵 STAGE 4: Audio Platform Filtering")
    print("-" * 60)
    
    audio_filter = AudioContentFilter()
    audio_links = audio_filter.filter_audio_links(links)
    
    if not audio_links:
        print("🔍 No audio platform links found")
        print("⚠️ Pipeline completed but no audio content detected")
        return

    audio_file = os.path.join(cfg.OUTPUT_DIR, f"4_snapshot_{snapshot_id}_audio_links.csv")
    pd.DataFrame(audio_links).to_csv(audio_file, index=False)
    print(f"🎵 Found {len(audio_links)} audio platform links!")
    print(f"📁 Saved to: {audio_file}")
    
    # Show platform breakdown
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
    
    print("\n📊 Audio platforms breakdown:")
    for platform, count in platform_counts.items():
        print(f"   {platform}: {count}")

    # Stage 5: Voice Content Verification
    print("\n🎙️ STAGE 5: Voice Content Verification")
    print("-" * 60)
    
    voice_verifier = VoiceContentVerifier(timeout=15)
    verified_links = voice_verifier.verify_voice_content(audio_links)
    
    # Save all verification results
    verified_file = os.path.join(cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_verified_voice.csv")
    pd.DataFrame(verified_links).to_csv(verified_file, index=False)
    print(f"📁 Saved verification results to: {verified_file}")
    
    # Save only confirmed voice content
    confirmed_voice = [link for link in verified_links if link['has_voice']]
    
    if confirmed_voice:
        confirmed_file = os.path.join(cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_confirmed_voice.csv")
        pd.DataFrame(confirmed_voice).to_csv(confirmed_file, index=False)
        print(f"🎙️ Found {len(confirmed_voice)} confirmed voice content links!")
        print(f"📁 Saved to: {confirmed_file}")
        
        # Detailed voice content analysis
        voice_types = {}
        platforms = {}
        confidence_levels = {}
        
        for link in confirmed_voice:
            voice_type = link.get('voice_type', 'unknown')
            platform = link.get('platform_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            
            voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
            platforms[platform] = platforms.get(platform, 0) + 1
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
        
        print("\n📊 Voice Content Analysis:")
        print("   Voice Types:")
        for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      {voice_type}: {count}")
        print("   Platforms:")
        for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
            print(f"      {platform}: {count}")
        print("   Confidence Levels:")
        for confidence, count in confidence_levels.items():
            print(f"      {confidence}: {count}")
        
        # Show sample confirmed voice links
        print("\n🎙️ Sample Confirmed Voice Content:")
        for i, link in enumerate(confirmed_voice[:5], 1):
            username = link.get('username', 'unknown')
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            url = link.get('url', '')[:60] + '...' if len(link.get('url', '')) > 60 else link.get('url', '')
            print(f"   {i}. @{username} | {voice_type} | {confidence} confidence")
            print(f"      {url}")
            
    else:
        print("❌ No voice content confirmed after verification")
        print("💡 This means the audio links found were likely music, sounds, or non-voice content")

    # Final comprehensive summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total accounts processed: {len(valid_accounts)}")
    print(f"📥 Profiles downloaded: {len(profiles)}")
    print(f"🔗 External links found: {len(links)}")
    print(f"🎵 Audio platform links: {len(audio_links)}")
    print(f"🎙️ Voice content confirmed: {len(confirmed_voice) if confirmed_voice else 0}")
    print(f"📈 Voice confirmation rate: {(len(confirmed_voice) / len(audio_links) * 100):.1f}%" if audio_links else "0%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")
    
    # Final output files summary
    print(f"\n📄 Output Files Generated:")
    print(f"   1. {existing_accounts_file} - Validated accounts")
    print(f"   2. {profiles_file} - Profile data")
    print(f"   3. {links_file} - External links")
    print(f"   4. {audio_file} - Audio platform links")
    print(f"   5. {verified_file} - Voice verification results")
    if confirmed_voice:
        print(f"   6. {confirmed_file} - ⭐ CONFIRMED VOICE CONTENT ⭐")

def run_stage5_only(audio_links_file, output_dir="output"):
    """Run only Stage 5 voice verification on existing audio links"""
    
    print("🎙️ STAGE 5: Voice Content Verification (Standalone)")
    print("=" * 60)
    
    # Load existing audio links
    if not os.path.exists(audio_links_file):
        print(f"❌ Audio links file not found: {audio_links_file}")
        return
    
    try:
        df = pd.read_csv(audio_links_file)
        audio_links = df.to_dict('records')
        print(f"📥 Loaded {len(audio_links)} audio links from: {audio_links_file}")
    except Exception as e:
        print(f"❌ Error loading audio links: {e}")
        return
    
    if not audio_links:
        print("❌ No audio links found in file")
        return
    
    # Run voice verification
    voice_verifier = VoiceContentVerifier(timeout=15)
    verified_links = voice_verifier.verify_voice_content(audio_links)
    
    # Generate output filename from input filename
    base_name = os.path.splitext(os.path.basename(audio_links_file))[0]
    
    # Save all verification results
    verified_file = os.path.join(output_dir, f"{base_name}_verified_voice.csv")
    pd.DataFrame(verified_links).to_csv(verified_file, index=False)
    print(f"📁 Saved verification results to: {verified_file}")
    
    # Save only confirmed voice content
    confirmed_voice = [link for link in verified_links if link['has_voice']]
    
    if confirmed_voice:
        confirmed_file = os.path.join(output_dir, f"{base_name}_confirmed_voice.csv")
        pd.DataFrame(confirmed_voice).to_csv(confirmed_file, index=False)
        print(f"🎙️ Found {len(confirmed_voice)} confirmed voice content links!")
        print(f"📁 Saved to: {confirmed_file}")
        
        # Analysis
        voice_types = {}
        confidence_levels = {}
        
        for link in confirmed_voice:
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            
            voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
        
        print("\n📊 Voice Content Analysis:")
        print("   Voice Types:")
        for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      {voice_type}: {count}")
        print("   Confidence Levels:")
        for confidence, count in confidence_levels.items():
            print(f"      {confidence}: {count}")
            
    else:
        print("❌ No voice content confirmed")
    
    print(f"\n✅ Stage 5 completed!")
    print(f"📊 Total links processed: {len(audio_links)}")
    print(f"🎙️ Voice content found: {len(confirmed_voice)}")

def show_help():
    """Show detailed usage help"""
    help_text = """
🎙️ X/TWITTER VOICE CONTENT PIPELINE

This pipeline validates X/Twitter accounts, collects profile data through Bright Data API,
extracts external links, filters for audio platforms, and verifies voice/speech content.

USAGE:
  python main_pipeline.py --input <file> [options]

REQUIRED ARGUMENTS:
  --input FILE          Input CSV/TXT file with usernames (one per line)

OPTIONAL ARGUMENTS:
  --force-recheck       Force recheck all accounts (ignore validation cache)
  --stage5-only FILE    Run only Stage 5 on existing audio links CSV file
  --show-log           Show processed accounts summary
  --show-snapshots     Show all Bright Data snapshots
  --analyze-duplicates  Analyze duplicate snapshots
  --clear-log          Clear processed accounts cache
  --help-detailed      Show this detailed help

EXAMPLES:
  # Normal run (uses cache and duplicate prevention)
  python main_pipeline.py --input usernames.csv

  # Force recheck all accounts from scratch
  python main_pipeline.py --input usernames.csv --force-recheck

  # Run only voice verification on existing audio links
  python main_pipeline.py --stage5-only output/4_snapshot_123_audio_links.csv

  # Show statistics and logs
  python main_pipeline.py --show-log
  python main_pipeline.py --show-snapshots
  python main_pipeline.py --analyze-duplicates

PIPELINE STAGES:
  1. Account Validation - Check if Twitter accounts exist (with persistent cache)
  2. Snapshot Management - Create/reuse Bright Data collection job (duplicate prevention)
  3. Data Download - Download profile data and extract external links
  4. Audio Filtering - Filter for audio/video platforms (YouTube, Spotify, etc.)
  5. Voice Verification - Verify links contain voice/speech content (NOT music)

OUTPUT FILES:
  - 1_existing_accounts.csv - Validated existing accounts
  - 2_snapshot_<id>_results.csv - Full profile data from Bright Data
  - 3_snapshot_<id>_external_links.csv - All external links found
  - 4_snapshot_<id>_audio_links.csv - Audio platform links only
  - 5_snapshot_<id>_verified_voice.csv - All links with voice verification
  - 5_snapshot_<id>_confirmed_voice.csv - 🎙️ CONFIRMED VOICE CONTENT ONLY
  - processed_accounts.json - Account validation cache
  - snapshots/ - Snapshot metadata and tracking

SUPPORTED PLATFORMS:
  Audio Platforms:
    - High: Spotify, SoundCloud, Apple Music, Apple Podcasts, Anchor
    - Medium: YouTube, Twitch, TikTok  
    - Low: Instagram, Discord, Kick

  Voice Content Types Detected:
    - Podcasts (episodes, shows)
    - Interviews and conversations
    - Educational content (lectures, presentations)
    - Commentary and analysis
    - Radio shows and broadcasts
    - Narrative content (audiobooks, storytelling)
    - Live talks (Twitch Just Chatting, etc.)

FEATURES:
  ✅ Persistent logging (remembers processed accounts)
  ✅ Duplicate prevention (reuses existing snapshots)
  ✅ Voice-specific verification (excludes music)
  ✅ Comprehensive error handling
  ✅ Detailed progress reporting
  ✅ Platform-specific detection strategies
"""
    print(help_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X/Twitter Voice Content Pipeline with Advanced Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=False, help="Input CSV/TXT file with usernames")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")
    parser.add_argument("--stage5-only", help="Run only Stage 5 on existing audio links file")
    parser.add_argument("--show-log", action="store_true", help="Show account validation log summary")
    parser.add_argument("--show-snapshots", action="store_true", help="Show Bright Data snapshots summary")
    parser.add_argument("--analyze-duplicates", action="store_true", help="Analyze duplicate snapshots")
    parser.add_argument("--clear-log", action="store_true", help="Clear processed accounts log")
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed usage help")
    
    args = parser.parse_args()
    
    # Handle help and info commands
    if args.help_detailed:
        show_help()
        sys.exit(0)
    
    # Handle Stage 5 only execution
    if args.stage5_only:
        if not os.path.exists(args.stage5_only):
            print(f"❌ Audio links file not found: {args.stage5_only}")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        run_stage5_only(args.stage5_only, output_dir)
        sys.exit(0)
    
    if args.show_log:
        try:
            validator = AccountValidator()
            validator.show_log_summary()
        except Exception as e:
            print(f"❌ Error showing log: {e}")
        sys.exit(0)
    
    if args.show_snapshots:
        try:
            sm = SnapshotManager()
            sm.print_snapshot_summary()
        except Exception as e:
            print(f"❌ Error showing snapshots: {e}")
        sys.exit(0)
        
    if args.analyze_duplicates:
        try:
            sm = SnapshotManager()
            sm.print_duplicate_analysis()
        except Exception as e:
            print(f"❌ Error analyzing duplicates: {e}")
        sys.exit(0)
    
    if args.clear_log:
        try:
            validator = AccountValidator()
            validator.clear_log()
            print("✅ Account validation log cleared")
        except Exception as e:
            print(f"❌ Error clearing log: {e}")
        sys.exit(0)
    
    # Validate required arguments for pipeline execution
    if not args.input:
        print("❌ Error: --input argument is required for pipeline execution")
        print("💡 Use --help-detailed for usage examples")
        print("📝 Or use --show-log, --show-snapshots for information")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        print("💡 Make sure the file path is correct")
        sys.exit(1)
    
    # Run main pipeline with comprehensive error handling
    try:
        print(f"🚀 Starting pipeline with input: {args.input}")
        print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
        main(args.input, args.force_recheck)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Pipeline interrupted by user (Ctrl+C)")
        print("💾 All progress has been saved and can be resumed")
        print("🔄 Run the same command again to continue from where you left off")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("💡 Check your file paths and configuration")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("💡 Check your configuration, API tokens, and network connection")
        print("📋 Use --show-log to see processed accounts")
        print("🔍 Use --show-snapshots to see snapshot history")
        sys.exit(1)

