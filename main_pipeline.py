import os
import pandas as pd
import argparse
import sys
from config import Config
from step1_validate_accounts import AccountValidator
from step2_bright_data_trigger import BrightDataTrigger
from step3_bright_data_download import BrightDataDownloader
from step4_audio_filter import AudioContentFilter
from step4_5_audio_detector import AudioContentDetector
from step5_voice_verification import VoiceContentVerifier
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    """Main pipeline execution"""
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
        print(f"  {platform}: {count}")

    # Stage 4.5: Audio Content Detection  
    print("\n🎵 STAGE 4.5: Audio Content Detection")
    print("-" * 60)

    audio_detector = AudioContentDetector(timeout=10)
    audio_detected_links = audio_detector.detect_audio_content(audio_links)

    if not audio_detected_links:
        print("🔍 No audio content detected in links")
        print("⚠️ Pipeline completed but no actual audio found")
        return

    # Save audio detection results
    audio_detected_file = os.path.join(cfg.OUTPUT_DIR, f"4_5_snapshot_{snapshot_id}_audio_detected.csv")
    pd.DataFrame(audio_detected_links).to_csv(audio_detected_file, index=False)

    print(f"🎵 Found {len(audio_detected_links)} links with actual audio content!")
    print(f"📁 Saved to: {audio_detected_file}")

    # Show audio type breakdown
    audio_types = {}
    confidence_levels = {}
    for link in audio_detected_links:
        audio_type = link.get('audio_type', 'unknown')
        confidence = link.get('audio_confidence', 'unknown')
        audio_types[audio_type] = audio_types.get(audio_type, 0) + 1
        confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1

    print("\n📊 Audio Content Breakdown:")
    print("  Audio Types:")
    for audio_type, count in sorted(audio_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {audio_type}: {count}")
    print("  Confidence Levels:")
    for confidence, count in confidence_levels.items():
        print(f"    {confidence}: {count}")

    # Stage 5: Voice Content Verification (now only on audio-confirmed links)
    print("\n🎙️ STAGE 5: Voice Content Verification (Audio-Confirmed Links Only)")
    print("-" * 60)

    voice_verifier = VoiceContentVerifier(timeout=15)
    verified_links = voice_verifier.verify_voice_content(audio_detected_links)

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
        print("  Voice Types:")
        for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {voice_type}: {count}")
        print("  Platforms:")
        for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
            print(f"    {platform}: {count}")
        print("  Confidence Levels:")
        for confidence, count in confidence_levels.items():
            print(f"    {confidence}: {count}")

        # Show sample confirmed voice links
        print("\n🎙️ Sample Confirmed Voice Content:")
        for i, link in enumerate(confirmed_voice[:5], 1):
            username = link.get('username', 'unknown')
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            url = link.get('url', '')[:60] + '...' if len(link.get('url', '')) > 60 else link.get('url', '')
            print(f"  {i}. @{username} | {voice_type} | {confidence} confidence")
            print(f"     {url}")

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
    print(f"🔊 Audio content confirmed: {len(audio_detected_links)}")
    print(f"🎙️ Voice content confirmed: {len(confirmed_voice) if confirmed_voice else 0}")
    print(f"📈 Voice confirmation rate: {(len(confirmed_voice) / len(audio_links) * 100):.1f}%" if audio_links else "0%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")

    # Final output files summary
    print(f"\n📄 Output Files Generated:")
    print(f"  1. {existing_accounts_file} - Validated accounts")
    print(f"  2. {profiles_file} - Profile data")
    print(f"  3. {links_file} - External links")
    print(f"  4. {audio_file} - Audio platform links")
    print(f"  5. {audio_detected_file} - Audio content detected")
    print(f"  6. {verified_file} - Voice verification results")
    if confirmed_voice:
        print(f"  7. {confirmed_file} - ⭐ CONFIRMED VOICE CONTENT ⭐")

# Individual Stage Runner Functions

def run_stage1_only(input_file, force_recheck=False):
    """Run only Stage 1: Account Validation"""
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("✅ STAGE 1 ONLY: Account Validation")
    print("=" * 50)
    
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
    
    print(f"\n✅ Stage 1 completed!")
    print(f"📊 Valid accounts found: {len(valid_accounts)}")
    print(f"📁 Output file: {existing_accounts_file}")
    print(f"💡 Next: Run Stage 2 with --stage2-only {existing_accounts_file}")

def run_stage2_only(accounts_file):
    """Run only Stage 2: Bright Data Trigger"""
    cfg = Config()
    
    print("🚀 STAGE 2 ONLY: Bright Data Trigger")
    print("=" * 50)
    
    # Load accounts from previous stage
    if not os.path.exists(accounts_file):
        print(f"❌ Accounts file not found: {accounts_file}")
        return None
        
    df = pd.read_csv(accounts_file)
    valid_accounts = df.to_dict('records')
    usernames = [acc['username'] for acc in valid_accounts]
    
    print(f"📥 Loaded {len(usernames)} accounts from: {accounts_file}")
    
    # Check for existing snapshot
    sm = SnapshotManager(cfg.OUTPUT_DIR)
    existing_snapshot = sm.get_reusable_snapshot(usernames)
    
    if existing_snapshot:
        print(f"🔄 Using existing snapshot: {existing_snapshot}")
        snapshot_id = existing_snapshot
    else:
        # Create new snapshot
        trigger = BrightDataTrigger(cfg.BRIGHT_DATA_API_TOKEN, cfg.BRIGHT_DATA_DATASET_ID)
        snapshot_id = trigger.create_snapshot_from_usernames(usernames)
        if snapshot_id:
            sm.register_snapshot(snapshot_id, valid_accounts)
        else:
            print("❌ Failed to create snapshot")
            return None
        
    print(f"✅ Stage 2 completed!")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"💡 Next: Run Stage 3 with --stage3-only {snapshot_id}")
    return snapshot_id

def run_stage3_only(snapshot_id):
    """Run only Stage 3: Data Download"""
    cfg = Config()
    
    print("⬇️ STAGE 3 ONLY: Data Download")
    print("=" * 50)
    
    downloader = BrightDataDownloader(cfg.BRIGHT_DATA_API_TOKEN)
    profiles = downloader.wait_and_download_snapshot(snapshot_id, cfg.MAX_SNAPSHOT_WAIT)
    
    if not profiles:
        print("❌ Failed to download snapshot data")
        return None
        
    # Save profile data
    profiles_file = os.path.join(cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
    pd.DataFrame(profiles).to_csv(profiles_file, index=False)
    print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
    
    # Extract external links
    links = downloader.extract_external_links(profiles)
    if not links:
        print("🔗 No external links found")
        return None
        
    links_file = os.path.join(cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
    pd.DataFrame(links).to_csv(links_file, index=False)
    
    print(f"✅ Stage 3 completed!")
    print(f"🔗 External links found: {len(links)}")
    print(f"📁 Links file: {links_file}")
    print(f"💡 Next: Run Stage 4 with --stage4-only {links_file}")
    return links_file

def run_stage4_only(links_file):
    """Run only Stage 4: Audio Platform Filtering"""
    print("🎵 STAGE 4 ONLY: Audio Platform Filtering")
    print("=" * 50)
    
    if not os.path.exists(links_file):
        print(f"❌ Links file not found: {links_file}")
        return None
        
    df = pd.read_csv(links_file)
    links = df.to_dict('records')
    print(f"📥 Loaded {len(links)} links from: {links_file}")
    
    audio_filter = AudioContentFilter()
    audio_links = audio_filter.filter_audio_links(links)
    
    if not audio_links:
        print("🔍 No audio platform links found")
        return None
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(links_file))[0]
    audio_file = os.path.join("output", f"4_{base_name.replace('3_', '')}_audio_links.csv")
    pd.DataFrame(audio_links).to_csv(audio_file, index=False)
    
    # Show platform breakdown
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1

    print(f"✅ Stage 4 completed!")
    print(f"🎵 Audio platform links: {len(audio_links)}")
    print("📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count}")
    print(f"📁 Audio file: {audio_file}")
    print(f"💡 Next: Run Stage 4.5 with --stage4_5-only {audio_file}")
    return audio_file

def run_stage4_5_only(audio_links_file):
    """Run only Stage 4.5: Audio Content Detection"""
    print("🔊 STAGE 4.5 ONLY: Audio Content Detection")
    print("=" * 50)
    
    if not os.path.exists(audio_links_file):
        print(f"❌ Audio links file not found: {audio_links_file}")
        return None
        
    df = pd.read_csv(audio_links_file)
    audio_links = df.to_dict('records')
    print(f"📥 Loaded {len(audio_links)} audio links from: {audio_links_file}")
    
    audio_detector = AudioContentDetector(timeout=10)
    audio_detected_links = audio_detector.detect_audio_content(audio_links)
    
    if not audio_detected_links:
        print("🔍 No audio content detected")
        return None
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(audio_links_file))[0]
    audio_detected_file = os.path.join("output", f"4_5_{base_name.replace('4_', '')}_audio_detected.csv")
    pd.DataFrame(audio_detected_links).to_csv(audio_detected_file, index=False)
    
    # Show audio type breakdown
    audio_types = {}
    confidence_levels = {}
    for link in audio_detected_links:
        audio_type = link.get('audio_type', 'unknown')
        confidence = link.get('audio_confidence', 'unknown')
        audio_types[audio_type] = audio_types.get(audio_type, 0) + 1
        confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1

    print(f"✅ Stage 4.5 completed!")
    print(f"🔊 Audio content detected: {len(audio_detected_links)}")
    print("📊 Audio Content Breakdown:")
    print("  Audio Types:")
    for audio_type, count in sorted(audio_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {audio_type}: {count}")
    print("  Confidence Levels:")
    for confidence, count in confidence_levels.items():
        print(f"    {confidence}: {count}")
    print(f"📁 Audio detected file: {audio_detected_file}")
    print(f"💡 Next: Run Stage 5 with --stage5-only {audio_detected_file}")
    return audio_detected_file

def run_stage5_only(audio_links_file, output_dir="output"):
    """Run only Stage 5: Voice Content Verification"""
    print("🎙️ STAGE 5 ONLY: Voice Content Verification")
    print("=" * 50)

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
    verified_file = os.path.join(output_dir, f"5_{base_name.replace('4_5_', '').replace('4_', '')}_verified_voice.csv")
    pd.DataFrame(verified_links).to_csv(verified_file, index=False)
    print(f"📁 Saved verification results to: {verified_file}")

    # Save only confirmed voice content
    confirmed_voice = [link for link in verified_links if link['has_voice']]

    if confirmed_voice:
        confirmed_file = os.path.join(output_dir, f"5_{base_name.replace('4_5_', '').replace('4_', '')}_confirmed_voice.csv")
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
        print("  Voice Types:")
        for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {voice_type}: {count}")
        print("  Confidence Levels:")
        for confidence, count in confidence_levels.items():
            print(f"    {confidence}: {count}")

        # Show sample
        print("\n🎙️ Sample Confirmed Voice Content:")
        for i, link in enumerate(confirmed_voice[:3], 1):
            username = link.get('username', 'unknown')
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            url = link.get('url', '')[:50] + '...' if len(link.get('url', '')) > 50 else link.get('url', '')
            print(f"  {i}. @{username} | {voice_type} | {confidence}")
            print(f"     {url}")

    else:
        print("❌ No voice content confirmed")

    print(f"\n✅ Stage 5 completed!")
    print(f"📊 Total links processed: {len(audio_links)}")
    print(f"🎙️ Voice content found: {len(confirmed_voice)}")

def show_help():
    """Show detailed usage help"""
    help_text = """
🎙️ X/TWITTER VOICE CONTENT PIPELINE

USAGE:
    python main_pipeline.py [options]

FULL PIPELINE:
    --input FILE                 Run full pipeline on input file
    --force-recheck             Force recheck all accounts

INDIVIDUAL STAGES:
    --stage1-only FILE          Run only Stage 1 (Account Validation)
    --stage2-only FILE          Run only Stage 2 (Bright Data Trigger)
    --stage3-only SNAPSHOT_ID   Run only Stage 3 (Data Download)
    --stage4-only FILE          Run only Stage 4 (Audio Platform Filter)
    --stage4_5-only FILE        Run only Stage 4.5 (Audio Content Detection)
    --stage5-only FILE          Run only Stage 5 (Voice Verification)

INFORMATION:
    --show-log                  Show account validation summary
    --show-snapshots            Show Bright Data snapshots
    --analyze-duplicates        Analyze duplicate snapshots
    --clear-log                 Clear account validation cache
    --help-detailed             Show this help

EXAMPLES:

    # Full pipeline
    python main_pipeline.py --input usernames.csv

    # Stage by stage
    python main_pipeline.py --stage1-only usernames.csv
    python main_pipeline.py --stage2-only output/1_existing_accounts.csv
    python main_pipeline.py --stage3-only snap_12345
    python main_pipeline.py --stage4-only output/3_snapshot_snap_12345_external_links.csv
    python main_pipeline.py --stage4_5-only output/4_snapshot_snap_12345_audio_links.csv
    python main_pipeline.py --stage5-only output/4_5_snapshot_snap_12345_audio_detected.csv

PIPELINE STAGES:
    1. Account Validation       - Validate X/Twitter accounts exist
    2. Snapshot Management      - Create/reuse Bright Data collection
    3. Data Download           - Download profiles and extract links
    4. Audio Platform Filter   - Filter for audio/video platforms
    4.5 Audio Content Detection - Verify actual audio content exists
    5. Voice Verification      - Confirm voice/speech content (not music)
"""
    print(help_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X/Twitter Voice Content Pipeline with Advanced Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", help="Input CSV/TXT file with usernames for full pipeline")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")

    # Individual stage arguments
    parser.add_argument("--stage1-only", help="Run only Stage 1 - Account validation on input file")
    parser.add_argument("--stage2-only", help="Run only Stage 2 - Trigger Bright Data with accounts file")
    parser.add_argument("--stage3-only", help="Run only Stage 3 - Download snapshot data by snapshot ID")
    parser.add_argument("--stage4-only", help="Run only Stage 4 - Filter audio platforms from links file")
    parser.add_argument("--stage4_5-only", help="Run only Stage 4.5 - Detect audio content from audio links file")
    parser.add_argument("--stage5-only", help="Run only Stage 5 - Voice verification on audio links file")

    # Information commands
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

    # Handle individual stages
    if args.stage1_only:
        if not os.path.exists(args.stage1_only):
            print(f"❌ Input file not found: {args.stage1_only}")
            sys.exit(1)
        run_stage1_only(args.stage1_only, args.force_recheck)
        sys.exit(0)

    if args.stage2_only:
        if not os.path.exists(args.stage2_only):
            print(f"❌ Accounts file not found: {args.stage2_only}")
            sys.exit(1)
        snapshot_id = run_stage2_only(args.stage2_only)
        sys.exit(0)

    if args.stage3_only:
        snapshot_id = run_stage3_only(args.stage3_only)
        sys.exit(0)

    if args.stage4_only:
        if not os.path.exists(args.stage4_only):
            print(f"❌ Links file not found: {args.stage4_only}")
            sys.exit(1)
        audio_file = run_stage4_only(args.stage4_only)
        sys.exit(0)

    if args.stage4_5_only:
        if not os.path.exists(args.stage4_5_only):
            print(f"❌ Audio links file not found: {args.stage4_5_only}")
            sys.exit(1)
        audio_detected_file = run_stage4_5_only(args.stage4_5_only)
        sys.exit(0)

    if args.stage5_only:
        if not os.path.exists(args.stage5_only):
            print(f"❌ Audio links file not found: {args.stage5_only}")
            sys.exit(1)
        run_stage5_only(args.stage5_only, "output")
        sys.exit(0)

    # Handle information commands
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

    # Validate required arguments for full pipeline execution
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found")
            sys.exit(1)

        # Run main pipeline with comprehensive error handling
        try:
            print(f"🚀 Starting full pipeline with input: {args.input}")
            print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
            main(args.input, args.force_recheck)
        except KeyboardInterrupt:
            print("\n\n⏹️ Pipeline interrupted by user (Ctrl+C)")
            print("💾 All progress has been saved and can be resumed")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            print("💡 Check your configuration and try individual stages")
            sys.exit(1)

    else:
        print("❌ No action specified.")
        print("💡 Use --input FILE for full pipeline or --stage1-only FILE to start")
        print("📖 Use --help-detailed for complete usage guide")
        sys.exit(1)
