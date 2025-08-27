
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
from step6_voice_sample_extractor import VoiceSampleExtractor
from step7_advanced_voice_processor import AdvancedVoiceProcessor
from step8_noise_reduction import NoiseReducer
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    """Main pipeline execution - YouTube, Twitch & TikTok Voice Content Pipeline (8 stages with noise reduction)"""
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("🎙️ YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE")
    print("=" * 60)
    print("🎯 Focus: YouTube, Twitch, and TikTok voice content extraction")
    print("🎤 Output: voice samples → noise reduction → voice-only filtering")
    print("🔍 Stages: 8 comprehensive processing stages (with noise reduction)")


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

 
    # Stage 4: YouTube, Twitch & TikTok Audio Platform Filtering
    print("\n🎯 STAGE 4: YouTube, Twitch & TikTok Audio Platform Filtering")
    print("-" * 60)
    audio_filter = AudioContentFilter()
    audio_links = audio_filter.filter_audio_links(links)
    if not audio_links:
        print("🔍 No YouTube, Twitch or TikTok links found")
        print("⚠️ Pipeline completed but no supported platforms detected")
        return

    print(f"🎯 Found {len(audio_links)} YouTube/Twitch/TikTok audio links!")


    audio_file = os.path.join(cfg.OUTPUT_DIR, f"4_snapshot_{snapshot_id}_audio_links.csv")
    pd.DataFrame(audio_links).to_csv(audio_file, index=False)
    print(f"🎯 Found {len(audio_links)} YouTube/Twitch audio links!")
    print(f"📁 Saved to: {audio_file}")

    # Show platform breakdown
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1

    print("\n📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f" {platform}: {count}")


    # Stage 4.5: YouTube, Twitch & TikTok Audio Content Detection  
    print("\n🎵 STAGE 4.5: YouTube, Twitch & TikTok Audio Content Detection")
    print("-" * 60)

    audio_detector = AudioContentDetector(timeout=10)
    audio_detected_links = audio_detector.detect_audio_content(audio_links)

    if not audio_detected_links:
        print("🔍 No audio content detected in YouTube/Twitch links")
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

    # Stage 5: YouTube, Twitch & TikTok Voice Content Verification
    print("\n🎙️ STAGE 5: YouTube, Twitch & TikTok Voice Content Verification")
    print("-" * 60)

    voice_verifier = VoiceContentVerifier(timeout=15)
    verified_links = voice_verifier.verify_voice_content(audio_detected_links)

    # Save all verification results
    verified_file = os.path.join(cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_verified_voice.csv")
    pd.DataFrame(verified_links).to_csv(verified_file, index=False)
    print(f"📁 Saved verification results to: {verified_file}")

    # Save only confirmed voice content
    confirmed_voice = [link for link in verified_links if link.get('has_voice')]

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
        for i, link in enumerate(confirmed_voice[:3], 1):
            username = link.get('username', 'unknown')
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            url = link.get('url', '')[:60] + '...' if len(link.get('url', '')) > 60 else link.get('url', '')
            print(f"  {i}. @{username} | {voice_type} | {confidence} confidence")
            print(f"     {url}")

    else:
        print("❌ No voice content confirmed after verification")
        print("💡 This means the YouTube/Twitch links found were likely music or non-voice content")
        confirmed_voice = []  # Ensure it's an empty list for Stage 6

    # Stage 6: Voice Sample Extraction (30-second samples)
    print("\n🎤 STAGE 6: Voice Sample Extraction ")
    print("-" * 60)

    if confirmed_voice:
        sample_extractor = VoiceSampleExtractor(
            output_dir="voice_samples",
            max_duration_hours=1,  # Максимум 1 час
            quality="192"
        )

        
        extracted_samples = sample_extractor.extract_voice_samples(confirmed_voice)
        
        if extracted_samples:
            # Save extraction results
            extraction_file = os.path.join(cfg.OUTPUT_DIR, f"6_snapshot_{snapshot_id}_voice_samples.csv")
            pd.DataFrame(extracted_samples).to_csv(extraction_file, index=False)
            print(f"📁 Saved extraction results to: {extraction_file}")
            
            # Generate samples report
            report_file = sample_extractor.generate_samples_report(extracted_samples)
            
            print(f"\n🎤 Voice Sample Extraction Summary:")
            print(f"  📊 Total voice links: {len(confirmed_voice)}")
            print(f"  ✅ Successful extractions: {len(extracted_samples)}")
            print(f"  📁 Samples directory: {sample_extractor.output_dir}")
            print(f"  📄 Report file: {report_file}")
            print(f"  ⏱️ Sample duration: 30 seconds each")
            
            # Show extracted files
            print(f"\n🎵 Extracted Sample Files:")
            for sample in extracted_samples:
                filename = sample.get('sample_filename', 'N/A')
                username = sample.get('processed_username', 'unknown')
                platform = sample.get('platform_source', 'unknown')
                start_time = sample.get('start_time_formatted', '0:00')
                file_size = sample.get('file_size_bytes', 0)
                print(f"  📄 {filename} (@{username} {platform} from {start_time}, {file_size//1000}KB)")
            
        else:
            print("❌ No voice samples could be extracted")
            print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
            extracted_samples = []
    else:
        print("⏭️ Skipping voice sample extraction - no confirmed voice content")
        extracted_samples = []

    # Stage 8: Background Noise Reduction (NEW STAGE)
    print("\n🎛️ STAGE 8: Background Noise Reduction")
    print("-" * 60)

    if extracted_samples:
        # Determine samples directory from extracted samples
        first_sample_path = extracted_samples[0].get('sample_file')
        if first_sample_path and os.path.exists(first_sample_path):
            samples_dir_for_nr = os.path.dirname(first_sample_path)
        else:
            samples_dir_for_nr = os.path.join(cfg.OUTPUT_DIR, "voice_samples")

        # Initialize noise reducer with default settings
        noise_reducer = NoiseReducer(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "voice_analysis"),
            mode="quick",  # Default to quick mode
            sample_rate=16000,
            highpass_hz=80,
            lowpass_hz=6000,
            afftdn_nr=24.0,
            afftdn_nf=12.0
        )
        
        # Process the samples directory
        nr_results = noise_reducer.process_directory(samples_dir_for_nr)
        denoised_dir = os.path.join(cfg.OUTPUT_DIR, "voice_analysis", "denoised_audio")
        
        successful_denoising = sum(1 for r in nr_results if r.get('output_file'))
        print(f"✅ Noise reduction completed: {successful_denoising} files denoised")
        print(f"📁 Denoised files saved to: {denoised_dir}")
        
        # Update extracted_samples to point to denoised files where available
        if successful_denoising > 0:
            for sample in extracted_samples:
                orig_path = sample.get('sample_file', '')
                if orig_path:
                    base_name = os.path.splitext(os.path.basename(orig_path))[0]
                    denoised_path = os.path.join(denoised_dir, f"{base_name}_denoised.wav")
                    if os.path.exists(denoised_path):
                        sample['sample_file'] = denoised_path
                        sample['is_denoised'] = True
                    else:
                        sample['is_denoised'] = False
    else:
        print("⏭️ Skipping noise reduction - no extracted samples")

    # Stage 7: Audio Content Filtering (Voice-Only Detection) - moved after Stage 8
    print("\n🔍 STAGE 7: Audio Content Filtering (Voice-Only Detection)")
    print("-" * 60)

    if extracted_samples:
        # Use AdvancedVoiceProcessor instead of SimpleAudioContentAnalyzer
        processor = AdvancedVoiceProcessor(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "audio_analysis"),
            min_voice_confidence=0.6,
            voice_segment_min_length=2.0
        )
        
        # Create temporary directory with audio files for processing
        temp_audio_dir = os.path.join(cfg.OUTPUT_DIR, "temp_audio_for_processing")
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        # Copy/link audio files to temp directory for batch processing
        for sample in extracted_samples:
            sample_file = sample.get('sample_file')
            if sample_file and os.path.exists(sample_file):
                import shutil
                dest_file = os.path.join(temp_audio_dir, os.path.basename(sample_file))
                if not os.path.exists(dest_file):
                    shutil.copy2(sample_file, dest_file)
        
        # Process the audio directory
        voice_only_results = processor.process_audio_directory(temp_audio_dir)
        
        if voice_only_results:
            # Save results
            results_file = processor.save_results(voice_only_results)
            report_file = processor.generate_report(voice_only_results)
            
            # Save simplified CSV for compatibility
            voice_only_file = os.path.join(cfg.OUTPUT_DIR, f"7_snapshot_{snapshot_id}_voice_only.csv")
            simplified_results = []
            for result in voice_only_results:
                simplified_results.append({
                    'processed_username': result.get('username', 'unknown'),
                    'platform_source': result.get('platform', 'unknown'),
                    'voice_only_file': result.get('voice_only_file', ''),
                    'speech_text': result.get('speech_analysis', {}).get('combined_text', ''),
                    'voice_confidence': result.get('final_analysis', {}).get('final_confidence', 0),
                    'word_count': result.get('speech_analysis', {}).get('word_count', 0),
                    'voice_duration': result.get('voice_duration', 0)
                })
            
            pd.DataFrame(simplified_results).to_csv(voice_only_file, index=False)
            
            print(f"🔍 Audio Content Filtering Summary:")
            print(f"  📊 Total audio samples: {len(extracted_samples)}")
            print(f"  ✅ Voice-only samples: {len(voice_only_results)}")
            print(f"  ❌ Filtered out: {len(extracted_samples) - len(voice_only_results)}")
            print(f"  📈 Voice detection rate: {(len(voice_only_results) / len(extracted_samples) * 100):.1f}%")
            print(f"  📁 Voice-only CSV: {voice_only_file}")
            print(f"  📄 Advanced analysis report: {report_file}")
            print(f"  🎵 Voice-only audio: {processor.voice_only_dir}")
            
            # Show sample results
            print(f"\n🎤 Sample Voice-Only Content:")
            for i, result in enumerate(voice_only_results[:3], 1):
                username = result.get('username', 'unknown')
                speech_text = result.get('speech_analysis', {}).get('combined_text', '')[:50]
                confidence = result.get('final_analysis', {}).get('final_confidence', 0)
                word_count = result.get('speech_analysis', {}).get('word_count', 0)
                platform = result.get('platform', 'unknown')
                print(f"  {i}. @{username} ({platform}) | \"{speech_text}...\"")
                print(f"     📊 Confidence: {confidence:.2f} | Words: {word_count}")
            
            voice_only_samples = simplified_results
        else:
            print("❌ No voice-only content found after filtering")
            print("💡 All audio samples contained music, noise, or unclear content")
            voice_only_samples = []
        
        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_audio_dir):
            shutil.rmtree(temp_audio_dir)
            
    else:
        print("⏭️ Skipping audio content filtering - no audio samples extracted")
        voice_only_samples = []

    # Final comprehensive summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total accounts processed: {len(valid_accounts)}")
    print(f"📥 Profiles downloaded: {len(profiles)}")
    print(f"🔗 External links found: {len(links)}")
    print(f"🎯 YouTube/Twitch/TikTok links: {len(audio_links)}")
    print(f"🔊 Audio content confirmed: {len(audio_detected_links)}")
    print(f"🎙️ Voice content confirmed: {len(confirmed_voice)}")
    print(f"🎤 Voice samples extracted: {len(extracted_samples) if 'extracted_samples' in locals() else 0}")
    print(f"🎛️ Samples denoised: {successful_denoising if 'successful_denoising' in locals() else 0}")
    print(f"✅ Voice-only samples (filtered): {len(voice_only_samples) if 'voice_only_samples' in locals() else 0}")
    print(f"📈 Voice confirmation rate: {(len(confirmed_voice) / len(audio_links) * 100):.1f}%" if audio_links else "0%")
    print(f"📈 Voice-only filtering rate: {(len(voice_only_samples) / len(extracted_samples) * 100):.1f}%" if 'extracted_samples' in locals() and extracted_samples and 'voice_only_samples' in locals() else "0%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")

    # Final output files summary
    print(f"\n📄 Output Files Generated:")
    print(f"  1. {existing_accounts_file} - Validated accounts")
    print(f"  2. {profiles_file} - Profile data")
    print(f"  3. {links_file} - External links")
    print(f"  4. {audio_file} - YouTube/Twitch links")
    print(f"  5. {audio_detected_file} - Audio content detected")
    print(f"  6. {verified_file} - Voice verification results")
    if confirmed_voice:
        print(f"  7. {confirmed_file} - ⭐ CONFIRMED VOICE CONTENT")
    if 'extracted_samples' in locals() and extracted_samples:
        print(f"  8. {extraction_file} - 🎤 VOICE SAMPLE EXTRACTION RESULTS")
        print(f"  9. {sample_extractor.output_dir} - 🎵 VOICE SAMPLES DIRECTORY")
        print(f"  10. {denoised_dir} - 🎛️ DENOISED AUDIO FILES (Stage 8)")
    if 'voice_only_samples' in locals() and voice_only_samples:
        print(f"  11. {voice_only_file} - ✅ VOICE-ONLY FILTERED RESULTS")
        print(f"  12. {processor.voice_only_dir} - 🎤 VOICE-ONLY AUDIO FILES")

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
    """Run only Stage 4: YouTube, Twitch & TikTok Audio Platform Filtering"""
    print("🎯 STAGE 4 ONLY: YouTube, Twitch & TikTok Audio Platform Filtering")
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
        print("🔍 No YouTube or Twitch links found")
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
    print(f"🎯 YouTube/Twitch/TickTok links: {len(audio_links)}")
    print("📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count}")
    print(f"📁 Audio file: {audio_file}")
    print(f"💡 Next: Run Stage 4.5 with --stage4_5-only {audio_file}")
    return audio_file

def run_stage4_5_only(audio_links_file):
    """Run only Stage 4.5: Audio Content Detection"""
    print("🎵 STAGE 4.5 ONLY: YouTube, Twitch & TikTok Audio Content Detection")
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
    print("🎙️ STAGE 5 ONLY: YouTube, Twitch & TikTok Voice Content Verification")
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
    confirmed_voice = [link for link in verified_links if link.get('has_voice')]

    if confirmed_voice:
        confirmed_file = os.path.join(output_dir, f"5_{base_name.replace('4_5_', '').replace('4_', '')}_confirmed_voice.csv")
        pd.DataFrame(confirmed_voice).to_csv(confirmed_file, index=False)
        print(f"🎙️ Found {len(confirmed_voice)} confirmed voice content links!")
        print(f"📁 Saved to: {confirmed_file}")

        # Analysis
        voice_types = {}
        confidence_levels = {}
        platforms = {}
        for link in confirmed_voice:
            voice_type = link.get('voice_type', 'unknown')
            confidence = link.get('voice_confidence', 'unknown')
            platform = link.get('platform_type', 'unknown')
            voice_types[voice_type] = voice_types.get(voice_type, 0) + 1
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1
            platforms[platform] = platforms.get(platform, 0) + 1

        print("\n📊 Voice Content Analysis:")
        print("  Voice Types:")
        for voice_type, count in sorted(voice_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {voice_type}: {count}")
        print("  Platforms:")
        for platform, count in platforms.items():
            print(f"    {platform}: {count}")
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

        print(f"💡 Next: Run Stage 6 with --stage6-only {confirmed_file}")

    else:
        print("❌ No voice content confirmed")

    print(f"\n✅ Stage 5 completed!")
    print(f"📊 Total links processed: {len(audio_links)}")
    print(f"🎙️ Voice content found: {len(confirmed_voice)}")

def run_stage6_only(confirmed_voice_file, output_dir="output"):
    """Run only Stage 6: Voice Sample Extraction (30s samples)"""
    print("🎤 STAGE 6 ONLY: Voice Sample Extraction (30s samples)")
    print("=" * 50)
    
    # Load confirmed voice links
    if not os.path.exists(confirmed_voice_file):
        print(f"❌ Confirmed voice file not found: {confirmed_voice_file}")
        return
    
    try:
        df = pd.read_csv(confirmed_voice_file)
        confirmed_voice = df.to_dict('records')
        print(f"📥 Loaded {len(confirmed_voice)} confirmed voice links from: {confirmed_voice_file}")
    except Exception as e:
        print(f"❌ Error loading confirmed voice links: {e}")
        return
    
    if not confirmed_voice:
        print("❌ No confirmed voice links found in file")
        return
    
    
    sample_extractor = VoiceSampleExtractor(
        output_dir=os.path.join(output_dir, "voice_samples"),
        max_duration_hours=1, 
        quality="192",
        min_duration=30,    
        max_duration=3600     
        )

    
    extracted_samples = sample_extractor.extract_voice_samples(confirmed_voice)
    
    if extracted_samples:
        # Save results
        base_name = os.path.splitext(os.path.basename(confirmed_voice_file))[0]
        extraction_file = os.path.join(output_dir, f"6_{base_name}_voice_samples.csv")
        pd.DataFrame(extracted_samples).to_csv(extraction_file, index=False)
        
        # Generate report
        report_file = sample_extractor.generate_samples_report(extracted_samples)
        
        print(f"✅ Stage 6 completed!")
        print(f"🎤 Successfully extracted {len(extracted_samples)} voice samples ")
        print(f"📁 Results: {extraction_file}")
        print(f"📄 Report: {report_file}")
        print(f"🎵 Samples directory: {sample_extractor.output_dir}")
        print(f"⏱️ Sample duration: 30 seconds per sample")
        print(f"💡 Next: Run Stage 8 with --stage8-only {sample_extractor.output_dir}")
        
        # Show extracted files
        print(f"\n🎵 Extracted Sample Files:")
        for sample in extracted_samples:
            filename = sample.get('sample_filename', 'N/A')
            username = sample.get('processed_username', 'unknown')
            platform = sample.get('platform_source', 'unknown')
            start_time = sample.get('start_time_formatted', '0:00')
            file_size = sample.get('file_size_bytes', 0)
            print(f"  📄 {filename} (@{username} {platform} from {start_time}, {file_size//1000}KB)")
        
        # Clean temporary files
        sample_extractor.clean_temp_files()
        
    else:
        print("❌ No voice samples could be extracted")
        print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
        print("💡 Also ensure the confirmed voice links are accessible")

def run_stage8_only(wavs_dir, output_dir="voice_analysis", nr_mode="quick", nr_noise_file=None, nr_sr=16000, nr_highpass=80, nr_lowpass=6000, nr_db=24.0, nr_floor=12.0):
    """Run only Stage 8: Noise reduction on directory with WAV files"""
    print("🎛️ STAGE 8 ONLY: Background Noise Reduction")
    print("=" * 50)
    
    if not os.path.isdir(wavs_dir):
        print(f"❌ Directory not found: {wavs_dir}")
        return
    
    reducer = NoiseReducer(
        output_dir=output_dir,
        mode=nr_mode,
        noise_profile_file=nr_noise_file,
        sample_rate=nr_sr,
        highpass_hz=nr_highpass,
        lowpass_hz=nr_lowpass,
        afftdn_nr=nr_db,
        afftdn_nf=nr_floor
    )
    
    results = reducer.process_directory(wavs_dir)
    
    successful_count = sum(1 for r in results if r.get('output_file'))
    print(f"\n✅ Stage 8 completed!")
    print(f"🎛️ Successfully denoised: {successful_count} files")
    print(f"📁 Denoised files saved to: {reducer.denoised_dir}")
    print(f"💡 Next: Run Stage 7 with audio samples (will use denoised files)")

def run_stage7_only(extracted_samples_file, output_dir="output"):
    """Run only Stage 7: Audio Content Filtering (Voice-Only Detection)"""
    print("🔍 STAGE 7 ONLY: Audio Content Filtering (Voice-Only Detection)")
    print("=" * 50)
    
    # Load extracted samples
    if not os.path.exists(extracted_samples_file):
        print(f"❌ Extracted samples file not found: {extracted_samples_file}")
        return
    
    try:
        df = pd.read_csv(extracted_samples_file)
        extracted_samples = df.to_dict('records')
        print(f"📥 Loaded {len(extracted_samples)} audio samples from: {extracted_samples_file}")
    except Exception as e:
        print(f"❌ Error loading extracted samples: {e}")
        return
    
    if not extracted_samples:
        print("❌ No audio samples found in file")
        return
    
    # Use AdvancedVoiceProcessor
    processor = AdvancedVoiceProcessor(
        output_dir=os.path.join(output_dir, "audio_analysis"),
        min_voice_confidence=0.6,
        voice_segment_min_length=2.0
    )
    
    # Create temporary directory with audio files for processing
    temp_audio_dir = os.path.join(output_dir, "temp_audio_for_processing")
    os.makedirs(temp_audio_dir, exist_ok=True)
    
    # Check for denoised files first
    denoised_dir = os.path.join(output_dir, "voice_analysis", "denoised_audio")
    
    # Copy/link audio files to temp directory for batch processing
    for sample in extracted_samples:
        sample_file = sample.get('sample_file')
        if sample_file and os.path.exists(sample_file):
            # Check for denoised version first
            base_name = os.path.splitext(os.path.basename(sample_file))[0]
            denoised_path = os.path.join(denoised_dir, f"{base_name}_denoised.wav")
            
            if os.path.exists(denoised_path):
                source_file = denoised_path
                print(f"  🎛️ Using denoised version: {os.path.basename(denoised_path)}")
            else:
                source_file = sample_file
                print(f"  🎵 Using original file: {os.path.basename(sample_file)}")
            
            import shutil
            dest_file = os.path.join(temp_audio_dir, os.path.basename(source_file))
            if not os.path.exists(dest_file):
                shutil.copy2(source_file, dest_file)
    
    # Process the audio directory
    voice_only_results = processor.process_audio_directory(temp_audio_dir)
    
    if voice_only_results:
        # Save results
        results_file = processor.save_results(voice_only_results)
        report_file = processor.generate_report(voice_only_results)
        
        # Save simplified CSV for compatibility
        base_name = os.path.splitext(os.path.basename(extracted_samples_file))[0]
        voice_only_file = os.path.join(output_dir, f"7_{base_name}_voice_only.csv")
        simplified_results = []
        for result in voice_only_results:
            simplified_results.append({
                'processed_username': result.get('username', 'unknown'),
                'platform_source': result.get('platform', 'unknown'),
                'voice_only_file': result.get('voice_only_file', ''),
                'speech_text': result.get('speech_analysis', {}).get('combined_text', ''),
                'voice_confidence': result.get('final_analysis', {}).get('final_confidence', 0),
                'word_count': result.get('speech_analysis', {}).get('word_count', 0),
                'voice_duration': result.get('voice_duration', 0)
            })
        
        pd.DataFrame(simplified_results).to_csv(voice_only_file, index=False)
        
        print(f"✅ Stage 7 completed!")
        print(f"🎤 Voice-only samples found: {len(voice_only_results)}")
        print(f"📁 Voice-only results: {voice_only_file}")
        print(f"📄 Advanced analysis report: {report_file}")
        print(f"🎵 Voice-only audio files: {processor.voice_only_dir}")
        print(f"📈 Voice detection rate: {(len(voice_only_results) / len(extracted_samples) * 100):.1f}%")
        
        # Show sample results
        print(f"\n🎤 Sample Voice-Only Content:")
        for i, result in enumerate(voice_only_results[:3], 1):
            username = result.get('username', 'unknown')
            speech_text = result.get('speech_analysis', {}).get('combined_text', '')[:40]
            confidence = result.get('final_analysis', {}).get('final_confidence', 0)
            word_count = result.get('speech_analysis', {}).get('word_count', 0)
            print(f"  {i}. @{username} | \"{speech_text}...\" | {confidence:.2f} confidence ({word_count} words)")
        
    else:
        print("❌ No voice-only samples found")
        print("💡 Try lowering minimum confidence or check audio quality")
        print("💡 All samples may contain music or non-voice content")
    
    # Cleanup temp directory
    import shutil
    if os.path.exists(temp_audio_dir):
        shutil.rmtree(temp_audio_dir)

def show_help():
    """Show detailed usage help"""
    help_text = """
🎙️ YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE (8-Stage Processing)
This pipeline validates X/Twitter accounts, collects profile data through Bright Data API,
extracts external links, filters for YouTube, Twitch & TikTok platforms, detects audio content,
verifies voice/speech content, extracts  voice samples, and filters for voice-only content.

SUPPORTED PLATFORMS:
- YouTube (youtube.com, youtu.be)
- Twitch (twitch.tv, m.twitch.tv)  
- TikTok (tiktok.com, www.tiktok.com, vm.tiktok.com, m.tiktok.com)

VOICE CONTENT TYPES DETECTED:
- Podcasts (episodes, shows)
- Interviews and conversations
- Educational content (lectures, tutorials)
- Commentary and analysis
- Live talks (Twitch Just Chatting)
- Gaming commentary
- TikTok voice content (stories, tutorials, reactions)



USAGE:
python main_pipeline.py [options]

FULL PIPELINE:
--input FILE                 Run full pipeline on input file
--force-recheck             Force recheck all accounts

INDIVIDUAL STAGES:
--stage1-only FILE          Run only Stage 1 (Account Validation)
--stage2-only FILE          Run only Stage 2 (Bright Data Trigger)
--stage3-only SNAPSHOT_ID   Run only Stage 3 (Data Download)
--stage4-only FILE          Run only Stage 4 (YouTube/Twitch Filter)
--stage4_5-only FILE        Run only Stage 4.5 (Audio Content Detection)
--stage5-only FILE          Run only Stage 5 (Voice Verification)
--stage6-only FILE          Run only Stage 6 (Voice Sample Extraction)
--stage8-only DIR           Run only Stage 8 (Noise Reduction on WAV directory)
--stage7-only FILE          Run only Stage 7 (Voice-Only Filtering)

STAGE 8 NOISE REDUCTION OPTIONS:
--nr-mode {quick,profile}   Noise reduction mode (default: quick)
--nr-noise-file FILE        Noise profile WAV for 'profile' mode
--nr-sr INT                 Target sample rate (default: 16000)
--nr-highpass INT           Highpass cutoff Hz (default: 80)
--nr-lowpass INT            Lowpass cutoff Hz (default: 6000)
--nr-db FLOAT               afftdn noise reduction dB (default: 24.0)
--nr-floor FLOAT            afftdn noise floor dB (default: 12.0)

INFORMATION:
--show-log                  Show account validation summary
--show-snapshots            Show Bright Data snapshots
--analyze-duplicates        Analyze duplicate snapshots
--clear-log                 Clear account validation cache
--help-detailed             Show this help

EXAMPLES:

Full pipeline:
python main_pipeline.py --input usernames.csv

Stage 8 only (noise reduction):
python main_pipeline.py --stage8-only output/voice_samples --nr-mode quick
python main_pipeline.py --stage8-only output/voice_samples --nr-mode profile --nr-noise-file noise.wav

Stage by stage:
python main_pipeline.py --stage1-only usernames.csv
python main_pipeline.py --stage2-only output/1_existing_accounts.csv
python main_pipeline.py --stage3-only snap_12345
python main_pipeline.py --stage4-only output/3_snapshot_snap_12345_external_links.csv
python main_pipeline.py --stage4_5-only output/4_snapshot_snap_12345_audio_links.csv
python main_pipeline.py --stage5-only output/4_5_snapshot_snap_12345_audio_detected.csv
python main_pipeline.py --stage6-only output/5_snapshot_snap_12345_confirmed_voice.csv
python main_pipeline.py --stage8-only output/voice_samples --nr-mode quick
python main_pipeline.py --stage7-only output/6_snapshot_snap_12345_voice_samples.csv

PIPELINE STAGES:
1. Account Validation       - Validate X/Twitter accounts exist
2. Snapshot Management      - Create/reuse Bright Data collection
3. Data Download           - Download profiles and extract links
4. Audio Platform Filter   - Filter for YouTube/Twitch only
4.5 Audio Content Detection - Verify actual audio content exists
5. Voice Verification      - Confirm voice/speech content (not music)
6. Voice Sample Extraction - Extract voice samples
8. Background Noise Reduction - Clean audio samples using ffmpeg filters
7. Voice-Only Filtering    - Advanced voice activity detection & speech recognition

SUPPORTED PLATFORMS:
- YouTube (youtube.com, youtu.be)
- Twitch (twitch.tv, m.twitch.tv)

SAMPLE SPECIFICATIONS:
- Duration: 30 seconds per sample
- Quality: 192 kbps original, 16kHz WAV after noise reduction
- YouTube timing: 0:00-0:30 (from beginning)
- Twitch timing: 3:00-3:30 (skip intro music)
- Noise reduction: ffmpeg afftdn + filters
- Voice-only filtering: Advanced VAD + Speech Recognition

DEPENDENCIES:
pip install requests pandas playwright yt-dlp SpeechRecognition numpy
playwright install chromium
ffmpeg (required for noise reduction)
"""
    print(help_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube, Twitch & TikTok Voice Content Pipeline with 8-Stage Processing (Noise Reduction included)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input", help="Input CSV/TXT file with usernames for full pipeline")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")

    # Individual stage arguments
    parser.add_argument("--stage1-only", help="Run only Stage 1 - Account validation on input file")
    parser.add_argument("--stage2-only", help="Run only Stage 2 - Trigger Bright Data with accounts file")
    parser.add_argument("--stage3-only", help="Run only Stage 3 - Download snapshot data by snapshot ID")
    parser.add_argument("--stage4-only", help="Run only Stage 4 - Filter YouTube/Twitch from links file")
    parser.add_argument("--stage4_5-only", help="Run only Stage 4.5 - Detect audio content from audio links file")
    parser.add_argument("--stage5-only", help="Run only Stage 5 - Voice verification on audio links file")
    parser.add_argument("--stage6-only", help="Run only Stage 6 - Extract voice samples from confirmed voice file")
    parser.add_argument("--stage8-only", help="Run only Stage 8 - Noise reduction on WAV directory")
    parser.add_argument("--stage7-only", help="Run only Stage 7 - Filter voice-only content from extracted samples file")

    # Stage 8 Noise Reduction options
    parser.add_argument("--nr-mode", choices=["quick", "profile"], default="quick", help="Noise reduction mode")
    parser.add_argument("--nr-noise-file", help="Noise profile WAV for 'profile' mode")
    parser.add_argument("--nr-sr", type=int, default=16000, help="Target sample rate for denoised output")
    parser.add_argument("--nr-highpass", type=int, default=80, help="Highpass cutoff Hz")
    parser.add_argument("--nr-lowpass", type=int, default=6000, help="Lowpass cutoff Hz")
    parser.add_argument("--nr-db", type=float, default=24.0, help="afftdn noise reduction dB")
    parser.add_argument("--nr-floor", type=float, default=12.0, help="afftdn noise floor dB")

    # Information commands
    parser.add_argument("--show-log", action="store_true", help="Show account validation log summary")
    parser.add_argument("--show-snapshots", action="store_true", help="Show Bright Data snapshots summary")
    parser.add_argument("--analyze-duplicates", action="store_true", help="Analyze duplicate snapshots")
    parser.add_argument("--clear-log", action="store_true", help="Clear processed accounts log")
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed usage help")

    args = parser.parse_args()

    # Handle help and info commands FIRST
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
        run_stage2_only(args.stage2_only)
        sys.exit(0)

    if args.stage3_only:
        run_stage3_only(args.stage3_only)
        sys.exit(0)

    if args.stage4_only:
        if not os.path.exists(args.stage4_only):
            print(f"❌ Links file not found: {args.stage4_only}")
            sys.exit(1)
        run_stage4_only(args.stage4_only)
        sys.exit(0)

    if args.stage4_5_only:
        if not os.path.exists(args.stage4_5_only):
            print(f"❌ Audio links file not found: {args.stage4_5_only}")
            sys.exit(1)
        run_stage4_5_only(args.stage4_5_only)
        sys.exit(0)

    if args.stage5_only:
        if not os.path.exists(args.stage5_only):
            print(f"❌ Audio links file not found: {args.stage5_only}")
            sys.exit(1)
        run_stage5_only(args.stage5_only, "output")
        sys.exit(0)

    if args.stage6_only:
        if not os.path.exists(args.stage6_only):
            print(f"❌ Confirmed voice file not found: {args.stage6_only}")
            sys.exit(1)
        run_stage6_only(args.stage6_only, "output")
        sys.exit(0)

    if args.stage8_only:
        run_stage8_only(
            args.stage8_only,
            output_dir="voice_analysis",
            nr_mode=args.nr_mode,
            nr_noise_file=args.nr_noise_file,
            nr_sr=args.nr_sr,
            nr_highpass=args.nr_highpass,
            nr_lowpass=args.nr_lowpass,
            nr_db=args.nr_db,
            nr_floor=args.nr_floor
        )
        sys.exit(0)

    if args.stage7_only:
        if not os.path.exists(args.stage7_only):
            print(f"❌ Extracted samples file not found: {args.stage7_only}")
            sys.exit(1)
        run_stage7_only(args.stage7_only, "output")
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
            print(f"🚀 Starting full 8-stage pipeline with input: {args.input}")
            print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
            print(f"⏱️ Configuration: 30-second samples -> noise reduction -> voice-only filtering")
            main(args.input, args.force_recheck)
        except KeyboardInterrupt:
            print("\n\n⏹️ Pipeline interrupted by user (Ctrl+C)")
            print("💾 All progress has been saved and can be resumed")
            print("🔄 Run individual stages to continue from where you left off")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            print("💡 Check your configuration and try individual stages for debugging")
            print("📋 Use --show-log to see processed accounts")
            print("🔍 Use --show-snapshots to see snapshot history")
            sys.exit(1)
    else:
        # NO ARGUMENTS PROVIDED - Show help
        print("❌ No action specified.")
        print("💡 Use --input FILE for full pipeline or --stage1-only FILE to start")
        print("📖 Use --help-detailed for complete usage guide")
        print("\n🎯 Quick start examples:")
        print("  python main_pipeline.py --input usernames.csv")
        print("  python main_pipeline.py --stage8-only output/voice_samples --nr-mode quick")
        print("  python main_pipeline.py --stage7-only output/6_voice_samples.csv")
        print("  python main_pipeline.py --show-log")
        print("\n⏱️ Current configuration: 8-stage processing with noise reduction")
        sys.exit(1)
