import os
import pandas as pd
import argparse
import sys
import subprocess
from config import Config
from step1_validate_accounts import AccountValidator
from step2_bright_data_trigger import BrightDataTrigger
from step3_bright_data_download import BrightDataDownloader
from step3_5_youtube_twitch_runner import Step3_5_YouTubeTwitchRunner
from step4_audio_filter import AudioContentFilter
from step5_voice_sample_extractor import AudioDownloader, save_results
from step6_voice_detector_advance import AdvancedVoiceDetector # Updated import
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    """Main pipeline execution - Whisper Enhanced Pipeline with MP3 to WAV conversion handling"""
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("🎙️ WHISPER ENHANCED VOICE CONTENT PIPELINE")
    print("=" * 60)
    print("🎯 Focus: YouTube, Twitch, and TikTok voice content extraction")
    print("🤖 AI Model: OpenAI Whisper for speech recognition and overlap detection")
    print("🎤 Pipeline: MP3 → WAV conversion → Whisper Processing → Transcription")
    print("🔍 Stages: 6 comprehensive processing stages (1→2→3→4→5→6)")
    print("🔄 Audio Flow: Stage 5 (MP3) → Stage 6 (WAV+Transcripts)")

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
    sm = SnapshotManager(cfg.OUTPUT_DIR)
    existing_snapshot = sm.get_reusable_snapshot(usernames)
    
    if existing_snapshot:
        print(f"🔄 Using existing snapshot: {existing_snapshot}")
        snapshot_id = existing_snapshot
    else:
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
    
    sm.update_snapshot_status(snapshot_id, "completed", profiles)
    profiles_file = os.path.join(cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
    pd.DataFrame(profiles).to_csv(profiles_file, index=False)
    print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
    
    links = downloader.extract_external_links(profiles)
    if not links:
        print("🔗 No external links found in profiles")
        print("⚠️ Pipeline completed but no links to process further")
        return
    
    links_file = os.path.join(cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
    pd.DataFrame(links).to_csv(links_file, index=False)
    print(f"🔗 Saved {len(links)} external links to: {links_file}")
    print(f"✅ Stage 3 completed!")
    print(f"🔗 External links found: {len(links)}")
    print(f"📁 Links file: {links_file}")
    print(f"💡 Next: Run Stage 3.5 with --stage3_5-only {links_file}")
    print(f"💡 Alternative: Skip Stage 3.5 and run Stage 4 with --stage4-only {links_file}")
    return links_file

    # Stage 3.5: YouTube & Twitch Channel Discovery
    print("\n🔍 STAGE 3.5: YouTube & Twitch Channel Discovery")
    print("-" * 60)
    runner = Step3_5_YouTubeTwitchRunner(cfg.OUTPUT_DIR)
    enhanced_file = runner.run_scraper_for_snapshot(snapshot_id)
    
    if enhanced_file:
        print(f"✅ Stage 3.5 completed: {enhanced_file}")
        enhanced_links = pd.read_csv(enhanced_file).to_dict('records')
    else:
        print("⚠️ Stage 3.5 failed, using original external links")
        enhanced_links = links

    # Stage 4: YouTube, Twitch & TikTok Audio Platform Filtering
    print("\n🎯 STAGE 4: YouTube, Twitch & TikTok Audio Platform Filtering")
    print("-" * 60)
    audio_filter = AudioContentFilter()
    audio_links = audio_filter.filter_audio_links(enhanced_links)
    
    if not audio_links:
        print("🔍 No YouTube, Twitch or TikTok links found")
        print("⚠️ Pipeline completed but no supported platforms detected")
        return
    
    print(f"🎯 Found {len(audio_links)} YouTube/Twitch/TikTok audio links!")
    audio_file = os.path.join(cfg.OUTPUT_DIR, f"4_snapshot_{snapshot_id}_audio_links.csv")
    pd.DataFrame(audio_links).to_csv(audio_file, index=False)
    print(f"📁 Saved to: {audio_file}")
    
    # Platform breakdown
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
    
    print("\n📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count}")

    # Stage 5: Voice Sample Extraction (Outputs MP3 files)
    print("\n🎤 STAGE 5: Voice Sample Extraction (MP3 Output)")
    print("-" * 60)
    confirmed_voice = audio_links  # Use audio_links directly, skipping step 5
    
    if confirmed_voice:
        sample_extractor = AudioDownloader(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "voice_samples")
        )
        
        extracted_samples = sample_extractor.download_audio_for_all(confirmed_voice)
        if extracted_samples:
            extraction_file = os.path.join(cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_voice_samples.csv")
            save_results(extracted_samples, extraction_file)
            print(f"📁 Saved extraction results to: {extraction_file}")
            print(f"✅ Successfully processed {len(extracted_samples)} voice samples")
            print(f"📁 Samples directory: {sample_extractor.output_dir}")
            
            print(f"\n🎤 Voice Sample Extraction Summary:")
            print(f"  📊 Total voice links: {len(confirmed_voice)}")
            print(f"  ✅ Successful extractions: {len(extracted_samples)}")
            print(f"  📁 Samples directory: {sample_extractor.output_dir}")
            print(f"  ⏱️ Sample duration: up to 1 hour each")
            print(f"  🎵 Output format: MP3 (192kbps)")
            
            print(f"\n🎵 Extracted MP3 Sample Files:")
            for i, sample in enumerate(extracted_samples[:10], 1):
                username = sample.get('username', 'unknown')
                platform = sample.get('platform', 'unknown')
                chunks = sample.get('chunks', 1)
                success = sample.get('success', False)
                print(f"  {i}. @{username} ({platform}) - {chunks} chunk(s) - {'✅' if success else '❌'}")
        else:
            print("❌ No voice samples could be extracted")
            print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
            extracted_samples = []
    else:
        print("⏭️ Skipping voice sample extraction - no confirmed voice content")
        extracted_samples = []

    # Stage 6: Advanced Voice Detection (Enhanced voice analysis)
    if extracted_samples:
        print("\n🎤 STAGE 6: Advanced Voice Detection (Enhanced Analysis)")
        print("-" * 60)
        print("🎤 Processing audio files with advanced voice detection")
        print("📝 Using Whisper, Pyannote VAD, and SpeechBrain for comprehensive analysis")
        
        try:
            cfg = Config()
            advanced_processor = AdvancedVoiceDetector(
                output_dir=os.path.join(cfg.OUTPUT_DIR, "stage6_processed"),
                threshold=0.5,
                min_duration=5.0,
                huggingface_token=cfg.HUGGINGFACE_TOKEN
            )
            
            wav_files = [f for f in os.listdir(sample_extractor.output_dir) if f.endswith('.wav')]
            mp3_files = [f for f in os.listdir(sample_extractor.output_dir) if f.endswith('.mp3')]
            
            if not wav_files and not mp3_files:
                print(f"❌ No audio files (WAV/MP3) found in: {sample_extractor.output_dir}")
                return
                
            print(f"📁 Found {len(wav_files)} WAV files and {len(mp3_files)} MP3 files")
            
            all_results = []
            
            # Process WAV files directly
            for wav_file in wav_files:
                wav_path = os.path.join(sample_extractor.output_dir, wav_file)
                print(f"🎵 Processing WAV: {wav_file}")
                try:
                    results = advanced_processor.process_audio_file(wav_path)
                    all_results.extend(results)
                except Exception as e:
                    print(f"⚠️ Error processing {wav_file}: {e}")
                    continue
            
            # Process MP3 files (they'll be converted to WAV internally)
            for mp3_file in mp3_files:
                mp3_path = os.path.join(sample_extractor.output_dir, mp3_file)
                print(f"🎵 Processing MP3: {mp3_file}")
                try:
                    results = advanced_processor.process_audio_file(mp3_path)
                    all_results.extend(results)
                except Exception as e:
                    print(f"⚠️ Error processing {mp3_file}: {e}")
                    continue
            
            processed_results = all_results
            
            if processed_results:
                print(f"✅ Stage 6 completed!")
                print(f"🤖 Successfully processed: {len(processed_results)} audio files")
                print(f"📁 Output directory: {advanced_processor.output_dir}")
                
                # Save enhanced results with transcriptions
                results_file = os.path.join(cfg.OUTPUT_DIR, f"6_voice_detection_results_{snapshot_id}.csv")
                enhanced_results = []
                
                for result in processed_results:
                    enhanced_results.append({
                        'input_file': result.get('clean_chunk_file', ''),
                        'output_file': result.get('clean_chunk_file', ''),
                        'transcription': result.get('transcription', ''),
                        'voice_percentage': result.get('voice_percentage', 0),
                        'overlap_percentage': result.get('overlap_percentage', 0),
                        'speakers_detected': result.get('speakers_detected', 1),
                        'word_count': result.get('word_count', 0),
                        'char_count': result.get('char_count', 0),
                        'avg_confidence': result.get('avg_confidence', 0),
                        'processing_method': result.get('processing_method', 'whisper'),
                        'chunk_number': result.get('chunk_number', 1),
                        'total_chunks': result.get('total_chunks', 1),
                        'processed_username': result.get('processed_username', 'unknown'),
                        'platform_source': result.get('platform_source', 'unknown'),
                        'input_format': 'Audio',
                        'output_format': 'WAV + Transcription',
                        'model_used': 'OpenAI-Whisper-Base'
                    })
                
                pd.DataFrame(enhanced_results).to_csv(results_file, index=False)
                print(f"📊 Enhanced processing results saved: {results_file}")
                
                print(f"\n🎤 Sample Voice Detection Results:")
                for i, result in enumerate(processed_results[:3], 1):
                    input_file = os.path.basename(result.get('clean_chunk_file', 'unknown'))
                    transcription = result.get('transcription', '')[:80] + "..." if result.get('transcription') else 'N/A'
                    voice_pct = result.get('voice_percentage', 0)
                    word_count = result.get('word_count', 0)
                    confidence = result.get('avg_confidence', 0)
                    print(f"  {i}. {input_file}")
                    print(f"     Voice Quality: {voice_pct:.1f}% | Words: {word_count} | Confidence: {confidence:.3f}")
                    print(f"     Transcription: {transcription}")
                    print(f"     Model: OpenAI-Whisper-Base | Format: WAV + Transcript")
                
                print(f"\n🔄 Complete Audio Processing Pipeline Summary:")
                print(f"  📥 Stage 5 Output: MP3 files ({len(extracted_samples)} samples)")
                print(f"  🤖 Stage 6 Processing: Advanced Voice Detection")
                print(f"  📤 Stage 6 Output: Enhanced WAV + transcripts ({len(processed_results)} files)")
                print(f"  🎯 AI Model: OpenAI Whisper throughout pipeline")
            else:
                print("❌ Stage 6 Voice Detection failed - no results returned")
                processed_results = []
                
        except Exception as e:
            print(f"❌ Stage 6 Voice Detection failed: {e}")
            print(f"💡 Check that WAV files exist in: {sample_extractor.output_dir}")
            processed_results = []
    else:
        print("\n⏭️ Skipping Stage 6 - no clean WAV chunks available")
        processed_results = []

    # Final comprehensive summary
    print("\n🎉 WHISPER ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total accounts processed: {len(valid_accounts)}")
    print(f"📥 Profiles downloaded: {len(profiles)}")
    print(f"🔗 External links found: {len(links)}")
    print(f"🎯 YouTube/Twitch/TikTok links: {len(audio_links)}")
    print(f"🔊 Audio content confirmed: {len(audio_detected_links)}")
    print(f"🎙️ Voice content confirmed: {len(confirmed_voice)}")
    print(f"🎤 Voice samples extracted (MP3): {len(extracted_samples)}")
    print(f"🤖 Whisper processed + transcripts: {len(processed_results) if 'processed_results' in locals() else 0}")
    
    voice_confirmation_rate = (len(confirmed_voice) / len(audio_links) * 100) if audio_links else 0
    print(f"📈 Voice confirmation rate: {voice_confirmation_rate:.1f}%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")
    print(f"🔄 Pipeline order: 1 → 2 → 3 → 4 → 5 → 6")
    print(f"🤖 AI Enhancement: OpenAI Whisper integration in stages 6")
    print(f"🎵 Audio format flow: MP3 (Stage 5) → WAV + Whisper (Stage 6)")

    # Final output files summary
    print(f"\n📄 Output Files Generated:")
    print(f"  1. {existing_accounts_file} - Validated accounts")
    print(f"  2. {profiles_file} - Profile data")
    print(f"  3. {links_file} - External links")
    print(f"  4. {audio_file} - YouTube/Twitch links")
    print(f"  5. {audio_detected_file} - Audio content detected")
    if confirmed_voice:
        print(f"  6. ⭐ CONFIRMED VOICE CONTENT: {len(confirmed_voice)} links")
    if extracted_samples:
        print(f"  7. {extraction_file} - 🎤 VOICE SAMPLE EXTRACTION RESULTS")
        print(f"  8. {sample_extractor.output_dir} - 🎵 VOICE SAMPLES DIRECTORY (MP3)")
    if 'processed_results' in locals() and processed_results:
        print(f"  9. {results_file} - 📝 WHISPER ENHANCED RESULTS + TRANSCRIPTS")
        print(f"  10. {advanced_processor.output_dir} - 🎤 FINAL WHISPER PROCESSED FILES")


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
    
    if not os.path.exists(accounts_file):
        print(f"❌ Accounts file not found: {accounts_file}")
        return None
    
    df = pd.read_csv(accounts_file)
    valid_accounts = df.to_dict('records')
    usernames = [acc['username'] for acc in valid_accounts]
    print(f"📥 Loaded {len(usernames)} accounts from: {accounts_file}")
    
    sm = SnapshotManager(cfg.OUTPUT_DIR)
    existing_snapshot = sm.get_reusable_snapshot(usernames)
    
    if existing_snapshot:
        print(f"🔄 Using existing snapshot: {existing_snapshot}")
        snapshot_id = existing_snapshot
    else:
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
    
    profiles_file = os.path.join(cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
    pd.DataFrame(profiles).to_csv(profiles_file, index=False)
    print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
    
    links = downloader.extract_external_links(profiles)
    if not links:
        print("🔗 No external links found")
        return None
    
    links_file = os.path.join(cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
    pd.DataFrame(links).to_csv(links_file, index=False)
    print(f"✅ Stage 3 completed!")
    print(f"🔗 External links found: {len(links)}")
    print(f"📁 Links file: {links_file}")
    print(f"💡 Next: Run Stage 3.5 with --stage3_5-only {links_file}")
    print(f"💡 Alternative: Skip Stage 3.5 and run Stage 4 with --stage4-only {links_file}")
    return links_file


def run_stage3_5_only(links_file):
    """Run only Stage 3.5: YouTube & Twitch Channel Discovery"""
    print("🔍 STAGE 3.5 ONLY: YouTube & Twitch Channel Discovery")
    print("=" * 50)
    
    if not os.path.exists(links_file):
        print(f"❌ Links file not found: {links_file}")
        return None

    df = pd.read_csv(links_file)
    links = df.to_dict('records')
    print(f"📥 Loaded {len(links)} links from: {links_file}")
    
    runner = Step3_5_YouTubeTwitchRunner(cfg.OUTPUT_DIR)
    enhanced_file = runner.run_scraper_for_snapshot(links)

    if enhanced_file:
        print(f"✅ Stage 3.5 completed: {enhanced_file}")
        print(f"📁 Enhanced links file: {enhanced_file}")
        print(f"💡 Next: Run Stage 4 with --stage4-only {enhanced_file}")
    else:
        print("⚠️ Stage 3.5 failed, using original external links")
        print(f"💡 Next: Run Stage 4 with --stage4-only {links_file}")
    
    return enhanced_file


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
        print("🔍 No YouTube, Twitch or TikTok links found")
        return None
    
    base_name = os.path.splitext(os.path.basename(links_file))[0]
    audio_file = os.path.join("output", f"4_{base_name.replace('3_', '')}_audio_links.csv")
    pd.DataFrame(audio_links).to_csv(audio_file, index=False)
    
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
    
    print(f"✅ Stage 4 completed!")
    print(f"🎯 YouTube/Twitch/TikTok links: {len(audio_links)}")
    print("📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count}")
    print(f"📁 Audio file: {audio_file}")
    print(f"💡 Next: Run Stage 5 with --stage5-only {audio_file}")
    return audio_file


def run_stage5_only(confirmed_voice_file, output_dir="output"):
    """Run only Stage 5: Voice Sample Extraction (MP3 Output) - Direct import, no subprocess"""
    print("🎤 STAGE 5 ONLY: Voice Sample Extraction (MP3 Output)")
    print("=" * 50)
    
    if not os.path.exists(confirmed_voice_file):
        print(f"❌ Confirmed voice file not found: {confirmed_voice_file}")
        return
    
    try:
        df = pd.read_csv(confirmed_voice_file)
        confirmed_voice = df.to_dict('records')
        print(f"📥 Loaded {len(confirmed_voice)} confirmed voice links from: {confirmed_voice_file}")
        
        if not confirmed_voice:
            print("❌ No confirmed voice links found in file")
            return
        
        voice_samples_dir = os.path.join(output_dir, "voice_samples")
        sample_extractor = AudioDownloader(voice_samples_dir)
        print("🚀 Starting direct audio download process...")
        
        extracted_samples = sample_extractor.download_audio_for_all(confirmed_voice)
        if extracted_samples:
            result_csv = os.path.join(output_dir, "6_voice_samples_results.csv")
            save_results(extracted_samples, result_csv)
            
            print("✅ Stage 5 completed successfully!")
            print(f"🎤 Successfully processed {len(extracted_samples)} voice samples")
            print(f"📁 Voice samples directory: {voice_samples_dir}")
            print(f"📄 Results CSV: {result_csv}")
            print(f"💡 Next: Run Stage 6 with --stage6-only {voice_samples_dir}")
        else:
            print("❌ No voice samples could be extracted")
            print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
    
    except Exception as e:
        print(f"❌ Error running Stage 5: {e}")


def run_stage6_only(audio_dir):
    """Run only Stage 6: Advanced Voice Detection (Enhanced voice analysis)"""
    print("🎤 STAGE 6 ONLY: Advanced Voice Detection (Enhanced Analysis)")
    print("-" * 60)
    print("🎤 Processing audio files with advanced voice detection")
    print("📝 Using Whisper, Pyannote VAD, and SpeechBrain for comprehensive analysis")
    
    try:
        cfg = Config()
        advanced_processor = AdvancedVoiceDetector(
            output_dir=os.path.join(audio_dir, "stage6_processed"),
            threshold=0.5,
            min_duration=5.0,
            huggingface_token=cfg.HUGGINGFACE_TOKEN
        )
        
        wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        mp3_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
        
        if not wav_files and not mp3_files:
            print(f"❌ No audio files (WAV/MP3) found in: {audio_dir}")
            return
            
        print(f"📁 Found {len(wav_files)} WAV files and {len(mp3_files)} MP3 files")
        
        all_results = []
        
        # Process WAV files directly
        for wav_file in wav_files:
            wav_path = os.path.join(audio_dir, wav_file)
            print(f"🎵 Processing WAV: {wav_file}")
            try:
                results = advanced_processor.process_audio_file(wav_path)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error processing {wav_file}: {e}")
                continue
        
        # Process MP3 files (they'll be converted to WAV internally)
        for mp3_file in mp3_files:
            mp3_path = os.path.join(audio_dir, mp3_file)
            print(f"🎵 Processing MP3: {mp3_file}")
            try:
                results = advanced_processor.process_audio_file(mp3_path)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error processing {mp3_file}: {e}")
                continue
        
        processed_results = all_results
        
        if processed_results:
            print(f"✅ Stage 6 completed!")
            print(f"🤖 Successfully processed: {len(processed_results)} audio files")
            print(f"📁 Output directory: {advanced_processor.output_dir}")
            
            # Save enhanced results with transcriptions
            results_file = os.path.join(audio_dir, f"6_voice_detection_results.csv")
            enhanced_results = []
            
            for result in processed_results:
                enhanced_results.append({
                    'input_file': result.get('clean_chunk_file', ''),
                    'output_file': result.get('clean_chunk_file', ''),
                    'transcription': result.get('transcription', ''),
                    'voice_percentage': result.get('voice_percentage', 0),
                    'overlap_percentage': result.get('overlap_percentage', 0),
                    'speakers_detected': result.get('speakers_detected', 1),
                    'word_count': result.get('word_count', 0),
                    'char_count': result.get('char_count', 0),
                    'avg_confidence': result.get('avg_confidence', 0),
                    'processing_method': result.get('processing_method', 'whisper'),
                    'chunk_number': result.get('chunk_number', 1),
                    'total_chunks': result.get('total_chunks', 1),
                    'processed_username': result.get('processed_username', 'unknown'),
                    'platform_source': result.get('platform_source', 'unknown'),
                    'input_format': 'Audio',
                    'output_format': 'WAV + Transcription',
                    'model_used': 'OpenAI-Whisper-Base'
                })
            
            pd.DataFrame(enhanced_results).to_csv(results_file, index=False)
            print(f"📊 Enhanced processing results saved: {results_file}")
            
            print(f"\n🎤 Sample Voice Detection Results:")
            for i, result in enumerate(processed_results[:3], 1):
                input_file = os.path.basename(result.get('clean_chunk_file', 'unknown'))
                transcription = result.get('transcription', '')[:80] + "..." if result.get('transcription') else 'N/A'
                voice_pct = result.get('voice_percentage', 0)
                word_count = result.get('word_count', 0)
                confidence = result.get('avg_confidence', 0)
                print(f"  {i}. {input_file}")
                print(f"     Voice Quality: {voice_pct:.1f}% | Words: {word_count} | Confidence: {confidence:.3f}")
                print(f"     Transcription: {transcription}")
                print(f"     Model: OpenAI-Whisper-Base | Format: WAV + Transcript")
            
            print(f"\n🔄 Stage 6 Processing Summary:")
            print(f"  📥 Input: {len(wav_files + mp3_files)} audio files")
            print(f"  🤖 Processing: Advanced Voice Detection")
            print(f"  📤 Output: Enhanced WAV + transcripts ({len(processed_results)} files)")
            print(f"  🎯 AI Model: OpenAI Whisper + Pyannote + SpeechBrain")
        else:
            print("❌ Stage 6 Voice Detection failed - no results returned")
            
    except Exception as e:
        print(f"❌ Stage 6 Voice Detection failed: {e}")
        print(f"💡 Check that audio files exist in: {audio_dir}")


def show_help():
    help_text = """
🤖 WHISPER ENHANCED YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE

PIPELINE FLOW:
1→2→3→4→5→6

INDIVIDUAL STAGES:
--stage1-only FILE     Stage 1: Account Validation
--stage2-only FILE     Stage 2: Bright Data Trigger
--stage3-only SNAPSHOT Stage 3: Data Download
--stage3_5-only FILE   Stage 3.5: YouTube/Twitch Channel Discovery
--stage4-only FILE     Stage 4: YouTube/Twitch Filter
--stage5-only FILE     Stage 5: Voice Sample Extraction (MP3 Output)
--stage6-only DIR      Stage 6: Advanced Voice Detection + Transcription

🤖 WHISPER ENHANCEMENTS:
- Stage 6: Advanced Voice Detection using Whisper, Pyannote VAD, and SpeechBrain
- Automatic MP3 → WAV conversion for Whisper processing
- Enhanced speech quality analysis with confidence scores
- Full transcription generation for all audio chunks

📝 TRANSCRIPTION FEATURES:
- Real-time speech-to-text using OpenAI Whisper
- Voice activity detection using Pyannote
- Speaker verification using SpeechBrain
- Voice quality assessment and confidence scoring
- Word count and character count metrics per chunk

NOTES:
- Whisper model automatically downloads on first use
- Memory usage optimized with configurable worker limits
- All transcriptions saved with processing results
- Compatible with existing pipeline stages 1-6
"""
    print(help_text)


# FIXED MAIN EXECUTION LOGIC
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper Enhanced YouTube/Twitch/TikTok Voice Pipeline with Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", help="Input CSV/TXT file with usernames for full pipeline")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")
    
    # Individual stage arguments
    parser.add_argument("--stage1-only", help="Run only Stage 1 - Account validation")
    parser.add_argument("--stage2-only", help="Run only Stage 2 - Bright Data trigger")
    parser.add_argument("--stage3-only", help="Run only Stage 3 - Data download")
    parser.add_argument("--stage3_5-only", help="Run only Stage 3.5 - YouTube/Twitch channel discovery")
    parser.add_argument("--stage4-only", help="Run only Stage 4 - YouTube/Twitch filter")
    parser.add_argument("--stage5-only", help="Run only Stage 5 - Voice sample extraction (MP3)")
    parser.add_argument("--stage6-only", help="Run only Stage 6 - Advanced Whisper analysis + transcription")
    
    # Information commands
    parser.add_argument("--show-log", action="store_true", help="Show account validation log")
    parser.add_argument("--show-snapshots", action="store_true", help="Show snapshot summary")
    parser.add_argument("--clear-log", action="store_true", help="Clear processed accounts log")
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed help")
    
    # Parse arguments FIRST - THIS WAS THE CRITICAL FIX
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
        run_stage2_only(args.stage2_only)
        sys.exit(0)
    
    if args.stage3_only:
        run_stage3_only(args.stage3_only)
        sys.exit(0)
    
    if args.stage3_5_only:
        if not os.path.exists(args.stage3_5_only):
            print(f"❌ Links file not found: {args.stage3_5_only}")
            sys.exit(1)
        run_stage3_5_only(args.stage3_5_only)
        sys.exit(0)
    
    if args.stage4_only:
        if not os.path.exists(args.stage4_only):
            print(f"❌ Links file not found: {args.stage4_only}")
            sys.exit(1)
        run_stage4_only(args.stage4_only)
        sys.exit(0)
    
    if args.stage5_only:
        if not os.path.exists(args.stage5_only):
            print(f"❌ Audio links file not found: {args.stage5_only}")
            sys.exit(1)
        run_stage5_only(args.stage5_only)
        sys.exit(0)
    
    if args.stage6_only:
        if not os.path.exists(args.stage6_only):
            print(f"❌ Audio directory not found: {args.stage6_only}")
            sys.exit(1)
        run_stage6_only(args.stage6_only)
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
    
    if args.clear_log:
        try:
            validator = AccountValidator()
            validator.clear_log()
            print("✅ Account validation log cleared")
        except Exception as e:
            print(f"❌ Error clearing log: {e}")
        sys.exit(0)
    
    # Run main pipeline
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        
        try:
            print(f"🚀 Starting Whisper enhanced 6-stage pipeline")
            print(f"🤖 AI Model: OpenAI Whisper for speech processing and transcription")
            print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
            print(f"🎵 Audio flow: Stage 5 (MP3) → Stage 6 (WAV+Transcripts)")
            main(args.input, args.force_recheck)
        except KeyboardInterrupt:
            print("\n⏹️ Pipeline interrupted by user (Ctrl+C)")
            print("💾 All progress has been saved and can be resumed")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            print("💡 Check your configuration and try individual stages for debugging")
            sys.exit(1)
    else:
        # No arguments provided - show help
        print("❌ No action specified.")
        print("💡 Use --input FILE for full pipeline or --help-detailed for usage guide")
        print("\n🎯 Quick start examples:")
        print("  python main_pipeline.py --input usernames.csv")
        print("  python main_pipeline.py --stage6-only output/voice_samples/")
        print("\n🔄 Pipeline: 1→2→3→4→5→6")
        print("🤖 Enhanced: OpenAI Whisper speech processing and transcription in stages 6")
        print("🎵 Audio format flow: MP3 (Stage 5) → WAV + Whisper (Stage 6)")
        sys.exit(1)
