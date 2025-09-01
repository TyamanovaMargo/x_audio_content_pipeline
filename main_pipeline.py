import os
import pandas as pd
import argparse
import sys

from config import Config
from step1_validate_accounts import AccountValidator
from step2_bright_data_trigger import BrightDataTrigger
from step3_bright_data_download import BrightDataDownloader
from step4_audio_filter import AudioContentFilter
from step5_audio_detector import AudioContentDetector
from step6_voice_sample_extractor import VoiceSampleExtractor
from step6_5_overlap_detector import OverlapDetector
from step7_diarization_processor import Step7DiarizationProcessor
from snapshot_manager import SnapshotManager

def main(input_file, force_recheck=False):
    """Main pipeline execution - Pipeline with MP3 to WAV conversion handling"""
    
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("🎙️ YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE")
    print("=" * 60)
    print("🎯 Focus: YouTube, Twitch, and TikTok voice content extraction")
    print("🎤 Pipeline: MP3 → WAV conversion → Overlap Detection → Diarization Processing")
    print("🔍 Stages: 7 comprehensive processing stages (1→2→3→4→4.5→5→6→6.5→7)")
    print("🔄 Audio Flow: Stage 6 (MP3) → Stage 6.5 (MP3→WAV) → Stage 7 (WAV)")

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
    print(f"📁 Saved to: {audio_file}")
    
    # Show platform breakdown
    platform_counts = {}
    for link in audio_links:
        platform = link.get('platform_type', 'unknown')
        platform_counts[platform] = platform_counts.get(platform, 0) + 1
    
    print("\n📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f" {platform}: {count}")

    # Stage 5: YouTube, Twitch & TikTok Audio Content Detection
    print("\n🎵 STAGE 5: YouTube, Twitch & TikTok Audio Content Detection")
    print("-" * 60)
    
    # Initialize enhanced detector WITHOUT Twitch API
    audio_detector = AudioContentDetector(
        timeout=15,
        huggingface_token=getattr(cfg, 'HUGGINGFACE_TOKEN', None)  # Only HuggingFace token needed
    )
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
    print(" Audio Types:")
    for audio_type, count in sorted(audio_types.items(), key=lambda x: x[1], reverse=True):
        print(f" {audio_type}: {count}")
    print(" Confidence Levels:")
    for confidence, count in confidence_levels.items():
        print(f" {confidence}: {count}")


    # Stage 6: Voice Sample Extraction (Outputs MP3 files)
    print("\n🎤 STAGE 6: Voice Sample Extraction (MP3 Output)")
    print("-" * 60)
    
    if confirmed_voice:
        sample_extractor = VoiceSampleExtractor(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "voice_samples"),
            max_duration_hours=1,  # Maximum 1 hour
            quality="192"  # MP3 quality
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
            print(f" 📊 Total voice links: {len(confirmed_voice)}")
            print(f" ✅ Successful extractions: {len(extracted_samples)}")
            print(f" 📁 Samples directory: {sample_extractor.output_dir}")
            print(f" 📄 Report file: {report_file}")
            print(f" ⏱️ Sample duration: up to 1 hour each")
            print(f" 🎵 Output format: MP3 (192kbps)")
            
            # Show extracted files
            print(f"\n🎵 Extracted MP3 Sample Files:")
            for sample in extracted_samples:
                filename = sample.get('sample_filename', 'N/A')
                username = sample.get('processed_username', 'unknown')
                platform = sample.get('platform_source', 'unknown')
                file_size = sample.get('file_size_bytes', 0)
                file_format = "MP3" if filename.lower().endswith('.mp3') else "Unknown"
                print(f" 📄 {filename} (@{username} {platform}, {file_size//1000}KB, {file_format})")
        else:
            print("❌ No voice samples could be extracted")
            print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
            extracted_samples = []
    else:
        print("⏭️ Skipping voice sample extraction - no confirmed voice content")
        extracted_samples = []

    # Stage 6.5: Audio Chunking and Overlap Detection (MP3 → WAV conversion)
    if extracted_samples:
        print("\n🔍 STAGE 6.5: Audio Chunking and Overlap Detection (MP3 → WAV Processing)")
        print("-" * 60)
        print("🔄 Converting MP3 files to WAV for overlap detection and chunking")
        
        overlap_detector = OverlapDetector(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "clean_chunks"),
            chunk_duration_minutes=5,  # 5-minute chunks
            overlap_threshold=0.3,     # 30% overlap = remove chunk
            huggingface_token=getattr(cfg, 'HUGGINGFACE_TOKEN', None)
        )
        
        clean_chunks = overlap_detector.process_extracted_samples(extracted_samples)
        
        if clean_chunks:
            # Save clean chunks results
            clean_chunks_file = os.path.join(cfg.OUTPUT_DIR, f"6_5_snapshot_{snapshot_id}_clean_chunks.csv")
            pd.DataFrame(clean_chunks).to_csv(clean_chunks_file, index=False)
            
            # Generate report
            report_file = overlap_detector.generate_report(clean_chunks)
            
            print(f"✅ Stage 6.5 completed!")
            print(f"🔍 Clean chunks created: {len(clean_chunks)}")
            print(f"📁 Clean chunks directory: {overlap_detector.output_dir}")
            print(f"📄 Report: {report_file}")
            print(f"📊 Results CSV: {clean_chunks_file}")
            print(f"🎵 Output format: WAV (16kHz mono)")
            
            # Statistics
            original_samples_count = len(extracted_samples)
            clean_chunks_count = len(clean_chunks)
            removed_count = original_samples_count - clean_chunks_count
            
            print(f"\n🎯 MP3 → WAV Conversion & Overlap Detection Summary:")
            print(f"📊 Original MP3 samples: {original_samples_count}")
            print(f"✅ Clean WAV chunks kept: {clean_chunks_count}")
            print(f"❌ Overlapping chunks removed: {removed_count}")
            print(f"📈 Clean chunk rate: {(clean_chunks_count / original_samples_count * 100):.1f}%" if original_samples_count > 0 else "0%")
            print(f"🔄 Format conversion: MP3 → WAV (16kHz mono)")
            
            # Show sample clean chunks
            print(f"\n🎵 Sample Clean WAV Chunks:")
            for i, chunk in enumerate(clean_chunks[:3], 1):
                clean_file = chunk.get('clean_chunk_file', 'N/A')
                username = chunk.get('processed_username', 'unknown')
                platform = chunk.get('platform_source', 'unknown')
                chunk_num = chunk.get('chunk_number', 1)
                total_chunks = chunk.get('total_chunks', 1)
                overlap_pct = chunk.get('overlap_percentage', 0)
                
                print(f" {i}. {os.path.basename(clean_file)} (@{username} {platform})")
                print(f"    Chunk: {chunk_num}/{total_chunks} | Overlap: {overlap_pct:.1f}% | Format: WAV")
        else:
            print("❌ No clean chunks found - all audio had overlapping voices")
            clean_chunks = []
    else:
        print("⏭️ Skipping Stage 6.5 - no MP3 audio samples extracted")
        clean_chunks = []

    # Stage 7: Diarization Processing (Processes WAV files)
    if clean_chunks and 'overlap_detector' in locals() and overlap_detector.output_dir:
        print("\n🎤 STAGE 7: Diarization Processing (WAV Input)")
        print("-" * 60)
        print("🔄 Processing clean WAV chunks with advanced diarization-first approach")
        
        try:
            # Initialize Step 7 Diarization processor
            processor = Step7DiarizationProcessor(config_path="config.json")
            
            # Process the clean chunks directory (contains WAV files)
            clean_audio_dir = overlap_detector.output_dir
            
            # Count WAV files in directory
            wav_files = [f for f in os.listdir(clean_audio_dir) if f.endswith('.wav')]
            print(f"📁 Processing {len(wav_files)} WAV files from: {clean_audio_dir}")
            
            processed_results = processor.process_folder(clean_audio_dir)
            
            if processed_results:
                print(f"✅ Stage 7 diarization processing completed!")
                print(f"🎤 Successfully processed: {len(processed_results)} WAV files")
                print(f"📁 Final output directory: {processor.config.output_dir}")
                
                # Save processing results
                stage7_results_file = os.path.join(cfg.OUTPUT_DIR, f"7_diarization_results_{snapshot_id}.csv")
                
                # Create simplified results for CSV
                simplified_results = []
                for result in processed_results:
                    simplified_results.append({
                        'input_file': result.get('input_file', ''),
                        'output_file': result.get('output_file', ''),
                        'primary_speaker': result.get('primary_speaker', ''),
                        'segments_count': result.get('segments_count', 0),
                        'voice_duration': result.get('voice_duration', 0),
                        'processing_method': result.get('processing_method', ''),
                        'processing_status': result.get('processing_status', ''),
                        'input_format': 'WAV',
                        'output_format': result.get('output_format', 'WAV'),
                        'is_chunk': result.get('is_chunk', False),
                        'original_mp3_source': result.get('original_mp3_source', '')
                    })
                
                pd.DataFrame(simplified_results).to_csv(stage7_results_file, index=False)
                print(f"📊 Processing results saved: {stage7_results_file}")
                
                # Show sample results
                print(f"\n🎤 Sample Diarization Results:")
                for i, result in enumerate(processed_results[:3], 1):
                    input_file = os.path.basename(result.get('input_file', 'unknown'))
                    primary_speaker = result.get('primary_speaker', 'unknown')
                    voice_duration = result.get('voice_duration', 0)
                    segments = result.get('segments_count', 0)
                    
                    print(f" {i}. {input_file}")
                    print(f"    Primary speaker: {primary_speaker}")
                    print(f"    Voice duration: {voice_duration:.1f}s ({segments} segments)")
                    print(f"    Input: WAV → Output: WAV (processed)")
                
                # Audio format summary
                print(f"\n🔄 Audio Format Pipeline Summary:")
                print(f"📥 Stage 6 Output: MP3 files ({len(extracted_samples)} samples)")
                print(f"🔄 Stage 6.5 Processing: MP3 → WAV conversion + overlap detection")
                print(f"📤 Stage 6.5 Output: WAV files ({len(clean_chunks)} clean chunks)")
                print(f"🎤 Stage 7 Processing: WAV → processed WAV with diarization")
                print(f"📤 Stage 7 Output: WAV files ({len(processed_results)} processed)")
                
            else:
                print("❌ Stage 7 diarization processing failed - no results returned")
                processed_results = []
                
        except Exception as e:
            print(f"❌ Stage 7 diarization processing failed: {e}")
            print(f"💡 Check that WAV files exist in: {clean_audio_dir}")
            processed_results = []
    else:
        print("\n⏭️ Skipping Stage 7 - no clean WAV chunks available")
        processed_results = []

    # Final comprehensive summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total accounts processed: {len(valid_accounts)}")
    print(f"📥 Profiles downloaded: {len(profiles)}")
    print(f"🔗 External links found: {len(links)}")
    print(f"🎯 YouTube/Twitch/TikTok links: {len(audio_links)}")
    print(f"🔊 Audio content confirmed: {len(audio_detected_links)}")
    print(f"🎙️ Voice content confirmed: {len(confirmed_voice)}")
    print(f"🎤 Voice samples extracted (MP3): {len(extracted_samples) if 'extracted_samples' in locals() else 0}")
    print(f"🔍 Clean chunks (WAV, no overlaps): {len(clean_chunks) if 'clean_chunks' in locals() else 0}")
    print(f"🎤 Diarization processed (WAV): {len(processed_results) if 'processed_results' in locals() else 0}")
    
    print(f"📈 Voice confirmation rate: {(len(confirmed_voice) / len(audio_links) * 100):.1f}%" if audio_links else "0%")
    print(f"📈 Clean chunk rate: {(len(clean_chunks) / len(extracted_samples) * 100):.1f}%" if 'extracted_samples' in locals() and extracted_samples and 'clean_chunks' in locals() else "0%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")
    print(f"🔄 Pipeline order: 1 → 2 → 3 → 4 → 5 → 5 → 6 → 6.5 → 7")
    print(f"🎵 Audio format flow: MP3 (Stage 6) → WAV (Stage 6.5) → Processed WAV (Stage 7)")

    # Final output files summary
    print(f"\n📄 Output Files Generated:")
    print(f" 1. {existing_accounts_file} - Validated accounts")
    print(f" 2. {profiles_file} - Profile data")
    print(f" 3. {links_file} - External links")
    print(f" 4. {audio_file} - YouTube/Twitch links")
    print(f" 5. {audio_detected_file} - Audio content detected")
    print(f" 6. {verified_file} - Voice verification results")
    
    if confirmed_voice:
        print(f" 7. {confirmed_file} - ⭐ CONFIRMED VOICE CONTENT")
    
    if 'extracted_samples' in locals() and extracted_samples:
        print(f" 8. {extraction_file} - 🎤 VOICE SAMPLE EXTRACTION RESULTS")
        print(f" 9. {sample_extractor.output_dir} - 🎵 VOICE SAMPLES DIRECTORY (MP3)")
    
    if 'clean_chunks' in locals() and clean_chunks:
        print(f" 10. {clean_chunks_file} - 🔍 CLEAN CHUNKS METADATA")
        print(f" 11. {overlap_detector.output_dir} - 🎵 CLEAN CHUNK AUDIO FILES (WAV)")
    
    if 'processed_results' in locals() and processed_results:
        print(f" 12. {stage7_results_file} - 🎤 STAGE 7 DIARIZATION RESULTS")
        print(f" 13. {processor.config.output_dir} - 🎤 FINAL PROCESSED AUDIO FILES (WAV)")

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
        print("🔍 No YouTube, Twitch or TikTok links found")
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
    print(f"🎯 YouTube/Twitch/TikTok links: {len(audio_links)}")
    print("📊 Platform breakdown:")
    for platform, count in platform_counts.items():
        print(f" {platform}: {count}")
    print(f"📁 Audio file: {audio_file}")
    print(f"💡 Next: Run Stage 5 with --stage4_5-only {audio_file}")
    
    return audio_file

def run_stage4_5_only(audio_links_file):
    """Run only Stage 5: Enhanced Audio Content Detection (No API)"""
    print("🎵 STAGE 5 ONLY: Enhanced YouTube, Twitch & TikTok Audio Content Detection")
    print("=" * 50)
    
    if not os.path.exists(audio_links_file):
        print(f"❌ Audio links file not found: {audio_links_file}")
        return None

    df = pd.read_csv(audio_links_file)
    audio_links = df.to_dict('records')
    print(f"📥 Loaded {len(audio_links)} audio links from: {audio_links_file}")

    cfg = Config()
    
    # Initialize enhanced detector WITHOUT Twitch API
    audio_detector = AudioContentDetector(
        timeout=15,
        huggingface_token=getattr(cfg, 'HUGGINGFACE_TOKEN', None)
    )
    
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
    
    print(f"✅ Stage 5 completed!")
    print(f"🔊 Audio content detected: {len(audio_detected_links)}")
    print("📊 Audio Content Breakdown:")
    print(" Audio Types:")
    for audio_type, count in sorted(audio_types.items(), key=lambda x: x[1], reverse=True):
        print(f" {audio_type}: {count}")
    print(" Confidence Levels:")
    for confidence, count in confidence_levels.items():
        print(f" {confidence}: {count}")
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
        
        print(f"💡 Next: Run Stage 6 with --stage6-only {confirmed_file}")
    else:
        print("❌ No voice content confirmed")
    
    print(f"\n✅ Stage 5 completed!")
    print(f"📊 Total links processed: {len(audio_links)}")
    print(f"🎙️ Voice content found: {len(confirmed_voice)}")

def run_stage6_only(confirmed_voice_file, output_dir="output"):
    """Run only Stage 6: Voice Sample Extraction (MP3 Output)"""
    print("🎤 STAGE 6 ONLY: Voice Sample Extraction (MP3 Output)")
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
        quality="192",  # MP3 quality
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
        print(f"🎤 Successfully extracted {len(extracted_samples)} voice samples")
        print(f"📁 Results: {extraction_file}")
        print(f"📄 Report: {report_file}")
        print(f"🎵 Samples directory: {sample_extractor.output_dir}")
        print(f"🎵 Output format: MP3 (192kbps)")
        print(f"💡 Next: Run Stage 6.5 with --stage6_5-only {extraction_file}")
        
        # Show extracted MP3 files
        print(f"\n🎵 Extracted MP3 Files:")
        for sample in extracted_samples:
            filename = sample.get('sample_filename', 'N/A')
            username = sample.get('processed_username', 'unknown')
            platform = sample.get('platform_source', 'unknown')
            file_size = sample.get('file_size_bytes', 0)
            print(f" 📄 {filename} (@{username} {platform}, {file_size//1000}KB, MP3)")
        
        # Clean temporary files
        sample_extractor.clean_temp_files()
    else:
        print("❌ No voice samples could be extracted")
        print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")

def run_stage6_5_only(extracted_samples_file, output_dir="output"):
    """Run only Stage 6.5: Audio Chunking and Overlap Detection (MP3 → WAV)"""
    print("🔍 STAGE 6.5 ONLY: Audio Chunking and Overlap Detection (MP3 → WAV)")
    print("=" * 50)
    
    if not os.path.exists(extracted_samples_file):
        print(f"❌ Extracted samples file not found: {extracted_samples_file}")
        return
    
    try:
        df = pd.read_csv(extracted_samples_file)
        extracted_samples = df.to_dict('records')
        print(f"📥 Loaded {len(extracted_samples)} extracted samples from: {extracted_samples_file}")
    except Exception as e:
        print(f"❌ Error loading extracted samples: {e}")
        return
    
    if not extracted_samples:
        print("❌ No extracted samples found in file")
        return
    
    # Check if samples are MP3
    mp3_count = sum(1 for sample in extracted_samples 
                   if sample.get('sample_filename', '').lower().endswith('.mp3'))
    print(f"🎵 Found {mp3_count} MP3 files to process")
    
    cfg = Config()
    
    overlap_detector = OverlapDetector(
        output_dir=os.path.join(output_dir, "clean_chunks"),
        chunk_duration_minutes=5,
        overlap_threshold=0.3,
        huggingface_token=getattr(cfg, 'HUGGINGFACE_TOKEN', None)
    )
    
    print("🔄 Converting MP3 to WAV and detecting overlaps...")
    clean_chunks = overlap_detector.process_extracted_samples(extracted_samples)
    
    if clean_chunks:
        # Save results
        base_name = os.path.splitext(os.path.basename(extracted_samples_file))[0]
        clean_chunks_file = os.path.join(output_dir, f"6_5_{base_name}_clean_chunks.csv")
        pd.DataFrame(clean_chunks).to_csv(clean_chunks_file, index=False)
        
        # Generate report
        report_file = overlap_detector.generate_report(clean_chunks)
        
        print(f"✅ Stage 6.5 completed!")
        print(f"🔍 Clean chunks created: {len(clean_chunks)}")
        print(f"📁 Clean chunks directory: {overlap_detector.output_dir}")
        print(f"📄 Report: {report_file}")
        print(f"📊 Results CSV: {clean_chunks_file}")
        print(f"🔄 Format conversion: MP3 → WAV (16kHz mono)")
        print(f"💡 Next: Run Stage 7 with --stage7-only {overlap_detector.output_dir}")
        
        # Show sample WAV outputs
        print(f"\n🎵 Sample Clean WAV Chunks:")
        for i, chunk in enumerate(clean_chunks[:3], 1):
            clean_file = chunk.get('clean_chunk_file', 'N/A')
            username = chunk.get('processed_username', 'unknown')
            platform = chunk.get('platform_source', 'unknown')
            print(f" {i}. {os.path.basename(clean_file)} (@{username} {platform}, WAV)")
        
    else:
        print("❌ No clean chunks found - all audio had overlapping voices")

def run_stage7_only(clean_audio_dir, output_dir="stage7_output"):
    """Run only Stage 7: Diarization Processing (WAV Input)"""
    print("🎤 STAGE 7 ONLY: Diarization Processing (WAV Input)")
    print("=" * 50)
    
    if not os.path.exists(clean_audio_dir):
        print(f"❌ Clean audio directory not found: {clean_audio_dir}")
        return
    
    # Count WAV and MP3 files
    wav_files = [f for f in os.listdir(clean_audio_dir) if f.endswith('.wav')]
    mp3_files = [f for f in os.listdir(clean_audio_dir) if f.endswith('.mp3')]
    
    if not wav_files and not mp3_files:
        print(f"❌ No audio files (WAV/MP3) found in: {clean_audio_dir}")
        return
    
    print(f"📥 Found {len(wav_files)} WAV files and {len(mp3_files)} MP3 files to process")
    if mp3_files:
        print("🔄 MP3 files will be converted to WAV for processing")
    
    try:
        processor = Step7DiarizationProcessor(config_path="config.json")
        processed_results = processor.process_folder(clean_audio_dir)
        
        if processed_results:
            print(f"✅ Stage 7 diarization processing completed!")
            print(f"🎤 Successfully processed: {len(processed_results)} files")
            print(f"📁 Output directory: {processor.config.output_dir}")
            
            # Show sample results
            print(f"\n🎤 Sample Diarization Results:")
            for i, result in enumerate(processed_results[:3], 1):
                input_file = os.path.basename(result.get('input_file', 'unknown'))
                primary_speaker = result.get('primary_speaker', 'unknown')
                voice_duration = result.get('voice_duration', 0)
                input_format = "WAV" if input_file.lower().endswith('.wav') else "MP3→WAV"
                
                print(f" {i}. {input_file} ({input_format})")
                print(f"    Primary speaker: {primary_speaker} ({voice_duration:.1f}s)")
        else:
            print("❌ No files could be processed successfully")
            
    except Exception as e:
        print(f"❌ Stage 7 diarization processing error: {e}")
def show_help():
    help_text = """
🎙️ YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE

PIPELINE FLOW:
1→2→3→4→5→6→6.5→7  (Step 5 removed)

INDIVIDUAL STAGES:
--stage1-only FILE     Stage 1: Account Validation
--stage2-only FILE     Stage 2: Bright Data Trigger  
--stage3-only SNAPSHOT Stage 3: Data Download
--stage4-only FILE     Stage 4: YouTube/Twitch Filter
--stage5-only FILE     Stage 5: Audio Detection (FINAL VOICE DECISION)
--stage6-only FILE     Stage 6: Voice Sample Extraction
--stage6_5-only FILE   Stage 6.5: Audio Chunking & Overlap Detection
--stage7-only DIR      Stage 7: Diarization Processing

NOTES:
- Step 5 (Voice Verification) removed - using VAD results from step 4.5
- Step 4.5 now makes final decision on voice presence
"""
    print(help_text)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube/Twitch/TikTok Voice Pipeline with MP3→WAV Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", help="Input CSV/TXT file with usernames for full pipeline")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck all accounts")
    
    # Individual stage arguments  
    parser.add_argument("--stage1-only", help="Run only Stage 1 - Account validation")
    parser.add_argument("--stage2-only", help="Run only Stage 2 - Bright Data trigger")
    parser.add_argument("--stage3-only", help="Run only Stage 3 - Data download")
    parser.add_argument("--stage4-only", help="Run only Stage 4 - Platform filtering")
    parser.add_argument("--stage5-only", help="Run only Stage 5 - Audio detection")
    parser.add_argument("--stage6-only", help="Run only Stage 6 - Voice sample extraction (MP3)")
    parser.add_argument("--stage6_5-only", help="Run only Stage 6.5 - Audio chunking & overlap detection (MP3→WAV)")
    parser.add_argument("--stage7-only", help="Run only Stage 7 - Diarization processing (WAV)")
    
    # Information commands
    parser.add_argument("--show-log", action="store_true", help="Show account validation log")
    parser.add_argument("--show-snapshots", action="store_true", help="Show snapshot summary")
    parser.add_argument("--clear-log", action="store_true", help="Clear processed accounts log")
    parser.add_argument("--help-detailed", action="store_true", help="Show detailed help")
    
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
    
    if args.stage4_only:
        if not os.path.exists(args.stage4_only):
            print(f"❌ Links file not found: {args.stage4_only}")
            sys.exit(1)
        run_stage4_only(args.stage4_only)
        sys.exit(0)
    
    if args.stage5_only:
        if not os.path.exists(args.stage4_5_only):
            print(f"❌ Audio links file not found: {args.stage4_5_only}")
            sys.exit(1)
        run_stage4_5_only(args.stage4_5_only)
        sys.exit(0)
    
    
    if args.stage6_only:
        if not os.path.exists(args.stage6_only):
            print(f"❌ Confirmed voice file not found: {args.stage6_only}")
            sys.exit(1)
        run_stage6_only(args.stage6_only, "output")
        sys.exit(0)
    
    if args.stage6_5_only:
        if not os.path.exists(args.stage6_5_only):
            print(f"❌ Extracted samples file not found: {args.stage6_5_only}")
            sys.exit(1)
        run_stage6_5_only(args.stage6_5_only, "output")
        sys.exit(0)
    
    if args.stage7_only:
        if not os.path.exists(args.stage7_only):
            print(f"❌ Clean audio directory not found: {args.stage7_only}")
            sys.exit(1)
        run_stage7_only(args.stage7_only, "stage7_output")
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
            print(f"🚀 Starting 7-stage pipeline with MP3→WAV conversion")
            print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
            print(f"🎵 Audio flow: Stage 6 (MP3) → Stage 6.5 (MP3→WAV) → Stage 7 (WAV)")
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
        print("❌ No action specified.")
        print("💡 Use --input FILE for full pipeline or --help-detailed for usage guide")
        print("\n🎯 Quick start examples:")
        print(" python main_pipeline.py --input usernames.csv")
        print(" python main_pipeline.py --stage6_5-only output/6_voice_samples.csv")
        print(" python main_pipeline.py --stage7-only output/clean_chunks")
        print("\n🔄 Pipeline: 1→2→3→4→4.5→5→6→6.5→7")
        print("🎵 Audio: MP3 (Stage 6) → WAV (Stage 6.5) → Processed WAV (Stage 7)")
        sys.exit(1)
