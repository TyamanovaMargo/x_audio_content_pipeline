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
from step5_audio_detector import EnhancedVoiceDetector as AudioContentDetector
from step6_voice_sample_extractor import AudioDownloader, save_results
from step7_overlap_detector import PyannoteWhisperProcessor # Updated import
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
    print("🔍 Stages: 7 comprehensive processing stages (1→2→3→4→5→6→6.5→7)")
    print("🔄 Audio Flow: Stage 6 (MP3) → Stage 6.5 (MP3→WAV+Whisper) → Stage 7 (WAV+Transcripts)")

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

    # Stage 5: YouTube, Twitch & TikTok Audio Content Detection
    print("\n🎵 STAGE 5: YouTube, Twitch & TikTok Audio Content Detection")
    print("-" * 60)
    audio_detector = AudioContentDetector("config.json")

    
    audio_detected_links = audio_detector.detect_audio_content(audio_links)
    if not audio_detected_links:
        print("🔍 No audio content detected in YouTube/Twitch links")
        print("⚠️ Pipeline completed but no actual audio found")
        return
    
    audio_detected_file = os.path.join(cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_audio_detected.csv")
    pd.DataFrame(audio_detected_links).to_csv(audio_detected_file, index=False)
    print(f"🎵 Found {len(audio_detected_links)} links with actual audio content!")
    print(f"📁 Saved to: {audio_detected_file}")
    
    # Audio content analysis
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
    
    confirmed_voice = audio_detected_links

    # Stage 6: Voice Sample Extraction (Outputs MP3 files)
    print("\n🎤 STAGE 6: Voice Sample Extraction (MP3 Output)")
    print("-" * 60)
    if confirmed_voice:
        sample_extractor = AudioDownloader(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "voice_samples")
        )
        
        extracted_samples = sample_extractor.download_audio_for_all(confirmed_voice)
        if extracted_samples:
            extraction_file = os.path.join(cfg.OUTPUT_DIR, f"6_snapshot_{snapshot_id}_voice_samples.csv")
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

    # Stage 6.5: Whisper Audio Processing and Overlap Detection (MP3 → WAV conversion)
    if extracted_samples:
        print("\n🤖 STAGE 6.5: Whisper Audio Processing and Overlap Detection")
        print("-" * 60)
        print("🔄 Converting MP3 files to WAV for Whisper processing")
        print("🗣️ Using OpenAI Whisper for voice activity detection and transcription")
        print("🔍 Using Whisper for overlapping speech detection")
        
        whisper_overlap_detector = PyannoteWhisperProcessor(
            output_dir=os.path.join(cfg.OUTPUT_DIR, "clean_chunks"),
            chunk_duration_minutes=5,
            huggingface_token=cfg.HUGGINGFACE_TOKEN
        )
        
        # Process the voice samples directory directly
        voice_samples_dir = os.path.join(cfg.OUTPUT_DIR, "voice_samples")
        
        # Get all audio files from the directory
        audio_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            audio_files.extend([f for f in os.listdir(voice_samples_dir) if f.lower().endswith(ext)])
        
        print(f"📁 Found {len(audio_files)} audio files to process")
        
        all_results = []
        for audio_file in audio_files:
            audio_path = os.path.join(voice_samples_dir, audio_file)
            print(f"🎵 Processing: {audio_file}")
            try:
                results = whisper_overlap_detector.process_audio_file(audio_path)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error processing {audio_file}: {e}")
                continue
        
        clean_chunks = all_results
        
        if clean_chunks:
            clean_chunks_file = os.path.join(cfg.OUTPUT_DIR, f"6_5_snapshot_{snapshot_id}_clean_chunks.csv")
            pd.DataFrame(clean_chunks).to_csv(clean_chunks_file, index=False)
            print(f"✅ Stage 6.5 completed!")
            print(f"🤖 Whisper processing successful")
            print(f"🔍 Clean chunks created: {len(clean_chunks)}")
            print(f"📁 Clean chunks directory: {whisper_overlap_detector.output_dir}")
            print(f"📊 Results CSV: {clean_chunks_file}")
            print(f"💡 Next: Run Stage 7 with --stage7-only {whisper_overlap_detector.output_dir}")
            
            # Calculate statistics
            mp3_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.mp3')]
            original_samples_count = len(mp3_files)
            clean_chunks_count = len(clean_chunks)
            removed_count = max(0, original_samples_count - clean_chunks_count)
            
            print(f"\n🎯 Whisper MP3 → WAV Conversion & Processing Summary:")
            print(f"  📊 Original MP3 samples: {original_samples_count}")
            print(f"  ✅ Clean WAV chunks kept: {clean_chunks_count}")
            print(f"  ❌ Overlapping/poor quality chunks removed: {removed_count}")
            clean_chunk_rate = (clean_chunks_count / original_samples_count * 100) if original_samples_count > 0 else 0
            print(f"  📈 Clean chunk rate: {clean_chunk_rate:.1f}%")
            print(f"  🔄 Format conversion: MP3 → WAV (16kHz mono)")
            print(f"  🤖 AI Model: OpenAI Whisper throughout pipeline")
            
            print(f"\n🎵 Sample Clean WAV Chunks with Transcriptions:")
            for i, chunk in enumerate(clean_chunks[:3], 1):
                clean_file = chunk.get('clean_chunk_file', 'N/A')
                username = chunk.get('processed_username', 'unknown')
                platform = chunk.get('platform_source', 'unknown')
                chunk_num = chunk.get('chunk_number', 1)
                total_chunks = chunk.get('total_chunks', 1)
                overlap_pct = chunk.get('overlap_percentage', 0)
                transcription = chunk.get('transcription', '')[:50] + "..." if chunk.get('transcription') else "N/A"
                voice_pct = chunk.get('voice_percentage', 0)
                print(f"  {i}. {os.path.basename(clean_file)} (@{username} {platform})")
                print(f"     Chunk: {chunk_num}/{total_chunks} | Voice: {voice_pct:.1f}% | Overlap: {overlap_pct:.1f}%")
                print(f"     Transcription: {transcription}")
                print(f"     Format: WAV | Model: Whisper")
        else:
            print("❌ No clean chunks found - all audio had overlapping voices or poor quality")
            clean_chunks = []
    else:
        print("⏭️ Skipping Stage 6.5 - no MP3 audio samples extracted")
        clean_chunks = []

    # Stage 7: Advanced Whisper Processing (Enhanced transcription and analysis)
    if clean_chunks and 'whisper_overlap_detector' in locals() and whisper_overlap_detector.output_dir:
        print("\n🎤 STAGE 7: Advanced Whisper Processing (Enhanced Analysis)")
        print("-" * 60)
        print("🤖 Processing clean WAV chunks with advanced Whisper analysis")
        print("📝 Generating detailed transcriptions and voice profiles")
        
        try:
            advanced_processor = PyannoteWhisperProcessor(
                output_dir=os.path.join(cfg.OUTPUT_DIR, "stage7_processed"),
                chunk_duration_minutes=5,
                huggingface_token=cfg.HUGGINGFACE_TOKEN
            )
            
            clean_audio_dir = voice_samples_dir
            
            # Get all WAV files from the directory
            wav_files = []
            if os.path.exists(clean_audio_dir):
                wav_files = [f for f in os.listdir(clean_audio_dir) if f.lower().endswith('.wav')]
            
            print(f"📁 Processing {len(wav_files)} clean WAV files from: {clean_audio_dir}")
            
            all_results = []
            for wav_file in wav_files:
                wav_path = os.path.join(clean_audio_dir, wav_file)
                print(f"🎵 Processing: {wav_file}")
                try:
                    results = advanced_processor.process_audio_file(wav_path)
                    all_results.extend(results)
                except Exception as e:
                    print(f"⚠️ Error processing {wav_file}: {e}")
                    continue
            
            processed_results = all_results
            
            if processed_results:
                print(f"✅ Stage 7 Whisper advanced processing completed!")
                print(f"🤖 Successfully processed: {len(processed_results)} WAV files")
                print(f"📁 Output directory: {advanced_processor.output_dir}")
                
                # Save enhanced results with transcriptions
                results_file = os.path.join(cfg.OUTPUT_DIR, f"7_whisper_results_{snapshot_id}.csv")
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
                        'input_format': 'WAV',
                        'output_format': 'WAV + Transcription',
                        'model_used': 'OpenAI-Whisper-Base'
                    })
                
                pd.DataFrame(enhanced_results).to_csv(results_file, index=False)
                print(f"📊 Enhanced processing results saved: {results_file}")
                
                print(f"\n🎤 Sample Whisper Advanced Processing Results:")
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
                print(f"  📥 Stage 6 Output: MP3 files ({len(extracted_samples)} samples)")
                print(f"  🔄 Stage 6.5 Processing: MP3 → WAV + Whisper analysis")
                print(f"  📤 Stage 6.5 Output: Clean WAV files ({len(clean_chunks)} chunks)")
                print(f"  🤖 Stage 7 Processing: Advanced Whisper analysis + transcription")
                print(f"  📤 Stage 7 Output: Enhanced WAV + transcripts ({len(processed_results)} files)")
                print(f"  🎯 AI Model: OpenAI Whisper throughout pipeline")
            else:
                print("❌ Stage 7 Whisper advanced processing failed - no results returned")
                processed_results = []
                
        except Exception as e:
            print(f"❌ Stage 7 Whisper advanced processing failed: {e}")
            print(f"💡 Check that WAV files exist in: {clean_audio_dir}")
            processed_results = []
    else:
        print("\n⏭️ Skipping Stage 7 - no clean WAV chunks available")
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
    print(f"🤖 Whisper clean chunks (WAV): {len(clean_chunks) if 'clean_chunks' in locals() else 0}")
    print(f"📝 Whisper processed + transcripts: {len(processed_results) if 'processed_results' in locals() else 0}")
    
    voice_confirmation_rate = (len(confirmed_voice) / len(audio_links) * 100) if audio_links else 0
    clean_chunk_rate = (len(clean_chunks) / len(extracted_samples) * 100) if extracted_samples and 'clean_chunks' in locals() else 0
    print(f"📈 Voice confirmation rate: {voice_confirmation_rate:.1f}%")
    print(f"📈 Whisper clean chunk rate: {clean_chunk_rate:.1f}%")
    print(f"🆔 Snapshot ID: {snapshot_id}")
    print(f"📁 Results saved in: {cfg.OUTPUT_DIR}")
    print(f"🔄 Pipeline order: 1 → 2 → 3 → 4 → 5 → 6 → 6.5 → 7")
    print(f"🤖 AI Enhancement: OpenAI Whisper integration in stages 6.5 and 7")
    print(f"🎵 Audio format flow: MP3 (Stage 6) → WAV + Whisper (Stage 6.5) → Enhanced WAV + Transcripts (Stage 7)")

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
    if 'clean_chunks' in locals() and clean_chunks:
        print(f"  9. {clean_chunks_file} - 🤖 WHISPER CLEAN CHUNKS METADATA")
        print(f"  10. {whisper_overlap_detector.output_dir} - 🎵 WHISPER CLEAN AUDIO FILES (WAV)")
    if 'processed_results' in locals() and processed_results:
        print(f"  11. {results_file} - 📝 WHISPER ENHANCED RESULTS + TRANSCRIPTS")
        print(f"  12. {advanced_processor.output_dir} - 🎤 FINAL WHISPER PROCESSED FILES")


# Individual Stage Runner Functions
def run_stage6_5_only(input_path, output_dir="output"):
    """Run only Stage 6.5: Whisper Audio Processing and Overlap Detection"""
    print("🤖 STAGE 6.5 ONLY: Whisper Audio Processing and Overlap Detection")
    print("=" * 60)
    
    if os.path.isdir(input_path):
        print(f"📁 Processing audio files from directory: {input_path}")
        cfg = Config()
        whisper_overlap_detector = PyannoteWhisperProcessor(
            output_dir=os.path.join(output_dir, "clean_chunks"),
            chunk_duration_minutes=5,
            huggingface_token=cfg.HUGGINGFACE_TOKEN
        )
        
        # Process the voice samples directory directly
        voice_samples_dir = input_path
        
        # Get all audio files from the directory
        audio_files = []
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            audio_files.extend([f for f in os.listdir(voice_samples_dir) if f.lower().endswith(ext)])
        
        print(f"📁 Found {len(audio_files)} audio files to process")
        
        all_results = []
        for audio_file in audio_files:
            audio_path = os.path.join(voice_samples_dir, audio_file)
            print(f"🎵 Processing: {audio_file}")
            try:
                results = whisper_overlap_detector.process_audio_file(audio_path)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error processing {audio_file}: {e}")
                continue
        
        clean_chunks = all_results
        
        if clean_chunks:
            import time
            timestamp = int(time.time())
            clean_chunks_file = os.path.join(output_dir, f"6_5_whisper_audio_dir_{timestamp}_clean_chunks.csv")
            pd.DataFrame(clean_chunks).to_csv(clean_chunks_file, index=False)
            print(f"✅ Stage 6.5 completed!")
            print(f"🤖 Whisper processing successful")
            print(f"🔍 Clean chunks created: {len(clean_chunks)}")
            print(f"📁 Clean chunks directory: {whisper_overlap_detector.output_dir}")
            print(f"📊 Results CSV: {clean_chunks_file}")
            print(f"💡 Next: Run Stage 7 with --stage7-only {whisper_overlap_detector.output_dir}")
            
            # Show sample transcriptions
            print(f"\n📝 Sample Whisper Transcriptions:")
            for i, chunk in enumerate(clean_chunks[:3], 1):
                transcription = chunk.get('transcription', '')[:100] + "..." if chunk.get('transcription') else 'N/A'
                voice_pct = chunk.get('voice_percentage', 0)
                print(f"  {i}. Voice: {voice_pct:.1f}% | Text: {transcription}")
        else:
            print("❌ No clean chunks found")
    
    else:
        if not os.path.exists(input_path):
            print(f"❌ Input file/directory not found: {input_path}")
            return
        
        try:
            df = pd.read_csv(input_path)
            extracted_samples = df.to_dict('records')
            print(f"📥 Loaded {len(extracted_samples)} extracted samples from CSV: {input_path}")
            
            cfg = Config()
            whisper_overlap_detector = PyannoteWhisperProcessor(
                output_dir=os.path.join(output_dir, "clean_chunks"),
                chunk_duration_minutes=5,
                huggingface_token=cfg.HUGGINGFACE_TOKEN
            )
            
            clean_chunks = whisper_overlap_detector.process_extracted_samples(extracted_samples)
            if clean_chunks:
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                clean_chunks_file = os.path.join(output_dir, f"6_5_whisper_{base_name}_clean_chunks.csv")
                pd.DataFrame(clean_chunks).to_csv(clean_chunks_file, index=False)
                print(f"✅ Stage 6.5 completed!")
                print(f"🤖 Whisper processing successful")
                print(f"🔍 Clean chunks created: {len(clean_chunks)}")
                print(f"📁 Clean chunks directory: {whisper_overlap_detector.output_dir}")
                print(f"📊 Results CSV: {clean_chunks_file}")
                print(f"💡 Next: Run Stage 7 with --stage7-only {whisper_overlap_detector.output_dir}")
            else:
                print("❌ No clean chunks found")
        except Exception as e:
            print(f"❌ Error processing input: {e}")


def run_stage7_only(voice_samples_dir, output_dir="stage7_output"):
    """Run only Stage 7: Advanced Whisper Processing (Enhanced Analysis)"""
    print("🤖 STAGE 7 ONLY: Advanced Whisper Processing (Enhanced Analysis)")
    print("=" * 60)
    
    cfg = Config()
    if not os.path.exists(voice_samples_dir):
        print(f"❌ Voice samples directory not found: {voice_samples_dir}")
        return
    
    wav_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]
    mp3_files = [f for f in os.listdir(voice_samples_dir) if f.endswith('.mp3')]
    
    if not wav_files and not mp3_files:
        print(f"❌ No audio files (WAV/MP3) found in: {voice_samples_dir}")
        return
    
    print(f"📥 Found {len(wav_files)} WAV files and {len(mp3_files)} MP3 files to process")
    print(f"🤖 Using OpenAI Whisper for advanced speech analysis and transcription")
    
    if mp3_files:
        print("🔄 MP3 files will be converted to WAV for Whisper processing")
    
    try:
        advanced_processor = PyannoteWhisperProcessor(
            output_dir=output_dir,
            chunk_duration_minutes=5,
            huggingface_token=cfg.HUGGINGFACE_TOKEN
        )
        
        clean_audio_dir = voice_samples_dir
        wav_files = [f for f in os.listdir(clean_audio_dir) if f.endswith('.wav')]
        print(f"📁 Processing {len(wav_files)} clean WAV files from: {clean_audio_dir}")
        
        all_results = []
        for wav_file in wav_files:
            wav_path = os.path.join(clean_audio_dir, wav_file)
            print(f"🎵 Processing: {wav_file}")
            try:
                results = advanced_processor.process_audio_file(wav_path)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️ Error processing {wav_file}: {e}")
                continue
        
        processed_results = all_results
        
        if processed_results:
            print(f"✅ Stage 7 Whisper advanced processing completed!")
            print(f"🤖 Successfully processed: {len(processed_results)} files")
            print(f"📁 Output directory: {advanced_processor.output_dir}")
            
            # Save enhanced results with transcriptions
            results_file = os.path.join(output_dir, "whisper_enhanced_results.csv")
            enhanced_results = []
            
            for result in processed_results:
                enhanced_results.append({
                    'input_file': result.get('clean_chunk_file', ''),
                    'transcription': result.get('transcription', ''),
                    'voice_percentage': result.get('voice_percentage', 0),
                    'overlap_percentage': result.get('overlap_percentage', 0),
                    'word_count': result.get('word_count', 0),
                    'char_count': result.get('char_count', 0),
                    'avg_confidence': result.get('avg_confidence', 0),
                    'speakers_detected': result.get('speakers_detected', 1),
                    'processing_method': result.get('processing_method', 'whisper'),
                    'model_used': 'OpenAI-Whisper-Base'
                })
            
            pd.DataFrame(enhanced_results).to_csv(results_file, index=False)
            print(f"📊 Enhanced results saved: {results_file}")
            
            print(f"\n📝 Sample Whisper Processing Results:")
            for i, result in enumerate(processed_results[:3], 1):
                input_file = os.path.basename(result.get('clean_chunk_file', 'unknown'))
                transcription = result.get('transcription', '')[:80] + "..." if result.get('transcription') else 'N/A'
                voice_pct = result.get('voice_percentage', 0)
                word_count = result.get('word_count', 0)
                print(f"  {i}. {input_file}")
                print(f"     Voice: {voice_pct:.1f}% | Words: {word_count}")
                print(f"     Transcript: {transcription}")
                print(f"     Model: OpenAI-Whisper-Base")
        else:
            print("❌ No files could be processed successfully")
    
    except Exception as e:
        print(f"❌ Stage 7 Whisper processing error: {e}")


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
    print(f"💡 Next: Run Stage 4 with --stage4-only {links_file}")
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
    
    runner = Step3_5_YouTubeTwitchRunner("output")
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


def run_stage5_only(audio_links_file):
    """Run only Stage 5: Enhanced Audio Content Detection"""
    print("🎵 STAGE 5 ONLY: Enhanced YouTube, Twitch & TikTok Audio Content Detection")
    print("=" * 50)
    
    if not os.path.exists(audio_links_file):
        print(f"❌ Audio links file not found: {audio_links_file}")
        return None
    
    df = pd.read_csv(audio_links_file)
    audio_links = df.to_dict('records')
    print(f"📥 Loaded {len(audio_links)} audio links from: {audio_links_file}")
    
    cfg = Config()
    audio_detector = AudioContentDetector("config.json")

    
    audio_detected_links = audio_detector.detect_audio_content(audio_links)
    if not audio_detected_links:
        print("🔍 No audio content detected")
        return None
    
    base_name = os.path.splitext(os.path.basename(audio_links_file))[0]
    audio_detected_file = os.path.join("output", f"5_{base_name.replace('4_', '')}_audio_detected.csv")
    pd.DataFrame(audio_detected_links).to_csv(audio_detected_file, index=False)
    
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
    print("  Audio Types:")
    for audio_type, count in sorted(audio_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {audio_type}: {count}")
    print("  Confidence Levels:")
    for confidence, count in confidence_levels.items():
        print(f"    {confidence}: {count}")
    print(f"📁 Audio detected file: {audio_detected_file}")
    print(f"💡 Next: Run Stage 6 with --stage6-only {audio_detected_file}")
    return audio_detected_file


def run_stage6_only(confirmed_voice_file, output_dir="output"):
    """Run only Stage 6: Voice Sample Extraction (MP3 Output) - Direct import, no subprocess"""
    print("🎤 STAGE 6 ONLY: Voice Sample Extraction (MP3 Output)")
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
            
            print("✅ Stage 6 completed successfully!")
            print(f"🎤 Successfully processed {len(extracted_samples)} voice samples")
            print(f"📁 Voice samples directory: {voice_samples_dir}")
            print(f"📄 Results CSV: {result_csv}")
            print(f"💡 Next: Run Stage 6.5 with --stage6_5-only {voice_samples_dir}")
        else:
            print("❌ No voice samples could be extracted")
            print("💡 Check internet connection and ensure yt-dlp/ffmpeg are installed")
    
    except Exception as e:
        print(f"❌ Error running Stage 6: {e}")


def show_help():
    help_text = """
🤖 WHISPER ENHANCED YOUTUBE, TWITCH & TIKTOK VOICE CONTENT PIPELINE

PIPELINE FLOW:
1→2→3→4→5→6→6.5(Whisper)→7(Whisper Enhanced)

INDIVIDUAL STAGES:
--stage1-only FILE     Stage 1: Account Validation
--stage2-only FILE     Stage 2: Bright Data Trigger
--stage3-only SNAPSHOT Stage 3: Data Download
--stage3_5-only FILE   Stage 3.5: YouTube/Twitch Channel Discovery
--stage4-only FILE     Stage 4: YouTube/Twitch Filter
--stage5-only FILE     Stage 5: Audio Detection (FINAL VOICE DECISION)
--stage6-only FILE     Stage 6: Voice Sample Extraction (MP3 Output)
--stage6_5-only DIR    Stage 6.5: Whisper Processing & Overlap Detection
--stage7-only DIR      Stage 7: Advanced Whisper Analysis + Transcription

🤖 WHISPER ENHANCEMENTS:
- Stage 6.5: Whisper voice activity detection and overlap detection
- Stage 7: Advanced Whisper transcription and analysis
- Automatic MP3 → WAV conversion for Whisper processing
- Enhanced speech quality analysis with confidence scores
- Full transcription generation for all audio chunks

📝 TRANSCRIPTION FEATURES:
- Real-time speech-to-text using OpenAI Whisper
- Voice quality assessment and confidence scoring
- Intelligent overlap detection using transcription analysis
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
    parser.add_argument("--stage4-only", help="Run only Stage 4 - Platform filtering")
    parser.add_argument("--stage5-only", help="Run only Stage 5 - Audio detection")
    parser.add_argument("--stage6-only", help="Run only Stage 6 - Voice sample extraction (MP3)")
    parser.add_argument("--stage6_5-only", help="Run only Stage 6.5 - Whisper processing & overlap detection")
    parser.add_argument("--stage7-only", help="Run only Stage 7 - Advanced Whisper analysis + transcription")
    
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
            print(f"❌ Confirmed voice file not found: {args.stage6_only}")
            sys.exit(1)
        run_stage6_only(args.stage6_only, "output")
        sys.exit(0)
    
    if args.stage6_5_only:
        if not os.path.exists(args.stage6_5_only):
            print(f"❌ Input path not found: {args.stage6_5_only}")
            sys.exit(1)
        run_stage6_5_only(args.stage6_5_only, "output")
        sys.exit(0)
    
    if args.stage7_only:
        if not os.path.exists(args.stage7_only):
            print(f"❌ Audio directory not found: {args.stage7_only}")
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
            print(f"🚀 Starting Whisper enhanced 7-stage pipeline")
            print(f"🤖 AI Model: OpenAI Whisper for speech processing and transcription")
            print(f"🔄 Force recheck: {'Yes' if args.force_recheck else 'No (using cache)'}")
            print(f"🎵 Audio flow: Stage 6 (MP3) → Stage 6.5 (Whisper) → Stage 7 (Enhanced)")
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
        print("  python main_pipeline.py --stage6_5-only output/voice_samples/")
        print("  python main_pipeline.py --stage7-only output/clean_chunks")
        print("\n🔄 Pipeline: 1→2→3→4→5→6→6.5(Whisper)→7(Enhanced)")
        print("🤖 Enhanced: OpenAI Whisper speech processing and transcription in stages 6.5 and 7")
        sys.exit(1)
