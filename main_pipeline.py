import os
import pandas as pd
import argparse
import sys
import shutil
import time
import logging

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
from step9_speaker_diarization import SpeakerDiarization
from snapshot_manager import SnapshotManager

class PipelineManager:
    """Enhanced YouTube & Twitch Voice Content Pipeline Manager"""
    
    def __init__(self, output_dir=None, verbose=False):
        self.cfg = Config()
        if output_dir:
            self.cfg.OUTPUT_DIR = output_dir
            
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            
        self.snapshot_id = None
        self.data_files = {}  # Store file paths between stages
        
    def run_stage_1(self, input_file, force_recheck=False):
        """Stage 1: Account Validation with Persistent Logging"""
        print("\n✅ STAGE 1: Account Validation with Persistent Logging")
        print("-" * 60)
        
        log_file = os.path.join(self.cfg.OUTPUT_DIR, "processed_accounts.json")
        validator = AccountValidator(
            max_concurrent=self.cfg.MAX_CONCURRENT_VALIDATIONS,
            delay_min=self.cfg.VALIDATION_DELAY_MIN,
            delay_max=self.cfg.VALIDATION_DELAY_MAX,
            log_file=log_file
        )
        
        existing_accounts_file = os.path.join(self.cfg.OUTPUT_DIR, "1_existing_accounts.csv")
        valid_accounts = validator.validate_accounts_from_file(
            input_file, existing_accounts_file, force_recheck=force_recheck
        )
        
        if not valid_accounts:
            print("❌ No valid accounts found.")
            return None
            
        self.data_files['valid_accounts'] = valid_accounts
        self.data_files['accounts_file'] = existing_accounts_file
        return valid_accounts
        
    def run_stage_2(self, valid_accounts=None):
        """Stage 2: Bright Data Snapshot Management"""
        print("\n🚀 STAGE 2: Bright Data Snapshot Management")
        print("-" * 60)
        
        if valid_accounts is None:
            valid_accounts = self.data_files.get('valid_accounts')
            if not valid_accounts:
                print("❌ No valid accounts available. Run Stage 1 first.")
                return None
        
        trigger = BrightDataTrigger(self.cfg.BRIGHT_DATA_API_TOKEN, self.cfg.BRIGHT_DATA_DATASET_ID)
        usernames = [acc['username'] for acc in valid_accounts]
        
        sm = SnapshotManager(self.cfg.OUTPUT_DIR)
        existing_snapshot = sm.get_reusable_snapshot(usernames)
        
        if existing_snapshot:
            print(f"🔄 Using existing snapshot: {existing_snapshot}")
            self.snapshot_id = existing_snapshot
        else:
            print(f"🆕 Creating new snapshot for {len(usernames)} usernames")
            self.snapshot_id = trigger.create_snapshot_from_usernames(usernames)
            if not self.snapshot_id:
                print("❌ Failed to create snapshot")
                return None
            sm.register_snapshot(self.snapshot_id, valid_accounts)
            
        self.data_files['snapshot_id'] = self.snapshot_id
        return self.snapshot_id
        
    def run_stage_3(self, snapshot_id=None):
        """Stage 3: Data Download & External Link Extraction"""
        print("\n⬇️ STAGE 3: Data Download & External Link Extraction")
        print("-" * 60)
        
        if snapshot_id is None:
            snapshot_id = self.data_files.get('snapshot_id')
            if not snapshot_id:
                print("❌ No snapshot ID available. Run Stage 2 first.")
                return None
        
        downloader = BrightDataDownloader(self.cfg.BRIGHT_DATA_API_TOKEN)
        profiles = downloader.wait_and_download_snapshot(snapshot_id, self.cfg.MAX_SNAPSHOT_WAIT)
        
        if not profiles:
            print("❌ Failed to download snapshot data")
            return None
            
        profiles_file = os.path.join(self.cfg.OUTPUT_DIR, f"2_snapshot_{snapshot_id}_results.csv")
        pd.DataFrame(profiles).to_csv(profiles_file, index=False)
        print(f"📊 Saved {len(profiles)} profiles to: {profiles_file}")
        
        links = downloader.extract_external_links(profiles)
        if not links:
            print("🔗 No external links found in profiles")
            return None
            
        links_file = os.path.join(self.cfg.OUTPUT_DIR, f"3_snapshot_{snapshot_id}_external_links.csv")
        pd.DataFrame(links).to_csv(links_file, index=False)
        print(f"🔗 Saved {len(links)} external links to: {links_file}")
        
        self.data_files['profiles'] = profiles
        self.data_files['links'] = links
        self.data_files['links_file'] = links_file
        return links
        
    def run_stage_4(self, links=None, links_file=None):
        """Stage 4: YouTube & Twitch Audio Platform Filtering"""
        print("\n🎯 STAGE 4: YouTube & Twitch Audio Platform Filtering")
        print("-" * 60)
        
        if links is None:
            if links_file and os.path.exists(links_file):
                links = pd.read_csv(links_file).to_dict('records')
            else:
                links = self.data_files.get('links')
                if not links:
                    print("❌ No links available. Run Stage 3 first or provide --links-file.")
                    return None
        
        audio_filter = AudioContentFilter()
        audio_links = audio_filter.filter_audio_links(links)
        
        if not audio_links:
            print("🔍 No YouTube or Twitch links found")
            return None
            
        snapshot_id = self.data_files.get('snapshot_id', 'unknown')
        audio_file = os.path.join(self.cfg.OUTPUT_DIR, f"4_snapshot_{snapshot_id}_audio_links.csv")
        pd.DataFrame(audio_links).to_csv(audio_file, index=False)
        print(f"🎯 Found {len(audio_links)} YouTube/Twitch audio links!")
        
        self.data_files['audio_links'] = audio_links
        self.data_files['audio_file'] = audio_file
        return audio_links
        
    def run_stage_4_5(self, audio_links=None, audio_file=None):
        """Stage 4.5: Audio Content Detection"""
        print("\n🎵 STAGE 4.5: YouTube & Twitch Audio Content Detection")
        print("-" * 60)
        
        if audio_links is None:
            if audio_file and os.path.exists(audio_file):
                audio_links = pd.read_csv(audio_file).to_dict('records')
            else:
                audio_links = self.data_files.get('audio_links')
                if not audio_links:
                    print("❌ No audio links available. Run Stage 4 first or provide --audio-file.")
                    return None
        
        audio_detector = AudioContentDetector(timeout=10)
        audio_detected_links = audio_detector.detect_audio_content(audio_links)
        
        if not audio_detected_links:
            print("🔍 No audio content detected")
            return None
            
        snapshot_id = self.data_files.get('snapshot_id', 'unknown')
        audio_detected_file = os.path.join(self.cfg.OUTPUT_DIR, f"4_5_snapshot_{snapshot_id}_audio_detected.csv")
        pd.DataFrame(audio_detected_links).to_csv(audio_detected_file, index=False)
        print(f"🎵 Found {len(audio_detected_links)} links with actual audio content!")
        
        self.data_files['audio_detected_links'] = audio_detected_links
        self.data_files['audio_detected_file'] = audio_detected_file
        return audio_detected_links
        
    def run_stage_5(self, audio_detected_links=None, audio_detected_file=None):
        """Stage 5: Voice Content Verification"""
        print("\n🎙️ STAGE 5: YouTube & Twitch Voice Content Verification")
        print("-" * 60)
        
        if audio_detected_links is None:
            if audio_detected_file and os.path.exists(audio_detected_file):
                audio_detected_links = pd.read_csv(audio_detected_file).to_dict('records')
            else:
                audio_detected_links = self.data_files.get('audio_detected_links')
                if not audio_detected_links:
                    print("❌ No audio detected links available. Run Stage 4.5 first or provide --audio-detected-file.")
                    return None
        
        voice_verifier = VoiceContentVerifier(timeout=15)
        verified_links = voice_verifier.verify_voice_content(audio_detected_links)
        
        snapshot_id = self.data_files.get('snapshot_id', 'unknown')
        verified_file = os.path.join(self.cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_verified_voice.csv")
        pd.DataFrame(verified_links).to_csv(verified_file, index=False)
        
        confirmed_voice = [link for link in verified_links if link.get('has_voice')]
        
        if confirmed_voice:
            confirmed_file = os.path.join(self.cfg.OUTPUT_DIR, f"5_snapshot_{snapshot_id}_confirmed_voice.csv")
            pd.DataFrame(confirmed_voice).to_csv(confirmed_file, index=False)
            print(f"🎙️ Found {len(confirmed_voice)} confirmed voice content links!")
        else:
            print("❌ No voice content confirmed after verification")
            
        self.data_files['verified_links'] = verified_links
        self.data_files['confirmed_voice'] = confirmed_voice
        self.data_files['confirmed_file'] = confirmed_file if confirmed_voice else None
        return confirmed_voice
        
    def run_stage_6(self, confirmed_voice=None, confirmed_file=None):
        """Stage 6: Voice Sample Extraction → MP3 Files"""
        print("\n🎤 STAGE 6: Voice Sample Extraction → MP3 Files")
        print("-" * 60)
        
        if confirmed_voice is None:
            if confirmed_file and os.path.exists(confirmed_file):
                confirmed_voice = pd.read_csv(confirmed_file).to_dict('records')
            else:
                confirmed_voice = self.data_files.get('confirmed_voice')
                if not confirmed_voice:
                    print("❌ No confirmed voice links available. Run Stage 5 first or provide --confirmed-file.")
                    return None
        
        voice_samples_dir = os.path.join(self.cfg.OUTPUT_DIR, "voice_samples")
        sample_extractor = VoiceSampleExtractor(
            output_dir=voice_samples_dir,
            min_duration=self.cfg.MIN_SAMPLE_DURATION,
            max_duration=self.cfg.MAX_SAMPLE_DURATION,
            quality="192"
        )
        
        extracted_samples = sample_extractor.download_audio_for_all(confirmed_voice)
        
        if extracted_samples:
            snapshot_id = self.data_files.get('snapshot_id', 'unknown')
            extraction_file = os.path.join(self.cfg.OUTPUT_DIR, f"6_snapshot_{snapshot_id}_voice_samples.csv")
            pd.DataFrame(extracted_samples).to_csv(extraction_file, index=False)
            
            report_file = sample_extractor.generate_download_report(extracted_samples)
            
            durations = [sample.get('audio_duration', 0) for sample in extracted_samples]
            total_hours = sum(durations) / 3600
            
            print(f"\n🎤 Voice Sample Extraction Summary:")
            print(f" 📊 Total voice links: {len(confirmed_voice)}")
            print(f" ✅ Successful MP3 extractions: {len(extracted_samples)}")
            print(f" ⏱️ Total audio extracted: {total_hours:.2f} hours")
            print(f" 📊 Average sample duration: {sum(durations)/len(durations):.1f} seconds")
            print(f" 📁 MP3 files saved in: {voice_samples_dir}")
            print(f" 📄 Report file: {report_file}")
        else:
            print("❌ No voice samples could be extracted")
            
        self.data_files['extracted_samples'] = extracted_samples
        self.data_files['voice_samples_dir'] = voice_samples_dir
        self.data_files['extraction_file'] = extraction_file if extracted_samples else None
        return extracted_samples
        
    def run_stage_7(self, voice_samples_dir=None, extracted_samples=None):
        """Stage 7: Advanced Voice Processing → Voice-Only WAV Files"""
        print("\n🔍 STAGE 7: Advanced Voice Processing → Voice-Only WAV Files")
        print("-" * 60)
        
        if voice_samples_dir is None:
            voice_samples_dir = self.data_files.get('voice_samples_dir')
            if not voice_samples_dir:
                voice_samples_dir = os.path.join(self.cfg.OUTPUT_DIR, "voice_samples")
                
        if not os.path.exists(voice_samples_dir) or not os.listdir(voice_samples_dir):
            print(f"❌ Voice samples directory not found or empty: {voice_samples_dir}")
            print("💡 Run Stage 6 first to extract MP3 files")
            return None
            
        processor = AdvancedVoiceProcessor(
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "voice_analysis"),
            min_voice_confidence=0.6,
            voice_segment_min_length=2.0
        )
        
        voice_only_results = processor.process_audio_directory(voice_samples_dir)
        
        voice_only_samples = []
        if voice_only_results:
            results_file = processor.save_results(voice_only_results)
            report_file = processor.generate_report(voice_only_results)
            
            for result in voice_only_results:
                voice_only_file = result.get('voice_only_file')
                if voice_only_file and os.path.exists(voice_only_file):
                    voice_only_samples.append({
                        'processed_username': result.get('username', 'unknown'),
                        'platform_source': result.get('platform', 'unknown'),
                        'voice_only_file': voice_only_file,
                        'sample_file': voice_only_file,
                        'speech_text': result.get('speech_analysis', {}).get('combined_text', ''),
                        'voice_confidence': result.get('final_analysis', {}).get('final_confidence', 0),
                        'word_count': result.get('speech_analysis', {}).get('word_count', 0),
                        'voice_duration': result.get('voice_duration', 0)
                    })
            
            snapshot_id = self.data_files.get('snapshot_id', 'unknown')
            voice_only_file = os.path.join(self.cfg.OUTPUT_DIR, f"7_snapshot_{snapshot_id}_voice_only.csv")
            pd.DataFrame(voice_only_samples).to_csv(voice_only_file, index=False)
            
            extracted_count = len(extracted_samples) if extracted_samples else len(os.listdir(voice_samples_dir))
            
            print(f"🔍 Advanced Voice Processing Summary:")
            print(f" 📊 Total MP3 files processed: {extracted_count}")
            print(f" ✅ Voice-only WAV files created: {len(voice_only_samples)}")
            if extracted_count > 0:
                print(f" 📈 Voice detection rate: {(len(voice_only_samples) / extracted_count * 100):.1f}%")
            print(f" 📁 Voice-only WAV files in: voice_analysis/voice_only_audio/")
            print(f" 📊 Detailed results: {results_file}")
            print(f" 📄 Report: {report_file}")
        else:
            print("❌ No voice-only content found after processing")
            
        self.data_files['voice_only_samples'] = voice_only_samples
        self.data_files['voice_only_file'] = voice_only_file if voice_only_samples else None
        return voice_only_samples
        
    def run_stage_8(self, voice_only_samples=None, voice_only_file=None):
        """Stage 8: Background Noise Reduction → Denoised WAV Files"""
        print("\n🎛️ STAGE 8: Background Noise Reduction → Denoised WAV Files")
        print("-" * 60)
        
        if voice_only_samples is None:
            if voice_only_file and os.path.exists(voice_only_file):
                voice_only_samples = pd.read_csv(voice_only_file).to_dict('records')
            else:
                voice_only_samples = self.data_files.get('voice_only_samples')
                if not voice_only_samples:
                    print("❌ No voice-only samples available. Run Stage 7 first or provide --voice-only-file.")
                    return None
        
        voice_only_dir = os.path.join(self.cfg.OUTPUT_DIR, "voice_analysis", "voice_only_audio")
        
        if not os.path.exists(voice_only_dir) or not os.listdir(voice_only_dir):
            print(f"❌ Voice-only directory not found or empty: {voice_only_dir}")
            print("💡 Run Stage 7 first to create voice-only WAV files")
            return None
            
        noise_reducer = NoiseReducer(
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "voice_analysis"),
            mode="quick",
            sample_rate=16000
        )
        
        nr_results = noise_reducer.process_directory(voice_only_dir)
        successful_denoising = sum(1 for r in nr_results if r.get('output_file'))
        
        print(f"✅ Noise reduction completed: {successful_denoising} WAV files denoised")
        
        denoised_dir = os.path.join(self.cfg.OUTPUT_DIR, "voice_analysis", "denoised_audio")
        denoised_count = 0
        
        for sample in voice_only_samples:
            orig_path = sample.get('voice_only_file', '')
            if orig_path:
                base_name = os.path.splitext(os.path.basename(orig_path))[0]
                denoised_path = os.path.join(denoised_dir, f"{base_name}_denoised.wav")
                if os.path.exists(denoised_path):
                    sample['voice_only_file'] = denoised_path
                    sample['sample_file'] = denoised_path
                    sample['is_denoised'] = True
                    denoised_count += 1
        
        snapshot_id = self.data_files.get('snapshot_id', 'unknown')
        voice_only_denoised_file = os.path.join(self.cfg.OUTPUT_DIR, f"8_snapshot_{snapshot_id}_voice_only_denoised.csv")
        pd.DataFrame(voice_only_samples).to_csv(voice_only_denoised_file, index=False)
        
        print(f"📁 Denoised WAV files saved in: voice_analysis/denoised_audio/")
        print(f"✅ Updated {denoised_count} file paths to denoised versions")
        
        self.data_files['voice_only_samples'] = voice_only_samples  # Updated with denoised paths
        self.data_files['voice_only_denoised_file'] = voice_only_denoised_file
        return voice_only_samples
        
    def run_stage_9(self, voice_only_samples=None, voice_only_denoised_file=None):
        """Stage 9: Speaker Diarization → Lead Speaker WAV Files"""
        print("\n🎭 STAGE 9: Speaker Diarization → Lead Speaker WAV Files")
        print("-" * 60)
        
        if voice_only_samples is None:
            if voice_only_denoised_file and os.path.exists(voice_only_denoised_file):
                voice_only_samples = pd.read_csv(voice_only_denoised_file).to_dict('records')
            else:
                voice_only_samples = self.data_files.get('voice_only_samples')
                if not voice_only_samples:
                    print("❌ No voice-only samples available. Run Stage 8 first or provide --voice-only-denoised-file.")
                    return None
        
        diarizer = SpeakerDiarization(
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "voice_analysis")
        )
        
        speaker_identified_samples = diarizer.process_voice_samples(voice_only_samples)
        
        if speaker_identified_samples:
            snapshot_id = self.data_files.get('snapshot_id', 'unknown')
            final_results_file = os.path.join(self.cfg.OUTPUT_DIR, f"9_snapshot_{snapshot_id}_speaker_identified.csv")
            pd.DataFrame(speaker_identified_samples).to_csv(final_results_file, index=False)
            
            single_speaker = sum(1 for s in speaker_identified_samples if not s.get('multiple_speakers', False))
            multi_speaker = sum(1 for s in speaker_identified_samples if s.get('multiple_speakers', False))
            
            print(f"🎭 Speaker Diarization Summary:")
            print(f" 📊 Total denoised WAV samples: {len(voice_only_samples)}")
            print(f" ✅ Speaker-identified samples: {len(speaker_identified_samples)}")
            print(f" 🎤 Single speaker: {single_speaker}")
            print(f" 👥 Multiple speakers: {multi_speaker}")
            print(f" 📈 Multi-speaker rate: {(multi_speaker / len(speaker_identified_samples) * 100):.1f}%")
            print(f" 📁 Lead speaker WAV files in: voice_analysis/speaker_samples/")
        else:
            print("❌ No samples could be processed for speaker diarization")
            
        self.data_files['speaker_identified_samples'] = speaker_identified_samples
        self.data_files['final_results_file'] = final_results_file if speaker_identified_samples else None
        return speaker_identified_samples

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced YouTube & Twitch Voice Content Pipeline with Individual Stage Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main_pipeline.py usernames.txt

  # Run specific stage
  python main_pipeline.py usernames.txt --stage 7
  
  # Run specific stage with input file
  python main_pipeline.py usernames.txt --stage 8 --voice-only-file output/7_snapshot_123_voice_only.csv
  
  # Run stages 7-9 only
  python main_pipeline.py usernames.txt --stages 7 8 9

Audio Processing Flow:
  Stage 6: YouTube/Twitch → MP3 files (voice_samples/)
  Stage 7: MP3 → Voice-only WAV files (voice_analysis/voice_only_audio/)
  Stage 8: WAV → Denoised WAV files (voice_analysis/denoised_audio/)
  Stage 9: Denoised WAV → Speaker samples (voice_analysis/speaker_samples/)
        """
    )
    
    parser.add_argument("input_file", help="Input file with usernames (CSV or TXT)")
    parser.add_argument("--force-recheck", action="store_true", help="Force recheck of all accounts")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Stage control
    parser.add_argument("--stage", type=int, help="Run specific stage only (1-9)")
    parser.add_argument("--stages", nargs="+", type=int, help="Run specific stages (e.g., --stages 1 2 3)")
    
    # Input files for individual stages
    parser.add_argument("--links-file", help="Input links file for Stage 4")
    parser.add_argument("--audio-file", help="Input audio links file for Stage 4.5")
    parser.add_argument("--audio-detected-file", help="Input audio detected file for Stage 5")
    parser.add_argument("--confirmed-file", help="Input confirmed voice file for Stage 6")
    parser.add_argument("--voice-samples-dir", help="Input voice samples directory for Stage 7")
    parser.add_argument("--voice-only-file", help="Input voice-only file for Stage 8")
    parser.add_argument("--voice-only-denoised-file", help="Input denoised file for Stage 9")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Initialize pipeline manager
    pipeline = PipelineManager(output_dir=args.output_dir, verbose=args.verbose)
    
    try:
        start_time = time.time()
        
        # Determine which stages to run
        if args.stage:
            stages_to_run = [args.stage]
        elif args.stages:
            stages_to_run = sorted(args.stages)
        else:
            stages_to_run = list(range(1, 10))  # All stages 1-9
        
        print("🎙️ ENHANCED YOUTUBE & TWITCH VOICE CONTENT PIPELINE")
        print("=" * 60)
        print(f"🎯 Stages to run: {stages_to_run}")
        
        # Run stages
        for stage in stages_to_run:
            if stage == 1:
                result = pipeline.run_stage_1(args.input_file, args.force_recheck)
            elif stage == 2:
                result = pipeline.run_stage_2()
            elif stage == 3:
                result = pipeline.run_stage_3()
            elif stage == 4:
                result = pipeline.run_stage_4(links_file=args.links_file)
            elif stage == 5:
                result = pipeline.run_stage_4_5(audio_file=args.audio_file)
            elif stage == 5:
                result = pipeline.run_stage_5(audio_detected_file=args.audio_detected_file)
            elif stage == 6:
                result = pipeline.run_stage_6(confirmed_file=args.confirmed_file)
            elif stage == 7:
                result = pipeline.run_stage_7(voice_samples_dir=args.voice_samples_dir)
            elif stage == 8:
                result = pipeline.run_stage_8(voice_only_file=args.voice_only_file)
            elif stage == 9:
                result = pipeline.run_stage_9(voice_only_denoised_file=args.voice_only_denoised_file)
            
            if result is None and stage in [1, 2, 3]:  # Critical early stages
                print(f"❌ Stage {stage} failed. Stopping pipeline.")
                break
        
        end_time = time.time()
        print(f"\n🎉 PIPELINE COMPLETED!")
        print(f"⏱️ Total execution time: {(end_time - start_time) / 60:.1f} minutes")
        
        # Generate final report if full pipeline was run
        if stages_to_run == list(range(1, 10)):
            _generate_final_report(pipeline)
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def _generate_final_report(pipeline):
    """Generate final report if available"""
    try:
        snapshot_id = pipeline.data_files.get('snapshot_id', 'unknown')
        
        stats = {
            'total_accounts': len(pipeline.data_files.get('valid_accounts', [])),
            'external_links': len(pipeline.data_files.get('links', [])),
            'audio_links': len(pipeline.data_files.get('audio_links', [])),
            'confirmed_voice': len(pipeline.data_files.get('confirmed_voice', [])),
            'extracted_samples': len(pipeline.data_files.get('extracted_samples', [])),
            'voice_only_samples': len(pipeline.data_files.get('voice_only_samples', [])),
            'speaker_identified': len(pipeline.data_files.get('speaker_identified_samples', [])),
            'total_hours': 0
        }
        
        extracted_samples = pipeline.data_files.get('extracted_samples', [])
        if extracted_samples:
            stats['total_hours'] = sum(sample.get('audio_duration', 0) for sample in extracted_samples) / 3600
        
        report_file = os.path.join(pipeline.cfg.OUTPUT_DIR, f"FINAL_PIPELINE_REPORT_{snapshot_id}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🎙️ ENHANCED YOUTUBE & TWITCH VOICE CONTENT PIPELINE - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Snapshot ID: {snapshot_id}\n\n")
            
            f.write("📊 PIPELINE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in stats.items():
                if key == 'total_hours':
                    f.write(f"{key.replace('_', ' ').title()}: {value:.2f} hours\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        print(f"📄 Final pipeline report saved: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Could not generate final report: {e}")

if __name__ == "__main__":
    main()
