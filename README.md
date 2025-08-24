# Voice Extraction Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This project is a comprehensive Python-based pipeline designed to process X/Twitter usernames, validate accounts, collect profile data using Bright Data, extract external links, filter for audio content from platforms like YouTube and Twitch, detect and verify voice content, extract audio samples, and perform advanced voice processing to isolate voice-only segments.

The pipeline is modular, with each step building on the previous one to automate the extraction of high-quality voice samples from social media profiles. It includes duplicate prevention, logging, and reporting features for efficient data management.

Key goals:
- Validate and filter valid X/Twitter accounts.
- Collect profile data while avoiding redundant snapshots.
- Focus on voice-centric content from linked YouTube/Twitch channels.
- Extract and process voice samples for further analysis (e.g., AI voice modeling).

**Note:** This pipeline requires external dependencies like Bright Data API access, yt-dlp for downloads, ffmpeg for audio processing, and Playwright for web scraping. Ensure you comply with all platform terms of service and data usage policies.

## Features

- **Account Validation:** Checks if X/Twitter accounts exist using web scraping.
- **Snapshot Management:** Tracks and reuses Bright Data snapshots to prevent duplicates.
- **Data Collection:** Triggers and downloads profile data via Bright Data API.
- **Link Extraction & Filtering:** Identifies and filters external links to YouTube and Twitch.
- **Audio & Voice Detection:** Detects audio content and verifies voice presence (e.g., podcasts, streams).
- **Sample Extraction:** Downloads short audio samples with filename including username and source.
- **Advanced Voice Processing:** Uses VAD (Voice Activity Detection) and speech recognition to extract voice-only segments.
- **Logging & Reporting:** Maintains logs, summaries, and reports for snapshots and extractions.
- **Concurrency & Delays:** Built-in delays and concurrency limits to avoid rate limiting.

## Requirements

- **Python:** 3.8 or higher.
- **Libraries:** Install via `pip install -r requirements.txt` (create this file with the following):

git clone https://github.com/your-repo/voice-extraction-pipeline.git
cd voice-extraction-pipeline


2. Install Python dependencies:
pip install -r requirements.txt


3. Install external tools:
- Install ffmpeg: Download from official site or use package manager (e.g., `brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu).
- Install yt-dlp: `pip install yt-dlp`.
- Install Playwright browsers: `playwright install`.

4. Set up configuration:
- Create a `.env` file or set environment variables for Bright Data:
  ```
  BRIGHT_DATA_TOKEN=your_api_token
  BRIGHT_DATA_DATASET_ID=your_dataset_id
  ```

## Usage

The pipeline is divided into sequential steps. Run them in order, or integrate into a main script. Outputs from one step feed into the next (e.g., via CSV files).

### Step 1: Validate Accounts
Validate usernames from an input file and output valid accounts.
python step1_validate_accounts.py --input sample_usernames.csv --output validated_accounts.csv

- Options: `--force-recheck` to reprocess all, `--max-accounts N` to limit processing.

### Step 2: Manage & Trigger Snapshots
Use SnapshotManager to check for duplicates and trigger Bright Data snapshots.
- Integrate with `step2_bright_data_trigger.py` for API calls.

### Step 3: Download Snapshot Data
Download data from Bright Data snapshots.
python step3_bright_data_download.py --snapshot-id YOUR_SNAPSHOT_ID


### Step 4: Filter Audio Links
Filter extracted links to YouTube/Twitch.
- Use `step4_audio_filter.py` on downloaded data.

### Step 4.5: Detect Audio Content
Detect if links contain audio.
- Run `step4_5_audio_detector.py`.

### Step 5: Verify Voice Content
Verify presence of voice (e.g., speech vs. music).
- Run `step5_voice_verification.py`.

### Step 6: Extract Voice Samples
Extract MP3 samples from confirmed voice links.
python step6_voice_sample_extractor.py --input confirmed_voice.csv --output-dir voice_samples --duration 120 --quality 192

- Generates samples in `voice_samples/` with filenames like `username_source_timestamp.mp3`.
- Options: `--list-files`, `--clean-temp` for maintenance.

### Step 7: Advanced Voice Processing
Process extracted samples to isolate voice-only audio.
python step7_advanced_voice_processor.py voice_samples/ --output-dir voice_analysis

- Outputs voice-only WAV files in `voice_analysis/voice_only_audio/`.
- Generates reports and CSV results.

### Full Pipeline Example
Create a main.py to chain steps, e.g.:
Pseudocode

validated = validate_accounts('usernames.txt')
snapshot_id = trigger_snapshot(validated)
data = download_snapshot(snapshot_id)
links = extract_links(data)
audio_links = filter_audio(links)
voice_links = verify_voice(audio_links)
samples = extract_samples(voice_links)
processed = process_voice(samples)


## Configuration

- **Output Directories:** Defaults to `output/`, `voice_samples/`, `voice_analysis/`. Customize via CLI args.
- **Delays & Concurrency:** Adjustable in classes (e.g., `AccountValidator` for min/max delays).
- **API Tokens:** Store securely in environment variables.
- **Logging:** Processed logs saved as JSON (e.g., `processed_accounts.json`).

## File Structure

- `snapshot_manager.py`: Manages Bright Data snapshots.
- `step1_validate_accounts.py`: Account validation.
- `step2_bright_data_trigger.py`: Snapshot triggering.
- `step3_bright_data_download.py`: Data downloading.
- `step4_audio_filter.py`: Audio platform filtering.
- `step4_5_audio_detector.py`: Audio detection.
- `step5_voice_verification.py`: Voice verification.
- `step6_voice_sample_extractor.py`: Sample extraction.
- `step7_advanced_voice_processor.py`: Voice processing.
- Utilities: `checker_web.py` (web checker), `io_utils.py` (I/O helpers), `username_utils.py` (username handling).

## Troubleshooting

- **Timeouts/Errors:** Increase delays or reduce concurrency.
- **Dependencies Missing:** Ensure ffmpeg and yt-dlp are in PATH.
- **API Issues:** Verify Bright Data token and dataset ID.
- **No Voice Detected:** Adjust confidence thresholds in processors.

## Contributing

Pull requests welcome! For major changes, open an issue first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Generated on Monday, August 18, 2025.*
