# Voice Extraction Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This is a comprehensive Python-based pipeline designed to extract and process voice content from X/Twitter profiles. The pipeline validates accounts, collects profile data using Bright Data, extracts external links, filters for audio content from platforms like YouTube and Twitch, and uses OpenAI Whisper for advanced voice processing and transcription.

The pipeline is modular with 7 main stages, each building on the previous one to automate the extraction of high-quality voice samples from social media profiles. It includes comprehensive timing measurement, duplicate prevention, logging, and reporting features for efficient data management.

**🤖 AI Enhancement:** Integrated OpenAI Whisper for speech recognition, overlap detection, and transcription in stage 7.

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
 5. config.json and config.py ( put your token in both files )


## Usage

The pipeline is divided into sequential steps. Run them in order, or integrate into a main script. Outputs from one step feed into the next (e.g., via CSV files).

### Individual Stages
Run specific stages independently:

```bash
# Stage 1: Account Validation
python main_pipeline.py --stage1-only sample_usernames.csv

# Stage 2: Bright Data Trigger
python main_pipeline.py --stage2-only output/1_existing_accounts.csv

# Stage 3: Data Download
python main_pipeline.py --stage3-only s_snapshot_id

# Stage 3.5: YouTube/Twitch Discovery (Optional)
python main_pipeline.py --stage3_5-only output/3_snapshot_xyz_external_links.csv

# Stage 4: Audio Platform Filtering
python main_pipeline.py --stage4-only output/3_snapshot_xyz_external_links.csv

# Stage 5: Audio Content Detection
python main_pipeline.py --stage5-only output/4_snapshot_xyz_audio_links.csv

# Stage 6: Voice Sample Extraction
python main_pipeline.py --stage6-only output/5_snapshot_xyz_audio_detected.csv

# Stage 7: Advanced Whisper Analysis
python main_pipeline.py --stage7-only output/voice_samples/
```

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

## Project Structure

```
x-audio-content-pipeline/
├── 📁 Core Pipeline Files
│   ├── main_pipeline.py                    # 🎯 Main orchestrator with all 7 stages
│   ├── config.py                          # ⚙️ Configuration settings
│   ├── config_example.py                  # 📋 Configuration template
│   ├── config.json                        # 🔧 Additional JSON configuration
│   └── requirements.txt                   # 📦 Python dependencies
│
├── 📁 Pipeline Stages
│   ├── step1_validate_accounts.py         # ✅ Stage 1: Account validation
│   ├── step2_bright_data_trigger.py       # 🚀 Stage 2: Snapshot management
│   ├── step3_bright_data_download.py      # ⬇️ Stage 3: Data download
│   ├── step3_5_youtube_twitch_runner.py   # 🔍 Stage 3.5: Channel discovery
│   ├── step4_audio_filter.py              # 🎯 Stage 4: Audio platform filter
│   ├── step5_voice_detector.py            # 🎵 Stage 5: Voice detection
│   ├── step6_voice_sample_extractor.py    # 🎤 Stage 6: Sample extraction
│   └── step7_advanced_voice_processor.py  # 🤖 Stage 7: Whisper processing
│
├── 📁 Utilities
│   ├── utils/
│   │   ├── __init__.py                    # Package initialization
│   │   ├── checker_web.py                 # 🌐 Web validation utilities
│   │   ├── io_utils.py                    # 📂 I/O helper functions
│   │   └── username_utils.py              # 👤 Username processing
│   ├── snapshot_manager.py                # 📊 Snapshot lifecycle management
│   ├── audio_from_youtube.py              # 🎵 YouTube audio utilities
│   └── split_chunks.py                    # ✂️ Audio chunking utilities
│
├── 📁 External Scraper
│   └── youtube-twitch-x-scraper/          # 🔍 Enhanced channel discovery
│       ├── youtube_twitch_scraper.py      # Main scraper implementation
│       ├── enhanced_matching.py           # Advanced matching algorithms
│       ├── config.py                      # Scraper configuration
│       ├── requirements.txt               # Scraper dependencies
│       ├── proxy/                         # Proxy configuration
│       └── README.md                      # Scraper documentation
│
├── 📁 Input/Output
│   ├── input/                             # 📥 Input files directory
│   │   ├── merged_user_all_usernames.csv  # Combined username datasets
│   │   ├── merged_user_mbti.csv           # MBTI personality data
│   │   └── users_twitter.csv              # Twitter user data
│   ├── output/                            # 📤 Pipeline output directory
│   │   ├── snapshots/                     # 📊 Snapshot metadata
│   │   ├── voice_samples/                 # 🎤 Extracted MP3 files
│   │   ├── processed_accounts.json        # 📋 Account validation log
│   │   ├── 1_existing_accounts.csv        # ✅ Stage 1 output
│   │   ├── 2_snapshot_*_results.csv       # 📊 Stage 2-3 output
│   │   ├── 3_snapshot_*_external_links.csv # 🔗 Stage 3 output
│   │   ├── 4_*_audio_links.csv            # 🎯 Stage 4 output
│   │   ├── 5_*_audio_detected.csv         # 🎵 Stage 5 output
│   │   ├── 6_voice_samples_results.csv    # 🎤 Stage 6 output
│   │   └── 7_whisper_results.csv          # 📝 Stage 7 output
│   └── output_audio2/                     # 🎵 Alternative audio output
│
├── 📁 Configuration & Samples
│   └── sample_usernames.csv               # 📋 Example input file
│
└── 📁 Development
    ├── .gitignore                         # 🚫 Git ignore rules
    ├── .venv/                             # 🐍 Python virtual environment
    ├── __pycache__/                       # 🗂️ Python cache files
    └── README.md                          # 📖 This documentation
```

### Key Components

#### 🎯 Main Pipeline (`main_pipeline.py`)
- **Full Pipeline:** Complete 7-stage execution with timing
- **Individual Stages:** Run any stage independently
- **Timing Measurement:** Comprehensive execution time tracking
- **Error Handling:** Robust error management and recovery
- **CLI Interface:** Rich command-line interface with help

#### 🤖 Whisper Integration
- **Stage 7:** Advanced Whisper analysis with transcription
- **Features:** Voice activity detection, overlap detection, transcription generation
- **Output:** Clean WAV files + detailed transcription metadata

#### 📊 Data Flow
```
Input CSV → Stage 1 → Stage 2 → Stage 3 → (Stage 3.5) → Stage 4 → Stage 5 → Stage 6 → Stage 7
    ↓           ↓         ↓         ↓           ↓            ↓         ↓         ↓           ↓
Usernames → Accounts → Snapshots → Links → Enhanced → Audio → Voice → MP3 → WAV+Whisper → Transcripts
```

#### 🔧 Configuration Files
- **`config.py`:** Main configuration (API tokens, settings)
- **`config_example.py`:** Template for new setups
- **`config.json`:** Additional JSON-based configuration

#### 📦 Dependencies
- **Core:** pandas, requests, asyncio, aiohttp
- **Audio:** yt-dlp, ffmpeg-python, pydub
- **AI:** openai-whisper, torch, transformers
- **Web:** playwright, beautifulsoup4, selenium

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
