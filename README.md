# X Audio Content Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This is a comprehensive Python-based pipeline designed to extract and process voice content from X/Twitter profiles. The pipeline validates accounts, collects profile data using Bright Data, extracts external links, filters for audio content from platforms like YouTube and Twitch, and uses advanced AI models for sophisticated voice detection and analysis.

The pipeline is modular with **6 main stages**, each building on the previous one to automate the extraction of high-quality voice samples from social media profiles. It includes comprehensive timing measurement, duplicate prevention, logging, and reporting features for efficient data management.

**🤖 AI Enhancement:** Integrated multiple AI models including OpenAI Whisper (transcription) and Pyannote (voice activity detection) for advanced voice processing in Stage 6.

## Key Features

- **Account Validation:** Checks if X/Twitter accounts exist using web scraping
- **Snapshot Management:** Tracks and reuses Bright Data snapshots to prevent duplicates
- **Data Collection:** Triggers and downloads profile data via Bright Data API
- **Link Extraction & Filtering:** Identifies and filters external links to YouTube and Twitch
- **Audio Content Detection:** Detects audio content from social media links
- **Voice Sample Extraction:** Downloads audio samples with intelligent filename generation
- **Advanced Voice Detection:** Uses multiple AI models (Whisper, Pyannote VAD) to distinguish voice from music
- **Music Filtering:** Sophisticated algorithms to filter out music and retain only voice content
- **Logging & Reporting:** Maintains comprehensive logs, summaries, and reports
- **Concurrency & Rate Limiting:** Built-in delays and concurrency limits to avoid rate limiting

## Pipeline Stages

### Stage 1: Account Validation
- Validates X/Twitter account existence
- Maintains persistent log to avoid re-checking
- Output: `1_existing_accounts.csv`

### Stage 2: Bright Data Snapshot Management
- Triggers Bright Data collection for valid accounts
- Prevents duplicate snapshots
- Output: Snapshot IDs for data collection

### Stage 3: Data Download & Link Extraction
- Downloads profile data from Bright Data
- Extracts external links from profiles
- Output: `3_snapshot_*_external_links.csv`

### Stage 4: Audio Platform Filtering
- Filters links for YouTube, Twitch, and TikTok content
- Identifies potential audio/video content
- Output: `4_*_audio_links.csv`

### Stage 5: Voice Sample Extraction (MP3 Output)
- Downloads audio samples from filtered links
- Converts to MP3 format for processing
- Output: MP3 files in `voice_samples/` directory

### Stage 6: Advanced Voice Detection
- **Dual-Model AI Analysis:** Uses Whisper and Pyannote VAD for efficient voice detection
- **Voice vs Music Detection:** Sophisticated algorithms to filter music
- **Transcription Analysis:** Analyzes speech patterns and content quality
- **Quality Metrics:** Speech duration, word count, speaking rate analysis
- **Smart Filtering:** Moves voice-confirmed files to `stage6_processed/`
- **Outputs:**
  - `6_voice_detection_results_*.csv` - Detailed analysis results with voice scores, transcriptions, and detection metrics
  - `stage6_processed/` directory - Contains only MP3 files confirmed to contain human voice
  - Voice detection log with processing status for each audio file
  - Comprehensive metrics: speech duration, word count, speaking rate, voice confidence scores

## Requirements

### System Requirements
- **Python:** 3.8 or higher
- **FFmpeg:** For audio processing
- **Git:** For version control

### Python Dependencies
Install via `pip install -r requirements.txt`:

```txt
pandas>=1.3.0
requests>=2.25.0
aiohttp>=3.8.0
asyncio-throttle>=1.0.0
yt-dlp>=2023.1.0
ffmpeg-python>=0.2.0
pydub>=0.25.0
openai-whisper>=20230314
pyannote.audio>=2.1.0
playwright>=1.20.0
beautifulsoup4>=4.9.0
selenium>=4.0.0
```

### External Tools
- **FFmpeg:** Download from [official site](https://ffmpeg.org/) or use package manager
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install ffmpeg`
- **Playwright browsers:** `playwright install`

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/TyamanovaMargo/x_audio_content_pipeline.git
cd x-audio-content-pipeline
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
playwright install
```

4. **Configure API access:**
Create environment variables or update `config.py`:
```bash
export BRIGHT_DATA_TOKEN=your_api_token
export BRIGHT_DATA_DATASET_ID=your_dataset_id
```

## Usage

### Full Pipeline
Run the complete 6-stage pipeline:
```bash
python main_pipeline.py --input sample_usernames.csv
```

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

# Stage 5: Voice Sample Extraction
python main_pipeline.py --stage5-only output/4_*_audio_links.csv

# Stage 6: Advanced Voice Detection
python main_pipeline.py --stage6-only output/voice_samples/
```

### Pipeline Management
```bash
# Show processed accounts log
python main_pipeline.py --show-log

# Show snapshot summary
python main_pipeline.py --show-snapshots

# Clear processed accounts log
python main_pipeline.py --clear-log

# Detailed help
python main_pipeline.py --help-detailed
```

## Project Structure

```
x-audio-content-pipeline/
├── 📁 Core Pipeline Files
│   ├── main_pipeline.py                    # 🎯 Main orchestrator with all 6 stages
│   ├── config.py                          # ⚙️ Configuration settings
│   ├── config.json                        # 🔧 JSON configuration
│   ├── requirements.txt                   # 📦 Python dependencies
│   └── snapshot_manager.py                # 📊 Snapshot lifecycle management
│
├── 📁 Pipeline Stages
│   ├── step1_validate_accounts.py         # ✅ Stage 1: Account validation
│   ├── step2_bright_data_trigger.py       # 🚀 Stage 2: Snapshot management
│   ├── step3_bright_data_download.py      # ⬇️ Stage 3: Data download
│   ├── step3_5_youtube_twitch_runner.py   # 🔍 Stage 3.5: Channel discovery
│   ├── step4_audio_filter.py              # 🎯 Stage 4: Audio platform filter
│   ├── step5_voice_sample_extractor.py    # 🎤 Stage 5: MP3 sample extraction
│   └── step6_voice_detector_advance.py    # 🤖 Stage 6: Advanced AI voice detection
│
├── 📁 Utilities
│   ├── utils/
│   │   ├── __init__.py                    # Package initialization
│   │   ├── checker_web.py                 # 🌐 Web validation utilities
│   │   ├── io_utils.py                    # 📂 I/O helper functions
│   │   └── username_utils.py              # 👤 Username processing
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
│   ├── output/                            # 📤 Pipeline output directory
│   │   ├── snapshots/                     # 📊 Snapshot metadata
│   │   ├── voice_samples/                 # 🎤 Extracted MP3 files
│   │   │   └── stage6_processed/          # ✅ Voice-confirmed files
│   │   ├── processed_accounts.json        # 📋 Account validation log
│   │   ├── 1_existing_accounts.csv        # ✅ Stage 1 output
│   │   ├── 2_snapshot_*_results.csv       # 📊 Stage 2-3 output
│   │   ├── 3_snapshot_*_external_links.csv # 🔗 Stage 3 output
│   │   ├── 4_*_audio_links.csv            # 🎯 Stage 4 output
│   │   ├── 5_*_audio_detected.csv         # 🎵 Stage 5 output
│   │   └── 6_voice_detection_results_*.csv # 🤖 Stage 6 AI analysis
│   └── output_audio2/                     # 🎵 Alternative audio output
│
└── 📁 Configuration & Development
    ├── sample_usernames.csv               # 📋 Example input file
    ├── .gitignore                         # 🚫 Git ignore rules
    ├── .venv/                             # 🐍 Python virtual environment
    └── README.md                          # 📖 This documentation
```

## Data Flow

```
Input CSV → Stage 1 → Stage 2 → Stage 3 → (Stage 3.5) → Stage 4 → Stage 5 → Stage 6
    ↓           ↓         ↓         ↓           ↓            ↓         ↓         ↓
Usernames → Accounts → Snapshots → Links → Enhanced → Audio → MP3 → AI Analysis
                                                      Links   Samples   (Voice/Music)
```

## Advanced Voice Detection (Stage 6)

Stage 6 uses multiple AI models for sophisticated voice vs music detection:

### AI Models Used
- **OpenAI Whisper:** Speech transcription and analysis
- **Pyannote Audio:** Voice Activity Detection (VAD)

### Detection Criteria
- **Speech Duration:** Minimum 5 seconds of detected speech
- **Word Count:** At least 5 meaningful words transcribed
- **Speech Ratio:** Minimum 30% speech content vs total duration
- **Speaking Rate:** 50-200 words per minute (filters music/noise)
- **Content Analysis:** Repetition patterns and vocabulary diversity
- **Voice Score:** Multi-model confidence scoring

### Music Filtering
- Detects repetitive patterns common in music
- Analyzes speaking rate to identify non-speech content
- Uses speech-to-total-duration ratio
- Combines multiple heuristics for accurate classification

## Configuration

### Environment Variables
```bash
BRIGHT_DATA_TOKEN=your_api_token
BRIGHT_DATA_DATASET_ID=your_dataset_id
```

### Pipeline Settings
- **Output Directories:** Configurable via `config.py`
- **Delays & Concurrency:** Adjustable rate limiting
- **AI Model Thresholds:** Customizable detection sensitivity
- **Logging Levels:** Comprehensive logging configuration

## Troubleshooting

### Common Issues
- **Dependencies Missing:** Ensure FFmpeg is in PATH
- **API Issues:** Verify Bright Data token and dataset ID
- **Memory Issues:** Reduce batch sizes for large datasets
- **Rate Limiting:** Increase delays between requests

### Performance Optimization
- **Concurrent Processing:** Adjust `MAX_CONCURRENT_VALIDATIONS`
- **Model Loading:** Models are cached after first load
- **Disk Space:** Monitor output directory size
- **Memory Usage:** Consider processing in smaller batches

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Pyannote Audio for voice activity detection
- Bright Data for profile data collection

---

*Last updated: September 14, 2024*
