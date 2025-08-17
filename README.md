# X/Twitter Audio Content Pipeline

A full-featured pipeline for validating X/Twitter accounts, collecting profile data through Bright Data API, extracting external links, and filtering for audio/video platforms with potential voice content such as YouTube, Spotify, TikTok, Twitch, etc.

---

## 📌 Pipeline Overview

**Stages:**
1. **Account Validation** — Check if accounts exist using Playwright web scraping  
2. **Bright Data Integration** — Create snapshots containing **only validated accounts**  
3. **Data Download** — Retrieve Bright Data snapshot results and extract external links  
4. **Audio Link Filtering** — Keep only links from audio/video platforms  
5. **Snapshot Tracking** — Save metadata for each snapshot, its accounts, and results  
6. **Results Saving** — All outputs saved in a structured `output/` folder

---

## 📂 Project Structure

x-audio-content-pipeline/
├── main_pipeline.py # Main orchestrator with snapshot tracking
├── step1_validate_accounts.py # Stage 1 - account validation
├── step2_bright_data_trigger.py # Stage 2 - create Bright Data snapshot
├── step3_bright_data_download.py # Stage 3 - snapshot download & link extraction
├── step4_audio_filter.py # Stage 4 - filter for audio platforms
├── snapshot_manager.py # Manage snapshot metadata & registry
├── utils/
│ ├── checker_web.py # Playwright-based account checker
│ ├── io_utils.py # Read/write files, handle logs
│ ├── username_utils.py # Username parsing & normalization
├── config.py # Pipeline configuration
├── requirements.txt # Dependencies
├── output/
│ ├── 1_existing_accounts.csv
│ ├── 2_snapshot_results.csv
│ ├── 3_external_links.csv
│ ├── 4_audio_links_.csv
│ └── snapshots/
│ ├── snapshot_registry.json
│ ├── s_metadata.json
│ └── s*_accounts.csv
└── README.md


---

## 🛠 Installation

Clone repo

git clone <your-repo-url>
cd x-audio-content-pipeline
Install dependencies

pip install -r requirements.txt
Install Playwright browsers

playwright install chromium


Create a `.env` file with:


---

## 📊 Input Formats

**CSV:**
username
elonmusk
naval

---

## 🚀 Running the Pipeline

### Full Run
python main_pipeline.py --input usernames.csv --output-dir output/


### With Account Limit
python main_pipeline.py --input usernames.csv --max-accounts 50


### List Snapshots
python main_pipeline.py --list-snapshots


---

## 📈 Outputs

- `1_existing_accounts.csv` — Validated accounts  
- `2_snapshot_*_results.csv` — Full Bright Data profile data  
- `3_external_links_*.csv` — Extracted external links  
- `4_audio_links_*.csv` — Filtered audio/video platform links  
- `output/snapshots/` — Metadata & account lists for each snapshot  
- `pipeline_summary.json` — Summary stats for last run  

---

## 🎵 Platforms Detected

**High:** Spotify, SoundCloud, Apple Music, Apple Podcasts, Anchor  
**Medium:** YouTube, Twitch, TikTok  
**Low:** Instagram, Discord, Kick

---

# Launch in stages (for example, for debugging)

Account verification only:
python step1_validate_accounts.py --input usernames.csv --output output/1_existing_accounts.csv

Create snapshot directly:
python step2_bright_data_trigger.py --usernames output/1_existing_accounts.csv

Downloading snapshot data:
python step3_bright_data_download.py --snapshot-id s_abc123

Filtering already collected links:
python step4_audio_filter.py --input output/3_snapshot_s_abc123_external_links.csv --output output/4_snapshot_s_abc123_audio_links.csv


# Standard execution
python main_pipeline.py --input usernames.csv

# Management commands  
python main_pipeline.py --show-log
python main_pipeline.py --show-snapshots
python main_pipeline.py --analyze-duplicates
python main_pipeline.py --help-detailed


📄 Complete Output Files:

    1_existing_accounts.csv - Validated accounts

    2_snapshot_<id>_results.csv - Full profile data

    3_snapshot_<id>_external_links.csv - All external links

    4_snapshot_<id>_audio_links.csv - Audio platform links

    5_snapshot_<id>_verified_voice.csv - All verification results

    5_snapshot_<id>_confirmed_voice.csv - 🎙️ FINAL VOICE CONTENT RESULTS


 Use Your Existing Audio Links File
python run_stage5_only.py --input output/4_snapshot_s_meb9udmcf8hu9g5yv_audio_links.csv


# Step by step execution
python main_pipeline.py --stage1-only usernames.csv
python main_pipeline.py --stage2-only output/1_existing_accounts.csv
python main_pipeline.py --stage3-only snap_12345
python main_pipeline.py --stage4-only output/3_snapshot_snap_12345_external_links.csv
python main_pipeline.py --stage5-only output/4_snapshot_snap_12345_audio_links.csv

# Check progress anytime
python main_pipeline.py --show-log
python main_pipeline.py --show-snapshots
