x-account-validator/
├── validate_x_accounts.py      # Main CLI entry point
├── io_utils.py                 # File I/O operations
├── username_utils.py           # Username validation and URL building
├── checker_web.py              # Playwright account verification logic
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── sample_usernames.csv        # Example input file
└── tests/                      # Unit tests
    ├── test_username_utils.py
    └── test_io_utils.py


# X/Twitter Account Validator

A Python tool that uses Playwright to verify whether X/Twitter accounts exist without using the official API.

## ⚠️ Important Legal Notice

This tool uses automated web browsing to check account existence. Please note:
- X/Twitter's Terms of Service prohibit unauthorized scraping and automated access
- Use this tool responsibly with low request rates and proper delays
- Consider obtaining explicit permission from X/Twitter before use
- This tool is for educational and research purposes

## Installation

1. Install Python dependencies:
pip install -r requirements.txt


2. Install Playwright browsers:
playwright install chromium


## Usage

### Basic usage:
python validate_x_accounts.py --input usernames.csv --output existing_accounts.csv