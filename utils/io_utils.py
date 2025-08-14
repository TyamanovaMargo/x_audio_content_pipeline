import csv
import os
from typing import List, Dict, Optional
import pandas as pd
import json
import os
from datetime import datetime
from typing import Set, Dict

def load_processed_log(log_file: str = "processed_usernames.json") -> Dict:
    """
    Load previously processed usernames from log file.
    
    Returns:
        Dict with username as key and processing info as value
    """
    if not os.path.exists(log_file):
        return {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"⚠️  Could not read log file {log_file}, starting fresh")
        return {}

def save_processed_log(processed_usernames: Dict, log_file: str = "processed_usernames.json"):
    """Save processed usernames to log file."""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(processed_usernames, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Could not save log file: {e}")

def add_to_processed_log(username: str, status: str, processed_dict: Dict):
    """Add a username and its status to the processed log."""
    processed_dict[username] = {
        'status': status,
        'checked_at': datetime.now().isoformat(),
        'profile_url': f"https://x.com/{username}"
    }

def filter_new_usernames(usernames: list, processed_log: Dict) -> list:
    """Filter out usernames that have already been processed."""
    new_usernames = []
    skipped_count = 0
    
    for username in usernames:
        if username in processed_log:
            skipped_count += 1
            print(f"⏭️  Skipping {username} (already checked: {processed_log[username]['status']})")
        else:
            new_usernames.append(username)
    
    if skipped_count > 0:
        print(f"📋 Skipped {skipped_count} already processed usernames")
    
    return new_usernames

def read_usernames(file_path: str, username_column: Optional[str] = None) -> List[str]:
    """
    Read usernames from CSV or TXT file.
    
    Args:
        file_path: Path to input file
        username_column: Column name for CSV files (default: 'username')
    
    Returns:
        List of usernames
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
        
        # Determine column to use
        if username_column:
            if username_column not in df.columns:
                raise ValueError(f"Column '{username_column}' not found in CSV")
            column = username_column
        else:
            # Try common column names
            possible_columns = ['username', 'user', 'handle', 'account']
            column = None
            for col in possible_columns:
                if col in df.columns:
                    column = col
                    break
            
            if not column:
                # Use first column
                column = df.columns[0]
        
        return df[column].astype(str).tolist()
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def write_results_csv(results: List[Dict], output_path: str):
    """Write results to CSV file."""
    if not results:
        return
    
    fieldnames = ['username', 'profile_url', 'status']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def filter_existing_accounts(results: List[Dict]) -> List[Dict]:
    """Filter results to only include existing accounts."""
    return [result for result in results if result['status'] == 'exists']
