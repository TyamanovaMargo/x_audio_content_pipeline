import csv
import os
from typing import List, Dict, Optional
import pandas as pd

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
