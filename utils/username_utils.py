import re
from typing import Optional

def normalize_username(username: str) -> Optional[str]:
    """
    Normalize username by trimming whitespace, removing @ prefix,
    and validating against X/Twitter username rules.
    
    Returns None if username is invalid.
    """
    if not username:
        return None
    
    # Trim whitespace and remove leading @
    normalized = username.strip().lstrip('@')
    
    # Validate username: letters, digits, underscore only
    if not re.match(r'^[A-Za-z0-9_]+$', normalized):
        return None
    
    # X usernames must be 1-15 characters
    if len(normalized) < 1 or len(normalized) > 15:
        return None
    
    return normalized

def build_profile_url(username: str) -> str:
    """Build X.com profile URL for a given username."""
    return f"https://x.com/{username}"

def validate_username(username: str) -> bool:
    """Check if username follows X/Twitter rules."""
    return normalize_username(username) is not None
