#!/usr/bin/env python3
"""
YouTube Audio Downloader using yt-dlp
Recommended approach - most reliable and up-to-date
"""
import os
import sys
import argparse
from pathlib import Path

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

def download_audio_yt_dlp(url: str, output_dir: str = ".", quality: str = "192") -> str:
    """
    Download audio from YouTube using yt-dlp
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save audio file
        quality: Audio quality (192, 128, 96, etc.)
    
    Returns:
        Path to downloaded audio file
    """
    if not YT_DLP_AVAILABLE:
        print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
        return ""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': quality,
        }],
        'extractaudio': True,
        'audioformat': 'mp3',
        'noplaylist': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            print(f"🎵 Title: {title}")
            print(f"⏱️ Duration: {duration//60}:{duration%60:02d}")
            print(f"📁 Output: {output_dir}")
            print("🔄 Downloading...")
            
            # Download the audio
            ydl.download([url])
            
            # Find the downloaded file
            expected_filename = f"{title}.mp3"
            mp3_files = list(Path(output_dir).glob("*.mp3"))
            
            if mp3_files:
                # Get the most recently created MP3 file
                latest_file = max(mp3_files, key=os.path.getctime)
                print(f"✅ Audio downloaded successfully: {latest_file}")
                return str(latest_file)
            else:
                print("❌ Audio file not found after download")
                return ""
                
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", default="./audio", help="Output directory (default: ./audio)")
    parser.add_argument("--quality", "-q", default="192", help="Audio quality in kbps (default: 192)")
    
    args = parser.parse_args()
    
    print("🎙️ YouTube Audio Downloader (yt-dlp)")
    print("=" * 50)
    
    result = download_audio_yt_dlp(args.url, args.output, args.quality)
    
    if result:
        print(f"\n🎉 Success! Audio saved to: {result}")
    else:
        print("\n💥 Download failed. Check the URL and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
