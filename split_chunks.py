import os
import subprocess
from typing import List

def split_mp3_to_wav_chunks(input_mp3_path: str, output_dir: str, chunk_length_sec: int = 300) -> List[str]:
    """
    Splits an input MP3 file into WAV chunks of specified length (default 5 minutes).

    Args:
        input_mp3_path: Path to the input MP3 file.
        output_dir: Directory to save output WAV chunks.
        chunk_length_sec: Length of each chunk in seconds (default 300s = 5 minutes).

    Returns:
        List of output WAV chunk file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get audio duration from mp3 using ffprobe
    def get_duration(path):
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    duration = get_duration(input_mp3_path)
    chunks = []
    num_chunks = int(duration // chunk_length_sec) + (1 if duration % chunk_length_sec > 0 else 0)

    for i in range(num_chunks):
        start = i * chunk_length_sec
        current_duration = min(chunk_length_sec, duration - start)
        output_wav_path = os.path.join(output_dir, f"chunk_{i+1:02d}.wav")

        cmd = [
            'ffmpeg',
            '-y',  # overwrite output
            '-ss', str(start),
            '-t', str(current_duration),
            '-i', input_mp3_path,
            '-ar', '16000',  # 16 kHz
            '-ac', '1',      # mono
            '-c:a', 'pcm_s16le',  # WAV PCM 16-bit
            output_wav_path
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(output_wav_path):
            chunks.append(output_wav_path)

    return chunks

# Example usage:
if __name__ == '__main__':
    input_mp3 = '/Users/margotiamanova/Desktop/PROJECTS/x-audio-content-pipeline/output_audio2/cdawgva_audio_part1mp3.mp3'
    output_directory = 'wav_chunks'
    result_chunks = split_mp3_to_wav_chunks(input_mp3, output_directory)
    print(f'Created {len(result_chunks)} WAV chunks:')
    for chunk_file in result_chunks:
        print(chunk_file)
