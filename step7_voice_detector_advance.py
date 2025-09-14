
import os
import shutil
import argparse
import logging
from glob import glob
from pydub import AudioSegment
import torch
import whisper
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition


def load_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # mono
        audio = audio.set_frame_rate(16000)  # 16 kHz
        return audio
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def save_file_if_voice_detected(src_file, dest_dir, detection_results, threshold, min_duration, verbose):
    detected_score, speech_duration = detection_results
    if verbose:
        print(f"Score: {detected_score}, Duration: {speech_duration}")

    if detected_score >= threshold and speech_duration >= min_duration:
        shutil.copy(src_file, dest_dir)
        if verbose:
            print(f"Copied {src_file} to {dest_dir}")
        return True
    return False


def main(args):
    logging.basicConfig(filename='voice_detection.log', level=logging.INFO)

    # Load Whisper model
    whisper_model = whisper.load_model("base")

    # Initialize Pyannote pipeline for VAD
    pyannote_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

    # Load SpeechBrain pretrained speaker recognition model
    speechbrain_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    os.makedirs(args.dest, exist_ok=True)

    files = glob(os.path.join(args.source, "*.mp3"))

    total_files = len(files)
    processed_files = 0
    passed_files = 0

    for file_path in files:
        processed_files += 1
        if args.verbose:
            print(f"Processing {file_path} ({processed_files}/{total_files})")

        audio = load_audio(file_path)
        if audio is None:
            logging.warning(f"Skipping {file_path} due to load failure.")
            continue

        # Use pyannote VAD pipeline
        audio_data = audio.raw_data
        try:
            vad_result = pyannote_pipeline({'waveform': torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0), 'sample_rate': 16000})
            speech_segments = vad_result.get_timeline().support()
            speech_duration = sum(segment.duration() for segment in speech_segments)
        except Exception as e:
            logging.error(f"Pyannote error on {file_path}: {e}")
            continue

        # Skip if no speech duration
        if speech_duration < args.min_duration:
            if args.verbose:
                print(f"No sufficient speech detected in {file_path}")
            logging.info(f"No speech detected in {file_path}")
            continue

        # Use Whisper for transcription to check speech content
        try:
            result = whisper_model.transcribe(file_path)
            transcript = result.get('text', '').strip()
        except Exception as e:
            logging.error(f"Whisper error on {file_path}: {e}")
            continue

        if len(transcript) < 10:  # arbitrary threshold for meaningful speech
            if args.verbose:
                print(f"No meaningful speech in {file_path}")
            logging.info(f"No meaningful speech in {file_path}")
            continue

        # Use SpeechBrain speaker recognition model to confirm voice
        try:
            score = speechbrain_model.verify_files(file_path, file_path).item()  # compare file with itself
        except Exception as e:
            logging.error(f"SpeechBrain error on {file_path}: {e}")
            score = 0

        detected = save_file_if_voice_detected(file_path, args.dest, (score, speech_duration), args.threshold, args.min_duration, args.verbose)

        if detected:
            passed_files += 1

    # Summary
    print(f"Processed {processed_files} files. {passed_files} files contained voices and were saved.")
    logging.info(f"Processed {processed_files} files. {passed_files} files saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice detection in MP3 files')
    parser.add_argument('--source', type=str, required=True, help='Source directory path')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Voice detection score threshold')
    parser.add_argument('--min_duration', type=float, default=5.0, help='Minimum speech duration in seconds')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    main(args)
