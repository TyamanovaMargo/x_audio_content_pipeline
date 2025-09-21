import os
import nemo.collections.asr as nemo_asr
from huggingface_hub import login, hf_hub_download
import librosa
import soundfile as sf
import tempfile

def authenticate_huggingface():
    """
    Authenticate with Hugging Face using your token.
    """
    token = "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"
    login(token=token)

def convert_to_mono_wav(input_path):
    """
    Convert any audio file to mono WAV format that NeMo can process.
    """
    try:
        # Load audio with librosa (handles many formats, converts to mono)
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Save as WAV
        sf.write(temp_path, audio, 16000)
        return temp_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None

def is_probably_singing(text: str):
    """
    Heuristic to detect singing based on transcription patterns.
    """
    if len(text) < 10:
        return False
    
    words = text.lower().split()
    repeated_words = any(words.count(word) > 2 for word in set(words))
    
    song_words = {'la', 'oh', 'yeah', 'baby', 'sing', 'love', 'ohh', 'na', 'hey', 'whoa'}
    common_song_word = any(word in text.lower() for word in song_words)
    
    repetitive_pattern = any(word * 3 in text.lower().replace(' ', '') for word in ['la', 'na', 'oh', 'ah'])
    
    return repeated_words or common_song_word or repetitive_pattern

def main(audio_folder):
    print("Authenticating with Hugging Face...")
    authenticate_huggingface()
    
    print("Loading NVIDIA Parakeet-TDT-0.6b-v3 model...")
    try:
        # Download and load model using the method that worked
        from nemo.core.classes import ModelPT
        nemo_file = hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v3", 
            filename="parakeet-tdt-0.6b-v3.nemo"
        )
        asr_model = ModelPT.restore_from(restore_path=nemo_file)
        print(f"Model loaded successfully! Type: {type(asr_model)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process audio files
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    temp_files = []  # Keep track of temporary files
    
    for filename in audio_files:
        file_path = os.path.join(audio_folder, filename)
        print(f"\nProcessing {filename}...")
        
        # Convert to mono WAV first
        temp_wav = convert_to_mono_wav(file_path)
        if temp_wav is None:
            print(f"❌ Could not convert {filename} to processable format")
            continue
            
        temp_files.append(temp_wav)
        
        try:
            # Transcribe using converted audio
            transcriptions = asr_model.transcribe([temp_wav])
            transcription = transcriptions[0] if transcriptions else ""
            
            # Analyze for singing
            singing_flag = is_probably_singing(transcription)
            
            # Display results
            if singing_flag:
                print(f"🎵 SINGING detected in {filename}")
            else:
                print(f"🗣️  SPEECH only in {filename}")
            
            print(f"Transcription: {transcription}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    audio_folder = "/Users/margotiamanova/Desktop/PROJECTS/x-audio-content-pipeline/sample_audio"
    
    print("🦜 Parakeet-TDT Singing Detection Script")
    print("=" * 50)
    
    main(audio_folder)

