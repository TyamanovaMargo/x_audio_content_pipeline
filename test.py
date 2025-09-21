import os
import nemo.collections.asr as nemo_asr
from huggingface_hub import login

def authenticate_huggingface():
    """
    Authenticate with Hugging Face using your token.
    Get your token from: https://huggingface.co/settings/tokens
    """
    token = "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"  # Replace with your actual token
    login(token=token)

def is_probably_singing(text: str):
    """
    Heuristic to detect singing based on transcription patterns.
    """
    if len(text) < 10:
        return False
    
    # Check for repeated words (common in songs)
    words = text.lower().split()
    repeated_words = any(words.count(word) > 2 for word in set(words))
    
    # Common song-like words
    song_words = {'la', 'oh', 'yeah', 'baby', 'sing', 'love', 'ohh', 'na', 'hey', 'whoa'}
    common_song_word = any(word in text.lower() for word in song_words)
    
    # Check for repetitive patterns (like "la la la")
    repetitive_pattern = any(word * 3 in text.lower().replace(' ', '') for word in ['la', 'na', 'oh', 'ah'])
    
    return repeated_words or common_song_word or repetitive_pattern

def main(audio_folder):
    # Authenticate with Hugging Face first
    print("Authenticating with Hugging Face...")
    authenticate_huggingface()
    
    # Load the model using NeMo - OFFICIAL METHOD from documentation
    print("Loading NVIDIA Parakeet-TDT-0.6b-v3 model...")
    try:
        # This is the official method from HuggingFace model page
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        print("Model loaded successfully!")
        print(f"Model type: {type(asr_model)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative method...")
        try:
            # Alternative: Let's try downloading the .nemo file and loading with restore_from
            from huggingface_hub import hf_hub_download
            nemo_file = hf_hub_download(repo_id="nvidia/parakeet-tdt-0.6b-v3", filename="parakeet-tdt-0.6b-v3.nemo")
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(nemo_file)
            print("Model loaded with restore_from method!")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return
    
    # Process each audio file
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for filename in audio_files:
        file_path = os.path.join(audio_folder, filename)
        print(f"\nProcessing {filename}...")
        
        try:
            # Transcribe using NeMo model
            transcriptions = asr_model.transcribe([file_path])
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

if __name__ == "__main__":
    # Set your audio folder path
    audio_folder = "/Users/margotiamanova/Desktop/PROJECTS/x-audio-content-pipeline/sample_audio"
    
    print("🦜 Parakeet-TDT Singing Detection Script")
    print("=" * 50)
    
    main(audio_folder)
