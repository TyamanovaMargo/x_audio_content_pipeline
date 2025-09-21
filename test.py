import os
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download
import librosa
import soundfile as sf
import tempfile

def convert_to_mono_wav(input_path, max_duration=25):
    """Convert audio to mono WAV with duration limit."""
    try:
        # Load and process audio
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Limit duration to avoid memory issues
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, audio, sr)
        return temp_path
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return None

def extract_transcription_text(result):
    """Extract text from NeMo 2.x transcription result."""
    if not result:
        return ""
    
    try:
        # NeMo 2.x returns list of Hypothesis objects
        if isinstance(result, list) and len(result) > 0:
            hypothesis = result[0]
            
            # Extract text from Hypothesis object
            if hasattr(hypothesis, 'text'):
                return hypothesis.text
            elif hasattr(hypothesis, 'words'):
                # Extract from word-level results
                if hypothesis.words:
                    return ' '.join([w.word if hasattr(w, 'word') else str(w) for w in hypothesis.words])
            elif hasattr(hypothesis, '__str__'):
                return str(hypothesis)
                
        return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def is_probably_singing(text: str):
    """Enhanced singing detection heuristic."""
    if not text or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    if len(words) < 3:
        return False
    
    # Singing indicators
    singing_score = 0
    
    # 1. Repeated words (very common in songs)
    word_counts = {}
    for word in words:
        if len(word) > 2:  # Skip very short words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    repeated_words = sum(1 for count in word_counts.values() if count >= 3)
    if repeated_words > 0:
        singing_score += 2
    
    # 2. Song-like sounds and words
    song_sounds = {
        'la', 'oh', 'yeah', 'baby', 'love', 'ohh', 'na', 'hey', 'whoa',
        'uh', 'ah', 'mm', 'hmm', 'ooh', 'aah', 'wow', 'huh', 'dance', 'tonight'
    }
    song_word_count = sum(1 for word in words if word in song_sounds)
    if song_word_count >= 2:
        singing_score += 2
    elif song_word_count >= 1:
        singing_score += 1
    
    # 3. Repetitive patterns
    repetitive_patterns = ['lalala', 'nanana', 'ahahah', 'ohohoh', 'heyheyhey']
    if any(pattern in text_lower.replace(' ', '') for pattern in repetitive_patterns):
        singing_score += 3
    
    # 4. Musical/emotional expressions
    musical_words = ['music', 'song', 'melody', 'rhythm', 'feel', 'heart', 'soul', 'dream']
    if any(word in text_lower for word in musical_words):
        singing_score += 1
    
    # Decision: score >= 2 indicates likely singing
    return singing_score >= 2

def main(audio_folder):
    print("Loading NVIDIA Parakeet-TDT-0.6b-v3 model...")
    
    try:
        # For NeMo 2.x, use the correct loading method
        from nemo.core.classes import ModelPT
        nemo_file = hf_hub_download(
            repo_id="nvidia/parakeet-tdt-0.6b-v3", 
            filename="parakeet-tdt-0.6b-v3.nemo",
            local_files_only=True
        )
        asr_model = ModelPT.restore_from(restore_path=nemo_file)
        print(f"✅ Model loaded successfully! (NeMo {nemo_asr.__version__ if hasattr(nemo_asr, '__version__') else '2.4.0'})")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get audio files (limit for testing)
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.ogg'))][:5]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files to process\n")
    temp_files = []
    results = []
    
    for i, filename in enumerate(audio_files, 1):
        file_path = os.path.join(audio_folder, filename)
        print(f"[{i}/{len(audio_files)}] Processing: {filename}")
        
        # Convert to processable format
        temp_wav = convert_to_mono_wav(file_path, max_duration=25)
        if not temp_wav:
            print(f"❌ Could not convert {filename}")
            continue
        
        temp_files.append(temp_wav)
        
        try:
            # Transcribe
            print(f"   🎯 Transcribing...")
            transcription_result = asr_model.transcribe([temp_wav])
            
            # Extract text
            transcription_text = extract_transcription_text(transcription_result)
            
            if not transcription_text:
                print(f"   ❌ No transcription obtained")
                continue
            
            # Analyze for singing
            singing_detected = is_probably_singing(transcription_text)
            
            # Store and display results
            result = {
                'file': filename,
                'singing': singing_detected,
                'text': transcription_text[:150] + "..." if len(transcription_text) > 150 else transcription_text
            }
            results.append(result)
            
            # Display result
            status = "🎵 SINGING" if singing_detected else "🗣️  SPEECH"
            print(f"   {status}")
            print(f"   Text: {result['text']}\n")
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Summary
    print("=" * 60)
    print("SUMMARY RESULTS:")
    print("=" * 60)
    
    singing_count = sum(1 for r in results if r['singing'])
    speech_count = len(results) - singing_count
    
    print(f"Total processed: {len(results)}")
    print(f"🎵 Singing detected: {singing_count}")
    print(f"🗣️  Speech only: {speech_count}")
    
    if results:
        print("\nDetailed results:")
        for result in results:
            status = "🎵 SINGING" if result['singing'] else "🗣️  SPEECH"
            print(f"{status}: {result['file']}")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

if __name__ == "__main__":
    audio_folder = "/Users/margotiamanova/Desktop/PROJECTS/x-audio-content-pipeline/sample_audio"
    
    print("🦜 Parakeet-TDT Singing Detection Script (NeMo 2.4.0)")
    print("=" * 60)
    
    main(audio_folder)
