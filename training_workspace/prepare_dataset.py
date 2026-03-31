import os
import shutil
from pathlib import Path
import soundfile as sf

def prepare_dataset():
    """
    Helps organize and validate the dataset for F5-TTS training.
    Expected User Input Structure:
      training_workspace/
        raw_data/
           audio1.wav
           audio2.wav
           metadata.csv (audio_name|text)
    
    Output Structure (Ready for F5-TTS):
      training_workspace/dataset/
         wavs/
         metadata.csv
    """
    workspace = Path(__file__).parent
    raw_data_dir = workspace / "raw_data"
    output_dir = workspace / "dataset"
    output_wavs = output_dir / "wavs"
    
    if not raw_data_dir.exists():
        print(f"Error: '{raw_data_dir}' does not exist.")
        print("Please create 'raw_data' folder and put your audio/.csv there.")
        return

    # Create output structure
    output_wavs.mkdir(parents=True, exist_ok=True)
    
    input_csv = raw_data_dir / "metadata.csv"
    if not input_csv.exists():
        print("Error: 'metadata.csv' not found in raw_data.")
        return

    print(f"Processing dataset from {raw_data_dir}...")
    
    valid_entries = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if "|" not in line:
            continue
            
        parts = line.strip().split("|")
        audio_name = parts[0].strip()
        text = parts[1].strip()
        
        # Handle extension variations
        if not audio_name.endswith(".wav"):
            audio_name += ".wav"
            
        src_audio = raw_data_dir / audio_name
        
        if src_audio.exists():
            # Validate Audio (Check if readable)
            try:
                data, samplerate = sf.read(src_audio)
                # Ensure filename is clean for F5
                dst_audio = output_wavs / audio_name
                if not dst_audio.exists():
                    shutil.copy2(src_audio, dst_audio)
                
                valid_entries.append(f"{audio_name}|{text}")
            except Exception as e:
                print(f"Skipping corrupted file {audio_name}: {e}")
        else:
            print(f"Warning: Audio file {audio_name} not found.")

    # Write cleaned metadata
    out_csv_path = output_dir / "metadata.csv"
    with open(out_csv_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(valid_entries))
        
    print(f"\nSuccess! Processed {len(valid_entries)} items.")
    print(f"Ready for training in: {output_dir}")

if __name__ == "__main__":
    prepare_dataset()
