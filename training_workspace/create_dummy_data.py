import os
import wave
import random
import struct
from pathlib import Path

def create_dummy_dataset():
    """
    Creates a dummy dataset in 'training_workspace/raw_data' 
    to test the training pipeline without finding real data.
    """
    workspace = Path(__file__).parent
    raw_data_dir = workspace / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating dummy data in {raw_data_dir}...")
    
    metadata_lines = []
    
    # Generate 5 dummy files
    for i in range(1, 6):
        filename = f"dummy_{i}.wav"
        filepath = raw_data_dir / filename
        
        # specific dummy text
        text = f"This is dummy sentence number {i} for testing purposes."
        
        # Create a random noise/tone wav file (1-3 seconds)
        duration = random.uniform(1.0, 3.0)
        sample_rate = 24000 # F5 default
        n_samples = int(sample_rate * duration)
        
        with wave.open(str(filepath), 'w') as wav_file:
            wav_file.setnchannels(1) # Mono
            wav_file.setsampwidth(2) # 2 bytes (16-bit)
            wav_file.setframerate(sample_rate)
            
            # Write random frames (noise)
            # Generating 16-bit PCM data (random bytes)
            # This is just for VRAM testing, doesn't need to sound good
            data = os.urandom(n_samples * 2) 
            wav_file.writeframes(data)
            
        metadata_lines.append(f"{filename}|{text}")
        print(f"Created {filename} ({duration:.2f}s)")

    # Write metadata
    csv_path = raw_data_dir / "metadata.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(metadata_lines))
        
    print("Dummy dataset created.")
    print("Now run 'prepare_dataset.py' then 'run_train.bat'")

if __name__ == "__main__":
    create_dummy_dataset()
