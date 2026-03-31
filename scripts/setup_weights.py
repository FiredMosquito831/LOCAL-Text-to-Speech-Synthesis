import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_gpt_sovits():
    print("Downloading GPT-SoVITS v2 weights...")
    repo = "lj1995/GPT-SoVITS"
    base_path = Path("GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained")
    base_path.mkdir(parents=True, exist_ok=True)
    
    files = [
        "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "s2G2333k.pth"
    ]
    
    for f in files:
        target = base_path / f
        if not target.exists():
            print(f"Downloading {f}...")
            hf_hub_download(repo_id=repo, filename=f"gsv-v2final-pretrained/{f}", local_dir="GPT-SoVITS/GPT_SoVITS/pretrained_models", local_dir_use_symlinks=False)

def download_fish_speech():
    print("Downloading Fish Speech 1.5 weights...")
    repo = "fishaudio/fish-speech-1.5"
    
    # We'll download the mini model weights as a default
    # Usually: firefly-gan, llama-0.5b etc.
    # For now let's just ensure the directory exists and download a few key files
    base_path = Path("fish-speech/checkpoints/fish-speech-1.5")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # In practice, users should use 'huggingface-cli download'
    # But we can try to get the essentials
    print("Fish Speech models are quite large. Recommend running:")
    print(f"huggingface-cli download {repo} --local-dir fish-speech/checkpoints/fish-speech-1.5")


def download_romanian_adapter():
    """Download Romanian F5-TTS adapter weights."""
    print("Downloading Romanian F5-TTS Adapter...")
    # Using the racai-ro/Ro-F5TTS repository (assuming HF mirror or similar structure)
    # If not on HF, we might need a direct link or different repo.
    # For now, we will try to download from a likely community repo if official isn't on HF.
    # We'll use a placeholder safe repo or the official one if known. 
    # Let's assume the user wants to use the local git repo's weights if they were there, 
    # but since they aren't, we'll try to fetch from the release.
    
    # Actually, often these are in the root or a 'checkpoints' dir.
    # We will try to download 'model_last.pt' from 'racai-ro/Ro-F5TTS' if it exists on HF.
    # If not, we'll create a placeholder for the user to manually place file.
    
    target_dir = Path("adapters/romanian")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # User requested Ro-F5TTS. Let's try downloading the finetuned model.
        # Primary candidate: Respair/F5-TTS-Ro (Community finetune)
        # Backup: Mihaiii/Ro-F5-TTS
        
        candidates = [
            ("Respair/F5-TTS-Ro", "model_last.pt"),
            ("Mihaiii/Ro-F5-TTS", "model_1200000.safetensors"),
        ]
        
        success = False
        last_error = None
        
        print(f"Created adapter directory at {target_dir}")
        
        for repo, filename in candidates:
            try:
                print(f"Attempting download from '{repo}' ({filename})...")
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False
                )
                print(f"Romanian adapter downloaded successfully from {repo}.")
                success = True
                break
            except Exception as e:
                print(f"Failed to download from {repo}: {e}")
                last_error = e
        
        if not success:
            raise last_error
            
    except Exception as e:
        print(f"Could not auto-download Romanian adapter: {e}")
        
        # Create a README
        readme_path = target_dir / "README.txt"
        readme_path.write_text(
            "Automatic download failed.\n"
            "Please manually download the adapter weights from the official repository:\n"
            "https://github.com/racai-ro/Ro-F5TTS\n\n"
            "Look for 'Releases' or a Google Drive/HuggingFace link in the README.\n"
            "Place the .pt or .safetensors file in this directory."
        )
        print(f"Please manually download the adapter from https://github.com/racai-ro/Ro-F5TTS")
        print(f"and place it in: {target_dir}")


if __name__ == "__main__":
    download_gpt_sovits()
    download_fish_speech()
    download_romanian_adapter()
