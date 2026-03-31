import os
import subprocess
import sys
from pathlib import Path

def setup_training_env():
    """
    Sets up the F5-TTS training environment.
    1. Clones the F5-TTS repository.
    2. Installs requirements.
    """
    workspace_dir = Path(__file__).parent.absolute()
    f5_repo_dir = workspace_dir / "F5-TTS"

    print(f"--- Setting up Training Environment in {workspace_dir} ---")

    # 1. Clone F5-TTS
    if not f5_repo_dir.exists():
        print("Cloning F5-TTS repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/SWivid/F5-TTS.git"],
                cwd=str(workspace_dir),
                check=True
            )
            print("Successfully cloned F5-TTS.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            return
    else:
        print("F5-TTS repository already exists. Skipping clone.")

    # 2. Install Dependencies (accelerate, etc. in addition to F5 requirements)
    print("Installing training dependencies...")
    requirements = [
        "accelerate",
        "bitsandbytes",  # For 8-bit optimizer if needed
        "scipy",
        "soundfile",
        "librosa"
    ]
    
    # We also need to install the F5-TTS package in editable mode or install its reqs
    # pip install -e F5-TTS/
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + requirements,
            check=True
        )
        print("Installed base training tools.")
        
        print("Installing F5-TTS requirements and package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=str(f5_repo_dir),
            check=True
        )
        print("Successfully installed F5-TTS in editable mode.")

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return

    print("\n--- Setup Complete ---")
    print(f"F5-TTS is located at: {f5_repo_dir}")
    print("You can now organize your dataset in 'training_workspace/dataset/'")

if __name__ == "__main__":
    setup_training_env()
