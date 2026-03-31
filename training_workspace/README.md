# Training F5-TTS (Romanian)

This folder contains everything you need to train a custom F5-TTS model (e.g., for Romanian).

## 1. Setup
Run the setup script to download the F5-TTS code and install dependencies.
```bash
python setup_training.py
```

## 2. Prepare Dataset
1.  Create a folder `raw_data/` inside `training_workspace/`.
2.  Put your `.wav` files and a `metadata.csv` file inside `raw_data/`.
    *   Format of `metadata.csv`: `filename|transcript`
    *   Example: `audio_01|Salut, ce mai faci?`
3.  Run the preparation script:
    ```bash
    python prepare_dataset.py
    ```
    This validates your audio and creates a cleaned `dataset/` folder ready for F5-TTS.

## 3. Train
Run the training launcher:
```bash
run_train.bat
```
*   This uses `accelerate` (mixed precision `fp16`) to save memory.
*   Edit `run_train.bat` to change batch size if you run out of VRAM.
