@echo off
REM VoiceSynth Studio - Complete Backend Setup
REM ===========================================
REM This script installs all dependencies for full TTS functionality

echo.
echo ============================================
echo VoiceSynth Studio - Backend Setup
echo ============================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/6] Installing core dependencies...
pip install nvidia-ml-py librosa scipy spacy numpy soundfile sounddevice --quiet

echo [2/6] Installing ML dependencies...
pip install transformers sentencepiece peft torchaudio x_transformers rotary_embedding_torch --quiet

echo [3/6] Installing GPT-SoVITS dependencies...
pip install numba pytorch-lightning pypinyin g2p_en chardet jieba tqdm funasr cn2an modelscope --quiet
pip install fast_langdetect split_lang wordsegment jieba_fast --quiet

echo [4/6] Installing ONNX and TensorRT... 
pip install onnxruntime-gpu tensorboard --quiet

echo [5/6] Cloning TTS repositories...
if not exist "GPT-SoVITS" (
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git --depth 1
)
if not exist "fish-speech" (
    git clone https://github.com/fishaudio/fish-speech.git --depth 1
)

echo [6/6] Setting up Python paths...
echo C:\Users\fgghk\PycharmProjects\VoiceSynthesisLocal\GPT-SoVITS > venv\Lib\site-packages\voicesynth_repos.pth
echo C:\Users\fgghk\PycharmProjects\VoiceSynthesisLocal\GPT-SoVITS\GPT_SoVITS >> venv\Lib\site-packages\voicesynth_repos.pth
echo C:\Users\fgghk\PycharmProjects\VoiceSynthesisLocal\fish-speech >> venv\Lib\site-packages\voicesynth_repos.pth

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Run the app with: python main.py
echo.

pause
