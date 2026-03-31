@echo off
call .\venv\Scripts\activate
echo Starting F5-TTS WebUI...
f5-tts_infer-gradio --port 7860 --host 127.0.0.1
pause
