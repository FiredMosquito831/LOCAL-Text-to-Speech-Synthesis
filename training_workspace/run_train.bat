@echo off
setlocal

:: Configuration
set WORKSPACE_DIR=%~dp0
set F5_REPO=%WORKSPACE_DIR%F5-TTS
set DATASET_NAME=romanian_custom
set RUN_NAME=romanian_f5_run

:: Paths
set TRAIN_SCRIPT=%F5_REPO%\src\f5_tts\train\train.py
set CONFIG_FILE=%F5_REPO%\src\f5_tts\configs\F5TTS_Base_train.yaml

echo --- Launching F5-TTS Training ---
echo Workspace: %WORKSPACE_DIR%

:: 1. Accelerate Launch with optimized flags for memory saving
:: Gradient accumulation is handled inside the python script/config usually, 
:: but here we launch with mixed precision (fp16) which is crucial.

accelerate launch --mixed_precision=fp16 ^
    "%TRAIN_SCRIPT%" ^
    --config_file "%CONFIG_FILE%" ^
    --dataset_name "%DATASET_NAME%" ^
    --tokenizer "char" ^
    --learning_rate 0.0001 ^
    --batch_size_per_gpu 2 ^
    --batch_size_type frame ^
    --grad_accumulation_steps 4 ^
    --max_samples 50 ^
    --logger "tensorboard" ^
    --exp_name "%RUN_NAME%"

echo.
echo Training process finished (or crashed). Check logs.
pause
