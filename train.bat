@echo off
setlocal EnableExtensions

rem Always run relative to this script, including when it is double-clicked.
cd /d "%~dp0"

rem Defaults can be overridden before running this script, for example:
rem   set DATASET=Brain
rem   set SAVE_PATH=ckpt\brain_patch_graph
rem   train.bat
if not defined PYTHON set "PYTHON=python"
if not defined MODEL_NAME set "MODEL_NAME=ViT-L-14-336"
if not defined DATASET set "DATASET=VisA"
if not defined TRAINING_MODE set "TRAINING_MODE=full_shot"
if not defined SHOT set "SHOT=32"
if not defined SAVE_PATH set "SAVE_PATH=ckpt\aaclip_patch_graph"
if not defined IMG_SIZE set "IMG_SIZE=518"
if not defined SURGERY_UNTIL_LAYER set "SURGERY_UNTIL_LAYER=20"
if not defined SEED set "SEED=111"

if not defined TEXT_BATCH_SIZE set "TEXT_BATCH_SIZE=16"
if not defined IMAGE_BATCH_SIZE set "IMAGE_BATCH_SIZE=2"
if not defined TEXT_EPOCH set "TEXT_EPOCH=5"
if not defined IMAGE_EPOCH set "IMAGE_EPOCH=20"
if not defined TEXT_LR set "TEXT_LR=0.00001"
if not defined IMAGE_LR set "IMAGE_LR=0.0005"

if not defined TEXT_NORM_WEIGHT set "TEXT_NORM_WEIGHT=0.1"
if not defined TEXT_ADAPT_WEIGHT set "TEXT_ADAPT_WEIGHT=0.1"
if not defined IMAGE_ADAPT_WEIGHT set "IMAGE_ADAPT_WEIGHT=0.1"
if not defined TEXT_ADAPT_UNTIL set "TEXT_ADAPT_UNTIL=3"
if not defined IMAGE_ADAPT_UNTIL set "IMAGE_ADAPT_UNTIL=6"

if not defined PATCH_GRAPH_K set "PATCH_GRAPH_K=8"
if not defined PATCH_GRAPH_ALPHA set "PATCH_GRAPH_ALPHA=0.7"
if not defined PATCH_GRAPH_RESIDUAL_WEIGHT set "PATCH_GRAPH_RESIDUAL_WEIGHT=0.2"

if not defined DISABLE_PATCH_GRAPH set "DISABLE_PATCH_GRAPH=0"
if not defined DISABLE_PATCH_GRAPH_SPATIAL set "DISABLE_PATCH_GRAPH_SPATIAL=0"
if not defined RELU set "RELU=0"

set "OPTIONAL_ARGS="
if "%DISABLE_PATCH_GRAPH%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --disable_patch_graph"
if "%DISABLE_PATCH_GRAPH_SPATIAL%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --disable_patch_graph_spatial"
if "%RELU%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --relu"

where "%PYTHON%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found: %PYTHON%
    echo Activate the aaclip environment or set PYTHON to python.exe.
    goto :failed
)

if not exist "%SAVE_PATH%" mkdir "%SAVE_PATH%"
if errorlevel 1 goto :failed

echo Training patch-graph AA-CLIP
echo   dataset:       %DATASET%
echo   mode:          %TRAINING_MODE%
echo   save_path:     %SAVE_PATH%
echo   graph k/alpha: %PATCH_GRAPH_K% / %PATCH_GRAPH_ALPHA%

"%PYTHON%" train.py ^
  --model_name "%MODEL_NAME%" ^
  --dataset "%DATASET%" ^
  --training_mode "%TRAINING_MODE%" ^
  --shot %SHOT% ^
  --save_path "%SAVE_PATH%" ^
  --img_size %IMG_SIZE% ^
  --surgery_until_layer %SURGERY_UNTIL_LAYER% ^
  --seed %SEED% ^
  --text_batch_size %TEXT_BATCH_SIZE% ^
  --image_batch_size %IMAGE_BATCH_SIZE% ^
  --text_epoch %TEXT_EPOCH% ^
  --image_epoch %IMAGE_EPOCH% ^
  --text_lr %TEXT_LR% ^
  --image_lr %IMAGE_LR% ^
  --text_norm_weight %TEXT_NORM_WEIGHT% ^
  --text_adapt_weight %TEXT_ADAPT_WEIGHT% ^
  --image_adapt_weight %IMAGE_ADAPT_WEIGHT% ^
  --text_adapt_until %TEXT_ADAPT_UNTIL% ^
  --image_adapt_until %IMAGE_ADAPT_UNTIL% ^
  --patch_graph_k %PATCH_GRAPH_K% ^
  --patch_graph_alpha %PATCH_GRAPH_ALPHA% ^
  --patch_graph_residual_weight %PATCH_GRAPH_RESIDUAL_WEIGHT% ^
  %OPTIONAL_ARGS%

if errorlevel 1 goto :failed

echo Training completed successfully.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:failed
echo Training failed with exit code %ERRORLEVEL%.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
