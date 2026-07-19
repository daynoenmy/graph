@echo off
setlocal EnableExtensions

rem Always run relative to this script, including when it is double-clicked.
cd /d "%~dp0"

rem Defaults can be overridden before running this script, for example:
rem   set DATASETS=Brain Liver Retina
rem   set SAVE_PATH=ckpt\aaclip_patch_graph
rem   test.bat
if not defined PYTHON set "PYTHON=python"
if not defined MODEL_NAME set "MODEL_NAME=ViT-L-14-336"
if not defined SAVE_PATH set "SAVE_PATH=ckpt\aaclip_patch_graph"
if not defined IMG_SIZE set "IMG_SIZE=518"
if not defined SHOT set "SHOT=4"
if not defined BATCH_SIZE set "BATCH_SIZE=32"
if not defined SEED set "SEED=111"

if not defined TEXT_ADAPT_WEIGHT set "TEXT_ADAPT_WEIGHT=0.1"
if not defined IMAGE_ADAPT_WEIGHT set "IMAGE_ADAPT_WEIGHT=0.1"
if not defined TEXT_ADAPT_UNTIL set "TEXT_ADAPT_UNTIL=3"
if not defined IMAGE_ADAPT_UNTIL set "IMAGE_ADAPT_UNTIL=6"

if not defined PATCH_GRAPH_K set "PATCH_GRAPH_K=8"
if not defined PATCH_GRAPH_ALPHA set "PATCH_GRAPH_ALPHA=0.7"
if not defined PATCH_GRAPH_RESIDUAL_WEIGHT set "PATCH_GRAPH_RESIDUAL_WEIGHT=0.2"

if not defined DATASETS set "DATASETS=MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300"
if not defined DISABLE_PATCH_GRAPH set "DISABLE_PATCH_GRAPH=0"
if not defined DISABLE_PATCH_GRAPH_SPATIAL set "DISABLE_PATCH_GRAPH_SPATIAL=0"
if not defined RELU set "RELU=0"
if not defined VISUALIZE set "VISUALIZE=0"

set "OPTIONAL_ARGS="
if "%DISABLE_PATCH_GRAPH%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --disable_patch_graph"
if "%DISABLE_PATCH_GRAPH_SPATIAL%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --disable_patch_graph_spatial"
if "%RELU%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --relu"
if "%VISUALIZE%"=="1" set "OPTIONAL_ARGS=%OPTIONAL_ARGS% --visualize"

where "%PYTHON%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found: %PYTHON%
    echo Activate the aaclip environment or set PYTHON to python.exe.
    goto :failed
)

echo Testing patch-graph AA-CLIP
echo   save_path: %SAVE_PATH%
echo   datasets:  %DATASETS%

for %%D in (%DATASETS%) do (
    echo.
    echo Testing %%D
    "%PYTHON%" test.py ^
      --model_name "%MODEL_NAME%" ^
      --save_path "%SAVE_PATH%" ^
      --dataset "%%D" ^
      --shot %SHOT% ^
      --batch_size %BATCH_SIZE% ^
      --img_size %IMG_SIZE% ^
      --seed %SEED% ^
      --text_adapt_weight %TEXT_ADAPT_WEIGHT% ^
      --image_adapt_weight %IMAGE_ADAPT_WEIGHT% ^
      --text_adapt_until %TEXT_ADAPT_UNTIL% ^
      --image_adapt_until %IMAGE_ADAPT_UNTIL% ^
      --patch_graph_k %PATCH_GRAPH_K% ^
      --patch_graph_alpha %PATCH_GRAPH_ALPHA% ^
      --patch_graph_residual_weight %PATCH_GRAPH_RESIDUAL_WEIGHT% ^
      %OPTIONAL_ARGS%
    if errorlevel 1 goto :failed
)

echo.
echo All tests completed successfully.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 0

:failed
echo Testing failed with exit code %ERRORLEVEL%.
if "%PAUSE_ON_EXIT%"=="1" pause
exit /b 1
