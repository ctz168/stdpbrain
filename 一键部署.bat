@echo off
chcp 65001 >nul 2>&1
title 类人脑AI - 国内一键部署脚本
color 0A

echo ============================================================
echo          类人脑双系统全闭环 AI架构 - 一键部署
echo              专为国内用户优化 - ModelScope镜像
echo ============================================================
echo.

REM 设置项目目录
pushd "%~dp0"

REM ===== 第一步：检查Python =====
echo [1/4] 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [错误] 未检测到Python！
    echo.
    echo 请先安装Python 3.10或以上版本：
    echo  - 华为镜像：https://mirrors.huaweicloud.com/python/
    echo  - 官方网站：https://www.python.org/downloads/
    echo.
    echo 安装时务必勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)
echo       [OK] Python已安装
echo.

REM ===== 第二步：创建虚拟环境 =====
echo [2/4] 创建虚拟环境...
if exist venv\Scripts\python.exe (
    echo       [跳过] 虚拟环境已存在
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败！
        pause
        exit /b 1
    )
    echo       [OK] 虚拟环境创建成功
)
echo.

REM ===== 第三步：安装依赖 =====
echo [3/4] 安装依赖包（使用清华镜像源）...
echo       这可能需要几分钟，请耐心等待...
echo.

REM 配置pip镜像源
venv\Scripts\python.exe -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

REM 升级pip
echo       [1/2] 升级pip...
venv\Scripts\python.exe -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 安装依赖
echo.
echo       [2/2] 安装依赖包...
venv\Scripts\python.exe -m pip install torch transformers huggingface_hub numpy accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
venv\Scripts\python.exe -m pip install python-telegram-bot aiohttp pydantic python-dotenv loguru modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo       [OK] 依赖安装完成
echo.

REM ===== 第四步：下载模型 =====
echo [4/4] 下载Qwen3.5-0.8B模型（国内镜像加速）...

if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" goto :model_exists

echo       正在下载模型（约1.6GB，请耐心等待）...
echo.
echo       方法1: ModelScope国内镜像...

REM 创建下载脚本
call :create_download_script

venv\Scripts\python.exe _download_temp.py
del _download_temp.py 2>nul

REM 检查ModelScope是否成功
if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" (
    echo       [OK] ModelScope下载成功
    goto :model_exists
)

echo       [尝试] 方法2: HuggingFace镜像 (hf-mirror.com)...

REM 创建HF镜像下载脚本
call :create_hf_download_script

venv\Scripts\python.exe _download_hf.py
del _download_hf.py 2>nul

if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" (
    echo       [OK] HF-Mirror下载成功
    goto :model_exists
)

echo.
echo [错误] 模型下载失败！
echo.
echo 请手动下载模型：
echo   1. 访问 https://modelscope.cn/models/Qwen/Qwen3.5-0.8B/files
echo   2. 下载所有文件到 models\Qwen3.5-0.8B\ 目录
echo.
pause
exit /b 1

:model_exists
echo       [OK] 模型已就绪
echo.

REM ===== 完成部署 =====
echo ============================================================
echo                    部署完成！
echo ============================================================
echo.
echo 运行模式选择：
echo.
echo   1. 对话模式 (chat)     - 与AI进行对话
echo   2. 生成模式 (continuous) - 连续文本生成
echo   3. 评估模式 (eval)     - 系统评估
echo   4. 退出
echo.

set /p choice="请选择运行模式 [1-4]: "

if "%choice%"=="1" goto :run_chat
if "%choice%"=="2" goto :run_generate
if "%choice%"=="3" goto :run_eval
goto :show_usage

:run_chat
echo.
echo 启动对话模式...
echo 输入 'quit' 或 'exit' 退出
echo.
venv\Scripts\python.exe main.py --mode chat
pause
goto :end

:run_generate
echo.
set /p input_text="请输入文本: "
venv\Scripts\python.exe main.py --mode generate --input "%input_text%"
pause
goto :end

:run_eval
echo.
echo 启动评估模式...
venv\Scripts\python.exe main.py --mode eval
pause
goto :end

:show_usage
echo.
echo 使用方法：
echo   - 对话模式: venv\Scripts\python.exe main.py --mode chat
echo   - 连续对话模式: venv\Scripts\python.exe main.py --mode continuous
echo   - 评估模式: venv\Scripts\python.exe main.py --mode eval
echo.
echo 或双击运行 对话模式.bat
echo.
pause
goto :end

REM ===== 子程序：创建下载脚本 =====
:create_download_script
(
echo from modelscope import snapshot_download
echo import os, shutil
echo print("正在从ModelScope下载..."^)
echo result = snapshot_download^('Qwen/Qwen3.5-0.8B', cache_dir='models'^)
echo print^(f"下载路径: {result}"^)
echo model_path = "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
echo if result and not os.path.exists^(model_path^):
echo     if os.path.exists^(result^):
echo         os.makedirs^("models/Qwen3.5-0.8B", exist_ok=True^)
echo         for f in os.listdir^(result^):
echo             src = os.path.join^(result, f^)
echo             dst = os.path.join^("models/Qwen3.5-0.8B", f^)
echo             if os.path.isfile^(src^):
echo                 shutil.copy2^(src, dst^)
echo print("ModelScope下载完成"^)
) > _download_temp.py
exit /b

REM ===== 子程序：创建HF下载脚本 =====
:create_hf_download_script
(
echo import os
echo os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
echo from huggingface_hub import snapshot_download
echo snapshot_download^("Qwen/Qwen3.5-0.8B", local_dir="models/Qwen3.5-0.8B", local_dir_use_symlinks=False^)
echo print("HF-Mirror下载完成"^)
) > _download_hf.py
exit /b

:end
popd
pause
