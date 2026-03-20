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
if exist venv (
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
call venv\Scripts\activate.bat

REM 配置pip镜像源
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1

REM 升级pip
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1

REM 安装依赖
echo       正在安装核心依赖...
python -m pip install torch transformers huggingface_hub numpy accelerate -q
python -m pip install python-telegram-bot aiohttp pydantic python-dotenv loguru modelscope -q

if %errorlevel% neq 0 (
    echo [警告] 部分依赖安装失败，尝试继续...
)
echo       [OK] 依赖安装完成
echo.

REM ===== 第四步：下载模型 =====
echo [4/4] 下载Qwen3.5-0.8B模型（国内镜像加速）...
if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" (
    echo       [跳过] 模型已存在
) else (
    echo       正在下载模型（约1.6GB，请耐心等待）...
    echo.
    echo       方法1: ModelScope国内镜像...
    
    REM 创建下载脚本
    echo from modelscope import snapshot_download > _download_temp.py
    echo import os, shutil >> _download_temp.py
    echo print("正在从ModelScope下载...") >> _download_temp.py
    echo result = snapshot_download('Qwen/Qwen3.5-0.8B', cache_dir='models') >> _download_temp.py
    echo print(f"下载路径: {result}") >> _download_temp.py
    echo if result and not os.path.exists("models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"): >> _download_temp.py
    echo     if os.path.exists(result): >> _download_temp.py
    echo         os.makedirs("models/Qwen3.5-0.8B", exist_ok=True) >> _download_temp.py
    echo         for f in os.listdir(result): >> _download_temp.py
    echo             src = os.path.join(result, f) >> _download_temp.py
    echo             dst = os.path.join("models/Qwen3.5-0.8B", f) >> _download_temp.py
    echo             if os.path.isfile(src): >> _download_temp.py
    echo                 shutil.copy2(src, dst) >> _download_temp.py
    echo print("ModelScope下载完成") >> _download_temp.py
    
    python _download_temp.py
    del _download_temp.py 2>nul
    
    REM 检查ModelScope是否成功
    if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" (
        echo       [OK] ModelScope下载成功
    ) else (
        echo       [尝试] 方法2: HuggingFace镜像 (hf-mirror.com)...
        
        REM 创建HF镜像下载脚本
        echo import os > _download_hf.py
        echo os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" >> _download_hf.py
        echo from huggingface_hub import snapshot_download >> _download_hf.py
        echo snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="models/Qwen3.5-0.8B", local_dir_use_symlinks=False) >> _download_hf.py
        echo print("HF-Mirror下载完成") >> _download_hf.py
        
        python _download_hf.py
        del _download_hf.py 2>nul
        
        if exist "models\Qwen3.5-0.8B\model.safetensors-00001-of-00001.safetensors" (
            echo       [OK] HF-Mirror下载成功
        ) else (
            echo.
            echo [错误] 模型下载失败！
            echo.
            echo 请手动下载模型：
            echo   1. 访问 https://modelscope.cn/models/Qwen/Qwen3.5-0.8B/files
            echo   2. 下载所有文件到 models\Qwen3.5-0.8B\ 目录
            echo.
            pause
            exit /b 1
        )
    )
)
echo.

REM ===== 完成部署 =====
echo ============================================================
echo                    部署完成！
echo ============================================================
echo.
echo 运行模式选择：
echo.
echo   1. 对话模式 (chat)     - 与AI进行对话
echo   2. 生成模式 (generate) - 单次文本生成
echo   3. 评估模式 (eval)     - 系统评估
echo   4. 退出
echo.

set /p choice="请选择运行模式 [1-4]: "

if "%choice%"=="1" (
    echo.
    echo 启动对话模式...
    echo 输入 'quit' 或 'exit' 退出
    echo.
    python main.py --mode chat
    pause
) else if "%choice%"=="2" (
    echo.
    set /p input_text="请输入文本: "
    python main.py --mode generate --input "%input_text%"
    pause
) else if "%choice%"=="3" (
    echo.
    echo 启动评估模式...
    python main.py --mode eval
    pause
) else (
    echo.
    echo 使用方法：
    echo   - 对话模式: python main.py --mode chat
    echo   - 生成模式: python main.py --mode generate --input "你的文本"
    echo   - 评估模式: python main.py --mode eval
    echo.
    echo 或双击运行 对话模式.bat
    echo.
    pause
)

popd
