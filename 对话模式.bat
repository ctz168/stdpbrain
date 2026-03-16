@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo          类人脑AI - 增强对话模式
echo              (支持异步独白流)
echo ============================================================
echo.

call venv\Scripts\activate.bat
python main.py --mode chat

REM 如需使用简单模式，运行：
REM python main.py --mode chat --simple

pause
