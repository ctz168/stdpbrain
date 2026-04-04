@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo          类人脑AI - 增强对话模式
echo              (支持流式独白显示)
echo ============================================================
echo.
echo 特性:
echo   * 实时显示AI内心独白（潜意识）
echo   * 流式回复输出
echo   * 类人脑思维状态机
echo   * 优化的独白内容（更像人脑思考）
echo.
echo 使用说明:
echo   * 输入消息与AI对话
echo   * 输入 quit 或 exit 退出
echo.

call venv\Scripts\activate.bat
python main.py --mode chat

REM 如需使用简单模式（无独白），运行：
REM python main.py --mode chat --simple

pause
