@echo off
chcp 65001 >nul
cd /d %~dp0

echo ============================================================
echo          类人脑AI - 持续独白观察模式
echo              (观察实时思维流)
echo ============================================================
echo.
echo 特性:
echo   * 后台持续生成内心独白 (每3-6秒)
echo   * 随时观察AI的思维过程
echo   * 输入消息打断独白并与AI对话
echo   * 输入 quit 或 exit 退出
echo.

call venv\Scripts\activate.bat
python main.py --mode continuous

pause
