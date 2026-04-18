@echo off
chcp 65001 >nul
cd /d "%~dp0"
call install.bat < _input.txt
