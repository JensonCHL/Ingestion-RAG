@echo off
REM Git Initialization Script for Windows

echo Initializing Git repository...

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git is not installed. Please install Git first.
    pause
    exit /b 1
)

REM Initialize repository
git init

REM Add all files
git add .

REM Create initial commit
git commit -m "Initial commit: RAG Document Processing Application"

REM Rename master to main if needed
git branch -M main

echo Git repository initialized successfully!
echo.
echo To push to GitHub:
echo 1. Create a new repository on GitHub
echo 2. Run: git remote add origin https://github.com/yourusername/your-repo-name.git
echo 3. Run: git push -u origin main
pause