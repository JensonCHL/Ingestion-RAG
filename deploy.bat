@echo off
REM RAG Application Deployment Script for Windows

echo Starting RAG Application Deployment...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Compose is not available. Please install Docker Desktop which includes Compose.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo Error: .env file not found!
    echo Please create a .env file with your configuration.
    pause
    exit /b 1
)

REM Build and start services
echo Building and starting services...
docker-compose up --build -d

echo Deployment completed!
echo Access your application at http://localhost:8501
echo To stop the application, run: docker-compose down
pause