#!/bin/bash

# RAG Application Deployment Script

echo "Starting RAG Application Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your configuration."
    exit 1
fi

# Build and start services
echo "Building and starting services..."
docker-compose up --build -d

echo "Deployment completed!"
echo "Access your application at http://localhost:8501"
echo "To stop the application, run: docker-compose down"