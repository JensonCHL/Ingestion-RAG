#!/bin/bash
# Docker Compose Installation Script for Ubuntu (Docker already installed)

echo "Installing Docker Compose..."

# Update package index
echo "Updating package index..."
sudo apt update

# Install Docker Compose
echo "Installing Docker Compose..."
sudo apt install -y docker-compose

echo "Installation completed!"
echo ""
echo "Verify the installation with:"
echo "  docker-compose --version"
echo ""
echo "Then you can run your application with:"
echo "  docker-compose up --build -d"

echo ""
echo "If the above installation fails, you can install Docker Compose manually:"
echo "  sudo curl -L \"https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose"
echo "  sudo chmod +x /usr/local/bin/docker-compose"