#!/bin/bash
# Git Initialization Script

echo "Initializing Git repository..."

# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Initialize repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: RAG Document Processing Application"

# Rename master to main if needed
git branch -M main

echo "Git repository initialized successfully!"
echo ""
echo "To push to GitHub:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin https://github.com/yourusername/your-repo-name.git"
echo "3. Run: git push -u origin main"