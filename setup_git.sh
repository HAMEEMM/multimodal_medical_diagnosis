#!/bin/bash

# This script adds files to Git following the project's organization
# Run this from the project root directory

echo "Starting Git setup for multimodal_medical_diagnosis project..."

# Add .gitignore and README changes
git add .gitignore README.md

# Add modified files
git add download_medical_speech_dataset.py requirements.txt

# Add .streamlit directory
git add .streamlit/.gitkeep

# Add new files
git add SETUP_GUIDE.md
git add test_environment.py

# Add directory structure for data
git add data/Medical\ Speech,\ Transcription,\ and\ Intent/README.md
git add data/Medical\ Speech,\ Transcription,\ and\ Intent/overview-of-recordings.csv
git add data/Medical\ Speech,\ Transcription,\ and\ Intent/recordings/test/.gitkeep
git add data/Medical\ Speech,\ Transcription,\ and\ Intent/recordings/train/.gitkeep
git add data/Medical\ Speech,\ Transcription,\ and\ Intent/recordings/validate/.gitkeep

# Add docs structure (ignoring large files per .gitignore)
git add docs/images/.gitkeep

# Add models directory
git add models/.gitkeep

# Add notebook directory
git add notebooks/.gitkeep
git add notebooks/*.ipynb

# Add the setup scripts
git add setup_git.sh setup_git.bat

# Remove deleted streamlit files that are now ignored
git rm --cached .streamlit/config.toml .streamlit/secrets.toml

echo "Git setup completed!"
echo "Run 'git status' to verify changes"
echo "To commit, run: git commit -m 'Update project structure and documentation'"
