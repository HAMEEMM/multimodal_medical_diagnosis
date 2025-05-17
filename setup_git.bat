@echo off
REM This script adds files to Git following the project's organization
REM Run this from the project root directory

echo Starting Git setup for multimodal_medical_diagnosis project...

REM Add .gitignore and README changes
git add .gitignore README.md

REM Add modified files
git add download_medical_speech_dataset.py requirements.txt

REM Add .streamlit directory
git add .streamlit\.gitkeep

REM Add new files
git add SETUP_GUIDE.md
git add test_environment.py

REM Add directory structure for data
git add "data\Medical Speech, Transcription, and Intent\README.md"
git add "data\Medical Speech, Transcription, and Intent\overview-of-recordings.csv"
git add "data\Medical Speech, Transcription, and Intent\recordings\test\.gitkeep"
git add "data\Medical Speech, Transcription, and Intent\recordings\train\.gitkeep"
git add "data\Medical Speech, Transcription, and Intent\recordings\validate\.gitkeep"

REM Add docs structure (ignoring large files per .gitignore)
git add docs\images\.gitkeep

REM Add models directory
git add models\.gitkeep

REM Add notebook directory
git add notebooks\.gitkeep
git add notebooks\*.ipynb

REM Add the setup scripts
git add setup_git.sh setup_git.bat

REM Remove deleted streamlit files that are now ignored
git rm --cached .streamlit/config.toml .streamlit/secrets.toml

echo Git setup completed!
echo Run 'git status' to verify changes
echo To commit, run: git commit -m "Update project structure and documentation"
