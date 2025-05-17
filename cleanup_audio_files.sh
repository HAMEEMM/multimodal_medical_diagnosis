#!/bin/bash

# This script addresses the wav files in the repository
# to reduce the number of pending changes

echo "Handling large number of audio files in Git..."

# First make sure our .gitignore changes are committed
git add .gitignore
git commit -m "Update .gitignore to properly exclude audio files"

# Now remove any wav files that might be tracked
echo "Removing wav files from Git tracking (but keeping them on disk)..."
git rm --cached "data/Medical Speech, Transcription, and Intent/recordings/**/*.wav" 2>/dev/null || true

# Remove specific directories from Git index but keep them locally
echo "Removing audio directories from Git index..."
git rm -r --cached "data/Medical Speech, Transcription, and Intent/recordings/test/" 2>/dev/null || true
git rm -r --cached "data/Medical Speech, Transcription, and Intent/recordings/train/" 2>/dev/null || true
git rm -r --cached "data/Medical Speech, Transcription, and Intent/recordings/validate/" 2>/dev/null || true

# Add back the directory structure with .gitkeep files
echo "Adding directory structure back with .gitkeep files..."
git add "data/Medical Speech, Transcription, and Intent/recordings/test/.gitkeep"
git add "data/Medical Speech, Transcription, and Intent/recordings/train/.gitkeep"
git add "data/Medical Speech, Transcription, and Intent/recordings/validate/.gitkeep"

echo "Audio files cleanup completed!"
echo "Run 'git status' to check the remaining changes."
