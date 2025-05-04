#!/bin/bash  

# Script to create dissertation project directory structure  

# Set the main project directory  
PROJECT_NAME="medical-diagnosis-nlp-dl"  

# Create main project directory  
echo "Creating project directory: $PROJECT_NAME"  
mkdir -p $PROJECT_NAME  
cd $PROJECT_NAME  

# Create data directory structure matching the Kaggle dataset organization  
echo "Creating data directory structure..."  
mkdir -p data  
touch data/overview-of-recordings.csv  # Placeholder for the CSV file  

# Create recording folder with test, train, and validate subfolders  
mkdir -p data/recording/test  
mkdir -p data/recording/train  
mkdir -p data/recording/validate  

# Create processed data directory for transformed datasets  
mkdir -p data/processed  

# Create notebooks directory  
echo "Creating notebooks directory..."  
mkdir -p notebooks  

# Create notebook files  
touch notebooks/1_data_exploration.ipynb  
touch notebooks/2_text_preprocessing.ipynb  
touch notebooks/3_audio_preprocessing.ipynb  
touch notebooks/4_text_classification.ipynb  
touch notebooks/5_audio_classification.ipynb  
touch notebooks/6_evaluation.ipynb  

# Create src directory and subdirectories  
echo "Creating src directory structure..."  
mkdir -p src/data_processing  
mkdir -p src/feature_extraction  
mkdir -p src/models  
mkdir -p src/visualization  

# Add __init__.py files  
touch src/__init__.py  
touch src/data_processing/__init__.py  
touch src/feature_extraction/__init__.py  
touch src/models/__init__.py  
touch src/visualization/__init__.py  

# Create results directory  
echo "Creating results directory..."  
mkdir -p results/figures  
mkdir -p results/tables  
mkdir -p results/models  

# Create docs directory  
echo "Creating docs directory..."  
mkdir -p docs  

# Create README file  
cat > README.md << 'EOF'  
# NLP and Deep Learning for Text and Audio Classification in Medical Diagnosis  

## Overview  
This repository contains the code, data, and results for my PhD dissertation on applying Natural Language Processing (NLP) and Deep Learning techniques to analyze text and audio data for medical diagnosis classification.  

## Dataset  
The study uses the [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) dataset from Kaggle, which contains:  
- "overview-of-recordings.csv" file with metadata about the recordings  
- "recording" folder with audio files organized in train, test, and validate subfolders  

## Repository Structure  
- `/data`: Contains the dataset files  
- `/notebooks`: Jupyter notebooks demonstrating the analysis pipeline  
- `/src`: Source code modules for data processing, feature extraction, and modeling  
- `/results`: Output figures, tables, and model performance metrics  
- `/docs`: Documentation and change tracking  

## Requirements  
To run the code in this repository, install the required dependencies: