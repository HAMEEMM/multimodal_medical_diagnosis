# Multimodal Medical Diagnosis Project Setup Guide

This guide provides step-by-step instructions for setting up the development environment for the multimodal medical diagnosis project. The environment uses Python 3.9.6 and incorporates all necessary libraries for multimodal (audio, text, image) medical data analysis and classification.

## Environment Overview

### Environment Specifications

- Python version: 3.9.6
- Virtual Environment: `.medical_diagnosis` (created with venv)
- Compatible with Windows, macOS, and Linux

### Key Packages

The environment includes the following:

**Core Data Science**

- NumPy 1.23.5
- Pandas 2.0.2
- SciPy 1.10.1
- Matplotlib 3.7.1, Seaborn 0.12.2, Plotly 5.14.1

**Machine Learning & Deep Learning**

- Scikit-Learn 1.2.2
- XGBoost 1.7.5
- TensorFlow 2.12.0 (CPU version)
- Keras 2.12.0
- Imbalanced-Learn 0.12.4

**Natural Language Processing**

- NLTK 3.8.1
- TextBlob 0.17.1
- spaCy 3.5.0 with en_core_web_sm model
- Transformers 4.29.2
- Gensim 4.3.0
- Wordcloud 1.9.2

**Audio Processing**

- Librosa 0.10.0
- SoundFile 0.12.1
- SpeechPy 2.4 (replacement for deprecated python-speech-features)

**Development Tools**

- Jupyter 1.0.0
- IPyWidgets 8.0.6
- Streamlit 1.22.0
- TQDM 4.65.0

## Prerequisites

1. Git installed on your system
2. Python 3.9.6 installed on your system
   - Download from: https://www.python.org/downloads/release/python-396/
   - Make sure to select "Add Python to PATH" during installation
3. VS Code installed
   - Download from: https://code.visualstudio.com/download

## Step 1: Clone the Repository

Open a terminal/command prompt and run:

```bash
git clone https://github.com/HAMEEMM/multimodal_medical_diagnosis.git
cd multimodal_medical_diagnosis
```

## Step 2: Set Up VS Code

1. Open VS Code
2. Install recommended extensions:

   - Python extension (ms-python.python)
   - Jupyter (ms-toolsai.jupyter)
   - Pylance (ms-python.vscode-pylance)

   You can install these by pressing `Ctrl+Shift+X`, searching for each extension, and clicking "Install".
3. Open the project folder in VS Code:

   - File → Open Folder → select the `multimodal_medical_diagnosis` folder

## Step 3: Create and Activate Virtual Environment

### On Windows:

```bash
# Create a virtual environment
python -m venv .medical_diagnosis

# Activate the virtual environment
.medical_diagnosis/Scripts/activate
```

### On macOS/Linux:

```bash
# Create a virtual environment
python -m venv .medical_diagnosis

# Activate the virtual environment
source .medical_diagnosis/bin/activate
```

## Step 4: Install Dependencies

```bash
# Install pip packages from the requirements file
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

Note: If you encounter any issues with the pip installation:

1. Make sure pip is up to date: `python -m pip install --upgrade pip`
2. If there are issues with specific packages, you can try installing them individually

## Step 5: Verify the Installation

Run the environment test script to ensure all packages are correctly installed:

```bash
python test_environment.py
```

This script will test the functionality of all key libraries and report any issues.

## Step 6: Configure VS Code to Use the Virtual Environment

1. In VS Code, press `Ctrl+Shift+P` to open the command palette
2. Type and select "Python: Select Interpreter"
3. Choose the Python interpreter from your `.medical_diagnosis` virtual environment

## Step 7: Run Jupyter Notebooks

Once the environment is set up, you can run the project notebooks:

```bash
jupyter notebook
```

This will open Jupyter in your browser, where you can navigate to the `notebooks` directory and open any of the project notebooks.

Alternatively, you can open the notebooks directly in VS Code by clicking on them in the file explorer.

## Important Notes and Technical Considerations

- JAX was downgraded to version 0.4.13 for compatibility with Python 3.9.6
- SpeechPy 2.4 was used as a replacement for the deprecated python-speech-features library
- All packages were verified to work together without conflicts

## Common Issues and Solutions

### Package Installation Errors

If you encounter errors during package installation:

1. For TensorFlow-related errors:

   ```bash
   pip install tensorflow==2.12.0
   ```
2. For JAX-related errors:

   ```bash
   pip install jax==0.4.13 jaxlib==0.4.13
   ```

### Python Version Mismatch

This project requires Python 3.9.6. If you have multiple Python versions installed, make sure to create the virtual environment using Python 3.9.6:

```bash
# On Windows
py -3.9 -m venv .medical_diagnosis

# On macOS/Linux
python3.9 -m venv .medical_diagnosis
```

### Audio Processing Libraries on Windows

If you experience issues with audio libraries on Windows:

```bash
pip install librosa==0.10.0 --no-deps
pip install soundfile==0.12.1
```

## Project Structure

- `data/`: Contains the medical speech dataset and other data files
- `docs/`: Project documentation and research papers
- `notebooks/`: Jupyter notebooks for data exploration and model development
- `test_environment.py`: Script to test the environment setup
- `requirements.txt`: Package dependencies with exact versions for reproducibility

## Dataset Information

The project contains a medical speech dataset with recordings in the `data/Medical Speech, Transcription, and Intent/recordings/` directory organized into train, test, and validation sets.

## Additional Resources

- Documentation for key libraries:
  - TensorFlow: https://www.tensorflow.org/api_docs
  - spaCy: https://spacy.io/usage
  - Librosa: https://librosa.org/doc/latest/index.html

If you encounter any issues not covered in this guide, please reach out to Mahdi.Hameem@gmail.com
