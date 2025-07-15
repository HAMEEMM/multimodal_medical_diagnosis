# 🏥 Multimodal Medical Diagnosis System

[![Python](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Hypotheses &amp; Questions](#-research-hypotheses--questions)
- [Dataset Information](#-dataset-information)
- [System Architecture](#-system-architecture)
- [Installation &amp; Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Implementation Details](#-implementation-details)
- [Results &amp; Performance](#-results--performance)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [Author &amp; Contact](#-author--contact)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## 🔬 Overview

This project implements a comprehensive **Multimodal Medical Diagnosis System** through three distinct classification approaches that leverage **Text Analysis**, **Audio Analysis**, and **Combined Audio+Text Analysis** to assist healthcare providers in clinical decision-making. The system analyzes patient symptom descriptions through voice recordings and transcribed text data to predict medical conditions with varying degrees of accuracy depending on the modality used.

> **🛠️ Tested Environment**: This project has been successfully tested and deployed with **Python 3.12.10** in a virtual environment (`.medical_diagnosis`) on **Windows 11**. All dependencies are automatically managed through the virtual environment for reproducible results.

### 🎯 Key Features

- **📝 Text Classification**: Advanced NLP using TF-IDF, lemmatization, and deep learning models achieving 99.55% F1-Score
- **🎤 Audio Classification**: Acoustic feature extraction (MFCCs, spectral, temporal features) with traditional ML models achieving 39.22% F1-Score
- **🔀 Multimodal (Audio + Text) Classification**: Combined analysis achieving 99.55% F1-Score for enhanced diagnostic accuracy
- **🤖 Multiple Algorithm Support**: Support Vector Machines, Logistic Regression, Naive Bayes, CNN, Feedforward Neural Networks
- **📊 Comprehensive Evaluation**: Advanced metrics including accuracy, precision, recall, F1-score, Cohen's Kappa, Matthews Correlation
- **⚡ Clinical Validation**: Rigorous testing against clinical deployment thresholds (75% minimum performance)
- **📈 Interactive Visualizations**: Plotly-based charts, confusion matrices, and performance analysis
- **🔬 Research-Grade Implementation**: Three separate Jupyter notebooks for reproducible research

### 📚 Three Notebook Implementation

1. **Text Classification** (`medical_diagnosis_text_classification.ipynb`)

   - Pure NLP approach to symptom text analysis
   - Advanced text preprocessing and feature engineering
   - Multiple traditional ML and deep learning models
   - **Result**: 99.55% F1-Score (Clinical threshold: PASSED)
2. **Audio Classification** (`medical_diagnosis_audio_classification.ipynb`)

   - Pure acoustic feature extraction and analysis
   - Comprehensive audio signal processing
   - Traditional ML models optimized for audio features
   - **Result**: 39.22% F1-Score (Clinical threshold: FAILED)
3. **Multimodal Classification** (`medical_diagnosis_audio_and_text_classification.ipynb`)

   - Combined audio and text feature analysis
   - Feature fusion and multimodal learning
   - Enhanced performance through complementary modalities
   - **Result**: 99.55% F1-Score (Clinical threshold: PASSED)

### 🏥 Clinical Applications

- **Primary Care**: Initial symptom assessment and triage (Text/Multimodal recommended)
- **Telemedicine**: Remote patient evaluation with transcribed consultations
- **Emergency Medicine**: Rapid symptom classification using available data modalities
- **Healthcare Analytics**: Population-level health monitoring and trend analysis
- **Medical Education**: Training tools demonstrating different diagnostic approaches

## 🔬 Research Hypotheses & Questions

### 1. Text Classification

**RQ1:** What is the effectiveness of NLP algorithms in classifying patient symptoms from text data on the population level?

- **H10 (Null):** Text analysis of patient symptoms results in insufficient precision and recall for provider decision support.
- **H1a (Alternative):** Text analysis of patient symptoms results in precision and recall sufficient for provider decision support.

### 2. Audio Classification

**RQ2:** How effective is NLP in classifying patient symptoms from audio data on the population level?

- **H20 (Null):** Audio analysis of patient symptoms yields both precision and recall metrics that are insufficient for effective provider decision support.
- **H2a (Alternative):** Audio analysis of patient symptoms results in precision and recall sufficient for provider decision support.

### 3. Multimodal (Audio + Text) Classification

**RQ3:** How effective is NLP in classifying patient symptoms from audio and text data on the population level?

- **H30 (Null):** Audio and text analysis of patient symptoms yields both precision and recall metrics that are insufficient for effective provider decision support.
- **H3a (Alternative):** Audio and text analysis of patient symptoms results in precision and recall sufficient for provider decision support.

### Performance Thresholds

- **Minimum Acceptable:** 75% precision, recall, and F1-score
- **High Performance:** 85% precision, recall, and F1-score
- **Clinical Deployment:** All metrics must exceed minimum threshold

## 📊 Dataset Information

### Source

- **Dataset**: Medical Speech, Transcription, and Intent Dataset
- **Platform**: Kaggle
- **Owner**: Paul Mooney
- **URL**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

### Dataset Characteristics

- **Total Records**: 6,661 medical recordings with transcriptions
- **Audio Files**:
  - Format: WAV files
  - Quality ratings: 3.0-4.67 scale
  - Background noise levels: Various (no_noise, light_noise)
  - Audio clipping status: Tracked for quality control
  - Total Files: 6,661 recordings across train/test/validation splits
- **Text Data**:
  - Patient symptom descriptions (transcribed from audio)
  - Medical diagnostic categories
  - Character length: Variable complexity
  - Word count: Variable from short phrases to detailed descriptions
  - Total Samples: 6,661 entries
- **Medical Categories**: Multiple diagnostic categories including:
  - **Pain-related**: Back pain, Head ache, Knee pain, Shoulder pain, Neck pain, Muscle pain
  - **Respiratory**: Hard to breath
  - **Digestive**: Stomach ache
  - **Skin conditions**: Skin issue, Acne, Hair falling out
  - **Systemic**: Body feels weak, Feeling dizzy, Emotional pain
  - **Injuries**: Infected wound, Injury from sports, Open wound
  - **Vision**: Blurry vision
  - **Cardiovascular**: Heart hurts
  - **Neurological**: Internal pain
  - **Joint-related**: Joint pain, Foot ache
  - And more specialized categories

### Data Variables by Classification Type

#### Text Classification Variables

- **phrase**: Text transcriptions containing patient symptom descriptions
- **prompt**: Corresponding medical diagnosis/category for each text entry

#### Audio Classification Variables

- **file_name**: Audio file identifier and path to WAV files
- **prompt**: Corresponding medical diagnosis/category for each audio sample

#### Multimodal (Audio + Text) Classification Variables

- **phrase**: Text transcriptions for text component analysis
- **file_name**: Audio files for acoustic feature extraction
- **prompt**: Target diagnostic categories for both modalities

## 🏗️ System Architecture

The system implements three distinct classification pipelines, each optimized for different input modalities:

### Three-Notebook Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER - MULTIMODAL DATA                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                    Text Data                │           Audio Files             │
│              (Symptom Descriptions)         │         (.wav recordings)         │
│                                             │                                   │
│    ┌─────────────────────────────────────┐  │  ┌─────────────────────────────┐  │
│    │      TEXT CLASSIFICATION            │  │  │    AUDIO CLASSIFICATION     │  │
│    │         (Notebook 1)                │  │  │       (Notebook 2)          │  │
│    │                                     │  │  │                             │  │
│    │  • Text Preprocessing               │  │  │  • Audio Feature Extraction │  │
│    │  • TF-IDF Vectorization             │  │  │  • MFCCs (74 features)      │  │
│    │  • Tokenization & Cleaning          │  │  │  • Spectral Analysis        │  │
│    │  • Stopword Removal                 │  │  │  • Temporal Features        │  │
│    │                                     │  │  │  • Harmonic Analysis        │  │
│    │  Models:                            │  │  │                             │  │
│    │  • Traditional ML:                  │  │  │  Models:                    │  │
│    │    - SVM, Random Forest             │  │  │  • Traditional ML:          │  │
│    │    - Logistic Regression            │  │  │    - SVM, Random Forest     │  │
│    │    - Naive Bayes                    │  │  │    - Logistic Regression    │  │
│    │  • Deep Learning:                   │  │  │  • Deep Learning:           │  │
│    │    - CNN, FNN                       │  │  │    - CNN, FNN               │  │
│    └─────────────────────────────────────┘  │  └─────────────────────────────┘  │
│                           │                  │                │                 │
│                           ▼                  │                ▼                 │
│    ┌─────────────────────────────────────────────────────────────────────────┐  │
│    │                   MULTIMODAL CLASSIFICATION                             │  │
│    │                        (Notebook 3)                                     │  │
│    │                                                                         │  │
│    │  • Combined Feature Processing                                          │  │
│    │  • Text + Audio Feature Fusion                                          │  │
│    │  • Joint Representation Learning                                        │  │
│    │  • Multimodal Deep Learning Models                                      │  │
│    │                                                                         │  │
│    │  Models:                                                                │  │
│    │  • Traditional ML: SVM, Random Forest, Logistic Regression              │  │
│    │  • Deep Learning: CNN, FNN, LSTM                                        │  │
│    │  • Ensemble Methods: Voting, Stacking                                   │  │
│    └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION & COMPARISON                                 │
│                                                                                 │
│  • Cross-Validation (5-fold)              • Advanced Metrics                    │
│  • Performance Comparison                 • Cohen's Kappa                       │
│  • Statistical Significance Testing       • Matthews Correlation                │
│  • Confusion Matrix Analysis              • AUC-ROC Analysis                    │
│  • Clinical Threshold Assessment          • Per-Class Performance               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      CLINICAL DECISION SUPPORT                                  │
│                                                                                 │
│  • Real-time Inference                    • Confidence Scoring                  │
│  • Model Deployment Pipeline              • Performance Monitoring              │
│  • Clinical Integration                   • Quality Assurance                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Components

1. **Modular Design**: Each classification approach (text, audio, multimodal) operates independently
2. **Feature Engineering**: Specialized preprocessing for each data type
3. **Model Comparison**: Multiple algorithms evaluated per modality
4. **Performance Validation**: Rigorous testing against clinical thresholds (75% minimum)
5. **Scalable Deployment**: Optimized for clinical integration

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.12.10 (Active Version in Virtual Environment)
- **Virtual Environment**: `.medical_diagnosis` (Pre-configured and Active)
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **Jupyter Kernel**: Python 3.12.10 configured for notebook execution

### Environment Setup

> **Note**: The project environment is already configured. The `.medical_diagnosis` virtual environment with Python 3.12.10 is active and ready to use.

1. **Repository Access**

```bash
# Navigate to the project directory
cd "d:\Msc\NCU\Doctoral Record\multimodal_medical_diagnosis"
```

2. **Virtual Environment** (Already Configured)

```bash
# Current active environment: .medical_diagnosis
# Python version: 3.12.10
# Status: ✅ Active and configured

# To verify environment status:
python --version  # Should show: Python 3.12.10

# Environment is automatically activated - look for (.medical_diagnosis) in terminal prompt
```

3. **Package Management** (Auto-managed)

```bash
# All packages are installed and up-to-date
# To view installed packages:
pip list

# To install additional packages (if needed):
pip install package_name

# All dependencies are automatically resolved through the active environment
```

### Environment Verification

After installation, verify your environment setup:

```bash
# Check Python version (should be 3.12.10)
python --version

# Verify virtual environment is activated (should show environment name)
echo $VIRTUAL_ENV  # Linux/macOS
echo $env:VIRTUAL_ENV  # Windows PowerShell

# Test key package imports
python -c "import numpy, pandas, tensorflow, librosa, sklearn; print('All packages imported successfully!')"

# Check TensorFlow version and GPU availability (if applicable)
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Current Environment Status

✅ **Environment Configuration**:

- **Python Version**: 3.12.10 ✓
- **Virtual Environment**: `.medical_diagnosis` (Active) ✓
- **Jupyter Kernel**: Python 3.12.10 configured ✓
- **Package Dependencies**: All major packages updated and verified ✓

✅ **Key Package Versions** (Currently Installed):

- **TensorFlow**: 2.19.0 (Latest)
- **Scikit-learn**: 1.6.1 (Latest)
- **NumPy**: 2.1.3 (Latest)
- **Pandas**: 2.3.1 (Latest)
- **Librosa**: 0.11.0 (Latest)
- **Jupyter**: 1.1.1 with JupyterLab 4.4.4

### Virtual Environment Status Indicators

**✅ Environment Activated:**

- Terminal prompt shows `(.medical_diagnosis)` prefix
- Example: `(.medical_diagnosis) PS D:\your\project\path>`

**❌ Environment Not Activated:**

- No prefix in terminal prompt
- Example: `PS D:\your\project\path>`

**To Activate Environment:**

```bash
# Windows PowerShell
.\.medical_diagnosis\Scripts\Activate.ps1

# Windows Command Prompt  
.\.medical_diagnosis\Scripts\activate.bat

# Linux/macOS
source .medical_diagnosis/bin/activate
```

### Required Libraries (Current Installation)

```python
# Core Data Science Libraries (Currently Installed)
numpy==2.1.3                  # Multi-dimensional array computing
pandas==2.3.1                 # Data manipulation and analysis library
scipy==1.16.0                 # Scientific computing library

# Machine Learning Libraries
scikit-learn==1.6.1           # Machine learning algorithms and tools
xgboost==3.0.2                # Gradient boosting framework
imbalanced-learn==0.13.0      # Tools for handling imbalanced datasets

# Deep Learning
tensorflow==2.19.0            # Deep learning framework
keras==3.10.0                 # High-level neural networks API

# Audio Processing
librosa==0.11.0               # Audio and music analysis library
soundfile==0.13.1             # Audio file reading/writing
noisereduce==3.0.3            # Audio noise reduction

# Visualization
matplotlib==3.10.3            # Comprehensive plotting and visualization library
seaborn==0.13.2               # Statistical data visualization
plotly==6.2.0                 # Interactive visualization library
pillow==11.3.0                # Python Imaging Library (PIL) for image processing

# Development and Notebook Tools
jupyter==1.1.1                # Interactive computing environment
jupyterlab==4.4.4             # JupyterLab interface
ipykernel==6.29.5             # Jupyter kernel for Python
ipywidgets==8.1.7             # Interactive widgets for Jupyter notebooks
tqdm==4.67.1                  # Progress bar for loops and data processing

# Utilities
joblib==1.5.1                 # Lightweight pipelining in Python
requests==2.32.4              # HTTP library for data downloading
psutil==7.0.0                 # System and process utilities
```

### Virtual Environment Information

- **Environment Name**: `.medical_diagnosis`
- **Python Version**: 3.12.10
- **Environment Type**: Virtual Environment
- **Activation Status**: ✅ Active
- **Package Management**: pip-based with automatic dependency resolution
- **Jupyter Kernel**: Configured for notebook execution

### Jupyter Notebook Configuration

The project notebooks are configured to use the Python 3.12.10 kernel from the `.medical_diagnosis` virtual environment:

```bash
# Start Jupyter Notebook (recommended)
jupyter notebook

# Or start JupyterLab (alternative interface)
jupyter lab

# Notebooks are located in the notebooks/ directory:
# - medical_diagnosis_text_classification.ipynb
# - medical_diagnosis_audio_classification.ipynb  
# - medical_diagnosis_audio_and_text_classification.ipynb
```

**Kernel Information**:

- **Kernel Name**: Python 3.12.10 (.medical_diagnosis)
- **Environment**: `.medical_diagnosis` virtual environment
- **Python Path**: `D:/Msc/NCU/Doctoral Record/multimodal_medical_diagnosis/.medical_diagnosis/Scripts/python.exe`
- **Package Access**: All installed packages available automatically

## 🚀 Usage Guide

### Quick Start

1. **Data Preparation**

```python
# Place your dataset in the data directory
data/
├── Medical Speech, Transcription, and Intent/
│   ├── overview-of-recordings.csv
│   └── recordings/
│       ├── test
│       ├── train
│       └── validate
```

2. **Run Text Classification**

```python
# Execute the text classification notebook
jupyter notebook notebooks/medical_diagnosis_text_classification.ipynb
```

3. **Run Audio Classification**

```python
# Execute the audio classification notebook
jupyter notebook notebooks/medical_diagnosis_audio_classification.ipynb
```

### Advanced Usage

#### Custom Model Training

```python
from src.models.text_classifier import TextClassifier
from src.models.audio_classifier import AudioClassifier

# Initialize models
text_model = TextClassifier(model_type='bert')
audio_model = AudioClassifier(model_type='cnn')

# Train models
text_model.train(X_text, y_text)
audio_model.train(X_audio, y_audio)

# Evaluate performance
text_results = text_model.evaluate(X_test_text, y_test_text)
audio_results = audio_model.evaluate(X_test_audio, y_test_audio)
```

#### Multimodal Prediction

```python
from src.multimodal.fusion import MultimodalPredictor

# Combine text and audio predictions
predictor = MultimodalPredictor(text_model, audio_model)
combined_prediction = predictor.predict(text_description, audio_file)
```

## 🔧 Implementation Details

### Text Classification Pipeline

1. **Text Preprocessing**

   - **Basic Cleaning**: Converts to lowercase, expands contractions (what's → what is, can't → cannot)
   - **Advanced NLP**: NLTK tokenization, lemmatization using WordNetLemmatizer
   - **Stopword Removal**: Filters common words while preserving medical terms (pain, ache, fever, swelling, rash)
   - **Feature Engineering**: TF-IDF vectorization with medical domain optimization
2. **Model Architectures**

   - **Traditional ML**: Support Vector Machine, Naive Bayes, Logistic Regression
   - **Deep Learning**:
     - **CNN**: Convolutional layers with text embeddings, GlobalMaxPooling1D
     - **Feedforward NN**: Dense layers with dropout for regularization
   - **Best Performance**: CNN achieving 99.55% F1-Score
3. **Evaluation Framework**

   - **Metrics**: Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, Matthews Correlation
   - **Cross-validation**: Stratified K-fold for robust performance estimation
   - **Threshold Analysis**: Clinical deployment threshold (75% minimum)

### Audio Classification Pipeline

1. **Audio Feature Extraction**

   - **MFCCs**: 13 Mel-frequency cepstral coefficients capturing spectral envelope
   - **Spectral Features**:
     - Spectral centroid (brightness)
     - Spectral bandwidth (frequency spread)
     - Spectral rolloff (frequency concentration)
   - **Temporal Features**: Zero crossing rate for speech analysis
   - **Harmonic Features**: Chroma features for tonal content
   - **Tonnetz Features**: Harmonic network analysis
   - **Statistical Features**: Mean, standard deviation, skewness, kurtosis for each feature
2. **Data Preprocessing**

   - **Audio Loading**: Librosa for consistent WAV file processing
   - **Feature Standardization**: StandardScaler for normalized feature ranges
   - **Quality Control**: Audio quality and noise level assessment
3. **Model Architectures**

   - **Traditional ML**: Support Vector Machine, Logistic Regression
   - **Feature Selection**: Comprehensive acoustic feature vector
   - **Best Performance**: SVM achieving 39.22% F1-Score (below clinical threshold)

### Multimodal (Audio + Text) Classification Pipeline

1. **Feature Integration**

   - **Text Component**: Full NLP preprocessing pipeline from text classification
   - **Audio Component**: Complete acoustic feature extraction from audio classification
   - **Feature Fusion**: Combined feature vector leveraging both modalities
   - **Dimensionality Management**: Optimized feature selection for computational efficiency
2. **Model Architecture**

   - **CNN for Audio**: Specialized convolutional layers for acoustic pattern recognition
   - **Enhanced Performance**: Multimodal approach improves upon individual modalities
   - **Best Performance**: CNN achieving 99.55% F1-Score
3. **Integration Strategy**

   - **Early Fusion**: Feature-level combination before model training
   - **Balanced Weighting**: Equal consideration of text and audio features
   - **Robust Validation**: Cross-modal performance verification

### Evaluation Methodology

#### Performance Metrics

- **Primary**: Accuracy, Precision, Recall, F1-Score
- **Advanced**: Cohen's Kappa (inter-rater agreement), Matthews Correlation Coefficient
- **Clinical**: Sensitivity, Specificity for medical deployment

#### Statistical Analysis

- **Threshold Testing**: Minimum acceptable (75%), High performance (85%)
- **Cross-validation**: Stratified sampling for class balance
- **Significance Testing**: Statistical validation of performance differences

#### Research Hypothesis Framework

- **Text Classification (RQ1)**: H1a ACCEPTED - 99.55% F1-Score exceeds threshold
- **Audio Classification (RQ2)**: H20 ACCEPTED - 39.22% F1-Score below threshold
- **Multimodal Classification (RQ3)**: H3a ACCEPTED - 99.55% F1-Score exceeds threshold

## 📈 Results & Performance

### Text Classification Results (RQ1)

**Research Question 1**: What is the effectiveness of the NLP algorithm in classifying patient symptoms from the text data on the population level?

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | Status |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ------ |
| CNN                 | **0.9955** | **0.9958** | **0.9955** | **0.9955** | ✅      |
| Linear SVM          | 0.9947     | 0.9952     | 0.9947     | 0.9947     | ✅      |
| Logistic Regression | 0.9940     | 0.9945     | 0.9940     | 0.9940     | ✅      |
| Naive Bayes         | 0.9797     | 0.9813     | 0.9797     | 0.9795     | ✅      |

**Outcome**: **H1a ACCEPTED** (Alternative hypothesis accepted)

- All models achieved >75% clinical threshold
- Best model: CNN with 99.55% F1-Score
- Text analysis demonstrates HIGH precision and recall sufficient for provider decision support

### Audio Classification Results (RQ2)

**Research Question 2**: How effective is NLP in classifying patient symptoms from audio data on the population level?

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | Status |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ------ |
| SVM                 | **0.5139** | **0.3482** | **0.5139** | **0.3922** | ❌      |
| Logistic Regression | 0.3870     | 0.3870     | 0.3870     | 0.3870     | ❌      |

**Outcome**: **H20 ACCEPTED** (Null hypothesis accepted)

- No models achieved 75% clinical threshold
- Best model: SVM with only 39.22% F1-Score
- Audio analysis yields precision and recall metrics INSUFFICIENT for effective provider decision support
- **Clinical Recommendation**: Further development needed for audio-only classification

### Multimodal (Audio + Text) Classification Results (RQ3)

**Research Question 3**: How effective is NLP in classifying patient symptoms from audio and text data on the population level?

| Model                  | Accuracy   | Precision  | Recall     | F1-Score   | Status |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ------ |
| CNN for Audio and Text | **0.9955** | **0.9958** | **0.9955** | **0.9955** | ✅      |

**Outcome**: **H3a ACCEPTED** (Alternative hypothesis accepted)

- Model achieved >75% clinical threshold
- CNN achieved 99.55% F1-Score
- Multimodal analysis results in HIGH precision and recall sufficient for provider decision support
- **Clinical Recommendation**: Deploy for clinical use

### Performance Summary by Modality

| Classification Type | Best Model | F1-Score   | Clinical Threshold (75%) | Deployment Ready |
| ------------------- | ---------- | ---------- | ------------------------ | ---------------- |
| **Text Only**       | CNN        | **99.55%** | ✅ PASSED                 | ✅ YES            |
| **Audio Only**      | SVM        | **39.22%** | ❌ FAILED                 | ❌ NO             |
| **Audio + Text**    | CNN        | **99.55%** | ✅ PASSED                 | ✅ YES            |

### Research Hypothesis Outcomes

#### Text Classification (RQ1)

- **Hypothesis**: H1a ACCEPTED
- **Evidence**: CNN achieves 99.55% F1-Score (>75% threshold)
- **Clinical Impact**: Text analysis provides sufficient precision and recall for provider decision support

#### Audio Classification (RQ2)

- **Hypothesis**: H20 ACCEPTED (Null hypothesis)
- **Evidence**: Best SVM achieves only 39.22% F1-Score (<75% threshold)
- **Clinical Impact**: Audio-only analysis insufficient for effective provider decision support

#### Multimodal Classification (RQ3)

- **Hypothesis**: H3a ACCEPTED
- **Evidence**: CNN achieves 99.55% F1-Score (>75% threshold)
- **Clinical Impact**: Combined audio+text analysis provides sufficient precision and recall for provider decision support

### Key Findings

1. **Text Classification Excellence**: NLP approaches to text analysis demonstrate exceptional performance with 99.55% F1-Score
2. **Audio Classification Challenges**: Pure acoustic feature extraction shows limitations with only 39.22% F1-Score
3. **Multimodal Enhancement**: Combining audio and text maintains high performance (99.55% F1-Score)
4. **Clinical Viability**: Text-based and multimodal approaches meet clinical deployment standards
5. **Research Validation**: 2 out of 3 research hypotheses strongly supported with >99% performance

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas

- **Data Science**: New models, feature engineering, evaluation metrics
- **Software Engineering**: API development, testing, deployment
- **Documentation**: Tutorials, examples, technical documentation
- **Healthcare**: Clinical validation, domain expertise, use case development

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation for API changes

## 👨‍💻 Author & Contact

**Hameem Mahdi**

- 📧 Email: [Mahdi.Hameem@gmail.com](mailto:Mahdi.Hameem@gmail.com)
- 🎓 Institution: [National University](https://www.nu.edu/degrees/engineering-data-and-computer-sciences/programs/doctor-of-philosophy-in-data-science/)
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/hameem-mahdi-m-s-e-ph-d-88686818b/)
- 🐙 GitHub: [@HAMEEMM](https://github.com/HAMEEMM)

### Research Supervision

- **Program**: Doctorate of Science in Data Science
- **Institution**: National University
- **Research Focus**: Multimodal AI for Healthcare Applications

## 🙏 Acknowledgments

### Data Provider

- **Paul Mooney** - For providing the comprehensive Medical Speech, Transcription, and Intent dataset on Kaggle
- **Kaggle Community** - For hosting and maintaining the dataset

### Technical Resources

- **TensorFlow Team** - Deep learning framework
- **Scikit-learn Contributors** - Machine learning library
- **Librosa Developers** - Audio processing library
- **NLTK Team** - Natural language processing toolkit

### Academic Support

- **National University Faculty** - Research guidance and support
- **Data Science Community** - Open-source tools and methodologies
- **National University Faculty** - Research guidance and support
- **Data Science Community** - Open-source tools and methodologies

### Special Thanks

- Healthcare professionals who provided domain expertise
- Beta testers who validated the clinical applications
- Open-source community for foundational tools and libraries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this work in your research, please cite:

```bibtex
Mahdi, H. (2025). Multimodal medical diagnosis system: Audio and text classification for clinical decision support. National University. Available at: https://github.com/HAMEEMM/multimodal_medical_diagnosis
```

---

<div align="center">

**🌟 If you find this project helpful, please consider giving it a star! 🌟**

[![GitHub stars](https://img.shields.io/github/stars/HAMEEMM/multimodal_medical_diagnosis.svg?style=social&label=Star)](https://github.com/HAMEEMM/multimodal_medical_diagnosis)
[![GitHub forks](https://img.shields.io/github/forks/HAMEEMM/multimodal_medical_diagnosis.svg?style=social&label=Fork)](https://github.com/HAMEEMM/multimodal_medical_diagnosis/fork)

**Built with ❤️ for healthcare innovation**

</div>
