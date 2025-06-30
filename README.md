# 🏥 Multimodal Medical Diagnosis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen.svg)]()

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Hypothesis](#-research-hypothesis)
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

This project implements a comprehensive **Multimodal Medical Diagnosis System** that leverages both **Audio Analysis** and **Text Classification** techniques to assist healthcare providers in clinical decision-making. The system analyzes patient symptom descriptions through voice recordings and text data to predict medical conditions with high accuracy.

### 🎯 Key Features

- **🎤 Audio Classification**: Advanced speech analysis using MFCCs, spectrograms, and deep learning models
- **📝 Text Classification**: Natural Language Processing for symptom description analysis
- **🤖 Machine Learning Models**: Support Vector Machines, Random Forest, Naive Bayes, Logistic Regression
- **🧠 Deep Learning**: CNN, RNN, and Feedforward Neural Networks
- **📊 Comprehensive Evaluation**: Advanced metrics including Cohen's Kappa, Matthews Correlation, AUC-ROC
- **⚡ Real-time Inference**: Optimized models for clinical deployment
- **📈 Visualization**: Interactive plots and confusion matrices for performance analysis

### 🏥 Clinical Applications

- **Primary Care**: Initial symptom assessment and triage
- **Telemedicine**: Remote patient evaluation
- **Emergency Medicine**: Rapid symptom classification
- **Healthcare Analytics**: Population-level health monitoring
- **Medical Education**: Training and simulation tools

## 🔬 Research Hypothesis

### Text Classification Research Question

**RQ1**: What is the effectiveness of NLP algorithms in classifying patient symptoms from text data on the population level?

- **H10** (Null): Text analysis of patient symptoms results in insufficient precision and recall for provider decision support.
- **H1a** (Alternative): Text analysis of patient symptoms results in precision and recall sufficient for provider decision support.

### Audio Classification Research Question

**RQ2**: What is the effectiveness of deep learning algorithms in classifying patient symptoms from voice recordings on the population level?

- **H20** (Null): Audio analysis of patient voice recordings results in insufficient precision and recall for provider decision support.
- **H2a** (Alternative): Audio analysis of patient voice recordings results in precision and recall sufficient for provider decision support.

### Performance Thresholds

- **Minimum Acceptable**: 75% precision, recall, and F1-score
- **High Performance**: 85% precision, recall, and F1-score
- **Clinical Deployment**: All metrics must exceed minimum threshold

## 📊 Dataset Information

### Source

- **Dataset**: Medical Speech, Transcription, and Intent Dataset
- **Platform**: Kaggle
- **Owner**: Paul Mooney
- **URL**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

### Dataset Characteristics

- **Audio Files**:

  - Format: WAV
  - Sample Rate: 16kHz, 22kHz, 44kHz
  - Duration: Variable (0.5-30 seconds)
  - Total Files: 6,661 recordings
- **Text Data**:

  - Patient symptom descriptions
  - Medical transcriptions
  - Intent classifications
  - Total Samples: 6,661 entries
- **Medical Categories**: 25 diagnostic categories including:

  - Cardiovascular conditions
  - Respiratory symptoms
  - Gastrointestinal issues
  - Neurological conditions
  - Musculoskeletal problems
  - Dermatological conditions
  - And more...

### Data Variables

#### Text Classification Variables

- **phrase**: Text entries containing patient symptom descriptions
- **prompt**: Corresponding medical diagnosis/category for each text entry

#### Audio Classification Variables

- **file_name**: Audio file identifier and path
- **prompt**: Corresponding medical diagnosis/category for each audio sample

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                              │
├─────────────────────┬───────────────────────────────────────┤
│    Text Data        │        Audio Files                    │
│ (Symptom Descriptions)│    (.wav, .mp3)                     │
└─────────────────────┴───────────────────────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐    ┌───────────────────────────────────┐
│  Text Preprocessing │    │    Audio Preprocessing            │
│  • Tokenization     │    │  • Feature Extraction            │
│  • Stopword Removal │    │  • MFCC Analysis                 │
│  • Lemmatization    │    │  • Spectrograms                  │
│  • TF-IDF Vector.   │    │  • Normalization                 │
└─────────────────────┘    └───────────────────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐    ┌───────────────────────────────────┐
│   Model Training    │    │      Model Training               │
│  • Traditional ML   │    │    • Traditional ML               │
│    - Naive Bayes    │    │      - SVM                        │
│    - SVM            │    │      - Random Forest              │
│    - Logistic Reg   │    │      - Logistic Regression        │
│  • Deep Learning    │    │    • Deep Learning                │
│    - CNN            │    │      - CNN                        │
│                     │    │      - Feedforward NN             │
└─────────────────────┘    └───────────────────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Evaluation                             │
│  • Cross-validation • Confusion Matrices • ROC Analysis    │
│  • Statistical Tests • Performance Metrics • Error Analysis│
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Clinical Decision Support                      │
│           • Real-time Inference                             │
│           • Confidence Scoring                              │
│           • Multimodal Integration                          │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Environment Setup

1. **Clone the Repository**

```bash
git clone https://github.com/HAMEEMM/multimodal_medical_diagnosis.git
cd multimodal_medical_diagnosis
```

2. **Create Virtual Environment**

```bash
# Using conda (recommended)
conda create -n medical_diagnosis python=3.8
conda activate medical_diagnosis

# Or using venv
python -m venv medical_diagnosis
# Windows
medical_diagnosis\Scripts\activate
# macOS/Linux
source medical_diagnosis/bin/activate
```

3. **Install Dependencies**

```bash
# Core dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install tensorflow-gpu

# For audio processing
pip install librosa soundfile pydub

# For visualization
pip install plotly seaborn matplotlib

# For advanced NLP
pip install transformers spacy
python -m spacy download en_core_web_sm
```

### Required Libraries

```python
# Core Data Science
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Deep Learning
tensorflow>=2.8.0
keras>=2.8.0

# Audio Processing
librosa>=0.9.0
soundfile>=0.10.0
pydub>=0.25.0

# Natural Language Processing
nltk>=3.7
textblob>=0.17.0
spacy>=3.4.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Machine Learning Enhancement
imbalanced-learn>=0.8.0
optuna>=2.10.0  # For hyperparameter optimization

# Utilities
tqdm>=4.62.0
joblib>=1.1.0
```

### NLTK Data Setup

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

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

### Text Processing Pipeline

1. **Preprocessing**

   - Tokenization using NLTK
   - Stopword removal (medical terms preserved)
   - Lemmatization for root word extraction
   - TF-IDF vectorization (max_features=5000)
2. **Feature Engineering**

   - N-gram features (1-3 grams)
   - Sentiment analysis scores
   - Text complexity metrics
   - Medical entity recognition
3. **Model Architectures**

   - **Traditional ML**: SVM, Naive Bayes, Logistic Regression
   - **Deep Learning**: CNN, Feedforward NN, BERT-based models

### Audio Processing Pipeline

1. **Feature Extraction**

   - **MFCCs**: 13 coefficients capturing spectral envelope
   - **Spectrograms**: Time-frequency representation
   - **Chroma Features**: Harmonic content analysis
   - **Zero Crossing Rate**: Speech vs silence detection
   - **Spectral Features**: Centroid, bandwidth, rolloff
2. **Data Augmentation**

   - Time stretching (±20%)
   - Pitch shifting (±2 semitones)
   - Noise addition (SNR: 20-40dB)
   - Volume normalization
3. **Model Architectures**

   - **CNN**: 3 convolutional layers + 2 dense layers
   - **RNN**: LSTM with 128 hidden units
   - **Hybrid**: CNN feature extraction + RNN temporal modeling

### Evaluation Metrics

- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: Cohen's Kappa, Matthews Correlation, AUC-ROC
- **Clinical Metrics**: Sensitivity, Specificity, NPV, PPV
- **Statistical Tests**: McNemar's test, paired t-tests

## 📈 Results & Performance

### Text Classification Results

| Model | Accuracy         | Precision        | Recall           | F1-Score         | AUC-ROC          |
| ----- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| CNN   | **0.9955** | **0.9958** | **0.9955** | **0.9955** | **0.9456** |

### Audio Classification Results

| Model | Accuracy         | Precision        | Recall           | F1-Score         | AUC-ROC          |
| ----- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| CNN   | **0.9925** | **0.9930** | **0.9925** | **0.9925** | **0.9234** |

### Research Hypothesis Outcomes

✅ **Text Classification**: H1a ACCEPTED - Models achieve >75% performance threshold (CNN: 99.55% F1-Score)
✅ **Audio Classification**: H2a ACCEPTED - Models achieve >75% performance threshold (CNN: 99.25% F1-Score)

Both modalities demonstrate sufficient precision and recall for clinical decision support, with text classification showing superior performance over audio classification in this implementation.

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
- 🎓 Institution: National University
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/hameem-mahdi)
- 🐙 GitHub: [@hameem-mahdi](https://github.com/hameem-mahdi)

### Research Supervision

- **Program**: Master of Science in Data Science
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

[![GitHub stars](https://img.shields.io/github/stars/hameem-mahdi/multimodal_medical_diagnosis.svg?style=social&label=Star)](https://github.com/hameem-mahdi/multimodal_medical_diagnosis)
[![GitHub forks](https://img.shields.io/github/forks/hameem-mahdi/multimodal_medical_diagnosis.svg?style=social&label=Fork)](https://github.com/hameem-mahdi/multimodal_medical_diagnosis/fork)

**Built with ❤️ for healthcare innovation**

</div>
