
# Model Card: CNN

## Model Details
- **Model Name**: CNN
- **Model Type**: Deep Learning
- **Task**: Multi-class Medical Diagnosis
- **Modality**: Audio-Text Feature Fusion
- **Date**: 2025-10-07

## Intended Use
- **Primary Use**: Medical symptom classification from audio and text inputs
- **Target Users**: Healthcare professionals, clinical decision support systems
- **Out-of-Scope Uses**: Not for standalone diagnosis without clinical oversight

## Performance Metrics

### Test Set Performance
- **Accuracy**: 84.88%
- **Precision**: 86.20%
- **Recall**: 84.88%
- **F1-Score**: 85.05%

### Generalization
- **Train Accuracy**: 97.36%
- **Validation Accuracy**: 87.65%
- **Test Accuracy**: 84.88%
- **Generalization Gap**: 2.78%

## Training Data
- **Source**: Multimodal medical diagnosis dataset (audio + text)
- **Size**: Training + Validation sets
- **Features**: Audio features + Text embeddings

## Limitations
- Performance varies across disease classes
- Lower accuracy for pain-related symptoms with overlapping descriptions
- Requires both audio and text inputs
- Trained on specific dataset distribution

## Ethical Considerations
- Should not replace professional medical diagnosis
- Requires validation in clinical settings
- Consider bias in training data representation
- Ensure patient privacy in deployment

## Contact
- **Researcher**: [Your Name/Institution]
- **Date**: 2025-10-07
