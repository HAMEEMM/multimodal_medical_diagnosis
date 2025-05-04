# =========================================================
# APPLICATION.PY - MEDICAL PROMPT PREDICTION APPLICATION
# =========================================================
import os
import pickle
import numpy as np
import librosa
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load all models
def load_models():
    models = {}
    models_dir = "models"
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found")
        return None
    
    # Load text model
    text_model_path = os.path.join(models_dir, "text_classification_model.pkl")
    if os.path.exists(text_model_path):
        with open(text_model_path, 'rb') as f:
            models['text'] = pickle.load(f)
        print("Text model loaded successfully")
    else:
        print(f"Text model not found at {text_model_path}")
    
    # Load audio model
    audio_model_path = os.path.join(models_dir, "audio_classification_model.pkl")
    if os.path.exists(audio_model_path):
        with open(audio_model_path, 'rb') as f:
            models['audio'] = pickle.load(f)
        print("Audio model loaded successfully")
    else:
        print(f"Audio model not found at {audio_model_path}")
    
    # Load combined model
    combined_model_path = os.path.join(models_dir, "combined_classification_model.pkl")
    if os.path.exists(combined_model_path):
        with open(combined_model_path, 'rb') as f:
            models['combined'] = pickle.load(f)
        print("Combined model loaded successfully")
    else:
        print(f"Combined model not found at {combined_model_path}")
    
    return models

# Audio feature extraction function
def extract_audio_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features (same as in training)
        # Time-domain features
        duration = librosa.get_duration(y=y, sr=sr)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        zero_crossing_rate_std = np.std(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        rms_std = np.std(librosa.feature.rms(y=y))
        
        # Frequency-domain features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_centroid_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_bandwidth_std = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Combine all features into a flat array
        feature_list = [
            duration, zero_crossing_rate, zero_crossing_rate_std, 
            rms, rms_std, spectral_centroid, spectral_centroid_std,
            spectral_bandwidth, spectral_bandwidth_std,
            spectral_rolloff, spectral_rolloff_std,
            tempo
        ]
        
        # Add MFCCs
        feature_list.extend(mfccs_mean)
        feature_list.extend(mfccs_std)
        
        # Add chroma features
        feature_list.extend(chroma_mean)
        feature_list.extend(chroma_std)
        
        return np.array([feature_list])
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Load models at startup
models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text input
        text = request.form.get('text', '')
        
        # Get audio file if uploaded
        audio_file = request.files.get('audio')
        
        results = {}
        confidence_scores = {}
        
        # Text prediction
        if text and 'text' in models:
            text_model = models['text']
            text_prediction = text_model.predict([text])[0]
            
            # Get confidence scores if available
            if hasattr(text_model, 'predict_proba'):
                text_probs = text_model.predict_proba([text])[0]
                text_confidence = np.max(text_probs)
            else:
                text_confidence = "Not available"
            
            results['text'] = text_prediction
            confidence_scores['text'] = text_confidence
        
        # Audio prediction
        if audio_file and 'audio' in models:
            # Save uploaded file temporarily
            temp_path = "temp_audio.wav"
            audio_file.save(temp_path)
            
            # Extract features
            audio_features = extract_audio_features(temp_path)
            
            if audio_features is not None:
                audio_model = models['audio']
                audio_prediction = audio_model.predict(audio_features)[0]
                
                # Get confidence scores if available
                if hasattr(audio_model, 'predict_proba'):
                    audio_probs = audio_model.predict_proba(audio_features)[0]
                    audio_confidence = np.max(audio_probs)
                else:
                    audio_confidence = "Not available"
                
                results['audio'] = audio_prediction
                confidence_scores['audio'] = audio_confidence
                
                # Remove temporary file
                os.remove(temp_path)
        
        # Combined prediction if both text and audio are available
        if text and audio_file and 'combined' in models and 'text' in models and 'audio' in models:
            combined_model = models['combined']
            text_model = combined_model['text_model']
            audio_model = combined_model['audio_model']
            meta_classifier = combined_model['meta_classifier']
            
            # Get text probabilities
            if hasattr(text_model, 'predict_proba'):
                text_probs = text_model.predict_proba([text])[0]
            else:
                text_probs = text_model.decision_function([text])
                if text_probs.ndim == 1:
                    text_probs = np.column_stack([1 - text_probs, text_probs])[0]
                else:
                    text_probs = np.exp(text_probs) / np.sum(np.exp(text_probs), axis=1, keepdims=True)[0]
            
            # Get audio probabilities
            if hasattr(audio_model, 'predict_proba'):
                audio_probs = audio_model.predict_proba(audio_features)[0]
            else:
                audio_probs = audio_model.decision_function(audio_features)
                if audio_probs.ndim == 1:
                    audio_probs = np.column_stack([1 - audio_probs, audio_probs])[0]
                else:
                    audio_probs = np.exp(audio_probs) / np.sum(np.exp(audio_probs), axis=1, keepdims=True)[0]
            
            # Combine probabilities and use meta-classifier
            combined_features = np.concatenate([text_probs, audio_probs]).reshape(1, -1)
            combined_prediction = meta_classifier.predict(combined_features)[0]
            
            # Get confidence scores if available
            if hasattr(meta_classifier, 'predict_proba'):
                combined_probs = meta_classifier.predict_proba(combined_features)[0]
                combined_confidence = np.max(combined_probs)
            else:
                combined_confidence = "Not available"
            
            results['combined'] = combined_prediction
            confidence_scores['combined'] = combined_confidence
        
        return render_template('result.html', 
                              results=results,
                              confidence_scores=confidence_scores,
                              text=text)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)