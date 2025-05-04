import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    # Load text classification model and vectorizer
    text_model = joblib.load('models/text_classifier.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    # Load audio classification model
    audio_model = load_model('models/audio_model.h5')
    
    # Load combined model if available
    try:
        combined_model = joblib.load('models/combined_classifier.pkl')
    except:
        combined_model = None
    
    return text_model, tfidf_vectorizer, audio_model, combined_model

# Initialize models
try:
    text_model, tfidf_vectorizer, audio_model, combined_model = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    model_loaded = False

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to string
    text = ' '.join(tokens)
    
    return text

# Audio preprocessing function
def preprocess_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # Pad or truncate to fixed length
    target_length = 100  # Adjust based on your model's expected input
    if mfccs.shape[1] > target_length:
        mfccs = mfccs[:, :target_length]
    else:
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    # Reshape for model input
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    return mfccs

# Prediction functions
def predict_from_text(text):
    preprocessed_text = preprocess_text(text)
    text_features = tfidf_vectorizer.transform([preprocessed_text])
    prediction = text_model.predict_proba(text_features)[0]
    return prediction

def predict_from_audio(audio_file):
    preprocessed_audio = preprocess_audio(audio_file)
    prediction = audio_model.predict(preprocessed_audio)[0]
    return prediction

def combined_prediction(text, audio_file):
    text_pred = predict_from_text(text)
    audio_pred = predict_from_audio(audio_file)
    
    if combined_model:
        # If we have a trained combined model
        combined_features = np.concatenate([text_pred, audio_pred])
        combined_features = np.expand_dims(combined_features, axis=0)
        prediction = combined_model.predict_proba(combined_features)[0]
    else:
        # Simple averaging if no combined model
        prediction = (text_pred + audio_pred) / 2
    
    return prediction

# Class labels
class_labels = [
    "emotional_pain", "infected_wound", "common_cold", 
    "joint_pain", "digestive_issues", "headache", 
    "fatigue", "breathing_difficulty", "skin_rash", "fever"
]

# App UI
st.markdown("<h1 class='main-header'>Medical Diagnosis Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered diagnosis using text and audio inputs</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/yourusername/multimodal_medical_diagnosis/main/docs/images/app_logo.png", width=100)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the analysis mode", 
    ["Home", "Text-based Diagnosis", "Audio-based Diagnosis", "Combined Analysis"])

# Home page
if app_mode == "Home":
    st.write("""
    ## Welcome to the Medical Diagnosis Assistant
    
    This application uses advanced AI models to predict potential medical conditions based on:
    - Textual descriptions of symptoms
    - Audio recordings (e.g., cough sounds)
    - Combined analysis of both text and audio
    
    ### How to use this app:
    1. Select your preferred analysis mode from the sidebar
    2. Enter your symptom information or upload audio recordings
    3. View the prediction results and confidence scores
    
    ### Important Note:
    This tool is for research and educational purposes only. Always consult with a qualified healthcare professional for medical advice and diagnosis.
    """)
    
    st.image("https://raw.githubusercontent.com/yourusername/multimodal_medical_diagnosis/main/docs/images/system_architecture.png", 
             caption="System Architecture: Text and Audio Processing Pipeline")

# Text-based Diagnosis
elif app_mode == "Text-based Diagnosis":
    st.markdown("<h2>Text-based Symptom Analysis</h2>", unsafe_allow_html=True)
    st.write("Enter a description of your symptoms below:")
    
    user_text = st.text_area("Symptom Description", height=150,
                            placeholder="Example: I've been experiencing a persistent dry cough for the past 3 days, along with a mild fever and sore throat...")
    
    if st.button("Analyze Text"):
        if user_text and model_loaded:
            with st.spinner("Analyzing your symptoms..."):
                # Get prediction
                prediction = predict_from_text(user_text)
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                # Convert prediction to DataFrame for visualization
                results_df = pd.DataFrame({
                    'Condition': class_labels,
                    'Probability': prediction
                }).sort_values('Probability', ascending=False)
                
                # Show top 3 conditions
                st.write("Top 3 potential conditions:")
                for i in range(3):
                    if i < len(results_df):
                        condition = results_df.iloc[i]['Condition'].replace('_', ' ').title()
                        prob = results_df.iloc[i]['Probability'] * 100
                        st.write(f"{i+1}. {condition}: {prob:.1f}%")
                
                # Show bar chart
                st.bar_chart(results_df.set_index('Condition'))
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.info("Remember: This analysis is for informational purposes only and does not constitute medical advice.")
        elif not user_text:
            st.warning("Please enter your symptoms to get a diagnosis.")
        elif not model_loaded:
            st.error("Models failed to load. Please try again later.")

# Audio-based Diagnosis
elif app_mode == "Audio-based Diagnosis":
    st.markdown("<h2>Audio-based Symptom Analysis</h2>", unsafe_allow_html=True)
    st.write("Upload an audio recording (e.g., cough, breathing sound):")
    
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Analyze Audio"):
            if model_loaded:
                with st.spinner("Analyzing your audio sample..."):
                    # Get prediction
                    prediction = predict_from_audio(audio_file)
                    
                    # Display results
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.subheader("Analysis Results")
                    
                    # Convert prediction to DataFrame for visualization
                    results_df = pd.DataFrame({
                        'Condition': class_labels,
                        'Probability': prediction
                    }).sort_values('Probability', ascending=False)
                    
                    # Show top 3 conditions
                    st.write("Top 3 potential conditions:")
                    for i in range(3):
                        if i < len(results_df):
                            condition = results_df.iloc[i]['Condition'].replace('_', ' ').title()
                            prob = results_df.iloc[i]['Probability'] * 100
                            st.write(f"{i+1}. {condition}: {prob:.1f}%")
                    
                    # Show bar chart
                    st.bar_chart(results_df.set_index('Condition'))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.info("Remember: This analysis is for informational purposes only and does not constitute medical advice.")
            else:
                st.error("Models failed to load. Please try again later.")

# Combined Analysis
elif app_mode == "Combined Analysis":
    st.markdown("<h2>Combined Text and Audio Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Enter a description of your symptoms:")
        user_text = st.text_area("Symptom Description", height=150,
                                placeholder="Describe your symptoms here...")
    
    with col2:
        st.write("Upload an audio recording:")
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if audio_file is not None:
            st.audio(audio_file)
    
    if st.button("Analyze Both"):
        if user_text and audio_file and model_loaded:
            with st.spinner("Analyzing your symptoms..."):
                # Get combined prediction
                prediction = combined_prediction(user_text, audio_file)
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.subheader("Combined Analysis Results")
                
                # Convert prediction to DataFrame for visualization
                results_df = pd.DataFrame({
                    'Condition': class_labels,
                    'Probability': prediction
                }).sort_values('Probability', ascending=False)
                
                # Show top 3 conditions
                st.write("Top 3 potential conditions:")
                for i in range(3):
                    if i < len(results_df):
                        condition = results_df.iloc[i]['Condition'].replace('_', ' ').title()
                        prob = results_df.iloc[i]['Probability'] * 100
                        st.write(f"{i+1}. {condition}: {prob:.1f}%")
                
                # Show bar chart
                st.bar_chart(results_df.set_index('Condition'))
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.write("### Individual Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Text Analysis:**")
                    text_pred = predict_from_text(user_text)
                    text_df = pd.DataFrame({
                        'Condition': class_labels,
                        'Probability': text_pred
                    }).sort_values('Probability', ascending=False).head(3)
                    st.dataframe(text_df)
                
                with col2:
                    st.write("**Audio Analysis:**")
                    audio_pred = predict_from_audio(audio_file)
                    audio_df = pd.DataFrame({
                        'Condition': class_labels,
                        'Probability': audio_pred
                    }).sort_values('Probability', ascending=False).head(3)
                    st.dataframe(audio_df)
                
                st.info("Remember: This analysis is for informational purposes only and does not constitute medical advice.")
        else:
            if not user_text:
                st.warning("Please enter your symptoms.")
            if not audio_file:
                st.warning("Please upload an audio file.")
            if not model_loaded:
                st.error("Models failed to load. Please try again later.")

# Footer
st.markdown("---")
st.markdown("© 2023 Medical Diagnosis Assistant | For Research Purposes Only | [GitHub Repository](https://github.com/HAMEEMM/multimodal_medical_diagnosis)")