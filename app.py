import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import io

# Initialize NLTK components
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e7f5ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_models():
    """Load all required models with caching for better performance"""
    try:
        # Load text classification model
        text_classifier = joblib.load('models/text_classifier.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load audio classification model 
        audio_model = load_model('models/audio_model.h5')
        
        # Load combined model if available
        try:
            combined_model = joblib.load('models/combined_classifier.pkl')
        except:
            combined_model = None
            
        return text_classifier, vectorizer, audio_model, combined_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def preprocess_text(text):
    """Preprocess text input for model prediction"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back to string
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_audio(audio_file):
    """Process audio file for model prediction"""
    # Load audio file
    audio_data, sr = librosa.load(audio_file, sr=22050)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # Standardize features
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # Pad or truncate to fixed length (based on your model requirements)
    target_length = 100  # Adjust based on your model's expected input
    if mfccs.shape[1] > target_length:
        mfccs = mfccs[:, :target_length]
    else:
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    # Reshape for CNN input
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    return mfccs

def predict_text(text, text_classifier, vectorizer):
    """Make prediction based on text input"""
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    prediction = text_classifier.predict_proba(text_features)[0]
    return prediction

def predict_audio(audio_file, audio_model):
    """Make prediction based on audio input"""
    processed_audio = preprocess_audio(audio_file)
    prediction = audio_model.predict(processed_audio)[0]
    return prediction

def predict_combined(text, audio_file, text_classifier, vectorizer, audio_model, combined_model):
    """Make prediction using both text and audio inputs"""
    text_pred = predict_text(text, text_classifier, vectorizer)
    audio_pred = predict_audio(audio_file, audio_model)
    
    if combined_model:
        # If we have a trained combined model, use it
        combined_input = np.concatenate([text_pred, audio_pred])
        combined_input = combined_input.reshape(1, -1)
        prediction = combined_model.predict_proba(combined_input)[0]
    else:
        # Simple averaging if no combined model
        prediction = (text_pred + audio_pred) / 2
        
    return prediction, text_pred, audio_pred

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return the file path"""
    try:
        # Create temp directory if it doesn't exist
        if not os.path.exists('temp'):
            os.makedirs('temp')
            
        # Save the file
        file_path = os.path.join('temp', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def visualize_predictions(predictions, class_labels, title="Prediction Results"):
    """Create bar chart of predictions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Format labels for better display
    display_labels = [label.replace('_', ' ').title() for label in class_labels]
    
    # Sort by probability
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = [display_labels[i] for i in sorted_indices]
    
    # Plot
    ax.barh(sorted_labels, sorted_predictions)
    ax.set_xlabel('Probability')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Return the figure
    return fig

# Load models
text_classifier, vectorizer, audio_model, combined_model = load_models()

# Define class labels (from your notebooks)
class_labels = [
    "emotional_pain", "infected_wound", "common_cold", 
    "joint_pain", "digestive_issues", "headache", 
    "fatigue", "breathing_difficulty", "skin_rash", "fever"
]

# Sidebar Navigation
st.sidebar.title("Medical Diagnosis")
st.sidebar.image("https://i.imgur.com/8koJsNY.png", width=100)

# Navigation
page = st.sidebar.radio(
    "Select Analysis Type", 
    ["Home", "Text-based Diagnosis", "Audio-based Diagnosis", "Combined Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses AI to analyze medical symptoms through text descriptions "
    "and audio recordings. **Important**: This is a research tool and not a substitute "
    "for professional medical advice."
)

# Home Page
if page == "Home":
    st.markdown("<h1 class='main-header'>Multimodal Medical Diagnosis System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## About This System
    
    This AI-powered system combines natural language processing and audio analysis 
    to help identify potential medical conditions based on symptom descriptions and sound recordings.
    
    ### Features:
    
    - **Text Analysis**: Processes written descriptions of symptoms
    - **Audio Analysis**: Analyzes sound recordings (e.g., coughs, breathing)
    - **Combined Analysis**: Integrates both inputs for improved accuracy
    
    ### How to Use:
    
    1. Select the type of analysis from the sidebar
    2. Provide the requested inputs (text description, audio file, or both)
    3. View the analysis results and potential conditions
    
    ### Research Background:
    
    This system represents a novel approach to medical symptom analysis by combining 
    multiple data modalities. Our research shows that this multimodal approach can 
    improve diagnostic accuracy compared to single-modality methods.
    """)
    
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.warning("⚠️ **Important Disclaimer**: This application is for research and educational purposes only. " 
               "It is not intended to provide medical advice or replace consultation with qualified healthcare professionals.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### System Architecture")
    st.image("https://i.imgur.com/JLSIfjL.png", caption="Multimodal Medical Diagnosis System Architecture")

# Text-based Diagnosis
elif page == "Text-based Diagnosis":
    st.markdown("<h1 class='main-header'>Text-based Symptom Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Describe your symptoms in detail below. Include information about:
    - When the symptoms started
    - Their intensity and duration
    - Any factors that make them better or worse
    - Any related symptoms
    """)
    
    # User input
    user_text = st.text_area(
        "Symptom Description", 
        height=150,
        placeholder="Example: I've been experiencing a persistent dry cough for the past 3 days, along with a mild fever and sore throat..."
    )
    
    # Analysis button
    if st.button("Analyze Symptoms"):
        if not user_text:
            st.warning("Please enter a symptom description to analyze.")
        elif not text_classifier or not vectorizer:
            st.error("Error: Required models could not be loaded. Please try again later.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                # Make prediction
                predictions = predict_text(user_text, text_classifier, vectorizer)
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.subheader("Analysis Results")
                
                # Create dataframe for results
                results_df = pd.DataFrame({
                    'Condition': [label.replace('_', ' ').title() for label in class_labels],
                    'Probability': predictions
                }).sort_values('Probability', ascending=False)
                
                # Show top conditions
                st.markdown("#### Top 3 Potential Conditions:")
                for i in range(3):
                    if i < len(results_df):
                        condition = results_df.iloc[i]['Condition']
                        prob = results_df.iloc[i]['Probability'] * 100
                        st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                
                # Visualization
                fig = visualize_predictions(predictions, class_labels, "Text Analysis Results")
                st.pyplot(fig)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.info("**Note**: This analysis is based solely on the text description. "
                        "For a more comprehensive assessment, try the combined analysis with both text and audio inputs.")
                st.markdown("</div>", unsafe_allow_html=True)

# Audio-based Diagnosis
elif page == "Audio-based Diagnosis":
    st.markdown("<h1 class='main-header'>Audio-based Symptom Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload an audio recording of relevant sounds (e.g., coughing, breathing, etc.). 
    For best results:
    - Record in a quiet environment
    - Position the microphone appropriately
    - Provide a clear recording of the relevant sound
    - Use WAV or MP3 format
    """)
    
    # File uploader
    audio_file = st.file_uploader("Upload Audio Recording", type=["wav", "mp3", "ogg"])
    
    if audio_file:
        # Display audio player
        st.audio(audio_file)
        
        # Analysis button
        if st.button("Analyze Audio"):
            if not audio_model:
                st.error("Error: Required models could not be loaded. Please try again later.")
            else:
                with st.spinner("Processing audio and analyzing..."):
                    # Save uploaded file temporarily
                    temp_path = save_uploaded_file(audio_file)
                    
                    if temp_path:
                        # Make prediction
                        predictions = predict_audio(temp_path, audio_model)
                        
                        # Display results
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.subheader("Analysis Results")
                        
                        # Create dataframe for results
                        results_df = pd.DataFrame({
                            'Condition': [label.replace('_', ' ').title() for label in class_labels],
                            'Probability': predictions
                        }).sort_values('Probability', ascending=False)
                        
                        # Show top conditions
                        st.markdown("#### Top 3 Potential Conditions:")
                        for i in range(3):
                            if i < len(results_df):
                                condition = results_df.iloc[i]['Condition']
                                prob = results_df.iloc[i]['Probability'] * 100
                                st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                        
                        # Visualization
                        fig = visualize_predictions(predictions, class_labels, "Audio Analysis Results")
                        st.pyplot(fig)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Audio feature visualization (optional)
                        with st.expander("View Audio Features"):
                            # Load audio and extract features for visualization
                            y, sr = librosa.load(temp_path, sr=22050)
                            
                            # Waveform
                            fig, ax = plt.subplots(figsize=(10, 4))
                            librosa.display.waveshow(y, sr=sr, ax=ax)
                            ax.set_title('Audio Waveform')
                            st.pyplot(fig)
                            
                            # MFCCs
                            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            librosa.display.specshow(mfccs, x_axis='time', ax=ax)
                            ax.set_title('MFCC Features')
                            fig.colorbar(ax.collections[0], format='%+2.f')
                            st.pyplot(fig)
                        
                        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                        st.info("**Note**: This analysis is based solely on the audio recording. "
                                "For a more comprehensive assessment, try the combined analysis with both text and audio inputs.")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Clean up temp file
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    else:
                        st.error("Error processing the audio file. Please try again.")

# Combined Analysis
elif page == "Combined Analysis":
    st.markdown("<h1 class='main-header'>Combined Text and Audio Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This analysis uses both symptom descriptions and audio recordings to provide more comprehensive results.
    Please provide both inputs below for the best diagnostic assessment.
    """)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Symptom Description")
        user_text = st.text_area(
            "Describe your symptoms in detail",
            height=150,
            placeholder="Example: I've been experiencing a persistent dry cough for the past 3 days, along with a mild fever and sore throat..."
        )
    
    with col2:
        st.markdown("### Audio Recording")
        audio_file = st.file_uploader("Upload relevant audio recording", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file)
    
    # Analysis button
    if st.button("Perform Combined Analysis"):
        if not user_text:
            st.warning("Please enter a symptom description.")
        elif not audio_file:
            st.warning("Please upload an audio recording.")
        elif not text_classifier or not vectorizer or not audio_model:
            st.error("Error: Required models could not be loaded. Please try again later.")
        else:
            with st.spinner("Performing comprehensive analysis..."):
                # Save uploaded file temporarily
                temp_path = save_uploaded_file(audio_file)
                
                if temp_path:
                    # Make combined prediction
                    combined_predictions, text_predictions, audio_predictions = predict_combined(
                        user_text, temp_path, text_classifier, vectorizer, audio_model, combined_model
                    )
                    
                    # Display main results
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.subheader("Combined Analysis Results")
                    
                    # Create dataframe for results
                    results_df = pd.DataFrame({
                        'Condition': [label.replace('_', ' ').title() for label in class_labels],
                        'Probability': combined_predictions
                    }).sort_values('Probability', ascending=False)
                    
                    # Show top conditions
                    st.markdown("#### Top 3 Potential Conditions:")
                    for i in range(3):
                        if i < len(results_df):
                            condition = results_df.iloc[i]['Condition']
                            prob = results_df.iloc[i]['Probability'] * 100
                            st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                    
                    # Visualization
                    fig = visualize_predictions(combined_predictions, class_labels, "Combined Analysis Results")
                    st.pyplot(fig)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Individual results comparison
                    st.markdown("### Comparison of Individual Analyses")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Text-Based Results")
                        # Create dataframe for text results
                        text_df = pd.DataFrame({
                            'Condition': [label.replace('_', ' ').title() for label in class_labels],
                            'Probability': text_predictions
                        }).sort_values('Probability', ascending=False).head(3)
                        
                        st.dataframe(text_df)
                    
                    with col2:
                        st.markdown("#### Audio-Based Results")
                        # Create dataframe for audio results
                        audio_df = pd.DataFrame({
                            'Condition': [label.replace('_', ' ').title() for label in class_labels],
                            'Probability': audio_predictions
                        }).sort_values('Probability', ascending=False).head(3)
                        
                        st.dataframe(audio_df)
                    
                    # Detailed comparison visualization
                    st.markdown("### Modality Comparison Visualization")
                    
                    # Prepare data for comparison chart
                    top_conditions = results_df.head(5)['Condition'].tolist()
                    top_indices = [list(results_df['Condition']).index(condition) for condition in top_conditions]
                    
                    comparison_data = {
                        'Condition': top_conditions,
                        'Text Analysis': [text_predictions[class_labels.index(condition.lower().replace(' ', '_'))] * 100 for condition in top_conditions],
                        'Audio Analysis': [audio_predictions[class_labels.index(condition.lower().replace(' ', '_'))] * 100 for condition in top_conditions],
                        'Combined Analysis': [combined_predictions[class_labels.index(condition.lower().replace(' ', '_'))] * 100 for condition in top_conditions]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.bar_chart(comparison_df.set_index('Condition'))
                    
                    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                    st.info("**Insight**: The combined analysis leverages both text descriptions and audio data "
                            "to provide a more comprehensive assessment. Areas where both modalities agree "
                            "generally indicate higher confidence in the diagnosis.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    st.error("Error processing the audio file. Please try again.")

# Footer
st.markdown("---")
st.markdown(
    "© 2025 Multimodal Medical Diagnosis System | For Research Purposes Only | "
    "[GitHub Repository](https://github.com/yourusername/multimodal_medical_diagnosis)"
)