import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Basic page setup
st.set_page_config(page_title="Medical Diagnosis System", page_icon="🏥")
st.title("Multimodal Medical Diagnosis System")

# Define constants (from your project)
CLASS_LABELS = [
    "emotional_pain", "infected_wound", "common_cold", 
    "joint_pain", "digestive_issues", "headache", 
    "fatigue", "breathing_difficulty", "skin_rash", "fever"
]

# Demo mode notice
st.warning("⚠️ **DEMO MODE**: Running without trained models. For full functionality, add model files to the repository.")

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis Type:", 
    ["Home", "Text Analysis", "Audio Analysis", "Combined Analysis"])

# Demo prediction functions
def demo_predict_text(text):
    """Generate simulated predictions for text"""
    # Random predictions with higher probability for certain conditions based on keywords
    probs = np.random.uniform(0.05, 0.15, len(CLASS_LABELS))
    
    # Simple keyword matching to make demo somewhat realistic
    text = text.lower()
    if "headache" in text or "head" in text:
        probs[5] = 0.7  # headache
    elif "cough" in text or "cold" in text or "throat" in text:
        probs[2] = 0.6  # common cold
    elif "stomach" in text or "nausea" in text:
        probs[4] = 0.65  # digestive issues
    elif "skin" in text or "rash" in text:
        probs[8] = 0.75  # skin rash
    elif "fever" in text:
        probs[9] = 0.68  # fever
        
    # Normalize
    probs = probs / np.sum(probs)
    return probs

def demo_predict_audio(audio_file=None):
    """Generate simulated predictions for audio"""
    # Random predictions
    probs = np.random.uniform(0.05, 0.2, len(CLASS_LABELS))
    
    # Make breathing_difficulty and common_cold more likely for audio
    probs[7] = 0.4  # breathing difficulty
    probs[2] = 0.3  # common cold
    
    # Normalize
    probs = probs / np.sum(probs)
    return probs

def visualize_predictions(predictions):
    """Create a bar chart of prediction probabilities"""
    # Format labels for better display
    display_labels = [label.replace('_', ' ').title() for label in CLASS_LABELS]
    
    # Create dataframe sorted by probability
    results_df = pd.DataFrame({
        'Condition': display_labels,
        'Probability': predictions
    }).sort_values('Probability', ascending=False)
    
    # Return the dataframe for display
    return results_df

# PAGE: Home
if page == "Home":
    st.header("Welcome to the Medical Diagnosis System")
    
    st.markdown("""
    ### About This System
    
    This AI-powered system helps identify potential medical conditions based on:
    
    - **Text descriptions** of symptoms
    - **Audio recordings** of relevant sounds (breathing, coughing, etc.)
    - **Combined analysis** of both inputs
    
    ### How to Use
    
    1. Select your preferred analysis type from the sidebar
    2. Enter your symptom information or upload audio recordings
    3. View the prediction results
    
    ### Important Note
    
    This application is for research and demonstration purposes only. 
    Always consult a healthcare professional for medical advice.
    """)

# PAGE: Text Analysis
elif page == "Text Analysis":
    st.header("Text-based Symptom Analysis")
    
    st.markdown("""
    Describe your symptoms in detail below. Include when they started, 
    their severity, and any other relevant information.
    """)
    
    # User input
    user_text = st.text_area(
        "Symptom Description",
        placeholder="Example: I've been experiencing a persistent dry cough for the past 3 days, along with a mild fever and sore throat...",
        height=150
    )
    
    # Analysis button
    if st.button("Analyze Symptoms"):
        if not user_text:
            st.warning("Please enter a symptom description to analyze.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                # Get demo predictions
                predictions = demo_predict_text(user_text)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create and display results dataframe
                results_df = visualize_predictions(predictions)
                
                # Show top 3 conditions
                st.markdown("#### Top 3 Potential Conditions:")
                for i in range(3):
                    condition = results_df.iloc[i]['Condition']
                    prob = results_df.iloc[i]['Probability'] * 100
                    st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                
                # Bar chart
                st.bar_chart(results_df.set_index('Condition'))
                
                st.info("Note: This analysis is running in demo mode with simulated predictions.")

# PAGE: Audio Analysis
elif page == "Audio Analysis":
    st.header("Audio-based Symptom Analysis")
    
    st.markdown("""
    Upload an audio recording of relevant sounds (e.g., coughing, breathing sounds).
    """)
    
    # File uploader
    audio_file = st.file_uploader("Upload Audio Recording", type=["wav", "mp3", "ogg"])
    
    if audio_file:
        # Display audio player
        st.audio(audio_file)
        
        # Analysis button
        if st.button("Analyze Audio"):
            with st.spinner("Processing audio..."):
                # Get demo predictions
                predictions = demo_predict_audio(audio_file)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create and display results dataframe
                results_df = visualize_predictions(predictions)
                
                # Show top 3 conditions
                st.markdown("#### Top 3 Potential Conditions:")
                for i in range(3):
                    condition = results_df.iloc[i]['Condition']
                    prob = results_df.iloc[i]['Probability'] * 100
                    st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                
                # Bar chart
                st.bar_chart(results_df.set_index('Condition'))
                
                st.info("Note: This analysis is running in demo mode with simulated predictions.")

# PAGE: Combined Analysis
elif page == "Combined Analysis":
    st.header("Combined Text and Audio Analysis")
    
    st.markdown("""
    This mode combines both text descriptions and audio recordings for more accurate analysis.
    Please provide both inputs below.
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Symptom Description")
        user_text = st.text_area(
            "Describe your symptoms",
            height=150,
            placeholder="Example: I've been experiencing a persistent dry cough..."
        )
    
    with col2:
        st.markdown("### Audio Recording")
        audio_file = st.file_uploader("Upload relevant audio", type=["wav", "mp3", "ogg"])
        if audio_file:
            st.audio(audio_file)
    
    # Analysis button
    if st.button("Perform Combined Analysis"):
        if not user_text:
            st.warning("Please enter a symptom description.")
        elif not audio_file:
            st.warning("Please upload an audio recording.")
        else:
            with st.spinner("Performing comprehensive analysis..."):
                # Get demo predictions
                text_predictions = demo_predict_text(user_text)
                audio_predictions = demo_predict_audio(audio_file)
                
                # Simple averaging for combined predictions
                combined_predictions = (text_predictions + audio_predictions) / 2
                
                # Display main results
                st.subheader("Combined Analysis Results")
                
                # Create and display results dataframe
                results_df = visualize_predictions(combined_predictions)
                
                # Show top 3 conditions
                st.markdown("#### Top 3 Potential Conditions:")
                for i in range(3):
                    condition = results_df.iloc[i]['Condition']
                    prob = results_df.iloc[i]['Probability'] * 100
                    st.markdown(f"**{i+1}. {condition}**: {prob:.1f}%")
                
                # Bar chart
                st.bar_chart(results_df.set_index('Condition'))
                
                # Comparison of individual analyses
                st.markdown("### Comparison of Individual Analyses")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Text Analysis Results")
                    text_df = visualize_predictions(text_predictions).head(3)
                    st.dataframe(text_df)
                
                with col2:
                    st.markdown("#### Audio Analysis Results")
                    audio_df = visualize_predictions(audio_predictions).head(3)
                    st.dataframe(audio_df)
                
                st.info("Note: This analysis is running in demo mode with simulated predictions.")

# Footer
st.markdown("---")
st.markdown("© 2025 Medical Diagnosis System | Research Purposes Only by Hameem Mahdi")