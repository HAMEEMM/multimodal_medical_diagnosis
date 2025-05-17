"""
Test script to verify the proper installation of all required packages
for the multimodal_medical_diagnosis project.
"""
import sys
import importlib
import pkg_resources

def import_or_report(package_name):
    """Attempts to import a package and reports success or failure"""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} successfully imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {str(e)}")
        return False

def test_tensorflow():
    """Test TensorFlow functionality"""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        print(f"  - GPU available: {'Yes' if tf.config.list_physical_devices('GPU') else 'No'}")
        print("  - Testing TensorFlow operation...")
        # Simple TensorFlow operation
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print(f"  - Matrix multiplication result: \n{c.numpy()}")
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {str(e)}")
        return False

def test_spacy():
    """Test spaCy functionality"""
    try:
        import spacy
        print(f"✓ spaCy version: {spacy.__version__}")
        print("  - Loading English model...")
        nlp = spacy.load("en_core_web_sm")
        text = "This is a test sentence for spaCy processing."
        doc = nlp(text)
        print(f"  - Tokenization: {[token.text for token in doc][:5]}...")
        print(f"  - Part-of-speech tags: {[(token.text, token.pos_) for token in doc][:3]}...")
        return True
    except Exception as e:
        print(f"✗ spaCy test failed: {str(e)}")
        return False
        
def test_librosa():
    """Test librosa functionality"""
    try:
        import librosa
        import numpy as np
        print(f"✓ librosa version: {librosa.__version__}")
        print("  - Creating test audio signal...")
        # Create a simple sine wave
        sr = 22050  # Sample rate
        duration = 0.5  # Duration in seconds
        frequency = 440  # Frequency in Hz (A4 note)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * frequency * t)
        
        # Extract features
        print("  - Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"  - MFCC shape: {mfccs.shape}")
        return True
    except Exception as e:
        print(f"✗ librosa test failed: {str(e)}")
        return False

def test_speechpy():
    """Test speechpy functionality as replacement for python-speech-features"""
    try:
        import speechpy
        import numpy as np
        print(f"✓ speechpy installed")
        print("  - Creating test audio signal...")
        # Create a simple signal
        fs = 16000  # Sample rate
        signal = np.random.uniform(low=-1, high=1, size=fs)  # 1 second of random noise
        
        # Extract features
        print("  - Extracting MFCC features...")
        mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=13)
        print(f"  - MFCC shape: {mfcc.shape}")
        return True
    except Exception as e:
        print(f"✗ speechpy test failed: {str(e)}")
        return False

def main():
    print("=" * 50)
    print("MULTIMODAL MEDICAL DIAGNOSIS ENVIRONMENT TEST")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print("-" * 50)
    
    # Core data science packages
    print("\nTesting core data science packages:")
    import_or_report("numpy")
    import_or_report("pandas")
    import_or_report("scipy")
    import_or_report("matplotlib")
    import_or_report("seaborn")
    
    # Machine learning packages
    print("\nTesting machine learning packages:")
    import_or_report("sklearn")
    import_or_report("xgboost")
    import_or_report("imblearn")  # imbalanced-learn
    
    # NLP packages
    print("\nTesting NLP packages:")
    import_or_report("nltk")
    import_or_report("textblob")
    import_or_report("transformers")
    import_or_report("gensim")
    test_spacy()
    
    # Audio processing
    print("\nTesting audio processing packages:")
    test_librosa()
    test_speechpy()
    
    # Deep learning
    print("\nTesting deep learning frameworks:")
    import_or_report("keras")
    test_tensorflow()
    
    # Interactive and visualization
    print("\nTesting interactive and visualization packages:")
    import_or_report("jupyter")
    import_or_report("ipywidgets")
    import_or_report("tqdm")
    import_or_report("plotly")
    import_or_report("wordcloud")
    import_or_report("streamlit")
    
    print("\n" + "=" * 50)
    print("Environment test completed")
    print("=" * 50)

if __name__ == "__main__":
    main()
