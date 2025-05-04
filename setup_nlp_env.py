
"""
Complete NLP Environment Setup Script
-------------------------------------
This script addresses common NLP environment setup issues:
1. Installs missing NLTK resources
2. Verifies spaCy model installation
3. Registers the current environment as a Jupyter kernel
"""

import sys
import subprocess
import importlib.util

def setup_nltk():
    """Install required NLTK resources"""
    print("\nSetting up NLTK resources...")
    try:
        import nltk
        
        # List of required NLTK resources
        resources = ['punkt', 'stopwords', 'wordnet']
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                print(f"✓ NLTK '{resource}' is already available")
            except LookupError:
                print(f"Downloading NLTK '{resource}'...")
                nltk.download(resource, quiet=True)
                print(f"✓ NLTK '{resource}' downloaded successfully")
        
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        return False

def setup_spacy():
    """Verify spaCy and download the English model if needed"""
    print("\nSetting up spaCy...")
    try:
        import spacy
        print(f"✓ spaCy {spacy.__version__} is installed")
        
        try:
            # Try to load the English model
            nlp = spacy.load('en_core_web_sm')
            print(f"✓ spaCy English model is already available")
        except OSError:
            # Download the model if not available
            print("Downloading spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✓ spaCy English model downloaded successfully")
        
        return True
    except ImportError:
        print("⚠️ spaCy is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            print("✓ spaCy installed successfully")
            # Now download the English model
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✓ spaCy English model downloaded successfully")
            return True
        except Exception as e:
            print(f"Error installing spaCy: {str(e)}")
            return False

def setup_jupyter_kernel():
    """Register the current environment as a Jupyter kernel"""
    print("\nSetting up Jupyter kernel for NLP...")
    try:
        # Check if ipykernel is installed
        if importlib.util.find_spec("ipykernel") is None:
            print("Installing ipykernel...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
        
        # Register the kernel
        kernel_name = "medical-nlp-env"
        display_name = "Python (Medical NLP)"
        
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", "--user",
            f"--name={kernel_name}", f"--display-name={display_name}"
        ])
        
        print(f"✓ Jupyter kernel '{display_name}' registered successfully")
        print(f"  Make sure to select this kernel in your Jupyter notebook")
        
        return True
    except Exception as e:
        print(f"Error setting up Jupyter kernel: {str(e)}")
        return False

def test_environment():
    """Test that everything is working correctly"""
    print("\nTesting NLP environment...")
    try:
        # Test spaCy
        import spacy
        nlp = spacy.load('en_core_web_sm')
        text = "Testing NLP functionality."
        doc = nlp(text)
        print(f"✓ spaCy test: {[token.text for token in doc]}")
        
        # Test NLTK
        import nltk
        from nltk.stem import WordNetLemmatizer
        tokens = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        print(f"✓ NLTK test: {tokens}")
        print(f"✓ NLTK lemmatization: {lemmas}")
        
        print("\nAll tests passed! Your environment is ready for NLP tasks.")
        return True
    except Exception as e:
        print(f"Environment test failed: {str(e)}")
        return False

def main():
    """Run all setup steps"""
    print("=" * 60)
    print("NLP ENVIRONMENT SETUP")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    
    nltk_ok = setup_nltk()
    spacy_ok = setup_spacy()
    jupyter_ok = setup_jupyter_kernel()
    
    if all([nltk_ok, spacy_ok, jupyter_ok]):
        test_environment()
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"NLTK: {'✓ OK' if nltk_ok else '⚠️ Issues found'}")
    print(f"spaCy: {'✓ OK' if spacy_ok else '⚠️ Issues found'}")
    print(f"Jupyter kernel: {'✓ OK' if jupyter_ok else '⚠️ Issues found'}")
    
    if all([nltk_ok, spacy_ok, jupyter_ok]):
        print("\n✅ Your NLP environment is ready!")
        print("   When you open Jupyter, select 'Python (Medical NLP)' kernel")
    else:
        print("\n⚠️ Some setup steps failed. Please check the messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()