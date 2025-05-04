"""  
NLTK Setup Utility  
------------------  
This module provides functions to download and check NLTK resources needed for the project.  
"""  

import os  
import sys  
from pathlib import Path  


def download_nltk_resources(quiet=False):  
    """  
    Download required NLTK resources if they don't exist already.  
    
    Parameters:  
    -----------  
    quiet : bool  
        If True, suppress download messages  
    
    Returns:  
    --------  
    bool  
        True if all resources are available, False otherwise  
    """  
    try:  
        import nltk  
        
        resources = ['punkt', 'stopwords', 'wordnet']  
        all_resources_available = True  
        
        for resource in resources:  
            try:  
                # Check if the resource exists  
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')  
                if not quiet:  
                    print(f"✓ NLTK resource '{resource}' is already available")  
            except LookupError:  
                # Download the resource  
                if not quiet:  
                    print(f"Downloading NLTK resource '{resource}'...")  
                nltk.download(resource, quiet=quiet)  
                all_resources_available = False  
        
        if all_resources_available and not quiet:  
            print("All required NLTK resources are available.")  
        
        return True  
    
    except Exception as e:  
        print(f"Error setting up NLTK resources: {str(e)}")  
        return False  


def check_nltk_resources():  
    """  
    Check if all required NLTK resources are available.  
    
    Returns:  
    --------  
    bool  
        True if all resources are available, False otherwise  
    """  
    try:  
        import nltk  
        
        resources = ['punkt', 'stopwords', 'wordnet']  
        all_available = True  
        
        for resource in resources:  
            try:  
                # Check if the resource exists  
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')  
            except LookupError:  
                print(f"NLTK resource '{resource}' is not available.")  
                all_available = False  
        
        return all_available  
    
    except Exception as e:  
        print(f"Error checking NLTK resources: {str(e)}")  
        return False  


if __name__ == "__main__":  
    # If run as a script, download all resources  
    download_nltk_resources(quiet=False)  