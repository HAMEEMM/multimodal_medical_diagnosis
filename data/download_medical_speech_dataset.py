import os  
import requests  
import zipfile  
import kaggle  
from kaggle.api.kaggle_api_extended import KaggleApi  

def setup_kaggle_credentials():  
    """  
    Setup Kaggle API credentials  
    Ensure you have kaggle.json in ~/.kaggle/ directory  
    """  
    # Path to Kaggle API credentials  
    kaggle_config_dir = os.path.expanduser('~/.kaggle/')  
    os.makedirs(kaggle_config_dir, exist_ok=True)  
    
    # Recommend manual credential setup  
    if not os.path.exists(os.path.join(kaggle_config_dir, 'kaggle.json')):  
        print("❌ Kaggle API credentials not found!")  
        print("Please follow these steps:")  
        print("1. Go to Kaggle > Account > Create New API Token")  
        print("2. Download kaggle.json")  
        print("3. Place kaggle.json in ~/.kaggle/ directory")  
        print("4. Set file permissions: chmod 600 ~/.kaggle/kaggle.json")  
        raise FileNotFoundError("Kaggle API credentials missing")  

def download_dataset(dataset_name, download_path):  
    """  
    Download dataset using Kaggle API  
    
    Args:  
        dataset_name (str): Kaggle dataset identifier  
        download_path (str): Local directory to save dataset  
    """  
    try:  
        # Initialize Kaggle API  
        api = KaggleApi()  
        api.authenticate()  
        
        # Create download directory if not exists  
        os.makedirs(download_path, exist_ok=True)  
        
        # Download dataset  
        print(f"📦 Downloading dataset: {dataset_name}")  
        api.dataset_download_files(  
            dataset_name,   
            path=download_path,   
            unzip=True  # Use Kaggle's built-in unzip  
        )  
        
        print("✅ Dataset downloaded successfully!")  
    
    except Exception as e:  
        print(f"❌ Download failed: {e}")  
        raise  

def clean_and_organize_dataset(download_path):  
    """  
    Clean and organize downloaded dataset  
    
    Args:  
        download_path (str): Path to downloaded dataset  
    """  
    # List all files in download directory  
    files = os.listdir(download_path)  
    
    # Remove any unnecessary zip files  
    for file in files:  
        if file.endswith('.zip'):  
            os.remove(os.path.join(download_path, file))  
    
    # List subdirectories  
    subdirs = [d for d in os.listdir(download_path)
            if os.path.isdir(os.path.join(download_path, d))]  
    
    # Check for nested directories (double unzip issue)  
    if len(subdirs) > 1:  
        print("⚠️ Multiple subdirectories detected. Consolidating...")  
        
        # Merge contents of all subdirectories  
        for subdir in subdirs:  
            subdir_path = os.path.join(download_path, subdir)  
            
            # Move files from nested directory to main download path  
            for item in os.listdir(subdir_path):  
                src = os.path.join(subdir_path, item)  
                dst = os.path.join(download_path, item)  
                
                # Move files, overwriting if necessary  
                if os.path.isfile(src):  
                    os.rename(src, dst)  
            
            # Remove empty subdirectory  
            os.rmdir(subdir_path)  
    
    print("✅ Dataset organized successfully!")  

def validate_dataset(download_path):  
    """  
    Validate downloaded dataset structure  
    
    Args:  
        download_path (str): Path to downloaded dataset  
    """  
    # Check for critical files  
    required_files = [  
        'overview-of-recordings.csv',  
        'recordings'  
    ]  
    
    missing_files = [  
        file for file in required_files   
        if not os.path.exists(os.path.join(download_path, file))  
    ]  
    
    if missing_files:  
        print("❌ Missing critical dataset components:")  
        for file in missing_files:  
            print(f"   - {file}")  
        raise FileNotFoundError("Incomplete dataset download")  
    
    # Display dataset structure  
    print("\n📂 Dataset Structure:")  
    for root, dirs, files in os.walk(download_path):  
        level = root.replace(download_path, '').count(os.sep)  
        indent = ' ' * 4 * level  
        print(f"{indent}📁 {os.path.basename(root)}/")  
        sub_indent = ' ' * 4 * (level + 1)  
        for file in files:  
            print(f"{sub_indent}📄 {file}")  

def main(dataset_name="paultimothymooney/medical-speech-transcription-and-intent"):  
    """  
    Main execution function for dataset download  
    
    Args:  
        dataset_name (str): Kaggle dataset identifier  
    """  
    # Define download path  
    download_path = os.path.join(  
        'G:',   
        'Msc',   
        'NCU',   
        'Doctoral Record',   
        'multimodal_medical_diagnosis',   
        'data',   
        'Medical Speech, Transcription, and Intent'  
    )  
    
    try:  
        # Setup Kaggle credentials  
        setup_kaggle_credentials()  
        
        # Download dataset  
        download_dataset(dataset_name, download_path)  
        
        # Clean and organize dataset  
        clean_and_organize_dataset(download_path)  
        
        # Validate dataset  
        validate_dataset(download_path)  
        
        print("\n🎉 Dataset successfully downloaded and prepared!")  
    
    except Exception as e:  
        print(f"❌ Dataset preparation failed: {e}")  

if __name__ == "__main__":  
    main()  