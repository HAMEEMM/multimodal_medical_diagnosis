
import os
import zipfile
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import time
from tqdm import tqdm
from colorama import init, Fore, Style
import sys

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def create_spinning_cursor():
    """Create a spinning cursor animation."""
    while True:
        for cursor in '|/-\\':
            yield cursor

def download_with_progress(api, dataset, download_path):
    """
    Download Kaggle dataset with a progress bar and spinner.
    
    Args:
        api (KaggleApi): Authenticated Kaggle API instance
        dataset (str): Dataset identifier
        download_path (str): Path to download the dataset
    
    Returns:
        str: Path to the downloaded zip file
    """
    # Prepare filename and full path
    zip_filename = f"{dataset.split('/')[-1]}.zip"
    full_zip_path = os.path.join(download_path, zip_filename)
    
    # Create a progress bar
    print(Fore.CYAN + "🔍 Preparing to download dataset..." + Style.RESET_ALL)
    
    # Use a spinner to show activity
    spinner = create_spinning_cursor()
    
    # Custom progress callback
    def progress_callback(chunk_size, total_size, total_written):
        if total_size > 0:
            percent = total_written * 100 / total_size
            sys.stdout.write(f"\r{Fore.YELLOW}Downloading: {next(spinner)} {percent:.1f}% complete{Style.RESET_ALL}")
            sys.stdout.flush()
    
    try:
        # Download the dataset with progress tracking
        print(Fore.GREEN + "📦 Initiating dataset download..." + Style.RESET_ALL)
        api.dataset_download_files(
            dataset, 
            path=download_path, 
            quiet=False, 
            progress=progress_callback
        )
        
        print("\n" + Fore.GREEN + "✅ Download complete!" + Style.RESET_ALL)
        return full_zip_path
    
    except Exception as e:
        print(Fore.RED + f"❌ Download failed: {e}" + Style.RESET_ALL)
        return None

def unzip_with_progress(zip_path, extract_path):
    """
    Unzip file with a progress bar.
    
    Args:
        zip_path (str): Path to the zip file
        extract_path (str): Path to extract the contents
    """
    print(Fore.CYAN + "🗂️ Extracting dataset..." + Style.RESET_ALL)
    
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files to extract
        file_list = zip_ref.namelist()
        
        # Create a progress bar for extraction
        with tqdm(
            total=len(file_list), 
            desc=Fore.GREEN + "Extracting" + Style.RESET_ALL, 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_path)
                pbar.update(1)
    
    print(Fore.GREEN + "✅ Extraction complete!" + Style.RESET_ALL)

def download_and_unzip_medical_speech_dataset(download_path):
    """
    Main function to download and unzip the medical speech dataset.
    
    Args:
        download_path (str): Full path where dataset will be downloaded and extracted
    """
    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)
    
    try:
        # Initialize the Kaggle API
        print(Fore.CYAN + "🔐 Authenticating Kaggle API..." + Style.RESET_ALL)
        api = KaggleApi()
        api.authenticate()
        
        # Dataset details
        dataset = "paultimothymooney/medical-speech-transcription-and-intent"
        
        # Download with progress
        zip_file = download_with_progress(api, dataset, download_path)
        
        if zip_file:
            # Unzip with progress
            unzip_with_progress(zip_file, download_path)
            
            # Optional: Remove the zip file after extraction
            os.remove(zip_file)
            print(Fore.GREEN + "🎉 Dataset successfully downloaded and extracted!" + Style.RESET_ALL)
        
    except Exception as e:
        print(Fore.RED + f"❌ An error occurred: {e}" + Style.RESET_ALL)

# Specify the exact path from your screenshot
download_path = r"G:\Msc\NCU\Doctoral Record\multimodal_medical_diagnosis\data"

# Run the download and unzip function
if __name__ == "__main__":
    # Add a welcome message
    print(Fore.CYAN + "🚀 Medical Speech Dataset Downloader" + Style.RESET_ALL)
    print(Fore.YELLOW + "Please be patient. Download may take some time." + Style.RESET_ALL)
    
    # Pause briefly to let the user read the message
    time.sleep(2)
    
    # Start the download
    download_and_unzip_medical_speech_dataset(download_path)
