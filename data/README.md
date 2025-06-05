# Medical Speech, Transcription, and Intent Dataset

This directory contains the Medical Speech dataset used for the multimodal medical diagnosis project.

## Dataset Structure

- `overview-of-recordings.csv`: Contains metadata about all recordings
- `recordings/`: Directory containing audio recordings
  - `test/`: Test set recordings
  - `train/`: Training set recordings
  - `validate/`: Validation set recordings

## Important Note

The audio files are not tracked in Git due to their large size. To obtain the dataset:

1. Run the `download_medical_speech_dataset.py` script from the project root:

   ```
   python download_medical_speech_dataset.py
   ```

2. Alternatively, download the dataset manually from the original source and place the files in the appropriate directories.

## Dataset Information

This dataset contains audio recordings of medical speech with corresponding transcriptions and intent labels. It is used for training models to classify medical symptoms, diagnoses, and intents from speech.
