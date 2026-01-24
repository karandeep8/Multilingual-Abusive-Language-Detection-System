# Multilingual Abusive Language Detection System
https://medium.com/@karandeepsaluja73/multilingual-abusive-language-detection-system-f6356cfde7d9

This repository contains a production-ready system for detecting abusive language in Hindi and Punjabi audio files. The system employs a concept-based multilingual approach, enabling seamless detection across languages by focusing on shared abusive concepts rather than direct translations. It integrates speech-to-text transcription with advanced text classification, optimized for noisy environments like audio recordings with background interference.

The core script (\`whole_code_final_abusive_detection.py\`) handles both training and inference pipelines, making it suitable for end-to-end development and deployment.

# Table of Contents


- [Features](#features)
- [Approach](#approach)
- [Usage](#usage)
- [Dataset and Dictionaries](#dataset-and-dictionaries)
- [Contributing](#contributing)

# Features


- **Multilingual Support**: Detects abusive content in Hindi (Devanagari script), Punjabi (Gurmukhi script), and their transliterations, treating equivalent terms as unified concepts.
- **Audio Processing**: Handles various audio formats (e.g., MP3, WAV) with noise reduction and volume normalization for robust transcription in noisy settings.
- **Timestamped Detection**: Identifies precise time segments of abusive content in audio files.
- **Threshold Optimization**: Minimizes false positives by dynamically tuning decision thresholds based on validation performance.
- **Synthetic Data Augmentation**: Generates training examples to reinforce conceptual equivalence across languages.
- **Tiered Classification**: Reduces errors by combining model confidence with dictionary-based evidence during inference.
- **Production-Ready**: Includes silent installation of dependencies, GPU/CPU fallback, and clean output for logging.

# Approach


The system is designed around a **concept-based detection paradigm**, which avoids the pitfalls of language-specific models or translation-based methods. Instead of translating text between Hindi and Punjabi, it maps equivalent abusive expressions (e.g., "भोसड़ी" in Hindi and "ਭੋਸੜੀ" in Punjabi) to shared conceptual meanings (e.g., "damaged vagina"). This allows a single model to learn patterns that generalize across scripts and languages.

## 1. **Data Preparation and Augmentation**

- **Swear Word Dictionary Management**: A centralized dictionary aggregates abusive terms from CSV sources, including Hindi transliterations, Devanagari script, Gurmukhi script, and multi-word phrases. English meanings serve as a bridge to link cross-language equivalents, enabling concept-level mapping.
- **Synthetic Data Generation**: To address data scarcity and reinforce conceptual links, synthetic sentences are created using language-specific templates (e.g., "तुम {} हो" in Hindi or "ਤੁਸੀਂ {} ਹੋ" in Punjabi). Abusive terms are inserted into these templates, producing balanced, labeled examples. This augments the training dataset by approximately 5%, ensuring the model learns that similar concepts in different languages warrant the same classification.

## 2. **Training Pipeline**

- **Model Selection**: Utilizes MuRIL (Multilingual Representations for Indian Languages), a pre-trained model from Google optimized for 17 Indian languages. It is fine-tuned for binary classification (abusive vs. non-abusive) using transfer learning, starting from its broad language understanding and adapting to the specific task.
- **Data Processing**: The combined original and synthetic dataset is tokenized, padded/truncated to a fixed length (128 tokens), and formatted for efficient batching. Training occurs over 3 epochs with a learning rate of 2e-5, mixed precision (FP16) for speed, and warmup scheduling to stabilize early training.
- **Evaluation During Training**: Validation loss is monitored after each epoch to select the best checkpoint, preventing overfitting.
- **Threshold Optimization**: Post-training, probabilities from the validation set are analyzed across thresholds (0.50 to 0.95). The optimal threshold is chosen to minimize false positives while maintaining recall above 75%, using metrics like precision, recall, F1-score, and false positive rate. This conservative strategy prioritizes avoiding wrongful flags in sensitive applications.

## 3. **Inference Pipeline**

- **Audio Preprocessing**: Audio files are loaded universally (supporting multiple formats), converted to mono, resampled to 16kHz, and processed with FFT-based spectral subtraction for noise reduction (targeting stationary noise like engine hum). Volume is normalized and boosted adaptively to an RMS level of ~0.225, enhancing transcription accuracy in low-volume recordings.
- **Chunking and Transcription**: Audio is divided into overlapping chunks (e.g., 3-second windows with 0.5-second overlap) to enable timestamped detection. Each chunk is transcribed using Whisper (preferring Large-v3 for superior Hindi/Punjabi handling), with filters to reject hallucinations (e.g., repetitive gibberish or encoding artifacts).
- **Text Classification**: Transcribed text is classified using the fine-tuned MuRIL model. A dictionary check provides a confidence boost (up to 20%) for matched abusive terms.
- **Tiered Decision Logic**: To further reduce false positives, classification uses a multi-tier system:
- High confidence (>90%): Requires dictionary match to confirm abusive; otherwise, treated as non-abusive (e.g., likely gibberish).
- Medium confidence (75-90%): Similarly requires dictionary evidence.
- Low confidence (<75%): Defaults to non-abusive.
- **Output**: Abusive segments are returned with timestamps, confidence scores, and matched words, allowing precise localization.

This approach ensures high accuracy (e.g., ~95% on text evaluation) while being robust to variations in script, accent, and noise. It emphasizes conceptual understanding over literal word matching, making it extensible to additional Indian languages.


# Usage


## Training Mode

Run the script to train the model:

- Loads data from specified CSVs.
- Generates synthetic samples.
- Trains and saves the model to `./concept_model`.
- Optimizes and saves the threshold.

## Inference Mode

The script includes an inference section for audio detection:
- Update paths in the `main()` function (e.g., audio file, dictionaries).
- Run to process an audio file and output timestamped abusive segments.

Example output:
```
FOUND 2 ABUSIVE SEGMENT(S):

1. Time: [1.50s - 4.50s]
   Confidence: 92.3%
   Classification: high_conf_dict:['बहनचोद']
   Transcription: "तुम बहनचोद हो"
```


# Dataset and Dictionaries


- **Training Data**: CSVs with columns `text` (Hindi/Punjabi sentences) and `label` (0: abusive, 1: non-abusive). Example files: `hindi_train.csv`, `hindi_val.csv`, `hindi_test.csv`.
- **Dictionaries**: CSVs for swear words and phrases (e.g., `hindi_swears.csv` with transliterations and meanings).
- **Audio Dataset**: Binary-labeled audio files for testing (e.g., in `/audio_files`).

Note: Datasets are not included; provide your own or use public abusive language datasets for Indian languages.

# Contributing


Contributions are welcome! Please:
- Open an issue for bugs or feature requests.
- Submit pull requests with improvements (e.g., support for more languages).
- Focus on enhancing the concept-based approach.
