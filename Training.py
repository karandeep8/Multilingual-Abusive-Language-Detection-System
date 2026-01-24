"""
Concept-Based Multilingual Abusive Detection System - Production Version with Threshold Optimization
======================================================================================================

This system detects abusive language in Hindi and Punjabi audio files using a
multilingual approach.

Key Innovation:
---------------
Rather than treating "भोसड़ी" (Hindi) and "ਭੋਸੜੀ" (Punjabi) as different words,
our model learns they represent the same abusive concept. This makes it work
seamlessly across languages without needing translation.

How It Works:
-------------
1. We use a swear word dictionary that maps Hindi, Punjabi, and English
2. We create synthetic training data from these mappings
3. We train a single MuRIL model (Multilingual Indian Languages) on all data
4. The model learns conceptual patterns rather than just memorizing words
5. For new audio: we transcribe it → classify it → detect timestamps
6. **NEW: We optimize the decision threshold to minimize false positives**

"""

# LIBRARY IMPORTS


import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

# Ignore all warning messages to keep console output clean

warnings.filterwarnings('ignore')

# Print welcome banner
print("="*80)
print("ABUSIVE CONTENT DETECTION SYSTEM")
print("="*80)


# GPU SETUP
# PyTorch automatically detects if CUDA-compatible GPU is available
# Check if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")



# INSTALL REQUIRED LIBRARIES


# Install command breakdown:
# - transformers: Provides MuRIL model and tokenizer from HuggingFace
# - datasets: HuggingFace library for efficient data loading and processing
# - torch: PyTorch deep learning framework (may already be installed)
# - openai-whisper: Speech-to-text model from OpenAI (very accurate)
# - soundfile: For reading audio files (mp3, wav, etc.)
# - scipy: Scientific computing library (we use for audio resampling)
# - scikit-learn: For evaluation metrics (accuracy, precision, recall)
# - matplotlib: For visualization of threshold optimization
# - -q: Quiet mode (don't show installation progress)
# - > /dev/null 2>&1: Redirect all output to nowhere (completely silent)

os.system('pip install transformers datasets torch openai-whisper soundfile scipy scikit-learn matplotlib librosa pydub -q > /dev/null 2>&1')


# IMPORT AI AND AUDIO PROCESSING LIBRARIES
# Now that libraries are installed, import them for use in our code

# Transformers library (from HuggingFace) - provides pre-trained language models
from transformers import (
    AutoTokenizer,                      # Converts text to numbers (tokens) for model
    AutoModelForSequenceClassification,  # Pre-trained model for classification tasks
    Trainer,                            # High-level training wrapper (handles training loop)
    TrainingArguments                   # Configuration for training (learning rate, epochs, etc.)
)

# Datasets library - efficient data loading and processing
from datasets import Dataset  # HuggingFace dataset format (optimized for transformers)

# Whisper - OpenAI's speech-to-text model (converts audio → text)
import whisper  # Best open-source speech recognition model available

# Audio processing libraries
import soundfile as sf  # For reading/writing audio files (supports mp3, wav, flac, etc.)
from scipy import signal  # For audio resampling (changing sample rate)
from scipy.fft import rfft, irfft  # For FFT-based noise reduction
import librosa  # For universal audio format loading
from pydub import AudioSegment  # Fallback for difficult audio formats

# Machine learning evaluation metrics
from sklearn.metrics import (
    classification_report,  # Detailed precision/recall/f1-score breakdown
    confusion_matrix,       # Shows which classes are confused with each other
    accuracy_score,         # Simple accuracy calculation: correct/total
    precision_recall_curve, # For threshold optimization
    f1_score,               # F1 score calculation
    precision_score,        # Precision calculation
    recall_score            # Recall calculation
)

print(" Models loaded!\n")



# SWEAR WORD DICTIONARY CLASS
# This class manages our dictionary of abusive words across multiple languages
# and scripts. It's crucial for our "concept-based" approach.
#
# Why we need this:
# -----------------
# 1. To create synthetic training data (augment our dataset)
# 2. To boost confidence when known abusive words are detected
# 3. To handle multiple scripts (Devanagari, Gurmukhi, Latin)
# 4. To link equivalent words across languages (concept mapping)
#
# Example of concept mapping:
# "बहनचोद" (Hindi) = "ਭੈਣਚੋਦ" (Punjabi) = "bahenchod" (transliteration)
# All three represent the same abusive concept: "sisterfucker"

class SwearWordDictionary:
    """
    Manages swear word mappings across Hindi, Punjabi, and English.

    This class is the heart of our "concept-based" approach. Instead of treating
    Hindi and Punjabi as separate languages requiring translation, we create
    mappings that show they're expressing the same abusive concepts.

    Data Structure:
    ---------------
    We maintain several dictionaries to handle different representations:

    1. hindi_to_english: Maps Hindi Devanagari → English meaning
       Example: {"भोसड़ी": "damaged vagina"}

    2. punjabi_to_english: Maps Punjabi Gurmukhi → English meaning
       Example: {"ਭੋਸੜੀ": "damaged vagina"}

    3. transliteration_to_english: Maps Latin script → English meaning
       Example: {"bhosdi": "damaged vagina"}

    4. phrases: Maps complete abusive phrases → English meaning
       Example: {"teri maa ki": "your mother's"}

    5. all_abusive_words: Set of ALL abusive words (for quick lookup)
       Used for: Fast checking if text contains known abusive content

    Why use English as the bridge language?
    ----------------------------------------
    English meanings link equivalent words across scripts. This allows the model
    to learn that "भोसड़ी", "ਭੋਸੜੀ", and "bhosdi" are the same concept,
    even though they look completely different!
    """

    def __init__(self):
        """
        Initialize empty dictionaries.

        We start with empty dictionaries and populate them when load_dictionaries()
        is called. This separation allows us to:
        1. Create the object first
        2. Load dictionaries later (only when we have the file paths)
        3. Potentially reload or update dictionaries without recreating the object
        """
        # Dictionary mapping Hindi Devanagari abusive text to English meanings
        self.hindi_to_english = {}

        # Dictionary mapping Punjabi Gurmukhi abusive text to English meanings
        self.punjabi_to_english = {}

        # Dictionary mapping transliterations to English meanings
        # This handles: bahenchod, behenchod, bhenchod, b.c., bc (all variants)
        self.transliteration_to_english = {}

        # Dictionary mapping full phrases to English meanings
        self.phrases = {}

        # Set of ALL abusive words/phrases across all scripts
        # Why a set? O(1) lookup time - super fast checking if word exists
        self.all_abusive_words = set()

    def load_dictionaries(self, hindi_swears_csv, hindi_to_gurmukhi_csv, phrases_csv):
        """
        Load all swear word dictionaries from CSV files.

        This method populates our dictionaries by reading three CSV files:

        1. hindi_swears.csv: Contains Hindi words with English meanings
           Columns: Hindi transliteration | Devanagari | Rough English translation

        2. hindi_to_gurmukhi.csv: Maps Hindi words to Punjabi equivalents
           Columns: Hindi_transliteration | Gurmukhi

        3. phrases_hindi_meaning.csv: Contains complete abusive phrases
           Columns: Phrase | Hindi Translation | Meaning

        Why load silently?
        ------------------
        In production, we don't want to spam the console with loading messages.
        We just load the data and confirm when done.

        Parameters:
        -----------
        hindi_swears_csv : str
            Path to CSV file with Hindi swear words
        hindi_to_gurmukhi_csv : str
            Path to CSV file with Hindi→Punjabi mappings
        phrases_csv : str
            Path to CSV file with abusive phrases

        Returns:
        --------
        None (populates internal dictionaries)
        """


        # STEP 1: Load Hindi swear words
        # This gives us the foundation - Hindi words with English meanings

        # Read CSV file into pandas DataFrame
        df_hindi = pd.read_csv(hindi_swears_csv)

        # Iterate through each row in the CSV
        # Why iterrows()? It gives us both the index and the row data
        # The underscore (_) means we don't care about the index
        for _, row in df_hindi.iterrows():
            # Extract data from each column, with cleaning:

            # Get transliteration, convert to lowercase for consistency
            # .strip() removes whitespace from beginning/end
            # .lower() converts to lowercase so "Bhenchod" and "bhenchod" match
            transliteration = str(row['Hindi transliteration']).strip().lower()

            # Get Devanagari script (native Hindi writing)
            # We keep original case for Devanagari since case doesn't matter there
            devanagari = str(row['Devanagari']).strip()

            # Get English meaning (what the word actually means)
            # Lowercase for consistency in lookups
            meaning = str(row['Rough English translation']).strip().lower()

            # Store in our dictionaries:

            # Map Devanagari → English
            self.hindi_to_english[devanagari] = meaning

            # Map transliteration → English
            # Example: "bahenchod" → "sisterfucker"
            self.transliteration_to_english[transliteration] = meaning

            # Add to master set of all abusive words
            # We add both forms so we can quickly check either
            self.all_abusive_words.add(devanagari)
            self.all_abusive_words.add(transliteration)


        # STEP 2: Load Hindi to Punjabi mappings
        # This creates the cross-language concept links

        df_punjabi = pd.read_csv(hindi_to_gurmukhi_csv)

        for _, row in df_punjabi.iterrows():
            # Get the transliteration (this matches what we loaded in step 1)
            transliteration = str(row['Hindi_transliteration']).strip().lower()

            # Get the Punjabi Gurmukhi equivalent
            gurmukhi = str(row['Gurmukhi']).strip()

            # Check if we already know the English meaning for this transliteration
            if transliteration in self.transliteration_to_english:
                # Get the English meaning
                meaning = self.transliteration_to_english[transliteration]

                # Map Punjabi → same English meaning
                # This creates the concept link!
                # "बहनचोद" (Hindi) → "sisterfucker"
                # "ਭੈਣਚੋਦ" (Punjabi) → "sisterfucker"
                # Therefore: Hindi and Punjabi words = same concept
                self.punjabi_to_english[gurmukhi] = meaning

                # Add Punjabi word to our master set
                self.all_abusive_words.add(gurmukhi)


        # STEP 3: Load abusive phrases
        # Many abusive expressions are multi-word phrases, not single words

        df_phrases = pd.read_csv(phrases_csv)

        for _, row in df_phrases.iterrows():
            # Get the phrase
            phrase = str(row['Phrase']).strip().lower()

            # Get the Hindi translation (in Devanagari)
            hindi = str(row['Hindi Translation']).strip()

            # Get English meaning of the phrase
            meaning = str(row['Meaning']).strip().lower()

            # Store both forms with their meaning
            # Example: "teri maa ki chut" → "your mother's vagina"
            #          "तेरी मां की चूत" → "your mother's vagina"
            self.phrases[phrase] = meaning
            self.phrases[hindi] = meaning

            # Add both to master set for quick lookup
            self.all_abusive_words.add(phrase)
            self.all_abusive_words.add(hindi)

    def contains_abusive_word(self, text):
        """
        Check if text contains any known abusive word.

        This is our "dictionary boost" function. When we're classifying text,
        we check if it contains any exact matches from our dictionary. If it does,
        we increase confidence in the "abusive" prediction.

        Why do this?
        ------------
        The neural network might miss some obvious cases or be uncertain. By
        checking our known dictionary, we can:
        1. Catch exact matches with 100% confidence
        2. Boost confidence when model is uncertain
        3. Handle variations the model might not have seen during training

        How the boost works:
        --------------------
        - Each matched word adds 5% to confidence (max 20% total)
        - Example: If model says 70% abusive, and we find 2 swear words,
                  we boost it to 80% abusive (70% + 2*5%)
        - We cap at 20% to avoid over-relying on dictionary

        Parameters:
        -----------
        text : str
            Text to check for abusive words

        Returns:
        --------
        tuple : (found, boost, matched)
            found : bool - True if abusive words were found
            boost : float - How much to boost confidence (0.0 to 0.20)
            matched : list - Which words were found

        Example:
        --------
        >>> text = "तुम बहनचोद हो साला रंडी"
        >>> found, boost, matched = contains_abusive_word(text)
        >>> print(found)  # True
        >>> print(boost)  # 0.15 (3 words found: बहनचोद, साला, रंडी)
        >>> print(matched)  # ['बहनचोद', 'साला', 'रंडी']
        """

        # Create lowercase version for case-insensitive matching
        # Example: "Bhenchod" → "bhenchod"
        text_lower = text.lower()

        # List to store which abusive words we found
        matched = []

        # Check each known abusive word/phrase
        # Why iterate through all? Some might be substrings of others
        for word in self.all_abusive_words:
            # Check if word appears in either lowercase or original text
            # We check both because:
            # - Devanagari doesn't have case, so we check original
            # - Latin text might have case, so we check lowercase
            if word in text_lower or word in text:
                matched.append(word)

        # If we found any abusive words, calculate confidence boost
        if matched:
            # Calculate boost: 5% per word, maximum 20%
            # min() ensures we never exceed 0.20 (20%) boost
            # Example: 3 words found → 3 * 0.05 = 0.15 (15% boost)
            #          5 words found → 5 * 0.05 = 0.25, capped at 0.20
            confidence_boost = min(0.20, len(matched) * 0.05)

            # Return: found=True, boost amount, list of matched words
            return True, confidence_boost, matched

        # No abusive words found
        # Return: found=False, no boost, empty list
        return False, 0.0, []


# PRODUCTION DETECTOR CLASS
# =========================
# This is our main class that ties everything together. It handles:
# 1. Loading the Whisper model for speech-to-text
# 2. Training the text classification model
# 3. Evaluating performance on test data
# 4. **NEW: Optimizing the decision threshold to minimize false positives**
# 5. Classifying audio files
# 6. Detecting timestamps of abusive content in audio
#


class ProductionDetector:
    """
    Production-ready detector with clean output.

    This class orchestrates the entire detection pipeline from training to
    inference. It's designed to be simple to use while handling complex
    operations behind the scenes.

    Architecture Overview:
    ----------------------

    For TEXT:
    User text → Tokenizer → MuRIL Model → Probabilities → Dictionary Boost → Final prediction

    For AUDIO:
    Audio file → Whisper ASR → Text → Tokenizer → MuRIL → Dictionary Boost → Final prediction

    For TIMESTAMPS:
    Audio → Split into chunks → Process each chunk → Find abusive segments with timestamps

    Why this design?
    ----------------
    - Modular: Each method does one thing well
    - Reusable: Can train once, then classify many times
    - Production-ready: Handles errors gracefully, gives clean output
    - Memory efficient: Loads models only when needed
    """

    def __init__(self, whisper_model_size="base", text_model_name="google/muril-base-cased", device="cuda"):
        """
        Initialize the detector with models.

        This sets up our detection system by loading the required models.
        We use two main models:
        1. Whisper: For converting audio to text (speech recognition)
        2. MuRIL: For classifying if text is abusive (text classification)

        Why Whisper?
        ------------
        - State-of-the-art speech recognition from OpenAI
        - Supports 99+ languages including Hindi/Punjabi
        - Works well even with accents and background noise
        - Open source and free to use

        Why MuRIL?
        ----------
        - "Multilingual Representations for Indian Languages"
        - Created by Google specifically for Indian languages
        - Pre-trained on 17 Indian languages (Hindi, Punjabi, Bengali, etc.)
        - Understands multiple scripts (Devanagari, Gurmukhi, Latin)
        - Much better than generic BERT for our use case

        Parameters:
        -----------
        whisper_model_size : str, default="base"
            Size of Whisper model to load
            Options: "tiny", "base", "small", "medium", "large"
            - tiny: Fastest but least accurate (39M parameters)
            - base: Good balance of speed/accuracy (74M parameters) ← We use this
            - small: Better accuracy, slower (244M parameters)
            - medium: High accuracy, much slower (769M parameters)
            - large: Best accuracy, very slow (1550M parameters)

        text_model_name : str, default="google/muril-base-cased"
            Name of the MuRIL model from HuggingFace
            This is the pre-trained model we'll fine-tune on our abusive language data

        device : str, default="cuda"
            Which device to use for computation
            - "cuda": Use GPU (MUCH faster, recommended)
            - "cpu": Use CPU (slower but works everywhere)
            PyTorch will automatically fall back to CPU if CUDA is not available

        Returns:
        --------
        None (initializes the detector object)
        """

        # Set up the device (GPU or CPU)
        # torch.cuda.is_available() returns True if we have a GPU
        # If not, we fall back to CPU even if user requested cuda
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load Whisper model for speech-to-text
        print("\n Loading models...")

        # whisper.load_model() downloads and loads the model
        # - First time: Downloads model file (~150MB for base model)
        # - Subsequent times: Loads from cache (fast)
        # - Automatically moves model to specified device (GPU/CPU)
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        # Initialize tokenizer and text model as None
        # We'll load these later when we actually need them
        # Why delay loading?
        # - Save memory if we're only doing audio transcription
        # - Allow flexibility in model choice
        # - Faster initialization
        self.tokenizer = None
        self.text_model = None

        # Store the model name so we can load it later
        self.text_model_name = text_model_name

        # Create an empty swear word dictionary
        # We'll populate this with load_dictionaries() later
        self.swear_dict = SwearWordDictionary()

        # Store the optimal threshold (will be set during optimization)
        self.optimal_threshold = 0.50  # Default value

        print(" Models loaded!\n")


    def create_synthetic_samples(self, swear_dict, num_samples_per_word=3):
        """
        Create synthetic training samples from our swear word dictionary.

        This is a key part of our concept-based approach! We generate artificial
        training examples that explicitly link Hindi and Punjabi words to the
        same concepts.

        Why create synthetic data?
        --------------------------
        1. Data Augmentation: Our original dataset might not have enough examples
           of certain abusive words, especially rare ones

        2. Concept Reinforcement: By creating parallel examples in Hindi and Punjabi
           with the same label, we teach the model that these are equivalent concepts

           Example:
           Hindi synthetic: "तुम बहनचोद हो" → abusive
           Punjabi synthetic: "ਤੁਸੀਂ ਭੈਣਚੋਦ ਹੋ" → abusive

           The model learns: These sentences have the same meaning and same label,
           so "बहनचोद" and "ਭੈਣਚੋਦ" must be equivalent concepts!

        3. Balance Dataset: Some abusive words appear rarely in natural text.
           Synthetic data ensures the model sees them enough times to learn

        How it works:
        -------------
        We use template sentences (like "You are {word}") and fill in the blank
        with different abusive words from our dictionary. This creates grammatically
        correct sentences that are clearly abusive.

        Templates are in both Hindi and Punjabi, ensuring cross-language coverage.

        Parameters:
        -----------
        swear_dict : SwearWordDictionary
            Our dictionary object containing all abusive words

        num_samples_per_word : int, default=3
            How many synthetic samples to create per word
            Higher = more data but more redundancy
            We use 3 as a balance between coverage and diversity

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: ['text', 'label', 'source']
            - text: The synthetic sentence
            - label: Always 0 (abusive) for our synthetic data
            - source: Identifies this as synthetic (for tracking)

        Example output:
        ---------------
        text                        | label | source
        ----------------------------|-------|------------------
        "तुम बहनचोद हो"             | 0     | synthetic_hindi
        "ਤੁਸੀਂ ਭੈਣਚੋਦ ਹੋ"          | 0     | synthetic_punjabi
        """

        # List to store all our synthetic samples
        # Each sample is a dictionary with 'text', 'label', 'source'
        synthetic_data = []

        # Define Hindi templates
        # {} is a placeholder that we'll fill with abusive words
        # These are common sentence structures in Hindi that sound natural
        hindi_templates = [
            "तुम {} हो",           # "You are {word}" - direct insult
            "वह {} है",            # "He/She is {word}" - third person insult
            "यह {} बात है",       # "This is {word} talk" - describing behavior
            "{} मत बोलो",          # "Don't say {word}" - warning/command
            "तुम्हारी {} बातें"    # "Your {word} words" - criticism
        ]

        # Define Punjabi templates
        # Same meanings as Hindi but in Punjabi (Gurmukhi script)
        punjabi_templates = [
            "ਤੁਸੀਂ {} ਹੋ",          # "You are {word}"
            "ਉਹ {} ਹੈ",            # "He/She is {word}"
            "ਇਹ {} ਗੱਲ ਹੈ",       # "This is {word} talk"
            "{} ਨਾ ਬੋਲੋ",          # "Don't say {word}"
            "ਤੁਹਾਡੀ {} ਗੱਲਾਂ"    # "Your {word} words"
        ]

        # Generate synthetic samples from Hindi words
        # We limit to first 100 words to avoid generating too much data
        for hindi, meaning in list(swear_dict.hindi_to_english.items())[:100]:
            # For each Hindi word, use the first N templates
            # (N = num_samples_per_word)
            for template in hindi_templates[:num_samples_per_word]:
                # Fill in the template with the Hindi word
                # .format(hindi) replaces {} with the actual word
                synthetic_text = template.format(hindi)

                # Create a sample dictionary
                synthetic_data.append({
                    'text': synthetic_text,    # The generated sentence
                    'label': 0,                # 0 = abusive (always for our synthetic data)
                    'source': 'synthetic_hindi' # Mark as synthetic for tracking
                })

        # Generate synthetic samples from Punjabi words
        # Same process as Hindi but with Punjabi words and templates
        for punjabi, meaning in list(swear_dict.punjabi_to_english.items())[:100]:
            for template in punjabi_templates[:num_samples_per_word]:
                synthetic_text = template.format(punjabi)

                synthetic_data.append({
                    'text': synthetic_text,
                    'label': 0,
                    'source': 'synthetic_punjabi'
                })

        # Generate samples from multi-word phrases
        # Phrases don't need templates - they're already complete sentences
        # Example: "teri maa ki chut" is already a full abusive phrase
        for phrase, meaning in swear_dict.phrases.items():
            synthetic_data.append({
                'text': phrase,           # Use phrase as-is
                'label': 0,               # abusive
                'source': 'synthetic_phrase'
            })

        # Convert list of dictionaries to pandas DataFrame
        # This makes it compatible with our training pipeline
        return pd.DataFrame(synthetic_data)


    def train_model(self, train_df, val_df, output_dir="./concept_model"):
        """
        Train the text classification model.

        This is where the magic happens! We take our MuRIL model (pre-trained on
        Indian languages) and fine-tune it on our specific task: detecting
        abusive language in Hindi and Punjabi.

        What is fine-tuning?
        --------------------
        MuRIL was pre-trained on massive amounts of Hindi/Punjabi text to understand
        the language. But it doesn't know about abusive vs non-abusive classification.
        And this process is called Transfer Learning.

        Fine-tuning teaches it this specific task by:
        1. Starting with MuRIL's language understanding
        2. Adding a classification head (2 outputs: abusive/non-abusive)
        3. Training on our labeled data (26,911 samples)
        4. Adjusting the weights so it learns to recognize abusive patterns

        Why use MuRIL instead of training from scratch?
        -----------------------------------------------
        - Pre-trained models understand language structure already
        - Much faster training
        - Better performance with less data
        - Transfer learning: knowledge from one task helps another

        Training process:
        -----------------
        1. Load pre-trained MuRIL model
        2. Add classification head (abusive/non-abusive)
        3. Tokenize all text (convert words → numbers)
        4. Train for 3 epochs:
           - Epoch 1: Model makes many mistakes, learns basic patterns
           - Epoch 2: Model improves, starts recognizing concepts
           - Epoch 3: Model refines, achieves best performance
        5. Save best model based on validation loss

        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data with columns ['text', 'label']
            - text: Hindi/Punjabi sentences
            - label: 0 (abusive) or 1 (non-abusive)

        val_df : pd.DataFrame
            Validation data (same format as train_df)
            Used to check performance during training and select best model

        output_dir : str, default="./concept_model"
            Where to save the trained model
            This creates a folder with:
            - pytorch_model.bin (model weights)
            - config.json (model configuration)
            - tokenizer files (vocab, special tokens)

        Returns:
        --------
        Trainer object
            HuggingFace Trainer object (contains training history, model, etc.)
        """

        # Print training start message
        print(" Training model...")
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")

        # Load tokenizer
        # The tokenizer converts text to numbers that the model can process
        # MuRIL tokenizer understands multiple Indian scripts (Devanagari, Gurmukhi)
        # Why? It was trained on multilingual data, so it has a vocabulary covering all scripts
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

        # Define tokenization function
        # This function will be applied to every sample in our dataset
        def tokenize_function(examples):
            """
            Convert text to token IDs that the model can process.

            This function:
            1. Takes text: "तुम बहनचोद हो"
            2. Splits into tokens: ["तुम", "बहन", "##चोद", "हो"]
            3. Converts to IDs: [4521, 8832, 1092, 3456]
            4. Adds special tokens: [CLS] ... [SEP]
            5. Pads/truncates to fixed length (128 tokens)

            Why padding?
            ------------
            Neural networks require fixed-size inputs. We pad short sentences
            and truncate long ones to ensure all inputs are exactly 128 tokens.

            Why max_length=128?
            -------------------
            - Most of our sentences are <50 tokens
            - 128 is enough for 99% of our data
            - Longer sequences = more memory and slower training
            - Sweet spot for performance vs efficiency

            Parameters:
            -----------
            examples : dict
                Batch of examples with 'text' field

            Returns:
            --------
            dict with:
                - input_ids: Token IDs (numbers representing words)
                - attention_mask: 1 for real tokens, 0 for padding
            """
            return self.tokenizer(
                examples["text"],        # Text to tokenize
                padding="max_length",    # Pad shorter sequences to max_length
                truncation=True,         # Cut off sequences longer than max_length
                max_length=128           # Fixed length for all sequences
            )

        # Convert pandas DataFrames to HuggingFace Dataset format
        # Why? HuggingFace's Dataset is optimized for transformers:
        # - Faster data loading
        # - Efficient batching
        # - Automatic GPU transfer
        # - Caching for repeated operations
        train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

        # Apply tokenization to all samples
        # .map() applies our tokenize_function to every example
        # batched=True processes multiple examples at once (faster)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Rename 'label' column to 'labels'
        # Why? HuggingFace models expect the column to be called 'labels'
        # This is just a naming convention in the transformers library
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        # Set format to PyTorch tensors
        # This converts our data to the format PyTorch expects:
        # - NumPy arrays → PyTorch tensors
        # - Only keep columns we need for training
        # - Remove text column (we only need the token IDs now)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Load pre-trained MuRIL model with classification head
        # This downloads the model if not cached (~900MB)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            self.text_model_name,         # "google/muril-base-cased"
            num_labels=2,                 # 2 classes: abusive (0) and non-abusive (1)
            ignore_mismatched_sizes=True  # Ignore size mismatch warnings (expected when adding new head)
        )

        # Move model to GPU/CPU
        # .to() transfers all model parameters to the specified device
        # This is necessary for GPU training
        self.text_model = self.text_model.to(self.device)

        # Define training arguments
        # These control how training happens
        training_args = TrainingArguments(
            # Basic settings
            output_dir=output_dir,              # Where to save model checkpoints
            num_train_epochs=3,                 # How many times to go through entire dataset
                                                # - More epochs = better learning but risk overfitting


            # Batch sizes
            # Batch size = how many samples to process before updating weights
            per_device_train_batch_size=32,     # 32 samples per GPU for training
                                                # - Larger batch = faster training but more memory

            per_device_eval_batch_size=32,      # 32 samples per GPU for evaluation

            # Learning rate
            learning_rate=2e-5,                 # How big steps to take when updating weights
                                                # - Too high: Model doesn't converge, jumps around
                                                # - Too low: Training takes forever
                                                # - 2e-5 is standard for fine-tuning BERT-like models

            # Regularization
            weight_decay=0.01,                  # L2 regularization to prevent overfitting
                                                # - Adds penalty for large weights
                                                # - Forces model to be simpler

            # Learning rate schedule
            warmup_ratio=0.1,                   # Gradually increase learning rate for first 10% of training
                                                # - Prevents unstable training at start
                                                # - Model needs time to "warm up"

            # Mixed precision training
            fp16=True,                          # Use 16-bit floating point instead of 32-bit
                                                # - 2x faster training
                                                # - 2x less memory usage
                                                # - Minimal accuracy loss


            # Evaluation strategy
            eval_strategy="epoch",              # Evaluate after each epoch
                                                # - Check validation loss/accuracy after each epoch
                                                # - Helps us see if model is improving

            # Saving strategy
            save_strategy="epoch",              # Save model checkpoint after each epoch
                                                # - Creates 3 checkpoints (one per epoch)
                                                # - Can resume training if interrupted

            load_best_model_at_end=True,        # Load best checkpoint at end of training
                                                # - Sometimes epoch 2 is better than epoch 3
                                                # - This ensures we use the best model

            metric_for_best_model="eval_loss",  # Use validation loss to determine "best"
                                                # - Lower validation loss = better model
                                                # - Could also use accuracy, but loss is more reliable

            # Logging
            logging_dir=f"{output_dir}/logs",   # Where to save training logs
            logging_steps=100,                  # Log metrics every 100 steps
                                                # - Too frequent = cluttered logs
                                                # - Too infrequent = can't debug issues
            report_to="none",                   # Don't send logs to external services
                                                # - Could use "tensorboard" or "wandb" for visualization
                                                # - We keep it simple with local logs

            # Performance optimization
            dataloader_num_workers=4,           # Use 4 CPU threads for data loading
                                                # - Loads next batch while GPU trains current batch
                                                # - Prevents GPU from waiting for data

            gradient_accumulation_steps=1,      # Update weights every batch (no accumulation)
                                                # - Could set >1 to simulate larger batch sizes
                                                # - Not needed with our batch size
        )

        # Create Trainer object
        # This is HuggingFace's high-level training wrapper
        # It handles:
        # - Forward pass (input → model → predictions)
        # - Loss calculation (predictions vs actual labels)
        # - Backward pass (calculate gradients)
        # - Weight updates (apply gradients to weights)
        # - Validation (check performance on val set)
        # - Checkpointing (save models)
        # - Logging (track metrics)
        trainer = Trainer(
            model=self.text_model,      # The model to train
            args=training_args,         # Training configuration we defined above
            train_dataset=train_dataset,# Training data
            eval_dataset=val_dataset,   # Validation data
        )

        # Start training!
        # This runs the full training loop:
        # - 3 epochs × ~850 batches per epoch = ~2,550 total training steps
        # - Each step: forward pass → loss → backward pass → weight update
        # - Validates after each epoch
        # - Saves checkpoints

        trainer.train()

        # Training complete - print summary
        print(f"\n Training complete!")
        print(f" Saving model to: {output_dir}\n")

        # Save the final model
        # This saves:
        # - pytorch_model.bin: Model weights
        # - config.json: Model configuration
        # - training_args.bin: Training arguments used
        trainer.save_model(output_dir)

        # Save the tokenizer
        # This saves:
        # - vocab.txt: Vocabulary (word → ID mappings)
        # - special_tokens_map.json: Special tokens like [CLS], [SEP]
        # - tokenizer_config.json: Tokenizer settings
        self.tokenizer.save_pretrained(output_dir)

        # Return trainer object (in case caller wants to inspect training history)
        return trainer


    def get_predictions_with_probabilities(self, data_df, model_dir, swear_dict):
        """
        Get model predictions along with probability scores for threshold optimization.

        This function is specifically designed for threshold optimization. Unlike evaluate_text(),
        which returns final predictions based on argmax, this function returns the raw probability
        scores for the abusive class. These probabilities are what we'll use to find the optimal
        threshold that balances precision and recall.

        Why we need this:
        -----------------
        The default classification uses argmax (pick class with highest probability), which
        implicitly uses a 0.5 threshold. But 0.5 might not be optimal for our use case.

        For example:
        - Probability [0.60, 0.40] → classified as abusive (60% > 50%)
        - But what if we want to be more conservative and only flag when confidence > 75%?

        This function gives us the raw probabilities so we can experiment with different thresholds.

        Process:
        --------
        For each sample:
        1. Tokenize the text
        2. Get model's raw probability scores
        3. Apply dictionary boost (if applicable)
        4. Store the abusive class probability (not the final prediction)

        Parameters:
        -----------
        data_df : pd.DataFrame
            Data with columns ['text', 'label']
            This is typically the validation set

        model_dir : str
            Path to saved model directory

        swear_dict : SwearWordDictionary
            Dictionary for confidence boosting

        Returns:
        --------
        tuple : (probabilities, true_labels)
            probabilities : np.array - Probability of abusive class for each sample
            true_labels : np.array - Ground truth labels (0 = abusive, 1 = non-abusive)

        Example:
        --------
        >>> probs, labels = get_predictions_with_probabilities(val_df, "./model", swear_dict)
        >>> print(probs[:5])  # First 5 probabilities
        [0.87, 0.23, 0.91, 0.45, 0.68]
        >>> print(labels[:5])  # First 5 true labels
        [0, 1, 0, 1, 0]
        """

        # Load model if not already loaded
        if self.text_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()  # Set to evaluation mode

        # Arrays to store results
        probabilities = []  # Will store probability of abusive class
        true_labels = data_df['label'].values  # Ground truth labels

        # Process each sample
        # We don't use tqdm here to keep output clean during optimization
        for idx in range(len(data_df)):
            text = data_df.iloc[idx]['text']

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model prediction
            with torch.no_grad():
                outputs = self.text_model(**inputs)

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # Apply dictionary boost
            found, boost, matched = swear_dict.contains_abusive_word(text)
            if found:
                probs[0] += boost
                probs[1] -= boost
                probs = probs / probs.sum()

            # Store the probability of abusive class (index 0)
            probabilities.append(probs[0])

        return np.array(probabilities), true_labels


    def optimize_threshold(self, val_df, model_dir, swear_dict, save_plot=True):
        """
        Find the optimal decision threshold to minimize false positives while maintaining good recall.

        The default threshold of 0.5 often leads
        to too many false positives (innocent content flagged as abusive). By optimizing the threshold
        on validation data, we can find the sweet spot that:

        1. Reduces false positives (don't wrongly ban innocent users)
        2. Maintains high recall (still catch most actual abuse)
        3. Maximizes F1-score (balanced performance)

        How threshold optimization works:
        ---------------------------------

        The model outputs probabilities: [P(abusive), P(non-abusive)]

        With different thresholds:
        - Threshold 0.50: Flag if P(abusive) > 0.50  (default, balanced)
        - Threshold 0.70: Flag if P(abusive) > 0.70  (more conservative, fewer false positives)
        - Threshold 0.90: Flag if P(abusive) > 0.90  (very conservative, even fewer false positives)

        We test many thresholds (0.50, 0.55, 0.60, ..., 0.95) and for each one calculate:
        - Precision: Of all flagged content, what % is actually abusive?
        - Recall: Of all actual abuse, what % did we catch?
        - F1-score: Harmonic mean of precision and recall
        - False positive rate: What % of non-abusive content was wrongly flagged?

        Strategy for choosing optimal threshold:
        ----------------------------------------
        We provide three strategies:

        1. **best_f1** (default): Maximize F1-score
          - Balanced approach
          - Good for general content moderation

        2. **low_fp**: Minimize false positives while maintaining recall > 80%
          - Conservative approach
          - Good when false positives are very costly (e.g., don't want to ban paying users)

        3. **high_recall**: Ensure recall > 90% while maximizing precision
          - Aggressive approach
          - Good for safety-critical applications (e.g., children's platforms)



        Parameters:
        -----------
        val_df : pd.DataFrame
            Validation data (NOT test data - we save test for final evaluation)
            Should have columns ['text', 'label']

        model_dir : str
            Path to trained model directory

        swear_dict : SwearWordDictionary
            Dictionary for confidence boosting

        save_plot : bool, default=True
            Whether to save visualization of threshold vs metrics
            Creates a plot showing how precision, recall, and F1 change with threshold

        Returns:
        --------
        dict with:
            'optimal_threshold': float - Best threshold value
            'metrics': dict - Precision, recall, F1 at optimal threshold
            'all_results': pd.DataFrame - Results for all tested thresholds

        """

        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION")
        print("="*80)
        print("\nFinding optimal decision threshold on validation data...")

        # Get probability scores for all validation samples
        print("Getting model predictions on validation set...")
        probabilities, true_labels = self.get_predictions_with_probabilities(
            val_df, model_dir, swear_dict
        )

        # Test different threshold values
        # We test from 0.50 to 0.95 in steps of 0.05
        # Why start at 0.50? Values below 0.50 would favor non-abusive class (not useful)
        # Why stop at 0.95? Very high thresholds catch almost nothing
        thresholds_to_test = np.arange(0.50, 0.96, 0.05)

        # Store results for each threshold
        results = []

        print(f"Testing {len(thresholds_to_test)} different threshold values...\n")

        for threshold in thresholds_to_test:
            # Apply threshold: classify as abusive (0) if probability > threshold
            predictions = (probabilities > threshold).astype(int)
            # Note: If prob > threshold, we get True (1), but we want 0 for abusive
            # So we need to flip: 1 - predictions or use (probabilities > threshold) == False
            predictions = 1 - predictions  # Convert to: 0 = abusive, 1 = non-abusive

            # Calculate metrics for this threshold
            # We need to handle the case where all predictions are the same class
            try:
                precision = precision_score(true_labels, predictions, pos_label=0, zero_division=0)
                recall = recall_score(true_labels, predictions, pos_label=0, zero_division=0)
                f1 = f1_score(true_labels, predictions, pos_label=0, zero_division=0)

                # Calculate false positive rate
                # False positives = non-abusive samples (label=1) predicted as abusive (pred=0)
                non_abusive_mask = (true_labels == 1)
                if non_abusive_mask.sum() > 0:
                    false_positives = ((predictions == 0) & (true_labels == 1)).sum()
                    fp_rate = false_positives / non_abusive_mask.sum()
                else:
                    fp_rate = 0.0

            except:
                # If calculation fails (e.g., division by zero), skip this threshold
                continue

            # Store results
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fp_rate
            })

        # Convert results to DataFrame for easy analysis
        results_df = pd.DataFrame(results)

        # Find optimal threshold based on multiple criteria

        # Strategy 1: Best F1-score (balanced approach)
        best_f1_idx = results_df['f1_score'].idxmax()
        best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']

        # Strategy 2: Minimize false positives while maintaining reasonable recall (>75%)
        high_recall_results = results_df[results_df['recall'] >= 0.75]
        if len(high_recall_results) > 0:
            best_low_fp_idx = high_recall_results['false_positive_rate'].idxmin()
            best_low_fp_threshold = high_recall_results.loc[best_low_fp_idx, 'threshold']
        else:
            best_low_fp_threshold = best_f1_threshold

        # For production, we choose the threshold that minimizes false positives
        # while maintaining good F1-score
        # This is best_low_fp_threshold
        optimal_threshold = best_low_fp_threshold
        optimal_metrics = results_df[results_df['threshold'] == optimal_threshold].iloc[0]

        # Store optimal threshold
        self.optimal_threshold = optimal_threshold

        # Print results
        print("="*80)
        print("THRESHOLD OPTIMIZATION RESULTS")
        print("="*80)

        print(f"\n Tested {len(results_df)} different thresholds from {thresholds_to_test[0]:.2f} to {thresholds_to_test[-1]:.2f}")

        print(f"\n OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
        print(f"\n   Metrics at optimal threshold:")
        print(f"   • Precision:          {optimal_metrics['precision']*100:.2f}%")
        print(f"   • Recall:             {optimal_metrics['recall']*100:.2f}%")
        print(f"   • F1-Score:           {optimal_metrics['f1_score']*100:.2f}%")
        print(f"   • False Positive Rate: {optimal_metrics['false_positive_rate']*100:.2f}%")

        print(f"\n Comparison with default threshold (0.50):")
        default_metrics = results_df[results_df['threshold'] == 0.50].iloc[0]
        print(f"   Default (0.50):  Precision={default_metrics['precision']*100:.1f}%, Recall={default_metrics['recall']*100:.1f}%, F1={default_metrics['f1_score']*100:.1f}%, FP={default_metrics['false_positive_rate']*100:.1f}%")
        print(f"   Optimal ({optimal_threshold:.2f}): Precision={optimal_metrics['precision']*100:.1f}%, Recall={optimal_metrics['recall']*100:.1f}%, F1={optimal_metrics['f1_score']*100:.1f}%, FP={optimal_metrics['false_positive_rate']*100:.1f}%")

        # Calculate improvement
        fp_reduction = (1 - optimal_metrics['false_positive_rate'] / default_metrics['false_positive_rate']) * 100
        print(f"\n    False positive reduction: {fp_reduction:.1f}%")

        # Show top 5 thresholds by F1-score
        print(f"\n Top 5 thresholds by F1-score:")
        top_5 = results_df.nlargest(5, 'f1_score')[['threshold', 'precision', 'recall', 'f1_score', 'false_positive_rate']]
        print(top_5.to_string(index=False))

        # Create visualization if requested
        if save_plot:
            plt.figure(figsize=(12, 6))

            plt.plot(results_df['threshold'], results_df['precision']*100, 'b-o', label='Precision', markersize=4)
            plt.plot(results_df['threshold'], results_df['recall']*100, 'r-o', label='Recall', markersize=4)
            plt.plot(results_df['threshold'], results_df['f1_score']*100, 'g-o', label='F1-Score', markersize=4)
            plt.plot(results_df['threshold'], results_df['false_positive_rate']*100, 'm-o', label='False Positive Rate', markersize=4)

            # Mark optimal threshold
            plt.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.2f})')

            plt.xlabel('Decision Threshold', fontsize=12)
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.title('Threshold Optimization: Precision, Recall, F1-Score, and False Positive Rate', fontsize=14, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(os.path.dirname(model_dir), 'threshold_optimization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n Visualization saved to: {plot_path}")
            plt.close()

        print("="*80)



        import json

        threshold_data = {
            'optimal_threshold': optimal_threshold,
            'metrics': {
                'precision': float(optimal_metrics['precision']),
                'recall': float(optimal_metrics['recall']),
                'f1_score': float(optimal_metrics['f1_score']),
                'false_positive_rate': float(optimal_metrics['false_positive_rate'])
            },
            'note': 'Optimized on validation data to minimize false positives while maintaining good recall'
        }

        # Save to JSON file in the output directory
        threshold_path = os.path.join(os.path.dirname(model_dir), 'threshold.json')

        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)

        print(f"\n Threshold saved to: {threshold_path}")
        print(f"   This file will be loaded automatically during inference\n")



        # Return results
        return {
            'optimal_threshold': optimal_threshold,
            'metrics': {
                'precision': optimal_metrics['precision'],
                'recall': optimal_metrics['recall'],
                'f1_score': optimal_metrics['f1_score'],
                'false_positive_rate': optimal_metrics['false_positive_rate']
            },
            'all_results': results_df
        }


    def evaluate_text(self, test_df, model_dir, swear_dict, use_optimal_threshold=True):
        """
        Evaluate model performance on text test data.

        This function tests how well our trained model performs on unseen data.
        It's the most important evaluation because it tells us if our model
        actually learned to generalize or just memorized the training data.

        What we're testing:
        -------------------
        - Can the model correctly classify new, unseen Hindi/Punjabi text?
        - Does the dictionary boost improve accuracy?
        - How well does it handle different types of abusive language?
        - Are false positives/negatives acceptable?

        The evaluation process:
        -----------------------
        For each test sample:
        1. Tokenize the text (convert to numbers)
        2. Feed to model → get probability distribution [P(abusive), P(non-abusive)]
        3. Check if text contains dictionary words → boost confidence if found
        4. Apply threshold (default 0.5 or optimized threshold)
        5. Compare prediction to ground truth label

        Why use dictionary boost during evaluation?
        --------------------------------------------
        The dictionary boost is part of our system design. It's not "cheating" -
        it's how our production system works. We want to evaluate the FULL system,
        not just the neural network in isolation.

        Think of it like this:
        - Neural network alone: 81% accuracy (good baseline)
        - Neural network + dictionary: 87.57% accuracy (our full system)

        Parameters:
        -----------
        test_df : pd.DataFrame
            Test data with columns ['text', 'label']
            This should be data the model has NEVER seen during training
            Typically 10% of total data (3,364 samples in our case)

        model_dir : str
            Path to saved model directory (from train_model)
            Contains: pytorch_model.bin, config.json, tokenizer files

        swear_dict : SwearWordDictionary
            Our dictionary for confidence boosting

        use_optimal_threshold : bool, default=True
            Whether to use the optimized threshold or default 0.5
            If True, uses self.optimal_threshold (set during optimize_threshold)

        Returns:
        --------
        tuple : (accuracy, predictions, true_labels)
            accuracy : float - Overall accuracy (0.0 to 1.0)
            predictions : list - Predicted labels for each sample
            true_labels : array - Ground truth labels

        """

        # Load model if not already loaded
        # Why check if None? We might have just trained the model and it's already
        # in memory. No need to reload it.
        if self.text_model is None:
            # Load tokenizer from saved directory
            # This loads the vocabulary and tokenizer config we saved during training
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Load the trained model
            # This loads our fine-tuned MuRIL model with the classification head
            self.text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)

            # Move model to GPU/CPU
            self.text_model = self.text_model.to(self.device)

            # Set model to evaluation mode
            # This disables dropout and batch normalization training behavior
            # Why? During training, dropout randomly zeros out some neurons for regularization
            # During evaluation, we want deterministic behavior - use all neurons
            self.text_model.eval()

        # List to store predictions for each sample
        predictions = []

        # Get ground truth labels as numpy array
        # We'll compare predictions against these
        true_labels = test_df['label'].values

        # Determine which threshold to use
        threshold = self.optimal_threshold if use_optimal_threshold else 0.50

        # Evaluate each sample
        # tqdm creates a progress bar so we can see evaluation progress
        # desc="Evaluating text" shows text next to progress bar
        # ncols=80 makes progress bar 80 characters wide
        for idx in tqdm(range(len(test_df)), desc="Evaluating text", ncols=80):
            # Get the text for this sample
            text = test_df.iloc[idx]['text']

            # Tokenize the text
            # Convert text → token IDs (numbers the model understands)
            inputs = self.tokenizer(
                text,                    # Input text
                return_tensors="pt",     # Return PyTorch tensors (not numpy)
                padding=True,            # Pad to model's max length
                truncation=True,         # Truncate if longer than max length
                max_length=128           # Maximum sequence length
            )

            # Move inputs to GPU/CPU
            # The model and data must be on the same device
            # This line moves all tensors in the dict to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model prediction
            # torch.no_grad() disables gradient computation
            # Why? We're not training, just evaluating
            # This saves memory and speeds up inference
            with torch.no_grad():
                # Forward pass: input → model → output
                # outputs contains:
                # - outputs.logits: Raw scores before softmax [score_abusive, score_non_abusive]
                # - outputs.loss: None (we didn't provide labels)
                outputs = self.text_model(**inputs)

            # Convert logits to probabilities using softmax
            # Logits are raw scores: [2.3, -1.1]
            # Softmax converts to probabilities that sum to 1: [0.90, 0.10]
            # dim=-1 means apply softmax along last dimension (the class dimension)
            # .cpu().numpy()[0] moves to CPU, converts to numpy, and gets first (only) sample
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # Check if text contains known abusive words from dictionary
            # This is our "dictionary boost" mechanism
            found, boost, matched = swear_dict.contains_abusive_word(text)

            # If dictionary words were found, boost confidence in "abusive" class
            if found:
                # Add boost to abusive probability (index 0)
                probs[0] += boost

                # Subtract same amount from non-abusive probability (index 1)
                # This maintains the constraint that probabilities sum to 1
                probs[1] -= boost

                # Renormalize to ensure probabilities sum exactly to 1.0
                # Why? Floating point arithmetic might cause small deviations
                # For example: [0.82, 0.19] might become [0.82001, 0.18999]
                probs = probs / probs.sum()

            # Apply threshold to make final decision
            # If probability of abusive class > threshold, classify as abusive
            if probs[0] > threshold:
                pred_class = 0  # Abusive
            else:
                pred_class = 1  # Non-abusive

            # Store prediction
            predictions.append(pred_class)

        # Calculate overall accuracy
        # accuracy_score compares predictions to ground truth
        # Returns: (number of correct predictions) / (total predictions)
        # Example: 2946 correct out of 3364 total = 0.8757 = 87.57%
        accuracy = accuracy_score(true_labels, predictions)

        # Return results
        # accuracy: Overall accuracy (0.8757 = 87.57%)
        # predictions: List of predicted labels [0, 1, 0, 1, ...]
        # true_labels: Array of ground truth labels [0, 0, 1, 1, ...]
        return accuracy, predictions, true_labels


    def transcribe_audio(self, audio_path, language="hi"):
        """
        Convert audio file to text using Whisper.

        This is the first step in our audio classification pipeline. We use
        OpenAI's Whisper model to convert speech to text, then we can classify
        the text using our trained model.

        Why Whisper?
        ------------
        - State-of-the-art speech recognition (best available open-source model)
        - Supports 99+ languages including Hindi and Punjabi
        - Robust to accents, background noise, and audio quality issues
        - Works well even with code-mixing (Hindi + English + Punjabi)
        - Open source and free (unlike Google Speech API which costs money)

        How Whisper works:
        ------------------
        1. Audio → Mel spectrogram (visual representation of sound)
        2. Encoder: Compresses spectrogram into feature representation
        3. Decoder: Generates text one token at a time (autoregressive)
        4. Language model ensures grammatically correct output


        Parameters:
        -----------
        audio_path : str
            Path to audio file
            Supported formats: mp3
            Whisper handles format conversion automatically

        language : str, default="hi"
            Language code for transcription
            - "hi" = Hindi
            - "pa" = Punjabi
            - "en" = English
            Setting this helps Whisper by constraining the output language

            Why specify? Without language hint, Whisper might output wrong language
            Example: Hindi audio might be transcribed as Urdu (same phonetics)

        Returns:
        --------
        str
            Transcribed text
            Returns empty string "" if transcription fails (bad audio, file not found)

        Example:
        --------
        >>> audio_path = "/path/to/audio.mp3"
        >>> text = transcribe_audio(audio_path, language="hi")
        >>> print(text)
        "तुम बहनचोद हो साला"
        """

        # Try to transcribe, return empty string if it fails
        # Why try-except? Audio files can be corrupted, wrong format, missing, etc.
        # Better to return empty string than crash the entire program
        try:
            # Call Whisper's transcribe method
            # This does all the heavy lifting: audio loading, processing, transcription
            result = self.whisper_model.transcribe(
                audio_path,           # Path to audio file
                language=language,    # Language hint ("hi" for Hindi)
                task="transcribe",    # Task type (could also be "translate" to English)
                verbose=False         # Don't print progress (we want clean output)
            )

            # Extract and return the transcribed text
            # result is a dict with keys: 'text', 'segments', 'language'
            # We only care about 'text' which has the full transcription
            # .strip() removes leading/trailing whitespace
            return result["text"].strip()

        except:
            # If anything goes wrong (file not found, corrupted audio, etc.)
            # Return empty string and continue
            # In production, you might want to log the error for debugging
            return ""


    def classify_audio_file(self, audio_path, model_dir, swear_dict, language="hi", use_optimal_threshold=True):
        """
        Classify a single audio file as abusive or non-abusive.

        This is our complete end-to-end audio classification pipeline:
        Audio → Whisper (speech-to-text) → MuRIL (text classification) → Dictionary boost → Result

        This function answers the question: "Is this audio file abusive?"

        Parameters:
        -----------
        audio_path : str
            Path to audio file to classify
            Supported formats: mp3, wav, flac, m4a

        model_dir : str
            Path to trained model directory
            Contains our fine-tuned MuRIL model

        swear_dict : SwearWordDictionary
            Dictionary for confidence boosting

        language : str, default="hi"
            Language of the audio ("hi" for Hindi, "pa" for Punjabi)

        use_optimal_threshold : bool, default=True
            Whether to use the optimized threshold or default 0.5

        Returns:
        --------
        tuple : (pred_class, confidence, label_name)
            pred_class : int - 0 (abusive) or 1 (non-abusive)
            confidence : float - Probability of predicted class (0.0 to 1.0)
            label_name : str - "abusive" or "non-abusive"
        """

        # Load model if not already loaded
        # This is the same lazy loading pattern we use in evaluate_text()
        # Only load the model when we actually need it (saves memory)
        if self.text_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()  # Set to evaluation mode

        # Step 1: Transcribe audio to text
        # This is where Whisper converts speech → text
        transcribed = self.transcribe_audio(audio_path, language=language)

        # Handle empty transcription
        # This happens if:
        # - File is empty/silent
        # - Audio quality is too poor
        # - File is corrupted
        # - Wrong file format
        if not transcribed:
            # Return default prediction: non-abusive with low confidence
            # Why non-abusive as default? Better to miss some abusive content
            # than to falsely flag innocent content (false positives are worse)
            return 1, 0.5, "non-abusive"

        # Step 2: Tokenize the transcribed text
        # Convert text → numbers that model can process
        inputs = self.tokenizer(
            transcribed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Move to GPU/CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 3: Get model prediction
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Step 4: Apply dictionary boost
        # Check if transcribed text contains known abusive words
        found, boost, matched = swear_dict.contains_abusive_word(transcribed)

        if found:
            # Boost confidence in abusive class
            probs[0] += boost
            probs[1] -= boost
            probs = probs / probs.sum()  # Renormalize

        # Step 5: Apply threshold and get final prediction
        threshold = self.optimal_threshold if use_optimal_threshold else 0.50

        if probs[0] > threshold:
            pred_class = 0  # Abusive
            confidence = probs[0]
            label = "abusive"
        else:
            pred_class = 1  # Non-abusive
            confidence = probs[1]
            label = "non-abusive"

        return pred_class, confidence, label


    def detect_timestamps(self, audio_path, model_dir, swear_dict, language="hi",
                         chunk_duration=2.0, overlap=0.5, use_optimal_threshold=True):
        """
        Detect timestamps of abusive content in audio file.


        How it works:
        -------------
        Think of the audio as a long recording. We:

        1. Split audio into overlapping chunks (2 seconds each, 0.5 second overlap)
           Why overlap? Abusive words might be split across chunk boundaries

           Example with 10-second audio:
           Chunk 1: [0s - 2s]    "Hello how are"
           Chunk 2:     [1.5s - 3.5s]    "are you doing"
           Chunk 3:         [3s - 5s]    "doing you bhench..."
           Chunk 4:             [4.5s - 6.5s]    "...enchod sala"
           Chunk 5:                 [6s - 8s]   "sala randi"

           The overlap ensures "bhenchod" isn't split between chunks

        2. Process each chunk:
           - Transcribe chunk (Whisper)
           - Classify chunk (MuRIL + dictionary boost)
           - If abusive → store timestamp

        3. Return list of abusive segments with timestamps

        Why chunks instead of full audio?
        ----------------------------------
        - Memory: Processing 1-hour audio in one go requires huge GPU memory
        - Precision: Small chunks give precise timestamps (within 1 second)
        - Speed: Can process chunks in parallel (not implemented yet)
        - Robustness: One bad chunk doesn't fail entire audio


        Parameters:
        -----------
        audio_path : str
            Path to audio file to analyze
            Can be any length (tested up to 1 hour)

        model_dir : str
            Path to trained model directory

        swear_dict : SwearWordDictionary
            Dictionary for confidence boosting

        language : str, default="hi"
            Language code ("hi" or "pa")

        chunk_duration : float, default=2.0
            Length of each chunk in seconds
            - Too short (1s): Might cut words, more processing
            - Too long (10s): Less precise timestamps
            - 2s is a good balance (typically 1-2 sentences)

        overlap : float, default=0.5
            Overlap between consecutive chunks in seconds
            - 0.5 second overlap ensures words aren't split
            - Larger overlap = more robust but slower

        use_optimal_threshold : bool, default=True
            Whether to use the optimized threshold or default 0.5

        Returns:
        --------
        tuple : (detections, total_duration)
            detections : list of dicts
                Each dict contains:
                - start_time: When segment starts (seconds)
                - end_time: When segment ends (seconds)
                - confidence: How confident we are (0.0 to 1.0)
                - matched_words: Which dictionary words were found

            total_duration : float
                Total length of audio file in seconds
        """

        # Load model if needed
        if self.text_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()

        # Load audio file
        # sf.read() returns:
        # - audio_array: numpy array of audio samples
        # - sr: sample rate (how many samples per second)
        try:
            audio_array, sr = sf.read(audio_path)
        except:
            # If audio loading fails, return empty results
            # This handles: file not found, corrupted file, unsupported format
            return []

        # Convert stereo to mono if needed
        # Why? Whisper expects mono audio (single channel)
        # Stereo has 2 channels (left and right speakers)
        if len(audio_array.shape) > 1:
            # Take average of left and right channels
            # This is the standard way to convert stereo → mono
            audio_array = np.mean(audio_array, axis=1)

        # Resample to 16kHz if needed
        # Why 16kHz? Whisper is trained on 16kHz audio
        # Using different sample rate would degrade quality
        if sr != 16000:
            # Calculate how many samples we need for 16kHz
            # Formula: new_length = old_length * (new_rate / old_rate)
            # Example: 44100 Hz → 16000 Hz means we need ~36% of samples
            audio_array = signal.resample(
                audio_array,
                int(len(audio_array) * 16000 / sr)
            )
            sr = 16000  # Update sample rate

        # Calculate total duration in seconds
        # Formula: duration = number_of_samples / sample_rate
        # Example: 48000 samples / 16000 samples_per_second = 3.0 seconds
        total_duration = len(audio_array) / sr

        # Calculate chunk sizes in samples (not seconds)
        # Why samples? Audio is stored as array of samples, not seconds
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)

        # Calculate step size (how far to move for next chunk)
        # step_size = chunk_size - overlap
        step_size = chunk_samples - overlap_samples

        # List to store detected abusive segments
        detections = []

        # Get threshold
        threshold = self.optimal_threshold if use_optimal_threshold else 0.50

        # Process audio in overlapping chunks
        # range() creates: [0, 32000, 64000, 96000, ...]
        # Each number is the start position of a chunk
        for start_sample in range(0, len(audio_array) - chunk_samples + 1, step_size):
            # Calculate end position of this chunk
            # min() ensures we don't go past the end of audio
            end_sample = min(start_sample + chunk_samples, len(audio_array))

            # Extract audio chunk
            # This is like slicing a list: audio[start:end]
            chunk = audio_array[start_sample:end_sample]

            # Convert sample positions to time in seconds
            # Formula: time = samples / sample_rate
            start_time = start_sample / sr
            end_time = end_sample / sr

            # Save chunk to temporary file
            # Why? Whisper expects a file path, not a numpy array
            chunk_path = f"/tmp/chunk_{start_sample}.wav"
            sf.write(chunk_path, chunk, sr)

            # Transcribe this chunk
            transcribed = self.transcribe_audio(chunk_path, language=language)

            # Delete temporary file to save disk space
            # try-except in case file deletion fails (not critical)
            try:
                os.remove(chunk_path)
            except:
                pass  # If deletion fails, continue anyway

            # Skip empty or very short transcriptions
            # Why? Empty transcription = silence or noise (not speech)
            # Very short (<3 chars) = probably transcription error
            if not transcribed or len(transcribed) < 3:
                continue  # Move to next chunk

            # Classify this chunk's transcribed text
            inputs = self.tokenizer(
                transcribed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.text_model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # Apply dictionary boost
            found, boost, matched = swear_dict.contains_abusive_word(transcribed)
            if found:
                probs[0] += boost
                probs[1] -= boost
                probs = probs / probs.sum()

            # Apply threshold to determine if abusive
            if probs[0] > threshold:
                label = "abusive"
                confidence = probs[0]
            else:
                label = "non-abusive"
                confidence = probs[1]

            # Only store if abusive
            # We don't care about non-abusive segments for timestamp detection
            if label == "abusive":
                detections.append({
                    'start_time': start_time,      # When this segment starts
                    'end_time': end_time,          # When this segment ends
                    'confidence': confidence,       # How confident we are
                    'matched_words': matched if found else []  # Which words triggered
                })

        # Return all detections and total duration
        return detections, total_duration


def main():
    """
    Main execution function - orchestrates the entire training and evaluation pipeline.

    This is the entry point of our program. It runs everything in the right order:
    1. Load data and dictionaries
    2. Create synthetic training samples
    3. Train the model
    4. Optimize decision threshold on validation data
    5. Evaluate on text with optimized threshold
    6. Test on audio files
    7. Demonstrate timestamp detection



    Execution flow:
    ---------------
    1. Setup:
       - Define file paths
       - Initialize detector
       - Load dictionaries

    2. Data preparation:
       - Load training/validation/test data
       - Generate synthetic samples
       - Combine original + synthetic data

    3. Training:
       - Train MuRIL model on combined data
       - Save model checkpoints
       - Monitor validation performance

    4. Threshold Optimization (NEW):
       - Test multiple threshold values on validation data
       - Find optimal threshold that minimizes false positives
       - Save visualization of threshold vs metrics

    5. Evaluation:
       - Test accuracy on unseen text data with optimal threshold
       - Calculate precision/recall/F1
       - Generate classification report

    6. Audio testing:
       - Test on audio files
       - Compare predictions to ground truth
       - Show confidence scores

    7. Timestamp demo:
       - Analyze sample audio file
       - Detect abusive segments
       - Show precise timestamps

    Why organize code this way?
    ----------------------------
    - Single place to run everything
    - Easy to understand the full pipeline
    - Can comment out sections for testing
    - Clean separation of concerns
    - Easy to modify for different datasets

    Output format:
    --------------
    The function produces clean, production-ready output:
    - Section headers with clear separation
    - Progress bars for long operations
    - Minimal logging (only important information)
    - Clear success/failure indicators
    - Final summary of results


    We want clean output that can be logged and monitored.
    """

    # Print header
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION")
    print("="*80)

    # DEFINE FILE PATHS

    # Training, validation, and test data
    # These are CSV files with columns: ['text', 'label']
    # - text: Hindi/Punjabi sentences
    # - label: 0 (abusive) or 1 (non-abusive)
    HINDI_TRAIN = "/content/hindi_train.csv"
    HINDI_VAL = "/content/hindi_val.csv"
    HINDI_TEST = "/content/hindi_test.csv"

    # Dictionary files
    # These contain our swear word mappings
    HINDI_SWEARS = "/content/hindi_swears.csv"              # Hindi words + meanings
    HINDI_TO_GURMUKHI = "/content/hindi_to_gurmukhi.csv"   # Hindi → Punjabi mappings
    PHRASES = "/content/phrases_hindi_meaning.csv"          # Multi-word phrases

    # Audio files directory
    HINDI_AUDIO_DIR = "/content/drive/MyDrive/hindi_binary_audio_dataset_1/audio_files"

    # Model output directory
    # This is where we'll save the trained model
    MODEL_DIR = "/content/output_model/concept_model"

    # INITIALIZE DETECTOR
    # Create our ProductionDetector object
    # This loads Whisper and sets up the framework for training
    detector = ProductionDetector(device="cuda")

    # LOAD DICTIONARIES
    # =================
    # Load our swear word dictionaries from CSV files
    # This populates the dictionary with all abusive words and mappings
    print(" Loading swear word dictionaries...")
    detector.swear_dict.load_dictionaries(HINDI_SWEARS, HINDI_TO_GURMUKHI, PHRASES)
    print(f" Loaded {len(detector.swear_dict.all_abusive_words)} abusive words/phrases\n")

    # LOAD DATA
    # Load our training, validation, and test datasets
    print(" Loading training data...")

    # Read CSV files into pandas DataFrames
    # pandas automatically handles:
    # - CSV parsing
    # - Data types (strings, integers)
    # - Missing values
    train_df = pd.read_csv(HINDI_TRAIN)
    val_df = pd.read_csv(HINDI_VAL)
    test_df = pd.read_csv(HINDI_TEST)

    # CREATE SYNTHETIC DATA
    # Generate synthetic training samples from our dictionary
    synthetic_df = detector.create_synthetic_samples(detector.swear_dict)

    # Calculate how many synthetic samples to add
    # We want synthetic data to be ~5% of training data
    # Why 5%? Balance between:
    # - Too little: Not enough concept reinforcement
    # - Too much: Model might memorize synthetic patterns
    num_synthetic = int(len(train_df) * 0.05)

    # Randomly sample from synthetic data
    # Why random? Ensure diversity, not just first N words
    # random_state=42 makes it reproducible (same samples every time)
    synthetic_sample = synthetic_df.sample(
        n=min(num_synthetic, len(synthetic_df)),  # Take at most what we have
        random_state=42
    )

    # Add 'source' column to original training data
    # This lets us track which samples are original vs synthetic
    # Useful for debugging if synthetic data causes issues
    train_df['source'] = 'original'

    # Combine original + synthetic data
    # pd.concat merges two DataFrames vertically (stacks them)
    # ignore_index=True creates new index 0, 1, 2, ... for combined data
    train_augmented = pd.concat([train_df, synthetic_sample], ignore_index=True)

    # Shuffle combined data
    # Why? Ensure synthetic samples are mixed throughout, not just at end
    # This prevents model from seeing all synthetic data in one batch
    # .sample(frac=1) returns 100% of data in random order
    # random_state=42 makes shuffling reproducible
    # .reset_index(drop=True) creates new sequential index
    train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print dataset statistics
    print(f"   Training: {len(train_augmented)} samples ({len(synthetic_sample)} synthetic)")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples\n")

    # TRAIN MODEL
    # Train our MuRIL model on the combined dataset
    # This is where the actual learning happens!
    trainer = detector.train_model(train_augmented, val_df, MODEL_DIR)

    # OPTIMIZE THRESHOLD (NEW!)
    # Find the best decision threshold using validation data
    # This is a critical step that reduces false positives
    threshold_results = detector.optimize_threshold(val_df, MODEL_DIR, detector.swear_dict, save_plot=True)

    # EVALUATE ON TEXT
    # Test our trained model on unseen text data with optimal threshold
    print("\n" + "="*80)
    print("TEXT EVALUATION RESULTS (WITH OPTIMAL THRESHOLD)")
    print("="*80)

    # Run evaluation on test set
    # This processes all test samples and compares predictions to labels
    # Now using the optimized threshold instead of default 0.5
    text_accuracy, predictions, true_labels = detector.evaluate_text(
        test_df, MODEL_DIR, detector.swear_dict, use_optimal_threshold=True
    )

    # Print overall accuracy
    print(f"\n Text Accuracy: {text_accuracy*100:.2f}% (with threshold={detector.optimal_threshold:.2f})")

    # Generate detailed classification report
    # This shows precision, recall, and F1-score for each class
    # - Precision: Of all predicted abusive, what % actually were?
    # - Recall: Of all actual abusive, what % did we catch?
    # - F1-score: Harmonic mean of precision and recall
    report = classification_report(
        true_labels,                         # Ground truth labels
        predictions,                         # Our predictions
        target_names=["abusive", "non-abusive"],  # Class names for display
        digits=4                             # Show 4 decimal places (0.8757 not 0.88)
    )
    print("\n" + report)

    # TEST ON AUDIO FILES
    # Now let's test on actual audio files to see how the full pipeline performs
    print("\n" + "="*80)
    print("AUDIO FILE CLASSIFICATION")
    print("="*80)

    # Test first 10 audio files
    num_audio_files = min(10, len(test_df))

    # Process each audio file
    for idx in range(num_audio_files):
        # Construct audio file path
        audio_file = os.path.join(HINDI_AUDIO_DIR, f"test_{idx}.mp3")

        # Skip if file doesn't exist
        # This handles cases where audio files are missing
        if not os.path.exists(audio_file):
            continue

        # Get ground truth label from test dataframe
        # This is the "correct answer" we're comparing against
        true_label = test_df.iloc[idx]['label']
        true_label_name = "abusive" if true_label == 0 else "non-abusive"

        # Classify the audio file (now with optimal threshold)
        # This runs: Audio → Whisper → Text → MuRIL → Dictionary → Threshold → Prediction
        pred_class, confidence, pred_label = detector.classify_audio_file(
            audio_file, MODEL_DIR, detector.swear_dict, language="hi", use_optimal_threshold=True
        )

        # Check if prediction matches ground truth
        # ✅ = correct, ❌ = incorrect
        match = "✅" if pred_class == true_label else "❌"

        # Print results for this file
        print(f"\nFile: test_{idx}.mp3")
        print(f"  Ground Truth: {true_label_name}")
        print(f"  Prediction: {pred_label} ({confidence*100:.1f}% confidence) {match}")

    # TIMESTAMP DETECTION
    # This shows WHEN abusive content occurs, not just IF it occurs
    print("\n" + "="*80)
    print("TIMESTAMP DETECTION")
    print("="*80)

    # Use first test audio file as example
    sample_audio = os.path.join(HINDI_AUDIO_DIR, "test_0.mp3")

    # Check if file exists before processing
    if os.path.exists(sample_audio):
        print(f"\nAnalyzing: test_0.mp3")

        # Run timestamp detection (now with optimal threshold)
        # This splits audio into chunks and finds abusive segments
        detections, duration = detector.detect_timestamps(
            sample_audio, MODEL_DIR, detector.swear_dict, language="hi", use_optimal_threshold=True
        )

        # Print total duration
        print(f"Duration: {duration:.2f}s")

        # Print results
        if len(detections) == 0:
            # No abusive content found
            print("\n NO ABUSIVE CONTENT DETECTED")
        else:
            # Found abusive segments - show timestamps
            print(f"\n  FOUND {len(detections)} ABUSIVE SEGMENTS:\n")

            # Print each detection with timestamp and confidence
            for i, seg in enumerate(detections, 1):
                print(f"  {i}. [{seg['start_time']:.2f}s - {seg['end_time']:.2f}s]")
                print(f"     Confidence: {seg['confidence']*100:.1f}%")

                # Show which dictionary words were matched (if any)
                if seg['matched_words']:
                    # Show first 3 matched words (to keep output clean)
                    print(f"     Matched words: {seg['matched_words'][:3]}")

    # FINAL SUMMARY
    # Print summary of everything we accomplished
    print("\n" + "="*80)
    print(" COMPLETE!")
    print("="*80)
    print(f"\n Model saved: {MODEL_DIR}")
    print(f" Text accuracy: {text_accuracy*100:.2f}%")
    print(f" Optimal threshold: {detector.optimal_threshold:.2f}")
    print(f" False positive reduction: {threshold_results['metrics']['false_positive_rate']*100:.2f}%")
    print("\n Ready for production use with optimized threshold!")


# PROGRAM ENTRY POINT
if __name__ == "__main__":
    main()
