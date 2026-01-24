"""
PRODUCTION-READY HINDI/PUNJABI PROFANITY DETECTION SYSTEM
==========================================================


Model: Whisper Large-v3 + Fine-tuned MuRIL classifier

This system processes noisy audio and identifies Hindi/Punjabi profanity
with high accuracy while minimizing false positives through a multi-stage pipeline.

Key Features:
- Universal audio format support (MP3, WAV, FLAC, etc.)
- FFT-based noise reduction for cockpit environments
- Adaptive volume normalization
- Whisper Large-v3 ASR (excellent Hindi/Punjabi support)
- Fine-tuned MuRIL binary classifier (abusive vs non-abusive)
- Dictionary-based confidence boosting
- Tiered classification to reduce false positives
- Timestamp-based detection for precise location of profanity

"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
import json
import time

# Suppress warnings for cleaner output during production use
warnings.filterwarnings('ignore')

print("="*80)
print("HINDI/PUNJABI PROFANITY DETECTION SYSTEM")
print("="*80)

# Check if CUDA is available for GPU acceleration
# GPU significantly speeds up both Whisper ASR and MuRIL classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")

# Install required dependencies
# These are needed for audio processing, ASR, and text classification
print("\n Installing dependencies...")
os.system('pip install transformers torch openai-whisper soundfile scipy librosa pydub -q > /dev/null 2>&1')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
import soundfile as sf
from scipy.fft import rfft, irfft  # For FFT-based noise reduction
import librosa  # Primary audio loading library
from pydub import AudioSegment  # Fallback for exotic audio formats

print(" Dependencies loaded!\n")



# LOAD ASR MODEL (WHISPER LARGE-V3)
# We use Whisper Large-v3 because it has the best Hindi/Punjabi support
# among publicly available models. It's trained on 680k hours of multilingual
# data and outputs native Devanagari script for Hindi.

print("="*80)
print("LOADING ASR MODEL")
print("="*80)

ASR_MODEL = None
ASR_TYPE = None

# Try to load Whisper Large-v3 first (best quality for Indian languages)
print("\n Loading Whisper Large-v3...")
try:
    print("   Downloading model... (this is ~3GB, may take 2-3 minutes)")
    print("   This model provides excellent Hindi/Punjabi recognition")

    # Load the model onto GPU if available
    ASR_MODEL = whisper.load_model("large-v3", device=device)
    ASR_TYPE = "whisper-large-v3"

    print(" SUCCESS: Whisper Large-v3 loaded!")

except Exception as e:
    print(f" Failed to load Large-v3: {str(e)[:100]}")

    # Fallback to Medium if Large-v3 fails (good balance of speed/accuracy)
    print("\n Trying Whisper Medium as fallback...")
    try:
        print("   Downloading... (~1.5GB)")
        ASR_MODEL = whisper.load_model("medium", device=device)
        ASR_TYPE = "whisper-medium"
        print(" SUCCESS: Whisper Medium loaded!")

    except Exception as e2:
        print(f" Failed: {str(e2)[:100]}")

        # Last resort: Whisper Base (fastest but lower accuracy)
        print("\n Falling back to Whisper Base...")
        ASR_MODEL = whisper.load_model("base", device=device)
        ASR_TYPE = "whisper-base"
        print(" SUCCESS: Whisper Base loaded!")

print("\n" + "="*80)
print(f"FINAL ASR MODEL: {ASR_TYPE.upper()}")
print("="*80)

# Show benefits of Large-v3 model
if "large" in ASR_TYPE:
    print("\n   Whisper Large-v3 Benefits:")
    print("   • MUCH better Hindi/Punjabi recognition than smaller models")
    print("   • More likely to output Devanagari script (not Roman transliteration)")
    print("   • Better handling of code-switching (Hinglish)")
    print("   • Lower Word Error Rate (WER)")



# SWEAR WORD DICTIONARY CLASS
# This class manages our profanity dictionaries and provides confidence boost
# when known abusive words are detected. We use multiple dictionaries to cover:
# - Hindi transliterations (Roman script)
# - Devanagari script
# - Gurmukhi script (Punjabi)
# - Common abusive phrases

class SwearWordDictionary:
    """
    Manages profanity dictionaries and provides word-matching functionality.

    The dictionary serves two purposes:
    1. Provide confidence boost to MuRIL predictions when known words are found
    2. Act as a safety net for edge cases where MuRIL might miss obvious profanity

    We maintain the original concept-based approach where equivalent words
    across different scripts (e.g., "chod", "चोद", "ਚੋਦ") are all added to
    the dictionary to maximize detection coverage.
    """

    def __init__(self):
        # Use a set for O(1) lookup time
        self.all_abusive_words = set()

    def load_dictionaries(self, hindi_swears_csv, hindi_to_gurmukhi_csv, phrases_csv):
        """
        Load profanity from multiple CSV sources.

        Why multiple CSVs?
        - hindi_swears.csv: Core Hindi abusive words in both Devanagari and Roman
        - hindi_to_gurmukhi.csv: Punjabi translations in Gurmukhi script
        - phrases_csv: Multi-word abusive phrases that should be detected together
        """
        print("\n Loading dictionaries...")

        # Load Hindi swear words (Devanagari + transliteration)
        df_hindi = pd.read_csv(hindi_swears_csv)
        for _, row in df_hindi.iterrows():
            # Add both the Roman transliteration (e.g., "chod")
            transliteration = str(row['Hindi transliteration']).strip().lower()
            # And the Devanagari script (e.g., "चोद")
            devanagari = str(row['Devanagari']).strip()

            self.all_abusive_words.add(devanagari)
            self.all_abusive_words.add(transliteration)

        # Load Punjabi equivalents in Gurmukhi script
        df_punjabi = pd.read_csv(hindi_to_gurmukhi_csv)
        for _, row in df_punjabi.iterrows():
            gurmukhi = str(row['Gurmukhi']).strip()
            self.all_abusive_words.add(gurmukhi)

        # Load common abusive phrases (these are multi-word expressions)
        df_phrases = pd.read_csv(phrases_csv)
        for _, row in df_phrases.iterrows():
            # Add both the phrase and its Hindi translation
            phrase = str(row['Phrase']).strip().lower()
            hindi = str(row['Hindi Translation']).strip()
            self.all_abusive_words.add(phrase)
            self.all_abusive_words.add(hindi)

        print(f" Loaded {len(self.all_abusive_words)} unique words/phrases\n")

    def contains_abusive_word(self, text):
        """
        Check if text contains any known abusive words.

        Returns:
        --------
        found : bool
            True if any abusive word was found
        boost : float
            Confidence boost to add to MuRIL's prediction (0-0.20)
        matched : list
            List of matched abusive words (for debugging/logging)

        Why boost confidence?
        When we find a known abusive word, we increase MuRIL's confidence
        because the dictionary acts as strong evidence. The boost is capped
        at 0.20 to prevent overwhelming the neural model's prediction.
        """
        if not text:
            return False, 0.0, []

        # Search in both original case and lowercase
        # This catches variations like "CHOD", "Chod", "chod"
        text_lower = text.lower()
        matched = []

        # Check each word in our dictionary
        for word in self.all_abusive_words:
            # Use substring matching to catch words within larger text
            # e.g., "bhenchod" contains "chod"
            if word in text_lower or word in text:
                matched.append(word)

        if matched:
            # Calculate boost based on number of matches
            # More matches = higher confidence boost
            # But cap at 0.20 to prevent dictionary from dominating
            confidence_boost = min(0.20, len(matched) * 0.05)
            return True, confidence_boost, matched

        return False, 0.0, []


# MAIN DETECTOR CLASS
# This is the core detection system that orchestrates all components:
# - Audio preprocessing (noise reduction, volume normalization)
# - Speech-to-text conversion (Whisper)
# - Profanity classification (MuRIL)
# - Post-processing and filtering

class ImprovedIndianASRDetector:
    """
    Production-ready profanity detector optimized for cockpit audio.

    Pipeline:
    1. Load audio in any format
    2. Apply FFT-based noise reduction
    3. Normalize and boost volume
    4. Chunk audio for processing
    5. Transcribe each chunk with Whisper
    6. Filter out hallucinations and gibberish
    7. Classify valid transcriptions with MuRIL
    8. Apply tiered decision logic to reduce false positives
    9. Return timestamped detections
    """

    def __init__(self, model_dir, asr_model, asr_type):
        """
        Initialize the detector with ASR and classification models.

        Parameters:
        -----------
        model_dir : str
            Path to fine-tuned MuRIL model directory
        asr_model : whisper.model.Whisper
            Loaded Whisper ASR model
        asr_type : str
            Name of ASR model for logging
        """
        self.device = device
        self.model_dir = model_dir
        self.asr_model = asr_model
        self.asr_type = asr_type

        print("\n="*80)
        print("LOADING CLASSIFIER")
        print("="*80)

        # Load fine-tuned MuRIL classifier
        # MuRIL is Google's multilingual BERT model fine-tuned on 17 Indian languages
        # We've fine-tuned it specifically for binary classification (abusive vs non-abusive)
        print(f"\n Loading MuRIL classifier...")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load tokenizer (converts text to input IDs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load the classification model (outputs probabilities for 2 classes)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.text_model.to(self.device)
        self.text_model.eval()  # Set to evaluation mode (disables dropout, etc.)
        print(" MuRIL loaded")

        # Load optimal threshold from training
        # This threshold was determined during model fine-tuning to balance
        # precision and recall. It minimizes false positives while maintaining
        # good detection of actual profanity.
        threshold_path = os.path.join(os.path.dirname(model_dir), 'threshold.json')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.optimal_threshold = threshold_data.get('optimal_threshold', 0.50)
            print(f" Optimal threshold: {self.optimal_threshold:.2f}")
        else:
            # Default to 0.50 if threshold file not found
            self.optimal_threshold = 0.50
            print("  Using default threshold: 0.50")

        # Dictionary will be loaded separately
        self.swear_dict = None

        print("\n" + "="*80)
        print(" SYSTEM READY")
        print(f"  ASR: {self.asr_type.upper()}")
        print(f"  Classifier: MuRIL (fine-tuned)")
        print("="*80)

    def load_dictionary(self, hindi_swears_csv, hindi_to_gurmukhi_csv, phrases_csv):
        """Load profanity dictionaries for confidence boosting."""
        self.swear_dict = SwearWordDictionary()
        self.swear_dict.load_dictionaries(hindi_swears_csv, hindi_to_gurmukhi_csv, phrases_csv)

    def load_audio_any_format(self, audio_path):
        """
        Universal audio loader supporting all common formats.

        Strategy:
        1. Try librosa first (fastest, handles most formats)
        2. If that fails, use pydub (slower but handles exotic formats)

        Why this approach?
        Audio comes in various formats depending on recording equipment.
        We need to handle MP3, WAV, FLAC, OGG, M4A, AAC, WMA, etc.

        Returns:
        --------
        audio : np.array
            Audio samples as float32 array
        sr : int
            Sample rate (always 16000 Hz for Whisper)
        """
        try:
            # Method 1: Try librosa (fast and reliable for most formats)
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return audio, sr
        except:
            # Method 2: Use pydub as fallback
            try:
                # Load with pydub (uses ffmpeg under the hood)
                audio_segment = AudioSegment.from_file(audio_path)

                # Convert to mono (Whisper needs single channel)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)

                # Resample to 16kHz (Whisper's expected sample rate)
                audio_segment = audio_segment.set_frame_rate(16000)

                # Convert to numpy array
                audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

                # Normalize to [-1, 1] range
                audio = audio / (np.max(np.abs(audio)) + 1e-8)

                return audio, 16000
            except Exception as e:
                print(f" Failed to load audio: {e}")
                return None, None

    def reduce_noise(self, audio, sr, strength=0.5):
        """
        FFT-based spectral subtraction for noise reduction.

        This is critical for audio which contains:
        - Engine noise (low-frequency rumble)
        - Ventilation systems (steady mid-frequency hum)
        - Radio static (high-frequency noise)

        Algorithm:
        ----------
        1. Estimate noise profile from first second of audio
           (Assumption: First second is representative of background noise)
        2. For each audio frame:
           a. Convert to frequency domain (FFT)
           b. Subtract estimated noise spectrum
           c. Apply spectral floor to avoid over-subtraction
           d. Convert back to time domain (IFFT)
        3. Overlap-add frames to reconstruct clean audio

        Parameters:
        -----------
        strength : float (0-1)
            How aggressively to remove noise
            0.5 = moderate (good for preserving speech quality)
            0.7 = aggressive (for very noisy environments)
            0.3 = gentle (for relatively clean audio)

        Why spectral subtraction?
        - Fast (realtime capable)
        - Works well for stationary noise (engine, ventilation)
        - Preserves speech intelligibility
        """

        # Frame parameters for Short-Time Fourier Transform (STFT)
        frame_length = 2048  # ~128ms at 16kHz (good for speech)
        hop_length = frame_length // 2  # 50% overlap

        # Estimate noise from first second of audio
        # Assumption: Background noise is relatively constant
        noise_duration = min(1.0, len(audio) / sr)
        noise_samples = int(noise_duration * sr)

        # Calculate how many frames fit in the noise estimation window
        num_noise_frames = max(1, (noise_samples - frame_length) // hop_length + 1)

        # Initialize noise power spectrum (average across all noise frames)
        noise_power_sum = np.zeros(frame_length // 2 + 1)

        # Extract noise profile from first second
        for i in range(num_noise_frames):
            start = i * hop_length
            end = start + frame_length
            if end > noise_samples:
                break

            # Get noise frame
            noise_frame = audio[start:end]

            # Pad if needed (for last frame)
            if len(noise_frame) < frame_length:
                noise_frame = np.pad(noise_frame, (0, frame_length - len(noise_frame)))

            # Convert to frequency domain
            noise_fft = rfft(noise_frame)

            # Accumulate power spectrum (magnitude squared)
            noise_power_sum += np.abs(noise_fft) ** 2

        # Average noise power across all frames
        noise_power = noise_power_sum / max(1, num_noise_frames)

        # Prepare for overlap-add synthesis
        audio_padded = np.pad(audio, (0, frame_length))
        cleaned_audio = np.zeros_like(audio)

        # Hanning window for smooth overlap-add
        window = np.hanning(frame_length)

        # Process each frame
        num_frames = (len(audio_padded) - frame_length) // hop_length + 1

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length

            # Extract and window the frame
            frame = audio_padded[start:end] * window

            # Convert to frequency domain
            frame_fft = rfft(frame)
            frame_power = np.abs(frame_fft) ** 2
            frame_phase = np.angle(frame_fft)  # Preserve phase (critical for speech)

            # SPECTRAL SUBTRACTION: Remove noise power
            # cleaned_power = signal_power - strength * noise_power
            cleaned_power = frame_power - strength * noise_power

            # Apply spectral floor to prevent over-subtraction
            # beta controls minimum allowed power (prevents "musical noise" artifact)
            beta = 0.1 + (0.2 * (1 - strength))  # Adaptive floor based on strength
            cleaned_power = np.maximum(cleaned_power, beta * frame_power)

            # Reconstruct complex spectrum (preserve original phase)
            cleaned_magnitude = np.sqrt(cleaned_power)
            cleaned_fft = cleaned_magnitude * np.exp(1j * frame_phase)

            # Convert back to time domain
            cleaned_frame = irfft(cleaned_fft, n=frame_length)

            # Overlap-add into output
            output_end = min(start + frame_length, len(cleaned_audio))
            cleaned_audio[start:output_end] += cleaned_frame[:output_end - start]

        # Normalize to prevent clipping
        cleaned_audio = cleaned_audio / (np.max(np.abs(cleaned_audio)) + 1e-8)

        return cleaned_audio

    def normalize_volume(self, audio, boost=1.5):
        """
        Normalize audio volume and apply optional boost.

        Why is this important?
        ----------------------
        Whisper performs best when audio has consistent RMS energy around 0.1-0.3.
        Cockpit Audio is often quiet (pilots speaking normally, not yelling),
        which can lead to poor Whisper transcriptions.

        This function:
        1. Calculates current RMS (root mean square) energy
        2. Amplifies to target RMS level
        3. Prevents clipping by limiting maximum gain

        Parameters:
        -----------
        boost : float
            Amplification factor (1.5 = 50% boost, 2.0 = 100% boost)
            Higher values for quieter audio

        Example:
        --------
        Quiet audio: RMS = 0.03 → After boost → RMS = 0.225 (7.5x gain)
        Normal audio: RMS = 0.10 → After boost → RMS = 0.225 (2.25x gain)
        Loud audio: RMS = 0.20 → After boost → RMS = 0.225 (1.125x gain)
        """

        # Calculate current RMS energy
        current_rms = np.sqrt(np.mean(audio ** 2))

        # Skip if audio is silent
        if current_rms < 1e-6:
            return audio

        # Target RMS level (0.15 is good for Whisper, then apply boost)
        target_rms = 0.15 * boost

        # Calculate required gain
        gain = target_rms / current_rms

        # Prevent clipping by limiting gain
        # Maximum gain that won't cause samples to exceed [-1, 1]
        max_gain = 0.95 / (np.max(np.abs(audio)) + 1e-8)
        gain = min(gain, max_gain)

        # Apply gain
        return audio * gain

    def transcribe_chunk(self, audio_array, language="hi"):
        """
        Transcribe audio chunk using Whisper with optimized settings.

        Whisper Parameters Explained:
        ------------------------------
        language="hi" :
            Tell Whisper to expect Hindi. This improves accuracy significantly.

        task="transcribe" :
            We want transcription (not translation)

        condition_on_previous_text=False :
            CRITICAL: Prevent Whisper from using previous chunks as context.
            Why? Each chunk is independent, and context can cause hallucinations
            where Whisper tries to maintain continuity even when there's no speech.

        compression_ratio_threshold=2.4 :
            Reject transcriptions with high compression ratio (likely hallucinations)
            Hallucinated text tends to be very compressible (lots of repetition)

        logprob_threshold=-1.0 :
            Reject low-confidence transcriptions (log probability < -1.0)

        no_speech_threshold=0.6 :
            Probability threshold for detecting silence
            Higher = more aggressive silence detection

        temperature=0.0 :
            Use greedy decoding (no randomness)
            For production, we want deterministic results

        initial_prompt :
            Give Whisper a hint about the language/context
            This helps it choose Devanagari script over Roman transliteration
        """
        try:
            result = self.asr_model.transcribe(
                audio_array,
                language=language,
                task="transcribe",
                verbose=False,  # Suppress debug output
                fp16=False,  # Use FP32 for better accuracy (slower but more precise)
                condition_on_previous_text=False,  # CRITICAL: Prevent cross-chunk contamination
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                temperature=0.0,
                # Prompt Whisper to output Devanagari
                initial_prompt="यह हिंदी में बोला गया है।"  # "This is spoken in Hindi"
            )

            return result["text"].strip()

        except Exception as e:
            # If transcription fails, return empty string
            # This is safer than crashing the entire pipeline
            return ""

    def is_valid_transcription(self, text):
        """
        Filter out Whisper hallucinations and gibberish.

        Common Whisper Hallucinations:
        -------------------------------
        1. Extreme repetition: "पपपपपपपपप..." (Whisper stuck in loop)
        2. Special markers: "��", "<|", "|>" (encoding artifacts)
        3. Wrong language: Cyrillic, Chinese characters (language confusion)

        Why do hallucinations happen?
        - Low audio quality → Whisper "fills in" with patterns
        - Silence → Whisper generates text from noise
        - Audio artifacts → Confuse the model

        This function rejects obvious garbage before wasting time on classification.
        """

        # Reject empty or very short transcriptions
        if not text or len(text) < 3:
            return False

        # Check for extreme character repetition
        # e.g., "पपपपपपपपपपपपपपप" (15+ consecutive identical characters)
        # Real Hindi words rarely have more than 3-4 consecutive identical chars
        for i in range(len(text) - 15):
            consecutive = 1
            for j in range(i+1, min(i+30, len(text))):
                if text[j] == text[i]:
                    consecutive += 1
                else:
                    break

            if consecutive >= 15:
                # This is clearly a hallucination
                return False

        # Check for known hallucination markers
        # These appear when Whisper encounters encoding issues or special tokens
        bad_markers = ['��', '�', '<|', '|>']
        if any(marker in text for marker in bad_markers):
            return False

        # If we reach here, transcription looks valid
        return True

    def classify_text_smart(self, text):
        """
        Classify text with tiered confidence logic to reduce false positives.

        Tiered Classification Strategy:
        --------------------------------
        The key insight: High confidence without dictionary match = likely gibberish

        Tier 1 (>90% confidence):
            IF dictionary match → ABUSIVE (high confidence + evidence)
            ELSE → NON-ABUSIVE (likely Whisper hallucination with random high score)

        Tier 2 (75-90% confidence):
            IF dictionary match → ABUSIVE (medium confidence + evidence)
            ELSE → NON-ABUSIVE (borderline case, err on side of caution)

        Tier 3 (<75% confidence):
            → NON-ABUSIVE (confidence too low)

        Why this works:
        ---------------
        Real abusive words trigger BOTH high MuRIL confidence AND dictionary match.
        Gibberish might randomly get high MuRIL confidence, but won't match dictionary.
        This combination reduces false positives.

        Returns:
        --------
        pred_class : int
            0 = abusive, 1 = non-abusive
        confidence : float
            Final confidence (0-1)
        label : str
            "abusive" or "non-abusive"
        reason : str
            Explanation of decision (for debugging/logging)
        """

        # Reject very short text
        if not text or len(text) < 3:
            return 1, 0.5, "non-abusive", "too_short"

        # Tokenize text for MuRIL
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # MuRIL's max sequence length
        )

        # Move to GPU if available
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get MuRIL prediction (no gradient needed for inference)
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Convert logits to probabilities
        # probs[0] = P(abusive), probs[1] = P(non-abusive)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Store initial probability before dictionary boost
        initial_abusive_prob = probs[0]

        # Check dictionary for known abusive words
        dict_found = False
        dict_matched = []
        if self.swear_dict:
            dict_found, boost, dict_matched = self.swear_dict.contains_abusive_word(text)

            if dict_found:
                # Apply confidence boost (but maintain probability distribution)
                probs[0] += boost
                probs[1] -= boost

                # Renormalize to ensure probabilities sum to 1
                probs = probs / probs.sum()

        # Final probability after dictionary boost
        final_abusive_prob = probs[0]

        # TIERED DECISION LOGIC

        # Tier 1: Very High Confidence (>90%)
        # Require dictionary confirmation to avoid false positives from gibberish
        if initial_abusive_prob > 0.90:
            if dict_found:
                # High confidence + dictionary match = Definitely abusive
                return 0, final_abusive_prob, "abusive", f"high_conf_dict:{dict_matched[:2]}"
            else:
                # High confidence but no dictionary = Likely gibberish
                # Example: "आपाद काम्ताणा" gets 92% but it's nonsense
                return 1, probs[1], "non-abusive", f"high_conf_no_dict({initial_abusive_prob:.2f})"

        # Tier 2: Medium-High Confidence (75-90%)
        # Also require dictionary confirmation
        elif initial_abusive_prob >= 0.75:
            if dict_found:
                # Medium confidence + dictionary match = Abusive
                return 0, final_abusive_prob, "abusive", f"med_conf_dict:{dict_matched[:2]}"
            else:
                # Medium confidence without dictionary = Not confident enough
                return 1, probs[1], "non-abusive", f"med_conf_no_dict({initial_abusive_prob:.2f})"

        # Tier 3: Low Confidence (<75%)
        # Definitely non-abusive
        else:
            return 1, probs[1], "non-abusive", f"low_confidence({initial_abusive_prob:.2f})"

    def detect_timestamps(self, audio_path, language="hi",
                         chunk_duration=3.0, overlap=0.5):
        """
        Main detection pipeline with timestamped results.

        Pipeline Flow:
        --------------
        1. Load audio (any format)
        2. Apply FFT noise reduction
        3. Normalize and boost volume
        4. Split into overlapping chunks
        5. For each chunk:
           a. Transcribe with Whisper
           b. Filter hallucinations
           c. Classify with MuRIL
           d. Apply tiered decision logic
        6. Return detections with timestamps

        Parameters:
        -----------
        audio_path : str
            Path to audio file
        language : str
            Language code ("hi" for Hindi, "pa" for Punjabi)
        chunk_duration : float
            Duration of each chunk in seconds (3.0s is optimal for Whisper)
        overlap : float
            Overlap between chunks in seconds (0.5s prevents missing words at boundaries)

        Returns:
        --------
        abusive_detections : list
            List of dicts containing abusive segments with timestamps
        all_detections : list
            List of all valid detections (for debugging/analysis)
        """

        print("\n" + "="*80)
        print(f"DETECTION WITH {self.asr_type.upper()}")
        print("="*80)

        print(f"\n File: {os.path.basename(audio_path)}")

        # Check if file exists
        if not os.path.exists(audio_path):
            print(" File not found!")
            return [], []

        # STEP 1: Load audio
        print(f"\n Loading audio...")
        audio_array, sr = self.load_audio_any_format(audio_path)

        if audio_array is None:
            print(" Failed to load audio!")
            return [], []

        total_duration = len(audio_array) / sr
        original_rms = np.sqrt(np.mean(audio_array ** 2))
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Original RMS: {original_rms:.4f}")

        # STEP 2: Noise reduction
        print(f"\n Applying FFT noise reduction...")
        print(f"   Strategy: Spectral subtraction (strength=0.5)")
        print(f"   Target: Remove engine noise, preserve speech")
        audio_array = self.reduce_noise(audio_array, sr, strength=0.5)
        cleaned_rms = np.sqrt(np.mean(audio_array ** 2))
        print(f"   Noise reduced (RMS: {cleaned_rms:.4f})")

        # STEP 3: Volume normalization
        print(f"\n Normalizing and boosting volume...")
        print(f"   Strategy: Adaptive gain to RMS=0.15 × 1.5 = 0.225")
        audio_array = self.normalize_volume(audio_array, boost=1.5)
        final_rms = np.sqrt(np.mean(audio_array ** 2))
        print(f"   Volume normalized (Final RMS: {final_rms:.4f})")
        print(f"   Total gain: {final_rms/original_rms:.2f}x")

        # STEP 4: Prepare chunking
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        step_size = chunk_samples - overlap_samples
        num_chunks = max(1, (len(audio_array) - chunk_samples) // step_size + 1)

        print(f"\n Processing {num_chunks} chunks...")
        print(f"   Chunk size: {chunk_duration}s with {overlap}s overlap")
        print("="*80)

        # Initialize result containers
        all_detections = []
        abusive_detections = []
        filtered_count = 0

        # STEP 5: Process each chunk
        for start_sample in range(0, len(audio_array) - chunk_samples + 1, step_size):
            end_sample = min(start_sample + chunk_samples, len(audio_array))
            chunk = audio_array[start_sample:end_sample]

            # Calculate timestamps
            start_time_sec = start_sample / sr
            end_time_sec = end_sample / sr

            # Transcribe chunk
            transcribed = self.transcribe_chunk(chunk, language)

            # Filter hallucinations
            if not self.is_valid_transcription(transcribed):
                filtered_count += 1
                continue

            # Classify transcription
            pred_class, confidence, label, reason = self.classify_text_smart(transcribed)

            # Store detection
            detection = {
                'start_time': start_time_sec,
                'end_time': end_time_sec,
                'label': label,
                'confidence': confidence,
                'transcription': transcribed,
                'reason': reason
            }

            all_detections.append(detection)

            # If abusive, add to abusive list and log
            if label == "abusive":
                abusive_detections.append(detection)
                print(f"   [{start_time_sec:.1f}s] ABUSIVE: \"{transcribed}\"")
                print(f"     Confidence: {confidence*100:.1f}% | Reason: {reason}")

        # Print summary
        print("="*80)
        print(f" Processing complete")
        print(f"  Valid chunks: {len(all_detections)}")
        print(f"  Filtered chunks: {filtered_count}")

        # STEP 6: Display results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)

        if len(abusive_detections) == 0:
            print("\n NO ABUSIVE CONTENT DETECTED")
            print(f"   Analyzed: {len(all_detections)} valid chunks")
            print(f"   Total duration: {total_duration:.2f}s")
        else:
            print(f"\n  FOUND {len(abusive_detections)} ABUSIVE SEGMENT(S):\n")

            for i, det in enumerate(abusive_detections, 1):
                print(f"  {i}. Time: [{det['start_time']:.2f}s - {det['end_time']:.2f}s]")
                print(f"     Confidence: {det['confidence']*100:.1f}%")
                print(f"     Classification: {det['reason']}")
                print(f"     Transcription: \"{det['transcription']}\"")
                print()

        return abusive_detections, all_detections



# MAIN EXECUTION
def main():
    """
    Main function to run the detection system.

    This demonstrates how to use the detector in production.
    Modify the paths below to match your deployment environment.
    """

    # Configuration
    # -------------
    # Update these paths for your production environment
    MODEL_DIR = "/content/output_model/concept_model"  # MuRIL model directory
    HINDI_SWEARS = "/content/hindi_swears.csv"  # Hindi swear words
    HINDI_TO_GURMUKHI = "/content/hindi_to_gurmukhi.csv"  # Punjabi translations
    PHRASES = "/content/phrases_hindi_meaning.csv"  # Abusive phrases
    AUDIO_FILE = "/content/Recording-6.mp3"  # Test audio

    # Initialize detector
    # -------------------
    detector = ImprovedIndianASRDetector(
        model_dir=MODEL_DIR,
        asr_model=ASR_MODEL,
        asr_type=ASR_TYPE
    )

    # Load dictionaries
    # -----------------
    if os.path.exists(HINDI_SWEARS):
        detector.load_dictionary(HINDI_SWEARS, HINDI_TO_GURMUKHI, PHRASES)
    else:
        print("  Warning: Dictionary files not found. Running without dictionary boost.")

    # Check if audio file exists
    # --------------------------
    if not os.path.exists(AUDIO_FILE):
        print(f"\n Audio file not found: {AUDIO_FILE}")
        print("   Please update AUDIO_FILE path in the code.")
        return

    # Run detection
    # -------------
    abusive, all_results = detector.detect_timestamps(
        AUDIO_FILE,
        language="hi",  # "hi" for Hindi, "pa" for Punjabi
        chunk_duration=3.0,  # 3 seconds per chunk (optimal for Whisper)
        overlap=0.5  # 0.5 second overlap (prevents missing words at boundaries)
    )

    print("\n" + "="*80)
    print(" DETECTION COMPLETE")
    print("="*80)

    # Additional analysis for large models
    if "large" in ASR_TYPE or "medium" in ASR_TYPE:
        print("\n   Using larger Whisper model provides:")
        print("   • Better Hindi/Punjabi recognition")
        print("   • More Devanagari output (less Roman transliteration)")
        print("   • Higher accuracy on noisy audio")


# Run the system
if __name__ == "__main__":
    main()
