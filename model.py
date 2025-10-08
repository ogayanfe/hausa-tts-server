from TTS.api import TTS
from huggingface_hub import hf_hub_download
import json
import scipy.io.wavfile as wavfile
import numpy as np
import os
import tempfile

# Download model files once at import
MODEL_REPO = "CLEAR-Global/TWB-Voice-Hausa-TTS-1.0"

# Pre-download required model assets


def setup_model():
    # Folder containing all your downloaded Hausa TTS model files
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_files")
    config_path = hf_hub_download(MODEL_REPO, "config.json")

    # Define paths to each required file
    config_path = os.path.join(MODEL_DIR, "config.json")
    model_path = os.path.join(MODEL_DIR, "best_model_498283.pth")
    speakers_file = os.path.join(MODEL_DIR, "speakers.pth")
    language_ids_file = os.path.join(MODEL_DIR, "language_ids.json")
    d_vector_file = os.path.join(MODEL_DIR, "d_vector.pth")
    config_se_file = os.path.join(MODEL_DIR, "config_se.json")
    model_se_file = os.path.join(MODEL_DIR, "model_se.pth")

    # Load and update config with correct local file paths
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["speakers_file"] = speakers_file
    config["language_ids_file"] = language_ids_file
    config["d_vector_file"] = [d_vector_file]
    config["model_args"]["speakers_file"] = speakers_file
    config["model_args"]["language_ids_file"] = language_ids_file
    config["model_args"]["d_vector_file"] = [d_vector_file]
    config["model_args"]["speaker_encoder_config_path"] = config_se_file
    config["model_args"]["speaker_encoder_model_path"] = model_se_file

    # Write updated config to a temp file (TTS expects a path)
    temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_config.name, "w", encoding="utf-8") as f:
        json.dump(config, f)

    # Initialize TTS with local model paths
    tts = TTS(model_path=model_path, config_path=temp_config.name)
    return tts


# Load once when server starts
print("🔄 Loading Hausa TTS model... (may take a minute)")
tts_instance = setup_model()
print("✅ Hausa TTS model loaded successfully.")


def generate_audio(text: str, speaker: str = "spk_m_2",) -> str:
    """
    Generates Hausa speech for the given text and returns a path to the output file.
    """
    wav_data = tts_instance.synthesizer.tts(  # type: ignore
        text=text.lower(), speaker_name=speaker
    )

    wav_array = np.array(wav_data, dtype=np.float32)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    wavfile.write(
        output_path, tts_instance.synthesizer.output_sample_rate, wav_array  # type: ignore
    )

    return output_path
