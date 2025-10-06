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
    config_path = hf_hub_download(MODEL_REPO, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_path = hf_hub_download(MODEL_REPO, "best_model_498283.pth")
    speakers_file = hf_hub_download(MODEL_REPO, "speakers.pth")
    language_ids_file = hf_hub_download(MODEL_REPO, "language_ids.json")
    d_vector_file = hf_hub_download(MODEL_REPO, "d_vector.pth")
    config_se_file = hf_hub_download(MODEL_REPO, "config_se.json")
    model_se_file = hf_hub_download(MODEL_REPO, "model_se.pth")

    config["speakers_file"] = speakers_file
    config["language_ids_file"] = language_ids_file
    config["d_vector_file"] = [d_vector_file]
    config["model_args"]["speakers_file"] = speakers_file
    config["model_args"]["language_ids_file"] = language_ids_file
    config["model_args"]["d_vector_file"] = [d_vector_file]
    config["model_args"]["speaker_encoder_config_path"] = config_se_file
    config["model_args"]["speaker_encoder_model_path"] = model_se_file

    temp_config = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_config.name, "w") as f:
        json.dump(config, f)

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
