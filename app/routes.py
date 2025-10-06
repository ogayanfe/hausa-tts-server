from flask import Blueprint, request, jsonify, send_file
import tempfile
import os
from model import generate_audio

main = Blueprint('main', __name__)


@main.route('/')
def home():
    return jsonify({"message": "Gani Hausa TTS Server Running ✅"})


@main.route('/api/tts', methods=['POST'])
def tts():
    """
    Expects JSON body:
    {
        "text": "some hausa text",
        "speaker": "spk_m_1"   # optional
    }
    """
    try:
        data = request.get_json()
        text: str = data.get("text", "").strip().lower()
        speaker: str = data.get("speaker", "spk_m_2")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate audio using the model
        audio_path = generate_audio(text, speaker)

        # Send the resulting WAV file
        return send_file(
            audio_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="output.wav"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
