#!/usr/bin/env python3
"""
Simple inference script for YourTTS Hausa model using TTS API
Usage: python3 hausa_inference.py /path/to/model.pth
   or: python3 hausa_inference.py /path/to/model/directory
"""
import os
import sys
import torch
from TTS.api import TTS
import scipy.io.wavfile as wavfile
import numpy as np

DEFAULT_DEVICE = "cuda"

EVAL_SENTS_PATH = 'HAU_HAJIYAEVAL_SENTS.txt'

# Test sentences
HAUSA_TEST_SENTENCES = [l.strip().lower() for l in open(EVAL_SENTS_PATH, 'r').readlines()]

SPEAKER_NAMES = ['spk_m_2', 'spk_f_1']
# SPEAKER_NAMES = ['spk_f_1']

INFERENCE_ID = "hajiya-eval-set"

def setup_paths(input_path):
    """Setup model, config, and output paths based on input"""
    input_path = input_path.rstrip('/')
    
    if input_path.endswith('.pth'):
        # Direct model file specified
        model_file = input_path
        model_dir = os.path.dirname(model_file)
        config_file = os.path.join(model_dir, "config.json")
        model_name = os.path.splitext(os.path.basename(model_file))[0]
    else:
        # Directory specified - find best_model.pth
        model_dir = input_path
        model_file = os.path.join(model_dir, "best_model.pth")
        config_file = os.path.join(model_dir, "config.json")
        model_name = "best_model"
    
    # Create output directory inside model directory
    output_dir = os.path.join(model_dir, f"inference_{INFERENCE_ID}_{model_name}")
    
    return model_file, config_file, output_dir

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 hausa_inference.py /path/to/model.pth")
        print("   or: python3 hausa_inference.py /path/to/model/directory")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Setup paths
    model_file, config_file, output_dir = setup_paths(input_path)
    
    # Verify files exist
    if not os.path.exists(model_file):
        print(f"Error: Model file not found: {model_file}")
        sys.exit(1)
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    print(f"Model: {model_file}")
    print(f"Config: {config_file}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = DEFAULT_DEVICE
    print(f"Using device: {device}")
    
    try:
        # Initialize TTS
        tts = TTS(model_path=model_file, config_path=config_file).to(device)
        print("Model loaded successfully!")
        
        # Get the synthesizer for direct access
        synthesizer = tts.synthesizer
        
        #Generate speech for each speaker
        for speaker_name in SPEAKER_NAMES:
            # Generate speech for each test sentence
            for i, text in enumerate(HAUSA_TEST_SENTENCES):
                output_file = os.path.join(output_dir, f"sample_{i+1:02d}_{speaker_name}.wav")
                
                print(f"Generating: '{text}'")
                
                try:
                    # Use direct synthesizer call with speaker_name (this works!)
                    wav = synthesizer.tts(
                        text=text,
                        speaker_name=speaker_name
                    )
                    
                    wav_array = np.array(wav, dtype=np.float32)
                    wavfile.write(output_file, synthesizer.output_sample_rate, wav_array)
                    
                    print(f"✅ Saved: {output_file}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                
        
        print(f"\n🎉 Inference complete! Check {output_dir}/ for audio files")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
