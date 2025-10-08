---
license: cc-by-nc-4.0
language:
- ha
- hau
datasets:
- CLEAR-Global/TWB-voice-TTS-Hausa-1.0-sampleset
---

# TWB Voice Hausa Multi-Speaker TTS 1.0

## Model Description

This is a multi-speaker Text-to-Speech (TTS) model for Hausa language, fine-tuned from the CML-TTS multilingual checkpoint using the YourTTS architecture. The model supports three distinct speakers: one female (spk_f_1) and two male speakers (spk_m_1, spk_m_2) and can generate high-quality Hausa speech from text input.

## Model Details

- **Model Architecture**: YourTTS (VITS-based)
- **Base Model**: CML-TTS Dataset multilingual checkpoint
- **Language**: Hausa (ha, hau)
- **Sample Rate**: 24 kHz  
- **Speakers**: 3 (spk_f_1: female, spk_m_1: male, spk_m_2: male)
- **Model Type**: Multi-speaker neural TTS
- **Framework**: Coqui TTS

### Model Architecture Details

- **Text Encoder**: 10-layer transformer with 2 attention heads
- **Hidden Channels**: 192
- **FFN Hidden Channels**: 768
- **Decoder**: HiFi-GAN with ResBlock type 2
- **Flow Layers**: 4 coupling layers
- **Posterior Encoder**: 16-layer WaveNet
- **Speaker Embedding**: 512-dimensional d-vectors
- **Language Embedding**: 4-dimensional language embeddings

## Training Data

The model was trained on approximately 30 hours of Hausa speech data from two high-quality sources:

### Female Speaker (spk_f_1)
- **Source**: TWB Voice project
- **Dialect**: Kenanci
- **Duration**: ~10 hours
- **Sample Dataset**: [TWB-voice-TTS-Hausa-1.0-sampleset](https://huggingface.co/datasets/CLEAR-Global/TWB-voice-TTS-Hausa-1.0-sampleset)
- **Description**: High-quality female voice recordings collected within the [TWB Voice 1.0](https://huggingface.co/datasets/CLEAR-Global/TWB-Voice-1.0) project.

### Male Speakers (spk_m_1, spk_m_2)  
- **Source**: Biblica open.bible project
- **Duration**: ~10 hours each (20 hours total)
- **Origin**: [open.bible](https://open.bible/)
- **Description**: Bible recordings featuring two consistent male speakers

### Data Preprocessing

- **Original Sample Rate**: 48 kHz → **Target**: 24 kHz (downsampled)
- **Audio Format**: Mono WAV files
- **Text Processing**: Lowercase conversion, **diacritics preserved**
- **Quality Filters**: 
  - Duration: 0.5s - 20s
  - Text length: minimum 10 characters
- **Train/Dev Split**: 95% train, 5% validation

## Character Set

The model supports standard Hausa orthography including diacritics:

**Characters**: `abcdefghijklmnopqrstuvwxyzăāɓɗƙƴū`

**Punctuation**: `!'(),-.:;?` and space

## Training 

The model was fine-tuned for 1000 epochs with the following configuration:
- **Batch Size**: 12
- **Learning Rate**: 0.0001 (generator and discriminator)
- **Mixed Precision**: FP16
- **Optimizer**: AdamW
- **Loss Components**: Mel loss (α=45.0), Speaker encoder loss (α=9.0)
- **GPU Setup**: 2x NVIDIA GeForce RTX 2080

### Evaluation Results

The model was evaluated on a set of 30 Hausa sentences by human evaluators using two criteria:

**Sample Evaluation Sentences:**
- "lokacin damuna shuka kan koriya shar."
- "lafiyarku tafi kuɗinku muhimmanci."
- "a kiyayi inda ake samun labarun magani ko kariya da cututtuka."

**Evaluation Metrics:**
- **Pronunciation Accuracy** (1-5 scale): **2.52** average
- **Speech Naturalness** (1-5 scale): **2.93** average

## Usage

### Installation

```bash
pip install coqui-tts torch scipy numpy huggingface_hub
```

### Quick Start

```python
from TTS.api import TTS
from huggingface_hub import hf_hub_download
import json
import tempfile
import scipy.io.wavfile as wavfile
import numpy as np
import os

# Download and setup model files
config_path = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Download required files
model_path = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "best_model_498283.pth")
speakers_file = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "speakers.pth")
language_ids_file = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "language_ids.json")
d_vector_file = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "d_vector.pth")
config_se_file = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "config_se.json")
model_se_file = hf_hub_download("CLEAR-Global/TWB-Voice-Hausa-TTS-1.0", "model_se.pth")

# Update config paths
config["speakers_file"] = speakers_file
config["language_ids_file"] = language_ids_file
config["d_vector_file"] = [d_vector_file]
config["model_args"]["speakers_file"] = speakers_file
config["model_args"]["language_ids_file"] = language_ids_file
config["model_args"]["d_vector_file"] = [d_vector_file]
config["model_args"]["speaker_encoder_config_path"] = config_se_file
config["model_args"]["speaker_encoder_model_path"] = model_se_file

# Save updated config
temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(config, temp_config, indent=2)
temp_config.close()

# Load TTS model
tts = TTS(model_path=model_path, config_path=temp_config.name)

# Generate speech
text = "lokacin damuna shuka kan koriya shar."
speaker = "spk_f_1"  # Options: spk_f_1 (female), spk_m_1 (male), spk_m_2 (male)

wav = tts.synthesizer.tts(text=text.lower(), speaker_name=speaker)
wav_array = np.array(wav, dtype=np.float32)
wavfile.write("output.wav", tts.synthesizer.output_sample_rate, wav_array)
```

### Batch Inference

For batch processing multiple sentences, use the provided inference script:

```bash
python3 inference.py /path/to/model.pth
# or
python3 inference.py /path/to/model/directory
```

The script will generate audio for all evaluation sentences with all available speakers.

## Model Limitations

- **Language**: Only supports Hausa language
- **Input Format**: **Requires lowercase text input**
- **Speakers**: Limited to 3 pre-trained speaker identities
- **Domain**: Trained primarily on religious/educational content and general speech
- **Code-switching**: Not optimized for mixed language input but the training data does include English names and places

## Ethical Considerations

- **Consent**: All training data used with appropriate permissions
- **Bias**: Model reflects the speech patterns and characteristics of the specific speakers in training data
- **Cultural Sensitivity**: Trained on religious and educational content; may not capture all dialectal variations
- **Use Cases**: Intended for educational, accessibility, and content creation purposes
- **Non-Commercial**: This model is released for non-commercial use only

## Licensing

This model is released under **CC-BY-NC** license. For commercial licensing or other uses, please contact tech@clearglobal.org.

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{yourtts-hausa-2025,
  title={YourTTS Hausa Multi-Speaker Text-to-Speech Model},
  author={Alp Öktem},
  year={2025},
  howpublished={Hugging Face Model Hub},
  url={https://huggingface.co/your-username/yourtts-hausa-multispeaker}
}
```

## Acknowledgments

This dataset was created by CLEAR Global with support from the Patrick J. McGovern Foundation. We acknowledge the following open source projects and resources that made this model possible:

- [**Idiap Coqui TTS**](github.com/idiap/coqui-ai-TTS): For the YourTTS architecture and training framework
- [**CML-TTS Dataset**](https://fredso.com.br/CML-TTS-Dataset/): For the multilingual base model
- [**TWB Voice Project**](https://twbvoice.org/): For high-quality Hausa voice data
- [**Biblica open.bible**](https://open.bible/): For Bible recording contributions

## Model Card Authors

Alp Öktem (alp.oktem@clearglobal.org)