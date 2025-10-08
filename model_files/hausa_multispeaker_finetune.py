#!/usr/bin/env python3
"""
Fine-tuning script for YourTTS Catalan model using pre-trained CML-TTS checkpoint
Based on the successful approach from African languages training
"""
import os
import torch
import json
import gdown
import tarfile
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.managers import save_file
from tqdm import tqdm

torch.set_num_threads(24)

def formatter(root_path, meta_file, **kwargs):
    """
    Formatter for Hausa dataset - converts CSV to TTS format
    """
    meta_path = os.path.join(root_path, meta_file)
    items = []
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2:
                continue
                
            audio_file = parts[0]  
            text = parts[1]
            
            # Create full path to audio file
            full_audio_path = os.path.join(root_path, audio_file)
            
            # Extract speaker name from filename: festcat_pau_001264.wav -> pau
            filename = os.path.basename(audio_file)
            name_parts = filename.split('_')
            if len(name_parts) >= 2:
                speaker_name = '_'.join(name_parts[1:4]) #FORMAT: spk_m_1, spk_m_2, spk_f_1
            else:
                speaker_name = "unknown"
            
            # Check if audio file actually exists
            if not os.path.exists(full_audio_path):
                print(f"Warning: Audio file not found: {full_audio_path}")
                continue
            
            items.append({
                "text": text,
                "audio_file": full_audio_path,
                "speaker_name": speaker_name,
                "language": "ha",
                "root_path": root_path,
                "audio_unique_name": filename
            })
    
    return items


def compute_embeddings(
    model_path,
    config_path,
    output_path,
    old_speakers_file=None,
    old_append=False,
    config_dataset_path=None,
    formatter=None,
    dataset_name=None,
    dataset_path=None,
    meta_file_train=None,
    meta_file_val=None,
    disable_cuda=False,
    no_eval=False,
):
    """Compute speaker embeddings - adapted from colleague's code"""
    use_cuda = torch.cuda.is_available() and not disable_cuda

    if config_dataset_path is not None:
        c_dataset = load_config(config_dataset_path)
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not no_eval)
    else:
        c_dataset = BaseDatasetConfig()
        c_dataset.dataset_name = dataset_name
        c_dataset.path = dataset_path
        if meta_file_train is not None:
            c_dataset.meta_file_train = meta_file_train
        if meta_file_val is not None:
            c_dataset.meta_file_val = meta_file_val
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=not no_eval, formatter=formatter)

    if meta_data_eval is None:
        samples = meta_data_train
    else:
        samples = meta_data_train + meta_data_eval

    encoder_manager = SpeakerManager(
        encoder_model_path=model_path,
        encoder_config_path=config_path,
        d_vectors_file_path=old_speakers_file,
        use_cuda=use_cuda,
    )

    class_name_key = encoder_manager.encoder_config.class_name_key

    # compute speaker embeddings
    if old_speakers_file is not None and old_append:
        speaker_mapping = encoder_manager.embeddings
    else:
        speaker_mapping = {}

    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        embedding_key = fields["audio_unique_name"] if "audio_unique_name" in fields else os.path.basename(fields["audio_file"])

        # Only update the speaker name when the embedding is already in the old file.
        if embedding_key in speaker_mapping:
            speaker_mapping[embedding_key]["name"] = class_name
            continue

        if old_speakers_file is not None and embedding_key in encoder_manager.clip_ids:
            # get the embedding from the old file
            embedd = encoder_manager.get_embedding_by_clip(embedding_key)
        else:
            # extract the embedding
            embedd = encoder_manager.compute_embedding_from_clip(audio_file)

        # create speaker_mapping if target dataset is defined
        speaker_mapping[embedding_key] = {}
        speaker_mapping[embedding_key]["name"] = class_name
        speaker_mapping[embedding_key]["embedding"] = embedd

    if speaker_mapping:
        # save speaker_mapping if target dataset is defined
        if os.path.isdir(output_path):
            mapping_file_path = os.path.join(output_path, "speakers.pth")
        else:
            mapping_file_path = output_path

        if os.path.dirname(mapping_file_path) != "":
            os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

        save_file(speaker_mapping, mapping_file_path)
        print("Speaker embeddings saved at:", mapping_file_path)



# Configuration
ROOT_DIR = "/home/twbgmy/play/coqui-play"
DATASET_PATH = os.path.join(ROOT_DIR, "yourtts_hausa/data_multi3")
OUT_PATH = os.path.join(ROOT_DIR, "yourtts_hausa")
RESTORE_PATH = os.path.join(ROOT_DIR, "checkpoints_yourtts_cml_tts_dataset/best_model.pth")

# Create output directory
os.makedirs(OUT_PATH, exist_ok=True)

# Name of the run for the Trainer
RUN_NAME = "YourTTS-Hausa-Multispeaker3-Finetune"

# Training parameters - smaller values for fine-tuning
BATCH_SIZE = 12  # Smaller batch size for fine-tuning
SAMPLE_RATE = 24000  # Match your dataset
MAX_AUDIO_LEN_IN_SECONDS = 20
MIN_AUDIO_LEN_IN_SECONDS = 0.5

# Check if metadata file exists
metadata_file = os.path.join(DATASET_PATH, "metadata.csv")
if not os.path.exists(metadata_file):
    print(f"ERROR: Metadata file not found at {metadata_file}")
    sys.exit(1)


# Dataset configuration
dataset_conf = BaseDatasetConfig(
    dataset_name="hausa_tts",
    meta_file_train="metadata.csv",
    meta_file_val="",
    language="ha",
    path=DATASET_PATH
)

# Speaker encoder paths from the pre-trained model
SPEAKER_ENCODER_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints_yourtts_cml_tts_dataset/model_se.pth")
SPEAKER_ENCODER_CONFIG_PATH = os.path.join(ROOT_DIR, "checkpoints_yourtts_cml_tts_dataset/config_se.json")

# Check if speaker embeddings exist, if not compute them
embeddings_file = os.path.join(DATASET_PATH, "speakers.pth")
if not os.path.isfile(embeddings_file):
    print(f">>> Computing speaker embeddings for Hausa dataset")
    compute_embeddings(
        SPEAKER_ENCODER_CHECKPOINT_PATH,
        SPEAKER_ENCODER_CONFIG_PATH,
        embeddings_file,
        formatter=formatter,
        dataset_name=dataset_conf.dataset_name,
        dataset_path=dataset_conf.path,
        meta_file_train=dataset_conf.meta_file_train,
        meta_file_val=dataset_conf.meta_file_val,
    )

D_VECTOR_FILES = [embeddings_file]

# Audio config
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Model arguments
model_args = VitsArgs(
    spec_segment_size=62,
    hidden_channels=192,
    hidden_channels_ffn_text_encoder=768,
    num_heads_text_encoder=2,
    num_layers_text_encoder=10,
    kernel_size_text_encoder=3,
    dropout_p_text_encoder=0.1,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",
    use_speaker_encoder_as_loss=False,
    use_language_embedding=True,
    embedded_language_dim=4
)

# Hausa characters - including special characters and diacritics
# HAUSA_CHARS = [
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
#     'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
#     'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
#     # Hausa specific characters with diacritics
#     'Ă', 'ă', 'Ɓ', 'ɓ', 'Ɗ', 'ɗ', 'Ƙ', 'ƙ', 'Ƴ', 'ƴ', 'Ƴ', 'ƴ'
# ]

HAUSA_CHARS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # Hausa specific characters with diacritics
    'ă', 'ā', 'ɓ', 'ɗ', 'ƙ', 'ƴ', 'ū'
]


HAUSA_PUNCT = [' ', '!', "'", '(', ')', ',', '-', '.', ':', ';','?']

# Test sentences in Hausa
# HAUSA_TEST_SENTENCES = [
#     "Sannu da zuwa, ina kwana?",
#     "Hausa harshe ne na Afirka ta yamma.",
#     "Kasuwar Kurmi ta shahara sosai a Kano.",
#     "Zubairu ya gaya wa Habiba cewa ba ya son taimakawa.",
#     "Mun koyi Hausa a makaranta tun muna kanana.",
#     "Nigeria kasa ce mai yawan al'umma a Afirka."
# ]

HAUSA_TEST_SENTENCES = [
    "sannu da zuwa, ina kwana?",
    "hausa harshe ne na afrika ta yamma.",
    "kasuwar kurmi ta shahara sosai a kano.",
    "gama mun ji labarin bangaskiyarku a cikin yesu kiristi da kuma ƙaunar da kuke yi saboda dukan tsarkaka."
]


# Get unique speakers from the dataset
speakers_list = set()

with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        filename = line.split('|')[0]
        if 'hausa_spk' in filename:
            # Extract speaker from filename like: wavs/hausa_spk54_...
            parts = os.path.basename(filename).split('_')
            if len(parts) >= 2:
                speakers_list.add('_'.join(parts[1:4]))  # spk54

print(f"Found speakers for testing: {speakers_list}")

# Create test sentences for available speakers
TEST_SENTENCES = []
for text in HAUSA_TEST_SENTENCES:  
    for speaker in speakers_list:  
        TEST_SENTENCES.append([text, speaker, None, "ha"])


# Training configuration
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description=f"""
            - YourTTS fine-tuned on Hausa TTS dataset.
            - Fine-tuning from CML-TTS multilingual checkpoint.
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=4,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    print_step=100,  
    plot_step=100,
    save_step=1000,  
    save_n_checkpoints=40,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=True,
    lr_gen=0.0001,          # Add this line
    lr_disc=0.0001,         # Add this line
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="no_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="".join(HAUSA_CHARS),
        punctuations="".join(HAUSA_PUNCT),
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=[dataset_conf],
    cudnn_benchmark=False,
    min_audio_len=int(SAMPLE_RATE * MIN_AUDIO_LEN_IN_SECONDS),
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=True,
    test_sentences=TEST_SENTENCES,
    eval_split_size=0.05,  # 5% for evaluation,
    speaker_encoder_loss_alpha=9.0,
    shuffle=True
)

print(f"Configuration:")
print(f"- Dataset path: {DATASET_PATH}")
print(f"- Output path: {OUT_PATH}")
print(f"- Restore path: {RESTORE_PATH}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Sample rate: {SAMPLE_RATE}")
print(f"- Available speakers: {len(speakers_list)}")
print(f"- Using default learning rates (no override)")

# Load dataset samples
print("Loading dataset...")
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    formatter=formatter,
    eval_split_size=config.eval_split_size,
)

print(f"Dataset loaded: {len(train_samples)} train, {len(eval_samples)} eval samples")

# Initialize model
print("Initializing model...")
model = Vits.init_from_config(config)

# Initialize trainer
print("Setting up trainer...")
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=False),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

#Debug 

speaker_embeddings = torch.load(embeddings_file)
print("First 5 speaker embedding keys:")
for i, key in enumerate(list(speaker_embeddings.keys())[:5]):
    print(f"  {key}")

print("\nFirst 5 audio_unique_names from dataset:")
for i, sample in enumerate(train_samples[:5]):
    if 'audio_unique_name' in sample:
        print(f"  {sample['audio_unique_name']}")
    else:
        print(f"  No audio_unique_name in sample {i}")

available_speakers = list(speaker_embeddings.keys())[:4]  # Just use first few for testing

print(f"Available speakers in embeddings: {available_speakers[:5]}")

# Start training
print("🚀 Starting fine-tuning...")
trainer.fit()
