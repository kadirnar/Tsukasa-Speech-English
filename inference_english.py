#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
English TTS Inference Script
----------------------------
This script generates speech from text using the English TTS model.
"""

import argparse
import os
import torch
import numpy as np
import librosa
import torchaudio
import yaml
from munch import Munch
import scipy.signal
from text_utils import TextCleaner
from models import build_model, load_ASR_models, load_F0_models
from utils import recursive_munch
from Utils.PLBERT.util import load_plbert
from Utils.phonemize.english_phonemizer import english_phonemize
import scipy.io.wavfile

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(0)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epoch_2nd_00044.pth")
print(f"Loading model from: {model_path}")

# Load configuration
config = yaml.safe_load(open("Configs/config_english_second_stage.yml"))
model_params = recursive_munch(config['model_params'])

# Load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)
print("text_aligner loaded")

# Load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)
print("pitch_extractor loaded")

# Load BERT model
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)
print("bert loaded")

# Build model
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# Load model parameters
params_whole = torch.load(model_path, map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print(f'{key} loaded')
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]

# Setup mel spectrogram transform
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    """Convert lengths to mask."""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    """Preprocess audio waveform to mel spectrogram."""
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def get_style_vector(wav_path):
    """Extract style vector from reference audio."""
    try:
        wave, sr = librosa.load(wav_path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)
    except Exception as e:
        print(f"Error extracting style vector: {e}")
        return None

def inference(text=None, ref_s=None, alpha=0.3, beta=0.7, diffusion_steps=0, embedding_scale=1, rate_of_speech=1.):
    """
    Generate speech from text using the trained model.
    
    Args:
        text: Input phonemes
        ref_s: Reference style vector
        alpha: Weight for reference style
        beta: Weight for content style
        diffusion_steps: Number of diffusion steps (not used)
        embedding_scale: Scale for embedding
        rate_of_speech: Controls the speed of speech
    
    Returns:
        numpy array: Generated audio waveform
    """
    textcleaner = TextCleaner()
    
    # Ensure text is not too long
    if text is None:
        return None
    
    if len(text) > 300:
        print("Warning: Text is too long, truncating...")
        text = text[:300]
    
    tokens = textcleaner(text)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        
        # Skip diffusion and use style vectors directly
        ref = ref_s[:, :128]
        s = ref_s[:, 128:]

        # Apply style mixing
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        # Generate speech
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        
        x = model.predictor.lstm(d)
        x_mod = model.predictor.prepare_projection(x)
        duration = model.predictor.duration_proj(x_mod)

        duration = torch.sigmoid(duration).sum(axis=-1) / rate_of_speech
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths[0].item(), int(pred_dur.sum().data)).to(device)
        
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # Encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        # Generate audio
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        # Get raw audio without trimming the end
        wav_out = out.squeeze().cpu().numpy()
    
    # Apply minimal post-processing to avoid muffled sound
    try:
        # Normalize the output (but not too aggressively)
        wav_out = wav_out / (np.max(np.abs(wav_out)) + 1e-7) * 0.9
        
        # Remove DC offset
        wav_out = wav_out - np.mean(wav_out)
        
        # Apply a very gentle high-pass filter just to remove sub-bass rumble
        b, a = scipy.signal.butter(2, 40/(24000/2), 'highpass')
        wav_out = scipy.signal.filtfilt(b, a, wav_out)
    except Exception as e:
        print(f"Warning: Error during audio post-processing: {e}")
    
    return wav_out

def main():
    parser = argparse.ArgumentParser(description="English TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--reference_audio", type=str, required=True, help="Path to reference audio file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for reference style")
    parser.add_argument("--beta", type=float, default=0.7, help="Weight for content style")
    parser.add_argument("--diffusion_steps", type=int, default=0, help="Number of diffusion steps")
    parser.add_argument("--embedding_scale", type=float, default=1.0, help="Scale for embedding")
    parser.add_argument("--rate_of_speech", type=float, default=1.0, help="Controls the speed of speech")
    
    args = parser.parse_args()
    
    print(f"Loading reference audio from: {args.reference_audio}")
    ref_s = get_style_vector(args.reference_audio)
    
    if ref_s is None:
        print("Error: Failed to extract style vector from reference audio.")
        return
    
    print(f"Generating speech for text: {args.text}")
    # Convert text to phonemes
    phonemes = english_phonemize(args.text)
    print(len(phonemes))
    
    # Generate speech
    wav = inference(
        text=phonemes,
        ref_s=ref_s,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
        rate_of_speech=args.rate_of_speech
    )
    
    if wav is None:
        print("Error: Failed to generate speech.")
        return
    
    # Save output
    print(f"Saving output to: {args.output}")
    scipy.io.wavfile.write(args.output, 24000, wav.astype(np.float32))
    print("Done!")

if __name__ == "__main__":
    main()
