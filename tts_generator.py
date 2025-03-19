#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
English TTS Generator
---------------------
A tool for generating speech using the English TTS model with various quality settings.
"""

import argparse
import os
import subprocess
import sys

def generate_speech(text, reference_audio, output_file, 
                   alpha=0.7, beta=0.3, 
                   diffusion_steps=0, 
                   embedding_scale=1.0,
                   rate_of_speech=1.0):
    """
    Generate speech using the English TTS model.
    
    Args:
        text (str): Text to synthesize
        reference_audio (str): Path to reference audio file
        output_file (str): Path to save the synthesized audio
        alpha (float): Weight for reference style (0.0-1.0)
        beta (float): Weight for content style (0.0-1.0)
        diffusion_steps (int): Number of diffusion steps (0 to skip)
        embedding_scale (float): Scale for embedding in diffusion
        rate_of_speech (float): Controls the speed of speech synthesis
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if reference audio exists
    if not os.path.exists(reference_audio):
        print(f"Error: Reference audio file {reference_audio} not found.")
        return False
    
    # Build command
    cmd = [
        "python", "inference_english.py",
        "--text", text,
        "--reference_audio", reference_audio,
        "--output", output_file,
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--diffusion_steps", str(diffusion_steps),
        "--embedding_scale", str(embedding_scale),
        "--rate_of_speech", str(rate_of_speech)
    ]
    
    print(f"Generating speech with the following settings:")
    print(f"  - Text: {text}")
    print(f"  - Reference audio: {reference_audio}")
    print(f"  - Output file: {output_file}")
    print(f"  - Alpha (reference style weight): {alpha}")
    print(f"  - Beta (content style weight): {beta}")
    print(f"  - Diffusion steps: {diffusion_steps}")
    print(f"  - Embedding scale: {embedding_scale}")
    print(f"  - Rate of speech: {rate_of_speech}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\nGeneration completed successfully!")
        print(f"Output saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print("\nError during generation:")
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate speech using the English TTS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    
    parser.add_argument("--reference_audio", type=str, default="/ephemeral/github/Tsukasa-Speech-English/testt.wav",
                        help="Path to reference audio file")
    
    parser.add_argument("--output", type=str, default="tts_output.wav",
                        help="Path to save the synthesized audio")
    
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for reference style (0.0-1.0)")
    
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight for content style (0.0-1.0)")
    
    parser.add_argument("--diffusion_steps", type=int, default=0,
                        help="Number of diffusion steps (0 to skip)")
    
    parser.add_argument("--embedding_scale", type=float, default=1.0,
                        help="Scale for embedding in diffusion")
    
    parser.add_argument("--rate_of_speech", type=float, default=1.0,
                        help="Controls the speed of speech synthesis")
    
    parser.add_argument("--preset", type=str, choices=["quality", "fast", "balanced"],
                        help="Use a preset configuration")
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset:
        if args.preset == "quality":
            args.alpha = 0.8
            args.beta = 0.2
            args.diffusion_steps = 0
            args.rate_of_speech = 0.95
        elif args.preset == "fast":
            args.alpha = 0.6
            args.beta = 0.4
            args.diffusion_steps = 0
            args.rate_of_speech = 1.1
        elif args.preset == "balanced":
            args.alpha = 0.7
            args.beta = 0.3
            args.diffusion_steps = 0
            args.rate_of_speech = 1.0
    
    # Generate speech
    success = generate_speech(
        text=args.text,
        reference_audio=args.reference_audio,
        output_file=args.output,
        alpha=args.alpha,
        beta=args.beta,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale,
        rate_of_speech=args.rate_of_speech
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
