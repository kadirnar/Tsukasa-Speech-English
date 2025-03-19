import subprocess
import os

# Test text to synthesize
test_text = "Vyvo is a technology-driven company that specializes in innovative wearable devices, AI-driven health and wellness solutions, and personal data monetization. The company was founded by Fabio Galdi and has positioned itself at the intersection of technology, health, and financial empowerment."

# Use testt.wav as reference as requested by the user
reference_audio = "/ephemeral/github/Tsukasa-Speech-English/testt.wav"
output_file = "/ephemeral/github/Tsukasa-Speech-English/english_tts_output.wav"

# Check if reference audio exists
if not os.path.exists(reference_audio):
    print(f"Reference audio file {reference_audio} not found. Please specify a valid file.")
    exit(1)

# Run the inference script
cmd = [
    "python", "inference_english.py",
    "--text", test_text,
    "--reference_audio", reference_audio,
    "--output", output_file,
    "--rate_of_speech", "1.0",
    "--alpha", "0.0",  # Higher weight for reference style
    "--beta", "0.0",   # Lower weight for content style
    "--diffusion_steps", "0",  # Skip diffusion as recommended
    "--embedding_scale", "1.0"
]

print("Running inference with command:")
print(" ".join(cmd))

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Inference completed successfully!")
    print(f"Output saved to: {output_file}")
    print("\nOutput from inference script:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Error during inference:")
    print(e.stderr)
