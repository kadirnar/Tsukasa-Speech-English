---
thumbnail: https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png
license: cc-by-nc-4.0
language:
- ja
pipeline_tag: text-to-speech
tags:
- '#StyleTTS'
- '#Japanese'
- Diffusion
- Prompt
- '#TTS'
- '#TexttoSpeech'
- '#speech'
- '#StyleTTS2'
---

<div style="text-align:center;">
  <img src="https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png" alt="Logo" style="width:300px; height:auto;">
</div>


# Tsukasa 司 Speech: Engineering the Naturalness and Rich Expressiveness 

**tl;dr** : I made a very cool japanese speech generation model.

日本語のモデルカードは[こちら](https://huggingface.co/Respair/Tsukasa_Speech/blob/main/README_JP.md)。

Part of a [personal project](https://github.com/Respaired/Project-Kanade), focusing on further advancing Japanese speech field. 

- Use the HuggingFace Space for **Tsukasa** (24khz): [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Shiki)
- HuggingFace Space for **Tsumugi** (48khz): [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Shiki)

- Join Shoukan lab's discord server, a comfy place I frequently visit -> [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/JrPSzdcM)

## What is this?

*Note*: This model only supports the Japanese language; but you can feed it Romaji if you use the Gradio demo.

This is a speech generation network, aimed at maximizing the expressiveness and Controllability of the generated speech. at its core it uses [StyleTTS 2](https://github.com/yl4579/StyleTTS2)'s architecture with the following changes:

- an entirely new data pre-processing pipeline
- Incorporating mLSTM Layers instead of regular PyTorch LSTM layers, and increasing the capacity of the text and prosody encoder by using a higher number of parameters
- Retrained PL-Bert, Pitch Extractor, Text Aligner from scratch
- Whisper's Encoder instead of WavLM for the SLM
- 48khz Config 
- improved Performance on non-verbal sounds and cues. such as sigh, pauses, etc. and also very slightly on laughter.
- a new way of sampling the Style Vectors.
- Promptable Speech Synthesizing.
- a Smart Phonemization algorithm that can handle Romaji inputs or a mixture of Japanese and Romaji.
- Fixed DDP and BF16 Training (mostly!)


There are two checkpoints you can use. Tsukasa & Tsumugi (placeholder).

Tsukasa was trained on ~800 hours of studio grade, high quality data. sourced mainly from games and novels, part of it from a private dataset.
So the Japanese is going to be the "anime japanese" (it's different than what people usually speak in real-life.)

For Tsumugi (placeholder) a subset of this data was used; at around ~300 hours but in a more controlled manner with additional manual cleaning & annotations. 

Unfortuantely Tsumugi's context length is capped and that means the model will not have enough information to handle the intonations as good as Tsukasa. 
it also only supports the first mode of Kotodama's inference, which means no voice design. 


Brought to you by:

- Soshyant (me)
- Auto Meta (Alignment AI Lab)
- Cryptowooser
- Buttercream

## Why does it matter?

Recently, there's a big trend towards larger models, increasing the scale. We're going the opposite way, trying to see how far we can push the limits by utilizing existing tools.
Maybe, just maybe, scale is not necessarily the answer.

There's also a few things that's related to Japanese. such as how we can improve the intonations for this language. what can be done to accurately annotate a text that can have various spellings depending on the context, etc.

## How to do ...

# Inference:

Gradio demo:
```bash
python app_tsuka.py
```

or check the inference notebook. before that, make sure you read the **Important Notes** section down below.

# Training:

First stage training:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
Second stage training: 
```bash
accelerate launch accelerate_train_second.py --config_path ./Configs/config.yml 
```

SLM Joint-Training doesn't work on multigpu. (you don't need it, i didn't use it too.)

or:

```bash
launch train_first.py --config_path ./Configs/config.yml
```

Third stage training (Kotodama, prompt encoding, etc.):
```
needs some cleaning, maybe not now.
```


## some ideas for future

I can think of a few things that can be improved, not nessarily by me, treat it as some sorts of suggestions:

- [o] changing the decoder ([fregrad](https://github.com/kaistmm/fregrad) looks promising)
- [o] retraining the Pitch Extractor using a different algorithm
- [o] while the quality of non-speech sounds have been improved, it cannot generate an entirely non-speech output, perhaps because of the hard alignement.
- [o] using the Style encoder as another modality in LLMs, since they have a detailed representation of the tone and expression of a speech (similar to Style-Talker).

## Pre-requisites
1. Python >= 3.11
2. Clone this repository:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```

## Training details

- 8x A40s + 2x V100s(32gb each)
- 750 ~ 800 hours of data
- Bfloat16
- Approximately 3 weeks of training, overall 3 months including the work spent on the data pipeline.
- Roughly 66.6 kg of CO2eq. of Carbon emitted if we base it on Google Cloud. (I didn't use Google, but the cluster is located in US, please treat it as a very rough approximation.) 


### Important Notes

1. Enabling the Diffusion sampler breaks the output:

***Unfortunately, depending on your hardware, the diffusion sampler may not work. This issue is beyond my control and seems to be related to how different hardware handles floating point operations. I can confirm that it works on A40, V100, and Google Colab's T4 GPUs, though it is not guaranteed to work even if you have the same GPUs. It doesn't matter if you use CPUs; the issue persists. This was a serious problem in the original StyleTTS2, but with the different ways of sampling i have included, you can simply turn off the diffusion sampler and use other methods for sampling the vectors, hopefully with minimal or no hit to the quality.***

2. Cannot reproduce the quality of the provided samples or cannot consistently get good results:

***Controllability comes at a cost, and that cost is the ease of use. This is especially true with a network composed of inherently non-deterministic modules. The system is highly sensitive to variations in your style vector. However, I'm confident that you can almost always achieve the expression you want in the most impressively natural way possible, but you should carefully tweak and play around with the inference parameters. also, not all speakers can handle all emotions consistently as some of them maybe didn't have that expression but you can cook a new one by using from a speaker that has it. I explain in more details on how to get consistently great results in the gradio space or inference notebook.***

3. [RuntimeError: The size of tensor a (512) must match the size of tensor b (some number) at non-singleton dimension 3]:

***your input is too long for a single inference run. use the Longform inference function. this is particularly challenging with the Tsumugi (placeholder) checkpoint as the context length of the mLSTM layer is capped at 512, meaning you cannot generate more than ~10 seconds of audio without relying on the Longform function. but this shouldn't be an issue with the other checkpoint. all in all, this should not be a serious problem. as there's no theoretical limit to the output thanks to the Longform algoirthm.***

3. short inputs sound un-impressive:

***everything said in 2, applies here. make sure your style vector is suitable for that. but generally it's not recommended to use a very short input.***

4. Nans in 2nd Stage:

***Your gradients are probably exploding. try clipping or your batch size is way too high. if that didn't work, feel free to do the first few epochs which is the pre-training stage, using the original DP script. or use DP entriely.***

4. Supporting English (or other languages):

***There is a wide gap between English and other languages, so I mostly focus on non-English projects. but the good folks at Shoukan labs are trying to train a multilingual model with English included. however, if i ever do myself, it'll be focused on something specific (let's say accents).***


## Some cool projects:

[Kokoro]("https://huggingface.co/spaces/hexgrad/Kokoro-TTS") - a very nice and light weight TTS, based on StyleTTS. supports Japanese and English.
[VoPho]("https://github.com/ShoukanLabs/VoPho") - a meta phonemizer to rule them all. it will automatically handle any languages with hand-picked high quality phonemizers.



## References
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

```
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}

```
