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

*tl;dr* : made a very cool japanese speech generation net.

日本語のモデルカードは[こちら](https://huggingface.co/Respair/Tsukasa_Speech/blob/main/README_JP.md)。

Part of a [personal project](https://github.com/Respaired/Project-Kanade), focusing on further advancing Japanese speech field. 

- Use the HuggingFace Space for *Tsukasa* (24khz): [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Tsukasa_Speech)
- HuggingFace Space for *Tsumugi* (48khz): [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Tsumugi_48khz)

- Join Shoukan lab's discord server, a comfy place I frequently visit -> [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/JrPSzdcM)

Github's repo:
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Respaired/Tsukasa-Speech)

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
- [Auto Meta](https://github.com/Alignment-Lab-AI)
- [Cryptowooser](https://github.com/cryptowooser)
- [Buttercream](https://github.com/korakoe)

Special thanks to Yinghao Aaron Li, the Author of StyleTTS which this work is based on top of that. <br> He is one of the most talented Engineers I've ever seen in this field. 
Also Karesto and Raven for their help in debugging some of the scripts. wonderful people.

## Why does it matter?

Recently, there's a big trend towards larger models, increasing the scale. We're going the opposite way, trying to see how far we can push the limits by utilizing existing tools.
Maybe, just maybe, scale is not necessarily the answer.

There's also a few things that's related to Japanese. such as how we can improve the intonations for this language. what can be done to accurately annotate a text that can have various spellings depending on the context, etc.

## How to do ...

## Pre-requisites
1. Python >= 3.11
2. Clone this repository:
```bash
git clone https://huggingface.co/Respair/Tsukasa_Speech
cd Tsukasa_Speech
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```


# Inference:

Gradio demo:
```bash
python app_tsuka.py
```

or check the inference notebook. before that, make sure you read the **Important Notes** section down below.

# Training:

## Pre-requisites

**First stage training**:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
**Second stage training**:
```bash
accelerate launch accelerate_train_second.py --config_path ./Configs/config.yml 
```

SLM Joint-Training doesn't work on multigpu. (you don't need it, i didn't use it too.)

or:

```bash
launch train_first.py --config_path ./Configs/config.yml
```

**Third stage training* (Kotodama, prompt encoding, etc.)**:
```
not planned right now, due to some constraints, but feel free to replicate.
```


## some ideas for future

I can think of a few things that can be improved, not nessarily by me, treat it as some sorts of suggestions:

- [o] changing the decoder ([fregrad](https://github.com/kaistmm/fregrad) looks promising)
- [o] retraining the Pitch Extractor using a different algorithm
- [o] while the quality of non-speech sounds have been improved, it cannot generate an entirely non-speech output, perhaps because of the hard alignement.
- [o] using the Style encoder as another modality in LLMs, since they have a detailed representation of the tone and expression of a speech (similar to Style-Talker).


## Training details

- 8x A40s + 2x V100s(32gb each)
- 750 ~ 800 hours of data
- Bfloat16
- Approximately 3 weeks of training, overall 3 months including the work spent on the data pipeline.
- Roughly 66.6 kg of CO2eq. of Carbon emitted if we base it on Google Cloud. (I didn't use Google, but the cluster is located in US, please treat it as a very rough approximation.) 


### Important Notes

Please Check [here](https://github.com/Respaired/Tsukasa-Speech/blob/main/important_notes.md).

Any questions?
```email
saoshiant@protonmail.com
```
or simply DM me on discord.

## Some cool projects:

[Kokoro]("https://huggingface.co/spaces/hexgrad/Kokoro-TTS") - a very nice and light weight TTS, based on StyleTTS. supports Japanese and English.<br>
[VoPho]("https://github.com/ShoukanLabs/VoPho") - a meta phonemizer to rule them all. it will automatically handle any languages with hand-picked high quality phonemizers.



## References
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)
- [litain's Moe Speech](https://huggingface.co/datasets/litagin/moe-speech) a very cool dataset you can use in case i couldn't release mine 
```
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}

```
