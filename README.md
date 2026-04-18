<div align="center">

# VoxCPM2 Portable

**Portable Windows build of VoxCPM2 — multilingual TTS with Voice Design, Cloning & LoRA fine-tuning.**

[![Stars](https://img.shields.io/github/stars/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/stargazers)
[![License](https://img.shields.io/github/license/timoncool/VoxCPM2_portable?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/timoncool/VoxCPM2_portable?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/commits/main)
[![Downloads](https://img.shields.io/github/downloads/timoncool/VoxCPM2_portable/total?style=flat-square)](https://github.com/timoncool/VoxCPM2_portable/releases)

**[Русская версия](README_RU.md)**

</div>

Generate natural multilingual speech, design brand-new voices from text descriptions, clone any voice from a reference clip, and **train your own LoRA on ~10 audio clips** — **100% local**, no cloud, no API keys. One-click install on Windows, runs on any NVIDIA GPU with 8+ GB VRAM.

Built on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) by OpenBMB — a tokenizer-free 2B-parameter diffusion autoregressive TTS model trained on 2M+ hours of speech.

## Why VoxCPM2 Portable?

- **Free forever** — no API keys, no credits, no usage limits
- **Private** — your voice data never leaves your machine
- **Portable** — everything in one folder, copy to USB, delete = uninstall
- **One-click** — `install.bat` → `run.bat` → generate speech
- **30 languages** — Russian, English, Chinese, French, German, Japanese, Korean and more
- **LoRA training in the UI** — fine-tune on your own 10-20 clips in 5-15 min on RTX 4090

## Features

### Text-to-Speech
- **30 languages** — RU/EN/ZH (+9 Chinese dialects)/AR/FR/DE/HI/IT/JA/KO/PT/ES + more — auto language detection
- **48 kHz studio output** via AudioVAE V2 super-resolution (16→48 kHz)
- **Natural prosody** — tokenizer-free diffusion autoregressive architecture
- Output formats: **MP3** (default), WAV, FLAC, OGG

### Voice Design
- Create voices from **text description** — gender, age, tone, emotion, pace, accent
- Zero-shot — no reference audio needed
- 6 ready-made examples (EN+RU) with one-click fill

### Voice Cloning (with optional Ultimate mode)
- Clone any voice from **5-50 seconds** of reference audio
- **Voice pack** bundled (~100 voices, RU/EN/FR/DE/JP/KR/AR)
- Extra **743 Russian voices** on-demand from `Slait/russia_voices`
- Style control: `slightly faster, cheerful tone` / `whispering, intimate` / `slow and dramatic`
- **Ultimate mode** — fill transcript field → model uses `prompt_wav_path + prompt_text + reference_wav_path` for max fidelity
- Optional **ZipEnhancer denoise** for noisy references

### LoRA Fine-Tuning (in the UI!)
- Train on **10-20 clips** (~5-10 min audio) → custom voice adaptation in 5-15 min on RTX 4090
- Bundled training script from OpenBMB (`training/scripts/train_voxcpm_finetune.py`)
- Live progress log
- Saved to `lora/<name>/step_XXXX/` — auto-appears in dropdown
- **Hot-swap**: switch between LoRAs without restart via `unload_lora` + `load_lora`
- Works across all modes (TTS / Voice Design / Voice Cloning)

### All model parameters exposed
CFG Scale · Inference Steps · Min/Max length · Retry-on-bad-case · Retry max attempts · Retry ratio threshold · Text normalization (wetext) · Denoise reference · Streaming (live progress) · Seed + Lock

### Interface
- **i18n RU/EN** via `gr.I18n` — switch via browser language
- **Dark theme** with gradient header
- **Bundled FFmpeg** portable (for MP3/OGG encoding)
- **Auto-download** — model (~4-5 GB) + voice pack on first run
- **Auto-port, auto-browser** — opens on `localhost` automatically

### GPU Accelerators (out of the box)
| GPU | Flash Attention 2 | SDPA flash | bfloat16 | AMP (training) |
|-----|:---:|:---:|:---:|:---:|
| RTX 30xx / 40xx / 50xx | ✅ | ✅ | ✅ | ✅ |
| GTX 10xx / RTX 20xx | ❌ | ✅ | ✅ | ✅ |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| Disk | 15 GB | 30 GB (with voice pack & LoRA) |
| OS | Windows 10/11 | Windows 11 |
| GPU | RTX 2080 / RTX 3060 | RTX 4070+ |

CPU-only mode supported but **very slow** (minutes per phrase).

## Quick Start

### 1. Clone
```bash
git clone https://github.com/timoncool/VoxCPM2_portable.git
cd VoxCPM2_portable
```

### 2. Install
```
install.bat
```
Select your GPU type (6 options). Installs portable Python 3.12 + PyTorch 2.7.1 + voxcpm + Flash Attention 2 + FFmpeg + default voice pack. Nothing system-wide.

### 3. Run
```
run.bat
```
Browser opens automatically. Model downloads on first run (~4-5 GB to `models/`).

## Launchers

| Script | Description |
|--------|-------------|
| `install.bat` | One-click installer — Python + PyTorch + voxcpm + accelerators + FFmpeg + voice pack |
| `run.bat` | Launch Gradio UI with full environment isolation |
| `update.bat` | Update portable wrapper + voxcpm package |

## LoRA Training Workflow

1. **Prepare dataset**: 10-50 WAV/MP3 clips (3-15 sec each) of one speaker + their transcripts
2. Open **LoRA** tab in the UI
3. Drag-and-drop audio files
4. Paste transcripts (format: `filename.wav|exact transcript text`, one per line)
5. Set **Name** for your LoRA
6. Defaults are tuned for **10-20 clips**: `r=16`, `alpha=16`, `steps=300`, `lr=0.0001`
7. Click **🎓 Start training** — wait 5-15 min on RTX 4090
8. Checkpoint appears in `lora/<name>/step_XXXX/` and in dropdown in all tabs
9. Select your LoRA in **Advanced settings → 🧬 LoRA** to activate (first activation takes ~30-60 sec to reload model with LoRA structure; subsequent switches are instant hot-swap)

### Recommended LoRA settings by dataset size

| Clips | Steps | r / α | LR |
|---|---|---|---|
| **10-20** (default) | **300** | 16/16 | 0.0001 |
| 50-100 | 500-1000 | 32/32 | 0.0001 |
| 200+ | 1000-2000 | 32/32 or 64/64 | 0.00005-0.0001 |
| 1000+ | 2000-5000 | 64/64 | 0.00005 |

## Architecture

```
VoxCPM2_portable/
├── app.py              # Gradio UI (4 tabs: TTS / Voice Design / Cloning / LoRA)
├── install.bat         # GPU selector + installer
├── run.bat             # Launcher with env isolation
├── update.bat          # Updater
├── requirements.txt    # Python dependencies
├── training/
│   ├── scripts/        # Official OpenBMB train & inference scripts (bundled)
│   └── conf/           # YAML config templates
├── python/             # Portable Python 3.12 (created by install.bat)
├── models/             # HuggingFace cache (VoxCPM2 ~4-5 GB, ZipEnhancer, SenseVoice)
├── voices/             # Voice pack (bundled default ~100 voices + user downloads)
├── lora/               # Trained LoRA checkpoints (lora/<name>/step_XXXX/)
├── train_data/         # User LoRA datasets (audio + transcripts)
├── ffmpeg/             # Portable FFmpeg (for MP3/OGG encoding)
├── output/             # Generated audio files with timestamps
├── cache/ / temp/      # General cache / tempdir
```

## Updating

```
update.bat
```

## Links

- [OpenBMB / VoxCPM](https://github.com/OpenBMB/VoxCPM) — original project
- [VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2) — weights
- [Demo page with audio samples](https://openbmb.github.io/voxcpm2-demopage/)
- [Fine-tuning Guide](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)

## Other Portable Neural Networks

| Project | Description |
|---------|-------------|
| [ACE-Step Studio](https://github.com/timoncool/ACE-Step-Studio) | Local AI music generation studio |
| [Foundation Music Lab](https://github.com/timoncool/Foundation-Music-Lab) | Music generation + timeline editor |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Speech recognition (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Audio quality enhancement |
| [Qwen3-TTS](https://github.com/timoncool/Qwen3-TTS_portable_rus) | Text-to-speech by Qwen |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Image captioning |
| [VideoSOS](https://github.com/timoncool/videosos) | AI video production |
| [RC Stable Audio Tools](https://github.com/timoncool/RC-stable-audio-tools-portable) | Music and audio generation |

## Authors

- **Nerual Dreming** — [Telegram](https://t.me/nerual_dreming) | [neuro-cartel.com](https://neuro-cartel.com) | [ArtGeneration.me](https://artgeneration.me)
- **Neiro-Soft** — [Telegram](https://t.me/neuroport) | portable neural network builds

## Acknowledgments

- **[OpenBMB / VoxCPM team](https://github.com/OpenBMB/VoxCPM)** — open source VoxCPM2 model
- **[Slait/russia_voices](https://huggingface.co/datasets/Slait/russia_voices)** — 743 Russian voice presets
- **[AIQuest Academy](https://github.com/TeamAIQ/Colab-notebooks)** — Colab UI reference
- **[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel)** — Windows Flash Attention 2 wheels
- **[Gradio](https://gradio.app/)** — UI framework
- **[FFmpeg](https://ffmpeg.org/)** — audio encoding

## Support This Project

I build software and do research in AI. Most of what I create is free and open source. Your donations allow me to keep creating and exploring without worrying about where the next meal comes from =)

**[All donation methods](https://dalink.to/nerual_dreming)** | **[dalink.to/nerual_dreming](https://dalink.to/nerual_dreming)** | **[boosty.to/neuro_art](https://boosty.to/neuro_art)**

- **BTC:** `1E7dHL22RpyhJGVpcvKdbyZgksSYkYeEBC`
- **ETH (ERC20):** `0xb5db65adf478983186d4897ba92fe2c25c594a0c`
- **USDT (TRC20):** `TQST9Lp2TjK6FiVkn4fwfGUee7NmkxEE7C`

---

## Star History

<a href="https://www.star-history.com/?repos=timoncool%2FVoxCPM2_portable&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=timoncool/VoxCPM2_portable&type=date&legend=top-left" />
 </picture>
</a>
