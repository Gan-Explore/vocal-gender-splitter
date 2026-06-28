# Vocal Gender Splitter — Pinokio App

Separates **male** and **female** vocals from audio files using:
- **Demucs** (htdemucs model) — strips music, isolates clean vocals
- **Pitch-based gender classification** — segments vocals and labels each by fundamental frequency (F0)

---

## Requirements

- Windows 10/11
- [Pinokio](https://pinokio.computer) installed
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with 4+ GB VRAM recommended (CPU fallback supported but slow)

---

## Install

1. In Pinokio, click **Discover** → search for this app, or use **Import from folder**
2. Click **Install** — this will:
   - Create a Python virtual environment
   - Install PyTorch (CUDA 11.8 build), Demucs, Gradio, librosa, SpeechBrain
   - First run will also auto-download the Demucs `htdemucs` model (~1 GB)

---

## Usage

1. Click **Launch App** in Pinokio
2. Upload any audio file (MP3, WAV, FLAC, OGG)
3. Optionally tick **Skip vocal isolation** if you already have a clean vocals-only file
4. Click **Separate vocals**
5. Download the **Male vocals** and **Female vocals** output files

---

## How it works

```
Input audio
    │
    ▼
[Demucs htdemucs]  ← skipped if "Skip vocal isolation" is checked
    │  strips drums, bass, guitar, piano
    ▼
Clean vocals stem
    │
    ▼
[Pitch estimator]  ← librosa pyin(), 2-second windowed segments
    │  F0 < 185 Hz → male
    │  F0 ≥ 185 Hz → female
    ▼
Masked output files
  male_vocals.wav
  female_vocals.wav
```

---

## Limitations

- Pitch-based separation works best on **duets or solo** recordings
- Voices with similar pitch ranges (e.g. tenor + mezzo-soprano) may bleed
- Simultaneous male+female singing in the same moment cannot be fully separated by any method
- CPU processing is significantly slower (~5–10× longer than GPU)

---

## Tips for better results

1. Pre-separate music with **UVR5** (also available on Pinokio) using the `BS-Roformer` model
2. Feed the clean vocal stem into this app with **Skip vocal isolation** checked
3. For podcasts or speech recordings, results are generally excellent

---

## Troubleshooting

**"CUDA out of memory"** — restart the app; Demucs will retry on CPU  
**Slow processing** — expected without a GPU; a 3-minute song takes ~5 min on CPU  
**Poor separation quality** — the recording likely has both voices singing simultaneously; no tool can cleanly separate them
