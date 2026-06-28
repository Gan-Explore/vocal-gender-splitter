import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import ssl
import certifi
import tempfile
from pathlib import Path

# Fix SSL cert issues with Pinokio's bundled miniconda Python
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context


# ── Demucs full stem separation ─────────────────────────────────────────────
def separate_stems(audio_path: str, progress=gr.Progress()):
    progress(0.05, desc="Loading audio...")
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    waveform, sr = torchaudio.load(audio_path)

    if sr != 44100:
        waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
        sr = 44100

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    progress(0.15, desc="Loading Demucs htdemucs model...")
    model = get_model("htdemucs")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    progress(0.30, desc="Separating stems (drums / bass / other / vocals)...")
    with torch.no_grad():
        sources = apply_model(model, waveform.unsqueeze(0).to(device), device=device)[0]

    stems = {
        "drums":  sources[0].cpu(),
        "bass":   sources[1].cpu(),
        "other":  sources[2].cpu(),
        "vocals": sources[3].cpu(),
    }

    out_dir = Path(tempfile.mkdtemp())
    paths = {}
    for name, tensor in stems.items():
        p = str(out_dir / f"{name}.wav")
        torchaudio.save(p, tensor, sr)
        paths[name] = p

    progress(0.45, desc="Stems saved.")
    return paths, sr


# ── Adaptive pitch-based gender split ───────────────────────────────────────
def classify_and_split(vocals_path: str, sr_hint: int, progress=gr.Progress()):
    import librosa
    from scipy.signal import medfilt

    waveform, sr = torchaudio.load(vocals_path)
    mono = waveform.mean(0).numpy() if waveform.shape[0] > 1 else waveform[0].numpy()

    total_samples = len(mono)
    segment_len   = int(sr * 1.5)   # 1.5-second windows
    hop           = int(sr * 0.75)  # 50% overlap

    progress(0.55, desc="Estimating pitch for every segment...")

    # Pass 1 — collect F0 for every voiced segment
    f0_list    = []   # median F0 per segment (None if unvoiced)
    seg_starts = list(range(0, total_samples - segment_len, hop))

    for start in seg_starts:
        seg = mono[start:start + segment_len]
        f0, voiced_flag, _ = librosa.pyin(
            seg,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced = f0[voiced_flag & ~np.isnan(f0)]
        f0_list.append(float(np.median(voiced)) if len(voiced) >= 3 else None)

    # Pass 2 — find adaptive threshold via K-means on voiced segments
    voiced_f0s = np.array([v for v in f0_list if v is not None])
    threshold  = 185.0   # fallback if not enough data

    if len(voiced_f0s) >= 4:
        # Simple 1-D K-means with 2 clusters (lower = male, higher = female)
        from scipy.cluster.vq import kmeans
        log_f0 = np.log(voiced_f0s)   # cluster in log-frequency (perceptually uniform)
        init_centers = np.array([np.log(130.0), np.log(230.0)])
        for _ in range(20):
            dists  = np.abs(log_f0[:, None] - init_centers[None, :])
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([log_f0[assign == k].mean() if (assign == k).any()
                               else init_centers[k] for k in range(2)])
            if np.allclose(new_c, init_centers, atol=1e-4):
                break
            init_centers = new_c
        centers   = np.sort(np.exp(init_centers))   # Hz, low → high
        threshold = float(np.sqrt(centers[0] * centers[1]))  # geometric midpoint
        spread    = centers[1] - centers[0]
        label_info = (
            "Adaptive threshold: " + str(int(threshold)) + " Hz  "
            "(male centre " + str(int(centers[0])) + " Hz, "
            "female centre " + str(int(centers[1])) + " Hz, "
            "spread " + str(int(spread)) + " Hz)"
        )
    else:
        label_info = "Fallback threshold: 185 Hz (not enough voiced segments to cluster)"

    progress(0.72, desc="Labelling segments...")

    # Pass 3 — label each segment using adaptive threshold
    raw_labels = []
    for f0 in f0_list:
        if f0 is None:
            raw_labels.append("unknown")
        elif f0 < threshold:
            raw_labels.append("male")
        else:
            raw_labels.append("female")

    # Smooth labels: majority vote over a 3-segment window to reduce flicker
    labels = list(raw_labels)
    for i in range(1, len(raw_labels) - 1):
        window = [raw_labels[i - 1], raw_labels[i], raw_labels[i + 1]]
        voiced_window = [l for l in window if l != "unknown"]
        if len(voiced_window) == 2 and voiced_window[0] == voiced_window[1]:
            labels[i] = voiced_window[0]

    # Pass 4 — build smooth masks with crossfade
    male_mask   = np.zeros(total_samples, dtype=np.float32)
    female_mask = np.zeros(total_samples, dtype=np.float32)
    fade        = int(sr * 0.06)

    for i, (start, label) in enumerate(zip(seg_starts, labels)):
        end = min(start + segment_len, total_samples)
        win = np.ones(end - start, dtype=np.float32)
        fi  = min(fade, end - start)
        win[:fi]  *= np.linspace(0, 1, fi)
        win[-fi:] *= np.linspace(1, 0, fi)
        if label == "male":
            male_mask[start:end]   = np.maximum(male_mask[start:end], win)
        elif label == "female":
            female_mask[start:end] = np.maximum(female_mask[start:end], win)

    progress(0.88, desc="Writing male / female files...")
    stereo = waveform if waveform.shape[0] == 2 else waveform.repeat(2, 1)
    male_t   = torch.from_numpy(male_mask).unsqueeze(0)
    female_t = torch.from_numpy(female_mask).unsqueeze(0)

    out_dir     = Path(tempfile.mkdtemp())
    male_path   = str(out_dir / "male_vocals.wav")
    female_path = str(out_dir / "female_vocals.wav")
    torchaudio.save(male_path,   stereo * male_t,   sr)
    torchaudio.save(female_path, stereo * female_t, sr)

    return male_path, female_path, labels, label_info


# ── Full pipeline ────────────────────────────────────────────────────────────
def run_pipeline(audio_file, skip_demucs, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")
    try:
        device_str = (
            "GPU: " + torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "CPU (no CUDA GPU detected)"
        )

        if skip_demucs:
            vocals_path = audio_file
            _, sr = torchaudio.load(audio_file)
            drums_path = bass_path = other_path = None
        else:
            stem_paths, sr = separate_stems(audio_file, progress)
            vocals_path = stem_paths["vocals"]
            drums_path  = stem_paths["drums"]
            bass_path   = stem_paths["bass"]
            other_path  = stem_paths["other"]

        male_path, female_path, labels, label_info = classify_and_split(vocals_path, sr, progress)

        male_count   = labels.count("male")
        female_count = labels.count("female")
        unknown      = labels.count("unknown")
        total        = len(labels)

        summary = (
            ("Stems separated + " if not skip_demucs else "") + "Gender split complete\n"
            + "-" * 33 + "\n"
            + "Segments analysed : " + str(total) + "\n"
            + "Male              : " + str(male_count) + " (" + str(100 * male_count // max(total, 1)) + "%)\n"
            + "Female            : " + str(female_count) + " (" + str(100 * female_count // max(total, 1)) + "%)\n"
            + "Unvoiced / silent : " + str(unknown) + "\n"
            + "-" * 33 + "\n"
            + label_info + "\n"
            + "-" * 33 + "\n"
            + device_str
        )

        progress(1.0, desc="Done!")
        return drums_path, bass_path, other_path, vocals_path, male_path, female_path, summary

    except Exception as e:
        raise gr.Error("Processing failed: " + str(e))


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Vocal Gender Splitter") as demo:

    gr.Markdown("# Vocal Gender Splitter")
    gr.Markdown("Full pipeline: **stem separation** (drums / bass / other / vocals) then **gender split** (male / female)")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Upload audio", type="filepath", sources=["upload"])
            skip_demucs = gr.Checkbox(
                label="Skip stem separation — file is already a vocals-only stem",
                value=False,
                info="Tick this if you pre-processed with UVR5 or another tool",
            )
            run_btn = gr.Button("Run pipeline", variant="primary", size="lg")

        with gr.Column(scale=1):
            summary_out = gr.Textbox(label="Summary", lines=9, interactive=False)

    gr.Markdown("### Separated stems")
    with gr.Row():
        drums_out  = gr.Audio(label="Drums",  type="filepath", interactive=False)
        bass_out   = gr.Audio(label="Bass",   type="filepath", interactive=False)
        other_out  = gr.Audio(label="Other (guitars / keys)", type="filepath", interactive=False)
        vocals_out = gr.Audio(label="Vocals (full)", type="filepath", interactive=False)

    gr.Markdown("### Gender split")
    with gr.Row():
        male_out   = gr.Audio(label="Male vocals",   type="filepath", interactive=False)
        female_out = gr.Audio(label="Female vocals", type="filepath", interactive=False)

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, skip_demucs],
        outputs=[drums_out, bass_out, other_out, vocals_out, male_out, female_out, summary_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
