import gradio as gr
import torch
import torchaudio
import numpy as np
import os
import tempfile
from pathlib import Path


# ── Demucs full stem separation ─────────────────────────────────────────────
def separate_stems(audio_path: str, progress=gr.Progress()):
    """Step 1: Demucs splits into drums / bass / other / vocals."""
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

    # htdemucs source order: drums, bass, other, vocals
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


# ── Pitch-based gender split ─────────────────────────────────────────────────
def classify_and_split(vocals_path: str, sr_hint: int, progress=gr.Progress()):
    """Step 2: Split the vocals stem into male / female by F0."""
    waveform, sr = torchaudio.load(vocals_path)
    if waveform.shape[0] > 1:
        mono = waveform.mean(0, keepdim=True)
    else:
        mono = waveform

    total_samples = mono.shape[1]
    segment_len   = int(sr * 2.0)
    hop           = int(sr * 1.0)

    male_mask   = torch.zeros(total_samples)
    female_mask = torch.zeros(total_samples)

    progress(0.60, desc="Classifying vocal segments by gender...")
    labels = []
    for start in range(0, total_samples - segment_len, hop):
        seg = mono[0, start:start + segment_len].numpy()
        f0  = estimate_pitch(seg, sr)
        if f0 is None:
            labels.append("unknown")
        elif f0 < 185:
            labels.append("male")
        else:
            labels.append("female")

    fade_samples = int(sr * 0.05)
    for i, label in enumerate(labels):
        start = i * hop
        end   = min(start + segment_len, total_samples)
        win   = torch.ones(end - start)
        fi    = min(fade_samples, end - start)
        win[:fi]  *= torch.linspace(0, 1, fi)
        win[-fi:] *= torch.linspace(1, 0, fi)
        if label == "male":
            male_mask[start:end]   = torch.max(male_mask[start:end], win)
        elif label == "female":
            female_mask[start:end] = torch.max(female_mask[start:end], win)

    progress(0.85, desc="Writing male / female files...")
    stereo = waveform if waveform.shape[0] == 2 else waveform.repeat(2, 1)

    out_dir     = Path(tempfile.mkdtemp())
    male_path   = str(out_dir / "male_vocals.wav")
    female_path = str(out_dir / "female_vocals.wav")
    torchaudio.save(male_path,   stereo * male_mask.unsqueeze(0),   sr)
    torchaudio.save(female_path, stereo * female_mask.unsqueeze(0), sr)

    return male_path, female_path, labels


def estimate_pitch(signal: np.ndarray, sr: int):
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            signal,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
        )
        voiced = f0[voiced_flag]
        return float(np.median(voiced)) if len(voiced) > 0 else None
    except Exception:
        return None


# ── Full pipeline ────────────────────────────────────────────────────────────
def run_pipeline(audio_file, skip_demucs, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")
    try:
        device_str = (
            f"GPU: {torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available() else "CPU (no CUDA GPU detected)"
        )

        if skip_demucs:
            # User supplies a clean vocal stem — skip straight to gender split
            vocals_path = audio_file
            _, sr = torchaudio.load(audio_file)
            drums_path = bass_path = other_path = None
        else:
            stem_paths, sr = separate_stems(audio_file, progress)
            vocals_path = stem_paths["vocals"]
            drums_path  = stem_paths["drums"]
            bass_path   = stem_paths["bass"]
            other_path  = stem_paths["other"]

        male_path, female_path, labels = classify_and_split(vocals_path, sr, progress)

        male_count   = labels.count("male")
        female_count = labels.count("female")
        unknown      = labels.count("unknown")
        total        = len(labels)

        summary = (
            f"{'Stems separated + ' if not skip_demucs else ''}Gender split complete\n"
            f"─────────────────────────────\n"
            f"Segments analysed : {total} ({total * 2}s)\n"
            f"Male              : {male_count} ({100 * male_count // max(total, 1)}%)\n"
            f"Female            : {female_count} ({100 * female_count // max(total, 1)}%)\n"
            f"Unvoiced / silent : {unknown}\n"
            f"─────────────────────────────\n"
            f"{device_str}"
        )

        progress(1.0, desc="Done!")
        return drums_path, bass_path, other_path, vocals_path, male_path, female_path, summary

    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


# ── Gradio UI ────────────────────────────────────────────────────────────────
# with gr.Blocks(
#    title="Vocal Gender Splitter",
#    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
#    css="""
with gr.Blocks(title="Vocal Gender Splitter") as demo:
        .gradio-container { max-width: 960px !important; margin: auto; }
        #title    { text-align: center; margin-bottom: 0.25rem; }
        #subtitle { text-align: center; color: #64748b; margin-bottom: 1.5rem; font-size: 0.9rem; }
        footer    { display: none !important; }
        .section-label { font-weight: 600; font-size: 0.85rem; color: #475569;
                          text-transform: uppercase; letter-spacing: 0.05em;
                          margin: 1rem 0 0.5rem; }
    """,
) as demo:

    gr.Markdown("# Vocal Gender Splitter", elem_id="title")
    gr.Markdown(
        "Full pipeline: **stem separation** (drums / bass / other / vocals) → **gender split** (male / female)",
        elem_id="subtitle",
    )

    # ── Input ──
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

    # ── Stems output ──
    gr.Markdown("### Separated stems", elem_classes=["section-label"])
    with gr.Row():
        drums_out  = gr.Audio(label="Drums",  type="filepath", interactive=False)
        bass_out   = gr.Audio(label="Bass",   type="filepath", interactive=False)
        other_out  = gr.Audio(label="Other (guitars / keys / etc.)", type="filepath", interactive=False)
        vocals_out = gr.Audio(label="Vocals (full)", type="filepath", interactive=False)

    # ── Gender output ──
    gr.Markdown("### Gender split", elem_classes=["section-label"])
    with gr.Row():
        male_out   = gr.Audio(label="Male vocals",   type="filepath", interactive=False)
        female_out = gr.Audio(label="Female vocals", type="filepath", interactive=False)

    gr.Markdown(
        "> **Tip:** On a clean duet recording results are excellent. "
        "For dense mixes with overlapping voices, some bleed between male and female tracks is expected.",
    )

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
