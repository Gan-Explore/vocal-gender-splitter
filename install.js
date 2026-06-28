module.exports = {
  title: "Install Vocal Gender Splitter",
  description: "Sets up Python environment and downloads AI models",
  icon: "fa-solid fa-download",
  run: [
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/pinokiocomputer/pinokio.git . || true",
        path: "app",
        done: ".",
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python -m venv venv",
        path: "app",
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "venv\\Scripts\\activate.bat",
          "python -m pip install --upgrade pip",
          "python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118",
          "python -m pip install demucs librosa soundfile gradio numpy scipy",
          "python -m pip install transformers speechbrain",
        ].join(" && "),
        path: "app",
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click <b>Launch App</b> to start.",
        type: "success",
      }
    }
  ]
}
