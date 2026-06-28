module.exports = {
  title: "Update Vocal Gender Splitter",
  icon: "fa-solid fa-rotate",
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "venv\\Scripts\\activate.bat",
          "python -m pip install --upgrade demucs librosa gradio transformers speechbrain",
        ].join(" && "),
        path: "app",
      }
    },
    {
      method: "notify",
      params: {
        html: "Update complete!",
        type: "success",
      }
    }
  ]
}
