module.exports = {
  title: "Launch Vocal Gender Splitter",
  icon: "fa-solid fa-play",
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "venv\\Scripts\\activate.bat",
          "python app.py",
        ].join(" && "),
        path: "app",
        env: {
          GRADIO_SERVER_PORT: "7860",
        },
      }
    },
    {
      method: "browser.open",
      params: {
        url: "http://localhost:7860"
      }
    }
  ]
}
