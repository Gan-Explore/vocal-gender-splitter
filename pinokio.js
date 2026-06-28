module.exports = {
  version: "2.0",
  title: "Vocal Gender Splitter",
  description: "Separate male and female vocals using AI (Demucs + gender classifier)",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = await kernel.exists(__dirname, "app", "venv")
    if (installed) {
      return [
        {
          text: "Launch App",
          icon: "fa-solid fa-play",
          href: "start.js",
        },
        {
          text: "Update",
          icon: "fa-solid fa-rotate",
          href: "update.js",
        },
      ]
    } else {
      return [
        {
          text: "Install",
          icon: "fa-solid fa-download",
          href: "install.js",
        },
      ]
    }
  }
}
