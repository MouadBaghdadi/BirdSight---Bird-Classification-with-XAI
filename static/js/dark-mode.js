document.addEventListener("DOMContentLoaded", () => {
    // Add dark mode toggle button to the header
    const header = document.querySelector(".app-header")
    const themeToggle = document.createElement("button")
    themeToggle.className = "theme-toggle"
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>'
    themeToggle.setAttribute("aria-label", "Toggle dark mode")
    header.appendChild(themeToggle)
  
    // Check for saved theme preference or respect OS preference
    const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)")
    const savedTheme = localStorage.getItem("theme")
  
    if (savedTheme === "dark" || (!savedTheme && prefersDarkScheme.matches)) {
      document.body.classList.add("dark-mode")
      themeToggle.innerHTML = '<i class="fas fa-sun"></i>'
    }
  
    // Toggle dark mode
    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode")
  
      if (document.body.classList.contains("dark-mode")) {
        localStorage.setItem("theme", "dark")
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>'
      } else {
        localStorage.setItem("theme", "light")
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>'
      }
    })
  })  