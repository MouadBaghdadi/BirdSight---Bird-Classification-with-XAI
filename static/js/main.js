document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const imageUpload = document.getElementById("image-upload")
    const previewImg = document.getElementById("preview-img")
    const uploadPlaceholder = document.getElementById("upload-placeholder")
    const predictBtn = document.getElementById("predict-btn")
    const clearBtn = document.getElementById("clear-btn")
    const errorMessage = document.getElementById("error-message")
    const resultsSection = document.getElementById("results-section")
    const speciesName = document.getElementById("species-name")
    const lifeStageContainer = document.getElementById("life-stage-container")
    const lifeStage = document.getElementById("life-stage")
    const confidenceBar = document.getElementById("confidence-bar")
    const confidenceValue = document.getElementById("confidence-value")
    const birdAudio = document.getElementById("bird-audio")
    const audioContainer = document.getElementById("audio-container")
    const heatmapContainer = document.getElementById("heatmap-container")
    const heatmapImage = document.getElementById("heatmap-image")
    const loadingOverlay = document.getElementById("loading-overlay")
  
    // Event Listeners
    imageUpload.addEventListener("change", handleImageUpload)
    predictBtn.addEventListener("click", predictBirdSpecies)
    clearBtn.addEventListener("click", clearImage)
  
    // Handle image upload
    function handleImageUpload(e) {
      const file = e.target.files[0]
      if (!file) return
  
      if (!file.type.match("image/jpeg") && !file.type.match("image/png")) {
        showError("Please upload a JPEG or PNG image.")
        return
      }
  
      const reader = new FileReader()
      reader.onload = (event) => {
        previewImg.src = event.target.result
        previewImg.style.display = "block"
        uploadPlaceholder.style.display = "none"
        predictBtn.disabled = false
        clearBtn.style.display = "block"
        resultsSection.style.display = "none"
        hideError()
      }
      reader.readAsDataURL(file)
    }
  
    // Predict bird species
    async function predictBirdSpecies() {
      if (!previewImg.src) {
        showError("Please upload an image first.")
        return
      }
  
      try {
        showLoading(true)
  
        // Create form data for the image
        const formData = new FormData()
        const file = imageUpload.files[0]
        formData.append("image", file)
  
        // Send prediction request
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        })
  
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || "Failed to process the image")
        }
  
        const result = await response.json()
  
        // Display results
        displayResults(result)
  
        // Generate and display heatmap
        generateHeatmap(formData)
      } catch (error) {
        console.error("Prediction error:", error)
        showError(error.message || "Failed to process the image. Please try again.")
      } finally {
        showLoading(false)
      }
    }
  
    // Generate heatmap
    async function generateHeatmap(formData) {
      try {
        const response = await fetch("/heatmap", {
          method: "POST",
          body: formData,
        });
    
        if (!response.ok) {
          console.error(`Failed to generate heatmap: ${response.status}`);
          return;
        }
    
        const result = await response.json();
    
        // Check for error in response
        if (result.error) {
          console.error("Heatmap error:", result.error);
          return;
        }
    
        // Display heatmap
        heatmapImage.src = result.heatmap_url + "?t=" + new Date().getTime();
        heatmapContainer.style.display = "block";
      } catch (error) {
        console.error("Failed to generate heatmap:", error);
      }
    }  
    // Display prediction results
    function displayResults(result) {
      // Display species name
      speciesName.textContent = result.species_name
  
      // Display life stage if available
      if (result.life_stage) {
        lifeStage.textContent = result.life_stage
        lifeStageContainer.style.display = "block"
      } else {
        lifeStageContainer.style.display = "none"
      }
  
      // Display confidence
      const confidencePercent = result.confidence.toFixed(1)
      confidenceBar.style.width = `${confidencePercent}%`
      confidenceValue.textContent = `${confidencePercent}%`
  
      // Set confidence bar color based on confidence level
      if (result.confidence >= 80) {
        confidenceBar.style.backgroundColor = "var(--success-color)"
      } else if (result.confidence >= 50) {
        confidenceBar.style.backgroundColor = "var(--secondary-color)"
      } else {
        confidenceBar.style.backgroundColor = "var(--error-color)"
      }
  
      // Display audio if available
      if (result.audio_path) {
        birdAudio.src = result.audio_path
        audioContainer.style.display = "block"
      } else {
        audioContainer.style.display = "none"
      }
  
      // Show results section
      resultsSection.style.display = "block"
    }
  
    // Clear image and results
    function clearImage() {
      imageUpload.value = ""
      previewImg.src = ""
      previewImg.style.display = "none"
      uploadPlaceholder.style.display = "flex"
      predictBtn.disabled = true
      clearBtn.style.display = "none"
      resultsSection.style.display = "none"
      hideError()
    }
  
    // Show error message
    function showError(message) {
      errorMessage.textContent = message
      errorMessage.style.display = "block"
    }
  
    // Hide error message
    function hideError() {
      errorMessage.style.display = "none"
    }
  
    // Show/hide loading overlay
    function showLoading(show) {
      loadingOverlay.style.display = show ? "flex" : "none"
    }
  
    // Add drag and drop functionality
    const imagePreview = document.getElementById("image-preview")
    ;["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      imagePreview.addEventListener(eventName, preventDefaults, false)
    })
  
    function preventDefaults(e) {
      e.preventDefault()
      e.stopPropagation()
    }
    ;["dragenter", "dragover"].forEach((eventName) => {
      imagePreview.addEventListener(eventName, highlight, false)
    })
    ;["dragleave", "drop"].forEach((eventName) => {
      imagePreview.addEventListener(eventName, unhighlight, false)
    })
  
    function highlight() {
      imagePreview.classList.add("highlight")
    }
  
    function unhighlight() {
      imagePreview.classList.remove("highlight")
    }
  
    imagePreview.addEventListener("drop", handleDrop, false)
  
    function handleDrop(e) {
      const dt = e.dataTransfer
      const files = dt.files
  
      if (files.length) {
        imageUpload.files = files
        handleImageUpload({ target: { files: files } })
      }
    }
  })
  