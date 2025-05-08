# Bird Classification and Visualization

This project implements a bird species classification system using the Swin Transformer architecture, enhanced with built-in attention-based visualizations for interpretability.

## Overview

- **Model Architecture:** Swin-Large (patch4-window12-384), fine-tuned on the Birds-1011 dataset using ImageNet-22K pre-trained weights.  
- **Accuracy:** Achieves over 89% top-1 accuracy on the validation set (1,011 species).  
- **Explainability:** Utilizes intrinsic attention visualization from SwinSelfAttention layers across multiple depths.  
- **Web Application:** Flask-based backend with a responsive frontend for image upload, prediction, and visualization.  
- **Audio Integration:** Includes bird calls for identified species.  

## System Pipeline

1. **Image Upload:** User provides an image of a bird.  
2. **Preprocessing:** Input is resized and normalized.  
3. **Prediction:** Swin Transformer model infers the bird species.  
4. **Visualization:** Self-attention weights are visualized as heatmaps.  
5. **Output:** Displays predicted species, confidence score, attention map, and optional audio.  

## Key Implementation Details

- **Attention Visualization:** Extracts and processes attention maps from shifted window mechanisms.  
- **Multi-Layer Support:** Enables visualization from early, middle, and late transformer layers.  
- **Web Integration:** Combines PyTorch inference with real-time frontend updates via Flask and JavaScript.
