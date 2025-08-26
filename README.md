# 🖼️ Image Generation with Pre-trained Models


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KritikMahesh/PRODIGY_GA_02/blob/main/Image_Generation_with_Pre_trained_Models.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![HuggingFace Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-blueviolet.svg)](https://huggingface.co/docs/diffusers/index)

> **Generating High-Quality Images from Text Prompts Using State-of-the-Art Pre-trained Models**

This repository contains a Jupyter Notebook implementation demonstrating the use of cutting-edge **pre-trained generative models** for image generation. The notebook leverages models from Hugging Face's Diffusers library, such as Stable Diffusion, to generate high-quality images from natural language prompts.

## 📋 Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Results](#-results)
- [How It Works](#-how-it-works)
- [Customization](#-customization)

## 🎯 About

This project showcases how to utilize **pre-trained diffusion models** for image generation based on text prompts. Using the **Stable Diffusion** model and associated libraries, it generates diverse images by conditioning on user inputs, demonstrating the power of generative AI without needing extensive training resources.

## ✨ Features

- 🤖 **Pre-trained Models** — Uses cutting-edge generative models like Stable Diffusion from Hugging Face  
- 📝 **Text-to-Image** — Generate images directly from descriptive natural language prompts  
- 🎨 **High Quality** — Produces photorealistic and artistic images at 512x512 resolution  
- ⚙️ **Easy Setup** — Minimal installation and easy-to-run notebook format  
- 📊 **Visualization** — Display generated images inline for easy comparison  
- 💡 **Extensible** — Notebook setup allows experimenting with different prompts and models  

## 🛠️ Installation

### Option 1: Google Colab (Recommended)
1. Click the **Open In Colab** badge above  
2. Run all cells sequentially — all dependencies install automatically  

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/KritikMahesh/PRODIGY_GA_04.git
cd PRODIGY_GA_04

# Install dependencies
pip install torch torchvision diffusers transformers matplotlib

# Run Jupyter Notebook
jupyter notebook Kaggle_image-generation-with-pre-trained-models.ipynb
```

## 🚀 Usage

### Quick Start
1. **Load the notebook** and run all cells sequentially
2. **Training** - Enter your own text prompts to generate images
3. **See results** - Visualize and save generated images for further use
4. **Experiment** - Modify model parameters for custom effects


## 🏗️ Model Details
- **Model Used**: Stable Diffusion (latent diffusion model for text-to-image synthesis)
- **Framework**: PyTorch and Hugging Face Diffusers
- **Resolution**: 512x512 pixels
- **Pipeline**: Text prompt tokenization → Latent diffusion → Image decoding
- **Sampling Steps**: Adjustable number of diffusion steps for quality-speed tradeoff

## 📈 Results
- The model generates images that reflect the input prompt creatively and with fine detail. Examples include landscapes, objects, and imaginative scenes generated from simple text descriptions 

## 🔬 How It Works

- **Text Encoding**: The input prompt is tokenized and encoded using a pre-trained text encoder.

- **Latent Diffusion**: The encoded text conditions the denoising diffusion process to generate latent representations.

- **Decoding**: The latent representation is decoded back into a high-resolution image.

- **Sampling**: The process is iterative, gradually refining noise into a coherent image.

## 🔧 Customization

- Change prompts in the notebook to generate different images
- Adjust number of inference steps to balance quality and speed
- Experiment with different pre-trained models from Hugging Face
- Use the saved images for your projects, presentations, or creative works
