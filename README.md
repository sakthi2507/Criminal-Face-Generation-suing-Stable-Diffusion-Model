Download the Dataset from : https://www.kaggle.com/datasets/ashwingupta3012/human-faces 
# 🔍 Criminal Face Generation using Stable Diffusion

This project explores the use of **Stable Diffusion**, a powerful text-to-image deep learning model, to generate **visual face representations of criminals** based on descriptive textual inputs. The goal is to assist law enforcement or forensic artists by creating photorealistic sketches from eyewitness descriptions.

## 🚀 Project Overview

- 🔎 **Objective**: To generate synthetic face images of suspects using descriptive criminal profiles.
- 🧠 **Model Used**: [Stable Diffusion v1.5/v2.1](https://github.com/CompVis/stable-diffusion) (by StabilityAI)
- 🛠️ **Tech Stack**: Python, Hugging Face Transformers, diffusers, Gradio (optional UI)

---

## 🧰 Features

- 📝 Input descriptive text (e.g., "male, 30s, beard, wearing a cap, angry expression")
- 🎨 Generates realistic face images using Stable Diffusion
- 📦 Saves output locally for further analysis
- 🔄 Optionally regenerate faces for the same prompt
- 🖼️ (Optional) GUI interface using Gradio

---

## 📁 Folder Structure

Criminal-Face-Generation/
│
├── app.py # Main app script
├── model_utils.py # Model loading and inference
├── requirements.txt # Dependencies
├── outputs/ # Generated face images
├── prompts/ # Sample criminal descriptions
└── README.md # Project description

 Dependencies
text
Copy
Edit
torch
diffusers
transformers
accelerate
Pillow
gradio (optional)
