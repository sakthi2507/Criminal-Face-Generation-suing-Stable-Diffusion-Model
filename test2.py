import torch
import clip
from torch import nn
from PIL import Image
import numpy as np
import dnnlib
import legacy
import os

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "ffhq.pkl"  # Path to downloaded model
text_input = "A rugged-looking man with tanned skin, a square jawline, a thick mustache, and a deep scar across his right eyebrow."  # Example text input

# --- Load StyleGAN2-ADA Pre-Trained Model ---
def load_stylegan2_model(model_path):
    print("Loading StyleGAN2 model...")
    with dnnlib.util.open_url(model_path) as f:
        G, D, G_ema = legacy.load_network_pkl(f)
    return G_ema

# --- Load CLIP Model ---
def load_clip_model():
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, preprocess

# --- Generate Image from Latent Vector ---
def generate_image_from_latent(G_ema, latent_vector, noise_mode='const'):
    # Generate a fake image from the latent vector
    images = G_ema.synthesis(latent_vector, noise_mode=noise_mode)
    images = (images + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]
    return images

# --- Text to Latent Space Using CLIP ---
def text_to_latent_space(text_input, clip_model, preprocess):
    # Preprocess the text input for CLIP model
    text_input = clip.tokenize([text_input]).to(device)
    
    # Get the CLIP text embeddings
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_input)

    # Normalize the text embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    # Generate a random latent vector (in this case, we'll use a simple random vector for now)
    latent_vector = torch.randn(1, 512).to(device)  # 512 is the latent size for FFHQ model
    latent_vector = latent_vector * 0.1  # Adjusting the scale of the latent vector

    return latent_vector, text_embeddings

# --- Main Code to Generate Face Based on Text ---
def generate_face_from_text(text_input, model_path):
    # Load StyleGAN2 model
    G_ema = load_stylegan2_model(model_path)
    
    # Load CLIP model
    clip_model, preprocess = load_clip_model()

    # Get latent vector from text input
    latent_vector, text_embeddings = text_to_latent_space(text_input, clip_model, preprocess)

    # Generate image from latent vector
    generated_image = generate_image_from_latent(G_ema, latent_vector)

    # Convert tensor to PIL image
    generated_image_pil = Image.fromarray(np.uint8(generated_image[0].cpu().numpy() * 255))

    # Show generated image
    generated_image_pil.show()

# --- Run the Function ---
generate_face_from_text(text_input, model_path)
