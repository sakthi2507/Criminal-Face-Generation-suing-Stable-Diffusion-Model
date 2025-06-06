import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# --- CONFIG ---
image_dir = "human_faces"  # Your image directory
batch_size = 2
epochs = 2  # Shortened for testing
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- DATASET ---
class SuspectFaceDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# --- CLIP + ViT + Decoder Architecture ---
class ViTWithDecoder(nn.Module):
    def __init__(self, prompt_dim=512):
        super(ViTWithDecoder, self).__init__()

        # Load CLIP model (non-OpenAI, using clip-vit-base-patch32)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load pretrained ViT for feature extraction
        from torchvision.models import ViT_B_16_Weights  # Import inside to avoid warning
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # Fix deprecated usage
        self.vit.heads = nn.Identity()  # Remove classification head

        # Decoder (Simple linear layers to convert features to image)
        # Adjust the input size to match the concatenated feature size (768 + 768 = 1536)
        self.decoder = nn.Sequential(
            nn.Linear(768 + 768, 1024),  # Updated to match the concatenated size (768 + 768 = 1536)
            nn.ReLU(),
            nn.Linear(1024, 256*256*3),  # For output image size (256x256 RGB)
            nn.Tanh()  # Image normalization to [-1, 1]
        )

    def forward(self, text_input, image_input):
        # Get CLIP text embeddings
        text_embeds = self.clip_model.get_text_features(input_ids=text_input["input_ids"].to(device), 
                                                        attention_mask=text_input["attention_mask"].to(device))

        # Pool the text embeddings (take the average across the sequence dimension)
        text_embeds = text_embeds.mean(dim=1)  # Now shape [batch_size, embedding_dim]

        # Ensure both features have same dimensions before concatenation
        text_embeds = text_embeds.unsqueeze(0) if text_embeds.dim() == 1 else text_embeds  # Add batch dim if missing

        # Extract ViT features from image_input
        vit_features = self.vit(image_input)  # Should return a single tensor with shape [batch_size, 768]

        # Ensure that both text_embeds and vit_features have the same batch dimension
        if vit_features.size(0) != text_embeds.size(0):
            # Ensure matching batch size by repeating text_embeds
            text_embeds = text_embeds.repeat(vit_features.size(0), 1)  # Repeat to match batch size

        # Concatenate ViT features with text embeddings
        combined_features = torch.cat((vit_features, text_embeds), dim=1)  # Concatenate along feature dimension

        # Pass through the decoder to generate image
        generated_image = self.decoder(combined_features)

        return generated_image.view(-1, 3, 256, 256)  # Reshape to image size




# --- Initialize Model and Optimizer ---
model = ViTWithDecoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# --- DATASET AND DATALOADER ---
dataset = SuspectFaceDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- CLIP Text Preprocessing Function ---
def preprocess_text(text_list):
    return model.clip_processor(text=text_list, return_tensors="pt", padding=True, truncation=True)

# --- TRAINING LOOP ---
model.train()
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]...")
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)

        # Example text input (you can replace this with your actual input)
        text_input = preprocess_text(["Generate a face of a suspect"])

        # Generate image from text and input image
        generated_image = model(text_input, batch)

        # Assume you have real target images (batch itself here as placeholder)
        loss = loss_fn(generated_image, batch)  # Compare with real image

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress for every 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# --- SAVE MODEL ---
torch.save(model.state_dict(), "vit_clip_decoder.pth")
print("✅ Training complete.")
