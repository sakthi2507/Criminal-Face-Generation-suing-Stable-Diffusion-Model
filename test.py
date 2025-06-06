import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIG ---
image_dir = "human_faces"
prompt_dim = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.transform(img), self.image_paths[idx]

# --- MODEL ---
class ViTWithPrompt(nn.Module):
    def __init__(self, prompt_dim=768):
        super(ViTWithPrompt, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()
        self.learned_prompt = nn.Parameter(torch.randn(1, prompt_dim))
        self.head = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, prompt_dim)
        )

    def forward(self, x):
        features = self.vit(x)
        prompt_expanded = self.learned_prompt.expand(x.size(0), -1)
        out = self.head(features)
        return out, prompt_expanded

# --- LOAD TRAINED MODEL ---
model = ViTWithPrompt().to(device)
model.load_state_dict(torch.load("vit_prompt_suspect.pth"))
model.eval()

# --- LOAD DATA ---
dataset = SuspectFaceDataset(image_dir, transform)
loader = DataLoader(dataset, batch_size=1)

# --- COMPUTE DISTANCE TO PROMPT ---
image_scores = []

with torch.no_grad():
    for img_tensor, path in tqdm(loader):
        img_tensor = img_tensor.to(device)
        out, prompt = model(img_tensor)
        distance = torch.norm(out - prompt).item()
        image_scores.append((distance, path[0]))

# --- SHOW TOP 5 CLOSEST IMAGES ---
top_k = sorted(image_scores, key=lambda x: x[0])[:5]

plt.figure(figsize=(15, 3))
for i, (score, path) in enumerate(top_k):
    img = Image.open(path)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Score: {score:.2e}")
    plt.axis("off")

plt.suptitle("Top 5 Closest Images to Prompt Vector")
plt.show()
