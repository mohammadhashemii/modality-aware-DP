import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np

# Constants
BASE_PATH = "fashion_indio/data"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "images/train")
VAL_IMAGES_PATH = os.path.join(BASE_PATH, "images/val")
TRAIN_JSON = os.path.join(BASE_PATH, "train_data.json")
VAL_JSON = os.path.join(BASE_PATH, "val_data.json")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class FashionDataset(Dataset):
    def __init__(self, json_file, images_dir, tokenizer, transform=None):
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.images_dir, os.path.basename(item["image_path"]))
        text = item["product_title"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_tokens = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        return image, text_tokens

# Transform and Tokenizer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Data Loaders
train_dataset = FashionDataset(TRAIN_JSON, TRAIN_IMAGES_PATH, tokenizer, transform)
val_dataset = FashionDataset(VAL_JSON, VAL_IMAGES_PATH, tokenizer, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Encoders and Projection Heads
image_encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(DEVICE)
text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

image_projection = ProjectionHead(768, 512).to(DEVICE)
text_projection = ProjectionHead(768, 512).to(DEVICE)

# Freeze all layers in the text encoder except the last layer
for name, param in text_encoder.named_parameters():
    if "transformer.layer" in name:
        # Enable gradients for the last layer only
        if "layer.5" in name:  # DistilBERT has 6 layers (index 0 to 5)
            param.requires_grad = True
        else:
            param.requires_grad = False
    else:
        param.requires_grad = False

# Criterion and Optimizers
criterion = torch.nn.CrossEntropyLoss()
image_optimizer = AdamW(
    list(image_encoder.parameters()) + list(image_projection.parameters()), lr=LEARNING_RATE
)
text_projection_optimizer = AdamW(text_projection.parameters(), lr=LEARNING_RATE)
text_encoder_optimizer = AdamW(
    [param for param in text_encoder.parameters() if param.requires_grad], 
    lr=LEARNING_RATE
)

# Privacy Engine for Text Encoder
EPSILON = 0.5
MAX_GRAD_NORM = 1.0
DELTA = 1 / (2 * len(train_dataset))
privacy_engine = PrivacyEngine(accountant="rdp")

text_encoder.train()

text_encoder, text_encoder_optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=text_encoder,
    optimizer=text_encoder_optimizer,
    data_loader=train_loader,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
    epochs=EPOCHS
)

# Training and Validation Functions
def train_one_epoch_with_dp(
    model_components, dataloader, image_optimizer, text_encoder_optimizer, text_proj_optimizer, criterion, device
):
    image_encoder, text_encoder, image_proj, text_proj = model_components
    image_encoder.train()
    text_encoder.train()
    image_proj.train()
    text_proj.train()

    epoch_loss = 0
    correct = 0
    total = 0

    with BatchMemoryManager(
        data_loader=dataloader,
        max_physical_batch_size=BATCH_SIZE,
        optimizer=text_encoder_optimizer,
    ) as memory_safe_dataloader:
        for images, text_tokens in tqdm(memory_safe_dataloader):
            images = images.to(device)
            input_ids = text_tokens["input_ids"].squeeze(1).to(device)
            attention_mask = text_tokens["attention_mask"].squeeze(1).to(device)

            # Zero the gradients for all optimizers
            image_optimizer.zero_grad()
            text_encoder_optimizer.zero_grad()
            text_proj_optimizer.zero_grad()

            # Forward pass
            image_features = image_proj(image_encoder(images))
            text_features = text_proj(
                text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            )

            similarity = torch.matmul(image_features, text_features.T)
            labels = torch.arange(images.size(0)).to(device)

            # Compute loss
            loss = criterion(similarity, labels) + criterion(similarity.T, labels)
            epoch_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(similarity, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Backward pass and optimization
            loss.backward()
            image_optimizer.step()
            text_encoder_optimizer.step()
            text_proj_optimizer.step()

    epoch_accuracy = correct / total * 100
    return epoch_loss / len(dataloader), epoch_accuracy


def validate(model_components, dataloader, criterion, device):
    image_encoder, text_encoder, image_proj, text_proj = model_components
    image_encoder.eval()
    text_encoder.eval()
    image_proj.eval()
    text_proj.eval()

    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, text_tokens in tqdm(dataloader):
            images = images.to(device)
            input_ids = text_tokens["input_ids"].squeeze(1).to(device)
            attention_mask = text_tokens["attention_mask"].squeeze(1).to(device)

            image_features = image_proj(image_encoder(images))
            text_features = text_proj(
                text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            )

            similarity = torch.matmul(image_features, text_features.T)
            labels = torch.arange(images.size(0)).to(device)

            loss = criterion(similarity, labels) + criterion(similarity.T, labels)
            epoch_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(similarity, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_accuracy = correct / total * 100
    return epoch_loss / len(dataloader), epoch_accuracy


def evaluate(model_components, dataloader, device):
    image_encoder, text_encoder, image_proj, text_proj = model_components
    image_encoder.eval()
    text_encoder.eval()
    image_proj.eval()
    text_proj.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, text_tokens in tqdm(dataloader):
            images = images.to(device)
            input_ids = text_tokens["input_ids"].squeeze(1).to(device)
            attention_mask = text_tokens["attention_mask"].squeeze(1).to(device)

            image_features = image_proj(image_encoder(images))
            text_features = text_proj(
                text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            )

            similarity = torch.matmul(image_features, text_features.T)
            predictions = torch.argmax(similarity, dim=1)

            labels = torch.arange(images.size(0)).to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Training Loop
model_components = (image_encoder, text_encoder, image_projection, text_projection)
for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_one_epoch_with_dp(
        model_components, train_loader, image_optimizer, text_encoder_optimizer, text_projection_optimizer, criterion, DEVICE
    )
    val_loss, val_accuracy = validate(model_components, val_loader, criterion, DEVICE)
    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
    )

# Final Evaluation
evaluate(model_components, val_loader, DEVICE)
