import json
import random
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def load_data(input_file):
    with open(input_file, 'r') as infile:
        data = [json.loads(line) for line in infile]
    return data

class ProductDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        title = item['product_title']
        label = item['class_label']
        inputs = self.tokenizer(title, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': int(label)
        }

def preprocess_data(data):
    unique_labels = list(set(item['class_label'] for item in data))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    for item in data:
        item['class_label'] = label_to_id[item['class_label']]
    return data, label_to_id

def sample_data(data, percentage):
    """
    Randomly samples a percentage of the data.
    :param data: The original dataset
    :param percentage: Percentage of data to retain (0-100)
    :return: Subsampled data
    """
    sample_size = int(len(data) * (percentage / 100))
    return random.sample(data, sample_size)

# Load and preprocess data
train_data = load_data('preprocess_train_data.json')
val_data = load_data('preprocess_val_data.json')
test_data = load_data('preprocess_test_data.json')

train_data, label_to_id = preprocess_data(train_data)
val_data, _ = preprocess_data(val_data)
test_data, _ = preprocess_data(test_data)

# Adjust dataset size by percentage
train_percentage = 50  # Use 10% of the training data
val_percentage = 100    # Use 10% of the validation data
test_percentage = 100   # Use 10% of the test data

train_data = sample_data(train_data, train_percentage)
val_data = sample_data(val_data, val_percentage)
test_data = sample_data(test_data, test_percentage)

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create the datasets
train_dataset = ProductDataset(train_data, tokenizer, max_len=128)
val_dataset = ProductDataset(val_data, tokenizer, max_len=128)
test_dataset = ProductDataset(test_data, tokenizer, max_len=128)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_to_id))

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training and evaluation loop
for epoch in range(10):
    print(f"Epoch {epoch + 1}")

    # Training loop
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        logits = outputs.logits
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == batch['label']).sum().item()
        total_train += batch['label'].size(0)

    train_accuracy = correct_train / total_train
    print(f"Train Loss: {total_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", unit="batch"):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label']
            )
            loss = outputs.loss
            logits = outputs.logits
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_val += (preds == batch['label']).sum().item()
            total_val += batch['label'].size(0)

    val_accuracy = correct_val / total_val
    print(f"Validation Loss: {total_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test loop
model.eval()
total_test_loss = 0
correct_test = 0
total_test = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        logits = outputs.logits
        total_test_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_test += (preds == batch['label']).sum().item()
        total_test += batch['label'].size(0)

test_accuracy = correct_test / total_test
print(f"Test Loss: {total_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save_pretrained("finetuned_distilbert.pth")
tokenizer.save_pretrained("finetuned_distilbert.tok")

print("Model fine-tuning complete and saved.")

# Epoch 1
# Train Loss: 188.5869, Train Accuracy: 0.8481
# Validation Loss: 29.1816, Validation Accuracy: 0.8492
# Epoch 2
# Train Loss: 178.2675, Train Accuracy: 0.8597
# Validation Loss: 33.5828, Validation Accuracy: 0.8344
# Epoch 3
# Train Loss: 224.5605, Train Accuracy: 0.8331
# Validation Loss: 42.0944, Validation Accuracy: 0.8124
# Epoch 4
# Train Loss: 199.3154, Train Accuracy: 0.8501
# Validation Loss: 39.5730, Validation Accuracy: 0.7825
# Epoch 5
# Train Loss: 190.8759, Train Accuracy: 0.8529
# Validation Loss: 38.8358, Validation Accuracy: 0.8183
# Epoch 6
# Train Loss: 304.0972, Train Accuracy: 0.7460
# Validation Loss: 42.0251, Validation Accuracy: 0.7595
# Epoch 7
# Train Loss: 189.8272, Train Accuracy: 0.8489
# Validation Loss: 38.7678, Validation Accuracy: 0.8212
# Epoch 8
# Train Loss: 187.2528, Train Accuracy: 0.8496
# Validation Loss: 31.9587, Validation Accuracy: 0.8353
# Epoch 9
# Train Loss: 153.1857, Train Accuracy: 0.8743
# Validation Loss: 29.6705, Validation Accuracy: 0.8407
# Epoch 10
# Train Loss: 150.2652, Train Accuracy: 0.8770
# Validation Loss: 30.5286, Validation Accuracy: 0.8347
# Test Loss: 31.1054, Test Accuracy: 0.8372