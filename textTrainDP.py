import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import sys

log_file = open('dp_last_layer_epshalf_log.txt','w')
sys.stdout = log_file

train_path = 'Corona_NLP_train.csv'
test_path = 'Corona_NLP_test.csv'

train_df = pd.read_csv(train_path, encoding="latin-1")
test_df = pd.read_csv(test_path, encoding="latin-1")

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

label_encoder = LabelEncoder()
train_df['Sentiment'] = label_encoder.fit_transform(train_df['Sentiment'])
test_df['Sentiment'] = label_encoder.transform(test_df['Sentiment'])

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

max_len = 128
batch_size = 32

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['OriginalTweet'].values,
    train_df['Sentiment'].values,
    test_size=0.1,
    random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len)
test_dataset = SentimentDataset(test_df['OriginalTweet'].values, test_df['Sentiment'].values, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        # Freeze all layers except the last
        for name, param in self.model.named_parameters():
            if "transformer.layer.5" in name:  # Fine-tune only the last layer
                param.requires_grad = trainable
            else:
                param.requires_grad = False

        # CLS token hidden representation
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        self.text_encoder = TextEncoder(model_name, pretrained=True, trainable=True)
        self.classifier = nn.Linear(self.text_encoder.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids, attention_mask)
        return self.classifier(features)

num_classes = len(label_encoder.classes_)
model = SentimentClassifier(model_name, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPSILON = 0.5
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
EPOCHS = 30
MAX_PHYSICAL_BATCH_SIZE = 32
LOGGING_INTERVAL = 200

privacy_engine = PrivacyEngine(accountant='rdp')

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_delta=DELTA,
    target_epsilon=EPSILON,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

def train_model_with_dp(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(1, epochs + 1):
        
        model.train()
        train_losses = []
        correct_train = 0
        total_train = 0

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for step, batch in enumerate(tqdm(memory_safe_data_loader, desc=f"Epoch {epoch}")):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                train_losses.append(loss.item())

                _, predicted = torch.max(outputs, dim=1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                if step > 0 and step % LOGGING_INTERVAL == 0:
                    train_loss = np.mean(train_losses)
                    eps = privacy_engine.get_epsilon(DELTA)

                    print(
                        f"Epoch: {epoch} | Step: {step} | Train Loss: {train_loss:.4f} | eps: {eps:.2f}"
                    )

        train_loss = np.mean(train_losses)
        train_accuracy = correct_train / total_train * 100
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, dim=1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = correct_val / total_val * 100
        print(f"Epoch {epoch}: Validation Accuracy: {val_accuracy:.2f}%")

    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

def evaluate_model(model, test_loader):
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    return correct_test / total_test * 100

train_model_with_dp(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

log_file.close()