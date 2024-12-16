import json
import random
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager

BATCH_SIZE = 1024
EPSILON = 0.5
EPOCHS = 10
LEARNING_RATE = 5e-3
MAX_LEN = 128
TRAIN_PERCENTAGE = 50
VAL_PERCENTAGE = 100
TEST_PERCENTAGE = 100

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
    sample_size = int(len(data) * (percentage / 100))
    return random.sample(data, sample_size)

train_data = load_data('preprocess_train_data.json')
val_data = load_data('preprocess_val_data.json')
test_data = load_data('preprocess_test_data.json')

train_data, label_to_id = preprocess_data(train_data)
val_data, _ = preprocess_data(val_data)
test_data, _ = preprocess_data(test_data)

train_data = sample_data(train_data, TRAIN_PERCENTAGE)
val_data = sample_data(val_data, VAL_PERCENTAGE)
test_data = sample_data(test_data, TEST_PERCENTAGE)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_dataset = ProductDataset(train_data, tokenizer, max_len=MAX_LEN)
val_dataset = ProductDataset(val_data, tokenizer, max_len=MAX_LEN)
test_dataset = ProductDataset(test_data, tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_to_id))

# for name, param in model.named_parameters():
#     if "classifier" not in name:
#         param.requires_grad = False

base_optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-2
)

optimizer = DPOptimizer(
    expected_batch_size=BATCH_SIZE,
    optimizer=base_optimizer,
    noise_multiplier=EPSILON,
    max_grad_norm=1.0
)

criterion = torch.nn.CrossEntropyLoss()


model.train()

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=EPSILON,
    max_grad_norm=1.0,
)


for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}")

    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:

        for batch in tqdm(memory_safe_data_loader, desc="Training", unit="batch"):
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


torch.save(model.state_dict(), f"finetuned_distilbert_last_layer_model_eps{EPSILON}.pth")
torch.save(tokenizer, f"finetuned_distilbert_last_layer_tokenizer_eps{EPSILON}.pth")


print("Model fine-tuning with Differential Privacy complete and saved.")


#eps = 10
# Train Loss: 141.5136, Train Accuracy: 0.3124
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:10<00:00, 23.87s/batch] 
# Validation Loss: 15.8096, Validation Accuracy: 0.3368
# Epoch 2
# Training: 60batch [22:44, 22.74s/batch]                                                                                                                      
# Train Loss: 95.9950, Train Accuracy: 0.4875
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:07<00:00, 23.48s/batch] 
# Validation Loss: 12.4068, Validation Accuracy: 0.4684
# Epoch 3
# Training: 65batch [22:18, 20.59s/batch]                                                                                                                      
# Train Loss: 85.3973, Train Accuracy: 0.5844
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.21s/batch] 
# Validation Loss: 10.2112, Validation Accuracy: 0.5875
# Epoch 4
# Training: 63batch [19:56, 18.99s/batch]                                                                                                                      
# Train Loss: 73.2566, Train Accuracy: 0.6467
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.07s/batch] 
# Validation Loss: 9.2110, Validation Accuracy: 0.6575
# Epoch 5
# Training: 65batch [20:07, 18.57s/batch]                                                                                                                      
# Train Loss: 71.4249, Train Accuracy: 0.6810
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.20s/batch] 
# Validation Loss: 8.0508, Validation Accuracy: 0.7068
# Epoch 6
# Training: 55batch [19:41, 21.49s/batch]                                                                                                                      
# Train Loss: 57.5915, Train Accuracy: 0.7091
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.16s/batch] 
# Validation Loss: 7.9477, Validation Accuracy: 0.7151
# Epoch 7
# Training: 58batch [19:47, 20.47s/batch]                                                                                                                      
# Train Loss: 59.8115, Train Accuracy: 0.7222
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.20s/batch] 
# Validation Loss: 7.9619, Validation Accuracy: 0.7341
# Epoch 8
# Training: 61batch [19:53, 19.57s/batch]                                                                                                                      
# Train Loss: 61.7333, Train Accuracy: 0.7324
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.12s/batch] 
# Validation Loss: 7.9448, Validation Accuracy: 0.7453
# Epoch 9
# Training: 61batch [19:55, 19.60s/batch]                                                                                                                      
# Train Loss: 62.5030, Train Accuracy: 0.7399
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.18s/batch] 
# Validation Loss: 7.8564, Validation Accuracy: 0.7587
# Epoch 10
# Training: 68batch [20:09, 17.79s/batch]                                                                                                                      
# Train Loss: 69.3244, Train Accuracy: 0.7461
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.21s/batch] 
# Validation Loss: 8.0635, Validation Accuracy: 0.7603
# Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.10s/batch] 
# Test Loss: 7.9962, Test Accuracy: 0.7536


#eps = 3
# Train Loss: 306.7727, Train Accuracy: 0.1758
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:19<00:00, 13.27s/batch] 
# Validation Loss: 41.0143, Validation Accuracy: 0.1265
# Epoch 2
# Training: 121batch [22:57, 11.38s/batch]                                                                                                                     
# Train Loss: 294.7081, Train Accuracy: 0.2388
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:19<00:00, 13.33s/batch] 
# Validation Loss: 38.9946, Validation Accuracy: 0.1493
# Epoch 3
# Training: 123batch [23:14, 11.34s/batch]                                                                                                                     
# Train Loss: 284.2126, Train Accuracy: 0.2706
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:18<00:00, 13.25s/batch] 
# Validation Loss: 37.1532, Validation Accuracy: 0.1736
# Epoch 4
# Training: 127batch [23:16, 11.00s/batch]                                                                                                                     
# Train Loss: 277.9035, Train Accuracy: 0.3070
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:18<00:00, 13.26s/batch] 
# Validation Loss: 35.2834, Validation Accuracy: 0.2171
# Epoch 5
# Training: 122batch [23:09, 11.39s/batch]                                                                                                                     
# Train Loss: 253.2415, Train Accuracy: 0.3420
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:18<00:00, 13.24s/batch] 
# Validation Loss: 33.8684, Validation Accuracy: 0.2564
# Epoch 6
# Training: 126batch [23:15, 11.08s/batch]                                                                                                                     
# Train Loss: 250.4433, Train Accuracy: 0.3761
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:20<00:00, 13.37s/batch] 
# Validation Loss: 32.3012, Validation Accuracy: 0.2868
# Epoch 7
# Training: 123batch [23:17, 11.37s/batch]                                                                                                                     
# Train Loss: 233.2645, Train Accuracy: 0.4060
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:19<00:00, 13.30s/batch] 
# Validation Loss: 31.2666, Validation Accuracy: 0.2900
# Epoch 8
# Training: 126batch [23:07, 11.02s/batch]                                                                                                                     
# Train Loss: 229.5243, Train Accuracy: 0.4285
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:22<00:00, 13.50s/batch] 
# Validation Loss: 29.8991, Validation Accuracy: 0.3297
# Epoch 9
# Training: 127batch [23:18, 11.01s/batch]                                                                                                                     
# Train Loss: 223.7458, Train Accuracy: 0.4388
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:19<00:00, 13.33s/batch] 
# Validation Loss: 28.9826, Validation Accuracy: 0.3404
# Epoch 10
# Training: 122batch [23:19, 11.47s/batch]                                                                                                                     
# Train Loss: 206.1603, Train Accuracy: 0.4631
# Validation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:18<00:00, 13.26s/batch] 
# Validation Loss: 28.1849, Validation Accuracy: 0.3519
# Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [03:19<00:00, 13.27s/batch] 
# Test Loss: 28.2511, Test Accuracy: 0.3529
# Model fine-tuning with Differential Privacy complete and saved.
# PS C:\Users\cy295\Desktop\school 8\modality-aware-DP> & C:/Python312/python.exe "c:/Users/cy295/Desktop/school 8/modality-aware-DP/finetune_dp.py"
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# C:\Python312\Lib\site-packages\opacus\privacy_engine.py:95: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.
#   warnings.warn(
# Epoch 1
# Training:   0%|                                                                                                                   | 0/45 [00:00<?, ?batch/s]C:\Users\cy295\AppData\Roaming\Python\Python312\site-packages\torch\nn\modules\module.py:1640: FutureWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
#   self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
# Training: 65batch [22:26, 20.71s/batch]
# Train Loss: 141.5136, Train Accuracy: 0.3124
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:10<00:00, 23.87s/batch] 
# Validation Loss: 15.8096, Validation Accuracy: 0.3368
# Epoch 2
# Training: 60batch [22:44, 22.74s/batch]                                                                                                                      
# Train Loss: 95.9950, Train Accuracy: 0.4875
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:07<00:00, 23.48s/batch] 
# Validation Loss: 12.4068, Validation Accuracy: 0.4684
# Epoch 3
# Training: 65batch [22:18, 20.59s/batch]                                                                                                                      
# Train Loss: 85.3973, Train Accuracy: 0.5844
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.21s/batch] 
# Validation Loss: 10.2112, Validation Accuracy: 0.5875
# Epoch 4
# Training: 63batch [19:56, 18.99s/batch]                                                                                                                      
# Train Loss: 73.2566, Train Accuracy: 0.6467
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.07s/batch] 
# Validation Loss: 9.2110, Validation Accuracy: 0.6575
# Epoch 5
# Training: 65batch [20:07, 18.57s/batch]                                                                                                                      
# Train Loss: 71.4249, Train Accuracy: 0.6810
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.20s/batch] 
# Validation Loss: 8.0508, Validation Accuracy: 0.7068
# Epoch 6
# Training: 55batch [19:41, 21.49s/batch]                                                                                                                      
# Train Loss: 57.5915, Train Accuracy: 0.7091
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.16s/batch] 
# Validation Loss: 7.9477, Validation Accuracy: 0.7151
# Epoch 7
# Training: 58batch [19:47, 20.47s/batch]                                                                                                                      
# Train Loss: 59.8115, Train Accuracy: 0.7222
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.20s/batch] 
# Validation Loss: 7.9619, Validation Accuracy: 0.7341
# Epoch 8
# Training: 61batch [19:53, 19.57s/batch]                                                                                                                      
# Train Loss: 61.7333, Train Accuracy: 0.7324
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.12s/batch] 
# Validation Loss: 7.9448, Validation Accuracy: 0.7453
# Epoch 9
# Training: 61batch [19:55, 19.60s/batch]                                                                                                                      
# Train Loss: 62.5030, Train Accuracy: 0.7399
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.18s/batch] 
# Validation Loss: 7.8564, Validation Accuracy: 0.7587
# Epoch 10
# Training: 68batch [20:09, 17.79s/batch]                                                                                                                      
# Train Loss: 69.3244, Train Accuracy: 0.7461
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.21s/batch] 
# Validation Loss: 8.0635, Validation Accuracy: 0.7603
# Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.10s/batch] 
# Test Loss: 7.9962, Test Accuracy: 0.7536
# Model fine-tuning with Differential Privacy complete and saved.
# PS C:\Users\cy295\Desktop\school 8\modality-aware-DP> & C:/Python312/python.exe "c:/Users/cy295/Desktop/school 8/modality-aware-DP/finetune_dp.py"
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# C:\Python312\Lib\site-packages\opacus\privacy_engine.py:95: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.
#   warnings.warn(
# Epoch 1
# Training:   0%|                                                                                                                   | 0/45 [00:00<?, ?batch/s]C:\Users\cy295\AppData\Roaming\Python\Python312\site-packages\torch\nn\modules\module.py:1640: FutureWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.
#   self._maybe_warn_non_full_backward_hook(args, result, grad_fn)
# Training: 63batch [20:10, 19.22s/batch]
# Train Loss: 153.3552, Train Accuracy: 0.2288
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:42<00:00, 20.36s/batch] 
# Validation Loss: 19.2564, Validation Accuracy: 0.1861
# Epoch 2
# Training: 60batch [20:01, 20.02s/batch]                                                                                                                      
# Train Loss: 122.5947, Train Accuracy: 0.3582
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:46<00:00, 20.82s/batch] 
# Validation Loss: 16.8070, Validation Accuracy: 0.3007
# Epoch 3
# Training: 61batch [20:00, 19.69s/batch]                                                                                                                      
# Train Loss: 110.9922, Train Accuracy: 0.4287
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:42<00:00, 20.32s/batch] 
# Validation Loss: 14.9131, Validation Accuracy: 0.3719
# Epoch 4
# Training: 64batch [20:16, 19.00s/batch]                                                                                                                      
# Train Loss: 107.5216, Train Accuracy: 0.4693
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:44<00:00, 20.55s/batch] 
# Validation Loss: 14.2558, Validation Accuracy: 0.3980
# Epoch 5
# Training: 56batch [20:20, 21.79s/batch]                                                                                                                      
# Train Loss: 89.5464, Train Accuracy: 0.5015
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:45<00:00, 20.69s/batch] 
# Validation Loss: 13.3640, Validation Accuracy: 0.4265
# Epoch 6
# Training: 65batch [20:24, 18.84s/batch]                                                                                                                      
# Train Loss: 98.5784, Train Accuracy: 0.5307
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:43<00:00, 20.40s/batch] 
# Validation Loss: 12.0689, Validation Accuracy: 0.5017
# Epoch 7
# Training: 59batch [20:03, 20.39s/batch]                                                                                                                      
# Train Loss: 85.2136, Train Accuracy: 0.5591
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:45<00:00, 20.64s/batch] 
# Validation Loss: 11.7217, Validation Accuracy: 0.5255
# Epoch 8
# Training: 59batch [20:11, 20.53s/batch]                                                                                                                      
# Train Loss: 82.0604, Train Accuracy: 0.5826
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:43<00:00, 20.43s/batch] 
# Validation Loss: 10.8315, Validation Accuracy: 0.5699
# Epoch 9
# Training: 62batch [20:14, 19.59s/batch]                                                                                                                      
# Train Loss: 85.1018, Train Accuracy: 0.5973
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:44<00:00, 20.62s/batch] 
# Validation Loss: 10.2157, Validation Accuracy: 0.6007
# Epoch 10
# Training: 61batch [20:15, 19.92s/batch]                                                                                                                      
# Train Loss: 81.6660, Train Accuracy: 0.6125
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:43<00:00, 20.48s/batch] 
# Validation Loss: 9.9368, Validation Accuracy: 0.6072
# Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:42<00:00, 20.37s/batch] 
# Test Loss: 10.0563, Test Accuracy: 0.6053


# Train Loss: 146.7269, Train Accuracy: 0.3030
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.20s/batch] 
# Validation Loss: 15.8245, Validation Accuracy: 0.3284
# Epoch 2
# Training: 63batch [21:56, 20.90s/batch]                                                                                                                      
# Train Loss: 100.2536, Train Accuracy: 0.4925
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:42<00:00, 20.28s/batch] 
# Validation Loss: 12.3719, Validation Accuracy: 0.4703
# Epoch 3
# Training: 63batch [19:59, 19.04s/batch]                                                                                                                      
# Train Loss: 83.1530, Train Accuracy: 0.5889
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.10s/batch] 
# Validation Loss: 10.6552, Validation Accuracy: 0.5724
# Epoch 4
# Training: 60batch [19:53, 19.90s/batch]                                                                                                                      
# Train Loss: 71.2510, Train Accuracy: 0.6430
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.11s/batch] 
# Validation Loss: 8.8796, Validation Accuracy: 0.6567
# Epoch 5
# Training: 60batch [19:51, 19.86s/batch]                                                                                                                      
# Train Loss: 65.7983, Train Accuracy: 0.6805
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.11s/batch] 
# Validation Loss: 8.3813, Validation Accuracy: 0.6861
# Epoch 6
# Training: 58batch [19:48, 20.50s/batch]                                                                                                                      
# Train Loss: 61.8979, Train Accuracy: 0.6989
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.23s/batch] 
# Validation Loss: 8.0299, Validation Accuracy: 0.7161
# Epoch 7
# Training: 57batch [19:47, 20.82s/batch]                                                                                                                      
# Train Loss: 60.7777, Train Accuracy: 0.7125
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:40<00:00, 20.12s/batch] 
# Validation Loss: 7.8404, Validation Accuracy: 0.7296
# Epoch 8
# Training: 65batch [20:02, 18.50s/batch]                                                                                                                      
# Train Loss: 68.2537, Train Accuracy: 0.7253
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 20.16s/batch] 
# Validation Loss: 7.6666, Validation Accuracy: 0.7411
# Epoch 9
# Training: 57batch [19:50, 20.89s/batch]                                                                                                                      
# Train Loss: 57.9568, Train Accuracy: 0.7373
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:45<00:00, 20.63s/batch] 
# Validation Loss: 7.9401, Validation Accuracy: 0.7503
# Epoch 10
# Training: 60batch [20:10, 20.17s/batch]                                                                                                                      
# Train Loss: 61.7153, Train Accuracy: 0.7423
# Validation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:44<00:00, 20.62s/batch] 
# Validation Loss: 7.6166, Validation Accuracy: 0.7619
# Testing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:44<00:00, 20.54s/batch] 
# Test Loss: 7.8305, Test Accuracy: 0.7555