# Modality-Level Differential Privacy for Multi-Modal CLIP Models
A PyTorch code of modality-aware Differential Private training for multimodal models

This repository implements **Modality-Level Differential Privacy** for training and fine-tuning **CLIP models**. The repository supports multi-stage training with modality-specific privacy guarantees, allowing flexibility to allocate privacy budgets efficiently.

We compare our approach with the baseline **DP-CLIP** framework.

## Dataset Setup
Ensure the datasets (**CIFAR-10** and **Fashion Indio**) are downloaded and placed in the appropriate directories. The default dataset folder is `data/`.

- **CIFAR-10**: Download via torchvision.
- **Fashion Indio**: [Kaggle Dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset).

---

## Running the Scripts

### 1. Finetuning Image Encoder
This script fine-tunes the image encoder (e.g., ViT-B/16) using DP-SGD.

**Script Path**: `scripts/finetune_image.sh`

Run the script:
```bash
sbatch scripts/finetune_image.sh
```

### 2. Running DP-CLIP (Baseline)
This script runs the baseline DP-CLIP framework with a uniform privacy budget across both image and text encoders.

**Script Path**: `scripts/dp_clip_baseline.sh`

Run the script:
```bash
sbatch scripts/dp_clip_baseline.sh
```

---

### 3. Multi-Stage Fine-Tuning of CLIP
This script implements the proposed **multi-stage fine-tuning** approach, splitting the privacy budget between two stages.

**Script Path**: `scripts/multistage_clip.sh`

Run the script:
```bash
sbatch scripts/multistage_clip.sh
```

