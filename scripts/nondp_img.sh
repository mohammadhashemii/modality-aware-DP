#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a100-8-gm640-c96-m1152
bash
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd /scratch/mhashe4/repos/dpclip_image_classification/modality_aware_DP
conda activate dpclip

echo '====start running===='


dataset=cifar10
image_encoder=vit_base_patch16_224
epochs=20
lr=0.0005
trainable=all-layers
bs=1000
mini_bs=50

python train.py \
--dataset $dataset \
--modality image \
--epochs $epochs \
--lr $lr \
--trainable $trainable \
--bs $bs \
--mini_bs $mini_bs \
>> ./logs/11-22/nondp/image_${dataset}_${image_encoder}_epochs${epochs}_lr${lr}_bs${mini_bs}_all_layers.log \
2>> ./logs/11-22/nondp/image_${dataset}_${image_encoder}_epochs${epochs}_lr${lr}_bs${mini_bs}_all_layers.err


echo '=====end======='