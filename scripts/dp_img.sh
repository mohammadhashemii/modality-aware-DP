#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --partition=a100-8-gm640-c96-m1152
bash
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd /scratch/mhashe4/repos/dpclip_image_classification/modality_aware_DP
conda activate dpclip

echo '====start running===='


dataset=cifar10
image_encoder=vit_base_patch16_224
epochs=5
lr=0.0005
trainable=last-layer
bs=1000
mini_bs=50

for eps in 10 3 1 0.5; do
    python train.py \
    --dataset $dataset \
    --modality image \
    --private \
    --epsilon $eps \
    --epochs $epochs \
    --lr $lr \
    --trainable $trainable \
    --bs $bs \
    --mini_bs $mini_bs \
    >> ./logs/11-22/dp/image_${dataset}_${image_encoder}_eps${eps}_epochs${epochs}_lr${lr}_bs${mini_bs}_${trainable}.log \
    2>> ./logs/11-22/dp/image_${dataset}_${image_encoder}_eps${eps}_epochs${epochs}_lr${lr}_bs${mini_bs}_${trainable}.err

done

echo '=====end======='