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
modality=image
image_encoder=resnet18
epochs=100
lr=1e-1
trainable=all-layers
bs=64

for eps in 10 3 1 0.5; do
    python train.py \
    --dataset $dataset \
    --modality $modality \
    --private \
    --epsilon $eps \
    --epochs $epochs \
    --lr $lr \
    --trainable $trainable \
    --bs $bs \
    >> ./logs/dp/image_${dataset}_${image_encoder}_eps${eps}_epochs${epochs}_lr${lr}_bs${bs}_all_layers.log \
    2>> ./logs/dp/image_${dataset}_${image_encoder}_eps${eps}_epochs${epochs}_lr${lr}_bs${bs}_all_layers.err

done

echo '=====end======='