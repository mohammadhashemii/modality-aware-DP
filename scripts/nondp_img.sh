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
modality=image
image_encoder=resnet18
epochs=20
lr=1e-1
trainable=all-layers
bs=64

python train.py \
--dataset $dataset \
--modality $modality \
--epochs $epochs \
--lr $lr \
--trainable $trainable \
--bs $bs \
>> ./logs/nondp/image_${dataset}_${image_encoder}_epochs${epochs}_lr${lr}_bs${bs}_all_layers.log \
2>> ./logs/nondp/image_${dataset}_${image_encoder}_epochs${epochs}_lr${lr}_bs${bs}_all_layers.err


echo '=====end======='