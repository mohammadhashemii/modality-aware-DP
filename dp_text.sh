#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a10g-8-gm192-c192-m768
bash
conda init bash >/dev/null 2>&1
source ~/.bashrc
cd /scratch/ycai222/modality-aware-DP
conda activate dp_clip
 
echo '====start running===='
python textTrainDP.py >> ./logs/epch30_textDP_finetune.log 2>> ./logs/epch30_textDP_finetune.err
echo '=====end======='