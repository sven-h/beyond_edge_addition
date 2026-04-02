#!/bin/bash
#SBATCH --partition=gpu-vram-94gb
#SBATCH --mem=400gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

/work/shertlin/miniconda3/envs/faiss/bin/python e_search_hard_negatives.py $1


