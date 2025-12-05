#!/bin/bash
#SBATCH --partition=gpu-vram-48gb
#SBATCH --mem=500gb
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

/work/shertlin/miniconda3/envs/wikitwo/bin/python c_embed_corpus.py $1
