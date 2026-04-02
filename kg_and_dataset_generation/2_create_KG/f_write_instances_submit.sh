#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=500gb
#SBATCH --time=24:00:00
#SBATCH --exclude=dws-[01-10]

/work/shertlin/miniconda3/envs/faiss/bin/python f_write_instances.py


