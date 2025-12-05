#!/bin/bash
#SBATCH --mem=500G
#SBATCH --partition=cpu
#SBATCH --time=48:00:00

cp ./wikidata-20251027-all-BETA.hdt /dev/shm/wikidata-20251027-all-BETA.hdt
cp ./wikidata-20251027-all-BETA.hdt.index.v1-1 /dev/shm/wikidata-20251027-all-BETA.hdt.index.v1-1

/work/shertlin/miniconda3/envs/wikitwo/bin/python a_create_kg_cache.py


