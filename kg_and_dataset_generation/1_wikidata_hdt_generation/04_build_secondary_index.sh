#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=200gb
#SBATCH --partition=cpu

/work/shertlin/miniconda3/envs/wikidata/bin/python -c 'from rdflib_hdt import HDTStore;HDTStore("wikidata-20251027-all-BETA.hdt")'
