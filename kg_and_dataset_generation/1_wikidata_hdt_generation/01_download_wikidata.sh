#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=50gb
#SBATCH --partition=cpu

wget https://dumps.wikimedia.org/wikidatawiki/entities/20251027/wikidata-20251027-all-BETA.nt.bz2