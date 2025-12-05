#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=400gb
#SBATCH --partition=cpu

#create option.hdtspec
export JAVA_OPTIONS="-Xmx256g"

# tmp folder is much faster because it is on a nvme
cp wikidata-20251027-all-BETA.nt.bz2 /tmp/wikidata-20251027-all-BETA.nt.bz2
bzip2 -cd /tmp/wikidata-20251027-all-BETA.nt.bz2 | qendpoint-cli-2.5.0/bin/rdf2hdt.sh -index -quiet -cattree -cattreelocation /tmp/cattree-wikidata-20251027 -options loader.cattree.kcat=20 wikidata-20251027-all-BETA.hdt
