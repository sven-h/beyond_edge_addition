#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --mem=50gb
#SBATCH --partition=cpu

wget https://github.com/the-qa-company/qEndpoint/releases/download/v2.5.0/qendpoint-cli.zip
unzip qendpoint-cli.zip
rm qendpoint-cli.zip