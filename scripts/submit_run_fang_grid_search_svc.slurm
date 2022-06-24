#!/bin/bash

#SBATCH --job-name=grid_search_mlp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/DepressionProjectNew/fang_code/outputs/logs/%j.out
#SBATCH --error=/home/asmyoo/DepressionProjectNew/fang_code/outputs/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=24:00:00

cd ..
python -u -m fang_code.grid_search_svc