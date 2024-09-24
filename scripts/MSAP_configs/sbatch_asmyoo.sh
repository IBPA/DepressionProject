#!/bin/bash

#SBATCH --job-name=dep_rfe_12to18ad
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/MSAP/scripts/logs/%j.out
#SBATCH --error=/home/asmyoo/MSAP/scripts/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres gpu:1
#SBATCH --mem=10G
#SBATCH --time=200:00:00

./scripts/run_msap_ts_dep_rfe.sh
