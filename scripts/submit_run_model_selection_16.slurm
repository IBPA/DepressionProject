#!/bin/bash

#SBATCH --job-name=dep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/share/taglab/Arielle/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/share/taglab/Arielle/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00

cd ../..
python -u -m DepressionProjectNew.run_model_selection \
    ./DepressionProjectNew/output/data_cleaned_encoded_16_yesmental.csv \
    ./DepressionProjectNew/output/output_16_yesmental/results.pkl \
    ./DepressionProjectNew/output/output_16_yesmental/preprocessed \
    y16CH_Dep_YN_192m
