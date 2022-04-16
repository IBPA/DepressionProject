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
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_12_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/preprocessed \
    y12CH_Dep_YN_144m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_12_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/ \
    y12CH_Dep_YN_144m
