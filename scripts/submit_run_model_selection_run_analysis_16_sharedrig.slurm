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
#SBATCH --time=48:00:00

cd ../..
python -u -m DepressionProjectNew.run_model_selection \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_16_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/preprocessed \
    y16CH_Dep_YN_192m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_16_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/ \
    y16CH_Dep_YN_192m
