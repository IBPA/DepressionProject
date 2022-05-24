#!/bin/bash

#SBATCH --job-name=dep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/home/asmyoo/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=72:00:00

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
