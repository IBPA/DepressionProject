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
#SBATCH --mem=500G
#SBATCH --time=72:00:00

cd ../..
python -u -m DepressionProjectNew.run_model_selection \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_18_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/preprocessed \
    y18CH_Dep_YN_216m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_18_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/ \
    y18CH_Dep_YN_216m