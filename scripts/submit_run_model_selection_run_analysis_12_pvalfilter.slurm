#!/bin/bash

#SBATCH --job-name=dep_12_pvalfilter
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/home/asmyoo/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=100:00:00

cd ../..
python -u -m DepressionProjectNew.run_model_selection \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    y12CH_Dep_YN_144m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/ \
    y12CH_Dep_YN_144m
