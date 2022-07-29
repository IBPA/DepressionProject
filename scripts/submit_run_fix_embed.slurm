#!/bin/bash

#SBATCH --job-name=dep_12_embed
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/home/asmyoo/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=100:00:00

cd ../..
python -u -m DepressionProjectNew.fix_embed_colors \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/ \
    y12CH_Dep_YN_144m
python -u -m DepressionProjectNew.test_old_embed_train \
   ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
   ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
   ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
   ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/test/ \
   y12CH_Dep_YN_144m