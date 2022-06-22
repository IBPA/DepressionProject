#!/bin/bash

#SBATCH --job-name=dep
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
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/ \
    y12to18_Dep_YN_216m