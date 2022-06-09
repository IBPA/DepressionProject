#!/bin/bash

#SBATCH --job-name=dep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/home/asmyoo/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/home/asmyoo/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=72:00:00

cd ../..
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_12_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental/ \
    y12CH_Dep_YN_144m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_12to18_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_12to18_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_12to18_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_12to18_yesmental/ \
    y12to18_Dep_YN_216m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_13_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_13_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_13_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_13_yesmental/ \
    y13CH_Dep_YN_162m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_16_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental/ \
    y16CH_Dep_YN_192m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_17_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_17_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_17_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_17_yesmental/ \
    y17CH_Dep_YN_204m
python -u -m DepressionProjectNew.run_analysis \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_18_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/ \
    y18CH_Dep_YN_216m