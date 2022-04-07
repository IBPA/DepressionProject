#!/bin/bash

#SBATCH --job-name=dep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asmyoo@ucdavis.edu
#SBATCH --output=/share/taglab/Arielle/DepressionProjectNew/output/logs/%j.out
#SBATCH --error=/share/taglab/Arielle/DepressionProjectNew/output/logs/%j.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=3:00:00

cd ../..
python -u -m DepressionProject.run_model_selection ./DepressionProjectNew/output/data_cleaned_encoded_18_yesmental.csv ./DepressionProjectNew/output/results.pkl ./DepressionProjectNew/output/preprocessed y18CH_Dep_YN_216m
