# DepressionProject

## Install python 3.10
## Install MSAP
```
git clone https://github.com/asmyoo/MSAP.git
pip install -e ./MSAP
```

## Install kneed
```
git clone https://github.com/asmyoo/kneed.git
pip install -e ./kneed
```

## Install kneebow
```
git clone https://github.com/asmyoo/kneebow.git
pip install -e ./kneebow
```

## Install requirements
```
pip install -r requirements.txt
```
- for additional info, might need some files in old_files folder
- hpc is what was used to run all the results except for ones requiring ipynb
- local is what was used to run ipynb results

## Get “preprocessed” data - run reformat_ml.py within src/preprocess
```
python reformat_ml.py
```

## Make 12to18 data if want to change meaning to be 1 anywhere even if missing data to be 1 within src/preprocess
```
python make_12to18.py
```
- Change model_selecting.py config to use the new dataset preprocessed_data_without_temporal_12to18.csv

## Make 12to18 average depression score within src/preprocess
```
python make_12to18ave.py
```
- Change model_selecting.py config to use the new dataset preprocessed_data_without_temporal_12to18ave.csv

## Change configs
- Cleaning.py for % missing value imputation and make sure columns_ignored contains child id variable name
- Model_selecting.py for age_cutoff and column_dependent

## Run get_config_info.py within src/preprocess
```
python get_config_info.py
```
- Make sure within get_config_info the default preprocessed data filename is correct
- Prediction label is 0/1 so does not need to be marked as categorical unless mistake is made
- Change preprocessing.py config categorical variables if needed (probably not)
- Change cleaning.py with columns_ignored to add mental health variables (don't do for now because our predictions seem to use these variables heavily to predict)

## Run depression-predictor run_eda.py and feature_analysis_correlations_iterativeimpute.ipynb
- Make sure to create a new conda environment for the requirements for depression-predictor
```
git clone https://github.com/asmyoo/depression-predictor.git
cd depression-predictor
pip install -r requirements.txt
cd ..
python -u -m depression-predictor.depp.run_eda
```
- Copy the Variables excel file and preprocessed data into the depression-predictor data folder
- Check filename for data in depression-predictor utils/dataset.py
- Takes approx 1 hr
- Copy vars_sorted.csv to DepressionProject/output
- Then run python notebook feature_analysis_correlations_iterativeimpute.ipynb

## Run run_cleaner.py
```
python -u -m DepressionProject.run_cleaner
```
- Make sure to not overwrite png's from feature_analysis_correlations_iterativeimpute.ipynb, missing_value png’s, and data_cleaned.csv's

## Run run_encode.py
```
python -m DepressionProject.run_encode DepressionProject/output/data_cleaned.csv DepressionProject/output/data_cleaned_encoded.csv
```
- Move output files into output folder (separated by age, include png's and etc)

## Run run_model_selection.py
- Use script

## Run run_analysis.py
- Use script

## Run univariate comparison
```
python -u -m DepressionProject.run_univariate \
    ./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/results.pkl \
    ./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/preprocessed \
    ./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/data_cleaned_encoded.csv \
    ./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/balanced_accuracy \
    y12to18_Dep_YN_216m \
    --use-balanced-accuracy
```

## Run fix_embed_colors for age 12/if colors are switched for depressed/not depressed
```
python -u -m DepressionProject.fix_embed_colors \
    ./DepressionProject/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProject/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    ./DepressionProject/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProject/output/pval_filter_60_MVI/output_12_yesmental/ \
    y12CH_Dep_YN_144m
```

## Run make_readable_all_var_sorted.py to change the description column of all vars_sorted_dir_ranked_rounded.csv to be more readable
```
python -u -m DepressionProject.make_readable_all_var_sorted ./DepressionProject/output/pval_filter_60_MVI
```

## Run make_readable_heatmapcsv.py if have pearson.csv of x and y variables that are highly correlated or anticorrelated after looking at the pearson heatmap
```
python -u -m DepressionProject.make_readable_heatmapcsv ./DepressionProject/output/rfe_pearson_spearman/output_12_yesmental
```

## Run get_unique_fts for getting list of unique features for each model
```
python -u -m DepressionProject.get_unique_fts ./DepressionProject/output/pval_filter_60_MVI
```

## Run rank_pearson_rfe for getting table of pearson correlations
```
python -u -m DepressionProject.rank_pearson_rfe ./DepressionProject/output/pval_filter_60_MVI
```

## Run run_tsne_cluster.py for age 12to18 to understand one cluster
```
python -u -m DepressionProject.run_tsne_cluster \
./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/results.pkl \
./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/preprocessed \
./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/data_cleaned_encoded.csv \
./DepressionProject/output/pval_filter_60_MVI/output_12to18_yesmental/f1 \
y12to18_Dep_YN_216m
```

## Run plot_rfe_jaccard.py to compare overlap of RFE selected features
```
python -u -m DepressionProject.plot_rfe_jaccard \
./DepressionProject/output/pval_filter_60_MVI/Supplementary\ Spreadsheet\ 3.xlsx
./DepressionProject/output/pval_filter_60_MVI/rfe_jaccard.svg
```

## Run get_top_10_rfe.py to get top 10 features from RFE for all ages into a csv
```
python -u -m DepressionProject.get_top_10_rfe \
./DepressionProject/output/pval_filter_60_MVI/Supplementary\ Spreadsheet\ 3.xlsx
./DepressionProject/output/pval_filter_60_MVI/rfe_jaccard.svg
```

## Run print_num_fts_missingvalratio.py to get number of features and missing value ratio before cleaning
```
python -u -m DepressionProject.print_num_fts_missvalratio
```

## Check duplicate samples that were created on accident prior to analysis
### Get “preprocessed” data with more info - run reformat_ml_checkdups.py within src/preprocess
```
python reformat_ml_checkdups.py
```
### Run clean_dups.py within src/preprocess
```
python clean_dups.py
```
### Run check_dups.py within src/preprocess to see if the duplicates affect the analysis
```
python check_dups.py
```

## Check missing value ratio before analysis again
```
python -u -m DepressionProject.print_num_fts_missvalratio --path_data ./DepressionProject/output/preprocessed_data_without_temporal_checkdup_cleaned_no_info.csv
```


## Run run_tsne.py (don't need)

## Run make_readable_pcc_sc_kendall.py and make_readable_list.py after pasting in the best rfe list and lists from run_univariate's output from src/preprocess

## Plot tsne using only best results from RFE/Elbow method (don't need)
Make sure to input the hardcoded variables for the rfe results
```
python -u -m DepressionProject.run_tsne_use_rfe_results_all \
    ./DepressionProject/output/10MVIout/output_12_yesmental \
    ./DepressionProject/output/10MVIout/output_16_yesmental \
    ./DepressionProject/output/10MVIout/output_17_yesmental \
    ./DepressionProject/output/10MVIout/output_18_yesmental \
    y12CH_Dep_YN_144m \
    y16CH_Dep_YN_192m \
    y17CH_Dep_YN_204m \
    y18CH_Dep_YN_216m
```

## Calculate F1 baselines and plot into confusion matrix (don't need)
```
python -u -m DepressionProject.run_f1_calcs_baseline_all \
    ./DepressionProject/output/10MVIout/output_12_yesmental \
    ./DepressionProject/output/10MVIout/output_16_yesmental \
    ./DepressionProject/output/10MVIout/output_17_yesmental \
    ./DepressionProject/output/10MVIout/output_18_yesmental \
    y12CH_Dep_YN_144m \
    y16CH_Dep_YN_192m \
    y17CH_Dep_YN_204m \
    y18CH_Dep_YN_216m
```

## Plot F1's with their baseline (don't need)
```
python -u -m DepressionProject.plot_f1_overall
    ./DepressionProject/output/10MVIout/f1s.png
```