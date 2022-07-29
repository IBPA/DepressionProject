# DepressionProjectNew

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
```
python -u -m depression-predictor.depp.run_eda
```
- Make sure using a NEW conda environment for running depression-predictor run_eda.py
- Copy the Variables excel file and preprocessed data into the depression-predictor data folder
- Check filename for data in depression-predictor utils/dataset.py
- Takes approx 1 hr
- Copy vars_sorted.csv to DepressionProjectNew/output
- Then run python notebook feature_analysis_correlations_iterativeimpute.ipynb

## Run run_cleaner.py
```
python -u -m DepressionProjectNew.run_cleaner
```
- Make sure to not overwrite png's from feature_analysis_correlations_iterativeimpute.ipynb, missing_value png’s, and data_cleaned.csv's

## Run run_encode.py
```
python -m DepressionProjectNew.run_encode DepressionProjectNew/output/data_cleaned.csv DepressionProjectNew/output/data_cleaned_encoded.csv
```
- Move output files into output folder (separated by age, include png's and etc)

## Run run_model_selection.py
- Use script

## Run run_analysis.py
- Use script

## Run univariate comparison
```
python -u -m DepressionProjectNew.run_univariate \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12to18_yesmental/balanced_accuracy \
    y12to18_Dep_YN_216m \
    --use-balanced-accuracy
```

## Run fix_embed_colors for age 12/if colors are switched for depressed/not depressed
```
python -u -m DepressionProjectNew.fix_embed_colors \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/ \
    y12CH_Dep_YN_144m
```

## Run run_tsne.py (don't need)

## Run make_readable_pcc_sc_kendall.py and make_readable_list.py after pasting in the best rfe list and lists from run_univariate's output

## Can also run model using current best results if low on time
```
python -u -m DepressionProjectNew.run_model_and_analysis ./DepressionProjectNew/output/output_18_yesmental/results.pkl ./DepressionProjectNew/output/output_12_yesmental/preprocessed ./DepressionProjectNew/output/data_cleaned_encoded_12_yesmental.csv ./DepressionProjectNew/output/output_12_yesmental y12CH_Dep_YN_144m
```
or use script

## Plot tsne using only best results from RFE/Elbow method (don't need)
Make sure to input the hardcoded variables for the rfe results
```
python -u -m DepressionProjectNew.run_tsne_use_rfe_results_all \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_17_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental \
    y12CH_Dep_YN_144m \
    y16CH_Dep_YN_192m \
    y17CH_Dep_YN_204m \
    y18CH_Dep_YN_216m
```

## Can also run analysis without rfe and calculate f1's (don't need)
```
python -u -m DepressionProjectNew.run_model_and_analysis_no_rfe_calc \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/results.pkl \
    ./DepressionProjectNew/output/output_18_yesmental/preprocessed \
    ./DepressionProjectNew/output/data_cleaned_encoded_18_yesmental_30MVI.csv \
    ./DepressionProjectNew/output/output_18_yesmental \
    y18CH_Dep_YN_216m
```

## Calculate F1 baselines and plot into confusion matrix (don't need)
```
python -u -m DepressionProjectNew.run_f1_calcs_baseline_all \
    ./DepressionProjectNew/output/10MVIout/output_12_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_16_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_17_yesmental \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental \
    y12CH_Dep_YN_144m \
    y16CH_Dep_YN_192m \
    y17CH_Dep_YN_204m \
    y18CH_Dep_YN_216m
```

## Plot F1's with their baseline (don't need)
```
python -u -m DepressionProjectNew.plot_f1_overall
    ./DepressionProjectNew/output/10MVIout/f1s.png
```