# DepressionProjectNew

## Install python 3.10
## Install MSAP
```
git clone https://github.com/asmyoo/MSAP.git
pip install -e ./MSAP
```

## Install requirements
```
pip install -r requirements.txt
```

## Get “preprocessed” data - run reformat_ml.py within src/preprocess
```
python reformat_ml.py
```

## Change configs
- Cleaning.py for % missing value imputation and make sure columns_ignored is empty
- Model_selecting.py for age_cutoff and column_dependent

## Run get_config_info.py within src/preprocess
```
python get_config_info.py
```
- Change preprocessing.py config categorical variables if needed (probably not)
- Change cleaning.py with columns_ignored to add mental health variables

## Run run_cleaner.py
```
python -u -m DepressionProjectNew.run_cleaner
```
- Make sure to not overwrite missing_value png’s and data_cleaned.csv

## Run run_encode.py
```
python -m DepressionProjectNew.run_encode DepressionProjectNew/output/data_cleaned.csv DepressionProjectNew/output/data_cleaned_encoded.csv
```

## Run run_model_selection.py
- Use script

## Run run_analysis.py
- Use script

## Run run_tsne.py

## Run make_readable_pcc_sc_kendall.py and make_readable_list.py after pasting in the best rfe list

## Can also run model using current best results if low on time
```
python -u -m DepressionProjectNew.run_model_and_analysis ./DepressionProjectNew/output/output_18_yesmental/results.pkl ./DepressionProjectNew/output/output_12_yesmental/preprocessed ./DepressionProjectNew/output/data_cleaned_encoded_12_yesmental.csv ./DepressionProjectNew/output/output_12_yesmental y12CH_Dep_YN_144m
```
or use script