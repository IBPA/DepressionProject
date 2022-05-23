import numpy as np
import pandas as pd
import pickle

#from utils import logging
#logger = logging.set_logging(__name__)

DEFAULT_VARIABLE_INFO = '../../data/Variables052122.csv'
#DEFAULT_CLEANED = '../../output/data_cleaned.pkl'
#DEFAULT_CLEANED = '../../output/data_cleaned.csv'
DEFAULT_CLEANED = '../../output/preprocessed_data_without_temporal.txt'

#cleaned_data = pickle.load(open(DEFAULT_CLEANED, 'rb'))
cleaned_data = pd.read_csv(
    DEFAULT_CLEANED, dtype='str', encoding='unicode_escape')

# Predicting for y18CH_Dep_YN_216m - change model_selecting.py COLUMN_DEPENDENT
# make sure to change age cutoff too

# get categorical - change preprocessing.py columns_categorical
df_var_info = pd.read_csv(DEFAULT_VARIABLE_INFO,
                          dtype='str', encoding='utf-8-sig')
# df_var_info = df_var_info.applymap(lambda x: x.strip())  # remove white space
print(
    f"Categorical Info is null: {df_var_info[df_var_info['Categorical'].isnull()]}")
categorical = df_var_info.loc[df_var_info['Categorical']
                              == '1', 'RelabeledName'].tolist()
categorical = [x for x in categorical if x in cleaned_data.columns]
print(f"Columns Categorical: {categorical}")

# get variables to ignore if ignoring mental health variables
# change cleaning.py COLUMNS_IGNORED
#print(f"{df_var_info['Child Mental Health Variable (1=Yes, 0 =No) '].unique()}")
# print(f"{df_var_info.columns}")
# print(f"{cleaned_data.columns}")
print(
    f"Mental Health is null: {df_var_info[df_var_info['Child Mental Health Variable (1=Yes, 0 =No) '].isnull()]}")
mental_health = df_var_info.loc[df_var_info['Child Mental Health Variable (1=Yes, 0 =No) '] == '1', 'RelabeledName'].tolist(
)
mental_health = [x for x in mental_health if x in cleaned_data.columns]
print(f"Mental Health: {mental_health}")
