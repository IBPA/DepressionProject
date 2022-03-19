# read ppc and sc files and output readable versions that can
# used for slides
import pandas as pd
import numpy as np

DEFAULT_VARIABLE_INFO = '../../data/Variables013122new.csv'
DEFAULT_PREPROCESSED = '../../output/preprocessed_data_without_temporal.txt'

df_variable_info = pd.read_csv(DEFAULT_VARIABLE_INFO, dtype='str', encoding = 'unicode_escape')

# if categorical, get new name
#categorical = df_variable_info.loc[df_variable_info['Categorical'] == '1', 'RelabeledName'].tolist()
want_readable = ['c601_32wg', 'Max_ed_32wg', 'kb010_6m', 'kd510a_18m', 'Adaptability_24m', 'Avg_FinDiff_61m', 'kz021_0m_1.0_0_2.0_1', 'f020a_8m_1.0_0_2.0_1']
relabeled = []
variable_description = []
unfound = []

mapper = dict(zip(
    list(df_variable_info['RelabeledName']),
    list(df_variable_info['Variable Label'])))

for name in want_readable:
    if len(name.split("_")) > 2:
        label_in_variable_info = "_".join(name.split("_")[0:2])
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        if var_desc == None:
            print(f"Len of name is 3 but not found in mapper if sliced: {name}")
            unfound.append(name)
            continue
        relabeled.append(name)
        variable_description.append(" ".join([var_desc, name]))
    else:
        relabeled.append(name)
        label_in_variable_info = name
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        variable_description.append(" ".join([var_desc, name]))

for name in unfound: # assume these are in info somewhere
    print(f"Looking for: {name}")
    relabeled.append(name)
    label_in_variable_info = name
    var_map = map(mapper.get, [label_in_variable_info])
    var_desc = list(var_map)[0]
    variable_description.append(" ".join([var_desc, name]))

mapper = dict(zip(
    relabeled,
    variable_description))

result = [mapper.get(item,item) for item in want_readable]
print(f"{result}")