# read ppc and sc files and output readable versions that can
# used for slides
import pandas as pd
import numpy as np

DEFAULT_VARIABLE_INFO = '../../data/Variables013122new_full.csv'
DEFAULT_PREPROCESSED = '../../output/preprocessed_data_without_temporal.txt'
DEFAULT_PPC = '../../output/analyses/ppc.csv'
DEFAULT_SC = '../../output/analyses/sc.csv'
OUT_PPC = '../../output/analyses/ppc_readable.csv'
OUT_SC = '../../output/analyses/sc_readable.csv'

df_variable_info = pd.read_csv(DEFAULT_VARIABLE_INFO, dtype='str')
ppc = pd.read_csv(DEFAULT_PPC, dtype='str')
sc = pd.read_csv(DEFAULT_SC, dtype='str')

# if categorical, get new name
#categorical = df_variable_info.loc[df_variable_info['Categorical'] == '1', 'RelabeledName'].tolist()
relabeled = []
variable_description = []

mapper = dict(zip(
    list(df_variable_info['RelabeledName']),
    list(df_variable_info['Variable Label'])))

for name in ppc["Variable"]:
    if len(name.split("_")) == 3:
        label_in_variable_info = "_".join(name.split("_")[0:2])
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        if var_desc == None:
            print(f"Len of name is 3 but not found in mapper if sliced: {name}")
            continue
        relabeled.append(name)
        variable_description.append("_".join([var_desc, name.split("_")[-1]]))

mapper = dict(zip(
    list(df_variable_info['RelabeledName']) + relabeled,
    list(df_variable_info['Variable Label']) + variable_description))


ppc.replace({"Variable": mapper}, inplace=True)
#print(ppc["Variable"])
ppc.to_csv(OUT_PPC)
print(f"Shape of PCC where p-value <= 0.05: {ppc.loc[pd.to_numeric(ppc['P-value']) <= 0.05].shape}")

sc.replace({"Variable": mapper}, inplace=True)
#print(sc["Variable"])
sc.to_csv(OUT_SC)
print(f"Shape of SC where p-value <= 0.05: {sc.loc[pd.to_numeric(sc['P-value']) <= 0.05].shape}")