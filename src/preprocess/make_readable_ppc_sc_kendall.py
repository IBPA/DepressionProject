# read ppc and sc files and output readable versions that can
# used for slides
import pandas as pd
import numpy as np

DEFAULT_VARIABLE_INFO = '../../data/Variables052122.csv'
# DEFAULT_PREPROCESSED = '../../output/preprocessed_data_without_temporal_12to18ave.csv'
# DEFAULT_PPC = '../../output/output_18_yesmental/pc_rank_pearson.csv'
# DEFAULT_SC = '../../output/output_18_yesmental/pc_rank_spearman.csv'
# DEFAULT_K = '../../output/output_18_yesmental/pc_rank_kendall.csv'
# OUT_PPC = '../../output/output_18_yesmental/pc_rank_pearson_readable.csv'
# OUT_SC = '../../output/output_18_yesmental/pc_rank_spearman_readable.csv'
# OUT_K = '../../output/output_18_yesmental/pc_rank_kendall_readable.csv'
DEFAULT_PPC = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_pearson_train.csv'
DEFAULT_SC = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_spearman_train.csv'
DEFAULT_K = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_kendall_train.csv'
OUT_PPC = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_pearson_train_readable.csv'
OUT_SC = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_spearman_train_readable.csv'
OUT_K = '../../output/pval_filter_60_MVI/output_17_yesmental/pc_rank_kendall_train_readable.csv'

df_variable_info = pd.read_csv(
    DEFAULT_VARIABLE_INFO, dtype='str', encoding='unicode_escape')
ppc = pd.read_csv(DEFAULT_PPC, dtype='str', encoding='unicode_escape')
sc = pd.read_csv(DEFAULT_SC, dtype='str', encoding='unicode_escape')
kendall = pd.read_csv(DEFAULT_K, dtype='str', encoding='unicode_escape')

# if categorical, get new name
# categorical = df_variable_info.loc[df_variable_info['Categorical'] == '1', 'RelabeledName'].tolist()
relabeled = []
variable_description = []
unfound = []

# make variable description
descriptions = ["{} ({})".format(a_, b_) for a_, b_ in zip(list(
    df_variable_info['Variable Label'].str.strip()), list(df_variable_info['Coding_details'].str.strip()))]

mapper = dict(zip(
    list(df_variable_info['RelabeledName'].str.strip()),
    descriptions
))
#print('y10CH_Dep_127m' in list(df_variable_info['RelabeledName']))

for name in ppc['Unnamed: 0']:
    if len(name.split("_")) > 2:
        label_in_variable_info = "_".join(name.split("_")[0:2])
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        if var_desc == None:
            print(
                f"Len of name is >=3 but not found in mapper if sliced: {name}")
            unfound.append(name)
            continue
        relabeled.append(name)
        variable_description.append(" ".join([var_desc, name]))
    else:
        label_in_variable_info = name
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        # if var_desc == None:
        #     print(
        #         f"Len of name is <=2 but not found in mapper if sliced: {name}")
        #     unfound.append(name)
        #     continue
        relabeled.append(name)
        variable_description.append(" ".join([var_desc, name]))

for name in unfound:  # assume these are in info somewhere
    print(f"Looking for: {name}")
    relabeled.append(name)
    label_in_variable_info = name
    var_map = map(mapper.get, [label_in_variable_info])
    var_desc = list(var_map)[0]
    variable_description.append(" ".join([var_desc, name]))

mapper = dict(zip(
    relabeled,
    variable_description))


ppc.replace({'Unnamed: 0': mapper}, inplace=True)
# print(ppc["Variable"])
ppc.to_csv(OUT_PPC, index=False)
print(
    f"Shape of PCC where p-value < 0.05: {ppc.loc[pd.to_numeric(ppc['p-value']) < 0.05].shape}")

sc.replace({'Unnamed: 0': mapper}, inplace=True)
# print(sc["Variable"])
sc.to_csv(OUT_SC, index=False)
print(
    f"Shape of SC where p-value < 0.05: {sc.loc[pd.to_numeric(sc['p-value']) < 0.05].shape}")

kendall.replace({'Unnamed: 0': mapper}, inplace=True)
# print(sc["Variable"])
kendall.to_csv(OUT_K, index=False)
print(
    f"Shape of Kendall where p-value < 0.05: {sc.loc[pd.to_numeric(sc['p-value']) < 0.05].shape}")
