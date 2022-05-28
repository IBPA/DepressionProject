import pandas as pd
import numpy as np

if __name__ == '__main__':
    raw_data = pd.read_csv(
        f"../../output/preprocessed_data_without_temporal.txt", dtype='str')
    raw_data = raw_data.applymap(
        lambda x: x.strip() if isinstance(x, str) else x)

    # print(raw_data)
    # y12CH_Dep_YN_144m
    # y13CH_Dep_YN_162m
    # y16CH_Dep_YN_192m
    # y17CH_Dep_YN_204m
    # y18CH_Dep_YN_216m
    # 'y12to18_Dep_YN_216m'
    # y12CH_Dep_144m
    # y13CH_Dep_162m
    # y16CH_Dep_192m
    # y17CH_Dep_204m
    # y18CH_Dep_216m
    # 'y12to18_Dep_Ave_216m'

    dep_YN = ['y12CH_Dep_YN_144m', 'y13CH_Dep_YN_162m',
              'y16CH_Dep_YN_192m', 'y17CH_Dep_YN_204m', 'y18CH_Dep_YN_216m']
    dep_scalar = ['y12CH_Dep_144m', 'y13CH_Dep_162m',
                  'y16CH_Dep_192m', 'y17CH_Dep_204m', 'y18CH_Dep_216m']
    df_dep_scalar = raw_data[dep_scalar].dropna(
        how='any')[dep_scalar].astype('float')

    # only take mean of columns where all dep scalars exist
    raw_data['y12to18_Dep_Ave_216m'] = df_dep_scalar.dropna(
        how='any').mean(1)

    df_dep_YN = raw_data[dep_YN].astype('float')
    for index, row in df_dep_YN.iterrows():
        if row[dep_YN].notnull().values.all():
            # set to max because values for all cols exist
            raw_data.loc[index, 'y12to18_Dep_YN_216m'] = max(row[dep_YN])
        else:
            if row[dep_YN].notnull().values.any() and np.nanmax(row[dep_YN]) > 0:
                # if there is any data and nanmax is not 0
                # set value to max (1)
                raw_data.loc[index, 'y12to18_Dep_YN_216m'] = np.nanmax(
                    row[dep_YN])
            else:
                # NaN because all do not exist or sum is 0
                raw_data.loc[index, 'y12to18_Dep_YN_216m'] = np.NaN

    raw_data.to_csv(
        f'../../output/preprocessed_data_without_temporal_12to18.csv')
