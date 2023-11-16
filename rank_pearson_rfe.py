"""Create rankings using feature_selection_corr_readable.csv
    for each age (from run_univariate, data preprocessing for best model
    is completed). Similar to figure 2d in the paper.

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu
    
"""
import pandas as pd
import click

from .run_univariate import make_readable


@click.command()
@click.argument(
    'path-input-dir',
    type=click.Path(exists=True))
def main(
        path_input_dir):
    ages = ['12', '13', '16', '17', '18', '12to18']
    # targets = ['y12CH_Dep_YN_144m', 'y12to18_Dep_YN_216m', 'y13CH_Dep_YN_162m',
    #            'y16CH_Dep_YN_192m', 'y17CH_Dep_YN_204m', 'y18CH_Dep_YN_216m']
    # columns = ['Variable', 'RFE Index', 'PearsonCorr Rank',
    #            'pearson_corr', 'pearson_pvalue', 'spearman_corr', 'spearman_pvalue']
    df_ages = []
    rfe_vars = []
    for i, age in enumerate(ages):
        if age == '12to18':
            df_age = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/f1/feature_selection_corr_readable.csv")
            # get rfe selected variables from file
            rfe = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/f1/pc_rank_pearson_rfe_train.csv")
        else:
            df_age = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/feature_selection_corr_readable.csv")
            # get rfe selected variables from file
            rfe = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/pc_rank_pearson_rfe_train.csv")
        rfe.rename(columns={'Unnamed: 0': 'Variable'}, inplace=True)
        rfe_vars_age = rfe['Variable'].tolist()
        rfe_readable = make_readable(rfe_vars_age)
        rfe_vars.append(rfe_readable)
        # drop univariate column
        df_age = df_age.drop(columns=['Univariate Index'])
        df_age['PearsonCorr Rank'] = df_age['pearson_corr'].abs().rank(
            ascending=False)
        # rename all columns to include age except for Variable
        df_age = df_age.rename(
            columns={col: f"{col}_{age}" for col in df_age.columns if col != 'Variable'})
        df_ages.append(df_age)
    # merge all ages on Variable
    df_merged = df_ages[0]
    for df_age in df_ages[1:]:
        df_merged = df_merged.merge(df_age, on='Variable', how='outer')

    # get ave rank
    pearson_rank_cols = [
        col for col in df_merged.columns if 'PearsonCorr Rank' in col]
    df_merged['ave_Pearson_rank'] = df_merged[pearson_rank_cols].mean(axis=1)

    # get rfe rank for variables selected by rfe
    for i, age in enumerate(ages):
        # mask
        mask = df_merged['Variable'].isin(rfe_vars[i])
        # get column name
        col_name = f"RFE Index_{age}"
        # get rfe index
        df_merged[f"masked_rfe_rank_{age}"] = df_merged.loc[mask, col_name] + 1
    # average the masked rfe index
    rfe_index_cols = [
        col for col in df_merged.columns if 'masked_rfe_rank' in col]
    df_merged['ave_rfe_rank'] = df_merged[rfe_index_cols].mean(axis=1)

    # get number of time selected by rfe
    df_merged['rfe_count'] = df_merged[rfe_index_cols].count(axis=1)

    # sort by rfe_count
    df_merged = df_merged.sort_values(by=['rfe_count'], ascending=False)

    # save
    df_merged.to_csv(
        f"{path_input_dir}/rank_pearson_rfe_allinfo.csv", index=False)

    # save only important columns
    cols = ['Variable', 'rfe_count', 'ave_rfe_rank']
    cols.extend([col for col in df_merged.columns if 'masked_rfe_rank' in col])
    cols.append('ave_Pearson_rank')
    cols.extend([col for col in df_merged.columns if 'PearsonCorr Rank' in col])
    df_merged[cols].to_csv(
        f"{path_input_dir}/rank_pearson_rfe.csv", index=False)


if __name__ == '__main__':
    main()
