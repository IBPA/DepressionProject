"""
    Get top 10 RFE selected features for different ages

    python -u -m DepressionProject.get_top_10_rfe \
        ./DepressionProject/output/pval_filter_60_MVI/Supplementary\ Spreadsheet\ 3.xlsx \
        ./DepressionProject/output/pval_filter_60_MVI/top_10_rfe.csv
    
    Authors:
        Arielle Yoo - asmyoo@ucdavis.edu
"""

import click
import pandas as pd


@click.command()
@click.argument(
    'excel_rfe_fts',
    type=click.Path(exists=True))
@click.argument(
    'output_path',
    type=click.Path())
def main(
        excel_rfe_fts,
        output_path):

    # Load data.
    rfe_excel = pd.ExcelFile(excel_rfe_fts)
    # get all ages RFE selected features
    ages = ['Dep12', 'Dep13', 'Dep16', 'Dep17', 'Dep18', 'Dep12-18']
    rfe_vars = []
    for age in ages:
        rfe = pd.read_excel(rfe_excel, sheet_name=age)
        rfe_vars_age = rfe['Variable'].tolist()[:10]
        rfe_vars.append(rfe_vars_age)

    # create table with feature names as index and ages as columns
    rfe_data = {}
    # get all features
    features = []
    for rfe_vars_age in rfe_vars:
        features.extend(rfe_vars_age)
    features = list(set(features))
    # create table
    for feature in features:
        rfe_data[feature] = [0] * len(ages)
    # fill in table with index
    for i in range(len(ages)):
        for j in range(len(rfe_vars[i])):
            rfe_data[rfe_vars[i][j]][i] = j + 1
    # create dataframe
    rfe_df = pd.DataFrame(rfe_data, index=ages).T
    # print(rfe_df)
    # rfe_df.to_csv(output_path)
    # sort by average rank
    rfe_df['Average Rank'] = rfe_df.mean(axis=1)
    rfe_df = rfe_df.sort_values(by=['Average Rank'])
    rfe_df = rfe_df.drop(columns=['Average Rank'])
    # print(rfe_df)
    rfe_df.to_csv(output_path)


if __name__ == '__main__':
    main()
