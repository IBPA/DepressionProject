"""
    Plot comparison of RFE selected features for different ages

    python -u -m DepressionProject.plot_rfe_jaccard \
        ./DepressionProject/output/pval_filter_60_MVI/Supplementary\ Spreadsheet\ 3.xlsx \
        ./DepressionProject/output/pval_filter_60_MVI/rfe_jaccard.svg
    
    Authors:
        Arielle Yoo - asmyoo@ucdavis.edu
"""

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def jaccard_similarity(list1, list2):
    """
    Calculate Jaccard similarity between two lists

    Args:
        list1 (list): first list
        list2 (list): second list

    Returns:
        float: Jaccard similarity between two lists
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


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
        rfe_vars_age = rfe['Variable'].tolist()
        rfe_vars.append(rfe_vars_age)
    # calculate jaccard similarity between each pair of ages and put in heatmap
    jaccard = []
    for i in range(len(ages)):
        jaccard_age = []
        for j in range(len(ages)):
            jaccard_age.append(jaccard_similarity(rfe_vars[i], rfe_vars[j]))
        jaccard.append(jaccard_age)
    jaccard = pd.DataFrame(jaccard, index=ages, columns=ages)
    # print(jaccard)
    # plot heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(jaccard, annot=True, cmap='Blues', ax=ax)
    ax.set_title(
        'Jaccard similarity between RFE selected features', fontsize=20)
    ax.set_xlabel('Age', fontsize=15)
    ax.set_ylabel('Age', fontsize=15)
    # increase font size of x and y tick labels
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    main()
