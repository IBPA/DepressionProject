# -*- coding: utf-8 -*-
"""Plot F1 scores for different mvis as bar graph
Must change info hard coded in main

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu
    Adapted from
    https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html

Todo:
    * maybe change to using seaborn instead and calculating f1s directly

python -u -m DepressionProjectNew.plot_f1_mvis
python -u -m DepressionProjectNew.plot_f1_mvis
./DepressionProjectNew/output/f1s_mvi.png
"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import click
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

@click.command()
@click.argument(
    'path-save',
    type=str,
    default = "")
def main(
        path_save):
    """
    """
    f1_scores = [0.36, 0.376, 0.325, 0.333, 0.326, 0.324, 0.322, 0.316, 0.323, 0.318]
    f1_scores_std = [0.019, 0.025, 0.019, 0.028, 0.036, 0.035, 0.034, 0.032, 0.029, 0.036]
    f1_bases = [0.314, 0.314, 0.314, 0.314, 0.314, 0.314, 0.314, 0.314, 0.314, 0.314]
    f1_bases_std = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    mvis = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    ind = np.arange(len(mvis))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind-width/2, f1_bases, width, yerr = f1_bases_std, label = 'Baseline')
    rects2 = ax.bar(ind+width/2, f1_scores, width, yerr = f1_scores_std, label = 'Best 10% MVI Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Scores')
    ax.set_title('Scores by % MVI Cutoff with Baseline')
    ax.set_xticks(ind)
    ax.set_xticklabels(('5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%'))
    ax.legend()

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')


    autolabel(rects1, "left")
    autolabel(rects2, "right")

    fig.tight_layout()

    if path_save != "":
        plt.savefig(path_save)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    main()