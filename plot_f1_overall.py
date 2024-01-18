# -*- coding: utf-8 -*-
"""Plot F1 scores for different ages as bar graph
Must change info hard coded in main

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu
    Adapted from
    https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html

Todo:
    * maybe change to using seaborn instead and calculating f1s directly

python -u -m DepressionProject.plot_f1_overall
python -u -m DepressionProject.plot_f1_overall
./DepressionProject/output/10MVIout/f1s.png
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
    default="")
def main(
        path_save):
    """
    """
    f1_scores = [0.173, 0.217, 0.317, 0.33, 0.389, 0.517]
    # f1_scores_std = [0.022, 0.026, 0.013, 0.025]
    f1_bases = [0.093, 0.156, 0.27, 0.3, 0.316, 0.515]
    # f1_bases_std = [0.008, 0.023, 0.014, 0.01]
    ages = [12, 13, 16, 17, 18, "12 to 18"]

    ind = np.arange(len(ages))
    width = 0.25

    fig, ax = plt.subplots()
    # rects1 = ax.bar(ind-width/2, f1_bases, width, yerr = f1_bases_std, label = 'Baseline')
    # rects2 = ax.bar(ind+width/2, f1_scores, width, yerr = f1_scores_std, label = 'Best Model')
    rects1 = ax.bar(ind-width/2, f1_bases, width, label='Baseline')
    rects2 = ax.bar(ind+width/2, f1_scores, width, label='Best Model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Scores')
    ax.set_title('Scores by age with baseline')
    ax.set_xticks(ind)
    ax.set_xticklabels(('12', '13', '16', '17', '18', "12 to 18"))
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
