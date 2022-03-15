# -*- coding: utf-8 -*-
"""From parsed logs, find best model + params

Authors:
    Arielle Yoo - asmyoo@ucdavis.edu

Todo:
    * does not work right now

"""

import logging
import sys
import numpy as np
import os
import pandas as pd

logging.getLogger(__file__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG)

def parse_final_result_from_parse_logs(
    results,
    scoring='f1_score'):
    results = results[results['classifier'] == 'svc']
    results = results.sort_values(by=[scoring], ascending=False)
    best = results.iloc[0]
    logging.info(f"Best Results: {best}")
    best_params = best['best_params']
    best_score = best[scoring]
    return best_params, best_score

def main():
    path_results = os.path.abspath(os.path.dirname(__file__)) + \
        "/final_result.csv"
    
    results = pd.read_csv(path_results)
    
    # TODO - write code for finding best results from final_result.csv
    best_params, best_score = parse_final_result_from_parse_logs(
        results,
        scoring='f1_score')

    logging.info(f"Best Params: {best_params}")
    print(f"Best Score: {best_score}")

if __name__ == '__main__':
    main()
