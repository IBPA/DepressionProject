import logging

import numpy as np


def parse_cv_results(
        results,
        report_score_using='f1',
        scoring=['f1', 'average_precision'],
        count=5):
    """
    """
    #logging.info(f"All results: f{results}")
    if isinstance(scoring, str):
        scoring = ['score']
        report_score_using = 'score'

    for score in scoring:
        logging.info(f"Scoring results of {score}")

        for rank in range(1, count + 1):
            runs = np.flatnonzero(results[f'rank_test_{score}'] == rank)

            for run in runs:
                logging.info(f"rank: {rank}")
                #logging.info(f"score: {results[f'mean_test_{score}'][run]}")
                #logging.info(f"train score: {results[f'mean_train_{score}'][run]}")
                #logging.info(f"std: {results[f'std_test_{score}'][run]}")
                #logging.info(f"train std: {results[f'std_train_{score}'][run]}")
                logging.info("scores:")
                for each_score in scoring:
                    logging.info(f"{each_score} score: {results[f'mean_test_{each_score}'][run]}")
                    logging.info(f"{each_score} train score: {results[f'mean_train_{each_score}'][run]}")
                    logging.info(f"{each_score} std: {results[f'std_test_{each_score}'][run]}")
                    logging.info(f"{each_score} train std: {results[f'std_train_{each_score}'][run]}")
                logging.info(results['params'][run])

            if (report_score_using == score) and (rank == 1):
                best_params = results['params'][run]
                best_score = results[f'mean_test_{score}'][run]

    return best_params, best_score
