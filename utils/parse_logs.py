import ast
from typing import List


def parse_non_grid_search_results(log_model: List[str]):
    """
    """
    f1_score = None

    for line in log_model[::-1]:
        if "INFO:root:f1 score: " in line:
            f1_score = float(line.split(': ')[-1].split(' +/- ')[0])
            break

    return f1_score


def parse_log_grid_search_results(log_model: List[str]):
    """
    """
    f1_score = None
    params = None

    for i, line in enumerate(log_model):
        if line == "INFO:root:rank: 1":
            f1_score = float(log_model[i + 2].split(': ')[-1]) # grabs f1 score
            params = ast.literal_eval(log_model[i + 18].split('INFO:root:')[-1]) # grabs params

            # Remove redundant prefix for param keys.
            params_new = {}
            for k, v in params.items():
                k_new = k.split('__')[-1]
                params_new[k_new] = v
            params = params_new
            break

    return f1_score, params


def parse_log(log: str):
    """
    """
    res = []
    log_lines = log.split('\n')
    log_models = []  # The lines of logs for each processed model.

    # Extracting log for each model processing.
    indices_start = []  # The starting points of each processed model' log.
    for i, line in enumerate(log_lines):
        if "INFO:root:Processing " in line and " combination..." in line:
            indices_start.append(i)
    if log_lines[-1] == "MANUAL_INFO_TIMEOUT_IGNORE_LAST_MODEL":
        pass
    else:
        indices_start = indices_start + [len(log_lines)]
    log_models.extend([log_lines[indices_start[i - 1]:indices_start[i]]
                       for i in range(1, len(indices_start))])

    for log_model in log_models:
        id_ = int(log_model[0].split(' ')[1].split('/')[0])
        (classifier_mode,
         scaling_mode,
         impute_mode,
         outlier_mode) = ast.literal_eval(log_model[1].split(':')[1].strip())

        if classifier_mode in ['decisiontreeclassifier',
                               'adaboostclassifier',
                               'randomforestclassifier',
                               'mlpclassifier']:
            score, params = parse_log_grid_search_results(log_model)
        else:
            score = parse_non_grid_search_results(log_model)
            params = None

        res.append(
            dict(
                model_id=id_,
                classifier=classifier_mode,
                scaling=scaling_mode,
                imputation=impute_mode,
                outlier=outlier_mode,
                f1_score=score,
                best_params=params))

    return res


if __name__ == '__main__':
    import os
    import pandas as pd

    path_logs_dir = (
        os.path.abspath(os.path.dirname(__file__))
        + "/../output/logs")

    path_logs = [(path_logs_dir + '/' + log)
                 for log in os.listdir(path_logs_dir) if log[-3:] == 'out']
    
    results = []
    for path_log in path_logs:
        with open(path_log, 'r') as f:
            log = f.read()
            results.extend(parse_log(log))

    pd.DataFrame(results).to_csv("final_result.csv", index=False)
