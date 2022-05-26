import pandas as pd
from msap.modeling.model_evaluation.statistics import get_selected_features
from msap.utils.plot import plot_rfe_line_from_metric_dict

if __name__ == "__main__":
    rfe_result = pd.read_csv(
        './output/old_results/10MVIout/output_12_yesmental/rfe_result.csv')
    print(rfe_result)
    plot_rfe_line_from_metric_dict(
        rfe_result, './output/old_results/10MVIout/output_12_yesmental/age12_10MVI.svg')
