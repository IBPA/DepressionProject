import os


class DefaultConfig:
    PATH_OUTPUT_REMOVED = (os.path.abspath(os.path.dirname(__file__))
                           + "/../output/overly_missing_data.csv")
    THRESHOLD_DROP_MISSING_RATIO = 0.6
    COLUMNS_IGNORED = ['cidB2846_0m']
    # 'MR_CH_SMFQDepressionScoreComplete_9y',
    # 'MR_CH_SMFQDepressionScoreProrated_9y',
    # 'CHR_CH_Depression_10y',
    # 'CHR_CH_DepressionYN_10y',
    # 'MR_CH_SMFQDepressionScoreComplete_11y',
    # 'MR_CH_Depression_11y',
    # 'MR_CH_DepressionYN_11y',
    # 'CHR_CH_Depression_12y',
    # 'CHR_CH_DepressionYN_12y',
    # 'MR_CH_Depression_13y',
    # 'CHR_CH_Depression_13y',
    # 'MR_CH_DepressionYN_13y',
    # 'CHR_CH_DepressionYN_13y',
    # 'CHR_CH_Depression_16y',
    # 'CHR_CH_DepressionYN_16y',
    # 'CHR_CH_DepressionMildYN_17y',
    # 'CHR_CH_DepressionModYN_17y',
    # 'CHR_CH_DepressionSevYN_17y',
    # 'CHR_CH_Depression_18y']
    # 'CHR_CH_FeltUnloved_18y',
    # 'CHR_CH_FeltTheyCouldNeverBeAsGood_18y',
    # 'CHR_CH_FeltTheyDidEverythingWrong_18y']
    DIFFERENT_FORMAT = [
        '_CH_BirthOrder_0d_A',
        '_CH_BirthOrder_0d_B']  # used for RFE
    CORRELATION_FILE = (os.path.abspath(os.path.dirname(
        __file__)) + "/../output/vars_sorted_dir.csv")
