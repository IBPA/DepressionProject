#!/bin/bash

# Data parameters
PATH_INPUT=data/preprocessed_dWt_stats_192m_train.csv
PATH_INPUT_TEST=data/preprocessed_dWt_stats_192m_test.csv
PATH_OUTPUTS_DIR=outputs/dep192mdo
COLUMN_TARGET=SMFQ_dep

# Model selection parameters
CLS_METHODS=rnn,lstm
OD_METHODS=none
MVI_METHODS=locf,nocb,simple
FS_METHODS=tsstandard,standard
OS_METHODS=resample
MAX_EPOCHS=1000
N_FOLDS=5
SCORING=f1


# Reproducibility parameters
RANDOM_STATE=42

python -m msap.ts.run_preprocess \
    $PATH_INPUT \
    $PATH_OUTPUTS_DIR \
    --column-target $COLUMN_TARGET \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --random-state $RANDOM_STATE \

python -m msap.ts.run_grid_search \
    $PATH_OUTPUTS_DIR \
    --column-target $COLUMN_TARGET \
    --cls-methods $CLS_METHODS \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --os-methods $OS_METHODS \
    --grid-search-n-splits $N_FOLDS \
    --grid-search-scoring $SCORING \
    --random-state $RANDOM_STATE \

python -m msap.ts.run_plot \
    $PATH_INPUT_TEST \
    $PATH_OUTPUTS_DIR \
    --column-target $COLUMN_TARGET \
    --cls-methods $CLS_METHODS \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --os-methods $OS_METHODS \
    --grid-search-n-splits $N_FOLDS \
    --grid-search-scoring $SCORING \
    --random-state $RANDOM_STATE \

python -m msap.ts.run_train_long \
    $PATH_INPUT_TEST \
    $PATH_OUTPUTS_DIR \
    --column-target $COLUMN_TARGET \
    --cls-methods $CLS_METHODS \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --os-methods $OS_METHODS \
    --max-epochs $MAX_EPOCHS \
    --grid-search-n-splits $N_FOLDS \
    --grid-search-scoring $SCORING \
    --random-state $RANDOM_STATE \
