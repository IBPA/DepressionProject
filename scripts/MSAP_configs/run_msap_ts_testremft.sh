#!/bin/bash

# Data parameters
PATH_INPUT=examples/basicmotions_0.1_0.05_ts_removedft.csv
PATH_OUTPUTS_DIR=outputs/example
COLUMN_TARGET=class

# Model selection parameters
CLS_METHODS=tsf
OD_METHODS=iforest,none
MVI_METHODS=locf,nocb,simple
FS_METHODS=tsstandard,standard,none
OS_METHODS=resample,none
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
