#!/bin/bash

set -e

# Data parameters
PATH_INPUT=examples/iris_0.1_0.05_train.csv
PATH_INPUT_TEST=examples/iris_0.1_0.05_test.csv
PATH_OUTPUTS_DIR=outputs/example_iris
COLUMN_TARGET=class

# Model selection parameters
CLS_METHODS=svc,rf,ada,nb,mlp
OD_METHODS=lof,iforest,none
MVI_METHODS=mforest,simple
FS_METHODS=minmax,standard,none
OS_METHODS=smote,none
N_FOLDS=5
SCORING=f1

# Reproducibility parameters
RANDOM_STATE=42

python -m msap.run_preprocess \
    $PATH_INPUT \
    $PATH_OUTPUTS_DIR \
    --column-target $COLUMN_TARGET \
    --od-methods $OD_METHODS \
    --mvi-methods $MVI_METHODS \
    --fs-methods $FS_METHODS \
    --random-state $RANDOM_STATE \

python -m msap.run_grid_search \
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

python -m msap.run_plot \
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
