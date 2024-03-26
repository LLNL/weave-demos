#!/bin/bash

# Baseline Simulation Study

rm -rf ./01_baseline_simulation/baseline/data/ && mkdir ./01_baseline_simulation/baseline/data/
rm -rf ./01_baseline_simulation/baseline/images/ && mkdir ./01_baseline_simulation/baseline/images/

SCRIPT=./ball_bounce.py

OUT_FILE=./01_baseline_simulation/baseline/data/baseline_0_output.dsv
X_POS_INITIAL=49.37
Y_POS_INITIAL=48.71
Z_POS_INITIAL=51.25
X_VEL_INITIAL=5.22
Y_VEL_INITIAL=4.83
Z_VEL_INITIAL=5.13
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=0
python $SCRIPT --output $OUT_FILE --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

OUT_FILE=./01_baseline_simulation/baseline/data/baseline_1_output.dsv
X_POS_INITIAL=50
Y_POS_INITIAL=50
Z_POS_INITIAL=50
X_VEL_INITIAL=5
Y_VEL_INITIAL=5
Z_VEL_INITIAL=5
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=1
python $SCRIPT --output $OUT_FILE --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

OUT_FILE=./01_baseline_simulation/baseline/data/baseline_2_output.dsv
X_POS_INITIAL=51
Y_POS_INITIAL=49
Z_POS_INITIAL=50
X_VEL_INITIAL=5.5
Y_VEL_INITIAL=4.9
Z_VEL_INITIAL=5.1
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=2
python $SCRIPT --output $OUT_FILE --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

OUT_FILE=./01_baseline_simulation/baseline/data/baseline_3_output.dsv
X_POS_INITIAL=49
Y_POS_INITIAL=50
Z_POS_INITIAL=51
X_VEL_INITIAL=5.25
Y_VEL_INITIAL=4.9
Z_VEL_INITIAL=5.0
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=3
python $SCRIPT --output $OUT_FILE --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

python ./dsv_to_sina.py ./01_baseline_simulation/baseline/data ./01_baseline_simulation/baseline/data/baseline_output.sqlite