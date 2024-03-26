#!/bin/bash

# Numerical Resolution Study

rm -rf ./01_baseline_simulation/num_res/data/ && mkdir ./01_baseline_simulation/num_res/data/
rm -rf ./01_baseline_simulation/num_res/images/ && mkdir ./01_baseline_simulation/num_res/images/

SCRIPT=./ball_bounce.py

OUT_FILE=./01_baseline_simulation/num_res/data/num_res_15_output.dsv
FREQUENCY=15
X_POS_INITIAL=49
Y_POS_INITIAL=50
Z_POS_INITIAL=51
X_VEL_INITIAL=5.25
Y_VEL_INITIAL=4.9
Z_VEL_INITIAL=5.0
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=3_15
python $SCRIPT --output $OUT_FILE --frequency $FREQUENCY --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

OUT_FILE=./01_baseline_simulation/num_res/data/num_res_20_output.dsv
FREQUENCY=20
X_POS_INITIAL=49
Y_POS_INITIAL=50
Z_POS_INITIAL=51
X_VEL_INITIAL=5.25
Y_VEL_INITIAL=4.9
Z_VEL_INITIAL=5.0
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=3_20
python $SCRIPT --output $OUT_FILE --frequency $FREQUENCY --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

OUT_FILE=./01_baseline_simulation/num_res/data/num_res_25_output.dsv
FREQUENCY=25
X_POS_INITIAL=49
Y_POS_INITIAL=50
Z_POS_INITIAL=51
X_VEL_INITIAL=5.25
Y_VEL_INITIAL=4.9
Z_VEL_INITIAL=5.0
GRAVITY=9.81
BOX_SIDE_LENGTH=100
GROUP_ID=47bcda
RUN_ID=3_25
python $SCRIPT --output $OUT_FILE --frequency $FREQUENCY --xpos $X_POS_INITIAL --ypos $Y_POS_INITIAL --zpos $Z_POS_INITIAL --xvel $X_VEL_INITIAL --yvel $Y_VEL_INITIAL --zvel $Z_VEL_INITIAL --gravity $GRAVITY --box_side_length $BOX_SIDE_LENGTH --group $GROUP_ID --run $RUN_ID

python ./dsv_to_sina.py ./01_baseline_simulation/num_res/data ./01_baseline_simulation/num_res/data/num_res_output.sqlite