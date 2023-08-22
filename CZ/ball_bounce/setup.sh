#!/bin/bash

python3 -m venv ball_bounce_demo_venv
source ball_bounce_demo_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name ball-bounce-demo

# Creating folders for data and images

mkdir ./01_baseline_simulation/baseline/data/
mkdir ./01_baseline_simulation/baseline/images/
mkdir ./01_baseline_simulation/num_res/data/
mkdir ./01_baseline_simulation/num_res/images/
mkdir ./03_simulation_ensembles/data/
mkdir ./04_manage_data/data/
mkdir ./05_post-process_data/images/
mkdir ./06_surrogate_model/images/


echo "Setup complete!"
