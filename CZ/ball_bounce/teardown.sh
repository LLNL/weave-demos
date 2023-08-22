#!/bin/bash

jupyter kernelspec uninstall -f ball-bounce-demo
rm -rf ball_bounce_demo_venv
rm -rf output*

rm -rf ./01_baseline_simulation/baseline/data/
rm -rf ./01_baseline_simulation/baseline/images/
rm -rf ./01_baseline_simulation/num_res/data/
rm -rf ./01_baseline_simulation/num_res/images/
rm -rf ./03_simulation_ensembles/data/
rm -rf ./04_manage_data/data/
rm -rf ./05_post-process_data/images/
rm -rf ./06_surrogate_model/images/

echo "Teardown complete! Virtual environment, kernel, and data removed."
