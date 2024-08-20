#!/bin/bash
#SBATCH -N 2
#SBATCH -J Merlin
#SBATCH -t 01:00:00
#SBATCH -p pbatch
#SBATCH -o merlin_workers_%j.out

#1. activate virtual env
#2. merlin run <yaml>
#3. sbatch ebox_workers.sbatch

# Turn off core files to work aroung flux exec issue.
ulimit -c 0

YAML=rayleigh_taylor_lc_build_venv_merlin.yaml

# VENV=/usr/apps/weave/weave-develop-cpu/
VENV=/usr/workspace/${USER}/virtualenvs/merlin_py_3_10
# VENV=/usr/workspace/${USER}/virtualenvs/test_venv

# Activate the virtual environment
source ${VENV}/bin/activate

#########################################
#          Running the Workers          #
#########################################

# Show the workers command
merlin run-workers ${YAML} --echo

# Start workers to run the tasks in the broker
merlin run-workers ${YAML}

# Keep the allocation alive until all workers stop
merlin -lvl debug monitor ${YAML}
# sleep inf
