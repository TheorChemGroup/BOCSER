#!/usr/bin/bash

#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=conf_search_1th
export KMP_STACKSIZE=10G
export OMP_STACKSIZE=10G
export OMP_NUM_THREADS=1,1
export MKL_NUM_THREADS=1

python bocser/conf_search.py --config=config.yaml
