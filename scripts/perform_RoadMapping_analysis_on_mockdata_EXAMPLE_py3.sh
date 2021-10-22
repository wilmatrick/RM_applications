#!/bin/bash -l
# Version: 2021-October-20

# Initial working directory:
#SBATCH -D /u/twilm/research/RM_applications/py

# Job Name:
#SBATCH -J 3_RoadMapping

# Number of nodes and MPI tasks per node:
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ../scripts/run3.log
#SBATCH -e ../scripts/run3.err
# SBATCH --exclude=freya???

# for OpenMP:
#SBATCH --cpus-per-task=40

# Notification:
#SBATCH --mail-type=all
#SBATCH --mail-user=trick@mpa-garching.mpg.de

# Wall clock limit:
#SBATCH --time=24:00:00

# Run the program:

conda activate py3_env

cd /u/twilm/research/RM_applications/py/

export OMP_NUM_THREADS=1

python perform_RoadMapping_analysis_on_mockdata_EXAMPLE_py3.py