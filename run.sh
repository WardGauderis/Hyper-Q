#!/bin/bash

# This file needs to be run with sbatch!!!

#SBATCH --job-name=Hyper-Q
#SBATCH --ntasks=1

# Number of CPUs, 20 seems to be a good compromise.

#SBATCH --cpus-per-task=20

# 100_000 iterations takes 5min for 20 runs
# 1_500_000 iterations take 1h15 for 20 runs
# Means in 5 days we can run 96 configs for 1_500_000 iterations and 20 runs

#SBATCH --time=01-00:00:00

#SBATCH --mail-user=fabian.luc.m.denoodt@vub.be
#SBATCH --mail-type=ALL

# Load the modules required.

module load buildenv/default-foss-2022a
module load Python/3.10.4-GCCcore-11.3.0
module load CMake/3.24.3-GCCcore-11.3.0

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads

# We should either use SCRATCH (90Gb and fast but is temporary) or DATA (50Gb and permanent)

rm -r $VSC_DATA/Hyper-Q2
cp -r Hyper-Q $VSC_DATA/Hyper-Q2

cd $VSC_DATA/Hyper-Q2
python scripts/test.py
