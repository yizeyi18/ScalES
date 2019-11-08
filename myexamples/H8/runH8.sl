#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=10
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=1

srun  ../../examples/dgdft -in ./dgdft.in
