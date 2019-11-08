#!/bin/bash -l
#SBATCH --qos debug
#SBATCH --nodes=12
#SBATCH --tasks-per-node=24
#SBATCH --constraint=haswell
#SBATCH -t 00:30:00

module load craype-hugepages2M
export OMP_NUM_THREADS=1
export MPICH_USE_DMAPP_COLL=1

srun -n 288 -c 1 ../../examples/dgdft
