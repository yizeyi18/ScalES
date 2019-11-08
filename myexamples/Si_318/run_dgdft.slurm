#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --nodes=64
#SBATCH --time=00:30:00
#SBATCH --tasks-per-node=12
#SBATCH --constraint=haswell

module load craype-hugepages2M
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=2
export MPICH_USE_DMAPP_COLL=1
srun -n 768 -c 2 --cpu-bind=cores ../../examples/dgdft 
