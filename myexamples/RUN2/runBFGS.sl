#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J runSi32bfgs
#SBATCH -t 00:30:00
 

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 32 -c 2 --cpu_bind=cores ../../examples/pwdft -in pwdft.bfgs.in
