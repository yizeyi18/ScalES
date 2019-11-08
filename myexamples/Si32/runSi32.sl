#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J runSi32
#SBATCH -t 00:30:00
 

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 8 -c 8 --cpu_bind=cores ../../examples/pwdft -in Si32pwdft.in
