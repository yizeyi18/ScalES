#!/bin/bash -l
#SBATCH --job-name Si
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --time=00:30:00
#SBATCH -C haswell  
cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
export OMP_NUM_THREADS=1
srun -n 8 -c 1 /global/homes/w/weihu/project_weihu/jielanli/ScalES/scales_test/build_scales_cpu/src/pwdft/pwdft
squeue -l -j $SLURM_JOB_ID
