#!/bin/bash
#SBATCH -J aimd
#SBATCH -N 128
#SBATCH -n 512
#SBATCH --ntasks-per-node=4
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
#SBATCH -p normal
#SBATCH --mem=90G
#SBATCH --gres=dcu:4
##SBATCH -t 1000
##SBATCH -x 

module purge
module add compiler/devtoolset/7.3.1  mpi/hpcx/2.7.4/gcc-7.3.1 compiler/rocm/3.3
source /opt/hpc/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/home/hw45888792/lijl/lib/FFTW/gcc/lib

srun hostname | sort -u > nd

sed -i 's/$/ slots=28/g' nd

NNODE=$(wc -l nd | awk '{print $1}' )
NP=$[1*NNODE]

APP=/public/home/hw45888792/jiaosz/soft/aimd-dcu/examples/pwdft
srun --mpi=pmix_v3 $APP
