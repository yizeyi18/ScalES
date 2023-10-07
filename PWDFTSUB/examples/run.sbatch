BATCH -J wrf
#SBATCH --comment=WRF
#SBATCH -n 128
#SBATCH --ntasks-per-node=32
#SBATCH -p kshctest
#SBATCH -o %j
#SBATCH -e %j

mpirun -np 128 ./pwdft
