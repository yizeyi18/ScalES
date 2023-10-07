export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/raid/software/fftw/icc/lib
mpirun --mca btl ^openib -np 20 /public3/home/fengjw/Davidson_pwdft/pwdft-CPU/aimd-cpu/examples


