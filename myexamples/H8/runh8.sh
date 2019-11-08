# srun --nodes=1 --ntasks-per-node=4 -c 4 --time=10 ../../examples/dgdft -in dgdft.in
srun --nodes=1 --ntasks-per-node=4 -c 4 --time=20 nvprof -o out.%q{OMPI_COMM_WORLD_RANK}.nvvp ../../examples/dgdft -in dgdft.in
