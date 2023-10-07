Software installation

1) Load mpi and gcc
1.Load mpi module (The 2017 version of intelmpi is the mpi version corresponding to the article test and is recommended).Or change CC/CXX/FC and LOADER in make.inc to the corresponding paths.
2.Load gcc(gcc7.3.0 is recommended)



2) Modify the make.inc file
1.Change DGDFT_DIR in make.inc to the root directory of PWDFT (/YourPath/PWDFTSUB).
2.Change FFTW_DIR, MKL_ROOT, LAPACK_LIB, and LIBXC_DIR in make.inc to the corresponding fftw, mkl, lapack, and libxc directories(fftw-3.3.8, mkl-2017, lapack-3.9.0, libxc-5.2.3 is recommended)

3)Execute compilation
1.sh install.sh

———————————————————————————————————————————————————————————————————————————————————————

Software usage method

You can run the pwdft executable directly (single core) or use mpirun to run the PWDFT executable, specifying the input file with the -in argument.

You can perform pwdft using mpirun directly in the examples directory, where you already have Si64.in test examples and corresponding pseudopotentials HGH_Si.bin.
(mpirun -np 4 /YourPath/PWDFTSUB/examples/pwdft -in /YourPath/PWDFTSUB/examples/Si64.in)
