## Installation

The program is built within PWDFT (Plane Wave Density Functional Theory) software package. It is written mainly in c++ with message passing interface (MPI) for parallel design. PWDFT depends on several external numerical libraries (BLAS, LAPACK, ScaLAPACK, FFTW and Libxc), which you should install first:

Libxc:  https://www.tddft.org/programs/libxc

FFTW 3.3.8 (or later): https://www.fftw.org/

We strongly suggest you to use Intel MKL (Math Kernel Library) to install and manage BLAS, LAPACK and ScaLAPACK rather than install them separately to avoid dependency problems if you are in X86 environment.

OpenMPI: https://www.open-mpi.org/. We suggest to use a stable release older than 3.0 to fit the interface such as MPI_Address et al.

PWDFT now only support GNU Make to automatically compile the project. At first you should modify the file  ``make.inc``  as a input of Makefile. For example:

```
DGDFT_DIR     = /staff/qcjiang/codes/hefeidgdft #Project root directory
FFTW_DIR      = /staff/qcjiang/lib/fftw-3.3.10 #FFTW root directory
MKL_ROOT      = /staff/qcjiang/intel/oneapi/mkl/2022.0.2 #MKL root directory
LIBXC_DIR     = /staff/qcjiang/lib/libxc-5.2.3 #Libxc root directory

...

CC            = mpicc #C compiler
CXX           = mpicxx #C++ compiler
FC            = mpif90 #fortran compiler
LOADER        = mpicxx #Loader
```

Afterwards, you can compile the project:

```
sh install.sh
```

## Usage

We provide several examples in ``tests``, you can run the example in this way:

```
cd tests/Si64
mpirun -n 4 ../../examples/pwdft
```

The parameters in the calculation is controlled by the file ``pwdft.in``, the parameters associated is as follows:

```
Extra_States:          0  #Extra states in the ground state calculation
Eig_MaxIter:           5  #The number of iterations within diagonalization in each SCF step 
SCF_Outer_MaxIter:     100 #SCF Indicates the maximum number of steps that can be executed
SCF_Outer_Tolerance:   1e-8 #Tolerance of SCF

PW_Solver:   PPCG, LOBPCG, CheFSI, Davidson,  PPCGScaLAPACK, LOBPCGScaLAPACK, DavidsonScaLAPACK  # the way to perform diagonalization
```



You can find further technical detail in our papar "Accelerating Parallel First-Principles Excited-State Calculation by Low-Rank Approximation with K-Means Clustering", accepted by ICPP'22 but not published yet. You can also refer to our previous work:

- [Hybrid MPI and OpenMP parallel implementation of large-scale linear-response time-dependent density functional theory with plane-wave basis set](https://iopscience.iop.org/article/10.1088/2516-1075/abfd1f)

- [Interpolative Separable Density Fitting Decomposition for Accelerating Hybrid Density Functional Calculations with Applications to Defects in Silicon](https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00807)
- [Interpolative Separable Density Fitting Decomposition for Accelerating Hartree–Fock Exchange Calculations within Numerical Atomic Orbitals](https://pubs.acs.org/doi/10.1021/acs.jpca.0c02826)
- [Accelerating Excitation Energy Computation in Molecules and Solids within Linear-Response Time-Dependent Density Functional Theory via Interpolative Separable Density Fitting Decomposition](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b01019)
- [Machine Learning K-Means Clustering Algorithm for Interpolative Separable Density Fitting to Accelerate Hybrid Functional Calculations with Numerical Atomic Orbitals](https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c06019)
