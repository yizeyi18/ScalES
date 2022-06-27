# LRTDDFT-ISDF

This branch is for performing LR-TDDFT (Linear Response Time-Dependent Density Functional Theory) with ISDF (Interpolative Separable Density Fitting).

## Installation

The LR-TDDFT program is built within PWDFT (Plane Wave Density Functional Theory) software package. It is written mainly in c++ with message passing interface (MPI) for parallel design. PWDFT depends on several external numerical libraries (BLAS, LAPACK, ScaLAPACK, FFTW and Libxc), which you should install first:

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
cd external/lbfgs
make cleanall && make
cd ../rqrcp
make cleanall && make
cd ../blopex/blopex_abstract
make clean && make
cd ../../../src
make cleanall && make -j
cd ../examples
make cleanall && make pwdft
```

## Usage

We provide several examples in ``tests/lrtddft``, you can run the example in this way:

```
cd tests/lrtddft/Si64
mpirun -n 4 ../../../examples/pwdft
```

The parameters in the calculation is controlled by the file ``pwdft.in``, the parameters associated with LRTDDFT-ISDF is as follows:

```
LRTDDFT:               1 #Perform LRTDDFT
Extra_States:          32 #Extra states in the ground state calculation
NvBand:                32 #Number of valence band, N_v in the paper
NcBand:                32 #Number of conduction band, N_c in the paper

LRTDDFT_ISDF:          1 #Perform LRTDDFT with ISDF
NumMuFac_LRTDDFT_ISDF: 0.1 #Ration of interpolation points, N_{\mu} in the paper
NumGaussianRandomFac_LRTDDFT_ISDF: 1.0 #Gaussian random matrices factor
Tolerance_LRTDDFT:     1e-8 #Parameter to control the convergence of LRTDDFT

IPType_LRTDDFT_ISDF:  Kmeans #The way to choose interpolation points
MaxIterKmeans_LRTDDFT_ISDF: 10 #Max iterations to perform ISDF
ToleranceKmeans_LRTDDFT_ISDF: 1e-5 #Parameter to control the convergence of ISDF

EigenSolver_LRTDDFT:   LOBPCG # LAPACK or ScaLAPACK or LOBPCG, the way to perform diagonalization
NkBand:                10 # N_k in LOBPCG procedure
EigMaxIter_LRTDDFT:    100 #Max iterations to perform LOBPCG
```



You can find further technical detail in our papar "Accelerating Parallel First-Principles Excited-State Calculation by Low-Rank Approximation with K-Means Clustering", accepted by ICPP'22 but not published yet. You can also refer to our previous work:

- [Hybrid MPI and OpenMP parallel implementation of large-scale linear-response time-dependent density functional theory with plane-wave basis set](https://iopscience.iop.org/article/10.1088/2516-1075/abfd1f)

- [Interpolative Separable Density Fitting Decomposition for Accelerating Hybrid Density Functional Calculations with Applications to Defects in Silicon](https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00807)
- [Interpolative Separable Density Fitting Decomposition for Accelerating Hartree–Fock Exchange Calculations within Numerical Atomic Orbitals](https://pubs.acs.org/doi/10.1021/acs.jpca.0c02826)
- [Accelerating Excitation Energy Computation in Molecules and Solids within Linear-Response Time-Dependent Density Functional Theory via Interpolative Separable Density Fitting Decomposition](https://pubs.acs.org/doi/full/10.1021/acs.jctc.9b01019)
- [Machine Learning K-Means Clustering Algorithm for Interpolative Separable Density Fitting to Accelerate Hybrid Functional Calculations with Numerical Atomic Orbitals](https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c06019)