TODO List   {#pageTODO}
=========
@todo
- A better way to handle various exchange correlation functionals
  - Initialization, finalizing phase
  - Category: (LDA, GGA, Hybrid, Meta, VdW) + (XCId / XId+CId). Or just
    always use XId + CId separate form (ABINIT might use this)
  - Need to make the same thing for DG
  - Design of XC class or something? (need discussion)
- Add support for libdbcsr and then for linear scaling? Or implement the
  native version?
  - Subspace problem in Chebyshev filtering. Use SCALAPACK with
    purification to handle the subspace problem. 
    o examples/diagonalize.cpp which has a SCALAPACK interface. Compare
      the performance of ScaLAPACK diagonalization and purification
      first for a fixed matrix.
  - LIBDBCSR format similar to PEXSI. Worth trying first. 
- OpenMP does not work for the new spinor multiplication due to the
  common dependence on the Fourier structure
  - The latest version is AddMultSpinorFineR2C, fft.inputVecR2C etc are
    shared between the threads, which limits the use of OpenMP. Perhaps
    increase the dimension of the buffer to the size of the local
    spinor, which allows the usage of FFTW_MANY or GURU interface in the
    future. In the case when number of threads equal to the number of
    local bands on the node, this implementation might be more advantageous.
  - Check with Weile on options
    1) OpenMP paralelization, multiple threads for a band
    2) add more buffer space for using thread parallelization for each
       band
   - Threaded FFTW can be tried with the new interface namely
     FFTWExecute
- Remove the unnecessary part of distinguishing coarse / fine grid
  (done), and
  document this a bit more properly. It is even possible to revamp the
  implementation by storing the wavefunction coefficients in the complex
  arithmetic (not sure)
  - When referring to the real space grid, there is no coarse grid,
    but only the fine real space grid. In the Fourier space, there are
    two grids. One is the same size as the real space grid, which is
    used for FFT. The second one is the Fourier grid restricted to a
    sphere, that is often used for storage of wavefunction coefficients,
    but nothing else.
  - In LOBPCG, the linear algebra is done with real arithmetic. If
    LOBPCG is to be performed for Fourier coefficients (like in KSSOLV),
    this LOBPCG part (linear algebra computatoin) need to be changed as
    well. This will eliminate the additional array which is Spinor in
    the real space with a coarse grid, and also requires eliminating the
    "coarse" in the code.
  - "coarse real space grid" might be useful for exchange calculation
    given experience from PEtot and VASP.
- Spinor class may be removed and moved to the Hamiltonian class. In
  the future different types of spinors should be treated with different
  classes of "Hamiltonian". The functions in spinor, such as
  preconditioners should also be moved to the Hamiltonian class (or
  KohnSham). (not sure still needed)
  - Hamitlonian should be a pure virtual class (does not implement
    anything actually)
  - Spin polarization should be kept in mind start from the beginning.
  - Spinor moved into Hamiltonian class. In KohnSham class, it is only
    for the spin-restricted calculation. Spinor has only two dimension,
    which is the number of grid points (in the Fourier space restricted to a
    sphere), and the number of bands. In spin-polarization, instead of
    having (ng,nc,nb) format, it might be better to have Spinor
    wavefun[numSpinComponent], where each wavefun has size (ng,nb).
    For the spin noncolinear polarization, the format of Spinor could be
    (ng,nc,nb) but this will be in the future. For k-point, the format is 
    Spinor wavefun[kpt]. Maybe k-point should be able to handle
    spin-polarization etc as well. But again in the future.
  - for eigensolver, in order to be compatible with multiple types of
    Hamiltonian classes, we need something like 
    LOBPCG( Hamiltonian, wavefun, param )
    where param is a simple struct to be declared in Hamiltonian class
    (or somewhere else) that is used to LABEL which part of the
    Hamiltonian class is used. This is immediately used when
    spin-polarization or k-point is needed.
  - In SCF, some "if statement" to detect the type of Hamiltonian should
    be needed. 
  - SCF, eigensolver supports multiple types of classes. Details in
    different realization of the Hamiltonian class.
  - Should be compatible with DGDFT. Currently DGDFT only supports
    KohnSham class essentially. It might be too difficult to support
    DGDFT for any class other than KohnSham at this point. Perhaps focus
    on hybrid functionals.
  - The new design should be combined with the design of spin
    polarization. This design instead should leave room for k-point
    implementation.
  - Super Hamiltonian class to handle spin?
  - 3/5/2016 Not sure Spinor class should be removed if we really only
    are going implement the spin case. Any extension to complex
    arithmetic (noncolinear spin and k-point seems to lead to very
    different code?)

- Hybrid is in the KohnSham class with sequential implementation. 
  - Parallel implementation for PWDFT
  - Hybrid for DG

- Nonlocal pseudopotential format is not very compatible with GPU like
  structure

DONE recently
c Remove BLOPEX
c Remove SCALAR design
c tentatively add core dumper 
  Better way to handle error: handling function taking a message as
  input is a more versatile way for handling error messaging. callstack
  procedure is slow and does not work for openmp. The DEBUG mode is too
  slow due to push/popstacks
  - Push/Popstack too expensive for simple operations. Throw error
    allows the error to be caught by the catch phrase, which gives the
    output of the callstack, but there is no way to use tools like gdb to
    look into the problem. Another simple way is to use `abort` but this
    implementation might be platform dependent and less informative. The
    third way is to rely on core dump, but not sure about massively parallel
    case.
  - try/catch and C++ exceptions not particularly useful for debugging.
  - coredumper
  - Encapsulate the error handling function (partially done)
c Combine PWDFT_bb and MD 
  c Should only have pwdft.cpp and dgdft.cpp
  c Standardize the output of initial and final results into subroutines
    that are shared between pwdft and dgdft. 
  c NVE, Nose-Hoover, BB, density extrapolation
  - into something called move_ions for geometry optimization and MD.
    o BFGS or preconditioned version (e.g. QE uses BFGS)
    o For 2000-10000 atoms, perhaps BFGS is too expensive. Maybe CG or
      FIRE alternative. PETOT uses CG?
    o MD: NVE, NH1, Langevin
c Refine Fourier to clean the normalization factors. Encapsulate the
  forward and backward Fourier transforms? In what convention?
  c Convention similar to QE can be considered.




12/6/2017: refactor2

@todo
- Merge refactor1 into master, and start a refactor2 branch (easy)

- Test the output from testSpecies.C from qbox, and benchmark with the
  output from MATLAB and figure out the unit convention etc.

- Adapt upf2qso to pspio.cpp/.hpp, and compare the results with MATLAB
  again.

- Introduce species.cpp/.hpp in parallel to periodtable.cpp/.hpp, to
  maintain backward compatibility and with DGDFT. The local part of the
  pseudopotential now removes the contribution from pseudocharge, and
  Gaussian pseudocharge needs to be added. The nonlocal one can reuse
  the current code structure.  The periodtable should eventually become
  obsolete.

- Coulomb interaction etc should be revamped. qbox can help.

- Add support from BigDFT's Poisson solver to evaluate the non-periodic
  boundary condition Coulomb. But perhaps the simplest is to find QE's
  treatment and use Makov-Payne type of correction first. This also
  outputs an estimate of the vacuum level. ESM / BigDFT's solver may be
  the right way to do things.

