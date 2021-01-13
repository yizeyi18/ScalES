Plan for Refactor2020
---------------------

Last revision: 09/15/2020 

This note serves as a todo list for the 3rd major refactoring of ``DGDFT``.  This has a different scope from the original plan 12/20/2019 (see below).

- [x] done
- [p] in progress
- [ ] planned but not started
- [d] still under consideration but deferred to future developments

Goal: Release DGDFT 1.0, and write a paper reporting the performance of
  PWDFT for hybrid functional calculations on multi-GPUs.


General thoughts
================

- The key component of DGDFT is PWDFT, which has the potential of being
  developed into a highly efficient, massively parallel code usable by a
  relatively wide audience. The features that make PWDFT standing out
  are that it has much better support of massively parallel and
  heterogeneous architecture, and that it supports new efficient
  algorithms for iterative solution of DFT and Hartree-Fock problems,
  like PPCG, ACE, ISDF etc.

- [x] The PWDFT module should be in a relatively clean folder pwdft/. It is
  by design a real arithmetic, gamma point code. Spin polarization may
  be supported. However, further developments, such as k-points and
  many-body perturbation theory etc should be in separate folders.

- [ ] The main difficulty of refactoring is the DG part. It is now in a
  separate folder called dg/ (also uses real arithemetic). The new
  developments by David should then be merged into dg/.

- [ ] TDDFT should be in a separate folder called tddft/. It defines complex
  spinors etc in its own folder and does not touch pwdft/.

- [p] GPU should be supported. We will put aside the issue of OpenMP for
  now. However, there should be a consistent way to multiple types of
  GPUs. A wrapper around the current cuNumMat etc so that we can write 
  xxx.device == GPU, xxx.device == HIP etc would be desirable.
  
- [x] Support of accelerators should be in separate files, e.g. spinor.cpp
  and spinor_device.cpp. The developers should try to make the
  corresponding files in sync with each other, and mark it so clearly in
  the code when syncing is not available / possible. For instance, when
  developing a new funcitonality, it should be in spinor.cpp (without
  GPU), and write in the documentation that spinor_device.cpp has not
  been modified correspondingly. 

- [ ] Unit tests can be in separate folders say pwdft/tests, and should be
  relatively up to date.

- [ ] Wei's group will be developing a number of features based on PWDFT in
  the near future. They are already relatively comfortable with the
  current code structure and style. So different programming models
  (such as realtively new features in C++) should be avoided when
  possible. However, the dg/ folder is a relatively separate piece,
  and can be used as a testbed for new programming models if needed
  (there may be also more benefit to be gained there)

- [p] A minimal sphinx based document for DGDFT, supporting mainly the PWDFT
  module.  Do not document the parameter values yet in sphinx yet (they are
  not stable for now). Directly refer to esdf.hpp and esdf.cpp


Target: refactor2020 for PWDFT
==============================


- [x] Introduce a "program" variable in esdf to determine the program
  mode. Deprecate the isDGDFT and isTDDFT flag

- [x] ``Use_VLocal`` should be deprecated. With ONCV pseudopotential
  being the default option, we should always use the branch
  ``Use_VLocal==1``.

- [x] Remove the KohnSham class and just have one Hamiltonian class.
  Future expansion of the functionality will not be based on inheritance
  but separate folders.

- [x] Simplified user input.

- [x] Change `SCF::Iterate` to `SCF::Execute()`. Inside this function,
  first call `SCF::IterateDensity` for all functionals (including
  hybrid ones, unless the hybrid mode is activated). Here for hybrid
  functional calculations, we run PBE first. Then execute
  `SCF::IterateWavefun` for hybrid functional calculations.

- [x] Separate contents of #ifdef GPU statements to the corresponding _device file

- [x] Make use of the `XCFamily_` and make the treatment of XC cleaner. 

- [x] Allow Hamiltonian to set the XC functional, instead of setting it
  in esdf

- [x] Properly handle the Gamma point and R2C, together with the Fourier
  interpolation problem (using an odd number of grid points). 
  
- [x] Maybe a better way is to avoid R2C during the coarse to fine grid
  interpolation, and use C2C instead. Then the coarse grid does not need
  to be restricted to an odd number of grid points.

- [x] EfreeHarris seems to be problematic for hybrid functional
  calculations.

- [x] Rename the awkward 'a3' in Hamiltonian and spinor to Hpsi

- [x] HSE calculation should not start with HSE w.o. exchange, this can
  create some instabilities. Instead it should start from e.g. PBE
  calculations. 

- [x] `SCF::IterateDensity` should be reused in `SCF::IterateWavefun`


- [ ] In Hamiltonian, add a pointer `ptablePtr_` for access to the
  information in the periodic table. Remove the pointer from the `SCF`
  class.

- [p] The structure of the code is largely based the current working code in
  the `GPU` branch, and will merge with `cuda_dg` branch `AMD_GPU_HIP`
  branch. Features in other branches are mainly in DG and can be merged
  later.

- [p] Encapsulate some GPU / ScaLAPACK operations to improve the readibility

- [p] A coherent, cmake environment for compiling src/pwdft and
  src/pwdft.cpp, with minimal requirement of external libraries. For
  instance, the default should be to compile pwdft, without the need of
  compiling PEXSI, etc.

- [p] Streamline the compilation of external packages in CMake: lbfgs (used in
  ionMove=="lbfgs"), rqrcp (used in ISDF), maybe compile by default.
  Other packages such as ELSI / PEXSI?
- [p] Can we encapsulate the ``AllForward / AllBackward`` operations (mainly in PWDFT)?

- [p] Give a few examples in examples/pwdft, maybe examples/tddft etc.

- [p] Document the code, at least for those being refactored

- [p] pwdft/spinor.cpp has too many deprecated implementations (in
  particular related to ISDF etc). Need to cleanup and remove the unused
  branches.

- [p] Prepare tutorial examples for PWDFT

- [p] Prepare a set of relatively robust default parameters, and simply
  the input file of the examples

- [p] LOBPCG should be reimplemented. There are two options: one is for safety (slower but accurate), and the other is for production (faster but may break).

- [ ] The ScaLAPACK diagonalization should be replaced by ELPA. More specifically, the diagonalization / PEXSI interface should be replaced by the ELSI interafce.

- [p] A consistent method to specify the input / to print the input
  parameters in pw/dg/td. so far only moves things to common/. Will see
  whether a cleaner solution is needed when organizing dg and td.



- [p] Atom positions should not be remapped back to [-0.5 a, a), where a
  is the lattice constant. In PW the mapping has been removed. Double
  check this with DG/TD.

- [ ] Remove the legacy support of the spin-orbit coupling
  pseudopotential (not supported by UPF anyway)

- [x] Add support for the HGH pseudopotential. This requires
  supporting non-off-diagonal DIJ (see KSSOLV's implementation
  pseudopotential/getvnl.m). However, fixing this requires at least one
  of the two actions:

    1. Diagonalize the DIJ matrix and store the eigenvectors. The
       problem with this is that the cutoffs from different nonlocal
       pseudopotentials will be mixed, which complicates the
       CalculateNonLocalPP process. (Lin CPU)

    2. Change vnl.weight from a scalar to a vector, storing each row of
       DIJ for a given J. Then when adding the contribution from the
       nonlocal pseudopotential, we first compute
       `<beta_J|psi>`, and then add `|beta_I>D_{IJ}<beta_J|psi>` to psi.
       We may add an if statement on `D_{IJ} != 0` to skip certain I's
       to reduce cost. This may affect other parts of the code such as
       DG. 
    
  Neither change is very simple, so we first need to decide whether we
  do need to support pseudopotentials where DIJ has off-diagonal
  entries (like HGH). Currently option 1 seems easier.

- [ ] Clean up the PWDFT source code, and make it more modular at the
  high level (after fixing geometry optimization). Create a separate
  file (e.g. md.cpp), and pwdft.cpp should stop at scf::Execute() (Wei)
  Make sure that in geometry optimization, the atomic position,
  atomic force, and convergence criterion are synced at the beginning of
  each iteration (maybe via MPI broadcast)
  Geometry optimization: should not reset to random wavefunctions
  each time. This is particularly problematic for hybrid functionals,
  where the Phi iteration starts from the beginning. In principle, the
  wavefunction should be reset only if something goes wrong. (see what
  QE does) Furthermore, in this case the next ion move should start with PBE
  instead of Phi iteration.

- [ ] Clean up the GPU part of the code to remove redundant copying.
  Also find a better way to remove the added argument `garbage` to
  distinguish the GPU and non-GPU versions of the same function. (Weile)


- [ ] Dynamic truncation criterion for eigensolver. In particular, the
  criterion is controlled by an energy like quantity. This should be
  implemented in all eigensolvers.

- [ ] CUFFT: One-by-one executation: is there a more efficient way to
  batched FFT? Why CUFFT does not suffer from the alignment issue? (i.e.
  we do not need to copy a vector into a saved buffer?) 
  
- [p] Eigensolver: in QE: reorder eigenvectors so that coefficients for
  unconverged roots come first. This allows to use quick matrix-matrix
  multiplications to set a new basis vector. Should we do the same? In
  the GPU version, this is replaced by reorder_evals_revecs. In the GPU
  based version pregterg_gpu, this is done by reorder_v, and
  subsequently redistribute the work for unconverged eigenvectors only.

- [p] The locking strategy in LOBPCG / PPCG. (David)

- [ ] Cleanup the AddMultSpinorEXXDF7 routine using the ScaLAPACK class.
  Remove the descriptors and contexts floating around. Decide whether to
  keep other EXXDF routines (David will first look at 2D distribution,
  and then pass to Weile)

- [ ] Make a decision about the best way to proceed with row<->col
  transformation among the methods of 
  
    a. the new bdist.redistribute_col_to_row and
    b. the old AlltoallForward / AlltoallBackward (e.g. used in MultSpinor) 
    c. methods based on pdgemr2d (not available in GPU, but according to
    Wei may be faster on CPU).

    We need:
    
    1. Benchmark results about the performance of each option.
    2. Leave at most two options (preferably one) for such a task. 
    3. In case pdgemr2d is needed in the end, it needs to be
           encapsulated.

    a. clean interface with bdist.redistribute_col_to_row (David). 
    b. Get rid of AlltoallForward / AlltoallBackward (David)
    c. test the performance of different implementations of bdist (Wei)

- [ ] pcdiis: cleanup the row<->col transformation. (Wei)

- [ ] Rename the project to Scalable Electronic Structure (ScalES,
  pronounced as "scales"). Change namespace etc, legal part etc

- [ ] The value of RGaussian should be properly set and tested for
  elements in the periodic table. In particular it should be checked
  that the overlap is not an issue (or better, implement the correction
  to the overlapping Gaussian charges in the self-interaction energy
  part c.f. Martin appendix). This may already be an issue, but would
  DEFINITELY be needed when changing to non-orthorhombic cells (see
  periodtable.cpp for more information under FIXME)

- [ ] The wavefun format, instead of (ir, icom, iband), maybe it is
  better to rearrange it to be (ir, iband, icom). By letting the last
  component of the tensor to be the component, we may use it for spin /
  k-points laters.


Plans for further developments in PWDFT
=======================================

- [d] OpenMP support? (most have been deleted so far)

- [d] Add some technical aspects of GPU support to `doc/developer.tex` 

- [d] Either make all string values of keywords to be lower case, or
  make string comparison case insensitive

- [d] HDF5 output of orbitals etc.

- [d] the blas.cpp / lapack.cpp shoulod be replaced by ``blas++`` and
  ``lapack++``. For now keep the scalapack interface as is. Recently
  looked into Slate. It seems still primitive.

- [d] Keep upfread up to date (c.f. the new implementation in KSSOLV
  @PpData/upfread.m. The current implementation is more like
  @PpData/upfreadold.m)

- [d] Coulomb norm in Anderson mixing.

- [d] Supporting FFT solvers other than FFTW (Wei)

- [d] Move esdf.cpp and esdf.hpp to the pwdft folder. In fact, each
  folder should be allowed to use its own esdfs (basically, separate
  folders should not be controlled by a central routine in the common/
  folder). The existing parser can be renamed esdf_common.hpp and
  esdf_common.cpp

- [d] Support of non-orthorhombic cells

- [d] Need to add SCAN functional (more generally, meta-GGA)

- [d] Need to provide API for an external electric field (w.o. using a
  velocity gauge?) 

- [d] Utilities to NumVec to clean up the spinor: 
  
    [ ] fine to coarse / coarse to fine grid
    [ ] element-wise product of two arrays (given by pointers) added to
    the third array. add to blas?


- [d] Change the default behavior from column partition to row partition
  in order to allow more processors than the number of bands (suggested
  by Wei Hu. This requires some discussion)

- [p] Remove the meaningless getters / setters in hamiltonian and scf,
  in the sense that the access subroutines provide full access to the
  variable without providing any additional information / explanation.
  [This requires some thoughts.]

- [d] The spinor class, other than storing the wavefunction, mainly
  provides information of the partition of the wavefunctions. If we
  would like to clean this up, it seems that the design in
  `hamiltonian_dg.hpp` is a better way to go, i.e. just store the
  wavefunction as something like `DistDblNumTns`, i.e.
  
  `typedef DistVec<IndexGroup, DblNumTns, IndexPrtn>   DblSpinor;`
  where the distribution is hidden under the key-data-partition
  structure. Then we may just embed spinor as a member Psi in
  Hamiltonian. Correspondingly the 

  This is potentially a BIG change. If we want to do this, we should
  think carefully about the data structure.



Plans for TDDFT 
===============

- [ ] Separate contents of #ifdef COMPLEX statements to tddft/ folder.

- [ ] Make tddft/ compile. maybe with cmake.


Plans for DGDFT
===============

- [ ] Make dg/ compile. Old fashioned Makefile is fine.



Meeting memos 
====================

**12/20/2019**:

It includes the note for initial discussion on 12/20/2019. together with new updates from 

Coding perspectives

-  [ ] Datatype. NumMatBase -> NumMat and NumMap for data structure that
   stores actual values / only view other NumMat. The allocator
   attribute decides whether the memory is allocated on CPU or GPU.
-  [ ] In order to use an architecture to support NumVec, NumMat and NumTns,
   it is better to have a base class supporting tensors of arbitrary
   dimension, and defines NumVec, NumMat and NumTns using
   ``structure binding``.
-  [ ] The complex arithmetic should be templated, using the ``constexpr`` syntax, which can evaluate the value of the function or variable at compile time (C++17 feature)
-  [ ] Use header files as much as possible, but for large classes use explicit instantiation.
-  [ ] the blas.cpp / lapack.cpp shoulod be replaced by ``blas++`` and ``lapack++``. For now keep the scalapack interface as is.
-  [ ] Input parameter: change to the INI format with hierarchical input structure. 
-  [ ] Use hdf5 to manage the output.
-  [ ] Instead of calling ``class.Setup()``, use a default constructor together with ``std::move``. 
-  [ ] There should be a default folder to store the UPF files (ONCV pseudopotential)
-  [ ] ``DistVec`` should allow send to / recv from multiple processors.
-  [ ] Encapsulate the ``AllForward / AllBackward`` operations (mainly in PWDFT).

Functionality

- [ ] LOBPCG should be reimplemented. There are two options: one is for safety (slower but accurate), and the other is for production (faster but may break).

- [ ] ``FFTW_MEASURE`` can create undesired randomness. Should add option to allow the usage of ``wisdom`` file.

- [ ] The ScaLAPACK diagonalization should be replaced by ELPA. More specifically, the diagonalization / PEXSI interface should be replaced by the ELSI interafce.

Input variables

- [ ] ``Use_VLocal`` should be deprecated. With ONCV pseudopotential being the default option, we should always use the branch  ``Use_VLocal==1``.

Tests

- [ ] Setup unit tests with google test.
- [ ] Test examples for PW / DG / TD.

**7/17/2020**:

- Confirm that pwdft/ and dg/ will only use real arithematics. Move all complex arithmetics to tddft/

- device level implementation can involve separate functions in
  xxx_device.hpp and xxx_device.cpp, but not separate classes. 

- We will implement wrappers around different implementation of GPUs
  based on Weile's plan.

- For the first step, Weile will perform the initial step of cleaning up
  the pwdft/ and tddft/ folders, and make them compilable (with some
  minimal dependency). Then we will merge with Wei and David's contributions



