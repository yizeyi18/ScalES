Plan for Refactor2020
---------------------

Last revision: 09/15/2020 

This note serves as a todo list for the 3rd major refactoring of ``ScalES``.  This has a different scope from the original plan 12/20/2019 (see below).

- [x] done
- [p] in progress
- [ ] planned but not started
- [d] still under consideration but deferred to future developments

Goal: Release ScalES 1.0, and write a paper reporting the performance of
  PWDFT for hybrid functional calculations on multi-GPUs.


General thoughts
================

- The key component of ScalES is PWDFT, which has the potential of being
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

- [p] A minimal sphinx based document for ScalES, supporting mainly the PWDFT
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

- [x] Rename the project to Scalable Electronic Structure (ScalES,
  pronounced as "scales"). Change namespace etc

- [x] Simplify the legal part of each file, update author contribution
  (David)

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

- [p] A consistent method to specify the input / to print the input
  parameters in pw/dg/td. so far only moves things to common/. Will see
  whether a cleaner solution is needed when organizing dg and td.


- [p] Atom positions should not be remapped back to [-0.5 a, a), where a
  is the lattice constant. In PW the mapping has been removed. Double
  check this with DG/TD.

- [p] Remove the legacy support of the spin-orbit coupling
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
       DG (discarded)
    
  Neither change is very simple, so we first need to decide whether we
  do need to support pseudopotentials where DIJ has off-diagonal
  entries (like HGH). Currently option 1 is implemented.

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
  distinguish the GPU and non-GPU versions of the same function. These
  functions will be removed and combined into a single interface using
  CPU/GPU (Weile)


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
  - [x] in LOBPCG
  - [ ] in PPCG
  - [ ] in Chebyshev

- [p] The locking strategy in LOBPCG / PPCG. (David)
  - [x] in LOBPCG
  - [ ] in PPCG

- [ ] Cleanup the AddMultSpinorEXXDF7 routine using the ScaLAPACK class.
  Remove the descriptors and contexts floating around. Decide whether to
  keep other EXXDF routines (David will first look at 2D distribution,
  and then pass to Weile)

- [x] Make a decision about the best way to proceed with row<->col
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

    a. [x] clean interface with bdist.redistribute_col_to_row (David). 

    b. [p] Get rid of AlltoallForward / AlltoallBackward (David)
           [x] in CPU code
           [ ] in GPU code

    c. [ ] test the performance of different implementations of bdist (Wei)

- [ ] Systematically test the cutoff values for commonly used
  pseudopotentials and put them in the etable structure in
  periodtable.cpp

- [ ] pcdiis: cleanup the row<->col transformation. (Wei)

- [ ] The value of RGaussian should be properly set and tested for
  elements in the periodic table. In particular it should be checked
  that the overlap is not an issue (or better, implement the correction
  to the overlapping Gaussian charges in the self-interaction energy
  part c.f. Martin appendix). This may already be an issue, but would
  likely be needed when changing to non-orthorhombic cells (see
  periodtable.cpp for more information under FIXME)

- [ ] The wavefun format, instead of (ir, icom, iband), maybe it is
  better to rearrange it to be (ir, iband, icom). By letting the last
  component of the tensor to be the component, we may use it for spin /
  k-points laters. (Wei)

- [ ] Absorb localing partitioning of rows / columns into bdist (David)

- [ ] ACE: VexxProj applied to only unlocked vectors? (Do it after
  locking for eigensolver)

- [ ] Make all string values of keywords to be lower case, also check
  the input options consistently in ESDFReadInput using the new format
  of if( not InArray(esdfParam.program, program_list) ) etc. (Weile?)


Plans for further developments in PWDFT
=======================================

- [ ] Further cleaning up periodtable.cpp (absorbing etable as an
  attribute to the PeriodTable class etc). This is not urgent.

- [ ] The ScaLAPACK diagonalization should be replaced by ELPA. More specifically, the diagonalization / PEXSI interface should be replaced by the ELSI interafce.

- [d] OpenMP support? (most have been deleted so far)

- [d] Add some technical aspects of GPU support to `doc/developer.tex` 


- [d] HDF5 output of orbitals etc.

- [x] the blas.cpp / lapack.cpp shoulod be replaced by ``blas++`` and
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




Citing ScalES (need to figure out a way how to do this)
=======================================================

For general usage of ScalES package for electronic structure calculation, 
**please cite the following two papers.**::

    @Article{JCP2012,
      Title                    = {{Adaptive local basis set for Kohn-Sham density functional theory in a discontinuous Galerkin framework I: Total energy calculation}},
      Author                   = {Lin, L. and Lu, J. and Ying, L. and E, W.},
      Journal                  = {J. Comput. Phys.},
      Year                     = {2012},
      Pages                    = {2140--2154},
      Volume                   = {231}
    }
    
    @Article{JCP2015,
      Title                    = {{DGDFT}: A massively parallel method for large scale density functional theory calculations},
      Author                   = {W. Hu and L. Lin and C. Yang},
      Journal                  = {J. Chem. Phys.},
      Year                     = {2015},
      Pages                    = {124110},
      Volume                   = {143}
    }

For hybrid functional calculations using PWDFT, 
**please also cite the following paper.**::

    @Article{JCTC2016,
      Title                    = {Adaptively Compressed Exchange Operator},
      Author                   = {Lin, L.},
      Journal                  = {J. Chem. Theory Comput.},
      Year                     = {2016},
      Pages                    = {2242},
      Volume                   = {12}
    }

For large scale calculations using DGDFT and Chebyshev filtering, 
**please also cite the following paper.**::

    @Article{JCTC2018_DG,
      Title                    = {Two-level {Chebyshev} filter based complementary subspace method for pushing the envelope of large-scale electronic structure calculations},
      Author                   = {A. S. Banerjee and L. Lin and P. Suryanarayana and C. Yang and J. E. Pask},
      Journal                  = {J. Chem. Theory Comput.},
      Year                     = {2018},
      Pages                    = {2930},
      Volume                   = {14}
    }

For large scale RT-TDDFT calculations,
**please also cite the following paper.**::

    @Article{JCTC2018_TD,
      Title                    = {Fast real-time time-dependent density functional theory calculations with the parallel transport gauge},
      Author                   = {W. Jia and D. An and L.-W. Wang and L. Lin},
      Journal                  = {J. Chem. Theory Comput.},
      Year                     = {2018},
      Pages                    = {5645},
      Volume                   = {14}
    }

More references on ScalES
=========================

**Method developments:**

    W. Jia, L.-W. Wang and L. Lin, Parallel transport time-dependent density
    functional theory calculations with hybrid functional on Summit, SC '19
    Proceedings of the International Conference for High Performance
    Computing, Article No. 79

    W. Jia and L. Lin, Fast real-time time-dependent hybrid functional
    calculations with the parallel transport gauge and the adaptively
    compressed exchange formulation, Comput. Phys. Commun. 240, 21, 2019

    W. Hu, Y. Huang, X. Qin, L. Lin, E. Kan, X. Li, C. Yang, J. Yang,
    Room-temperature magnetism and tunable energy gaps in
    edge-passivated zigzag graphene quantum dots, npj 2D Mater. Appl. 3,
    17, 2019

    Y. Li and L. Lin, Globally constructed adaptive local basis set for
    spectral projectors of second order differential operators, SIAM
    Multiscale Model. Simul., 17, 92, 2019

    A. S. Banerjee, L. Lin, P. Suryanarayana, C. Yang, J. E. Pask,
    Two-level Chebyshev filter based complementary subspace method for
    pushing the envelope of large-scale electronic structure
    calculations, J. Chem. Theory Comput. 14, 2930, 2018

    K. Dong, W. Hu and L. Lin, Interpolative separable density fitting
    through centroidal Voronoi tessellation with applications to hybrid
    functional electronic structure calculations, J. Chem. Theory
    Comput. 14, 1311, 2018

    A. Damle and L. Lin, Disentanglement via entanglement: A unified
    method for Wannier localization, SIAM Multiscale Model. Simul., 16,
    1392, 2018

    W. Hu, L. Lin and C. Yang, Interpolative separable density fitting
    decomposition for accelerating hybrid density functional
    calculations with applications to defects in silicon, J. Chem.
    Theory Comput. 13, 5420, 2017

    W. Hu, L. Lin and C. Yang, Projected Commutator DIIS Method for
    Accelerating Hybrid Functional Electronic Structure Calculations, J.
    Chem. Theory Comput. 13, 5458, 2017

    L. Lin and B. Stamm, A posteriori error estimates for discontinuous
    Galerkin methods using non-polynomial basis functions. Part II:
    Eigenvalue problems, Math. Model. Numer. Anal. 51, 1733, 2017

    W. Hu, L. Lin, A. Banerjee, E. Vecharynski and C. Yang, Adaptively
    compressed exchange operator for large scale hybrid density
    functional calculations with applications to the adsorption of water
    on silicene, J. Chem. Theory Comput. 13, 1188, 2017

    G. Zhang, L. Lin, W. Hu, C. Yang and J.E. Pask, Adaptive local basis
    set for Kohn-Sham density functional theory in a discontinuous
    Galerkin framework II: Force, vibration, and molecular dynamics
    calculations, J. Comput. Phys. 335, 426 2017

    A. S. Banerjee, L. Lin, W. Hu, C. Yang, J. E. Pask, Chebyshev
    polynomial filtered subspace iteration in the Discontinuous Galerkin
    method for large-scale electronic structure calculations, J. Chem.
    Phys. 145, 154101, 2016

    L. Lin, Adaptively compressed exchange operator, J. Chem. Theory
    Comput. 12, 2242, 2016


    L. Lin and B. Stamm, A posteriori error estimates for discontinuous
    Galerkin methods using non-polynomial basis functions. Part I:
    Second order linear PDE, Math. Model. Numer. Anal. 50, 1193, 2016

    A. Damle, L. Lin and L. Ying, Compressed representation of Kohn-Sham
    orbitals via selected columns of the density matrix, J. Chem. Theory
    Comput. 11, 1463, 2015

    W. Hu, L. Lin and C. Yang, DGDFT: A massively parallel method for
    large scale density functional theory calculations, J. Chem. Phys.
    143, 124110, 2015

    J. Kaye, L. Lin and C. Yang, A posteriori error estimator for
    adaptive local basis functions to solve Kohn-Sham density functional
    theory, Commun. Math. Sci. 13, 1741, 2015

    L. Lin and L. Ying, Element orbitals for Kohn-Sham density
    functional theory, Phys. Rev. B 85, 235144, 2012

    L. Lin, J. Lu, L. Ying and W. E, Optimized local basis set for
    Kohn-Sham density functional theory, J. Comput. Phys 231, 4515,
    2012

    L. Lin, J. Lu, L. Ying and W. E, Adaptive local basis set for
    Kohn-Sham density functional theory in a discontinuous Galerkin
    framework I: Total energy calculation, J. Comput. Phys. 231, 2140,
    2012
    
**Applications:**

    W. Hu, L. Lin, R. Zhang, C. Yang and J. Yang, Highly efficient
    photocatalytic water splitting over edge-modified phosphorene
    nanoribbons, J. Amer. Chem. Soc. 139, 15429, 2017

    W. Hu, L. Lin, C. Yang, J. Dai and J. Yang, Edge-modified
    phosphorene nanoflake heterojunctions as highly efficient solar
    cells, Nano Lett. 16 1675, 2016

    W. Hu, L. Lin and C. Yang, Edge reconstruction in armchair
    phosphorene nanoribbons revealed by discontinuous Galerkin density
    functional theory, Phys. Chem. Chem. Phys. 17, 31397, 2015

    W. Hu, L. Lin, C. Yang and J. Yang, Electronic structure of
    large-scale graphene nanoflakes, J. Chem. Phys. 141, 214704, 2014
