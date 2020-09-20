Plan for Refactor2020
---------------------

Last revision: 09/15/2020 

This note serves as a todo list for the 3rd major refactoring of ``DGDFT``.  This has a different scope from the original plan 12/20/2019 (see below).

- [x] done
- [p] in progress
- [ ] planned but not started
- [d] still under consideration but deferred to future developments

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



Target: refactor2020
====================

- [p] The structure of the code is largely based the current working code in
  the `GPU` branch, and will merge with `cuda_dg` branch `AMD_GPU_HIP`
  branch. Features in other branches are mainly in DG and can be merged
  later.

- [p] Encapsulate some GPU / ScaLAPACK operations to improve the readibility

- [p] A coherent, cmake environment for compiling src/pwdft and
  src/pwdft.cpp, with minimal requirement of external libraries. For
  instance, the default should be to compile pwdft, without the need of
  compiling PEXSI, etc.


- [x] ``Use_VLocal`` should be deprecated. With ONCV pseudopotential
  being the default option, we should always use the branch
  ``Use_VLocal==1``.

- [ ] Can we encapsulate the ``AllForward / AllBackward`` operations (mainly in PWDFT)?

- [x] Separate contents of #ifdef GPU statements to the corresponding _device file

- [ ] Separate contents of #ifdef COMPLEX statements to tddft/ folder.

- [ ] Make tddft/ compile. maybe with cmake.

- [ ] Make dg/ compile. Old fashioned Makefile is fine.

- [p] A minimal sphinx based document for DGDFT, supporting mainly the PWDFT
  module.

- [p] Give a few examples in examples/pwdft, maybe examples/tddft etc.

- [p] Document the code, at least for those being refactored

- [ ] Add some technical aspects of GPU support to `doc/developer.tex` 

- [ ] pwdft/spinor.cpp has too many deprecated implementations (in
  particular related to ISDF etc). Need to cleanup and remove the unused
  branches.

- [ ] Prepare tutorial examples for PWDFT

- [p] Prepare a set of relatively robust default parameters, and simply
  the input file of the examples

- [d] Simplified user input. Maybe INI, or benefit from a python interface
  like `ase`?

- [ ] HDF5 output of orbitals etc.

- [ ] LOBPCG should be reimplemented. There are two options: one is for safety (slower but accurate), and the other is for production (faster but may break).

- [d] the blas.cpp / lapack.cpp shoulod be replaced by ``blas++`` and ``lapack++``. For now keep the scalapack interface as is. Recently looked into Slate. It seems still primitive.

- [ ] The ScaLAPACK diagonalization should be replaced by ELPA. More specifically, the diagonalization / PEXSI interface should be replaced by the ELSI interafce.

- [p] A consistent method to specify the input / to print the input
  parameters in pw/dg/td. so far only moves things to common/. Will see
  whether a cleaner solution is needed when organizing dg and td.

- [p] Make use of the `XCFamily_` in esdfParam and make the treatment of XC
  cleaner. This is already done in pw

- [ ] Move the density and wavefunction extrapolation part into
  iondynamics

- [ ] Release DGDFT 1.0, and write a paper reporting the performance of
  PWDFT for hybrid functional calculations on multi-GPUs.


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

