Refactor3 of DGDFT
==================

Last revision: 01/03/2020 

This note serves as a todo list for the 3rd major refactoring of ``DGDFT``. It includes the note for the discussion on 12/20/2019, and **(to be added)**.

Coding perspectives
-------------------

-  [ ] Datatype. NumMatBase -> NumMat and NumMap for data structure that
   stores actual values / only view other NumMat. The allocator
   attribute decides whether the memory is allocated on CPU or GPU.
-  [ ] In order to use an architecture to support NumVec, NumMat and NumTns,
   it is better to have a base class supporting tensors of arbitrary
   dimension, and defines NumVec, NumMat and NumTns using
   ``structure binding``.
-  [ ] The complex arithmetic should be templated, using the ``constexpr`` syntax, which can evaluate the value of the function or variable at compile time (C++17 feature)
-  [ ] Use header files as much as possible, but for large classes use explicit instantiation.
-  [ ] the blas.cpp / lapack.cpp shoulod be replaced by ``blas++`` and ``lapack++``. For now keep the scalapack interface as in.
-  [ ] Input parameter: change to the INI format with hierarchical input structure. 
-  [ ] Use hdf5 to manage the output.
-  [ ] Instead of calling ``class.Setup()``, use a default constructor together with ``std::move``. 
-  [ ] There should be a default folder to store the UPF files (ONCV pseudopotential)
-  [ ] ``DistVec`` should allow send to / recv from multiple processors.
-  [ ] Encapsulate the ``AllForward / AllBackward`` operations (mainly in PWDFT).

Functionality
-------------

- [ ] LOBPCG should be reimplemented. There are two options: one is for safety (slower but accurate), and the other is for production (faster but may break).

- [ ] ``FFTW_MEASURE`` can create undesired randomness. Should add option to allow the usage of ``wisdom`` file.

- [ ] The ScaLAPACK diagonalization should be replaced by ELPA. More specifically, the diagonalization / PEXSI interface should be replaced by the ELSI interafce.

Input variables
---------------

- [ ] ``Use_VLocal`` should be deprecated. With ONCV pseudopotential being the default option, we should always use the branch  ``Use_VLocal==1``.

Tests
-----

- [ ] Setup unit tests with google test.
- [ ] Test examples for PW / DG / TD.


Bug fixes
---------
- [ ] The force evaluation of DG with ``Use_VLocal==1`` is incorrect. 
