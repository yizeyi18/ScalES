Introduction
------------

Overview
========

ScalES (Scalable Electronic Structure) is a software package aimed at 
perform electronic structure calculations in a scalable fashion. There
are a few relatively independent modules:

- Plane Wave DFT (under ``src/pwdft``), for standard ground state DFT
  calculations using the planewave basis set. 

  * Large scale 

- Time-dependent DFT (under ``src/tddft``), for real-time TDDFT
  calculations using the planewave basis set.

- Discontinuous Galerkin DFT (under ``src/dgdft``), for very large scale
  ground state DFT calculations using the discontinuous (see logo),
  adaptive local basis set.

There are many excellent electronic structure software packages
available. The ScalES package has the following features (some are
perhaps unique to this package) from a practical perspective:

- Very fast planewave DFT calculations for hundreds to thousands of
  atoms, in particular hybrid functional calculations (such as HSE).

- Excellent parallel scalability in almost all supported
  functionalities.

- Excellent multi-GPU support.

- Real-time TDDFT calculations with a relatively large time step (10 fs
  or more) in a planewave basis set.

- Fast, very large scale DFT calculations using DG.

- Written in C++, a relatively clean code (by the standard of the
  electronic structure community). It is modestly friendly to experienced
  electronic structure developers who also care about high performance
  computing.

The ScalES project started with a small team working part time on the
project in a math department (the situation has not changed drastically
so far). So almost all features of ScalES are built around a single
scenario: **perform fast, massively parallel, large scale, Gamma-point
only, pseudopotential DFT calculations** (with some more adjectives depending on the
detailed scenario, but the gist remains the same). ScalES strives to be
the **best electronic structure code** in this regime. For **anything
outside this regime**, we expect that users can easily find other
software packages that (certainly) have more functionalities and
outperform ScalES. The following common tasks are currently considered
out of scope of ScalES.  None of the limitation is fundamental. They are
all due to the desire to make the code maintainable (a.k.a.  lack of man
power and/or laziness).

- Band structure calculations (or in general k-point sampling).

- Non-orthorhombic supercells (originally because it takes some extra
  efforts to make DG work for non-orthorhombic cells. for large systems
  it is relatively easy to make orthorhombic supercells and is usually
  not a problem, with the help of software packages such as ASE).

- Anything other than norm-conserving pseudopotentials (e.g.
  all-electron, USPP, PAW).

- Many more...




Contributors
============

**Contributors to method and code developments:**

- Lin Lin, University of California, Berkeley and Lawrence Berkeley National Laboratory
- Wei Hu, University of Science and Technology of China
- Weile Jia, University of California, Berkeley
- Amartya Banerjee, University of California, Los Algeles
- David Williams-Young, Lawrence Berkeley National Laboratory
- Lexing Ying, Stanford University
- Subhajit Banerjee, University of California, Davis
- Gaigong Zhang, ASML
- Kun Dong, Google

**Contributors to method developments:**

- Weinan E, Princeton University
- Jianfeng Lu, Duke University
- John Pask, Lawrence Livermore National Laboratory
- Phanish Suryanarayana, Georgia Institute of Technology 
- Benjamin Stamm, RWTH Aachen University
- Lin-Wang Wang, Lawrence Berkeley National Laboratory
- Chao Yang, Lawrence Berkeley National Laboratory



ScalES version history
======================

- v1.0 (TBD)

  - First public release of ScalES project.
  - The publicly release version of ScalES include only the ``pwdft`` module. 
  - The ``dgdft`` and ``tddft`` module are available in the developer's
    branch and will be released later.

- Between v0.8 and v1.0 (2016-2020), we did not use a version system.
  Instead many branches with functionalties have been developed. The
  goal of the ``refactor2020`` branch is to merge many (not all) of the
  functionalties into a more or less uniform code, which turns into the
  new ``master`` branch for further developments. These branches
  include:

  - ``cuda_dg``: GPU development with cuda.
  - ``GPU`` and ``AMD_GPU_HIP``: GPU developments in PWDFT with cuda and
    hip, and tddft with cuda.
  - ``tddft``: Real-time TDDFT.
  - ``refactor1`` and ``refactor2`` (4/4/2016--12/27/2017): Major
    refactoring of the code as well as method developments.  Merge with
    ACE formulation of hybrid functionals.  Develop Chebyshev filtering
    for ``pwdft`` and ``dgdft``. ISDF algorithm for hybrid functionals. UPF file
    format for pseudopotentials.

- v0.8 (3/5/2016)

  - A number of developments including merging with hybrid functional branch.

- v0.7 (6/8/2015)

  - A number of developments including GGA functionals and OpenMP.
  
- v0.6 (8/8/2014)

  - A number of developments including intra-element parallelization

- v0.5 (3/27/2014)

  - A number of developments including integration with PEXSI.

- v0.4 (5/20/2013)

  - Discard the attempt to use PETSc.
  - A number of developments in v0.4.x including MD, parallel
    read/write, Harris functional, and OpenMP for LOBPCG
  - Another (not used) functionality is the evaluation of the a
    residual type posteriori error estimator in ``dgdft``, available at
    4/16 with commit ``bb22eb9``.

- v0.3 (1/20/2013)

  - Merge with the `elementorbital` developments (ultimately not used in
    favor of the purely discontinuous orbitals). The code are accessible
    in the ``OldElementOrbital`` branch.

- v0.2 (8/7/2012)

  - Get prepared to migrate to the PETSc environment.

- v0.1 (5/5/2012)

  - Migrated to Git from SVN.

- pre v0.1 (around 2010--2012)

  - Version control using SVN.
  - C code refactored into a C++ code. Perhaps refactored twice.
  - Already a reasonably parallel code!

