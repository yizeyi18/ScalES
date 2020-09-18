Introduction
------------

Overview
========

DGDFT (stands for Discontinuous Galerkin Density Functional Theory) is a
software package designed to perform large scale electronic structure
calculations (tens of thousands of atoms or larger). While the original
goal of the project was only on very large scale systems taking
advantage of the discontinuous Galerkin (DG) formulation, a few
useful, and relatively independent modules have been developed using the
software infrasture of DGDFT. 

- Plane Wave DFT (under ``src/pwdft``), for standard ground state DFT
  calculations using the planewave basis set. 

  * Large scale 

- Time-dependent DFT (under ``src/tddft``), for real-time TDDFT
  calculations using the planewave basis set.

- Discontinuous Galerkin DFT (under ``src/dg``), for very large scale
  ground state DFT calculations using the discontinuous (see logo),
  adaptive local basis set.

There are many excellent electronic structure software packages
available. The DGDFT package has the following features (some are
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

The DGDFT project started with a small team working part time on the
project in a math department (the situation has not changed drastically
so far). So almost all features of DGDFT are built around a single
scenario: **perform fast, massively parallel, large scale, Gamma-point
only DFT calculations** (with some more adjectives depending on the
detailed scenario, but the gist remains the same). DGDFT strives to be
the **best code** in this regime. For **anything outside this regime**,
we expect that users can easily find other software packages that
(certainly) have more functionalities and outperform DGDFT. The
following common tasks are currently considered out of scope of DGDFT.
None of the limitation is fundamental. They are all due to the desire to
make the code maintainable (a.k.a.  lack of man power and/or laziness).

- Band structure calculations (or in general k-point sampling).

- Non-orthorhombic supercells.

- Anything other than norm-conserving pseudopotentials (e.g. all-electron, USPP, PAW).

- Many more...




Contributors
============

**Contributors to method and code developments:**

- Lin Lin, University of California, Berkeley and Lawrence Berkeley National Laboratory, linlin@math.berkeley.edu
- Wei Hu, University of Science and Technology of China
- Weile Jia, University of California, Berkeley
- Amartya Banerjee, University of California, Los Algeles
- David Williams-Young, Lawrence Berkeley National Laboratory
- Lexing Ying, Stanford University
- Subhajit Banerjee, University of California, Davis
- Kun Dong, Google
- Jason Kaye, Flatiron Institute
- Gaigong Zhang, ASML

**Contributors to method developments:**

- Weinan E, Princeton University
- Jianfeng Lu, Duke University
- John Pask, Lawrence Livermore National Laboratory
- Phanish Suryanarayana, Georgia Institute of Technology 
- Lin-Wang Wang, Lawrence Berkeley National Laboratory
- Chao Yang, Lawrence Berkeley National Laboratory

Citing DGDFT
==============
For general usage of DGDFT package for electronic structure calculation, 
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

For large scale calculations using DGDFT, 
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


More references on DGDFT
========================

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

DGDFT version history
=====================

- v0.4 (5/20/2013)
  - Discard the attempt to use PETSc.
  - Start to focus on MD in the v0.4.x tags.

- v0.3 (1/20/2013)
  - Merge with the `elementorbital` developments (ultimately not used in
    favor of the purely discontinuous orbitals)

- v0.2 (8/7/2012)
  - Get prepared to migrate to the PETSc environment.


- v0.1 (5/5/2012)
  - Migrated to Git from SVN.
  - Has been refactored perhaps twice from a C code to a C++ code.
  - Already a reasonably parallel code!



License
=======
DGDFT is distributed under BSD license (modified by Lawrence Berkeley
National Laboratory).

DGDFT Copyright (c) 2012 The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of 
any required approvals from U.S. Dept. of Energy).  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
