/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Authors: Chris J. Pickard and Lin Lin

   This file is part of DGDFT. All rights reserved.

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
*/
/// @file esdf.hpp
/// @brief Electronic structure data format for reading the input data.
/// @date 2012-08-10
#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
#include "domain.hpp"
#include "periodtable.hpp"
#include "tinyvec_impl.hpp"

namespace dgdft{


// *********************************************************************
// Electronic structure data format
// *********************************************************************
namespace esdf{


  /************************************************************ 
   * Main routines
   ************************************************************/
  void esdf_bcast();
  void esdf_key();
  void esdf_init(const char *);
  void esdf_string(const char *, const char *, char *);
  int esdf_integer(const char *, int);
  float esdf_single(const char *, float);
  double esdf_double(const char *, double);
  double esdf_physical(const char *, double, char *);
  bool esdf_defined(const char *);
  bool esdf_boolean(const char *, bool);
  bool esdf_block(const char *, int *);
  char *esdf_reduce(char *);
  double esdf_convfac(char *, char *);
  int esdf_unit(int *);
  void esdf_file(int *, char *, int *);
  void esdf_lablchk(char *, char *, int *);
  void esdf_die(char *);
  void esdf_warn(char *);
  void esdf_close();

  /************************************************************ 
   * Utilities
   ************************************************************/
  void getaline(FILE *, char *);
  void getlines(FILE *fp, char **);
  char *trim(char *);
  void adjustl(char *,char *);
  int len_trim(char *);
  int indexstr(char *, char *);
  int indexch(char *, char);
  int countw(char *, char **, int);
  char *strlwr(char *);
  char *strupr(char *);


  // *********************************************************************
  // Input interface
  // *********************************************************************
  /// @struct ESDFInputParam
  /// @brief Main structure containing input parameters for the
  /// electronic structure calculation.
  struct ESDFInputParam{
    /// @brief Global computational domain.
    ///
    /// Not an input parameter by the user.
    Domain              domain;
    /// @brief Types and positions of all atoms in the global
    /// computational domain.
    ///
    /// Not an input parameter by the user.
    std::vector<Atom>   atomList;

    /// @brief Mixing maximum dimension.
    ///
    /// Default: 9
    ///
    /// This parameter is relevant for Anderson mixing.
    Int                 mixMaxDim;
    /// @brief Mixing type for self-consistent field iteration.
    ///
    /// Default: anderson
    ///
    /// - = "anderson"           : Anderson mixing
    /// - = "kerker+anderson"    : Anderson mixing with Kerker
    /// preconditioner
    ///
    /// @todo Preconditioner better than Kerker mixing.
    std::string         mixType;
    /// @brief Which variable to mix
    ///
    /// Default: density
    ///
    /// - = "density"            : Density mixing
    /// - = "potential"          : Potential mixing
    std::string         mixVariable;
    /// @brief Coefficient in front of the preconditioned residual to be
    /// mixed with the mixing variable in the previous step.
    ///
    /// Default: 0.8
    ///
    /// For metallic systems or small gapped semiconductors,
    /// mixStepLength is often needed to be smaller than 0.1.  In such
    /// case, a better preconditioner such as Kerker preconditioner can
    /// be helpful.
    Real                mixStepLength;            
    /// @brief Tolerance for inner %SCF iteration in DG calculation.
    ///
    /// Default: 1e-4
    ///
    /// @note When scfInnerMaxIter = 1 (which is used most of the cases
    /// in the current version), this parameter is not useful.
    Real                scfInnerTolerance;
    /// @brief Tolerance for outer %SCF iteration in DG calculation.
    ///
    /// Default: 1e-6
    ///
    /// The DG calculation stops when
    ///
    /// \f[
    /// \frac{\Vert v^{(k)} - v^{(k-1)} \Vert}{\Vert{v^{(k-1)}}\Vert} <
    /// scfOuterTolerance
    /// \f]
    ///
    /// where the variable \f$v\f$ can be the density 
    /// (mixVariable = "density") or potential 
    /// (mixVariable = "potential").
    Real                scfOuterTolerance;
    /// @brief The DG calculation stops when the difference between free
    /// energy and Harris energy per atom is less than scfOuterEnergyTolerance.
    ///
    /// Default: 1e-4
    Real                scfOuterEnergyTolerance;
    /// @brief Minimum number of inner %SCF iterations
    ///
    /// Default: 1
    Int                 scfInnerMinIter;
    /// @brief Maximum number of inner %SCF iterations
    ///
    /// Default: 1
    Int                 scfInnerMaxIter;
    /// @brief Minimum number of outer %SCF iterations
    ///
    /// Default: 3
    Int                 scfOuterMinIter;
    /// @brief Maximum number of outer %SCF iterations
    ///
    /// Default: 30
    Int                 scfOuterMaxIter;
    /// @brief Maximum number of iterations for hybrid functional
    /// iterations.
    /// 
    /// Default: 10
    Int                 scfPhiMaxIter;
    /// @brief Tolerance for hybrid functional iterations using Fock
    /// energy
    ///
    /// Default: 1e-6
    Real                scfPhiTolerance;
    /// @brief Whether to use the projection formulation for hybrid
    /// functional
    ///
    /// Default: 0
    bool                isHybridVexxProj;
    /// @brief Treatment of the divergence term in hybrid functional
    /// calculation.
    ///
    /// Default: 1
    ///
    /// - 0    : No regularization
    /// - 1    : Gygi-Baldereschi regularization
    Int                 exxDivergenceType;

    /// @brief Tolerance for the eigenvalue solver
    ///
    /// Default: 1e-6
    ///
    /// Currently the LOBPCG method is used as the eigenvalue solver for
    /// obtaining the adaptive local basis functions, and eigTolerance
    /// controls the tolerance for the LOBPCG solver.  
    ///
    /// In the case when the eigensolver tolerance is tunned dynamically
    /// (see 
    /// @ref dgdft::esdf::ESDFInputParam::isEigToleranceDynamic "isEigToleranceDynamic"), the tolerance for
    /// the eigensolver is controlled dynamically and can be larger than
    /// eigTolerance.
    Real                eigTolerance;
    /// @brief Minimum number of iterations for the eigensolver.
    ///
    /// Default: 1
    Int                 eigMinIter;
    /// @brief Maximum number of iterations for the eigensolver.
    ///
    /// Default: 10
    Int                 eigMaxIter;
    /// @brief Tolerance for thresholding the adaptive local basis functions.
    ///
    /// Default: 1e-6
    ///
    /// The adaptive local basis functions restricted to the element are
    /// not orthogonal and may be linearly dependent.  An local SVD
    /// decomposition is performed to eliminate the linearly dependent
    /// modes with tolerance SVDBasisTolerance.
    Real                SVDBasisTolerance;
    /// @brief Whether to use the saved electron density as the start.
    ///
    /// Default: 0
    bool                isRestartDensity;
    /// @brief Whether to use the saved basis functions in extended
    /// element as the start.
    ///
    /// Default: 0
    bool                isRestartWfn;
    /// @brief Whether to output the electron density.
    ///
    /// Default: 1
    ///
    /// When isOutputDensity = 1, files DEN_xxx_yyy will be generated,
    /// where by default xxx is the mpirank (starting from 0), and yyy
    /// is mpisize.
    ///
    /// This option is needed to restart the electron density using 
    /// @ref dgdft::esdf::ESDFInputParam::isRestartDensity "isRestartDensity".
    bool                isOutputDensity;
    /// @brief Whether to output the wavefunctions in the element on LGL
    /// grid.
    ///
    /// Default: 0
    ///
    /// This is mainly for debugging and visualization purpose and is
    /// not commonly used.
    bool                isOutputALBElemLGL;
    /// @brief Whether to output the wavefunctions in the element on
    /// uniform grid.
    ///
    /// Default: 0
    ///
    /// This is mainly for debugging and visualization purpose and is
    /// not commonly used.
    bool                isOutputALBElemUniform;
    /// @brief Whether to output the wavefunctions in the extended
    /// element.
    ///
    /// Default: 1
    ///
    /// When isOutputWfnExtElem = 1, the approximate eigenvectors in the
    /// extended element are given in the output, in the form
    /// WFNEXT_xxx_yyy, where by default xxx is the mpirank (starting
    /// from 0), and yyy is mpisize.
    bool                isOutputWfnExtElem;
    /// @brief Whether to output the potential in the extended
    /// element.
    ///
    /// Default: 0
    ///
    /// This is mainly for debugging and visualization purpose and is
    /// not commonly used.
    bool                isOutputPotExtElem;
    /// @brief Whether to calculate a posteriori error estimator for
    /// each %SCF iteration.
    ///
    /// Default: 0
    ///
    /// If this is 0, then the a posteriori error estimator is given in
    /// the output after the %SCF iteration is finished.
    ///
    /// @todo Sharp a posteriori error estimate than the current version.
    bool                isCalculateAPosterioriEachSCF; 
    /// @brief Whether to calculate the force each %SCF iteration.
    ///
    /// Default: 1
    bool                isCalculateForceEachSCF; 
    /// @brief Whether to output the DG Hamiltonian matrix in each %SCF
    /// iteration.
    ///
    /// Default: 0
    ///
    /// If isOutputHMatrix = 1, the H matrix is output in the file
    /// `H.csc` using the compressed sparse column (CSC) binary format.
    bool                isOutputHMatrix;


    /// @brief Inverse of temperature.
    ///
    /// Default: 1.0 / (100 K * k_b)
    ///
    /// This parameter is not controlled directly, but through 
    /// "Temperature" in the input file, in the unit of Kelvin.
    Real                Tbeta;       
    /// @brief Number of empty states for finite temperature
    /// calculation.
    ///
    /// Default: 0
    ///
    /// This parameter must be larger than 0 for small gapped systems or
    /// relatively high temperature calculations.
    Int                 numExtraState;
    /// @brief Some states for the planewave solver are unused in order
    /// to accelerate the convergence rate of the eigensolver.
    ///
    /// Default: 0
    Int                 numUnusedState;
    /// @brief Whether to control the tolerance of the eigensolver
    /// dynamically.
    ///
    /// Default: 1
    ///
    /// When isEigToleranceDynamic = 1, the tolerance for the
    /// eigensolver is controlled dynamically and is related to the
    /// error in the current %SCF iteration.  The lower limit of the
    /// tolerance is controlled by
    /// @ref dgdft::esdf::ESDFInputParam::eigTolerance "eigTolerance".
    bool                isEigToleranceDynamic;
    /// @brief File for storing the information of the pseudopotential.
    ///
    /// Default: "HGH.bin"
    ///
    /// The pseudopotential file is currently only for the
    /// Hartwigsen-Goedecker-Hutter pseudopotential, and is generated by
    /// the utility subroutine HGH.m.
    ///
    /// @note Only the master processor (mpirank == 0) reads this table,
    /// and the information is broadcast to other processors.
    std::string         periodTableFile;
    /// @brief Type of the pseudopotential
    ///
    /// Default: "HGH"
    ///
    /// Currently HGH is the only supported pseudopotential format.
    std::string         pseudoType;
    /// @brief Solver for the planewave problem.  
    ///
    /// @todo out-of-date description.  BLOPEX is to be removed.
    ///
    /// - = "modified_blopex"    : BLOPEX package (modified in the
    ///                            external/ directory using the LOBPCG
    ///                            method but with BLAS3 for linear
    ///                            algebra operations.  (default)
    std::string         PWSolver;                 
    /// @brief Method for solving the projected problem in the adaptive
    /// local basis set.
    ///
    /// Default: "diag"
    ///
    /// - = "diag"      : Diagonalization method using ScaLAPACK. 
    /// - = "pexsi"     : Pole expansion and selected inversion method.
    ///                   This option needs to turn on the macro -DPEXSI
    ///                   to support the libraries.
    std::string         solutionMethod; 
    /// @brief Type of the exchange-correlation functional.
    ///
    /// Default: "XC_LDA_XC_TETER93"
    ///
    /// The exchange-correlation functional is implemented using the
    /// libxc package. Currently only the LDA and GGA xc functionals is
    /// supported.
    std::string         XCType;
    /// @brief Type of the van der Waals correction.
    ///
    /// Default: "DFT-D2"
    ///
    /// Currently only the DFT-D2 correction is supported.
    std::string         VDWType;

    // DG related
    /// @brief Number of elements along x,y,z directions.
    ///
    /// Default: (1,1,1)
    Index3              numElem;
    /// @brief Number of uniform grids along x,y,z directions in the
    /// extended element.
    ///
    /// This is not directly controlled by the user, but through 
    /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction".
    Index3              numGridWavefunctionElem;
    /// @brief Number of uniform grids for representing the density
    /// along x,y,z directions in the element.
    ///
    /// The current implementation uses dual-grid implementation, which
    /// uses a denser grid to represent the electron density and
    /// potential than that of the wavefunction on the uniform grid.
    /// This parameter is controlled by
    /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction"
    /// and
    /// @ref dgdft::esdf::ESDFInputParam::densityGridFactor "densityGridFactor".
    Index3              numGridDensityElem;
    /// @brief Number of Legendre-Gauss-Lobatto (LGL) grids for
    /// representing the basis functions along x,y,z directions in the
    /// element.
    ///
    /// This parameter is controlled by
    /// @ref dgdft::esdf::ESDFInputParam::ecutWavefunction "ecutWavefunction"
    /// and
    /// @ref dgdft::esdf::ESDFInputParam::LGLGridFactor "LGLGridFactor".
    Index3              numGridLGL;
    /// @brief Penalty parameter 
    ///
    /// Default: 100.0
    ///
    /// The same penalty parameter is applied to all faces.
    ///
    /// @todo The automatic choice of penalty parameter is to be
    /// implemented.
    Real                penaltyAlpha;
    /// @brief Number of adaptive local basis functions in each element.
    ///
    /// Default: Must be provided by the user.
    ///
    /// Dimension: numElem[0] * numElem[1] * numElem[2]
    ///
    /// @note In the current implementation, the actual number of basis
    /// functions used in the element (i,j,k) is **no more than**
    /// numALBElem(i,j,k)-numUnusedState.
    IntNumTns           numALBElem;
    /// @brief Block size for ScaLAPACK.
    ///
    /// Default: 16
    ///
    /// Only used when ScaLAPACK is invoked.
    Int                 scaBlockSize;

    // Add a potential barrier in the extended element
    bool                isPotentialBarrier;
    Real                potentialBarrierW;
    Real                potentialBarrierS;
    Real                potentialBarrierR;

    // Periodization of the potential in the extended element
    bool                isPeriodizePotential;
    Point3              distancePeriodize;	

    /// @brief Kinetic energy cutoff for the wavefunction on the uniform
    /// grid.
    ///
    /// Default: 10.0 Ha
    ///
    /// The number of uniform grids for the wavefunction along each
    /// direction i (i=x,y,z) is given by the formula
    /// \f[
    ///    N_i = \sqrt{2*ecutWavefunction}*L_i
    /// \f]
    /// where \f$L_i\f$ is the dimension of the domain
    /// along the direction i. The domain can be the global domain,
    /// extended element or element.
    Real                ecutWavefunction;
    /// @brief The ratio between the number of grids for the density and
    /// the wavefunction in the uniform grid along each dimension.
    ///
    /// Default: 2.0
    ///
    /// The number of uniform grids for the density and potential along
    /// each direction i (i=x,y,z) is given by the formula
    /// \f[
    ///    N_i = densityGridFactor * \sqrt{2*ecutWavefunction}*L_i
    /// \f]
    /// where \f$L_i\f$ is the dimension of the domain
    /// along the direction i. The domain can be the global domain,
    /// extended element or element.
    Real                densityGridFactor;
    /// @brief The ratio between the number of LGL grid and uniform grid
    /// for the basis functions along each dimension.
    ///
    /// Default: 2.0
    ///
    /// The number of LGL grids along each direction i (i=x,y,z) is
    /// given by the formula
    /// \f[
    ///    N_i = LGLGridFactor * \sqrt{2*ecutWavefunction}*L_i
    /// \f]
    /// where \f$L_i\f$ is the dimension of the domain
    /// along the direction i. The domain can be the global domain,
    /// extended element or element.
    Real                LGLGridFactor;
    
    /// @brief The interp factor for Gaussian function for generating 
    /// the transfer matrix from LGL grid to uniform grid on each
    /// element with the Gaussian convolution interpolation method. 
    /// 
    /// Default: 4.0
    Real                GaussInterpFactor;

    /// @brief The sigma value for Gaussian function for generating 
    /// the transfer matrix from LGL grid to uniform grid on each
    /// element with the Gaussian convolution interpolation method. 
    /// 
    /// Default: 0.001
    Real                GaussSigma;
    
    /// @brief Number of processors for distributed FFT.
    ///
    Int                 numProcDistFFT;

    /// @brief Number of processors used by ScaLAPACK.
    ///
    /// Default: mpisize
    Int                 numProcScaLAPACK;

    // PEXSI
    /// @brief Number of processors in the row communication group for
    /// each pole.
    ///
    /// Default: 1
    Int                 numProcRowPEXSI;
    /// @brief Number of processors in the column communication group for
    /// each pole.
    ///
    /// Default: 1
    Int                 numProcColPEXSI;

    /// @brief Number of terms in the pole expansion.
    /// 
    /// Default: 60
    Int                 numPole;
    /// @brief Number of processors for PARMETIS/PT-SCOTCH.
    /// 
    /// Default: 1
    Int                 npSymbFact;
    /// @brief Spectral gap.
    ///
    /// Default: 0.0 Ha
    ///
    /// Setting this value to be 0.0 works for both metallic and
    /// insulating systems. A correct and larger energy gap can reduce
    /// the number of poles.
    Real                energyGap;
    /// @brief Spectral radius of the (H,S) pencil.
    ///
    /// Default: 100.0 Ha
    ///
    /// Often in practice this estimate does not need to be sharp.
    Real                spectralRadius;
    /// @brief Ordering strategy for factorization and selected
    /// inversion.  
    ///
    /// Default: 0
    ///
    /// - = 0   : Parallel ordering using ParMETIS/PT-SCOTCH (PARMETIS
    ///   option in SuperLU_DIST).
    /// - = 1   : Sequential ordering using METIS (METIS_AT_PLUS_A
    ///   option in SuperLU_DIST).
    /// - = 2   : Multiple minimum degree ordering (MMD_AT_PLUS_A
    ///   option in SuperLU_DIST).
    Int                 matrixOrdering;
    /// @brief Number of %SCF iterations before inertia counting is
    /// turned off.
    ///
    /// Default: 10
    Int                 inertiaCountSteps;
    /// @brief Maximum number of PEXSI iteration.
    ///
    /// Default: 5
    Int                 maxPEXSIIter;
    /// @brief Estimate of the lower bound of the chemical potential.
    ///
    /// Default: -2.0 Ha
    Real                muMin;
    /// @brief Estimate of the upper bound of the chemical potential.
    ///
    /// Default: 2.0 Ha
    Real                muMax;
    /// @brief Tolerance for the total number of electrons in the PEXSI
    /// iteration.
    /// 
    /// Default: 0.001
    Real                numElectronPEXSITolerance;
    /// @brief Tolerance for the chemical potential in the inertia
    /// counting procedure.
    ///
    /// Default: 0.05 Ha
    Real                muInertiaTolerance;
    /// @brief Step length for expanding the chemical potential
    /// interval.
    ///
    /// Default: 0.3 Ha
    ///
    /// If the chemical potential is not in the prescribed interval, the
    /// interval is then expanded by the prescribed step length.
    Real                muInertiaExpansion;
    /// @brief Safe guard for the chemical potential update.
    ///
    /// Default: 0.05 Ha
    ///
    /// If the difference of chemical potential between consequetive
    /// steps of the PEXSI iteration exceeds the safe guard value, the
    /// value is not trusted and the inertia count procedure is
    /// re-invoked.
    Real                muPEXSISafeGuard;

    /// @brief Maximum number of steps for geometry optimization
    ///
    /// Default: 10 
    Int                 geoOptMaxStep;
    /// @brief Maximum force for geometry optimization
    ///
    /// Default: 0.001 
    Real                geoOptMaxForce;

    /// @brief Maximum number of steps for MD simulation 
    ///
    /// Default: 10 
		Int									MDMaxStep; 
    /// @brief Time step for MD simulation.
    ///
    /// Default: 50.0
		Int									MDTimeStep; 
    /// @brief Extrapolation type for updating the density
    ///
    /// Default: "linear"
    ///
    /// Currently three extrapolation  types (linear, quadratic and
    /// Dario) are supported.
    std::string         MDExtrapolationType;
    /// @brief Temperature for ion.
    ///
    /// Default: K
    Real                ionTemperature;       
    /// @brief Inverse of ionTemperature.
    ///
    /// Default: 1.0 / (100 K * k_b)
    ///
    /// This parameter is not controlled directly, but through 
    /// "ionTemperature" in the input file, in the unit of Kelvin.
    Real                TbetaIonTemperature;       
    /// @brief Mass for Nose-Hoover thermostat
    ///
    /// Default: 10.0
		Real								qMass;								
    /// @brief Whether to use the previous position
    ///
    /// Default: 0
    bool                isRestartPosition;
    /// @brief Whether to use the previous thermostat state
    ///
    /// Default: 0
    bool                isRestartThermostat;
    /// @brief Whether to output position information
    ///
    /// Default: 1
    bool                isOutputPosition;
    /// @brief Whether to output thermostat information
    ///
    /// Default: 1
    bool                isOutputThermostat;
    /// @brief Output the atomic position in XYZ format. Used in MD
    /// simulation and geometry optimization
    ///
    /// Default: 1
    bool                isOutputXYZ;
  };

  void ESDFReadInput( ESDFInputParam& esdfParam, const std::string filename );

  void ESDFReadInput( ESDFInputParam& esdfParam, const char* filename );

} // namespace esdf
} // namespace dgdft
#endif // _ESDF_HPP_
