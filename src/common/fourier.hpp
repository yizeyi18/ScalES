//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin 


/// @file fourier.cpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
/// @date 2015-05-02 Add some dual grid functions
#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include <memory>
#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"

namespace scales{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
struct Fourier {
  std::shared_ptr<Domain> domain = nullptr;
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridTotalFine;
  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;
  fftw_plan backwardPlanFine;
  fftw_plan forwardPlanFine;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;
  fftw_plan backwardPlanR2CFine;
  fftw_plan forwardPlanR2CFine;

  fftw_plan mpiforwardPlanFine;
  fftw_plan mpibackwardPlanFine;
  MPI_Comm         comm;
  ptrdiff_t        localNz;
  ptrdiff_t        localNzStart;
  ptrdiff_t        numAllocLocal;
  Int              numGridLocal;
  bool             isMPIFFTW;
  CpxNumVec        inputComplexVecLocal;
  CpxNumVec        outputComplexVecLocal;     

  unsigned  plannerFlag;

  // Laplacian operator related
  DblNumVec                gkk;
  std::vector<CpxNumVec>   ik;
  DblNumVec                TeterPrecond;

  DblNumVec                gkkFine;
  std::vector<CpxNumVec>   ikFine;
  // FIXME Teter should be moved to Hamiltonian
  DblNumVec                TeterPrecondFine;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVec;     
  CpxNumVec                outputComplexVec;     

  CpxNumVec                inputComplexVecFine;     
  CpxNumVec                outputComplexVecFine;     


  // Real data Fourier transform
  Int       numGridTotalR2C;
  Int       numGridTotalR2CFine;

  DblNumVec                gkkR2C;
  std::vector<CpxNumVec>   ikR2C;
  DblNumVec                TeterPrecondR2C;

  DblNumVec                gkkR2CFine;
  std::vector<CpxNumVec>   ikR2CFine;
  DblNumVec                TeterPrecondR2CFine;

  // Temporary vectors that can also be used globally
  DblNumVec                inputVecR2C;     
  CpxNumVec                outputVecR2C;     

  DblNumVec                inputVecR2CFine;     
  CpxNumVec                outputVecR2CFine;     

  /// @brief index array for mapping a coarse grid to a fine grid
  IntNumVec                idxFineGrid;
  IntNumVec                idxFineGridR2C;


  Fourier() = delete;
  Fourier( std::shared_ptr<Domain> domain );
  ~Fourier();

  void Initialize();
  void InitializeFine();


};

void FFTWExecute( Fourier& fft, fftw_plan& plan );

// *********************************************************************
// Parallel FFTW interface
// *********************************************************************

/// @struct DistFourier
/// @brief Distributed memory (MPI only) parallel FFTW interface.
struct DistFourier {
  Domain           domain;                      
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridLocal;    
  ptrdiff_t        localNz;
  ptrdiff_t        localNzStart;
  // numAllocLocal is the size for the FFTW vectors.
  // numAllocLocal is close but may not be exactly the same as numGridLocal, which is
  // localNz * Ny * Nx.  This is because FFTW may need some intermediate
  // space.  For more information see FFTW's manual.
  ptrdiff_t        numAllocLocal;

  /// @brief Whether the processor according to the rank in domain
  /// participate in the FFTW calculation.
  bool             isInGrid;
  /// @brief The communicator used by parallel FFTW, should be
  /// consistent with inGrid.
  MPI_Comm         comm;

  // plans
  fftw_plan        backwardPlan;
  fftw_plan        forwardPlan;

  unsigned         plannerFlag;

  // Laplacian operator related
  DblNumVec                gkkLocal;
  std::vector<CpxNumVec>   ikLocal;
  DblNumVec                TeterPrecondLocal;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVecLocal;     
  CpxNumVec                outputComplexVecLocal;     

  DistFourier();
  ~DistFourier();

  /// @brief Initialize the FFTW variables.
  /// 
  /// @param[in] dm Domain for the FFTW calculation. 
  /// @param[in] numProc The number of processors actually participate
  /// in the FFTW calculation.  A processor participates in the FFTW
  /// calculation if mpirank(dm.comm) < numProc.
  ///
  /// FIXME: The initialization routine ASSUMES that a fine grid is
  /// generated from the domain. This can be confusing and need to be
  /// fixed later.
  void Initialize( const Domain& dm, Int numProc );
};

} // namespace scales


#endif // _FOURIER_HPP_
