//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia, and David Williams-Young

/// @file eigensolver.hpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2012-11-20
#ifndef _EIGENSOLVER_HPP_
#define _EIGENSOLVER_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "esdf.hpp"

#include <blacspp/grid.hpp>

namespace scales{

class EigenSolver
{
private:

  Hamiltonian*        hamPtr_;
  Fourier*            fftPtr_;
  Spinor*             psiPtr_;


  DblNumVec           eigVal_;
  DblNumVec           resVal_;


  // Routines for Chebyshev Filtering
  double Cheby_Upper_bound_estimator(DblNumVec& ritz_values, int Num_Lanczos_Steps);
  void Chebyshev_filter(int m, double a, double b);
  void Chebyshev_filter_scaled(int m, double a, double b, double a_L);


  // ScaLAPACK parameters
  Int           scaBlockSize_;
  Int           numProcScaLAPACK_;
  //Int           contxt_;
  //Int           nprow_, npcol_;
  std::unique_ptr<blacspp::Grid> blacs_grid_ = nullptr;



  Int PWDFT_PPCG_use_scala_ ;
  Int PWDFT_Cheby_use_scala_ ;

public:

  // ********************  LIFECYCLE   *******************************

  EigenSolver ();

  ~EigenSolver();

  // ********************  OPERATORS   *******************************

  void Setup(
      Hamiltonian& ham,
      Spinor& psi,
      Fourier& fft );

  // ********************  OPERATIONS  *******************************

  /// @brief Parallel LOBPCG solver with intra-element
  /// parallelization.
  ///
  /// The dense eigenvalue problem can be solved with LAPACK or
  /// ScaLAPACK depending on PWSolver
  ///
  /// @param[in] numEig  Number of eigenvalues to be counted in the
  /// convergence criterion.  numEig must be less than or equal to
  /// width.
  /// @param[in] eigMaxIter    Maximum number of iterations
  /// @param[in] eigMinTolerance Minimum tolerance must be reached
  /// during the LOBPCG iteration
  /// @param[in] eigTolerance  Residual tolerance.
  void LOBPCGSolveReal(
      Int          numEig,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );


  /// @brief Routines for Chebyshev filtering
  void FirstChebyStep(
      Int      numEig,
      Int      eigMaxIter,
      Int        filter_order );

  void GeneralChebyStep(
      Int      numEig,
      Int        filter_order );


  /// @brief Parallel PPCG solver
  /// by ScaLAPACK.
#ifdef DEVICE
  void devicePPCGSolveReal(
      Int          numEig,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance,
      Int          scf_iter );
#endif

  void PPCGSolveReal(
      Int          numEig,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance );


  // ********************  ACCESS      *******************************
  DblNumVec& EigVal() { return eigVal_; }
  DblNumVec& ResVal() { return resVal_; }


  Hamiltonian& Ham()  {return *hamPtr_;}
  Spinor&      Psi()  {return *psiPtr_;}
  Fourier&     FFT()  {return *fftPtr_;}

}; // -----  end of class  EigenSolver  ----- 

} // namespace scales
#endif // _EIGENSOLVER_HPP_
