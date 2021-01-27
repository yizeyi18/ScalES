//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Dong An, Weile Jia, Lin Lin

/// @file sgmres.hpp
/// @brief sgmres in the global domain or extended element.
/// @date 2017-10-17
#ifndef _SGMRES_HPP_
#define _SGMRES_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "hamiltonian.hpp"
#include  "spinor.hpp"
#include  "esdf.hpp"

#ifdef _COMPLEX_

namespace scales{

  // Solving linear system Ax = b using GMRES with preconditioning matrix M
  class Sgmres
  {
    public:

      Hamiltonian*        hamPtr_;
      Fourier*            fftPtr_;
      Spinor*             psiPtr_; // please note, the spinor here must be a 1 column psi.

      // input
      Complex* AMat_; // matrix A
      Complex* rhs_;  // vector b 
      double tol_;  // tolerance of GMRES iteration, default 1e-6
      int max_it_;  // maximum iteration of GMRES, default 20
      Complex* Precond_; // predondioning matrix, only supporting diagonal matrix whose only diagonal entries are stored
      int size_;  // dimension of the linear system

      // input & output
      Complex* xVec_; // initial value needed to be given. After GMRES, storing the solution

      // output
      double relres_;  // 2-norm of residue r = Ax - b after GMRES
      int iter_;   // total number of iterations
      bool flag_;  // ''true'' if GMRES converges, ''false'' otherwise

    public:
      Sgmres();
      ~Sgmres(){};

      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        int size);
 
      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        Complex* AMat, 
        Complex* rhs, 
        Complex* xVec, 
        Complex* Precond, 
        int size);
      
      void Setup(
        Hamiltonian& ham,
        Spinor& psi,
        Fourier& fft,
        Complex* AMat, 
        Complex* rhs, 
        Complex* xVec, 
        Complex* Precond, 
        int size, 
        double tol, 
        int max_it);

      //  operation: computing A * yVec = Vout
      //  suggest this function only used in sgmres::Solve, not manually
      void AMatdotVec(
        Complex* yVec, 
        Complex* Vout);

      void AMatdotVec(
        Complex omega,
        Complex* yVec,
        Complex* Vout);

      // operation: solving H * solu = b, where H is an upper triangular matrix
      // suggest this function only used in sgmres::Solve, not manually
      void HSqr(
          Complex* H, 
          Complex* b, 
          Complex* solu, 
          int size);

      // Main action: implementing GMRES
      void Solve( Complex * rhs, Complex *xVec, Complex omega);
  };



} // namespace scales



#endif


#endif
