/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu and Lexing Ying

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
/// @file hamiltonian_dg.cpp
/// @brief Implementation of the Hamiltonian class for DG calculation.
/// @date 2013-01-09
/// @date 2014-08-07 Add intra-element parallelization
#include  "hamiltonian_dg.hpp"
#include  "hamiltonian_dg_conversion.hpp"
#include  "mpi_interf.hpp"
#include  "blas.hpp"

namespace dgdft{

using namespace PseudoComponent;

// *********************************************************************
// Hamiltonian class for DG
// *********************************************************************

HamiltonianDG::HamiltonianDG() {
  XCInitialized_ = false;
}

HamiltonianDG::HamiltonianDG    ( const esdf::ESDFInputParam& esdfParam )
{

  Setup( esdfParam );


  return ;
}         // -----  end of method HamiltonianDG::HamiltonianDG  ----- 


void HamiltonianDG::Setup ( const esdf::ESDFInputParam& esdfParam )
{
  domain_            = esdfParam.domain;
  atomList_          = esdfParam.atomList;
  pseudoType_        = esdfParam.pseudoType;
  numExtraState_     = esdfParam.numExtraState;
  numElem_           = esdfParam.numElem;
  penaltyAlpha_      = esdfParam.penaltyAlpha;
  numLGLGridElem_    = esdfParam.numGridLGL;

  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  if( esdfParam.XCType == "XC_LDA_XC_TETER93" )
  { XCId_ = XC_LDA_XC_TETER93;
    // Teter 93
    // S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996) 
  }    
  else if( esdfParam.XCType == "XC_GGA_XC_PBE" )
  {
    XId_ = XC_GGA_X_PBE;
    CId_ = XC_GGA_C_PBE;
    // Perdew, Burke & Ernzerhof correlation
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
  }
  else
    ErrorHandling("Unrecognized exchange-correlation type");

  Int ntot = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();

  XCInitialized_ = false;

  // Only consider numSpin == 2 in the DG calculation.
  numSpin_ = 2;

  for( Int d = 0; d < DIM; d++ ){
    if( domain_.numGrid[d] % numElem_[d] != 0 ){
      ErrorHandling( 
          "The number of global wfc grid points is not divisible by the number of elements" );
    }
    if( domain_.numGridFine[d] % numElem_[d] != 0 ){
      ErrorHandling(
          "The number of global rho grid points is not divisible by the number of elements" );
    }
    numUniformGridElem_[d] = domain_.numGrid[d] / numElem_[d];
    numUniformGridElemFine_[d] = domain_.numGridFine[d] / numElem_[d];
  }

  dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
  dmRow_ = mpisize / dmCol_;
  if( (mpisize % dmCol_) != 0 ){
    std::ostringstream msg;
    msg << "Total number of processors do not fit to the number processors per element." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  // Setup the element domains
  domainElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );
  for( Int k=0; k< numElem_[2]; k++ )
    for( Int j=0; j< numElem_[1]; j++ )
      for( Int i=0; i< numElem_[0]; i++ ) {
        Index3 key( i, j, k );
        Domain& dm = domainElem_(i, j, k);
        for( Int d = 0; d < DIM; d++ ){
          dm.length[d]     = domain_.length[d] / numElem_[d];
          dm.numGrid[d]    = numUniformGridElem_[d];
          dm.numGridFine[d]    = numUniformGridElemFine_[d];
          dm.posStart[d]   = dm.length[d] * key[d];
        }

        dm.comm    = domain_.rowComm;
        dm.rowComm = domain_.rowComm;
        dm.colComm = domain_.rowComm;

      }


  // Partition by element

  IntNumTns& elemPrtnInfo = elemPrtn_.ownerInfo;
  elemPrtnInfo.Resize( numElem_[0], numElem_[1], numElem_[2] );

  // When intra-element parallelization is invoked, assign one element
  // to processors belong to the same processor row communicator.

  if( mpisize != dmRow_ * dmCol_ ){
    std::ostringstream msg;
    msg << "The number of processors is not equal to the total number of elements." << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  Int cnt = 0;
  for( Int k=0; k< numElem_[2]; k++ )
    for( Int j=0; j< numElem_[1]; j++ )
      for( Int i=0; i< numElem_[0]; i++ ) {
        elemPrtnInfo(i,j,k) = cnt++;
      }

#if ( _DEBUGlevel_ >= 1 )
  for( Int k=0; k< numElem_[2]; k++ )
    for( Int j=0; j< numElem_[1]; j++ )
      for( Int i=0; i< numElem_[0]; i++ ) {
        statusOFS << "Element # " << Index3(i,j,k) << " belongs to proc " << 
          elemPrtn_.Owner(Index3(i,j,k)) << std::endl;
      }
#endif

  // Initialize the DistNumVecs.
  // All quantities below are shared among all processors in the same
  // row communicator, and therefore they only communicate in the column
  // communicators.
  pseudoCharge_.SetComm( domain_.colComm );
  density_.SetComm( domain_.colComm );
  densityLGL_.SetComm( domain_.colComm );
  vext_.SetComm( domain_.colComm );
  vhart_.SetComm( domain_.colComm );
  vxc_.SetComm( domain_.colComm );
  epsxc_.SetComm( domain_.colComm );
  vtot_.SetComm( domain_.colComm );
  vtotLGL_.SetComm( domain_.colComm );

  gradDensity_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    gradDensity_[d].SetComm( domain_.colComm );
    gradDensity_[d].Prtn() = elemPrtn_;
  }

  eigvecCoef_.SetComm( domain_.colComm );
  pseudo_.SetComm( domain_.colComm );
  vnlCoef_.SetComm( domain_.colComm );

  // The exception is basis, for which all processors have distinct
  // values.
  // This also means that collective communication procedure
  // (GetBegin/GetEnd/PutBegin/PutEnd) are not directly used for
  // basisLGL_.
  basisLGL_.SetComm( domain_.comm );

  basisUniformFine_.SetComm( domain_.comm );

  // All quantities follow elemPrtn_, but the communication are only
  // performed in the column communicator.
  pseudoCharge_.Prtn()  = elemPrtn_;
  density_.Prtn()       = elemPrtn_;
  densityLGL_.Prtn()    = elemPrtn_;
  vext_.Prtn()          = elemPrtn_;
  vhart_.Prtn()         = elemPrtn_;
  vxc_.Prtn()           = elemPrtn_;
  epsxc_.Prtn()         = elemPrtn_;
  vtot_.Prtn()          = elemPrtn_;

  // Initialize the quantities shared among processors in the same row
  // communicator.
  for( Int k=0; k< numElem_[2]; k++ )
    for( Int j=0; j< numElem_[1]; j++ )
      for( Int i=0; i< numElem_[0]; i++ ) {
        Index3 key = Index3(i,j,k);
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          DblNumVec  empty( numUniformGridElemFine_.prod() );
          SetValue( empty, 0.0 );
          DblNumVec emptyLGL( numLGLGridElem_.prod() );
          density_.LocalMap()[key]     = empty;
          densityLGL_.LocalMap()[key]  = emptyLGL;
          vext_.LocalMap()[key]        = empty;
          vhart_.LocalMap()[key]       = empty;
          vxc_.LocalMap()[key]         = empty;
          epsxc_.LocalMap()[key]       = empty;
          vtot_.LocalMap()[key]        = empty;
          for( Int d = 0; d < DIM; d++ ){
            gradDensity_[d].LocalMap()[key] = empty;
          }
        } // own this element
      }  // for (i)

  vtotLGL_.Prtn()       = elemPrtn_;
  basisLGL_.Prtn()      = elemPrtn_;
  basisUniformFine_.Prtn()      = elemPrtn_;

  for( Int k=0; k< numElem_[2]; k++ )
    for( Int j=0; j< numElem_[1]; j++ )
      for( Int i=0; i< numElem_[0]; i++ ) {
        Index3 key = Index3(i,j,k);
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          DblNumVec  empty( numLGLGridElem_.prod() );
          SetValue( empty, 0.0 );
          vtotLGL_.LocalMap()[key]        = empty;
        }
      }

  eigvecCoef_.Prtn()    = elemPrtn_;

  // Pseudopotential
  pseudo_.Prtn()        = elemPrtn_;
  vnlCoef_.Prtn()       = elemPrtn_;
  vnlDrvCoef_.resize(DIM);
  for( Int d = 0; d < DIM; d++ ){
    vnlDrvCoef_[d].Prtn() = elemPrtn_;
    vnlDrvCoef_[d].SetComm( domain_.colComm );
  }

  // Partition of the DG matrix
  elemMatPrtn_.ownerInfo = elemPrtn_.ownerInfo;
  // Initialize HMat_
  HMat_.LocalMap().clear();
  HMat_.Prtn() = elemMatPrtn_;
  HMat_.SetComm( domain_.colComm );

  // Partition by atom
  // The ownership of an atom is determined by whether the element
  // containing the atom is owned

  Int numAtom = atomList_.size();

  std::vector<Int>& atomPrtnInfo = atomPrtn_.ownerInfo;
  atomPrtnInfo.resize( numAtom );

  for( Int a = 0; a < numAtom; a++ ){
    Point3 pos = atomList_[a].pos;
    bool isAtomIn = false;
    Index3 idx(-1, -1, -1);
    for( Int k = 0; k < numElem_[2]; k++ ){
      for( Int j = 0; j < numElem_[1]; j++ ){
        for( Int i = 0; i < numElem_[0]; i++ ){
          isAtomIn = IsInSubdomain( pos, domainElem_(i, j, k), domain_.length );
          if( isAtomIn == true ){
            idx = Index3(i, j, k);
            break;
          }
        }
      }
    }
    if( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ){
      std::ostringstream msg;
      msg << "Cannot find element for atom #" << a << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    // Determine the ownership by the ownership of the corresponding element
    atomPrtnInfo[a] = elemPrtn_.Owner( idx );
  } // for (a)

#if ( _DEBUGlevel_ >= 1 )
  for( Int a = 0; a < numAtom; a++ ){
    statusOFS << "Atom # " << a << " belongs to proc " << 
      atomPrtn_.Owner(a) << std::endl;
  }
#endif

  // Generate the grids
  //
  // When dual grid is used, the fine grid is used for quantities such
  // as density and potential.  The coarse grid is used for
  // wavefunctions (basis functions)

  UniformMesh( domain_, uniformGrid_ );
  UniformMeshFine( domain_, uniformGridFine_ );

  uniformGridElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );
  uniformGridElemFine_.Resize( numElem_[0], numElem_[1], numElem_[2] );

  LGLGridElem_.Resize( numElem_[0], numElem_[1], numElem_[2] );

  for( Int k = 0; k < numElem_[2]; k++ ){
    for( Int j = 0; j < numElem_[1]; j++ ){
      for( Int i = 0; i < numElem_[0]; i++ ){
        UniformMesh( domainElem_(i, j, k), 
            uniformGridElem_(i, j, k) );
        UniformMeshFine( domainElem_(i, j, k),
            uniformGridElemFine_(i, j, k) );
        LGLMesh( domainElem_(i, j, k),
            numLGLGridElem_,
            LGLGridElem_(i, j, k) );
      }
    }
  }

#if ( _DEBUGlevel_ >= 1 )
  for( Int k = 0; k < numElem_[2]; k++ ){
    for( Int j = 0; j < numElem_[1]; j++ ){
      for( Int i = 0; i < numElem_[0]; i++ ){
        statusOFS << "Uniform grid for element " << Index3(i,j,k) << std::endl;
        for( Int d = 0; d < DIM; d++ ){
          statusOFS << uniformGridElem_(i,j,k)[d] << std::endl;
        }
        statusOFS << "Uniform Fine grid for element " << Index3(i,j,k) << std::endl;
        for( Int d = 0; d < DIM; d++ ){
          statusOFS << uniformGridElemFine_(i,j,k)[d] << std::endl;
        }
        statusOFS << "LGL grid for element " << Index3(i,j,k) << std::endl;
        for( Int d = 0; d < DIM; d++ ){
          statusOFS << LGLGridElem_(i,j,k)[d] << std::endl;
        }
      }
    }
  } // for (k)
#endif

  // Generate the differentiation matrix on the LGL grid
  // NOTE: This assumes uniform mesh used for each element.
  DMat_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    DblNumVec  dummyX, dummyW;
    DblNumMat  dummyP;
    GenerateLGL( dummyX, dummyW, dummyP, DMat_[d], 
        numLGLGridElem_[d] );

    // Scale the differentiation matrix
    blas::Scal( numLGLGridElem_[d] * numLGLGridElem_[d],
        2.0 / (domain_.length[d] / numElem_[d]), 
        DMat_[d].Data(), 1 );
  }

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Sanity check " << std::endl;

  for( Int d = 0; d < DIM; d++ ){
    DblNumVec t( numLGLGridElem_[d] );
    DblNumVec s( numLGLGridElem_[d] );
    SetValue(t, 1.0);
    blas::Gemm( 'N', 'N', numLGLGridElem_[d], 1, numLGLGridElem_[d],
        1.0, DMat_[d].Data(), numLGLGridElem_[d],
        t.Data(), numLGLGridElem_[d], 0.0,
        s.Data(), numLGLGridElem_[d] );
    statusOFS << "Derivative of constant along dimension " << d 
      << " gives  " << s << std::endl;
  }
#endif

  // Generate the transfer matrix from LGL grid to uniform grid on each
  // element. The Lagrange polynomials involved in the transfer matrix
  // is computed using the Barycentric method.  For more information see
  //
  // [J.P. Berrut and L.N. Trefethen, Barycentric Lagrange Interpolation,
  // SIAM Rev. 2004]
  //
  // NOTE: This assumes uniform mesh used for each element.
  {
    LGLToUniformMat_.resize(DIM);
    LGLToUniformMatFine_.resize(DIM);
    Index3& numLGL                      = numLGLGridElem_;
    Index3& numUniform                  = numUniformGridElem_;
    Index3& numUniformFine              = numUniformGridElemFine_;
    // Small stablization parameter 
    const Real EPS                      = 1e-13; 
    for( Int d = 0; d < DIM; d++ ){
      DblNumVec& LGLGrid     = LGLGridElem_(0,0,0)[d];
      DblNumVec& uniformGrid = uniformGridElem_(0,0,0)[d];
      DblNumVec& uniformGridFine = uniformGridElemFine_(0,0,0)[d];
      // Stablization constant factor, according to Berrut and Trefethen
      Real    stableFac = 0.25 * domainElem_(0,0,0).length[d];
      DblNumMat& localMat = LGLToUniformMat_[d];
      DblNumMat& localMatFine = LGLToUniformMatFine_[d];
      localMat.Resize( numUniform[d], numLGL[d] );
      localMatFine.Resize( numUniformFine[d], numLGL[d] );
      DblNumVec lambda( numLGL[d] );
      DblNumVec denom( numUniform[d] );
      DblNumVec denomFine( numUniformFine[d] );
      SetValue( lambda, 0.0 );
      SetValue( denom, 0.0 );
      SetValue( denomFine, 0.0 );
      for( Int i = 0; i < numLGL[d]; i++ ){
        lambda[i] = 1.0;
        for( Int j = 0; j < numLGL[d]; j++ ){
          if( j != i ) 
            lambda[i] *= (LGLGrid[i] - LGLGrid[j]) / stableFac; 
        } // for (j)
        lambda[i] = 1.0 / lambda[i];
        for( Int j = 0; j < numUniform[d]; j++ ){
          denom[j] += lambda[i] / ( uniformGrid[j] - LGLGrid[i] + EPS );
        }
        for( Int j = 0; j < numUniformFine[d]; j++ ){
          denomFine[j] += lambda[i] / ( uniformGridFine[j] - LGLGrid[i] + EPS );
        }
      } // for (i)

      for( Int i = 0; i < numLGL[d]; i++ ){
        for( Int j = 0; j < numUniform[d]; j++ ){
          localMat( j, i ) = (lambda[i] / ( uniformGrid[j] - LGLGrid[i]
                + EPS )) / denom[j]; 
        } // for (j)
        for( Int j = 0; j < numUniformFine[d]; j++ ){
          localMatFine( j, i ) = (lambda[i] / ( uniformGridFine[j] - LGLGrid[i]
                + EPS )) / denomFine[j];
        } // for (j)
      } // for (i)
    } // for (d)
  }

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "LGLToUniformMat[0] = " << LGLToUniformMat_[0] << std::endl;
  statusOFS << "LGLToUniformMat[1] = " << LGLToUniformMat_[1] << std::endl; 
  statusOFS << "LGLToUniformMat[2] = " << LGLToUniformMat_[2] << std::endl; 
  statusOFS << "LGLToUniformMatFine[0] = " << LGLToUniformMatFine_[0] << std::endl;
  statusOFS << "LGLToUniformMatFine[1] = " << LGLToUniformMatFine_[1] << std::endl; 
  statusOFS << "LGLToUniformMatFine[2] = " << LGLToUniformMatFine_[2] << std::endl; 
#endif


  // Generate the transfer matrix from LGL grid to uniform grid on each
  // element with the Gaussian convolution interpolation method. 
  //
  // NOTE: This assumes uniform mesh used for each element.

  if(0){

    Real timeSta, timeEnd;

    Index3& numLGL                      = numLGLGridElem_;
    Index3& numUniform                  = numUniformGridElem_;
    Index3& numUniformFine              = numUniformGridElemFine_;
    // Small stablization parameter 
    const Real EPS                      = 1e-13;

    Real interpFactor = esdfParam.GaussInterpFactor;
    Real sigma = esdfParam.GaussSigma;
    Index3 NInterp;

    for( Int d = 0; d < DIM; d++ ){
      NInterp[d] = int(ceil(domain_.length[d] / numElem_[d] * interpFactor / sigma)); 
    }

    GetTime( timeSta );

    std::vector<DblNumVec>   LGLInterpGridElem;
    std::vector<DblNumVec> LGLInterpWeight1D;
    LGLInterpGridElem.resize(DIM);
    LGLInterpWeight1D.resize(DIM);

    Point3 length       = domainElem_(0,0,0).length;

    GetTime( timeSta );

    for( Int d = 0; d < DIM; d++ ){
      DblNumVec&  dummyX = LGLInterpGridElem[d];
      Domain& dm =  domainElem_(0, 0, 0);
      GenerateLGLMeshWeightOnly( dummyX, LGLInterpWeight1D[d], NInterp[d] );
      //DblNumMat  dummyP, dummpD;
      //GenerateLGL( dummyX, LGLInterpWeight1D[d], dummyP, dummpD, NInterp[d] );
      blas::Scal( NInterp[d], 0.5 * length[d], LGLInterpWeight1D[d].Data(), 1 );

      for( Int i = 0; i < NInterp[d]; i++ ){
        dummyX[i] = dm.posStart[d] + ( dummyX[i] + 1.0 ) * dm.length[d] * 0.5;
      }
    }

    GetTime( timeEnd );

    statusOFS << "Time for GenerateLGL = " << timeEnd - timeSta << std::endl;

    //Domain dmExtElem;
    Index3& numExtElem = numExtElem_; // The number of element in externd element

    // Setup the domain in the extended element
    for( Int k=0; k< numElem_[2]; k++ )
      for( Int j=0; j< numElem_[1]; j++ )
        for( Int i=0; i< numElem_[0]; i++ ) {
          Index3 key = Index3(i,j,k);
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            for( Int d = 0; d < DIM; d++ ){
              if( numElem_[d] == 1 ){
                numExtElem[d] = 1;
              }
              else if ( numElem_[d] >= 3 ){
                numExtElem[d] = 3;
              }
              else{
                ErrorHandling( "numElem[d] is either 1 or >=3." );
              }
            } // for d
          }
        }

    //LGLToUniformGaussMat_.Resize( numExtElem[0], numExtElem[1], numExtElem[2] );
    LGLToUniformGaussMatFine_.Resize( numExtElem[0], numExtElem[1], numExtElem[2] );

    Point3 posStartElem;

    for( Int kk = 0; kk < numExtElem[2]; kk++ ){
      for( Int jj = 0; jj < numExtElem[1]; jj++ ){
        for( Int ii = 0; ii < numExtElem[0]; ii++ ){

          // std::vector<DblNumMat>& LGLToUniformGaussMat     = LGLToUniformGaussMat_ (ii,jj,kk);
          std::vector<DblNumMat>& LGLToUniformGaussMatFine = LGLToUniformGaussMatFine_ (ii,jj,kk);

          //LGLToUniformGaussMat.resize(DIM);
          LGLToUniformGaussMatFine.resize(DIM);

          posStartElem[0] = length[0] * (ii-1);
          posStartElem[1] = length[1] * (jj-1);
          posStartElem[2] = length[2] * (kk-1);

          for( Int d = 0; d < DIM; d++ ){
            if (numExtElem[d] == 1){
              posStartElem[d] = 0;
            }
          }

          for( Int d = 0; d < DIM; d++ ){

            DblNumVec& LGLGrid             = LGLGridElem_(0,0,0)[d];
            DblNumVec& uniformGridTemp     = uniformGridElem_(0,0,0)[d];
            DblNumVec& uniformGridFineTemp = uniformGridElemFine_(0,0,0)[d];
            DblNumVec& LGLInterpGrid       = LGLInterpGridElem[d];

            // DblNumVec  uniformGrid(numUniform[d]);
            DblNumVec  uniformGridFine(numUniformFine[d]);

            //for( Int i = 0; i < numUniform[d]; i++ ){
            //  uniformGrid[i] = uniformGridTemp[i] + posStartElem[d];
            //}

            for( Int i = 0; i < numUniformFine[d]; i++ ){
              uniformGridFine[i] = uniformGridFineTemp[i] + posStartElem[d];
            }

            // Stablization constant factor, according to Berrut and Trefethen
            Real    stableFac = 0.25 * domainElem_(0,0,0).length[d];
            //DblNumMat& localMat = LGLToUniformGaussMat[d];
            DblNumMat& localMatFine = LGLToUniformGaussMatFine[d];
            //localMat.Resize( numUniform[d], numLGL[d] );
            localMatFine.Resize( numUniformFine[d], numLGL[d] );
            DblNumVec lambda( numLGL[d] );
            DblNumVec denom( NInterp[d] );
            SetValue( lambda, 0.0 );
            SetValue( denom, 0.0 );

            DblNumMat LGLMat;
            LGLMat.Resize ( NInterp[d], numLGL[d] ); 
            SetValue( LGLMat, 0.0 );

            for( Int i = 0; i < numLGL[d]; i++ ){
              lambda[i] = 1.0;
              for( Int j = 0; j < numLGL[d]; j++ ){
                if( j != i ) 
                  lambda[i] *= (LGLGrid[i] - LGLGrid[j]) / stableFac; 
              } // for (j)
              lambda[i] = 1.0 / lambda[i];
              for( Int k = 0; k < NInterp[d]; k++ ){
                denom[k] += lambda[i] / ( LGLInterpGrid[k] - LGLGrid[i] + EPS );
              }
            } // for (i)

            for( Int j = 0; j < numLGL[d]; j++ ){
              for( Int i = 0; i < NInterp[d]; i++ ){
                LGLMat( i, j )  = ( lambda[j] / ( LGLInterpGrid[i] - LGLGrid[j] + EPS ) ) / denom[i];

              }
            }

            DblNumVec& LGLInterpWeight = LGLInterpWeight1D[d];

            // Generate the Gaussian matrix by numerical integration
            Real guassFac = 1.0/sqrt(2.0*PI*sigma*sigma);

            if (0) {

              for( Int i = 0; i < numUniformFine[d]; i++ ){
                for( Int j = 0; j < numLGL[d]; j++ ){
                  Real sum=0.0;
                  for( Int k = 0; k < NInterp[d]; k++ ){
                    Real dist=0.0; 
                    dist = uniformGridFine[i] - LGLInterpGrid[k];
                    dist = dist - round(dist/domain_.length[d])*domain_.length[d];
                    sum += exp(-dist*dist/(2*sigma*sigma)) * LGLMat(k,j) * LGLInterpWeight(k);
                  } // for (k)
                  localMatFine( i, j ) = guassFac * sum;
                } // for (j)
              } // for (i)

            } // if (0)

            if (1) {

              DblNumMat guassTemp;
              guassTemp.Resize ( numUniformFine[d], NInterp[d]); 
              SetValue( guassTemp, 0.0 );

              for( Int k = 0; k < NInterp[d]; k++ ){
                for( Int i = 0; i < numUniformFine[d]; i++ ){
                  Real dist=0.0; 
                  dist = uniformGridFine[i] - LGLInterpGrid[k];
                  dist = dist - round(dist/domain_.length[d])*domain_.length[d];
                  guassTemp(i, k) = exp(-dist*dist/(2*sigma*sigma)) * LGLInterpWeight(k) * guassFac;
                }
              } 

              // Use Gemm
              Int m = numUniformFine[d], n = numLGL[d], k = NInterp[d];
              blas::Gemm( 'N', 'N', m, n, k, 1.0, guassTemp.Data(),
                  m, LGLMat.Data(), k, 0.0, localMatFine.Data(), m );

            } // if (1)

          } // for (d)

        } // for ii
      } // for jj
    } // for kk

  } // for Gaussian convolution interpolation


  // Compute the LGL weights at 1D, 2D, 3D
  {
    Point3 length       = domainElem_(0,0,0).length;
    Index3 numGrid      = numLGLGridElem_;             
    Int    numGridTotal = numGrid.prod();

    // Compute the integration weights
    // 1D
    LGLWeight1D_.resize(DIM);

    for( Int d = 0; d < DIM; d++ ){
      DblNumVec  dummyX;
      DblNumMat  dummyP, dummpD;
      GenerateLGL( dummyX, LGLWeight1D_[d], dummyP, dummpD, 
          numGrid[d] );
      blas::Scal( numGrid[d], 0.5 * length[d], 
          LGLWeight1D_[d].Data(), 1 );
    }

    // 2D: faces labeled by normal vectors, i.e. 
    // yz face : 0
    // xz face : 1
    // xy face : 2

    LGLWeight2D_.resize(DIM);

    // yz face
    LGLWeight2D_[0].Resize( numGrid[1], numGrid[2] );
    for( Int k = 0; k < numGrid[2]; k++ )
      for( Int j = 0; j < numGrid[1]; j++ ){
        LGLWeight2D_[0](j, k) = LGLWeight1D_[1](j) * LGLWeight1D_[2](k);
      } // for (j)

    // xz face
    LGLWeight2D_[1].Resize( numGrid[0], numGrid[2] );
    for( Int k = 0; k < numGrid[2]; k++ )
      for( Int i = 0; i < numGrid[0]; i++ ){
        LGLWeight2D_[1](i, k) = LGLWeight1D_[0](i) * LGLWeight1D_[2](k);
      } // for (i)

    // xy face
    LGLWeight2D_[2].Resize( numGrid[0], numGrid[1] );
    for( Int j = 0; j < numGrid[1]; j++ )
      for( Int i = 0; i < numGrid[0]; i++ ){
        LGLWeight2D_[2](i, j) = LGLWeight1D_[0](i) * LGLWeight1D_[1](j);
      }


    // 3D
    LGLWeight3D_.Resize( numGrid[0], numGrid[1], numGrid[2] );
    for( Int k = 0; k < numGrid[2]; k++ )
      for( Int j = 0; j < numGrid[1]; j++ )
        for( Int i = 0; i < numGrid[0]; i++ ){
          LGLWeight3D_(i, j, k) = LGLWeight1D_[0](i) * LGLWeight1D_[1](j) *
            LGLWeight1D_[2](k);
        } // for (i)

  }

  // Initialize the XC functional.  
  // Spin-unpolarized functional is used here
  //if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
  //  ErrorHandling( "XC functional initialization error." );
  //} 

  xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED);
  xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED);
  xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED);

  if( XCId_ == 20 )
  {
    if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
      ErrorHandling( "XC functional initialization error." );
    } 
  }    
  else if( ( XId_ == 101 ) && ( CId_ == 130 )  )
  {
    if( ( xc_func_init(&XFuncType_, XId_, XC_UNPOLARIZED) != 0 )
        && ( xc_func_init(&CFuncType_, CId_, XC_UNPOLARIZED) != 0 ) ){
      ErrorHandling( "XC functional initialization error." );
    }
  }
  else
    ErrorHandling("Unrecognized exchange-correlation type");

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          Int numBasisLGLTotal = esdfParam.numALBElem(i,j,k);
          Int numBasisLGLBlocksize = numBasisLGLTotal / mpisizeRow;
          Int numBasisLGLLocal = numBasisLGLBlocksize;

          if(mpirankRow < (numBasisLGLTotal % mpisizeRow)){
            numBasisLGLLocal = numBasisLGLBlocksize + 1;
          }
          basisLGLIdx_.Resize( numBasisLGLLocal );
          SetValue( basisLGLIdx_, 0 );
          for (Int i = 0; i < numBasisLGLLocal; i++){
            basisLGLIdx_[i] = i * mpisizeRow + mpirankRow ;
          }
        }
      }



  return ;
}         // -----  end of method HamiltonianDG::Setup  ----- 

void
HamiltonianDG::UpdateHamiltonianDG    ( std::vector<Atom>& atomList )
{

  atomList_          = atomList;

  // Repartitioning the atoms according to the new coordinate

  Int numAtom = atomList_.size();

  std::vector<Int>& atomPrtnInfo = atomPrtn_.ownerInfo;
  atomPrtnInfo.resize( numAtom );

  for( Int a = 0; a < numAtom; a++ ){
    Point3 pos = atomList_[a].pos;
    bool isAtomIn = false;
    Index3 idx(-1, -1, -1);
    for( Int k = 0; k < numElem_[2]; k++ ){
      for( Int j = 0; j < numElem_[1]; j++ ){
        for( Int i = 0; i < numElem_[0]; i++ ){
          isAtomIn = IsInSubdomain( pos, domainElem_(i, j, k), domain_.length );
          if( isAtomIn == true ){
            idx = Index3(i, j, k);
            break;
          }
        }
      }
    }
    if( idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ){
      std::ostringstream msg;
      msg << "Cannot find element for atom #" << a << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    // Determine the ownership by the ownership of the corresponding element
    atomPrtnInfo[a] = elemPrtn_.Owner( idx );
  } // for (a)

#if ( _DEBUGlevel_ >= 1 )
  for( Int a = 0; a < numAtom; a++ ){
    statusOFS << "Atom # " << a << " belongs to proc " << 
      atomPrtn_.Owner(a) << std::endl;
  }
#endif


  return ;
}         // -----  end of method HamiltonianDG::UpdateHamiltonianDG  ----- 

HamiltonianDG::~HamiltonianDG    ( )
{

  if( XCInitialized_ )
  {
    if( XCId_ == 20 )
    {
      xc_func_end(&XCFuncType_);
    }    
    else if( ( XId_ == 101 ) && ( CId_ == 130 )  )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else
      ErrorHandling("Unrecognized exchange-correlation type");
  }

}         // -----  end of method HamiltonianDG::HamiltonianDG  ----- 

void
HamiltonianDG::DiffPsi    (const Index3& numGrid, const Real* psi, Real* Dpsi, Int d)
{
  if( d == 0 ){
    // Use Gemm
    Int m = numGrid[0], n = numGrid[1]*numGrid[2];
    blas::Gemm( 'N', 'N', m, n, m, 1.0, DMat_[0].Data(),
        m, psi, m, 0.0, Dpsi, m );
  }
  else if( d == 1 ){
    // Middle dimension, use Gemv
    Int   m = numGrid[1], n = numGrid[0]*numGrid[2];
    Int   ptrShift;
    Int   inc = numGrid[0];
    for( Int k = 0; k < numGrid[2]; k++ ){
      for( Int i = 0; i < numGrid[0]; i++ ){
        ptrShift = i + k * numGrid[0] * numGrid[1];
        blas::Gemv( 'N', m, m, 1.0, DMat_[1].Data(), m, 
            psi + ptrShift, inc, 0.0, Dpsi + ptrShift, inc );
      }
    } // for (k)
  }
  else if ( d == 2 ){
    // Use Gemm
    Int m = numGrid[0]*numGrid[1], n = numGrid[2];
    blas::Gemm( 'N', 'T', m, n, n, 1.0, psi, m,
        DMat_[2].Data(), n, 0.0, Dpsi, m );
  }
  else{
    ErrorHandling("Wrong dimension.");
  }


  return ;
}         // -----  end of method HamiltonianDG::DiffPsi  ----- 


void
HamiltonianDG::InterpLGLToUniform    ( const Index3& numLGLGrid, const
    Index3& numUniformGridFine, const Real* rhoLGL, Real* rhoUniform )
{
  Index3 Ns1 = numLGLGrid;
  Index3 Ns2 = numUniformGridFine;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, LGLToUniformMatFine_[0].Data(),
        m, rhoLGL, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            LGLToUniformMatFine_[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        LGLToUniformMatFine_[2].Data(), n, 0.0, rhoUniform, m );
  }



  return ;
}         // -----  end of method HamiltonianDG::InterpLGLToUniform  ----- 


void
HamiltonianDG::GaussConvInterpLGLToUniform    ( const Index3& numLGLGrid, const
    Index3& numUniformGridFine, const Real* rhoLGL, Real* rhoUniform, 
    std::vector<DblNumMat> LGLToUniformGaussMatFine )
{
  Index3 Ns1 = numLGLGrid;
  Index3 Ns2 = numUniformGridFine;

  DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
  DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
  SetValue( tmp1, 0.0 );
  SetValue( tmp2, 0.0 );

  // x-direction, use Gemm
  {
    Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
    blas::Gemm( 'N', 'N', m, n, k, 1.0, LGLToUniformGaussMatFine[0].Data(),
        m, rhoLGL, k, 0.0, tmp1.Data(), m );
  }

  // y-direction, use Gemv
  {
    Int   m = Ns2[1], n = Ns1[1];
    Int   rhoShift1, rhoShift2;
    Int   inc = Ns2[0];
    for( Int k = 0; k < Ns1[2]; k++ ){
      for( Int i = 0; i < Ns2[0]; i++ ){
        rhoShift1 = i + k * Ns2[0] * Ns1[1];
        rhoShift2 = i + k * Ns2[0] * Ns2[1];
        blas::Gemv( 'N', m, n, 1.0, 
            LGLToUniformGaussMatFine[1].Data(), m, 
            tmp1.Data() + rhoShift1, inc, 0.0, 
            tmp2.Data() + rhoShift2, inc );
      } // for (i)
    } // for (k)
  }


  // z-direction, use Gemm
  {
    Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
    blas::Gemm( 'N', 'T', m, n, k, 1.0, 
        tmp2.Data(), m, 
        LGLToUniformGaussMatFine[2].Data(), n, 0.0, rhoUniform, m );
  }



  return ;
}         // -----  end of method HamiltonianDG::GaussConvInterpLGLToUniform  ----- 


void
HamiltonianDG::CalculatePseudoPotential    ( PeriodTable &ptable ){
  Int ntotFine = domain_.NumGridTotalFine();
  Int numAtom = atomList_.size();
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Real vol = domain_.Volume();

  // Very important especially when updating the atomic positions
  pseudo_.LocalMap().clear();

  // *********************************************************************
  // Atomic information
  // *********************************************************************

  // Calculate the number of occupied states
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      std::ostringstream msg;
      msg << "Cannot find the atom type for atom #" << a << std::endl;
      ErrorHandling( msg.str().c_str() );
    }
    nelec = nelec + ptable.ptemap()[atype].params(PTParam::ZION);
  }

  if( nelec % 2 != 0 ){
    ErrorHandling( "This is a spin-restricted calculation. nelec should be even." );
  }

  numOccupiedState_ = nelec / numSpin_;

#if ( _DEBUGlevel_ >= 0 )
  Print( statusOFS, "Number of Occupied States                    = ", numOccupiedState_ );
#endif

  // *********************************************************************
  // Generate the atomic pseudopotentials
  // *********************************************************************

  // Check that the cutoff radius of the pseudopotential is smaller than
  // the length of the element.
  {
    Real minLength = std::min( domainElem_(0,0,0).length[0],
        std::min( domainElem_(0,0,0).length[1], domainElem_(0,0,0).length[2] ) );
    Real Rzero;
    for( Int a = 0; a < numAtom; a++ ){
      Int type = atomList_[a].type;
      // For the case where there is no nonlocal pseudopotential
      if(ptable.ptemap()[type].cutoffs.m()>PTSample::NONLOCAL)      
        Rzero = ptable.ptemap()[type].cutoffs(PTSample::NONLOCAL);
      else
        Rzero = 0.0;

      if( Rzero >= minLength ){
        std::ostringstream msg;
        msg << "In order for the current DG partition to work, " 
          << "the support of the nonlocal pseudopotential must be smaller than "
          << "the length of the element along each dimension.  " << std::endl
          << "It is now found for atom " << a << ", which is of type " << type 
          << ", Rzero = " << Rzero << std::endl
          << "while the length of the element is " 
          << domainElem_(0,0,0).length[0] << " x " 
          << domainElem_(0,0,0).length[1] << " x " 
          << domainElem_(0,0,0).length[2] << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }
  }

  // Also prepare the integration weights for constructing the DG matrix later.
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << std::endl << "Prepare the integration weights for constructing the DG matrix." << std::endl;
#endif

  vnlWeightMap_.clear();
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          // std::map<Int, PseudoPot>  ppMap;
          std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
          std::vector<DblNumVec>&    gridpos = uniformGridElemFine_( i, j, k );
          for( Int a = 0; a < numAtom; a++ ){
            PTEntry& ptentry = ptable.ptemap()[atomList_[a].type];
            // Cutoff radius: Take the largest one
            Real Rzero = ptentry.cutoffs( PTSample::PSEUDO_CHARGE );
            if(ptentry.cutoffs.m()>PTSample::NONLOCAL)      
              Rzero = std::max( Rzero, ptentry.cutoffs(PTSample::NONLOCAL) );

            // Compute the minimum distance of this atom to all grid points
            Point3 minDist;
            Point3& length = domain_.length;
            Point3& pos    = atomList_[a].pos;
            for( Int d = 0; d < DIM; d++ ){
              minDist[d] = Rzero;
              Real dist;
              for( Int i = 0; i < gridpos[d].m(); i++ ){
                dist = gridpos[d](i) - pos[d];
                dist = dist - IRound( dist / length[d] ) * length[d];
                if( std::abs( dist ) < minDist[d] )
                  minDist[d] = std::abs( dist );
              }
            }
            // If this atom overlaps with this element, compute the pseudopotential
            if( minDist.l2() <= Rzero ){
              PseudoPot   pp;
              ptable.CalculatePseudoCharge( atomList_[a], 
                  domain_, uniformGridElemFine_(i, j, k), pp.pseudoCharge );
              ptable.CalculateNonlocalPP( atomList_[a], 
                  domain_, LGLGridElem_(i, j, k), pp.vnlList );
              ppMap[a] = pp;
              DblNumVec   weight( pp.vnlList.size() );
              for( Int l = 0; l < weight.Size(); l++ ){
                weight(l) = pp.vnlList[l].second;
              }
              vnlWeightMap_[a] = weight;
            }
          } // for (a)
          //pseudo_.LocalMap()[key] = ppMap;
        } // own this element
      } // for (i)

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << std::endl << "Atomic pseudocharge computed." << std::endl;
#endif

  // *********************************************************************
  // Local pseudopotential: computed by pseudocharge
  // *********************************************************************

  // Compute the pseudocharge by summing over contributions from all atoms
  {
    Real localSum = 0.0, sumRho = 0.0;

    pseudoCharge_.LocalMap().clear();

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
            DblNumVec  localVec( numUniformGridElemFine_.prod() );
            SetValue( localVec, 0.0 );
            for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
                mi != ppMap.end(); mi++ ){
              Int atomIdx = (*mi).first;
              PseudoPot& pp = (*mi).second;
              SparseVec& sp = pp.pseudoCharge;
              IntNumVec& idx = sp.first;
              DblNumMat& val = sp.second;
              for( Int l = 0; l < idx.m(); l++ ){
                localVec[idx(l)] += val(l, VAL);
              }
            } // for (mi)
            pseudoCharge_.LocalMap()[key] = localVec;
          } // own this element
        } // for (i)

    // Compute the sum of the pseudocharge and make adjustment.

    for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
        mi != pseudoCharge_.LocalMap().end(); mi++ ){
      DblNumVec& vec = (*mi).second;
      for( Int i = 0; i < vec.m(); i++ ){
        localSum += vec[i];
      }
    }

    localSum *= domain_.Volume() / domain_.NumGridTotalFine();

    mpi::Allreduce( &localSum, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    Print( statusOFS, "Sum of Pseudocharge                          = ", sumRho );
    Print( statusOFS, "numOccupiedState                             = ", 
        numOccupiedState_ );
#endif

    // Make adjustments to the pseudocharge
    Real fac = numSpin_ * numOccupiedState_ / sumRho;

    for( std::map<Index3, DblNumVec>::iterator mi = pseudoCharge_.LocalMap().begin();
        mi != pseudoCharge_.LocalMap().end(); mi++ ){
      DblNumVec& vec = (*mi).second;
      for( Int i = 0; i < vec.m(); i++ ){
        vec[i] *= fac;
      }
    }

#if ( _DEBUGlevel_ >= 0 )
    Print( statusOFS, "After adjustment, sum of Pseudocharge        = ", 
        (Real) numSpin_ * numOccupiedState_ );
#endif
  }


  return ;
}         // -----  end of method HamiltonianDG::CalculatePseudoPotential  ----- 

void
HamiltonianDG::CalculateDensity    ( 
    DistDblNumVec& rho,
    DistDblNumVec& rhoLGL )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  MPI_Barrier(domain_.rowComm);
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  MPI_Barrier(domain_.colComm);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  DblNumVec& occrate = occupationRate_;
  Int numEig = occrate.m();

  DistDblNumVec  psiUniform;
  psiUniform.Prtn() = elemPrtn_;

  // Clear the density
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec& localRho = rho.LocalMap()[key];
          DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];
          SetValue( localRho, 0.0 );
          SetValue( localRhoLGL, 0.0 );
        } // own this element
      } // for (i)

  // Method 1: Normalize each eigenfunctions.  This may take many
  // interpolation steps, and the communication cost may be large
  if(0)
  {
    // Loop over all the eigenfunctions
    for( Int g = 0; g < numEig; g++ ){
      // Normalization constants
      Real normPsiLocal  = 0.0;
      Real normPsi       = 0.0;
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumMat& localBasis = basisLGL_.LocalMap()[key];
              Int numGrid  = localBasis.m();
              Int numBasis = localBasis.n();

              DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
              if( localCoef.n() != numEig ){
                ErrorHandling( 
                    "Numbers of eigenfunction coefficients do not match.");
              }
              if( localCoef.m() != numBasis ){
                ErrorHandling(
                    "Number of LGL grids do not match.");
              }
              DblNumVec  localPsiLGL( numGrid );
              DblNumVec  localPsiUniformFine( numUniformGridElemFine_.prod() );
              SetValue( localPsiLGL, 0.0 );

              // Compute local wavefunction on the LGL grid
              blas::Gemv( 'N', numGrid, numBasis, 1.0, 
                  localBasis.Data(), numGrid, 
                  localCoef.VecData(g), 1, 0.0,
                  localPsiLGL.Data(), 1 );

              // Interpolate local wavefunction from LGL grid to uniform fine grid
              InterpLGLToUniform( 
                  numLGLGridElem_, 
                  numUniformGridElemFine_, 
                  localPsiLGL.Data(), 
                  localPsiUniformFine.Data() );

              // Compute the local norm
              normPsiLocal += Energy( localPsiUniformFine );

              psiUniform.LocalMap()[key] = localPsiUniformFine;

            } // own this element
          } // for (i)

      // All processors get the normalization factor
      mpi::Allreduce( &normPsiLocal, &normPsi, 1, MPI_SUM, domain_.comm );

      // pre-constant in front of psi^2 for density
      Real rhofac = (numSpin_ * domain_.NumGridTotalFine() / domain_.Volume() ) 
        * occrate[g] / normPsi;

      // Add the normalized wavefunction to density
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localRho = rho.LocalMap()[key];
              DblNumVec& localPsi = psiUniform.LocalMap()[key];
              for( Int p = 0; p < localRho.Size(); p++ ){
                localRho[p] += localPsi[p] * localPsi[p] * rhofac;
              }    
            } // own this element
          } // for (i)
    } // for (g)
    // Check the sum of the electron density
#if ( _DEBUGlevel_ >= 0 )
    Real sumRhoLocal = 0.0;
    Real sumRho      = 0.0;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho = rho.LocalMap()[key];
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += localRho[p];
            }    
          } // own this element
        } // for (i)
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

    sumRho *= domain_.Volume() / domain_.NumGridTotalFine();

    Print( statusOFS, "Sum rho = ", sumRho );
#endif
  } // Method 1


  // Method 2: Compute the electron density locally, and then normalize
  // only in the global domain. The result should be almost the same as
  // that in Method 1, but should be much faster.
  if(0)
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisLGL_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();

            // Skip the element if there is no basis functions.
            if( numBasis == 0 )
              continue;

            DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
            if( localCoef.n() != numEig ){
              ErrorHandling( 
                  "Numbers of eigenfunction coefficients do not match.");
            }
            if( localCoef.m() != numBasis ){
              ErrorHandling(
                  "Number of LGL grids do not match.");
            }

            DblNumVec& localRho = rho.LocalMap()[key];

            DblNumVec  localRhoLGL( numGrid );
            DblNumVec  localPsiLGL( numGrid );
            SetValue( localRhoLGL, 0.0 );
            SetValue( localPsiLGL, 0.0 );

            // Loop over all the eigenfunctions
            // 
            // NOTE: Gemm is not a feasible choice when a large number of
            // eigenfunctions are there.
            for( Int g = 0; g < numEig; g++ ){
              // Compute local wavefunction on the LGL grid
              blas::Gemv( 'N', numGrid, numBasis, 1.0, 
                  localBasis.Data(), numGrid, 
                  localCoef.VecData(g), 1, 0.0,
                  localPsiLGL.Data(), 1 );
              // Update the local density
              Real  occ    = occrate[g];
              for( Int p = 0; p < numGrid; p++ ){
                localRhoLGL(p) += pow( localPsiLGL(p), 2.0 ) * occ;
              }
            }

            // Interpolate the local density from LGL grid to uniform
            // grid
            InterpLGLToUniform( 
                numLGLGridElem_, 
                numUniformGridElem_, 
                localRhoLGL.Data(), 
                localRho.Data() );

            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }

          } // own this element
        } // for (i)

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

    Real rhofac = numSpin_ * numOccupiedState_ 
      * (domain_.NumGridTotal() / domain_.Volume()) / sumRho;

    // Normalize the electron density in the global domain
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho = rho.LocalMap()[key];
            blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
          } // own this element
        } // for (i)
  } // Method 2

  // Method 3: Method 3 is the same as the Method 2, but to output the
  // eigenfunctions locally. 
  if(1) // FIXME
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    Real sumRhoLGLLocal = 0.0, sumRhoLGL = 0.0;
    // Clear the density 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

            SetValue( localRho, 0.0 );
            SetValue( localRhoLGL, 0.0 );

          }
        } // for (i)

    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisLGL_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();

            Int numBasisTotal = 0;
            MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

            // Compute the density by matrix vector multiplication
            // This is done by going from column partition to row 
            // parition, perform Gemv locally, and transform the output
            // from row partition to column partition.

            // Skip the element if there is no basis functions.
            if( numBasisTotal == 0 )
              continue;

            DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];

            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];
            SetValue( localRhoLGL, 0.0 );


            // Loop over all the eigenfunctions
            // 
            // NOTE: Gemm is not a feasible choice when a large number of
            // eigenfunctions are there.
            {
              DblNumVec  localPsiLGL( numGrid );
              SetValue( localPsiLGL, 0.0 );

              // For thread safety, declare as private variable
              DblNumVec  localRhoLGLTmp( numGrid );
              SetValue( localRhoLGLTmp, 0.0 );


              // Compute the density by converting the basis function
              // from column partition to row partition, and then
              // compute the Kohn-Sham wavefunction on each local
              // processor, which contributes to the electron density.
              //
              // The electron density is reduced among all processors in
              // the same row processor communicator to obtain the
              // electron density in each element.  
              if(1){

                Int height = numGrid;
                Int width = numBasisTotal;

                Int widthBlocksize = width / mpisizeRow;
                Int heightBlocksize = height / mpisizeRow;

                Int widthLocal = widthBlocksize;
                Int heightLocal = heightBlocksize;

                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }

                if(mpirankRow < (height % mpisizeRow)){
                  heightLocal = heightBlocksize + 1;
                }

                Int numGridTotal = height;  
                Int numGridLocal = heightLocal;  

                DblNumMat localBasisRow( heightLocal, width );

                AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

                DblNumVec  localRhoLGLRow( numGridLocal ); 
                SetValue( localRhoLGLRow, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp parallel 
                {
#endif
                  DblNumVec  localPsiLGLRow( numGridLocal );
                  SetValue( localPsiLGLRow, 0.0 );

                  // For thread safety, declare as private variable

                  DblNumVec localRhoLGLRowTmp( numGridLocal );
                  SetValue( localRhoLGLRowTmp, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
                  for( Int g = 0; g < numEig; g++ ){
                    // Compute local wavefunction on the LGL grid
                    blas::Gemv( 'N', numGridLocal, numBasisTotal, 1.0, 
                        localBasisRow.Data(), numGridLocal, 
                        localCoef.VecData(g), 1, 0.0,
                        localPsiLGLRow.Data(), 1 );

                    // Update the local density
                    Real  occ    = occrate[g];

                    for( Int p = 0; p < numGridLocal; p++ ){
                      localRhoLGLRowTmp(p) += localPsiLGLRow(p) * localPsiLGLRow(p) * occ * numSpin_;
                    }
                  }

#ifdef _USE_OPENMP_
#pragma omp critical
                  {
#endif
                    // This is a reduce operation for an array, and should be
                    // done in the OMP critical syntax
                    blas::Axpy( numGridLocal, 1.0, localRhoLGLRowTmp.Data(), 1, localRhoLGLRow.Data(), 1 );
#ifdef _USE_OPENMP_
                  }
#endif

#ifdef _USE_OPENMP_
                }
#endif

                IntNumVec heightBlocksizeIdx( mpisizeRow );
                SetValue( heightBlocksizeIdx, 0 );
                for( Int i = 0; i < mpisizeRow; i++ ){
                  if((height % mpisizeRow) == 0){
                    heightBlocksizeIdx(i) = heightBlocksize  * i;
                  }
                  else{
                    if(i < (height % mpisizeRow)){
                      heightBlocksizeIdx(i) = (heightBlocksize + 1) * i;
                    }
                    else{
                      heightBlocksizeIdx(i) = (heightBlocksize + 1) * (height % mpisizeRow) + heightBlocksize * (i - (height % mpisizeRow));
                    }
                  }
                }
                
                DblNumVec  localRhoLGLTemp1( numGridTotal );
                SetValue( localRhoLGLTemp1, 0.0 );
                for( Int p = 0; p < numGridLocal; p++ ){
                  localRhoLGLTemp1( p + heightBlocksizeIdx(mpirankRow) ) = localRhoLGLRow(p);
                  //localRhoLGLTemp1( p + heightBlocksize * mpirankRow ) = localRhoLGLRow(p);
                }

                SetValue( localRhoLGLTmp, 0.0 );
                MPI_Allreduce( localRhoLGLTemp1.Data(),
                    localRhoLGLTmp.Data(), numGridTotal, MPI_DOUBLE,
                    MPI_SUM, domain_.rowComm );


              } 

              blas::Axpy( numGrid, 1.0, localRhoLGLTmp.Data(), 1, localRhoLGL.Data(), 1 );
            }


            // Interpolate the local density from LGL grid to uniform
            // grid
            InterpLGLToUniform( 
                numLGLGridElem_, 
                numUniformGridElemFine_, 
                localRhoLGL.Data(), 
                localRho.Data() );

            sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
                localRhoLGL.Data(), 1, 
                LGLWeight3D_.Data(), 1 );

            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }
          } // own this element
        } // for (i)

    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Sum Rho on LGL grid (raw data) = ", sumRhoLGL );
    Print( statusOFS, "Sum Rho on uniform grid (interpolated) = ", sumRho );
    statusOFS << std::endl;
#endif

  } // for Method 3


  // Method 4:
  if(0) // FIXME ME
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    Real sumRhoLGLLocal = 0.0, sumRhoLGL = 0.0;
    // Clear the density 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

            SetValue( localRho, 0.0 );
            SetValue( localRhoLGL, 0.0 );

          }
        } // for (i)

    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisLGL_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();

            Int numBasisTotal = 0;
            MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

            // Compute the density by matrix vector multiplication
            // This is done by going from column partition to row 
            // parition, perform Gemv locally, and transform the output
            // from row partition to column partition.

            // Skip the element if there is no basis functions.
            if( numBasisTotal == 0 )
              continue;

            DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];

            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];
            SetValue( localRhoLGL, 0.0 );

            // Loop over all the eigenfunctions
            // 
            // NOTE: Gemm is not a feasible choice when a large number of
            // eigenfunctions are there.
            {
              DblNumVec  localPsiLGL( numGrid );
              SetValue( localPsiLGL, 0.0 );

              // For thread safety, declare as private variable
              DblNumVec  localRhoLGLTmp( numGrid );
              SetValue( localRhoLGLTmp, 0.0 );


              // Compute the density by converting the basis function
              // from column partition to row partition, and then
              // compute the Kohn-Sham wavefunction on each local
              // processor, which contributes to the electron density.
              //
              // The electron density is reduced among all processors in
              // the same row processor communicator to obtain the
              // electron density in each element.  
              if(1){

                Int height = numGrid;
                Int width = numBasisTotal;

                Int widthBlocksize = width / mpisizeRow;
                Int heightBlocksize = height / mpisizeRow;

                Int widthLocal = widthBlocksize;
                Int heightLocal = heightBlocksize;

                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }

                if(mpirankRow < (height % mpisizeRow)){
                  heightLocal = heightBlocksize + 1;
                }

                Int numGridTotal = height;  
                Int numGridLocal = heightLocal;  

                DblNumMat localBasisRow( heightLocal, width );

                AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

                DblNumVec  localRhoLGLRow( numGridLocal ); 
                SetValue( localRhoLGLRow, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp parallel 
                {
#endif
                  DblNumVec  localPsiLGLRow( numGridLocal );
                  SetValue( localPsiLGLRow, 0.0 );

                  // For thread safety, declare as private variable

                  DblNumVec localRhoLGLRowTmp( numGridLocal );
                  SetValue( localRhoLGLRowTmp, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
                  for( Int g = 0; g < numEig; g++ ){
                    // Compute local wavefunction on the LGL grid
                    blas::Gemv( 'N', numGridLocal, numBasisTotal, 1.0, 
                        localBasisRow.Data(), numGridLocal, 
                        localCoef.VecData(g), 1, 0.0,
                        localPsiLGLRow.Data(), 1 );

                    // Update the local density
                    Real  occ    = occrate[g];

                    for( Int p = 0; p < numGridLocal; p++ ){
                      localRhoLGLRowTmp(p) += localPsiLGLRow(p) * localPsiLGLRow(p) * occ * numSpin_;
                    }
                  }

#ifdef _USE_OPENMP_
#pragma omp critical
                  {
#endif
                    // This is a reduce operation for an array, and should be
                    // done in the OMP critical syntax
                    blas::Axpy( numGridLocal, 1.0, localRhoLGLRowTmp.Data(), 1, localRhoLGLRow.Data(), 1 );
#ifdef _USE_OPENMP_
                  }
#endif

#ifdef _USE_OPENMP_
                }
#endif

                IntNumVec heightBlocksizeIdx( mpisizeRow );
                SetValue( heightBlocksizeIdx, 0 );
                for( Int i = 0; i < mpisizeRow; i++ ){
                  if((height % mpisizeRow) == 0){
                    heightBlocksizeIdx(i) = heightBlocksize  * i;
                  }
                  else{
                    if(i < (height % mpisizeRow)){
                      heightBlocksizeIdx(i) = (heightBlocksize + 1) * i;
                    }
                    else{
                      heightBlocksizeIdx(i) = (heightBlocksize + 1) * (height % mpisizeRow) + heightBlocksize * (i - (height % mpisizeRow));
                    }
                  }
                }
                
                DblNumVec  localRhoLGLTemp1( numGridTotal );
                SetValue( localRhoLGLTemp1, 0.0 );
                for( Int p = 0; p < numGridLocal; p++ ){
                  localRhoLGLTemp1( p + heightBlocksizeIdx(mpirankRow) ) = localRhoLGLRow(p);
                  //localRhoLGLTemp1( p + heightBlocksize * mpirankRow ) = localRhoLGLRow(p);
                }

                SetValue( localRhoLGLTmp, 0.0 );
                MPI_Allreduce( localRhoLGLTemp1.Data(),
                    localRhoLGLTmp.Data(), numGridTotal, MPI_DOUBLE,
                    MPI_SUM, domain_.rowComm );


              } 

              blas::Axpy( numGrid, 1.0, localRhoLGLTmp.Data(), 1, localRhoLGL.Data(), 1 );
            }

            if(1) 
            {

              for( Int kk = 0; kk < numExtElem_[2]; kk++ ){
                for( Int jj = 0; jj < numExtElem_[1]; jj++ ){
                  for( Int ii = 0; ii < numExtElem_[0]; ii++ ){
                    std::vector<DblNumMat>& LGLToUniformGaussMatFine = LGLToUniformGaussMatFine_ (ii,jj,kk);
                    Index3 keyTemp(ii,jj,kk);

                    for( Int d = 0; d < DIM; d++ ){
                      if (numExtElem_[d] == 1){
                        keyTemp[d] = key[d]; 
                      }

                      if (numExtElem_[d] == 3){
                        if ( (key[d] == 0) && (keyTemp[d] == 0) ){
                          keyTemp[d] = numElem_[d] - 1; 
                        }
                        else if ( (key[d] == (numElem_[d]-1) ) && (keyTemp[d] == 2) ){
                          keyTemp[d] = 0; 
                        }
                        else {
                          keyTemp[d] = key[d] + keyTemp[d] - 1;
                        } 
                      }

                    } // for (d)

                    DblNumVec& localRhoTemp = rho.LocalMap()[keyTemp];
                    localRhoTemp.Resize( numUniformGridElemFine_.prod() );
                    SetValue( localRhoTemp, 0.0 );

                    GaussConvInterpLGLToUniform( 
                        numLGLGridElem_, 
                        numUniformGridElemFine_, 
                        localRhoLGL.Data(), 
                        localRhoTemp.Data(),
                        LGLToUniformGaussMatFine );

                  }
                }
              }

            } // for if(1) 

          } // own this element
        } // for (i)

    if(1) 
    {
      std::vector<Index3>  keyIdx;
      for( std::map<Index3, DblNumVec>::iterator 
          mi  = rho.LocalMap().begin();
          mi != rho.LocalMap().end(); mi++ ){
        Index3 key = (*mi).first;
        if( rho.Prtn().Owner(key) != (mpirank / dmRow_) ){
          keyIdx.push_back( key );
        }
      }

      // Communication
      rho.PutBegin( keyIdx, NO_MASK );
      rho.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean up
      std::vector<Index3>  eraseKey;
      for( std::map<Index3, DblNumVec>::iterator 
          mi  = rho.LocalMap().begin();
          mi != rho.LocalMap().end(); mi++ ){
        Index3 key = (*mi).first;
        if( rho.Prtn().Owner(key) != (mpirank / dmRow_) ){
          eraseKey.push_back( key );
        }
      }

      for( std::vector<Index3>::iterator vi = eraseKey.begin();
          vi != eraseKey.end(); vi++ ){
        rho.LocalMap().erase( *vi );
      }

    }  // for if(1)

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];
            DblNumVec& localRho    = rho.LocalMap()[key];

            sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
                localRhoLGL.Data(), 1, 
                LGLWeight3D_.Data(), 1 );

            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }
          } // own this element
        } // for (i)

    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Sum Rho on LGL grid (raw data) = ", sumRhoLGL );
    Print( statusOFS, "Sum Rho on uniform grid (interpolated) = ", sumRho );
    statusOFS << std::endl;
#endif

  } // for Method 4 


  // Method 5: 
  if(0) // FIXME ME
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisUniformFine_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();


            Int numBasisTotal = 0;
            MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

            // Compute the density by matrix vector multiplication
            // This is done by going from column partition to row 
            // parition, perform Gemv locally, and transform the output
            // from row partition to column partition.

            // Skip the element if there is no basis functions.
            if( numBasisTotal == 0 )
              continue;

            DblNumMat& localCoef = eigvecCoef_.LocalMap()[key];
            DblNumVec& localRho  = rho.LocalMap()[key];

            // Loop over all the eigenfunctions
            // 
            // NOTE: Gemm is not a feasible choice when a large number of
            // eigenfunctions are there.
            if(1){
              // Compute the density by converting the basis function
              // from column partition to row partition, and then
              // compute the Kohn-Sham wavefunction on each local
              // processor, which contributes to the electron density.
              //
              // The electron density is reduced among all processors in
              // the same row processor communicator to obtain the
              // electron density in each element.  

              Int height = numGrid;
              Int width = numBasisTotal;

              Int widthBlocksize = width / mpisizeRow;
              Int heightBlocksize = height / mpisizeRow;

              Int widthLocal = widthBlocksize;
              Int heightLocal = heightBlocksize;

              if(mpirankRow < (width % mpisizeRow)){
                widthLocal = widthBlocksize + 1;
              }

              if(mpirankRow < (height % mpisizeRow)){
                heightLocal = heightBlocksize + 1;
              }

              Int numGridTotal = height;  
              Int numGridLocal = heightLocal;  

              DblNumMat localBasisRow( heightLocal, width );

              AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

              DblNumVec  localRhoRow( numGridLocal ); 
              SetValue( localRhoRow, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp parallel 
              {
#endif
                DblNumVec  localPsiRow( numGridLocal );
                SetValue( localPsiRow, 0.0 );

                // For thread safety, declare as private variable

                DblNumVec localRhoRowTmp( numGridLocal );
                SetValue( localRhoRowTmp, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif

                for( Int g = 0; g < numEig; g++ ){
                  blas::Gemv( 'N', numGridLocal, numBasisTotal, 1.0, 
                      localBasisRow.Data(), numGridLocal, 
                      localCoef.VecData(g), 1, 0.0,
                      localPsiRow.Data(), 1 );

                  // Update the local density
                  Real  occ    = occrate[g];

                  for( Int p = 0; p < numGridLocal; p++ ){
                    localRhoRowTmp(p) += localPsiRow(p) * localPsiRow(p) * occ * numSpin_;
                  }
                } // for g


#ifdef _USE_OPENMP_
#pragma omp critical
                {
#endif
                  // This is a reduce operation for an array, and should be
                  // done in the OMP critical syntax
                  blas::Axpy( numGridLocal, 1.0, localRhoRowTmp.Data(), 1, localRhoRow.Data(), 1 );
#ifdef _USE_OPENMP_
                }
#endif

#ifdef _USE_OPENMP_
              }
#endif

              IntNumVec heightBlocksizeIdx( mpisizeRow );
              SetValue( heightBlocksizeIdx, 0 );
              for( Int i = 0; i < mpisizeRow; i++ ){
                if((height % mpisizeRow) == 0){
                  heightBlocksizeIdx(i) = heightBlocksize  * i;
                }
                else{
                  if(i < (height % mpisizeRow)){
                    heightBlocksizeIdx(i) = (heightBlocksize + 1) * i;
                  }
                  else{
                    heightBlocksizeIdx(i) = (heightBlocksize + 1) * (height % mpisizeRow) + heightBlocksize * (i - (height % mpisizeRow));
                  }
                }
              }

              DblNumVec  localRhoTemp1( numGridTotal );
              SetValue( localRhoTemp1, 0.0 );
              for( Int p = 0; p < numGridLocal; p++ ){
                localRhoTemp1( p + heightBlocksizeIdx(mpirankRow) ) = localRhoRow(p);
                //localRhoTemp1( p + heightBlocksize * mpirankRow ) = localRhoRow(p);
              }


              DblNumVec  localRhoTemp2( numGridTotal );
              SetValue( localRhoTemp2, 0.0 );
              MPI_Allreduce( localRhoTemp1.Data(),
                  localRhoTemp2.Data(), numGridTotal, MPI_DOUBLE,
                  MPI_SUM, domain_.rowComm );

              blas::Axpy( numGrid, 1.0, localRhoTemp2.Data(), 1, localRho.Data(), 1 );

            } //if(1) 


            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }

          } // own this element
        } // for (i)

    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Sum Rho on uniform grid method 5 = ", sumRho );
    statusOFS << std::endl;
#endif

  } // for Method 5

  // Method 6: Normalize each eigenfunctions.  This may take many
  // interpolation steps, and the communication cost may be large.
  // This routine generates both density on the LGL grid and on the
  // uniform grid
  // FIXME: Only works now WITHOUT intra-element parallelization
  if(0)
  { 

    DistDblNumVec  psiLGL;
    psiLGL.Prtn() = elemPrtn_;

    // Loop over all the eigenfunctions
    for( Int g = 0; g < numEig; g++ ){
      // Normalization constants
      Real normPsiLocal  = 0.0;
      Real normPsi       = 0.0;
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumMat& localBasis = basisLGL_.LocalMap()[key];
              Int numGrid  = localBasis.m();
              Int numBasis = localBasis.n();

              DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
              if( localCoef.n() != numEig ){
                ErrorHandling( 
                    "Numbers of eigenfunction coefficients do not match.");
              }
              if( localCoef.m() != numBasis ){
                ErrorHandling(
                    "Number of LGL grids do not match.");
              }
              DblNumVec  localPsiLGL( numGrid );
              DblNumVec  localPsiUniformFine( numUniformGridElemFine_.prod() );
              SetValue( localPsiLGL, 0.0 );

              // Compute local wavefunction on the LGL grid
              blas::Gemv( 'N', numGrid, numBasis, 1.0, 
                  localBasis.Data(), numGrid, 
                  localCoef.VecData(g), 1, 0.0,
                  localPsiLGL.Data(), 1 );

              // Interpolate local wavefunction from LGL grid to uniform fine grid
              InterpLGLToUniform( 
                  numLGLGridElem_, 
                  numUniformGridElemFine_, 
                  localPsiLGL.Data(), 
                  localPsiUniformFine.Data() );

              // Compute the local norm
              normPsiLocal += Energy( localPsiUniformFine );

              psiUniform.LocalMap()[key] = localPsiUniformFine;
              psiLGL.LocalMap()[key] = localPsiLGL;

            } // own this element
          } // for (i)

      // All processors get the normalization factor
      mpi::Allreduce( &normPsiLocal, &normPsi, 1, MPI_SUM, domain_.comm );

      // pre-constant in front of psi^2 for density
      Real rhofac = (numSpin_ * domain_.NumGridTotalFine() / domain_.Volume() ) 
        * occrate[g] / normPsi;


      // Add the normalized wavefunction to density
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localRho = rho.LocalMap()[key];
              DblNumVec& localPsiUniformFine = psiUniform.LocalMap()[key];
              for( Int p = 0; p < localRho.Size(); p++ ){
                localRho[p] += localPsiUniformFine[p] * localPsiUniformFine[p] * rhofac;
              }    

              DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];
              DblNumVec& localPsiLGL = psiLGL.LocalMap()[key];
              for( Int p = 0; p < localPsiLGL.Size(); p++ ){
                localRhoLGL[p] += localPsiLGL[p] * localPsiLGL[p] * occrate[g] * numSpin_;
              }    
            } // own this element
          } // for (i)
    } // for (g)
    // Check the sum of the electron density
#if ( _DEBUGlevel_ >= 0 )
    {
      Real sumRhoLocal = 0.0;
      Real sumRho      = 0.0;
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localRho = rho.LocalMap()[key];
              for( Int p = 0; p < localRho.Size(); p++ ){
                sumRhoLocal += localRho[p];
              }    
            } // own this element
          } // for (i)
      mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

      sumRho *= domain_.Volume() / domain_.NumGridTotalFine();

      Print( statusOFS, "Sum rho on uniform fine = ", sumRho );
    }
#endif

#if ( _DEBUGlevel_ >= 0 )
    {
      Real sumRhoLGLLocal = 0.0;
      Real sumRhoLGL      = 0.0;
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

              sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
                  localRhoLGL.Data(), 1, 
                  LGLWeight3D_.Data(), 1 );

            } // own this element
          } // for (i)
      mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.colComm );

      Print( statusOFS, "Sum rho on LGL = ", sumRhoLGL );
    }
#endif

  } // Method 6


  return ;
}         // -----  end of method HamiltonianDG::CalculateDensity  ----- 


void
HamiltonianDG::CalculateDensityDM    (
    DistDblNumVec& rho,
    DistDblNumVec& rhoLGL,
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );


  if(1)
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    Real sumRhoLGLLocal = 0.0, sumRhoLGL = 0.0;

    // Clear the density 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

            SetValue( localRho, 0.0 );
            SetValue( localRhoLGL, 0.0 );

          }
        } // for (i)

    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisLGL_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();

            // Skip the element if there is no basis functions.
            if( numBasis == 0 )
              continue;

            DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key, key)];

            if( numBasis != localDM.m() ||
                numBasis != localDM.n() ){
              std::ostringstream msg;
              msg << std::endl
                << "Error happens in the element (" << key << ")" << std::endl
                << "The number of basis functions is " << numBasis << std::endl
                << "The size of the local density matrix is " 
                << localDM.m() << " x " << localDM.n() << std::endl;
              ErrorHandling( msg.str().c_str() );
            }

            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

#ifdef _USE_OPENMP_
#pragma omp parallel 
            {
#endif
              // For thread safety, declare as private variable
              DblNumVec  localRhoLGLTmp( numGrid );
              SetValue( localRhoLGLTmp, 0.0 );
              Real factor;

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
              // Explicit take advantage of the symmetry
              for( Int a = 0; a < numBasis; a++ )
                for( Int b = a; b < numBasis; b++ ){
                  factor = localDM(a,b);
                  if( b > a ) factor *= 2.0;
                  for( Int p = 0; p < numGrid; p++ ){
                    localRhoLGLTmp(p) += localBasis(p,a) * localBasis(p,b) * factor; 
                  }
                }
#ifdef _USE_OPENMP_
#pragma omp critical
              {
#endif
                // This is a reduce operation for an array, and should be
                // done in the OMP critical syntax
                blas::Axpy( numGrid, 1.0, localRhoLGLTmp.Data(), 1, localRhoLGL.Data(), 1 );
#ifdef _USE_OPENMP_
              }
#endif

#ifdef _USE_OPENMP_
            }
#endif

            statusOFS << "Before interpolation" << std::endl;

            // Interpolate the local density from LGL grid to uniform
            // grid
            InterpLGLToUniform( 
                numLGLGridElem_, 
                numUniformGridElemFine_,
                localRhoLGL.Data(), 
                localRho.Data() );
            statusOFS << "After interpolation" << std::endl;

            sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
                localRhoLGL.Data(), 1, 
                LGLWeight3D_.Data(), 1 );

            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }
          } // own this element
        } // for (i)

    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Sum Rho on LGL grid (raw data) = ", sumRhoLGL );
    Print( statusOFS, "Sum Rho on uniform grid (interpolated) = ", sumRho );
    statusOFS << std::endl;
#endif


    Real rhofac = numSpin_ * numOccupiedState_ / sumRho;


    // Normalize the electron density in the global domain
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho = rho.LocalMap()[key];
            blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
          } // own this element
        } // for (i)
  }

  return ;
}         // -----  end of method HamiltonianDG::CalculateDensityDM  ----- 


void
HamiltonianDG::CalculateDensityDM2    (
    DistDblNumVec& rho,
    DistDblNumVec& rhoLGL,
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  if(1)
  {
    Real sumRhoLocal = 0.0, sumRho = 0.0;
    Real sumRhoLGLLocal = 0.0, sumRhoLGL = 0.0;

    // Clear the density 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

            SetValue( localRho, 0.0 );
            SetValue( localRhoLGL, 0.0 );

          }
        } // for (i)

    // Compute the local density in each element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumMat& localBasis = basisLGL_.LocalMap()[key];
            Int numGrid  = localBasis.m();
            Int numBasis = localBasis.n();

            Int numBasisTotal = 0;
            MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

            // Skip the element if there is no basis functions.
            if( numBasisTotal == 0 )
              continue;

            // Convert the basis functions from column based
            // partition to row based partition
            Int height = localBasis.m();
            Int width = numBasisTotal;

            Int widthBlocksize = width / mpisizeRow;
            Int heightBlocksize = height / mpisizeRow;

            Int widthLocal = widthBlocksize;
            Int heightLocal = heightBlocksize;

            if(mpirankRow < (width % mpisizeRow)){
              widthLocal = widthBlocksize + 1;
            }

            if(mpirankRow < (height % mpisizeRow)){
              heightLocal = heightBlocksize + 1;
            }

            Int numLGLGridTotal = height;  
            Int numLGLGridLocal = heightLocal;  

            DblNumMat localBasisRow( heightLocal, width );

            AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

            DblNumMat& localDM = distDMMat.LocalMap()[ElemMatKey(key, key)];

            DblNumVec& localRho    = rho.LocalMap()[key];
            DblNumVec& localRhoLGL = rhoLGL.LocalMap()[key];

            {
              // For thread safety, declare as private variable
              DblNumVec  localRhoLGLTmp( numGrid );
              SetValue( localRhoLGLTmp, 0.0 );
              Real factor;
              
              IntNumVec heightBlocksizeIdx( mpisizeRow );
              SetValue( heightBlocksizeIdx, 0 );
              for( Int i = 0; i < mpisizeRow; i++ ){
                if((height % mpisizeRow) == 0){
                  heightBlocksizeIdx(i) = heightBlocksize  * i;
                }
                else{
                  if(i < (height % mpisizeRow)){
                    heightBlocksizeIdx(i) = (heightBlocksize + 1) * i;
                  }
                  else{
                    heightBlocksizeIdx(i) = (heightBlocksize + 1) * (height % mpisizeRow) + heightBlocksize * (i - (height % mpisizeRow));
                  }
                }
              }

              // Explicit take advantage of the symmetry
              for( Int a = 0; a < numBasisTotal; a++ )
                for( Int b = a; b < numBasisTotal; b++ ){
                  factor = localDM(a,b);
                  if( b > a ) factor *= 2.0;
                  Int idxSta = heightBlocksizeIdx(mpirankRow);
                  //Int idxSta = mpirankRow * heightBlocksize;
                  for( Int p = 0; p < heightLocal; p++ ){
                    localRhoLGLTmp(idxSta + p) += 
                      localBasisRow(p,a) * localBasisRow(p,b) * factor; 
                  }
                }
              MPI_Allreduce( localRhoLGLTmp.Data(), localRhoLGL.Data(),
                  numLGLGridTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );
            }

            statusOFS << "Before interpolation" << std::endl;

            // Interpolate the local density from LGL grid to uniform
            // grid
            InterpLGLToUniform( 
                numLGLGridElem_, 
                numUniformGridElemFine_, 
                localRhoLGL.Data(), 
                localRho.Data() );
            statusOFS << "After interpolation" << std::endl;

            sumRhoLGLLocal += blas::Dot( localRhoLGL.Size(),
                localRhoLGL.Data(), 1, 
                LGLWeight3D_.Data(), 1 );

            Real* ptrRho = localRho.Data();
            for( Int p = 0; p < localRho.Size(); p++ ){
              sumRhoLocal += (*ptrRho);
              ptrRho++;
            }
          } // own this element
        } // for (i)

    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 

    // All processors get the normalization factor
    mpi::Allreduce( &sumRhoLGLLocal, &sumRhoLGL, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl;
    Print( statusOFS, "Sum Rho on LGL grid (raw data) = ", sumRhoLGL );
    Print( statusOFS, "Sum Rho on uniform grid (interpolated) = ", sumRho );
    statusOFS << std::endl;
#endif

    Real rhofac = numSpin_ * numOccupiedState_ / sumRho;

    // Normalize the electron density in the global domain
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho = rho.LocalMap()[key];
            blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
          } // own this element
        } // for (i)
  }

  return ;
}         // -----  end of method HamiltonianDG::CalculateDensityDM2  ----- 


void HamiltonianDG::CalculateGradDensity( DistFourier&  fft ) {
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;

  DblNumVec  tempVecLocal;

  std::vector<DblNumVec>      gradDensityLocal(DIM);

  // Convert tempVec to tempVecLocal in distributed row vector format
  DistNumVecToDistRowVec(
      density_,
      tempVecLocal,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );

  if( fft.isInGrid ){

    for( Int i = 0; i < ntotLocal; i++ ){
      fft.inputComplexVecLocal(i) = Complex( 
          tempVecLocal(i), 0.0 );
    }

    fftw_execute( fft.forwardPlan );

    CpxNumVec  cpxVecLocal( tempVecLocal.Size() );
    blas::Copy( ntotLocal, fft.outputComplexVecLocal.Data(), 1,
        cpxVecLocal.Data(), 1 );

    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikLocal  = fft.ikLocal[d];
      for( Int i = 0; i < ntotLocal; i++ ){
        if( fft.gkkLocal(i) == 0 ){
          fft.outputComplexVecLocal(i) = Z_ZERO;
        }
        else{
          fft.outputComplexVecLocal(i) = cpxVecLocal(i) * ikLocal(i);
        }
      }

      fftw_execute( fft.backwardPlan );

      gradDensityLocal[d].Resize( tempVecLocal.Size() );

      for( Int i = 0; i < ntotLocal; i++ ){
        gradDensityLocal[d](i) = fft.inputComplexVecLocal(i).real() / ntot;
      }

    } // for (d)

  } // if (fft.isInGrid)

  for( Int d = 0; d < DIM; d++ ){
    DistRowVecToDistNumVec( 
        gradDensityLocal[d],
        gradDensity_[d],
        domain_.numGridFine,
        numElem_,
        fft.localNzStart,
        fft.localNz,
        fft.isInGrid,
        domain_.colComm );
  }

  return; 
}  // -----  end of method HamiltonianDG::CalculateGradDensity ----- 


void
HamiltonianDG::CalculateXC    ( 
    Real &Exc, 
    DistDblNumVec&   epsxc,
    DistDblNumVec&   vxc,
    DistFourier&    fft )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Real ExcLocal = 0.0;

  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;
  // Cutoff of the XC potential. Important for SCF convergence for GGA and above.
  Real epsRho = 1e-8, epsGRho = 1e-8;

  if( XCId_ == 20 ) // XC_FAMILY_LDA
  {
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho   = density_.LocalMap()[key];
            DblNumVec& localEpsxc = epsxc.LocalMap()[key];
            DblNumVec& localVxc   = vxc.LocalMap()[key];

            xc_lda_exc_vxc( &XCFuncType_, localRho.Size(), 
                localRho.Data(),
                localEpsxc.Data(), 
                localVxc.Data() );

            // Modify "bad points"
            if(1){
              for( Int i = 0; i < localRho.Size(); i++ ){
                if( localRho(i) < epsRho ){
                  localEpsxc(i) = 0.0;
                  localVxc(i) = 0.0;
                }
              }
            }

            ExcLocal += blas::Dot( localRho.Size(), 
                localRho.Data(), 1, localEpsxc.Data(), 1 );

          } // own this element
        } // for (i)
  } // XC_FAMILY_LDA
  else if( ( XId_ == 101 ) && ( CId_ == 130 ) ) //XC_FAMILY_GGA
  {
    DistDblNumVec vxc22;
    vxc22.SetComm( domain_.colComm );
    vxc22.Prtn() = elemPrtn_;

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            DblNumVec& localRho   = density_.LocalMap()[key];
            DblNumVec& localEpsxc = epsxc.LocalMap()[key];
            DblNumVec& localVxc   = vxc.LocalMap()[key];

            DblNumVec& vxc2 = vxc22.LocalMap()[key];

            DblNumVec     vxc1;             
            vxc1.Resize( localRho.Size() );
            vxc2.Resize( localRho.Size() );

            DblNumVec     vxc1temp;             
            DblNumVec     vxc2temp;             
            vxc1temp.Resize( localRho.Size() );
            vxc2temp.Resize( localRho.Size() );

            DblNumVec     epsx; 
            DblNumVec     epsc; 
            epsx.Resize( localRho.Size() );
            epsc.Resize( localRho.Size() );

            DblNumVec gradDensity;
            gradDensity.Resize( localRho.Size() );
            SetValue( gradDensity, 0.0 );
            DblNumVec& gradDensity0 = gradDensity_[0].LocalMap()[key];
            DblNumVec& gradDensity1 = gradDensity_[1].LocalMap()[key];
            DblNumVec& gradDensity2 = gradDensity_[2].LocalMap()[key];

            for(Int i = 0; i < localRho.Size(); i++){
              gradDensity(i) = gradDensity0(i) * gradDensity0(i)
                + gradDensity1(i) * gradDensity1(i)
                + gradDensity2(i) * gradDensity2(i);
            }

            Real timeXCSta, timeXCEnd;
            GetTime(timeXCSta);
            
            SetValue( epsx, 0.0 );
            SetValue( vxc1, 0.0 );
            SetValue( vxc2, 0.0 );
            xc_gga_exc_vxc( &XFuncType_, localRho.Size(), localRho.Data(), 
                gradDensity.Data(), epsx.Data(), vxc1.Data(), vxc2.Data() );

            SetValue( epsc, 0.0 );
            SetValue( vxc1temp, 0.0 );
            SetValue( vxc2temp, 0.0 );
            xc_gga_exc_vxc( &CFuncType_, localRho.Size(), localRho.Data(), 
                gradDensity.Data(), epsc.Data(), vxc1temp.Data(), vxc2temp.Data() );
            GetTime(timeXCEnd);
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << "Time for calling the XC kernel is " <<
              timeXCEnd - timeXCSta << " [s]" << std::endl << std::endl;
#endif

            for( Int i = 0; i < localRho.Size(); i++ ){
              localEpsxc(i) = epsx(i) + epsc(i) ;
              vxc1( i ) += vxc1temp( i );
              vxc2( i ) += vxc2temp( i );
              localVxc( i ) = vxc1( i );
            }

            // Modify "bad points"
            if(1){
              for( Int i = 0; i < localRho.Size(); i++ ){
                if( localRho(i) < epsRho || gradDensity(i) < epsGRho ){
                  localEpsxc(i) = 0.0;
                  vxc1(i) = 0.0;
                  vxc2(i) = 0.0;
                  localVxc(i) = 0.0;
                }
              }
            }


            ExcLocal += blas::Dot( localRho.Size(), 
                localRho.Data(), 1, localEpsxc.Data(), 1 );

          } // own this element
        } // for (i)

    for( Int d = 0; d < DIM; d++ ){

      DistDblNumVec gradDensityVxc22;
      gradDensityVxc22.SetComm( domain_.colComm );
      gradDensityVxc22.Prtn() = elemPrtn_;

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key = Index3( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& vxc2 = vxc22.LocalMap()[key];
              DblNumVec& gradDensityd     = gradDensity_[d].LocalMap()[key];
              DblNumVec& gradDensityVxc2  = gradDensityVxc22.LocalMap()[key];

              gradDensityVxc2.Resize( gradDensityd.Size() );
              for(Int i = 0; i < gradDensityd.Size(); i++){
                gradDensityVxc2(i) = gradDensityd( i ) * 2.0 * vxc2( i ); 
              }

            } // own this element
          } // for (i)

      DistDblNumVec gradGradDensityVxc22;
      gradGradDensityVxc22.SetComm( domain_.colComm );
      gradGradDensityVxc22.Prtn() = elemPrtn_;

      DblNumVec  tempVecLocal1;
      DblNumVec  tempVecLocal2;

      DistNumVecToDistRowVec(
          gradDensityVxc22,
          tempVecLocal1,
          domain_.numGridFine,
          numElem_,
          fft.localNzStart,
          fft.localNz,
          fft.isInGrid,
          domain_.colComm );

      tempVecLocal2.Resize( tempVecLocal1.Size() );
      SetValue( tempVecLocal2, 0.0 );

      if( fft.isInGrid ){

        for( Int i = 0; i < ntotLocal; i++ ){
          fft.inputComplexVecLocal(i) = Complex( 
              tempVecLocal1(i), 0.0 );
        }

        fftw_execute( fft.forwardPlan );

        CpxNumVec& ikLocal  = fft.ikLocal[d];
        for( Int i = 0; i < ntotLocal; i++ ){
          if( fft.gkkLocal(i) == 0 ){
            fft.outputComplexVecLocal(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecLocal(i) *= ikLocal(i);
          }
        }

        fftw_execute( fft.backwardPlan );

        for( Int i = 0; i < ntotLocal; i++ ){
          tempVecLocal2(i) = fft.inputComplexVecLocal(i).real() / ntot;
        }

      } // if (fft.isInGrid)

      DistRowVecToDistNumVec( 
          tempVecLocal2,
          gradGradDensityVxc22,
          domain_.numGridFine,
          numElem_,
          fft.localNzStart,
          fft.localNz,
          fft.isInGrid,
          domain_.colComm );



      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key = Index3( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec& localVxc   = vxc.LocalMap()[key];
              DblNumVec& gradGradDensityVxc2 = gradGradDensityVxc22.LocalMap()[key];
              for( Int i = 0; i < localVxc.Size(); i++ ){
                localVxc( i ) -= gradGradDensityVxc2(i);
              }
            }
          }

    } // for (d)

  } //XC_FAMILY_GGA
  else
    ErrorHandling( "Unsupported XC family!" );

  ExcLocal *= domain_.Volume() / domain_.NumGridTotalFine();

  mpi::Allreduce( &ExcLocal, &Exc, 1, MPI_SUM, domain_.colComm );


  return ;
}         // -----  end of method HamiltonianDG::CalculateXC  ----- 


void HamiltonianDG::CalculateHartree( 
    DistDblNumVec&  vhart,
    DistFourier&    fft ) {
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;

  vhart.SetComm(domain_.colComm);

  DistDblNumVec   tempVec;
  tempVec.SetComm(domain_.colComm);
  tempVec.Prtn() = elemPrtn_;

  // tempVec = density_ - pseudoCharge_
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key = Index3( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          tempVec.LocalMap()[key] = density_.LocalMap()[key];
          blas::Axpy( numUniformGridElemFine_.prod(), -1.0, 
              pseudoCharge_.LocalMap()[key].Data(), 1,
              tempVec.LocalMap()[key].Data(), 1 );
        }
      }

  // Convert tempVec to tempVecLocal in distributed row vector format
  DblNumVec  tempVecLocal;

  DistNumVecToDistRowVec(
      tempVec,
      tempVecLocal,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );

  // The contribution of the pseudoCharge is subtracted. So the Poisson
  // equation is well defined for neutral system.
  // Only part of the processors participate in the FFTW calculation

  if( fft.isInGrid ){

    for( Int i = 0; i < ntotLocal; i++ ){
      fft.inputComplexVecLocal(i) = Complex( 
          tempVecLocal(i), 0.0 );
    }
    fftw_execute( fft.forwardPlan );

    for( Int i = 0; i < ntotLocal; i++ ){
      if( fft.gkkLocal(i) == 0 ){
        fft.outputComplexVecLocal(i) = Z_ZERO;
      }
      else{
        // NOTE: gkk already contains the factor 1/2.
        fft.outputComplexVecLocal(i) *= 2.0 * PI / fft.gkkLocal(i);
      }
    }
    fftw_execute( fft.backwardPlan );

    // tempVecLocal saves the Hartree potential

    for( Int i = 0; i < ntotLocal; i++ ){
      tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
    }
  } // if (fft.isInGrid)

  // Convert tempVecLocal to vhart in the DistNumVec format

  DistRowVecToDistNumVec(
      tempVecLocal,
      vhart,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );


  return; 
}  // -----  end of method HamiltonianDG::CalculateHartree ----- 




void
HamiltonianDG::CalculateVtot    ( DistDblNumVec& vtot  )
{
  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  vtot.SetComm(domain_.colComm);

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key = Index3( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          DblNumVec&   localVtot  = vtot.LocalMap()[key];
          DblNumVec&   localVext  = vext_.LocalMap()[key];
          DblNumVec&   localVhart = vhart_.LocalMap()[key];
          DblNumVec&   localVxc   = vxc_.LocalMap()[key];

          localVtot.Resize( localVxc.Size() );
          for( Int p = 0; p < localVtot.Size(); p++){
            localVtot[p] = localVext[p] + localVhart[p] +
              localVxc[p];
          }
        }
      } // for (i)


  return ;
}         // -----  end of method HamiltonianDG::CalculateVtot  ----- 


void
HamiltonianDG::CalculateForce    ( DistFourier& fft )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  MPI_Barrier(domain_.rowComm);
  Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
  Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

  MPI_Barrier(domain_.colComm);
  Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
  Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

  // *********************************************************************
  // Initialize the force computation
  // *********************************************************************
  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;
  Int numAtom   = atomList_.size();


  DblNumMat   forceLocal( numAtom, DIM );
  DblNumMat   force( numAtom, DIM );
  SetValue( forceLocal, 0.0 );
  SetValue( force, 0.0 );


  // *********************************************************************
  // Compute the derivative of the Hartree potential for computing the 
  // local pseudopotential contribution to the Hellmann-Feynman force
  // *********************************************************************
  DistDblNumVec&              vhart = vhart_;
  std::vector<DistDblNumVec>  vhartDrv(DIM);
  std::vector<DblNumVec>      vhartDrvLocal(DIM);
  DistDblNumVec   tempVec;


  vhart.SetComm( domain_.colComm );
  tempVec.SetComm( domain_.colComm );

  tempVec.Prtn() = elemPrtn_;
  for( Int d = 0; d < DIM; d++ ){
    vhartDrv[d].Prtn() = elemPrtn_;
    vhartDrv[d].SetComm( domain_.colComm );
  }

  // tempVec = density_ - pseudoCharge_
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key = Index3( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          tempVec.LocalMap()[key] = density_.LocalMap()[key];
          blas::Axpy( numUniformGridElemFine_.prod(), -1.0, 
              pseudoCharge_.LocalMap()[key].Data(), 1,
              tempVec.LocalMap()[key].Data(), 1 );
        }
      }

  // Convert tempVec to tempVecLocal in distributed row vector format
  DblNumVec  tempVecLocal;

  DistNumVecToDistRowVec(
      tempVec,
      tempVecLocal,
      domain_.numGridFine,
      numElem_,
      fft.localNzStart,
      fft.localNz,
      fft.isInGrid,
      domain_.colComm );

  // The contribution of the pseudoCharge is subtracted. So the Poisson
  // equation is well defined for neutral system.
  // Only part of the processors participate in the FFTW calculation

  if( fft.isInGrid ){

    // cpxVecLocal saves the Fourier transform of 
    // density_ - pseudoCharge_ 
    CpxNumVec  cpxVecLocal( tempVecLocal.Size() );

    for( Int i = 0; i < ntotLocal; i++ ){
      fft.inputComplexVecLocal(i) = Complex( 
          tempVecLocal(i), 0.0 );
    }

    fftw_execute( fft.forwardPlan );

    blas::Copy( ntotLocal, fft.outputComplexVecLocal.Data(), 1,
        cpxVecLocal.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikLocal  = fft.ikLocal[d];
      for( Int i = 0; i < ntotLocal; i++ ){
        if( fft.gkkLocal(i) == 0 ){
          fft.outputComplexVecLocal(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecLocal(i) = cpxVecLocal(i) *
            2.0 * PI / fft.gkkLocal(i) * ikLocal(i);
        }
      }

      fftw_execute( fft.backwardPlan );

      // vhartDrvLocal saves the derivative of the Hartree potential in
      // the distributed row format
      vhartDrvLocal[d].Resize( tempVecLocal.Size() );

      for( Int i = 0; i < ntotLocal; i++ ){
        vhartDrvLocal[d](i) = fft.inputComplexVecLocal(i).real() / ntot;
      }

    } // for (d)
  } // if (fft.isInGrid)

  // Convert vhartDrvLocal to vhartDrv in the DistNumVec format

  for( Int d = 0; d < DIM; d++ ){
    DistRowVecToDistNumVec( 
        vhartDrvLocal[d],
        vhartDrv[d],
        domain_.numGridFine,
        numElem_,
        fft.localNzStart,
        fft.localNz,
        fft.isInGrid,
        domain_.colComm );
  }




  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Method 1: Using the derivative of the pseudopotential
  if(0){
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
            for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
                mi != ppMap.end(); mi++ ){
              Int atomIdx = (*mi).first;
              PseudoPot& pp = (*mi).second;
              SparseVec& sp = pp.pseudoCharge;
              IntNumVec& idx = sp.first;
              DblNumMat& val = sp.second;
              Real    wgt = domain_.Volume() / domain_.NumGridTotalFine();
              DblNumVec&  vhartVal = vhart.LocalMap()[key];
              Real resX = 0.0;
              Real resY = 0.0;
              Real resZ = 0.0;
              for( Int l = 0; l < idx.m(); l++ ){
                resX -= val(l, DX) * vhartVal[idx(l)] * wgt;
                resY -= val(l, DY) * vhartVal[idx(l)] * wgt;
                resZ -= val(l, DZ) * vhartVal[idx(l)] * wgt;
              }
              forceLocal( atomIdx, 0 ) += resX;
              forceLocal( atomIdx, 1 ) += resY;
              forceLocal( atomIdx, 2 ) += resZ;

            } // for (mi)
          } // own this element
        } // for (i)
  }


  // Method 2: Using integration by parts
  // This method only uses the value of the local pseudopotential and
  // does not use the derivative of the pseudopotential. This is done
  // through integration by parts, and the derivative is applied to the
  // Coulomb potential evaluated on a uniform grid. 
  // 
  // NOTE: For ONCV pseudopotential we can only use this version!
  if(1)
  {
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
            for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
                mi != ppMap.end(); mi++ ){
              Int atomIdx = (*mi).first;
              PseudoPot& pp = (*mi).second;
              SparseVec& sp = pp.pseudoCharge;
              IntNumVec& idx = sp.first;
              DblNumMat& val = sp.second;
              Real    wgt = domain_.Volume() / domain_.NumGridTotalFine();
              for( Int d = 0; d < DIM; d++ ){
                DblNumVec&  drv = vhartDrv[d].LocalMap()[key];
                Real res = 0.0;
                for( Int l = 0; l < idx.m(); l++ ){
                  res += val(l, VAL) * drv[idx(l)] * wgt;
                }
                forceLocal( atomIdx, d ) += res;
              }
            } // for (mi)
          } // own this element
        } // for (i)
  }


  if(0){
    // Output the local component of the force for debugging purpose
    mpi::Allreduce( forceLocal.Data(), force.Data(), numAtom * DIM,
        MPI_SUM, domain_.colComm );
    for( Int a = 0; a < numAtom; a++ ){
      Point3 ft(force(a,0),force(a,1),force(a,2));
      Print( statusOFS, "atom", a, "localforce ", ft );
    }
  }

  // *********************************************************************
  // Compute the force from nonlocal pseudopotential
  // *********************************************************************
  // This method only uses the value of the pseudopotential and does not
  // use the derivative of the pseudopotential. This is done through
  // integration by parts, and the derivative is applied to the basis functions
  // evaluated on a LGL grid. This is illustrated in 
  // hamiltonian_dg_matrix.cpp
  if(1)
  {
    // Step 1. Collect the eigenvectors from the neighboring elements
    // according to the support of the pseudopotential
    // Note: the following part is the same as that in 
    //
    // HamiltonianDG::CalculateDGMatrix
    //
    // Each element owns all the coefficient matrices in its neighbors
    // and then perform data processing later. It can be as many as 
    // 3^3-1 = 26 elements. 
    //
    // Note that it is assumed that the size of the element size cannot
    // be smaller than the pseudopotential (local or nonlocal) cutoff radius.
    //
    // Use std::set to avoid repetitive entries
    std::set<Index3>  pseudoSet;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            IntNumVec  idxX(3);
            IntNumVec  idxY(3);
            IntNumVec  idxZ(3); 

            // Previous
            if( i == 0 )  idxX(0) = numElem_[0]-1; else   idxX(0) = i-1;
            if( j == 0 )  idxY(0) = numElem_[1]-1; else   idxY(0) = j-1;
            if( k == 0 )  idxZ(0) = numElem_[2]-1; else   idxZ(0) = k-1;

            // Current
            idxX(1) = i;
            idxY(1) = j;
            idxZ(1) = k;

            // Next
            if( i == numElem_[0]-1 )  idxX(2) = 0; else   idxX(2) = i+1;
            if( j == numElem_[1]-1 )  idxY(2) = 0; else   idxY(2) = j+1;
            if( k == numElem_[2]-1 )  idxZ(2) = 0; else   idxZ(2) = k+1;

            // Tensor product 
            for( Int c = 0; c < 3; c++ )
              for( Int b = 0; b < 3; b++ )
                for( Int a = 0; a < 3; a++ ){
                  // Not the element key itself
                  if( idxX[a] != i || idxY[b] != j || idxZ[c] != k ){
                    pseudoSet.insert( Index3( idxX(a), idxY(b), idxZ(c) ) );
                  }
                } // for (a)
          }
        } // for (i)
    std::vector<Index3>  pseudoIdx;
    pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );

    eigvecCoef_.SetComm( domain_.colComm );
    eigvecCoef_.GetBegin( pseudoIdx, NO_MASK );
    eigvecCoef_.GetEnd( NO_MASK );

    // Step 2. Loop through the atoms and eigenvecs for the contribution
    // to the force
    //
    // Note: this procedure shall be substituted with the density matrix
    // formulation when PEXSI is used. 

    // Loop over atoms and pseudopotentials
    Int numEig = occupationRate_.m();
    for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
      if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
        DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];    
        Int numVnl = vnlWeight.Size();
        DblNumMat resVal ( numEig, numVnl );
        DblNumMat resDrvX( numEig, numVnl );
        DblNumMat resDrvY( numEig, numVnl );
        DblNumMat resDrvZ( numEig, numVnl );
        SetValue( resVal,  0.0 );
        SetValue( resDrvX, 0.0 );
        SetValue( resDrvY, 0.0 );
        SetValue( resDrvZ, 0.0 );

        // Loop over the elements overlapping with the nonlocal
        // pseudopotential
        for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
            ei  = vnlCoef_.LocalMap().begin();
            ei != vnlCoef_.LocalMap().end(); ei++ ){
          Index3 key = (*ei).first;
          std::map<Int, DblNumMat>& coefMap = (*ei).second; 
          std::map<Int, DblNumMat>& coefDrvXMap = vnlDrvCoef_[0].LocalMap()[key];
          std::map<Int, DblNumMat>& coefDrvYMap = vnlDrvCoef_[1].LocalMap()[key];
          std::map<Int, DblNumMat>& coefDrvZMap = vnlDrvCoef_[2].LocalMap()[key];

          if( eigvecCoef_.LocalMap().find( key ) == eigvecCoef_.LocalMap().end() ){
            ErrorHandling( "Eigenfunction coefficient matrix cannot be located." );
          }

          DblNumMat&  localCoef = eigvecCoef_.LocalMap()[key];

          Int numBasis = localCoef.m();

          if( coefMap.find( atomIdx ) != coefMap.end() ){

            DblNumMat&  coef      = coefMap[atomIdx];
            DblNumMat&  coefDrvX  = coefDrvXMap[atomIdx];
            DblNumMat&  coefDrvY  = coefDrvYMap[atomIdx];
            DblNumMat&  coefDrvZ  = coefDrvZMap[atomIdx];

            // Skip the calculation if there is no adaptive local
            // basis function.  
            if( coef.m() == 0 ){
              continue;
            }

            // Value
            blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
                1.0, localCoef.Data(), numBasis, 
                coef.Data(), numBasis,
                1.0, resVal.Data(), numEig );

            // Derivative
            blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
                1.0, localCoef.Data(), numBasis, 
                coefDrvX.Data(), numBasis,
                1.0, resDrvX.Data(), numEig );

            blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
                1.0, localCoef.Data(), numBasis, 
                coefDrvY.Data(), numBasis,
                1.0, resDrvY.Data(), numEig );

            blas::Gemm( 'T', 'N', numEig, numVnl, numBasis,
                1.0, localCoef.Data(), numBasis, 
                coefDrvZ.Data(), numBasis,
                1.0, resDrvZ.Data(), numEig );

          } // found the atom
        } // for (ei)

        // Add the contribution to the local force
        // The minus sign comes from integration by parts
        // The 4.0 comes from spin (2.0) and that |l> appears twice (2.0)
        for( Int g = 0; g < numEig; g++ ){
          for( Int l = 0; l < numVnl; l++ ){
            forceLocal(atomIdx, 0) += -4.0 * occupationRate_[g] * vnlWeight[l] *
              resVal(g, l) * resDrvX(g, l);
            forceLocal(atomIdx, 1) += -4.0 * occupationRate_[g] * vnlWeight[l] *
              resVal(g, l) * resDrvY(g, l);
            forceLocal(atomIdx, 2) += -4.0 * occupationRate_[g] * vnlWeight[l] *
              resVal(g, l) * resDrvZ(g, l);
          }
        }
      } // own this atom
    } // for (atomIdx)

  }
  // Clean the eigenvecs after the computation of the force
  {
    std::vector<Index3>  eraseKey;
    for( std::map<Index3, DblNumMat>::iterator 
        mi  = eigvecCoef_.LocalMap().begin();
        mi != eigvecCoef_.LocalMap().end(); mi++ ){
      Index3 key = (*mi).first;
      if( eigvecCoef_.Prtn().Owner(key) != (mpirank / dmRow_) ){
        eraseKey.push_back( key );
      }
    }
    for( std::vector<Index3>::iterator vi = eraseKey.begin();
        vi != eraseKey.end(); vi++ ){
      eigvecCoef_.LocalMap().erase( *vi );
    }
  }


  // *********************************************************************
  // Compute the total force and give the value to atomList
  // *********************************************************************
  mpi::Allreduce( forceLocal.Data(), force.Data(), numAtom * DIM,
      MPI_SUM, domain_.colComm );

  for( Int a = 0; a < numAtom; a++ ){
    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
  } 


  return ;
}         // -----  end of method HamiltonianDG::CalculateForce  ----- 


// Old version of the code.
//void
//HamiltonianDG::CalculateForceDM    ( 
//    DistFourier& fft, 
//    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
//{
//    if( !fft.isInitialized ){
//        ErrorHandling("Fourier is not prepared.");
//    }
// 
//    Int mpirank, mpisize;
//    MPI_Comm_rank( domain_.comm, &mpirank );
//    MPI_Comm_size( domain_.comm, &mpisize );
//
//  Real timeSta, timeEnd;
//
//    // *********************************************************************
//    // Initialize the force computation
//    // *********************************************************************
//  Int ntot      = fft.numGridTotal;
//    Int ntotLocal = fft.numGridLocal;
//    Int numAtom   = atomList_.size();
//
//
//    DblNumMat   forceLocal( numAtom, DIM );
//    DblNumMat   force( numAtom, DIM );
//    SetValue( forceLocal, 0.0 );
//    SetValue( force, 0.0 );
//    
//
//  distDMMat.SetComm( domain_.colComm );
//
//    // *********************************************************************
//    // Compute the derivative of the Hartree potential for computing the 
//    // local pseudopotential contribution to the Hellmann-Feynman force
//    // *********************************************************************
//
//#if ( _DEBUGlevel_ >= 1 )
//  statusOFS << "Starting the computation of local derivatives "
//    << std::endl;
//#endif
// 
//    DistDblNumVec&              vhart = vhart_;
//  vhart.SetComm( domain_.colComm );
//    
//  std::vector<DistDblNumVec>  vhartDrv(DIM);
//    std::vector<DblNumVec>      vhartDrvLocal(DIM);
//    DistDblNumVec   tempVec;
//  tempVec.SetComm( domain_.colComm );
//
//    tempVec.Prtn() = elemPrtn_;
//    for( Int d = 0; d < DIM; d++ ){
//        vhartDrv[d].Prtn() = elemPrtn_;
//    vhartDrv[d].SetComm( domain_.colComm );
//  }
//
//    // tempVec = density_ - pseudoCharge_
//    for( Int k = 0; k < numElem_[2]; k++ )
//        for( Int j = 0; j < numElem_[1]; j++ )
//            for( Int i = 0; i < numElem_[0]; i++ ){
//                Index3 key = Index3( i, j, k );
//                if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
//                    tempVec.LocalMap()[key] = density_.LocalMap()[key];
//                    blas::Axpy( numUniformGridElem_.prod(), -1.0, 
//                            pseudoCharge_.LocalMap()[key].Data(), 1,
//                            tempVec.LocalMap()[key].Data(), 1 );
//                }
//            }
//
//
//
//    // Convert tempVec to tempVecLocal in distributed row vector format
//    DblNumVec  tempVecLocal;
//
//  DistNumVecToDistRowVec(
//            tempVec,
//            tempVecLocal,
//            domain_.numGridFine,
//            numElem_,
//            fft.localNzStart,
//            fft.localNz,
//            fft.isInGrid,
//            domain_.colComm );
//
//    // The contribution of the pseudoCharge is subtracted. So the Poisson
//    // equation is well defined for neutral system.
//    // Only part of the processors participate in the FFTW calculation
//    
//
//    if( fft.isInGrid ){
//
//        // cpxVecLocal saves the Fourier transform of 
//        // density_ - pseudoCharge_ 
//        CpxNumVec  cpxVecLocal( tempVecLocal.Size() );
//
//        for( Int i = 0; i < ntotLocal; i++ ){
//            fft.inputComplexVecLocal(i) = Complex( 
//                    tempVecLocal(i), 0.0 );
//        }
//
//
//        fftw_execute( fft.forwardPlan );
//
//        blas::Copy( ntotLocal, fft.outputComplexVecLocal.Data(), 1,
//                cpxVecLocal.Data(), 1 );
//
//        // Compute the derivative of the Hartree potential via Fourier
//        // transform 
//        for( Int d = 0; d < DIM; d++ ){
//            CpxNumVec& ikLocal  = fft.ikLocal[d];
//            for( Int i = 0; i < ntotLocal; i++ ){
//                if( fft.gkkLocal(i) == 0 ){
//                    fft.outputComplexVecLocal(i) = Z_ZERO;
//                }
//                else{
//                    // NOTE: gkk already contains the factor 1/2.
//                    fft.outputComplexVecLocal(i) = cpxVecLocal(i) *
//                        2.0 * PI / fft.gkkLocal(i) * ikLocal(i);
//                }
//            }
//
//            fftw_execute( fft.backwardPlan );
//
//            // vhartDrvLocal saves the derivative of the Hartree potential in
//            // the distributed row format
//            vhartDrvLocal[d].Resize( tempVecLocal.Size() );
//
//            for( Int i = 0; i < ntotLocal; i++ ){
//                vhartDrvLocal[d](i) = fft.inputComplexVecLocal(i).real() / ntot;
//            }
//
//        } // for (d)
//
//
//    } // if (fft.isInGrid)
//
//    // Convert vhartDrvLocal to vhartDrv in the DistNumVec format
//
//    for( Int d = 0; d < DIM; d++ ){
//        DistRowVecToDistNumVec( 
//                vhartDrvLocal[d],
//                vhartDrv[d],
//                domain_.numGridFine,
//                numElem_,
//                fft.localNzStart,
//                fft.localNz,
//                fft.isInGrid,
//                domain_.colComm );
//    }
//
//
//    
//        
//    // *********************************************************************
//    // Compute the force from local pseudopotential
//    // *********************************************************************
//
//#if ( _DEBUGlevel_ >= 1 )
//  statusOFS << "Starting the local part of the force calculation "
//    << std::endl;
//#endif
//
//    // Method 1: Using the derivative of the pseudopotential
//    if(1){
//        for( Int k = 0; k < numElem_[2]; k++ )
//            for( Int j = 0; j < numElem_[1]; j++ )
//                for( Int i = 0; i < numElem_[0]; i++ ){
//                    Index3 key( i, j, k );
//                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
//                        std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
//                        for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
//                                 mi != ppMap.end(); mi++ ){
//                            Int atomIdx = (*mi).first;
//                            PseudoPot& pp = (*mi).second;
//                            SparseVec& sp = pp.pseudoCharge;
//                            IntNumVec& idx = sp.first;
//                            DblNumMat& val = sp.second;
//                            Real    wgt = domain_.Volume() / domain_.NumGridTotalFine();
//                            DblNumVec&  vhartVal = vhart.LocalMap()[key];
//                            Real resX = 0.0;
//                            Real resY = 0.0;
//                            Real resZ = 0.0;
//                            for( Int l = 0; l < idx.m(); l++ ){
//                                resX -= val(l, DX) * vhartVal[idx(l)] * wgt;
//                                resY -= val(l, DY) * vhartVal[idx(l)] * wgt;
//                                resZ -= val(l, DZ) * vhartVal[idx(l)] * wgt;
//                            }
//                            forceLocal( atomIdx, 0 ) += resX;
//                            forceLocal( atomIdx, 1 ) += resY;
//                            forceLocal( atomIdx, 2 ) += resZ;
//
//                        } // for (mi)
//                    } // own this element
//                } // for (i)
//    }
//
//
//    // Method 2: Using integration by parts
//    if(0)
//    {
//        for( Int k = 0; k < numElem_[2]; k++ )
//            for( Int j = 0; j < numElem_[1]; j++ )
//                for( Int i = 0; i < numElem_[0]; i++ ){
//                    Index3 key( i, j, k );
//                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
//                        std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
//                        for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
//                                 mi != ppMap.end(); mi++ ){
//                            Int atomIdx = (*mi).first;
//                            PseudoPot& pp = (*mi).second;
//                            SparseVec& sp = pp.pseudoCharge;
//                            IntNumVec& idx = sp.first;
//                            DblNumMat& val = sp.second;
//                            Real    wgt = domain_.Volume() / domain_.NumGridTotalFine();
//                            for( Int d = 0; d < DIM; d++ ){
//                                DblNumVec&  drv = vhartDrv[d].LocalMap()[key];
//                                Real res = 0.0;
//                                for( Int l = 0; l < idx.m(); l++ ){
//                                    res += val(l, VAL) * drv[idx(l)] * wgt;
//                                }
//                                forceLocal( atomIdx, d ) += res;
//                            }
//                        } // for (mi)
//                    } // own this element
//                } // for (i)
//    }
//
//
//    // *********************************************************************
//    // Compute the force from nonlocal pseudopotential
//    // *********************************************************************
//
//
//  // Use the density matrix instead of the eigenfunctions. 
//  if(1)
//    {
//#if ( _DEBUGlevel_ >= 1 )
//    statusOFS << "Starting the nonlocal part of the force calculation "
//      << std::endl;
//#endif
//
//    // Step 1. Collect the blocks of density matrices according to the
//    // support of the pseudopotential 
//        //
//        // Use std::set to avoid repetitive entries
//    GetTime( timeSta );
//        std::set<ElemMatKey>  pseudoSet;
//    for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
//      if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
//        // Loop over the elements (row indices in the density matrix)
//        // containing the atom
//        for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
//            ei  = vnlCoef_.LocalMap().begin();
//            ei != vnlCoef_.LocalMap().end(); ei++ ){
//          Index3 iKey = (*ei).first;
//
//          // vnlCoef finds the pseudopotential for atomIdx in the element
//          // iKey
//          std::map<Int, DblNumMat>::iterator mi = 
//            (*ei).second.find( atomIdx );
//          if( mi == (*ei).second.end() )
//            continue;
//
//          // It is important that the matrix is nonzero
//          DblNumMat& coef1 = (*mi).second;
//          if( coef1.m() == 0 )
//            continue;
//
//          // Loop over the elements (column indices in the density
//          // matrix) containing the atom
//          for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
//              ej  = vnlCoef_.LocalMap().begin();
//              ej != vnlCoef_.LocalMap().end(); ej++ ){
//            Index3 jKey = (*ej).first;
//
//            // vnlCoef finds the pseudopotential for atomIdx in the
//            // element jKey 
//            std::map<Int, DblNumMat>::iterator ni = 
//              (*ej).second.find( atomIdx );
//
//            if( ni == (*ej).second.end() )
//              continue;
//
//            // It is important that the matrix is nonzero
//            DblNumMat& coef2 = (*ni).second;
//            if( coef2.m() == 0 )
//              continue;
//
//            // Only add the (iKey, jKey) if it does not correspond to a
//            // zero matrix.
//            pseudoSet.insert( ElemMatKey( iKey, jKey ) );
//          }
//        }
//      }
//    }
//
//        std::vector<ElemMatKey>  pseudoIdx;
//        pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );
//
//
//
//#if ( _DEBUGlevel_ >= 1 )
//        statusOFS << "Required density matrix blocks " << std::endl;
//    for( std::vector<ElemMatKey>::iterator vi = pseudoIdx.begin();
//         vi != pseudoIdx.end(); vi++ ){
//      statusOFS << (*vi).first << " -- " << (*vi).second << std::endl;
//    }
//        statusOFS << "Owned density matrix blocks on this processor" << std::endl;
//    for( std::map<ElemMatKey, DblNumMat>::iterator 
//       mi  = distDMMat.LocalMap().begin();
//       mi != distDMMat.LocalMap().end(); mi++ ){
//      ElemMatKey key = (*mi).first;
//      statusOFS << key.first << " -- " << key.second << std::endl;
//    }
//    statusOFS << "Ownerinfo for ElemMatPrtn " <<
//      distDMMat.Prtn().ownerInfo << std::endl; 
//#endif
//    distDMMat.GetBegin( pseudoIdx, NO_MASK );
//    distDMMat.GetEnd( NO_MASK );
//        GetTime( timeEnd );
//#if ( _DEBUGlevel_ >= 1 )
//        statusOFS << "Time for getting the density matrix blocks is " <<
//            timeEnd - timeSta << " [s]" << std::endl << std::endl;
//#endif
//
//
//
//    // Step 2. Loop through the atoms, find the corresponding nonlocal
//    // pseudopotential and the density matrix for the contribution to
//    // the force
//    for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
//      if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
//        DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];    
//        Int numVnl = vnlWeight.Size();
//
//        // Loop over the elements (row indices in the density matrix)
//        // containing the atom
//        for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
//            ei  = vnlCoef_.LocalMap().begin();
//            ei != vnlCoef_.LocalMap().end(); ei++ ){
//          Index3 iKey = (*ei).first;
//
//          // vnlCoef finds the pseudopotential for atomIdx in the element
//          // iKey
//          std::map<Int, DblNumMat>::iterator mi = 
//            (*ei).second.find( atomIdx );
//          if( mi == (*ei).second.end() )
//            continue;
//
//          // It is important that the matrix is nonzero
//          DblNumMat& coef1 = (*mi).second;
//          if( coef1.m() == 0 )
//            continue;
//
//          std::map<Int, DblNumMat>& coefMap = (*ei).second; 
//
//          // Loop over the elements (column indices in the density
//          // matrix) containing the atom
//          for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
//              ej  = vnlCoef_.LocalMap().begin();
//              ej != vnlCoef_.LocalMap().end(); ej++ ){
//            Index3 jKey = (*ej).first;
//
//            // vnlCoef finds the pseudopotential for atomIdx in the
//            // element jKey 
//            std::map<Int, DblNumMat>::iterator ni = 
//              (*ej).second.find( atomIdx );
//
//            if( ni == (*ej).second.end() )
//              continue;
//
//            // It is important that the matrix is nonzero
//            DblNumMat& coef2 = (*ni).second;
//            if( coef2.m() == 0 )
//              continue;
//
//            std::map<Int, DblNumMat>& coefDrvXMap = vnlDrvCoef_[0].LocalMap()[jKey];
//            std::map<Int, DblNumMat>& coefDrvYMap = vnlDrvCoef_[1].LocalMap()[jKey];
//            std::map<Int, DblNumMat>& coefDrvZMap = vnlDrvCoef_[2].LocalMap()[jKey];
//
//            ElemMatKey matKey = ElemMatKey(iKey, jKey);
//
//            if( distDMMat.LocalMap().find( matKey ) == 
//                distDMMat.LocalMap().end() ){
//              std::ostringstream msg;
//              msg << std::endl
//                << "Cannot find the density matrix component." << std::endl
//                << "AtomIdx: " << atomIdx << std::endl
//                << "Row index3: " << iKey << std::endl
//                << "Col index3: " << jKey << std::endl;
//              ErrorHandling( msg.str().c_str() );
//            }
//
//            DblNumMat&  localDM   = distDMMat.LocalMap()[matKey];
//
//            DblNumMat&  coef      = coefMap[atomIdx];
//            DblNumMat&  coefDrvX  = coefDrvXMap[atomIdx];
//            DblNumMat&  coefDrvY  = coefDrvYMap[atomIdx];
//            DblNumMat&  coefDrvZ  = coefDrvZMap[atomIdx];
//
//
//            // Add to the force.
//            // The minus sign comes from integration by parts Spin = 2.0
//            // is assumed.  The 2.0 comes from that |l> appears twice,
//            // one in the value and one in the derivative
//
//            for( Int l = 0; l < numVnl; l++ ){
//              for( Int a = 0; a < localDM.m(); a++ ){
//                for( Int b = 0; b < localDM.n(); b++ ){
//                  forceLocal(atomIdx, 0) += -2.0 * vnlWeight[l] *
//                    coef(a, l) * coefDrvX(b, l) * localDM(a,b);
//                  forceLocal(atomIdx, 1) += -2.0 * vnlWeight[l] *
//                    coef(a, l) * coefDrvY(b, l) * localDM(a,b);
//                  forceLocal(atomIdx, 2) += -2.0 * vnlWeight[l] *
//                    coef(a, l) * coefDrvZ(b, l) * localDM(a,b);
//                }
//              }
//            }
//            
//          } // for (ej)
//        } // for (ei)
//      } // own this atom
//    } // for (atomIdx)
//
//    }
//
//
//    // *********************************************************************
//    // Compute the total force and give the value to atomList
//    // *********************************************************************
//    mpi::Allreduce( forceLocal.Data(), force.Data(), numAtom * DIM,
//            MPI_SUM, domain_.colComm );
//
//    for( Int a = 0; a < numAtom; a++ ){
//        atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
//    } 
//
//
//    return ;
//}         // -----  end of method HamiltonianDG::CalculateForceDM  ----- 



// New version of code 12/19/2014
// This version does not use FFT anymore, and use LGL grid to compute
// the local potential contribution of the force
void
HamiltonianDG::CalculateForceDM    ( 
    DistFourier& fft, 
    DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distDMMat )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int mpirank, mpisize;
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );

  Real timeSta, timeEnd;

  // *********************************************************************
  // Initialize the force computation
  // *********************************************************************
  Int ntot      = fft.numGridTotal;
  Int ntotLocal = fft.numGridLocal;
  Int numAtom   = atomList_.size();


  DblNumMat   forceLocal( numAtom, DIM );
  DblNumMat   force( numAtom, DIM );
  SetValue( forceLocal, 0.0 );
  SetValue( force, 0.0 );


  distDMMat.SetComm( domain_.colComm );

  // *********************************************************************
  // Compute the derivative of the Hartree potential for computing the 
  // local pseudopotential contribution to the Hellmann-Feynman force
  // *********************************************************************


#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Starting the computation of local derivatives "
    << std::endl;
#endif

  DistDblNumVec&              vhart = vhart_;
  vhart.SetComm( domain_.colComm );

  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Starting the local part of the force calculation "
    << std::endl;
#endif

  // Method 1: Using the derivative of the pseudopotential. Integrated
  // on a uniform grid
  if(1){
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            std::map<Int, PseudoPot>&  ppMap = pseudo_.LocalMap()[key];
            for( std::map<Int, PseudoPot>::iterator mi = ppMap.begin();
                mi != ppMap.end(); mi++ ){
              Int atomIdx = (*mi).first;
              PseudoPot& pp = (*mi).second;
              SparseVec& sp = pp.pseudoCharge;
              IntNumVec& idx = sp.first;
              DblNumMat& val = sp.second;
              Real    wgt = domain_.Volume() / domain_.NumGridTotalFine();
              DblNumVec&  vhartVal = vhart.LocalMap()[key];
              Real resX = 0.0;
              Real resY = 0.0;
              Real resZ = 0.0;
              for( Int l = 0; l < idx.m(); l++ ){
                resX -= val(l, DX) * vhartVal[idx(l)] * wgt;
                resY -= val(l, DY) * vhartVal[idx(l)] * wgt;
                resZ -= val(l, DZ) * vhartVal[idx(l)] * wgt;
              }
              forceLocal( atomIdx, 0 ) += resX;
              forceLocal( atomIdx, 1 ) += resY;
              forceLocal( atomIdx, 2 ) += resZ;

            } // for (mi)
          } // own this element
        } // for (i)
  }



  // *********************************************************************
  // Compute the force from nonlocal pseudopotential
  // *********************************************************************


  // Use the density matrix instead of the eigenfunctions. 
  if(1)
  {
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Starting the nonlocal part of the force calculation "
      << std::endl;
#endif

    // Step 1. Collect the blocks of density matrices according to the
    // support of the pseudopotential 
    //
    // Use std::set to avoid repetitive entries
    GetTime( timeSta );
    std::set<ElemMatKey>  pseudoSet;
    for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
      if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
        // Loop over the elements (row indices in the density matrix)
        // containing the atom
        for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
            ei  = vnlCoef_.LocalMap().begin();
            ei != vnlCoef_.LocalMap().end(); ei++ ){
          Index3 iKey = (*ei).first;

          // vnlCoef finds the pseudopotential for atomIdx in the element
          // iKey
          std::map<Int, DblNumMat>::iterator mi = 
            (*ei).second.find( atomIdx );
          if( mi == (*ei).second.end() )
            continue;

          // It is important that the matrix is nonzero
          DblNumMat& coef1 = (*mi).second;
          if( coef1.m() == 0 )
            continue;

          // Loop over the elements (column indices in the density
          // matrix) containing the atom
          for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
              ej  = vnlCoef_.LocalMap().begin();
              ej != vnlCoef_.LocalMap().end(); ej++ ){
            Index3 jKey = (*ej).first;

            // vnlCoef finds the pseudopotential for atomIdx in the
            // element jKey 
            std::map<Int, DblNumMat>::iterator ni = 
              (*ej).second.find( atomIdx );

            if( ni == (*ej).second.end() )
              continue;

            // It is important that the matrix is nonzero
            DblNumMat& coef2 = (*ni).second;
            if( coef2.m() == 0 )
              continue;

            // Only add the (iKey, jKey) if it does not correspond to a
            // zero matrix.
            pseudoSet.insert( ElemMatKey( iKey, jKey ) );
          }
        }
      }
    }

    std::vector<ElemMatKey>  pseudoIdx;
    pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );



#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Required density matrix blocks " << std::endl;
    for( std::vector<ElemMatKey>::iterator vi = pseudoIdx.begin();
        vi != pseudoIdx.end(); vi++ ){
      statusOFS << (*vi).first << " -- " << (*vi).second << std::endl;
    }
    statusOFS << "Owned density matrix blocks on this processor" << std::endl;
    for( std::map<ElemMatKey, DblNumMat>::iterator 
        mi  = distDMMat.LocalMap().begin();
        mi != distDMMat.LocalMap().end(); mi++ ){
      ElemMatKey key = (*mi).first;
      statusOFS << key.first << " -- " << key.second << std::endl;
    }
    statusOFS << "Ownerinfo for ElemMatPrtn " <<
      distDMMat.Prtn().ownerInfo << std::endl; 
#endif

    distDMMat.GetBegin( pseudoIdx, NO_MASK );
    distDMMat.GetEnd( NO_MASK );


    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for getting the density matrix blocks is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



    // Step 2. Loop through the atoms, find the corresponding nonlocal
    // pseudopotential and the density matrix for the contribution to
    // the force
    for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
      if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
        DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];    
        Int numVnl = vnlWeight.Size();

        // Loop over the elements (row indices in the density matrix)
        // containing the atom
        for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
            ei  = vnlCoef_.LocalMap().begin();
            ei != vnlCoef_.LocalMap().end(); ei++ ){
          Index3 iKey = (*ei).first;

          // vnlCoef finds the pseudopotential for atomIdx in the element
          // iKey
          std::map<Int, DblNumMat>::iterator mi = 
            (*ei).second.find( atomIdx );
          if( mi == (*ei).second.end() )
            continue;

          // It is important that the matrix is nonzero
          DblNumMat& coef1 = (*mi).second;
          if( coef1.m() == 0 )
            continue;

          std::map<Int, DblNumMat>& coefMap = (*ei).second; 

          // Loop over the elements (column indices in the density
          // matrix) containing the atom
          for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
              ej  = vnlCoef_.LocalMap().begin();
              ej != vnlCoef_.LocalMap().end(); ej++ ){
            Index3 jKey = (*ej).first;

            // vnlCoef finds the pseudopotential for atomIdx in the
            // element jKey 
            std::map<Int, DblNumMat>::iterator ni = 
              (*ej).second.find( atomIdx );

            if( ni == (*ej).second.end() )
              continue;

            // It is important that the matrix is nonzero
            DblNumMat& coef2 = (*ni).second;
            if( coef2.m() == 0 )
              continue;

            std::map<Int, DblNumMat>& coefDrvXMap = vnlDrvCoef_[0].LocalMap()[jKey];
            std::map<Int, DblNumMat>& coefDrvYMap = vnlDrvCoef_[1].LocalMap()[jKey];
            std::map<Int, DblNumMat>& coefDrvZMap = vnlDrvCoef_[2].LocalMap()[jKey];

            ElemMatKey matKey = ElemMatKey(iKey, jKey);

            if( distDMMat.LocalMap().find( matKey ) == 
                distDMMat.LocalMap().end() ){
              std::ostringstream msg;
              msg << std::endl
                << "Cannot find the density matrix component." << std::endl
                << "AtomIdx: " << atomIdx << std::endl
                << "Row index3: " << iKey << std::endl
                << "Col index3: " << jKey << std::endl;
              ErrorHandling( msg.str().c_str() );
            }

            DblNumMat&  localDM   = distDMMat.LocalMap()[matKey];

            DblNumMat&  coef      = coefMap[atomIdx];
            DblNumMat&  coefDrvX  = coefDrvXMap[atomIdx];
            DblNumMat&  coefDrvY  = coefDrvYMap[atomIdx];
            DblNumMat&  coefDrvZ  = coefDrvZMap[atomIdx];


            // Add to the force.
            // The minus sign comes from integration by parts Spin = 2.0
            // is assumed.  The 2.0 comes from that |l> appears twice,
            // one in the value and one in the derivative

            for( Int l = 0; l < numVnl; l++ ){
              for( Int a = 0; a < localDM.m(); a++ ){
                for( Int b = 0; b < localDM.n(); b++ ){
                  forceLocal(atomIdx, 0) += -2.0 * vnlWeight[l] *
                    coef(a, l) * coefDrvX(b, l) * localDM(a,b);
                  forceLocal(atomIdx, 1) += -2.0 * vnlWeight[l] *
                    coef(a, l) * coefDrvY(b, l) * localDM(a,b);
                  forceLocal(atomIdx, 2) += -2.0 * vnlWeight[l] *
                    coef(a, l) * coefDrvZ(b, l) * localDM(a,b);
                }
              }
            }

          } // for (ej)
        } // for (ei)
      } // own this atom
    } // for (atomIdx)

  }


  // *********************************************************************
  // Compute the total force and give the value to atomList
  // *********************************************************************
  mpi::Allreduce( forceLocal.Data(), force.Data(), numAtom * DIM,
      MPI_SUM, domain_.colComm );

  for( Int a = 0; a < numAtom; a++ ){
    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
  } 


  return ;
}         // -----  end of method HamiltonianDG::CalculateForceDM  ----- 

// FIXME This does not work when intra-element parallelization is used.
void
HamiltonianDG::CalculateAPosterioriError    ( 
    DblNumTns&       eta2Total,
    DblNumTns&       eta2Residual,
    DblNumTns&       eta2GradJump,
    DblNumTns&       eta2Jump    )
{

  Int mpirank, mpisize;
  Int numAtom = atomList_.size();
  MPI_Comm_rank( domain_.comm, &mpirank );
  MPI_Comm_size( domain_.comm, &mpisize );
  Real timeSta, timeEnd;

  // Here numGrid is the LGL grid
  Point3 length       = domainElem_(0,0,0).length;
  Index3 numGrid      = numLGLGridElem_;             
  Int    numGridTotal = numGrid.prod();
  Int    numSpin      = this->NumSpin();

  // Hamiltonian acting on the basis functions
  DistDblNumMat   HbasisLGL;

  // The derivative of basisLGL along x,y,z directions
  std::vector<DistDblNumMat>   Dbasis(DIM);

  // The inner product of <l|psi> for each pair of nonlocal projector
  // saved on the local processor and all eigenfunctions psi.
  //
  // The data attribute of this structure is 
  //
  // atom index (map to) Matrix value of the corresponding <l|psi>
  DistVec<Index3, std::map<Int, DblNumMat>, ElemPrtn>  vnlPsi;

  // Jump of the value of the basis, and average of the
  // derivative of the basis function, each of size 6 describing
  // the different faces along the X/Y/Z directions. L/R: left/right.
  enum{
    XL = 0,
    XR = 1,
    YL = 2,
    YR = 3,
    ZL = 4,
    ZR = 5,
    NUM_FACE = 6,
  };
  std::vector<DistDblNumMat>   basisJump(NUM_FACE);
  std::vector<DistDblNumMat>   DbasisJump(NUM_FACE);


  // Define the local estimators on each processor.
  DblNumTns eta2ResidualLocal( numElem_[0], numElem_[1], numElem_[2] );
  DblNumTns eta2GradJumpLocal( numElem_[0], numElem_[1], numElem_[2] );
  DblNumTns eta2JumpLocal( numElem_[0], numElem_[1], numElem_[2] );




  // *********************************************************************
  // Initial setup
  // *********************************************************************

  {
    HbasisLGL.Prtn()     = elemPrtn_;

    for( Int i = 0; i < NUM_FACE; i++ ){
      basisJump[i].Prtn()     = elemPrtn_;
      DbasisJump[i].Prtn()    = elemPrtn_;
    }

    for( Int i = 0; i < DIM; i++ ){
      Dbasis[i].Prtn() = elemPrtn_;
    }

    vnlPsi.Prtn()        = elemPrtn_;

    eta2Total.Resize( numElem_[0], numElem_[1], numElem_[2] );
    eta2Residual.Resize( numElem_[0], numElem_[1], numElem_[2] );
    eta2GradJump.Resize( numElem_[0], numElem_[1], numElem_[2] );
    eta2Jump.Resize( numElem_[0], numElem_[1], numElem_[2] );

    SetValue( eta2Total, 0.0 );
    SetValue( eta2Residual, 0.0 );
    SetValue( eta2GradJump, 0.0 );
    SetValue( eta2Jump, 0.0 );

    SetValue( eta2ResidualLocal, 0.0 );
    SetValue( eta2GradJumpLocal, 0.0 );
    SetValue( eta2JumpLocal, 0.0 );

  }


  // *********************************************************************
  // Collect the eigenvectors from the neighboring elements according to
  // the support of the pseudopotential
  // *********************************************************************
  {
    // Each element owns all the coefficient matrices in its neighbors
    // and then perform data processing later. It can be as many as 
    // 3^3-1 = 26 elements. 
    //
    // Note that it is assumed that the size of the element size cannot
    // be smaller than the pseudopotential (local or nonlocal) cutoff radius.
    //
    // Use std::set to avoid repetitive entries
    GetTime( timeSta );
    std::set<Index3>  pseudoSet;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            IntNumVec  idxX(3);
            IntNumVec  idxY(3);
            IntNumVec  idxZ(3); 

            // Previous
            if( i == 0 )  idxX(0) = numElem_[0]-1; else   idxX(0) = i-1;
            if( j == 0 )  idxY(0) = numElem_[1]-1; else   idxY(0) = j-1;
            if( k == 0 )  idxZ(0) = numElem_[2]-1; else   idxZ(0) = k-1;

            // Current
            idxX(1) = i;
            idxY(1) = j;
            idxZ(1) = k;

            // Next
            if( i == numElem_[0]-1 )  idxX(2) = 0; else   idxX(2) = i+1;
            if( j == numElem_[1]-1 )  idxY(2) = 0; else   idxY(2) = j+1;
            if( k == numElem_[2]-1 )  idxZ(2) = 0; else   idxZ(2) = k+1;

            // Tensor product 
            for( Int c = 0; c < 3; c++ )
              for( Int b = 0; b < 3; b++ )
                for( Int a = 0; a < 3; a++ ){
                  // Not the element key itself
                  if( idxX[a] != i || idxY[b] != j || idxZ[c] != k ){
                    pseudoSet.insert( Index3( idxX(a), idxY(b), idxZ(c) ) );
                  }
                } // for (a)
          }
        } // for (i)
    std::vector<Index3>  pseudoIdx;
    pseudoIdx.insert( pseudoIdx.begin(), pseudoSet.begin(), pseudoSet.end() );

    eigvecCoef_.GetBegin( pseudoIdx, NO_MASK );
    eigvecCoef_.GetEnd( NO_MASK );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for getting the coefficients is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // *********************************************************************
  // Compute the residual term.
  // *********************************************************************

  // Prepare the H*basis but without the nonlocal contribution
  {
    GetTime(timeSta);

    // Compute H * basis
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            DblNumMat&  basis = basisLGL_.LocalMap()[key];
            Int numBasis = basis.n();

            DblNumMat empty( basis.m(), basis.n() );
            SetValue( empty, 0.0 );
            HbasisLGL.LocalMap()[key] = empty; 

            // Skip the calculation if there is no basis functions in
            // the element.
            if( numBasis == 0 )
              continue;

            DblNumMat& Hbasis = HbasisLGL.LocalMap()[key];

            // Laplacian part
            for( Int d = 0; d < DIM; d++ ){
              DblNumMat D(basis.m(), basis.n());
              DblNumMat D2(basis.m(), basis.n());
              SetValue( D, 0.0 );
              SetValue( D2, 0.0 );
              for( Int g = 0; g < numBasis; g++ ){
                DiffPsi( numGrid, basis.VecData(g), D.VecData(g), d );
                DiffPsi( numGrid, D.VecData(g), D2.VecData(g), d );
              }
              Dbasis[d].LocalMap()[key] = D;
              blas::Axpy( D2.Size(), -0.5, D2.Data(), 1,
                  Hbasis.Data(), 1 );
            }

            // Local pseudopotential part
            {
              DblNumVec&  vtot  = vtotLGL_.LocalMap()[key];
              for( Int g = 0; g < numBasis; g++ ){
                Real*   ptrVtot   = vtot.Data();
                Real*   ptrBasis  = basis.VecData(g);
                Real*   ptrHbasis = Hbasis.VecData(g);
                for( Int p = 0; p < vtot.Size(); p++ ){
                  *(ptrHbasis++) += (*(ptrVtot++)) * (*(ptrBasis++));
                }
              }
            }

          } // if (own this element)
        } // for (i)
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for H * basis is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // Prepare <l|psi> for each nonlocal projector l and eigenfunction
  // psi.  This is done through the structure vnlPsi
  {
    GetTime(timeSta);

    Int numEig = occupationRate_.m();

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            std::map<Int, PseudoPot>& pseudoMap =
              pseudo_.LocalMap()[key];
            std::map<Int, DblNumMat>  vnlPsiMap;

            // Loop over atoms, regardless of whether this atom belongs
            // to this element or not.
            for(std::map<Int, PseudoPot>::iterator 
                mi  = pseudoMap.begin();
                mi != pseudoMap.end(); mi++ ){
              Int atomIdx = (*mi).first;
              std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
              Int numVnl = vnlList.size();
              DblNumMat   vnlPsiMat( numVnl, numEig );
              SetValue( vnlPsiMat, 0.0 );

              // Loop over all neighboring elements to compute the
              // inner product. 
              // vnlCoef saves the information of <l|phi> where phi are
              // the basis functions.  
              //
              // <l|psi_i> = \sum_{jk} <l|phi_{jk}> C_{jk;i}
              //
              // where C_{jk;i} is saved in eigvecCoef
              for(std::map<Index3, std::map<Int, DblNumMat> >::iterator 
                  ei  = vnlCoef_.LocalMap().begin();
                  ei != vnlCoef_.LocalMap().end(); ei++ ){
                Index3 keyNB                      = (*ei).first;
                std::map<Int, DblNumMat>& coefMap = (*ei).second; 
                DblNumMat&  eigCoef               = eigvecCoef_.LocalMap()[keyNB];

                if( coefMap.find( atomIdx ) != coefMap.end() ){

                  DblNumMat&  coef      = coefMap[atomIdx];

                  Int numBasis = coef.m();

                  // Skip the calculation if there is no adaptive local
                  // basis function.  
                  if( numBasis == 0 ){
                    continue;
                  }

                  // Inner product (NOTE The conjugate is done in a
                  // very crude way here, may need to be revised
                  // when complex arithmetic is considered)
                  blas::Gemm( 'T', 'N', numVnl, numEig, numBasis,
                      1.0, coef.Data(), numBasis, 
                      eigCoef.Data(), numBasis,
                      1.0, vnlPsiMat.Data(), numVnl );

                } // found the atom
              } // for (ei)
              vnlPsiMap[atomIdx] = vnlPsiMat;
            } // for (mi)

            vnlPsi.LocalMap()[key] = vnlPsiMap;

          } // if (own this element)
        } // for (i)
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing <l|psi> is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  {
    GetTime(timeSta);


    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            DblNumMat&  basis =  basisLGL_.LocalMap()[key];
            Int numLGL   = basis.m();
            Int numBasis = basis.n();

            // Do not directly skip the calculatoin even if numBasis ==
            // 0, due to the contribution from the nonlocal
            // pseudopotential. 


            DblNumMat& Hbasis     = HbasisLGL.LocalMap()[key];
            DblNumMat& localCoef  = eigvecCoef_.LocalMap()[key];
            DblNumVec& eig        = eigVal_;
            DblNumVec& occrate    = occupationRate_;

            Int numEig = localCoef.n();

            // Prefactor for the residual term.
            Real hK = length.l2(); // diameter of the domain
            Real pK = std::max( 1, numBasis );
            Real facR = ( hK * hK ) / ( pK * pK );

            DblNumVec  residual( numLGL );
            for( Int g = 0; g < numEig; g++ ){
              SetValue( residual, 0.0 );
              // Contribution from local part
              if( numBasis > 0 ){
                // r = Hbasis * V(g) 
                blas::Gemv( 'N', numLGL, numBasis, 1.0, Hbasis.Data(),
                    numLGL, localCoef.VecData(g), 1, 1.0, 
                    residual.Data(), 1 );
                // r = r - e(g) * basis * V(g)
                blas::Gemv( 'N', numLGL, numBasis, -eig(g), basis.Data(),
                    numLGL, localCoef.VecData(g), 1, 1.0,
                    residual.Data(), 1 );
              } // local part

              // Contribution from the nonlocal pseudopotential
              {
                std::map<Int, PseudoPot>& pseudoMap  = pseudo_.LocalMap()[key];
                std::map<Int, DblNumMat>&  vnlPsiMap = vnlPsi.LocalMap()[key];
                for(std::map<Int, PseudoPot>::iterator 
                    mi  = pseudoMap.begin();
                    mi != pseudoMap.end(); mi++ ){
                  Int atomIdx = (*mi).first;
                  std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
                  Int numVnl = vnlList.size();
                  DblNumMat&  vnlPsiMat = vnlPsiMap[atomIdx];
                  DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];    

                  // Loop over projector
                  for( Int l = 0; l < vnlList.size(); l++ ){
                    SparseVec&  vnl = vnlList[l].first;
                    IntNumVec&  idx = vnl.first;
                    DblNumMat&  val = vnl.second;


                    Real fac = vnlWeight[l] * vnlPsiMat( l, g );

                    for( Int p = 0; p < idx.Size(); p++ ){
                      residual[idx(p)] += fac * val(p, VAL);
                    }
                  }

                } // for (mi)
              }

              Real* ptrR = residual.Data();
              Real* ptrW = LGLWeight3D_.Data();
              Real  tmpR = 0.0;
              for( Int p = 0; p < numLGL; p++ ){
                tmpR += (*ptrR) * (*ptrR) * (*ptrW);
                ptrR++; ptrW++;
              }
              eta2ResidualLocal( i, j, k ) += 
                tmpR * occrate(g) * numSpin * facR;
            } // for (eigenfunction)


          } // if (own this element)
        } // for (i)

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing the local residual is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // *********************************************************************
  // Compute the jump term
  // *********************************************************************
  {
    GetTime(timeSta);
    // Compute average of derivatives and jump of values
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            DblNumMat&  basis = basisLGL_.LocalMap()[key];
            Int numBasis = basis.n();

            // NOTE Still construct empty matrices when numBasis = 0


            // x-direction
            {
              Int  numGridFace = numGrid[1] * numGrid[2];
              DblNumMat emptyX( numGridFace, numBasis );
              SetValue( emptyX, 0.0 );
              basisJump[XL].LocalMap()[key] = emptyX;
              basisJump[XR].LocalMap()[key] = emptyX;
              DbasisJump[XL].LocalMap()[key] = emptyX;
              DbasisJump[XR].LocalMap()[key] = emptyX;

              DblNumMat&  valL = basisJump[XL].LocalMap()[key];
              DblNumMat&  valR = basisJump[XR].LocalMap()[key];
              DblNumMat&  drvL = DbasisJump[XL].LocalMap()[key];
              DblNumMat&  drvR = DbasisJump[XR].LocalMap()[key];
              DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];

              // Form jumps of the values and derivatives of the basis
              // functions from volume to face.

              // basis(0,:,:)             -> valL
              // basis(numGrid[0]-1,:,:)  -> valR
              // Dbasis(0,:,:)            -> drvL
              // Dbasis(numGrid[0]-1,:,:) -> drvR
              for( Int g = 0; g < numBasis; g++ ){
                Int idx, idxL, idxR;
                for( Int gk = 0; gk < numGrid[2]; gk++ )
                  for( Int gj = 0; gj < numGrid[1]; gj++ ){
                    idx  = gj + gk*numGrid[1];
                    idxL = 0 + gj*numGrid[0] + gk * (numGrid[0] *
                        numGrid[1]);
                    idxR = (numGrid[0]-1) + gj*numGrid[0] + gk * (numGrid[0] *
                        numGrid[1]);

                    // 1.0, -1.0 comes from jump with different normal vectors
                    // [[a]] = -(1.0) a_L + (1.0) a_R
                    drvL(idx, g) = -1.0 * DbasisX( idxL, g );
                    drvR(idx, g) = +1.0 * DbasisX( idxR, g );

                    valL(idx, g) = -1.0 * basis( idxL, g );
                    valR(idx, g) = +1.0 * basis( idxR, g );
                  } // for (gj)
              } // for (g)

            } // x-direction


            // y-direction
            {
              Int  numGridFace = numGrid[0] * numGrid[2];
              DblNumMat emptyY( numGridFace, numBasis );
              SetValue( emptyY, 0.0 );
              basisJump[YL].LocalMap()[key] = emptyY;
              basisJump[YR].LocalMap()[key] = emptyY;
              DbasisJump[YL].LocalMap()[key] = emptyY;
              DbasisJump[YR].LocalMap()[key] = emptyY;

              DblNumMat&  valL = basisJump[YL].LocalMap()[key];
              DblNumMat&  valR = basisJump[YR].LocalMap()[key];
              DblNumMat&  drvL = DbasisJump[YL].LocalMap()[key];
              DblNumMat&  drvR = DbasisJump[YR].LocalMap()[key];
              DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];

              // Form jumps of the values and derivatives of the basis
              // functions from volume to face.

              // basis(0,:,:)             -> valL
              // basis(numGrid[0]-1,:,:)  -> valR
              // Dbasis(0,:,:)            -> drvL
              // Dbasis(numGrid[0]-1,:,:) -> drvR
              for( Int g = 0; g < numBasis; g++ ){
                Int idx, idxL, idxR;
                for( Int gk = 0; gk < numGrid[2]; gk++ )
                  for( Int gi = 0; gi < numGrid[0]; gi++ ){
                    idx  = gi + gk*numGrid[0];
                    idxL = gi + 0 *numGrid[0] +
                      gk * (numGrid[0] * numGrid[1]);
                    idxR = gi + (numGrid[1]-1)*numGrid[0] + 
                      gk * (numGrid[0] * numGrid[1]);

                    // 1.0, -1.0 comes from jump with different normal vectors
                    // [[a]] = -(1.0) a_L + (1.0) a_R
                    drvL(idx, g) = -1.0 * DbasisY( idxL, g );
                    drvR(idx, g) = +1.0 * DbasisY( idxR, g );

                    valL(idx, g) = -1.0 * basis( idxL, g );
                    valR(idx, g) = +1.0 * basis( idxR, g );
                  } // for (gj)
              } // for (g)

            } // y-direction

            // z-direction
            {
              Int  numGridFace = numGrid[0] * numGrid[1];
              DblNumMat emptyZ( numGridFace, numBasis );
              SetValue( emptyZ, 0.0 );
              basisJump[ZL].LocalMap()[key] = emptyZ;
              basisJump[ZR].LocalMap()[key] = emptyZ;
              DbasisJump[ZL].LocalMap()[key] = emptyZ;
              DbasisJump[ZR].LocalMap()[key] = emptyZ;

              DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
              DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
              DblNumMat&  drvL = DbasisJump[ZL].LocalMap()[key];
              DblNumMat&  drvR = DbasisJump[ZR].LocalMap()[key];
              DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

              // Form jumps of the values and derivatives of the basis
              // functions from volume to face.

              // basis(0,:,:)             -> valL
              // basis(numGrid[0]-1,:,:)  -> valR
              // Dbasis(0,:,:)            -> drvL
              // Dbasis(numGrid[0]-1,:,:) -> drvR
              for( Int g = 0; g < numBasis; g++ ){
                Int idx, idxL, idxR;
                for( Int gj = 0; gj < numGrid[1]; gj++ )
                  for( Int gi = 0; gi < numGrid[0]; gi++ ){
                    idx  = gi + gj*numGrid[0];
                    idxL = gi + gj*numGrid[0] +
                      0 * (numGrid[0] * numGrid[1]);
                    idxR = gi + gj*numGrid[0] +
                      (numGrid[2]-1) * (numGrid[0] * numGrid[1]);

                    // 1.0, -1.0 comes from jump with different normal vectors
                    // [[a]] = -(1.0) a_L + (1.0) a_R
                    drvL(idx, g) = -1.0 * DbasisZ( idxL, g );
                    drvR(idx, g) = +1.0 * DbasisZ( idxR, g );

                    valL(idx, g) = -1.0 * basis( idxL, g );
                    valR(idx, g) = +1.0 * basis( idxR, g );
                  } // for (gj)
              } // for (g)

            } // z-direction

          }
        } // for (i)

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for constructing the boundary terms is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  {
    GetTime(timeSta);
    std::set<Index3>   boundaryXset;
    std::set<Index3>   boundaryYset;
    std::set<Index3>   boundaryZset;

    std::vector<Index3>   boundaryXIdx;
    std::vector<Index3>   boundaryYIdx; 
    std::vector<Index3>   boundaryZIdx;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key(i, j, k);
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
            // Periodic boundary condition. Everyone only considers the
            // previous(left/back/down) directions and compute.
            Int p1; if( i == 0 )  p1 = numElem_[0]-1; else   p1 = i-1;
            Int p2; if( j == 0 )  p2 = numElem_[1]-1; else   p2 = j-1;
            Int p3; if( k == 0 )  p3 = numElem_[2]-1; else   p3 = k-1;
            boundaryXset.insert( Index3( p1, j,  k) );
            boundaryYset.insert( Index3( i, p2,  k ) );
            boundaryZset.insert( Index3( i,  j, p3 ) ); 
          }
        } // for (i)

    // The left element passes the values on the right face.

    boundaryXIdx.insert( boundaryXIdx.begin(), boundaryXset.begin(), boundaryXset.end() );
    boundaryYIdx.insert( boundaryYIdx.begin(), boundaryYset.begin(), boundaryYset.end() );
    boundaryZIdx.insert( boundaryZIdx.begin(), boundaryZset.begin(), boundaryZset.end() );

    DbasisJump[XR].GetBegin( boundaryXIdx, NO_MASK );
    DbasisJump[YR].GetBegin( boundaryYIdx, NO_MASK );
    DbasisJump[ZR].GetBegin( boundaryZIdx, NO_MASK );

    basisJump[XR].GetBegin( boundaryXIdx, NO_MASK );
    basisJump[YR].GetBegin( boundaryYIdx, NO_MASK );
    basisJump[ZR].GetBegin( boundaryZIdx, NO_MASK );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "After the GetBegin part of communication." << std::endl;
    statusOFS << "Time for GetBegin is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  {
    GetTime( timeSta );
    DbasisJump[XR].GetEnd( NO_MASK );
    DbasisJump[YR].GetEnd( NO_MASK );
    DbasisJump[ZR].GetEnd( NO_MASK );

    basisJump[XR].GetEnd( NO_MASK );
    basisJump[YR].GetEnd( NO_MASK );
    basisJump[ZR].GetEnd( NO_MASK );

    MPI_Barrier( domain_.comm );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for the remaining communication cost is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  {
    GetTime( timeSta );
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

            // x-direction
            {
              // keyL is the previous element received from GetBegin/GetEnd.
              // keyR is the current element
              Int p1; if( i == 0 )  p1 = numElem_[0]-1; else   p1 = i-1;
              Index3 keyL( p1, j, k );
              Index3 keyR = key;

              Int  numGridFace = numGrid[1] * numGrid[2];

              // Note that the notation can be very confusing here:
              // The left element (keyL) contributes to the right face
              // (XR), and the right element (keyR) contributes to the
              // left face (XL)
              DblNumMat&  valL = basisJump[XR].LocalMap()[keyL];
              DblNumMat&  valR = basisJump[XL].LocalMap()[keyR];
              DblNumMat&  drvL = DbasisJump[XR].LocalMap()[keyL];
              DblNumMat&  drvR = DbasisJump[XL].LocalMap()[keyR];

              Int numBasisL = valL.n();
              Int numBasisR = valR.n();

              DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
              DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
              DblNumVec& eig         = eigVal_;
              DblNumVec& occrate     = occupationRate_;

              // Diameter of the face
              Real hF = std::sqrt( length[1] * length[1] + length[2] * length[2] );
              Real pF = std::max( numBasisL, numBasisR );

              // Skip the computation if there is no basis function on
              // either side.  
              if( pF > 0 ){
                // factor in the estimator for jump of the derivative and
                // the jump of the values
                // NOTE:
                // Originally in [Giani 2012] the penalty term in the a
                // posteriori error estimator should be
                //
                // facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
                // should be
                //
                // gamma * p_F^2 / h_F = penaltyAlpha_
                //
                // used in this code.  So 
                //
                // facJ = penaltyAlpha_^2 * h_F / p_F


                // One 0.5 comes from double counting on the faces F,
                // and the other comes from the 1/2 in front of the
                // Laplacian operator.
                Real facGJ = 0.5 * 0.5 * hF / pF;
                Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

                if( localCoefL.n() != localCoefR.n() ){
                  ErrorHandling( 
                      "The number of eigenfunctions do not match." );
                }

                Int numEig = localCoefL.n();

                DblNumVec  jump( numGridFace );
                DblNumVec  gradJump( numGridFace );

                for( Int g = 0; g < numEig; g++ ){
                  SetValue( jump, 0.0 );
                  SetValue( gradJump, 0.0 );

                  // Left side: gradjump and jump
                  if( numBasisL > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
                        drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
                        valL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  // Right side: gradjump and jump
                  if( numBasisR > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
                        drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
                        valR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  Real* ptrGJ = gradJump.Data();
                  Real* ptrJ  = jump.Data();
                  Real* ptrW  = LGLWeight2D_[0].Data();
                  Real  tmpGJ = 0.0;
                  Real  tmpJ  = 0.0;
                  for( Int p = 0; p < numGridFace; p++ ){
                    tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
                    tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
                    ptrGJ++; ptrJ++; ptrW++;
                  }
                  // Previous element
                  eta2GradJumpLocal( p1, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( p1, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
                  // Current element
                  eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
                } // for (eigenfunction)
              } // if (pF>0)
            } // x-direction


            // y-direction
            {
              // keyL is the previous element received from GetBegin/GetEnd.
              // keyR is the current element
              Int p2; if( j == 0 )  p2 = numElem_[1]-1; else   p2 = j-1;
              Index3 keyL( i, p2, k );
              Index3 keyR = key;

              Int  numGridFace = numGrid[0] * numGrid[2];

              // Note that the notation can be very confusing here:
              // The left element (keyL) contributes to the right face
              // (YR), and the right element (keyR) contributes to the
              // left face (YL)
              DblNumMat&  valL = basisJump[YR].LocalMap()[keyL];
              DblNumMat&  valR = basisJump[YL].LocalMap()[keyR];
              DblNumMat&  drvL = DbasisJump[YR].LocalMap()[keyL];
              DblNumMat&  drvR = DbasisJump[YL].LocalMap()[keyR];

              Int numBasisL = valL.n();
              Int numBasisR = valR.n();

              DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
              DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
              DblNumVec& eig         = eigVal_;
              DblNumVec& occrate     = occupationRate_;

              Real hF = std::sqrt( length[0] * length[0] + length[2] * length[2] );
              Real pF = std::max( numBasisL, numBasisR );

              // Skip the computation if there is no basis function on
              // either side.  
              if( pF > 0 ){

                // factor in the estimator for jump of the derivative and
                // the jump of the values
                // NOTE:
                // Originally in [Giani 2012] the penalty term in the a
                // posteriori error estimator should be
                //
                // facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
                // should be
                //
                // gamma * p_F^2 / h_F = penaltyAlpha_
                //
                // used in this code.  So 
                //
                // facJ = penaltyAlpha_^2 * h_F / p_F
                //

                // One 0.5 comes from double counting on the faces F,
                // and the other comes from the 1/2 in front of the
                // Laplacian operator.
                Real facGJ = 0.5 * 0.5 * hF / pF;
                Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

                if( localCoefL.n() != localCoefR.n() ){
                  ErrorHandling( 
                      "The number of eigenfunctions do not match." );
                }

                Int numEig = localCoefL.n();

                DblNumVec  jump( numGridFace );
                DblNumVec  gradJump( numGridFace );

                for( Int g = 0; g < numEig; g++ ){
                  SetValue( jump, 0.0 );
                  SetValue( gradJump, 0.0 );

                  // Left side: gradjump and jump
                  if( numBasisL > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
                        drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
                        valL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  // Right side: gradjump and jump
                  if( numBasisR > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
                        drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
                        valR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  Real* ptrGJ = gradJump.Data();
                  Real* ptrJ  = jump.Data();
                  Real* ptrW  = LGLWeight2D_[1].Data();
                  Real  tmpGJ = 0.0;
                  Real  tmpJ  = 0.0;
                  for( Int p = 0; p < numGridFace; p++ ){
                    tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
                    tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
                    ptrGJ++; ptrJ++; ptrW++;
                  }
                  // Previous element
                  eta2GradJumpLocal( i, p2, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( i, p2, k )     += tmpJ  * occrate(g) * numSpin * facJ;
                  // Current element
                  eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
                } // for (eigenfunction)
              } // if (pF>0)
            } // y-direction


            // z-direction
            {
              // keyL is the previous element received from GetBegin/GetEnd.
              // keyR is the current element
              Int p3; if( k == 0 )  p3 = numElem_[2]-1; else   p3 = k-1;
              Index3 keyL( i, j, p3 );
              Index3 keyR = key;

              Int  numGridFace = numGrid[0] * numGrid[1];

              // Note that the notation can be very confusing here:
              // The left element (keyL) contributes to the right face
              // (ZR), and the right element (keyR) contributes to the
              // left face (ZL)
              DblNumMat&  valL = basisJump[ZR].LocalMap()[keyL];
              DblNumMat&  valR = basisJump[ZL].LocalMap()[keyR];
              DblNumMat&  drvL = DbasisJump[ZR].LocalMap()[keyL];
              DblNumMat&  drvR = DbasisJump[ZL].LocalMap()[keyR];

              Int numBasisL = valL.n();
              Int numBasisR = valR.n();

              DblNumMat& localCoefL  = eigvecCoef_.LocalMap()[keyL];
              DblNumMat& localCoefR  = eigvecCoef_.LocalMap()[keyR];
              DblNumVec& eig         = eigVal_;
              DblNumVec& occrate     = occupationRate_;


              Real hF = std::sqrt( length[0] * length[0] + length[1] * length[1] );
              Real pF = std::max( numBasisL, numBasisR );

              // Skip the computation if there is no basis function on
              // either side.  
              if( pF > 0 ){

                // factor in the estimator for jump of the derivative and
                // the jump of the values
                // NOTE:
                // Originally in [Giani 2012] the penalty term in the a
                // posteriori error estimator should be
                //
                // facJ = gamma^2 * p_F^3 / h_F.  However, there the jump term
                // should be
                //
                // gamma * p_F^2 / h_F = penaltyAlpha_
                //
                // used in this code.  So 
                //
                // facJ = penaltyAlpha_^2 * h_F / p_F
                //

                // One 0.5 comes from double counting on the faces F,
                // and the other comes from the 1/2 in front of the
                // Laplacian operator.
                Real facGJ = 0.5 * 0.5 * hF / pF;
                Real facJ  = 0.5 * 0.5 * penaltyAlpha_ * penaltyAlpha_ * hF / pF;

                if( localCoefL.n() != localCoefR.n() ){
                  ErrorHandling( 
                      "The number of eigenfunctions do not match." );
                }

                Int numEig = localCoefL.n();

                DblNumVec  jump( numGridFace );
                DblNumVec  gradJump( numGridFace );

                for( Int g = 0; g < numEig; g++ ){
                  SetValue( jump, 0.0 );
                  SetValue( gradJump, 0.0 );

                  // Left side: gradjump and jump
                  if( numBasisL > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0,
                        drvL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisL, 1.0, 
                        valL.Data(), numGridFace, localCoefL.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  // Right side: gradjump and jump
                  if( numBasisR > 0 ){
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0,
                        drvR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, gradJump.Data(), 1 );
                    blas::Gemv( 'N', numGridFace, numBasisR, 1.0, 
                        valR.Data(), numGridFace, localCoefR.VecData(g), 1,
                        1.0, jump.Data(), 1 );
                  }

                  Real* ptrGJ = gradJump.Data();
                  Real* ptrJ  = jump.Data();
                  Real* ptrW  = LGLWeight2D_[2].Data();
                  Real  tmpGJ = 0.0;
                  Real  tmpJ  = 0.0;
                  for( Int p = 0; p < numGridFace; p++ ){
                    tmpGJ += (*ptrGJ) * (*ptrGJ) * (*ptrW);
                    tmpJ  += (*ptrJ)  * (*ptrJ)  * (*ptrW);
                    ptrGJ++; ptrJ++; ptrW++;
                  }
                  // Previous element
                  eta2GradJumpLocal( i, j, p3 ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( i, j, p3 )     += tmpJ  * occrate(g) * numSpin * facJ;
                  // Current element
                  eta2GradJumpLocal( i, j, k ) += tmpGJ * occrate(g) * numSpin * facGJ;
                  eta2JumpLocal( i, j, k )     += tmpJ  * occrate(g) * numSpin * facJ;
                } // for (eigenfunction)
              } // if (pF>0)
            } // z-direction


          } // if (own this element)
        } // for (i)

    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for the face and jump term is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  // *********************************************************************
  // Reduce the computed error estimator among all elements.
  // *********************************************************************


  mpi::Allreduce( eta2ResidualLocal.Data(), eta2Residual.Data(),
      numElem_.prod(), MPI_SUM, domain_.colComm );

  mpi::Allreduce( eta2GradJumpLocal.Data(), eta2GradJump.Data(),
      numElem_.prod(), MPI_SUM, domain_.colComm );

  mpi::Allreduce( eta2JumpLocal.Data(), eta2Jump.Data(),
      numElem_.prod(), MPI_SUM, domain_.colComm );

  // Compute the total estimator
  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        eta2Total(i,j,k) = 
          eta2Residual(i,j,k) +
          eta2GradJump(i,j,k) +
          eta2Jump(i,j,k);
      } // for (i)




  return ;
}         // -----  end of method HamiltonianDG::CalculateAPosterioriError  ----- 

} // namespace dgdft
