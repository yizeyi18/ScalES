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
using namespace esdf;

// *********************************************************************
// Hamiltonian class for DG
// *********************************************************************

void HamiltonianDG::Setup_device ( )
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

  statusOFS << "DBWY IN DG SETUP_device" << std::endl;


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
  atomDensity_.SetComm( domain_.colComm );
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
  atomDensity_.Prtn()   = elemPrtn_;
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
  DMat_device_.clear();
  for( Int d = 0; d < DIM; d++ ){
    DblNumVec  dummyX, dummyW;
    DblNumMat  dummyP;
    GenerateLGL( dummyX, dummyW, dummyP, DMat_[d], 
        numLGLGridElem_[d] );

    // Scale the differentiation matrix
    blas::Scal( numLGLGridElem_[d] * numLGLGridElem_[d],
        2.0 / (domain_.length[d] / numElem_[d]), 
        DMat_[d].Data(), 1 );

    // Copy data to device
    // TODO: Port GenerateLGL over to CUDA
    DMat_device_.emplace_back( DMat_[d].Size() );
    cuda::memcpy_h2d( DMat_device_.back().data(), DMat_[d].Data(),
                      DMat_[d].Size() );
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

    sizeHMat_ = numElem_[0] * numElem_[1] * numElem_[2] * esdfParam.numALBElem(0,0,0);

  return ;
}         // -----  end of method HamiltonianDG::Setup_device  ----- 



void HamiltonianDG::DiffPsi_device_slow( const Index3& numGrid, Int numBasis, 
  const Real* psi_device, Real* Dpsi_device, Int d ) {

  statusOFS << "DBWY DiffPsi_device slow" << std::endl;

  const Int nx = numGrid[0];
  const Int ny = numGrid[1];
  const Int nz = numGrid[2];

  const Int ntot = nx * ny * nz;

  // x derivative
  if( d == 0 ) {

    for( Int g = 0; g < numBasis; g++ ) {
      auto* psi_ptr    = psi_device  + g * ntot;
      auto* dpsi_x_ptr = Dpsi_device + g * ntot;
      cublas::blas::gemm( handle, 'N', 'N', 
        nx, ny * nz, nx,
        1., DMat_device_[0].data(), nx, psi_ptr, nx,
        0., dpsi_x_ptr, nx
      );
    }

  // y derivative
  } else if( d == 1 ) {

    const auto m = nx;
    const auto n = ny;
    for( Int kg = 0; kg < numBasis * nz; ++kg ) {
      auto* psi_ptr    = psi_device  + kg * m * n;
      auto* dpsi_y_ptr = Dpsi_device + kg * m * n;
      cublas::blas::gemm( handle, 'N', 'T', m, n, n,
        1., psi_ptr, m, DMat_device_[1].data(), n, 
        0., dpsi_y_ptr, m 
      );
    }

  // z derivative
  } else if( d == 2 ) {

    const auto m = nx * ny;
    const auto n = nz;
    for( Int g = 0; g < numBasis; g++ ) {
      auto* psi_ptr    = psi_device  + g * ntot;
      auto* dpsi_z_ptr = Dpsi_device + g * ntot;

      cublas::blas::gemm( handle, 'N', 'T', m, n, n,
        1., psi_ptr, m, DMat_device_[2].data(), n, 
        0., dpsi_z_ptr, m
      );
    }

  } else {

    ErrorHandling("Wrong dimension.");

  }


}         // -----  end of method HamiltonianDG::DiffPsi_device_slow  ----- 

void HamiltonianDG::DiffPsi_device_fast( const Index3& numGrid, Int numBasis, 
  Real* psi_device, Real* Dpsi_device, Int d ) {

  statusOFS << "DBWY DiffPsi_device fast" << std::endl;

  const Int nx = numGrid[0];
  const Int ny = numGrid[1];
  const Int nz = numGrid[2];

  const Int ntot = nx * ny * nz;

  // x derivative
  if( d == 0 ) {

    auto m = nx;
    auto n = ny * nz * numBasis;

    cublas::blas::gemm( handle, 'N', 'N', m, n, m,
      1., DMat_device_[0].data(), m, psi_device, m,
      0., Dpsi_device, m
    );

  // y derivative
  } else if( d == 1 ) {

    auto m = nx;
    auto n = ny;
    auto gemm_stride = nx * ny;
    auto gemm_nbatch = nz * numBasis;

    cublas::blas::gemm_batched_strided( handle, 'N', 'T', m, n, n,
      1., psi_device, m, gemm_stride, DMat_device_[1].data(), n, 0,
      0., Dpsi_device, m, gemm_stride,
      gemm_nbatch
    );

  // z derivative
  } else if( d == 2 ) {

    auto m = nx * ny;
    auto n = nz;
    auto gemm_stride = ntot;
    auto gemm_nbatch = numBasis;

    cublas::blas::gemm_batched_strided( handle, 'N', 'T', m, n, n,
      1., psi_device, m, gemm_stride, DMat_device_[2].data(), n, 0,
      0., Dpsi_device, m, gemm_stride,
      gemm_nbatch
    );

  } else {

    ErrorHandling("Wrong dimension.");

  }


}         // -----  end of method HamiltonianDG::DiffPsi_device_fast  ----- 

} // namespace dgdft
