/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu, Lexing Ying and David Williams-Young

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
/// @file hamiltonian_dg_matrix.cpp
/// @brief Implementation of the Hamiltonian class for DG calculation.
/// @date 2013-01-09
/// @date 2014-08-07 Add intra-element paralelization.  NOTE: the OpenMP
/// parallelization is not used in this version.
#include  "hamiltonian_dg.hpp"
#include  "mpi_interf.hpp"
#include  "blas.hpp"


// FIXME Whether to include the non-local part
#define _NON_LOCAL_  1

// FIXME Check all numBasis == 0 and see whether it should be numBasisTotal = 0

namespace dgdft{

using namespace PseudoComponent;


// *********************************************************************
// Hamiltonian class for constructing the DG matrix
// *********************************************************************

void
  HamiltonianDG::CalculateDGMatrix_device    (  )
  {
    statusOFS << "DBWY IN CalculateDGMatrix_device" << std::endl;
    Int mpirank, mpisize;
    Int numAtom = atomList_.size();
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    MPI_Barrier(domain_.rowComm);
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

    MPI_Barrier(domain_.colComm);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    Real timeSta, timeEnd;

    // Here numGrid is the LGL grid
    Point3 length       = domainElem_(0,0,0).length;
    Index3 numGrid      = numLGLGridElem_;             
    Int    numGridTotal = numGrid.prod();

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
    std::vector<DistDblNumMat>   DbasisAverage(NUM_FACE);

    std::vector<DistVec<Index3, cuda::device_vector<Real>, ElemPrtn> >
      basisJump_device( NUM_FACE ),
      DbasisAverage_device( NUM_FACE );

    // IMPORTANT:
    //
    // When intra-element parallelization is invoked, the communication of
    // boundary terms are refined to each column processor communicator.
    for( Int k = 0; k < NUM_FACE; k++ ) {
      basisJump[k].SetComm(domain_.colComm);
      DbasisAverage[k].SetComm(domain_.colComm);

      basisJump_device[k].SetComm(domain_.colComm);
      DbasisAverage_device[k].SetComm(domain_.colComm);
    } 

    // The derivative of basisLGL along x,y,z directions
    std::vector<DistDblNumMat>   Dbasis(DIM);
    std::vector<DistVec<Index3, cuda::device_vector<Real>, ElemPrtn> >
      Dbasis_device(DIM);

    // Same as above
    for( Int k = 0; k < DIM; k++ ) {
      Dbasis[k].SetComm(domain_.colComm);
      Dbasis_device[k].SetComm(domain_.colComm);
    } 

    // Integration weights
    std::vector<DblNumVec>&  LGLWeight1D = LGLWeight1D_;
    std::vector<DblNumMat>&  LGLWeight2D = LGLWeight2D_;
    DblNumTns&               LGLWeight3D = LGLWeight3D_;

    // Square root of the LGL integration weight in 3D.
    DblNumTns    sqrtLGLWeight3D( numGrid[0], numGrid[1], numGrid[2] );

    // Square root of the LGL integration weight in 2D for each face.
    std::vector<DblNumMat>   sqrtLGLWeight2D;
    sqrtLGLWeight2D.resize(DIM);

    sqrtLGLWeight2D[0].Resize( numGrid[1], numGrid[2]);
    sqrtLGLWeight2D[1].Resize( numGrid[0], numGrid[2]);
    sqrtLGLWeight2D[2].Resize( numGrid[0], numGrid[1]);

    Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
    for( Int i = 0; i < numGrid.prod(); i++ ){
      *(ptr2++) = std::sqrt( *(ptr1++) );
    }

    // yz face
    for( Int k = 0; k < numGrid[2]; k++ )
      for( Int j = 0; j < numGrid[1]; j++ ){
        sqrtLGLWeight2D[0](j, k) = std::sqrt( LGLWeight2D[0](j, k) );
      } // for (j)

    // xz face
    for( Int k = 0; k < numGrid[2]; k++ )
      for( Int i = 0; i < numGrid[0]; i++ ){
        sqrtLGLWeight2D[1](i, k) = std::sqrt( LGLWeight2D[1](i, k) );
      } // for (i)

    // xy face
    for( Int j = 0; j < numGrid[1]; j++ )
      for( Int i = 0; i < numGrid[0]; i++ ){
        sqrtLGLWeight2D[2](i, j) = std::sqrt( LGLWeight2D[2](i, j) );
      }

    // Clear the DG Matrix
    HMat_.LocalMap().clear();

    // *********************************************************************
    // Initial setup
    // *********************************************************************

    {
      GetTime(timeSta);
      // Compute the indices for all basis functions
      IntNumTns  numBasisLocal(numElem_[0], numElem_[1], numElem_[2]);
      SetValue( numBasisLocal, 0 );
      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key(i, j, k);
            if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
              int numBasisLocalElem = basisLGL_.LocalMap()[key].n();
              numBasisLocal(i, j, k) = 0;
              mpi::Allreduce( &numBasisLocalElem, &numBasisLocal(i, j, k), 1, MPI_SUM, domain_.rowComm );
            }
          } // for (i)

      IntNumTns numBasis(numElem_[0], numElem_[1], numElem_[2]);
      mpi::Allreduce( numBasisLocal.Data(), numBasis.Data(),
          numElem_.prod(), MPI_SUM, domain_.colComm );

      // Every processor computes all index sets and its inverse mapping
      elemBasisIdx_.Resize(numElem_[0], numElem_[1], numElem_[2]);

      {
        Int cnt = 0;
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              std::vector<Int> idxVec;
              for(Int g = 0; g < numBasis(i, j, k); g++){
                idxVec.push_back( cnt++ );
              }
              elemBasisIdx_(i, j, k) = idxVec;
            } // for (i)

        sizeHMat_ = cnt;
      }

      elemBasisInvIdx_.resize(sizeHMat_);

      {
        Int cnt = 0;
        for( Int k = 0; k < numElem_[2]; k++ )
          for( Int j = 0; j < numElem_[1]; j++ )
            for( Int i = 0; i < numElem_[0]; i++ ){
              Index3 key = Index3(i,j,k);
              for(Int g = 0; g < numBasis(i, j, k); g++){
                elemBasisInvIdx_[cnt++] = key;
              }
            } // for (i)
      }





      for( Int i = 0; i < NUM_FACE; i++ ){
        basisJump[i].Prtn()     = elemPrtn_;
        DbasisAverage[i].Prtn() = elemPrtn_;

        basisJump_device[i].Prtn()     = elemPrtn_;
        DbasisAverage_device[i].Prtn() = elemPrtn_;
      }

      for( Int i = 0; i < DIM; i++ ){
        Dbasis[i].Prtn()        = elemPrtn_;
        Dbasis_device[i].Prtn() = elemPrtn_;
      }

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for initial setup is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }


    // *********************************************************************
    // Copy basis to the device
    // TODO: Construct basisLGL on device
    // *********************************************************************
    DistVec<Index3, cuda::device_vector<Real>, ElemPrtn> basisLGL_device;
    basisLGL_device.SetComm( basisLGL_.Comm() );
    basisLGL_device.Prtn() = basisLGL_.Prtn();
    for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
    for( Int i = 0; i < numElem_[0]; i++ ) {
      Index3 key( i, j, k );
      if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ) {
        auto& basis        = basisLGL_.LocalMap()[key];
        auto& basis_device = basisLGL_device.LocalMap()[key];

        basis_device.resize( basis.Size() );
        cuda::memcpy_h2d( basis_device.data(), basis.Data(), basis.Size() );
      }
    }

    // XXX: Device variables
    auto cublas_stream = handle.get_stream();



    // *********************************************************************
    // Compute the local derivatives
    // Start the communication of the derivatives
    // *********************************************************************
    {
      GetTime(timeSta);

      // Allocate DBasis (XXX: to factor out timings)
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          auto& basis        = basisLGL_.LocalMap()[key];

          // Device
          Dbasis_device[0].LocalMap()[key].resize( basis.Size() );
          Dbasis_device[1].LocalMap()[key].resize( basis.Size() );
          Dbasis_device[2].LocalMap()[key].resize( basis.Size() );
          
          // Host
          Dbasis[0].LocalMap()[key].Resize( basis.m(), basis.n() );
          Dbasis[1].LocalMap()[key].Resize( basis.m(), basis.n() );
          Dbasis[2].LocalMap()[key].Resize( basis.m(), basis.n() );

        } // own this element
      } // for (ijk)
      
      cuda::event diff_psi_st, diff_psi_en;
      diff_psi_st.record( cublas_stream );

      // Compute derivatives on each local element ( XXX: DEVICE )
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          auto& basis        = basisLGL_.LocalMap()[key];
          auto& basis_device = basisLGL_device.LocalMap()[key];
          Int numBasis = basis.n();

          auto& DbasisX_device = Dbasis_device[0].LocalMap()[key];
          auto& DbasisY_device = Dbasis_device[1].LocalMap()[key];
          auto& DbasisZ_device = Dbasis_device[2].LocalMap()[key];

          // Compute derivatives on device using batched formulation
          DiffPsi_device_fast( numGrid, numBasis, basis_device.data(),
                               DbasisX_device.data(), 0 );
          DiffPsi_device_fast( numGrid, numBasis, basis_device.data(),
                               DbasisY_device.data(), 1 );
          DiffPsi_device_fast( numGrid, numBasis, basis_device.data(),
                               DbasisZ_device.data(), 2 );

        } // own this element
      } // for (ijk)

      diff_psi_en.record( cublas_stream );
      cublas_stream.synchronize();
      auto diff_psi_dur = cuda::event::elapsed_time( diff_psi_st, diff_psi_en );

      Real diff_psi_st_host, diff_psi_en_host;












      GetTime( diff_psi_st_host );
      // Compute derivatives on each local element ( XXX: HOST )
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          auto& basis        = basisLGL_.LocalMap()[key];
          Int numBasis = basis.n();

          DblNumMat& DbasisX = Dbasis[0].LocalMap()[key];
          DblNumMat& DbasisY = Dbasis[1].LocalMap()[key];
          DblNumMat& DbasisZ = Dbasis[2].LocalMap()[key];

          // Compact implementation with the same efficiency
          for( Int g = 0; g < numBasis; g++ ){
            DiffPsi( numGrid, basis.VecData(g), DbasisX.VecData(g), 0 );
            DiffPsi( numGrid, basis.VecData(g), DbasisY.VecData(g), 1 );
            DiffPsi( numGrid, basis.VecData(g), DbasisZ.VecData(g), 2 );
          }

        } // own this element
      } // for (ijk)
      GetTime( diff_psi_en_host );

      # if 0
      // Compare host and device calculations
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          auto& basis        = basisLGL_.LocalMap()[key];
          Int numBasis = basis.n();

          // Host evaluation
          auto& DbasisX = Dbasis[0].LocalMap()[key];
          auto& DbasisY = Dbasis[1].LocalMap()[key];
          auto& DbasisZ = Dbasis[2].LocalMap()[key];

          // Device evaluation
          auto& DbasisX_device = Dbasis_device[0].LocalMap()[key];
          auto& DbasisY_device = Dbasis_device[1].LocalMap()[key];
          auto& DbasisZ_device = Dbasis_device[2].LocalMap()[key];

          // Communucate device values to host
          cuda::pinned_vector< double > DbasisX_test( DbasisX.Size() );
          cuda::pinned_vector< double > DbasisY_test( DbasisY.Size() );
          cuda::pinned_vector< double > DbasisZ_test( DbasisZ.Size() );
          cuda::copy( DbasisX_device, DbasisX_test );
          cuda::copy( DbasisY_device, DbasisY_test );
          cuda::copy( DbasisZ_device, DbasisZ_test );
          cuda::wrappers::device_sync();

          statusOFS << "DBWY MAX XVAL = " << 
            *std::max_element( DbasisX_test.begin(), DbasisX_test.end() ) 
          << std::endl;
          statusOFS << "DBWY MAX YVAL = " << 
            *std::max_element( DbasisY_test.begin(), DbasisY_test.end() ) 
          << std::endl;
          statusOFS << "DBWY MAX ZVAL = " << 
            *std::max_element( DbasisZ_test.begin(), DbasisZ_test.end() ) 
          << std::endl;

          for( auto ii = 0; ii < DbasisX.Size(); ++ii )
            DbasisX_test[ii] = std::abs( DbasisX_test[ii] - DbasisX.Data()[ii] );
          for( auto ii = 0; ii < DbasisY.Size(); ++ii )
            DbasisY_test[ii] = std::abs( DbasisY_test[ii] - DbasisY.Data()[ii] );
          for( auto ii = 0; ii < DbasisZ.Size(); ++ii )
            DbasisZ_test[ii] = std::abs( DbasisZ_test[ii] - DbasisZ.Data()[ii] );

          statusOFS << "DBWY MAX XDIFF = " << 
            *std::max_element( DbasisX_test.begin(), DbasisX_test.end() ) 
          << std::endl;
          statusOFS << "DBWY MAX YDIFF = " << 
            *std::max_element( DbasisY_test.begin(), DbasisY_test.end() ) 
          << std::endl;
          statusOFS << "DBWY MAX ZDIFF = " << 
            *std::max_element( DbasisZ_test.begin(), DbasisZ_test.end() ) 
          << std::endl;


        } // own this element
      } // for (ijk)
      #endif


      statusOFS << "Time for the local gradient calculation (Host) is " <<
        (diff_psi_en_host - diff_psi_st_host)*1000 << " [ms]" << std::endl;
      statusOFS << "Time for the local gradient calculation (Device) is " <<
        diff_psi_dur << " [ms]" << std::endl << std::endl;

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for the local gradient calculation is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime(timeSta);


      // Preallocate Device
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          auto&  basis        = basisLGL_.LocalMap()[key];
          Int numBasis = basis.n();

          const Int NX = numGrid[0];
          const Int NY = numGrid[1];
          const Int NZ = numGrid[2];

          // x-direction
          {
            const Int  numGridFace = NY * NZ;

            // Device alloc
            basisJump_device[XL].LocalMap()[key].resize( numGridFace * numBasis );
            basisJump_device[XR].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[XL].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[XR].LocalMap()[key].resize( numGridFace * numBasis );
          } // x-direction


          // y-direction
          {
            const Int  numGridFace = NX * NZ;

            // Device alloc
            basisJump_device[YL].LocalMap()[key].resize( numGridFace * numBasis );
            basisJump_device[YR].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[YL].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[YR].LocalMap()[key].resize( numGridFace * numBasis );
          } // y-direction

          // z-direction
          {
            const Int  numGridFace = NX * NY;

            // Device alloc
            basisJump_device[ZL].LocalMap()[key].resize( numGridFace * numBasis );
            basisJump_device[ZR].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[ZL].LocalMap()[key].resize( numGridFace * numBasis );
            DbasisAverage_device[ZR].LocalMap()[key].resize( numGridFace * numBasis );
          } // z-direction

        } // I own
      } // for (ijk)



      cuda::event jump_st, jump_en;
      jump_st.record();

      // Compute average of derivatives and jump of values (XXX: Device)
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          auto&  basis        = basisLGL_.LocalMap()[key];
          auto&  basis_device = basisLGL_device.LocalMap()[key];
          Int numBasis = basis.n();

          const Int NX = numGrid[0];
          const Int NY = numGrid[1];
          const Int NZ = numGrid[2];

          // x-direction
          {
            const Int  numGridFace = NY * NZ;

            auto&  valL_device = basisJump_device[XL].LocalMap()[key];
            auto&  valR_device = basisJump_device[XR].LocalMap()[key];
            auto&  drvL_device = DbasisAverage_device[XL].LocalMap()[key];
            auto&  drvR_device = DbasisAverage_device[XR].LocalMap()[key];
            auto&  DbasisX_device = Dbasis_device[0].LocalMap()[key];

            // Form jumps and averages from volume to face.
            //   - jkg in [0, NYZ * NBF)
            //   - ioff = (NX-1)

            // valL <- -basis( 0, :, :, : )
            //   valL[ jkg ] = -basis[ jkg * NX ]
            cuda::memcpy2d_d2d( valL_device.data(), 1, 
                              basis_device.data(), NX,
                              1, NY * NZ * numBasis );
            cublas::blas::scal( handle, NY * NZ * numBasis, -1., 
              valL_device.data(), 1 );

            // valR <- basis( nx-1, :, :, : )
            //   valR[ jkg ] = basis[ ioff + jkg * NX ]
            cuda::memcpy2d_d2d( valR_device.data(), 1, 
                              basis_device.data() + (NX-1), NX,
                              1, NY * NZ * numBasis );

            // drvL <- 0.5 * DbasisX( 0, :, :, : )
            //   drvL[ jkg ] = 0.5 * DbasisX[ jkg * NX ]
            cuda::memcpy2d_d2d( drvL_device.data(), 1, 
                              DbasisX_device.data(), NX,
                              1, NY * NZ * numBasis );
            cublas::blas::scal( handle, NY * NZ * numBasis, 0.5,
              drvL_device.data(), 1 );

            // drvR <- 0.5 * DbasisX( nx-1, :, :, : )
            //   drvR[ jkg ] = 0.5 * DbasisX[ ioff + jkg * NX ]
            cuda::memcpy2d_d2d( drvR_device.data(), 1, 
                              DbasisX_device.data() + (NX-1), NX,
                              1, NY * NZ * numBasis );
            cublas::blas::scal( handle, NY * NZ * numBasis, 0.5,
              drvR_device.data(), 1 );

          } // x-direction


          // y-direction
          {
            const Int  numGridFace = NX * NZ;

            auto&  valL_device = basisJump_device[YL].LocalMap()[key];
            auto&  valR_device = basisJump_device[YR].LocalMap()[key];
            auto&  drvL_device = DbasisAverage_device[YL].LocalMap()[key];
            auto&  drvR_device = DbasisAverage_device[YR].LocalMap()[key];
            auto&  DbasisY_device = Dbasis_device[1].LocalMap()[key];

            // Form jumps and averages from volume to face.
            //   - i  in [0, NX)
            //   - kg in [0, NZ * NBF)
            //   - ioff = (NY-1) * NX

            // valL <- -basis( :, 0, :, : )
            //   valL[ i + kg * NX ] = -basis[ i + kg * NXY ]
            cuda::memcpy2d_d2d( valL_device.data(), NX, 
                              basis_device.data(), NX * NY,
                              NX, NZ * numBasis );
            cublas::blas::scal( handle, NX * NZ * numBasis, -1., 
              valL_device.data(), 1 );

            // valR <- basis( :, ny-1, :, : )
            //   valR[ i + kg * NX ] = basis[ ioff + i + kg * NXY ]
            cuda::memcpy2d_d2d( valR_device.data(), NX, 
                              basis_device.data() + (NY-1)*NX, NX * NY,
                              NX, NZ * numBasis );

            // drvL <- 0.5 * DbasisY( :, 0, :, : )
            //   drvL[ i + kg * NX ] = 0.5 * DbasisY[ i + kg * NXY ]
            cuda::memcpy2d_d2d( drvL_device.data(), NX, 
                              DbasisY_device.data(), NX * NY,
                              NX, NZ * numBasis );
            cublas::blas::scal( handle, NX * NZ * numBasis, 0.5,
              drvL_device.data(), 1 );

            // drvR <- 0.5 * DbasisY( :, ny-1, :, : )
            //   drvR[ i + kg * NX ] = 0.5 * DbasisY[ ioff + i + kg * NXY ]
            cuda::memcpy2d_d2d( drvR_device.data(), NX, 
                              DbasisY_device.data() + (NY-1)*NX, NX * NY,
                              NX, NZ * numBasis );
            cublas::blas::scal( handle, NX * NZ * numBasis, 0.5,
              drvR_device.data(), 1 );


          } // y-direction

          // z-direction
          {
            const Int  numGridFace = NX * NY;

            auto&  valL_device = basisJump_device[ZL].LocalMap()[key];
            auto&  valR_device = basisJump_device[ZR].LocalMap()[key];
            auto&  drvL_device = DbasisAverage_device[ZL].LocalMap()[key];
            auto&  drvR_device = DbasisAverage_device[ZR].LocalMap()[key];
            auto&  DbasisZ_device = Dbasis_device[2].LocalMap()[key];

            // Form jumps and averages from volume to face.
            //   - ij in [0, NXY)
            //   - g  in [0, NBF)
            //   - ioff = (NZ-1) * NXY

            // valL <- -basis( :, :, 0, : )
            //   valL[ ij + g * NXY ] = -basis[ ij + g * NXYZ ]
            cuda::memcpy2d_d2d( valL_device.data(), NX * NY, 
                              basis_device.data(), NX * NY * NZ,
                              NX * NY, numBasis );
            cublas::blas::scal( handle, NX * NY * numBasis, -1., 
              valL_device.data(), 1 );

            // valR <- basis( :, :, nz-1, : )
            //   valR[ ij + g * NXY ] = basis[ ioff + ij + g * NXYZ ]
            cuda::memcpy2d_d2d( valR_device.data(), NX * NY, 
                              basis_device.data() + (NZ-1)*NX*NY, NX * NY * NZ,
                              NX * NY, numBasis );

            // drvL <- 0.5 * DbasisZ( :, :, 0, : )
            //   drvL[ ij + g * NXY ] = 0.5 * DbasisZ[ ij + g * NXYZ ]
            cuda::memcpy2d_d2d( drvL_device.data(), NX * NY, 
                              DbasisZ_device.data(), NX * NY * NZ,
                              NX * NY, numBasis );
            cublas::blas::scal( handle, NX * NY * numBasis, 0.5,
              drvL_device.data(), 1 );

            // drvR <- 0.5 * DbasisZ( :, :, nz-1, : )
            //   drvR[ ij + g * NXY ] = 0.5 * DbasisZ[ ioff + ij + g * NXYZ ]
            cuda::memcpy2d_d2d( drvR_device.data(), NX * NY, 
                              DbasisZ_device.data() + (NZ-1)*NX*NY, NX * NY * NZ,
                              NX * NY, numBasis );
            cublas::blas::scal( handle, NX * NY * numBasis, 0.5,
              drvR_device.data(), 1 );

          } // z-direction

        } // I own
      } // for (ijk)
      jump_en.record();
      jump_en.synchronize();
      auto jump_dur_device = cuda::event::elapsed_time( jump_st, jump_en );


      // Preallocate Host 
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          auto&  basis        = basisLGL_.LocalMap()[key];
          Int numBasis = basis.n();

          const Int NX = numGrid[0];
          const Int NY = numGrid[1];
          const Int NZ = numGrid[2];

          // x-direction
          {
            const Int  numGridFace = NY * NZ;

            // Host alloc
            DblNumMat emptyX( numGridFace, numBasis );
            SetValue( emptyX, 0.0 );
            basisJump[XL].LocalMap()[key] = emptyX;
            basisJump[XR].LocalMap()[key] = emptyX;
            DbasisAverage[XL].LocalMap()[key] = emptyX;
            DbasisAverage[XR].LocalMap()[key] = emptyX;
          } // x-direction


          // y-direction
          {
            const Int  numGridFace = NX * NZ;

            // Host alloc
            DblNumMat emptyY( numGridFace, numBasis );
            SetValue( emptyY, 0.0 );
            basisJump[YL].LocalMap()[key] = emptyY;
            basisJump[YR].LocalMap()[key] = emptyY;
            DbasisAverage[YL].LocalMap()[key] = emptyY;
            DbasisAverage[YR].LocalMap()[key] = emptyY;
          } // y-direction

          // z-direction
          {
            const Int  numGridFace = NX * NY;

            // Host alloc
            DblNumMat emptyZ( numGridFace, numBasis );
            SetValue( emptyZ, 0.0 );
            basisJump[ZL].LocalMap()[key] = emptyZ;
            basisJump[ZR].LocalMap()[key] = emptyZ;
            DbasisAverage[ZL].LocalMap()[key] = emptyZ;
            DbasisAverage[ZR].LocalMap()[key] = emptyZ;
          } // z-direction

        }
      } // for (ijk)


      Real jump_st_host, jump_en_host;

      // Compute average of derivatives and jump of values ( XXX: Host )
      GetTime( jump_st_host );
      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
          auto&  basis        = basisLGL_.LocalMap()[key];
          Int numBasis = basis.n();

          const Int NX = numGrid[0];
          const Int NY = numGrid[1];
          const Int NZ = numGrid[2];

          // x-direction
          {
            const Int  numGridFace = NY * NZ;

            DblNumMat&  valL = basisJump[XL].LocalMap()[key];
            DblNumMat&  valR = basisJump[XR].LocalMap()[key];
            DblNumMat&  drvL = DbasisAverage[XL].LocalMap()[key];
            DblNumMat&  drvR = DbasisAverage[XR].LocalMap()[key];
            DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];


            // Form jumps and averages from volume to face.
            // basis(0,:,:)             -> valL
            // basis(NX-1,:,:)  -> valR
            // Dbasis(0,:,:)            -> drvL
            // Dbasis(NX-1,:,:) -> drvR
            for( Int g = 0; g < numBasis; g++ ){
              Int idx, idxL, idxR;
              for( Int gk = 0; gk < NZ; gk++ )
                for( Int gj = 0; gj < NY; gj++ ){
                  idx  = gj + gk*NY;
                  idxL = 0 + gj*NX + gk * (NX *
                      NY);
                  idxR = (NX-1) + gj*NX + gk * (NX *
                      NY);

                  // 0.5 comes from average
                  // {{a}} = 1/2 (a_L + a_R)
                  drvL(idx, g) = +0.5 * DbasisX( idxL, g );
                  drvR(idx, g) = +0.5 * DbasisX( idxR, g );
                  // 1.0, -1.0 comes from jump with different normal vectors
                  // [[a]] = -(1.0) a_L + (1.0) a_R
                  valL(idx, g) = -1.0 * basis( idxL, g );
                  valR(idx, g) = +1.0 * basis( idxR, g );
                } // for (gj)
            } // for (g)

          } // x-direction


          // y-direction
          {
            const Int  numGridFace = NX * NZ;

            DblNumMat&  valL = basisJump[YL].LocalMap()[key];
            DblNumMat&  valR = basisJump[YR].LocalMap()[key];
            DblNumMat&  drvL = DbasisAverage[YL].LocalMap()[key];
            DblNumMat&  drvR = DbasisAverage[YR].LocalMap()[key];
            DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];

            // Form jumps and averages from volume to face.
            // basis(0,:,:)             -> valL
            // basis(NX-1,:,:)  -> valR
            // Dbasis(0,:,:)            -> drvL
            // Dbasis(NX-1,:,:) -> drvR
            for( Int g = 0; g < numBasis; g++ ){
              Int idx, idxL, idxR;
              for( Int gk = 0; gk < NZ; gk++ )
                for( Int gi = 0; gi < NX; gi++ ){
                  idx  = gi + gk*NX;
                  idxL = gi + 0 *NX +
                    gk * (NX * NY);
                  idxR = gi + (NY-1)*NX + 
                    gk * (NX * NY);

                  // 0.5 comes from average
                  // {{a}} = 1/2 (a_L + a_R)
                  drvL(idx, g) = +0.5 * DbasisY( idxL, g );
                  drvR(idx, g) = +0.5 * DbasisY( idxR, g );
                  // 1.0, -1.0 comes from jump with different normal vectors
                  // [[a]] = -(1.0) a_L + (1.0) a_R
                  valL(idx, g) = -1.0 * basis( idxL, g );
                  valR(idx, g) = +1.0 * basis( idxR, g );
                } // for (gj)
            } // for (g)

          } // y-direction

          // z-direction
          {
            const Int  numGridFace = NX * NY;

            DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
            DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
            DblNumMat&  drvL = DbasisAverage[ZL].LocalMap()[key];
            DblNumMat&  drvR = DbasisAverage[ZR].LocalMap()[key];
            DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

            // Form jumps and averages from volume to face.
            // basis(0,:,:)             -> valL
            // basis(NX-1,:,:)  -> valR
            // Dbasis(0,:,:)            -> drvL
            // Dbasis(NX-1,:,:) -> drvR
            for( Int g = 0; g < numBasis; g++ ){
              Int idx, idxL, idxR;
              for( Int gj = 0; gj < NY; gj++ )
                for( Int gi = 0; gi < NX; gi++ ){
                  idx  = gi + gj*NX;
                  idxL = gi + gj*NX +
                    0 * (NX * NY);
                  idxR = gi + gj*NX +
                    (NZ-1) * (NX * NY);

                  // 0.5 comes from average
                  // {{a}} = 1/2 (a_L + a_R)
                  drvL(idx, g) = +0.5 * DbasisZ( idxL, g );
                  drvR(idx, g) = +0.5 * DbasisZ( idxR, g );
                  // 1.0, -1.0 comes from jump with different normal vectors
                  // [[a]] = -(1.0) a_L + (1.0) a_R
                  valL(idx, g) = -1.0 * basis( idxL, g );
                  valR(idx, g) = +1.0 * basis( idxR, g );
                } // for (gj)
            } // for (g)

          } // z-direction

        }
      } // for (ijk)
      GetTime( jump_en_host);
      auto jump_dur_host = (jump_en_host - jump_st_host) * 1000;



      // Compare average of derivatives and jump of values (XXX: Check)
      auto face_check = [&]( auto& key, char dir, int LF, int RF ) {

        auto&  valL = basisJump[LF].LocalMap()[key];
        auto&  valR = basisJump[RF].LocalMap()[key];
        auto&  drvL = DbasisAverage[LF].LocalMap()[key];
        auto&  drvR = DbasisAverage[RF].LocalMap()[key];

        auto&  valL_device = basisJump_device[LF].LocalMap()[key];
        auto&  valR_device = basisJump_device[RF].LocalMap()[key];
        auto&  drvL_device = DbasisAverage_device[LF].LocalMap()[key];
        auto&  drvR_device = DbasisAverage_device[RF].LocalMap()[key];

        // Test correctness
        cuda::pinned_vector<double> 
          drvL_test( valL_device.size() ),
          drvR_test( valL_device.size() ),
          valL_test( valL_device.size() ),
          valR_test( valL_device.size() );

        cuda::memcpy_d2h( valL_test.data(), valL_device.data(),
                          valL_device.size() );
        cuda::memcpy_d2h( valR_test.data(), valR_device.data(),
                          valR_device.size() );
        cuda::memcpy_d2h( drvL_test.data(), drvL_device.data(),
                          drvL_device.size() );
        cuda::memcpy_d2h( drvR_test.data(), drvR_device.data(),
                          drvR_device.size() );


        statusOFS << "DBWY " << dir << "_VAL_L_MAX = " <<
          *std::max_element( valL_test.begin(), valL_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_VAL_R_MAX = " <<
          *std::max_element( valR_test.begin(), valR_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_DRV_L_MAX = " <<
          *std::max_element( drvL_test.begin(), drvL_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_DRV_R_MAX = " <<
          *std::max_element( drvR_test.begin(), drvR_test.end() )
        << std::endl;


        for( auto g = 0; g < valL_device.size(); ++g ) {
          valL_test[g] = std::abs( valL_test[g] - valL.Data()[g] );
          valR_test[g] = std::abs( valR_test[g] - valR.Data()[g] );
          drvL_test[g] = std::abs( drvL_test[g] - drvL.Data()[g] );
          drvR_test[g] = std::abs( drvR_test[g] - drvR.Data()[g] );
        }

        statusOFS << "DBWY " << dir << "_VAL_L_DIFF = " <<
          *std::max_element( valL_test.begin(), valL_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_VAL_R_DIFF = " <<
          *std::max_element( valR_test.begin(), valR_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_DRV_L_DIFF = " <<
          *std::max_element( drvL_test.begin(), drvL_test.end() )
        << std::endl;
        statusOFS << "DBWY " << dir << "_DRV_R_DIFF = " <<
          *std::max_element( drvR_test.begin(), drvR_test.end() )
        << std::endl;
      };

      for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

          face_check( key, 'X', XL, XR ); statusOFS << std::endl;
          face_check( key, 'Y', YL, YR ); statusOFS << std::endl;
          face_check( key, 'Z', ZL, ZR );

        } // I own
      } // for (ijk)

      statusOFS << "Time for constructing the boundary terms (host) is " <<
        jump_dur_host << " [ms]" << std::endl;
      statusOFS << "Time for constructing the boundary terms (device) is " <<
        jump_dur_device << " [ms]" << std::endl << std::endl;

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for constructing the boundary terms is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // *********************************************************************
    // Boundary terms, Part I
    // Start the communication of the boundary terms
    //
    // When intra-element parallelization is invoked, the communication of
    // boundary terms are refined to each column processor communicator.
    // *********************************************************************
    {
      GetTime(timeSta);
      // Set of element indicies for boundary term communication.
      std::set<Index3>   boundaryXset;
      std::set<Index3>   boundaryYset;
      std::set<Index3>   boundaryZset;

      // Vector of element indicies for boundary term communication.
      // Should contain the same information as the sets above.
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
      // This is due to the use of periodic boundary condition.

      boundaryXIdx.insert( boundaryXIdx.begin(), boundaryXset.begin(), boundaryXset.end() );
      boundaryYIdx.insert( boundaryYIdx.begin(), boundaryYset.begin(), boundaryYset.end() );
      boundaryZIdx.insert( boundaryZIdx.begin(), boundaryZset.begin(), boundaryZset.end() );

      // Host Communication
      DbasisAverage[XR].GetBegin( boundaryXIdx, NO_MASK );
      DbasisAverage[YR].GetBegin( boundaryYIdx, NO_MASK );
      DbasisAverage[ZR].GetBegin( boundaryZIdx, NO_MASK );

      basisJump[XR].GetBegin( boundaryXIdx, NO_MASK );
      basisJump[YR].GetBegin( boundaryYIdx, NO_MASK );
      basisJump[ZR].GetBegin( boundaryZIdx, NO_MASK );

      // Device Communication
      // XXX: This involves copy to the host, should avoid if possible through
      //      CUDA aware MPI
      DbasisAverage_device[XR].GetBegin( boundaryXIdx, NO_MASK );
      DbasisAverage_device[YR].GetBegin( boundaryYIdx, NO_MASK );
      DbasisAverage_device[ZR].GetBegin( boundaryZIdx, NO_MASK );

      basisJump_device[XR].GetBegin( boundaryXIdx, NO_MASK );
      basisJump_device[YR].GetBegin( boundaryYIdx, NO_MASK );
      basisJump_device[ZR].GetBegin( boundaryZIdx, NO_MASK );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "After the GetBegin part of communication." << std::endl;
      statusOFS << "Time for GetBegin is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }

    // *********************************************************************
    // Nonlocal pseudopotential term, Part I
    // Compute the coefficients for the nonlocal pseudopotential 
    // Start the communiation of the nonlocal pseudopotential terms
    //
    // When intra-element parallelization is invoked, the inner product of
    // basis functions and the nonlocal potential projectors are first
    // done independently on each processors, and then all processors in
    // the same processor row communicator share the same information.
    // *********************************************************************
      GetTime( timeSta );
      // Compute the coefficient (i.e. the inner product of the nonlocal
      // pseudopotential and basis functions in the form of <phi|l>) for
      // nonlocal pseudopotential projectors locally
      //
      // Also get the inner product of the form <D_{x,y,z} phi | l> for
      // nonlocal pseudopotential projectors locally

      vnlCoef_.LocalMap().clear();
      for( Int d = 0; d < DIM; d++ ){
        vnlDrvCoef_[d].LocalMap().clear();
      }


      // Setup vnlCoeff / vnlDrvCoeff on device
      DistVec< Index3, std::map< Int, cuda::device_vector<Real> >, ElemPrtn >
        vnlCoef_device;
      std::vector< decltype(vnlCoef_device) > vnlDrvCoef_device( DIM );

      vnlCoef_device.SetComm( vnlCoef_.Comm() );
      vnlCoef_device.Prtn() = vnlCoef_.Prtn();
      for( auto d = 0; d < DIM; ++d ) {
        vnlDrvCoef_device[d].SetComm( vnlDrvCoef_[d].Comm() );
        vnlDrvCoef_device[d].Prtn() = vnlDrvCoef_[d].Prtn();
      }


      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){

              std::map<Int, PseudoPot>& pseudoMap =
                pseudo_.LocalMap()[key];

              // Host Maps
              std::map<Int, DblNumMat>  coefMap;
              std::map<Int, DblNumMat>  coefDrvXMap;
              std::map<Int, DblNumMat>  coefDrvYMap;
              std::map<Int, DblNumMat>  coefDrvZMap;

              // Device Maps
              std::map<Int, cuda::device_vector<Real>>  coefMap_device;
              std::map<Int, cuda::device_vector<Real>>  coefDrvXMap_device;
              std::map<Int, cuda::device_vector<Real>>  coefDrvYMap_device;
              std::map<Int, cuda::device_vector<Real>>  coefDrvZMap_device;

              // Host Basis / DBasis
              DblNumMat&   basis = basisLGL_.LocalMap()[key];
              DblNumMat& DbasisX = Dbasis[0].LocalMap()[key];
              DblNumMat& DbasisY = Dbasis[1].LocalMap()[key];
              DblNumMat& DbasisZ = Dbasis[2].LocalMap()[key];

              // Device Basis / DBasis
              auto&   basis_device = basisLGL_device.LocalMap()[key];
              auto& DbasisX_device = Dbasis_device[0].LocalMap()[key];
              auto& DbasisY_device = Dbasis_device[1].LocalMap()[key];
              auto& DbasisZ_device = Dbasis_device[2].LocalMap()[key];

              Int numBasis = basis.n();
              Int numBasisTotal = 0;
              MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT,
                  MPI_SUM, domain_.rowComm );

              // Loop over atoms, regardless of whether this atom belongs
              // to this element or not.
              for(auto mi  = pseudoMap.begin(); mi != pseudoMap.end(); mi++ ){
                Int atomIdx = (*mi).first;
                std::vector<NonlocalPP>&  vnlList = (*mi).second.vnlList;
                // NOTE: in intra-element parallelization, coef and
                // coefDrvX/Y/Z contain different data among processors in
                // each processor row in the same element.

                DblNumMat coef( numBasisTotal, vnlList.size() );
                DblNumMat coefDrvX( numBasisTotal, vnlList.size() );
                DblNumMat coefDrvY( numBasisTotal, vnlList.size() ); 
                DblNumMat coefDrvZ( numBasisTotal, vnlList.size() );

                SetValue( coef, 0.0 );
                SetValue( coefDrvX, 0.0 );
                SetValue( coefDrvY, 0.0 );
                SetValue( coefDrvZ, 0.0 );

                // Loop over projector
                // Method 2: Efficient way of implementation
                // Local inner product computation
                for( Int g = 0; g < vnlList.size(); g++ ){
                  SparseVec&  vnl = vnlList[g].first;
                  Int         idxSize = vnl.first.Size();

                  if( idxSize > 0 ) {
                    Int        *ptrIdx = vnl.first.Data();
                    Real       *ptrVal = vnl.second.VecData(VAL);
                    Real       *ptrWeight = LGLWeight3D.Data();
                    Real       *ptrBasis, *ptrDbasisX, *ptrDbasisY, *ptrDbasisZ;
                    Real       *ptrCoef, *ptrCoefDrvX, *ptrCoefDrvY, *ptrCoefDrvZ;
                    // Loop over basis function
                    for( Int b = 0; b < numBasis; b++ ){
                      // Loop over grid point
                      ptrBasis    = basis.VecData(b);
                      ptrDbasisX  = DbasisX.VecData(b);
                      ptrDbasisY  = DbasisY.VecData(b);
                      ptrDbasisZ  = DbasisZ.VecData(b);
                      ptrCoef     = coef.VecData(g);
                      ptrCoefDrvX = coefDrvX.VecData(g);
                      ptrCoefDrvY = coefDrvY.VecData(g);
                      ptrCoefDrvZ = coefDrvZ.VecData(g);

                      Int a = basisLGLIdx_(b); 

                      for( Int l = 0; l < idxSize; l++ ){
                        ptrCoef[a]     += ptrWeight[ptrIdx[l]] * 
                          ptrBasis[ptrIdx[l]] * ptrVal[l];
                        ptrCoefDrvX[a] += ptrWeight[ptrIdx[l]] *
                          ptrDbasisX[ptrIdx[l]] * ptrVal[l];
                        ptrCoefDrvY[a] += ptrWeight[ptrIdx[l]] *
                          ptrDbasisY[ptrIdx[l]] * ptrVal[l];
                        ptrCoefDrvZ[a] += ptrWeight[ptrIdx[l]] *
                          ptrDbasisZ[ptrIdx[l]] * ptrVal[l];
                      }
                    }
                  } // non-empty
                } // for (g)


                DblNumMat coefTemp( numBasisTotal, vnlList.size() );

                SetValue( coefTemp, 0.0 );
                MPI_Allreduce( coef.Data(), coefTemp.Data(), numBasisTotal * vnlList.size(), MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                coefMap[atomIdx] = coefTemp;

                SetValue( coefTemp, 0.0 );
                MPI_Allreduce( coefDrvX.Data(), coefTemp.Data(), numBasisTotal * vnlList.size(), MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                coefDrvXMap[atomIdx] = coefTemp;

                SetValue( coefTemp, 0.0 );
                MPI_Allreduce( coefDrvY.Data(), coefTemp.Data(), numBasisTotal * vnlList.size(), MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                coefDrvYMap[atomIdx] = coefTemp;

                SetValue( coefTemp, 0.0 );
                MPI_Allreduce( coefDrvZ.Data(), coefTemp.Data(), numBasisTotal * vnlList.size(), MPI_DOUBLE, MPI_SUM, domain_.rowComm );
                coefDrvZMap[atomIdx] = coefTemp;

              } // mi


              // Save coef and its derivativees in vnlCoef and vnlDrvCoef
              // strctures, for the use of constructing DG matrix, force
              // computation and a posteriori error estimation.
              vnlCoef_.LocalMap()[key] = coefMap;
              vnlDrvCoef_[0].LocalMap()[key] = coefDrvXMap;
              vnlDrvCoef_[1].LocalMap()[key] = coefDrvYMap;
              vnlDrvCoef_[2].LocalMap()[key] = coefDrvZMap;
            } // own this element
          } // for (i)


      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << 
        "Time for computing the coefficient for nonlocal projector is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );

      // Inter-element communication of the coefficient matrices for
      // nonlocal pseudopotential.
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

      vnlCoef_.GetBegin( pseudoIdx, NO_MASK );
      for( Int d = 0; d < DIM; d++ )
        vnlDrvCoef_[d].GetBegin( pseudoIdx, NO_MASK );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << 
        "Time for starting the communication of pseudopotential coefficent is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif



    // *********************************************************************
    // Diagonal part of the DG Hamiltonian matrix:  
    // 1) Laplacian 
    // 2) Local potential
    // 3) Intra-element part of boundary terms
    // 
    // Overlap communication (basis on the boundary and nonlocal
    // potential) with computation
    // *********************************************************************
    {
      GetTime(timeSta);


      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
              DblNumMat&  basis = basisLGL_.LocalMap()[key];
              Int numBasis = basis.n();
              Int numBasisTotal = 0;
              MPI_Allreduce( &numBasis, &numBasisTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );
              DblNumMat   localMat( numBasisTotal, numBasisTotal );
              SetValue( localMat, 0.0 );
              // In all matrix assembly process, Only compute the upper
              // triangular matrix use symmetry later


              //#ifdef _USE_OPENMP_
              //#pragma omp parallel 
              {
                //#endif
                // Private pointer among the OpenMP sessions
                Real* ptrLocalMat = localMat.Data();

                // Laplacian part
                {
                  DblNumMat&  DbasisX = Dbasis[0].LocalMap()[key];
                  DblNumMat&  DbasisY = Dbasis[1].LocalMap()[key];
                  DblNumMat&  DbasisZ = Dbasis[2].LocalMap()[key];

                    // In order to compute the Laplacian part
                    //
                    // Dphi_i * w * Dphi_j, sqrt(w) is first multiplied to
                    // the derivative of the basis functions.
                    for( Int g = 0; g < basis.n(); g++ ){
                      Real *ptr = sqrtLGLWeight3D.Data();
                      Real *ptrX = DbasisX.VecData(g);
                      Real *ptrY = DbasisY.VecData(g);
                      Real *ptrZ = DbasisZ.VecData(g);
                      for( Int l = 0; l < basis.m(); l++ ){
                        *(ptrX++)  *= *ptr;
                        *(ptrY++)  *= *ptr;
                        *(ptrZ++)  *= *ptr;
                        ptr++;
                      }
                    }

                    // Convert the basis functions from column based
                    // partition to row based partition
                    Int height = basis.m();
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

                    DblNumMat DbasisXRow( heightLocal, width );
                    DblNumMat DbasisYRow( heightLocal, width );
                    DblNumMat DbasisZRow( heightLocal, width );

                    DblNumMat localMatXTemp( width, width );
                    DblNumMat localMatYTemp( width, width );
                    DblNumMat localMatZTemp( width, width );

                    DblNumMat localMatTemp1( width, width );
                    DblNumMat localMatTemp2( width, width );

                    AlltoallForward (DbasisX, DbasisXRow, domain_.rowComm);
                    AlltoallForward (DbasisY, DbasisYRow, domain_.rowComm);
                    AlltoallForward (DbasisZ, DbasisZRow, domain_.rowComm);

                    // Compute the local contribution to the Laplacian
                    // part for the row partitioned basis functions.
                    SetValue( localMatXTemp, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, DbasisXRow.Data(), numLGLGridLocal, 
                        DbasisXRow.Data(), numLGLGridLocal, 0.0,
                        localMatXTemp.Data(), numBasisTotal );

                    SetValue( localMatYTemp, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, DbasisYRow.Data(), numLGLGridLocal, 
                        DbasisYRow.Data(), numLGLGridLocal, 0.0,
                        localMatYTemp.Data(), numBasisTotal );

                    SetValue( localMatZTemp, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal,
                        numLGLGridLocal, 1.0, DbasisZRow.Data(),
                        numLGLGridLocal, DbasisZRow.Data(),
                        numLGLGridLocal, 0.0, localMatZTemp.Data(),
                        numBasisTotal );

                    SetValue( localMatTemp1, 0.0 );
                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMatTemp1(a,b) = localMatXTemp(a,b) +
                          localMatYTemp(a,b) + localMatZTemp(a,b);  
                      }

                    SetValue( localMatTemp2, 0.0 );
                    MPI_Allreduce( localMatTemp1.Data(),
                        localMatTemp2.Data(), numBasisTotal *
                        numBasisTotal, MPI_DOUBLE, MPI_SUM,
                        domain_.rowComm );

                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMat(a,b) += 0.5 * localMatTemp2(a,b);  
                      } 


                }
#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "After the Laplacian part." << std::endl;
#endif

                // Local potential part
                {
                  DblNumVec&  vtot  = vtotLGL_.LocalMap()[key];
                  //#ifdef _USE_OPENMP_
                  //#pragma omp for schedule(dynamic,1) nowait
                  //#endif

                  // For this step, we need a copy of the basis in the
                  // FourDotProduct process.  This is because the local
                  // potential has both positive and negative
                  // contribution, and the sqrt trick is not applicable
                  // here.

                    DblNumMat basisTemp (basis.m(), basis.n()); 
                    SetValue( basisTemp, 0.0 );

                    // This is the same as the FourDotProduct process.
                    for( Int g = 0; g < basis.n(); g++ ){
                      Real *ptr1 = LGLWeight3D.Data();
                      Real *ptr2 = vtot.Data();
                      Real *ptr3 = basis.VecData(g);
                      Real *ptr4 = basisTemp.VecData(g);
                      for( Int l = 0; l < basis.m(); l++ ){
                        *(ptr4++) = (*(ptr1++)) * (*(ptr2++)) * (*(ptr3++));
                      }
                    }


                    // Convert the basis functions from column based
                    // partition to row based partition
                    Int height = basis.m();
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

                    DblNumMat basisRow( heightLocal, width );
                    DblNumMat basisTempRow( heightLocal, width );

                    DblNumMat localMatTemp1( width, width );
                    DblNumMat localMatTemp2( width, width );

                    AlltoallForward (basis, basisRow, domain_.rowComm);
                    AlltoallForward (basisTemp, basisTempRow, domain_.rowComm);

                    SetValue( localMatTemp1, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, basisRow.Data(), numLGLGridLocal, 
                        basisTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp1.Data(), numBasisTotal );

                    SetValue( localMatTemp2, 0.0 );
                    MPI_Allreduce( localMatTemp1.Data(),
                        localMatTemp2.Data(), numBasisTotal *
                        numBasisTotal, MPI_DOUBLE, MPI_SUM,
                        domain_.rowComm );

                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMat(a,b) += localMatTemp2(a,b);  
                      } 


                }// Local potential part

#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "After the local potential part." << std::endl;
#endif

                // x-direction: intra-element part of the boundary term
                {
                  Int  numGridFace = numGrid[1] * numGrid[2];

                  DblNumMat&  valL = basisJump[XL].LocalMap()[key];
                  DblNumMat&  valR = basisJump[XR].LocalMap()[key];
                  DblNumMat&  drvL = DbasisAverage[XL].LocalMap()[key];
                  DblNumMat&  drvR = DbasisAverage[XR].LocalMap()[key];

                  // intra-element part of the boundary term
                  Real intByPartTerm, penaltyTerm;
                  //#ifdef _USE_OPENMP_
                  //#pragma omp for schedule(dynamic,1) nowait
                  //#endif



                    DblNumMat valLTemp(numGridFace, numBasis); 
                    DblNumMat valRTemp(numGridFace, numBasis); 
                    DblNumMat drvLTemp(numGridFace, numBasis); 
                    DblNumMat drvRTemp(numGridFace, numBasis); 

                    SetValue( valLTemp, 0.0 );
                    SetValue( valRTemp, 0.0 );
                    SetValue( drvLTemp, 0.0 );
                    SetValue( drvRTemp, 0.0 );

                    // sqrt(w) is first multiplied to the derivative of
                    // the derivatives and function values on the surface
                    // for inner product.
                    for( Int g = 0; g < numBasis; g++ ){
                      Real *ptr = sqrtLGLWeight2D[0].Data();
                      Real *ptr1 = valL.VecData(g);
                      Real *ptr2 = valR.VecData(g);
                      Real *ptr3 = drvL.VecData(g);
                      Real *ptr4 = drvR.VecData(g);
                      Real *ptr11 = valLTemp.VecData(g);
                      Real *ptr22 = valRTemp.VecData(g);
                      Real *ptr33 = drvLTemp.VecData(g);
                      Real *ptr44 = drvRTemp.VecData(g);
                      for( Int l = 0; l < numGridFace; l++ ){
                        *(ptr11++) = (*(ptr1++)) * (*ptr);
                        *(ptr22++) = (*(ptr2++)) * (*ptr);
                        *(ptr33++) = (*(ptr3++)) * (*ptr);
                        *(ptr44++) = (*(ptr4++)) * (*ptr);
                        ptr++;
                      }
                    }

                    // Convert the basis functions from column based
                    // partition to row based partition

                    Int height = numGridFace;
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

                    Int numBasisLocal = widthLocal;

                    DblNumMat valLTempRow( heightLocal, width );
                    DblNumMat valRTempRow( heightLocal, width );
                    DblNumMat drvLTempRow( heightLocal, width );
                    DblNumMat drvRTempRow( heightLocal, width );

                    DblNumMat localMatTemp1( width, width );
                    DblNumMat localMatTemp2( width, width );
                    DblNumMat localMatTemp3( width, width );
                    DblNumMat localMatTemp4( width, width );
                    DblNumMat localMatTemp5( width, width );
                    DblNumMat localMatTemp6( width, width );

                    DblNumMat localMatTemp7( width, width );
                    DblNumMat localMatTemp8( width, width );


                    AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                    AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                    AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                    AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);


                    SetValue( localMatTemp1, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp1.Data(), numBasisTotal );

                    SetValue( localMatTemp2, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        drvLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp2.Data(), numBasisTotal );

                    SetValue( localMatTemp3, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp3.Data(), numBasisTotal );

                    SetValue( localMatTemp4, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        drvRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp4.Data(), numBasisTotal );

                    SetValue( localMatTemp5, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp5.Data(), numBasisTotal );

                    SetValue( localMatTemp6, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp6.Data(), numBasisTotal );


                    SetValue( localMatTemp7, 0.0 );
                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMatTemp7(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                          - 0.5 * localMatTemp3(a,b) - 0.5 * localMatTemp4(a,b)
                          + penaltyAlpha_ * localMatTemp5(a,b) + penaltyAlpha_ * localMatTemp6(a,b);  
                      } 


                    SetValue( localMatTemp8, 0.0 );
                    MPI_Allreduce( localMatTemp7.Data(), localMatTemp8.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMat(a,b) += localMatTemp8(a,b);  
                      } 


                } // x-direction

                // y-direction: intra-element part of the boundary term
                {
                  Int  numGridFace = numGrid[0] * numGrid[2];

                  DblNumMat&  valL = basisJump[YL].LocalMap()[key];
                  DblNumMat&  valR = basisJump[YR].LocalMap()[key];
                  DblNumMat&  drvL = DbasisAverage[YL].LocalMap()[key];
                  DblNumMat&  drvR = DbasisAverage[YR].LocalMap()[key];

                  // intra-element part of the boundary term
                  Real intByPartTerm, penaltyTerm;
                  //#ifdef _USE_OPENMP_
                  //#pragma omp for schedule(dynamic,1) nowait
                  //#endif







                    DblNumMat valLTemp(numGridFace, numBasis); 
                    DblNumMat valRTemp(numGridFace, numBasis); 
                    DblNumMat drvLTemp(numGridFace, numBasis); 
                    DblNumMat drvRTemp(numGridFace, numBasis); 

                    SetValue( valLTemp, 0.0 );
                    SetValue( valRTemp, 0.0 );
                    SetValue( drvLTemp, 0.0 );
                    SetValue( drvRTemp, 0.0 );

                    // sqrt(w) is first multiplied to the derivative of
                    // the derivatives and function values on the surface
                    // for inner product.
                    for( Int g = 0; g < numBasis; g++ ){
                      Real *ptr = sqrtLGLWeight2D[1].Data();
                      Real *ptr1 = valL.VecData(g);
                      Real *ptr2 = valR.VecData(g);
                      Real *ptr3 = drvL.VecData(g);
                      Real *ptr4 = drvR.VecData(g);
                      Real *ptr11 = valLTemp.VecData(g);
                      Real *ptr22 = valRTemp.VecData(g);
                      Real *ptr33 = drvLTemp.VecData(g);
                      Real *ptr44 = drvRTemp.VecData(g);
                      for( Int l = 0; l < numGridFace; l++ ){
                        *(ptr11++) = (*(ptr1++)) * (*ptr);
                        *(ptr22++) = (*(ptr2++)) * (*ptr);
                        *(ptr33++) = (*(ptr3++)) * (*ptr);
                        *(ptr44++) = (*(ptr4++)) * (*ptr);
                        ptr++;
                      }
                    }


                    // Convert the basis functions from column based
                    // partition to row based partition
                    Int height = numGridFace;
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

                    Int numBasisLocal = widthLocal;

                    DblNumMat valLTempRow( heightLocal, width );
                    DblNumMat valRTempRow( heightLocal, width );
                    DblNumMat drvLTempRow( heightLocal, width );
                    DblNumMat drvRTempRow( heightLocal, width );

                    DblNumMat localMatTemp1( width, width );
                    DblNumMat localMatTemp2( width, width );
                    DblNumMat localMatTemp3( width, width );
                    DblNumMat localMatTemp4( width, width );
                    DblNumMat localMatTemp5( width, width );
                    DblNumMat localMatTemp6( width, width );

                    DblNumMat localMatTemp7( width, width );
                    DblNumMat localMatTemp8( width, width );

                    AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                    AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                    AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                    AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);

                    SetValue( localMatTemp1, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp1.Data(), numBasisTotal );

                    SetValue( localMatTemp2, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        drvLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp2.Data(), numBasisTotal );

                    SetValue( localMatTemp3, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp3.Data(), numBasisTotal );

                    SetValue( localMatTemp4, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        drvRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp4.Data(), numBasisTotal );

                    SetValue( localMatTemp5, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp5.Data(), numBasisTotal );

                    SetValue( localMatTemp6, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp6.Data(), numBasisTotal );


                    SetValue( localMatTemp7, 0.0 );
                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMatTemp7(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                          - 0.5 * localMatTemp3(a,b) - 0.5 * localMatTemp4(a,b)
                          + penaltyAlpha_ * localMatTemp5(a,b) + penaltyAlpha_ * localMatTemp6(a,b);  
                      } 


                    SetValue( localMatTemp8, 0.0 );
                    MPI_Allreduce( localMatTemp7.Data(), localMatTemp8.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMat(a,b) += localMatTemp8(a,b);  
                      } 




                } // y-direction

                // z-direction: intra-element part of the boundary term
                {
                  Int  numGridFace = numGrid[0] * numGrid[1];

                  DblNumMat&  valL = basisJump[ZL].LocalMap()[key];
                  DblNumMat&  valR = basisJump[ZR].LocalMap()[key];
                  DblNumMat&  drvL = DbasisAverage[ZL].LocalMap()[key];
                  DblNumMat&  drvR = DbasisAverage[ZR].LocalMap()[key];

                  // intra-element part of the boundary term
                  Real intByPartTerm, penaltyTerm;
                  //#ifdef _USE_OPENMP_
                  //#pragma omp for schedule(dynamic,1) nowait
                  //#endif







                    DblNumMat valLTemp(numGridFace, numBasis); 
                    DblNumMat valRTemp(numGridFace, numBasis); 
                    DblNumMat drvLTemp(numGridFace, numBasis); 
                    DblNumMat drvRTemp(numGridFace, numBasis); 

                    SetValue( valLTemp, 0.0 );
                    SetValue( valRTemp, 0.0 );
                    SetValue( drvLTemp, 0.0 );
                    SetValue( drvRTemp, 0.0 );

                    // sqrt(w) is first multiplied to the derivative of
                    // the derivatives and function values on the surface
                    // for inner product.
                    for( Int g = 0; g < numBasis; g++ ){
                      Real *ptr = sqrtLGLWeight2D[2].Data();
                      Real *ptr1 = valL.VecData(g);
                      Real *ptr2 = valR.VecData(g);
                      Real *ptr3 = drvL.VecData(g);
                      Real *ptr4 = drvR.VecData(g);
                      Real *ptr11 = valLTemp.VecData(g);
                      Real *ptr22 = valRTemp.VecData(g);
                      Real *ptr33 = drvLTemp.VecData(g);
                      Real *ptr44 = drvRTemp.VecData(g);
                      for( Int l = 0; l < numGridFace; l++ ){
                        *(ptr11++) = (*(ptr1++)) * (*ptr);
                        *(ptr22++) = (*(ptr2++)) * (*ptr);
                        *(ptr33++) = (*(ptr3++)) * (*ptr);
                        *(ptr44++) = (*(ptr4++)) * (*ptr);
                        ptr++;
                      }
                    }


                    // Convert the basis functions from column based
                    // partition to row based partition
                    Int height = numGridFace;
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

                    Int numBasisLocal = widthLocal;

                    DblNumMat valLTempRow( heightLocal, width );
                    DblNumMat valRTempRow( heightLocal, width );
                    DblNumMat drvLTempRow( heightLocal, width );
                    DblNumMat drvRTempRow( heightLocal, width );

                    DblNumMat localMatTemp1( width, width );
                    DblNumMat localMatTemp2( width, width );
                    DblNumMat localMatTemp3( width, width );
                    DblNumMat localMatTemp4( width, width );
                    DblNumMat localMatTemp5( width, width );
                    DblNumMat localMatTemp6( width, width );

                    DblNumMat localMatTemp7( width, width );
                    DblNumMat localMatTemp8( width, width );

                    AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                    AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                    AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                    AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);

                    SetValue( localMatTemp1, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp1.Data(), numBasisTotal );

                    SetValue( localMatTemp2, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        drvLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp2.Data(), numBasisTotal );

                    SetValue( localMatTemp3, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, drvRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp3.Data(), numBasisTotal );

                    SetValue( localMatTemp4, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        drvRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp4.Data(), numBasisTotal );

                    SetValue( localMatTemp5, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valLTempRow.Data(), numLGLGridLocal, 
                        valLTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp5.Data(), numBasisTotal );

                    SetValue( localMatTemp6, 0.0 );
                    blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
                        1.0, valRTempRow.Data(), numLGLGridLocal, 
                        valRTempRow.Data(), numLGLGridLocal, 0.0,
                        localMatTemp6.Data(), numBasisTotal );


                    SetValue( localMatTemp7, 0.0 );
                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMatTemp7(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                          - 0.5 * localMatTemp3(a,b) - 0.5 * localMatTemp4(a,b)
                          + penaltyAlpha_ * localMatTemp5(a,b) + penaltyAlpha_ * localMatTemp6(a,b);  
                      } 


                    SetValue( localMatTemp8, 0.0 );
                    MPI_Allreduce( localMatTemp7.Data(), localMatTemp8.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                    for( Int a = 0; a < numBasisTotal; a++ )
                      for( Int b = a; b < numBasisTotal; b++ ){
                        localMat(a,b) += localMatTemp8(a,b);  
                      } 
                } // z-direction

#if ( _DEBUGlevel_ >= 1 )
                statusOFS << "After the boundary part." << std::endl;
#endif
                //#ifdef _USE_OPENMP_
              }
              //#endif
              // Symmetrize the diagonal part of the DG matrix
              for( Int a = 0; a < numBasisTotal; a++ )
                for( Int b = 0; b < a; b++ ){
                  localMat(a,b) = localMat(b,a);
                }

              // Add localMat to HMat_
              ElemMatKey matKey( key, key );
              std::map<ElemMatKey, DblNumMat>::iterator mi = 
                HMat_.LocalMap().find( matKey );
              if( mi == HMat_.LocalMap().end() ){
                HMat_.LocalMap()[matKey] = localMat;
              }
              else{
                DblNumMat&  mat = (*mi).second;
                blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
                    mat.Data(), 1); // y = a*x + y -> mat = localMat + mat
              }
            }
          } // for (i)

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for the diagonal part of the DG matrix is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    } 



    // *********************************************************************
    // Nonlocal pseudopotential term, Part II
    // Finish the communication of the nonlocal pseudopotential
    // Update the nonlocal potential part of the matrix
    //
    // In the intra-element parallelization, every processor in the same
    // processor row communication is doing the same (repetitive) work.
    // *********************************************************************
    if(_NON_LOCAL_){
      GetTime( timeSta );
      // Finish the communication of the nonlocal pseudopotential

      vnlCoef_.GetEnd( NO_MASK );
      for( Int d = 0; d < DIM; d++ )
        vnlDrvCoef_[d].GetEnd( NO_MASK );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << 
        "Time for the communication of pseudopotential coefficent is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );
      // Update the nonlocal potential part of the matrix

      // Loop over atoms
      for( Int atomIdx = 0; atomIdx < numAtom; atomIdx++ ){
        if( atomPrtn_.Owner(atomIdx) == (mpirank / dmRow_) ){
          DblNumVec&  vnlWeight = vnlWeightMap_[atomIdx];    
          // Loop over element 1
          for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
              ei  = vnlCoef_.LocalMap().begin();
              ei != vnlCoef_.LocalMap().end(); ei++ ){
            Index3 key1 = (*ei).first;
            std::map<Int, DblNumMat>& coefMap1 = (*ei).second; 
            std::map<Int, PseudoPot>& pseudoMap = pseudo_.LocalMap()[key1];

            std::map<Int, DblNumMat>::iterator mi = 
              coefMap1.find( atomIdx );
            if( mi != coefMap1.end() ){
              DblNumMat&  coef1 = (*mi).second;
              // Skip the calculation if there is no adaptive local
              // basis function.  
              if( coef1.m() == 0 ){
                continue;
              }
              // Loop over element j
              for( std::map<Index3, std::map<Int, DblNumMat> >::iterator 
                  ej  = vnlCoef_.LocalMap().begin();
                  ej != vnlCoef_.LocalMap().end(); ej++ ){
                Index3 key2 = (*ej).first;
                std::map<Int, DblNumMat>& coefMap2 = (*ej).second;

                std::map<Int, DblNumMat>::iterator ni = 
                  coefMap2.find( atomIdx );
                // Compute the contribution to HMat_(key1, key2)
                if( ni != coefMap2.end() ){
                  DblNumMat& coef2 = (*ni).second;
                  // Skip the calculation if there is no adaptive local
                  // basis function.  
                  if( coef2.m() == 0 ){
                    continue;
                  }

                  DblNumMat localMat( coef1.m(), coef2.m() );
                  SetValue( localMat, 0.0 );
                  // Check size consistency
                  if( coef1.n() != coef2.n() ||
                      coef1.n() != vnlWeight.Size() ){
                    std::ostringstream msg;
                    msg 
                      << "Error in assembling the nonlocal pseudopotential part of the DG matrix." << std::endl
                      << "Atom number " << atomIdx << std::endl
                      << "Element 1: " << key1 << ", Element 2: " << key2 << std::endl
                      << "Coef matrix 1 size : " << coef1.m() << " x " << coef1.n() << std::endl
                      << "Coef matrix 2 size : " << coef2.m() << " x " << coef2.n() << std::endl
                      << "vnlWeight     size : " << vnlWeight.Size() << std::endl;

                    ErrorHandling( msg.str().c_str() );
                  }
                  // Outer product with the weight of the nonlocal
                  // pseudopotential to form local matrix
                  //
                  // localMat = coef1 * diag(weight) * coef2^T.
                  for( Int l = 0; l < vnlWeight.Size(); l++ ){
                    Real weight = vnlWeight(l);
                    blas::Ger( coef1.m(), coef2.m(), weight, 
                        coef1.VecData(l), 1, coef2.VecData(l), 1,
                        localMat.Data(), localMat.m() );
                  }
                  // Add to HMat_
                  ElemMatKey matKey( key1, key2 );
                  std::map<ElemMatKey, DblNumMat>::iterator mati = 
                    HMat_.LocalMap().find( matKey );
                  if( mati == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMat;
                  }
                  else{
                    DblNumMat&  mat = (*mati).second;
                    blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
                        mat.Data(), 1 );
                  }
                } // found atomIdx in element 2
              } // for (ej)
            } // found atomIdx in element 1
          } // for (ei)
        } // own this atom
      }

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << 
        "Time for updating the nonlocal potential part of the matrix is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }


    // *********************************************************************
    // Boundary terms, Part II
    // Finish the communication of boundary terms
    // Update the inter-element boundary part of the matrix
    //
    // In the intra-element parallelization, processors in the same
    // processor row communication are doing different work.
    // *********************************************************************

    {
      GetTime( timeSta );

      // Map sizes before XXX: Debug
      size_t dba_XR_sz_before = DbasisAverage[XR].LocalMap().size();
      size_t dba_YR_sz_before = DbasisAverage[YR].LocalMap().size();
      size_t dba_ZR_sz_before = DbasisAverage[ZR].LocalMap().size();
      size_t bj_XR_sz_before  = basisJump[XR].LocalMap().size();
      size_t bj_YR_sz_before  = basisJump[YR].LocalMap().size();
      size_t bj_ZR_sz_before  = basisJump[ZR].LocalMap().size();


      // Host deserialize
      DbasisAverage[XR].GetEnd( NO_MASK );
      DbasisAverage[YR].GetEnd( NO_MASK );
      DbasisAverage[ZR].GetEnd( NO_MASK );

      basisJump[XR].GetEnd( NO_MASK );
      basisJump[YR].GetEnd( NO_MASK );
      basisJump[ZR].GetEnd( NO_MASK );

      // Device deserialize
      // TODO: involves host communication, should optimize
      DbasisAverage_device[XR].GetEnd( NO_MASK );
      DbasisAverage_device[YR].GetEnd( NO_MASK );
      DbasisAverage_device[ZR].GetEnd( NO_MASK );

      basisJump_device[XR].GetEnd( NO_MASK );
      basisJump_device[YR].GetEnd( NO_MASK );
      basisJump_device[ZR].GetEnd( NO_MASK );

      // Map sizes after XXX: Debug
      size_t dba_XR_sz_after = DbasisAverage[XR].LocalMap().size();
      size_t dba_YR_sz_after = DbasisAverage[YR].LocalMap().size();
      size_t dba_ZR_sz_after = DbasisAverage[ZR].LocalMap().size();
      size_t bj_XR_sz_after  = basisJump[XR].LocalMap().size();
      size_t bj_YR_sz_after  = basisJump[YR].LocalMap().size();
      size_t bj_ZR_sz_after  = basisJump[ZR].LocalMap().size();

      // Check that communication is correct
      {
        statusOFS << "DBWY CHECK GetBegin/GetEnd" << std::endl;

        statusOFS << "LM DBA XR: " << dba_XR_sz_before << ", " << dba_XR_sz_after << std::endl;
        statusOFS << "LM DBA YR: " << dba_YR_sz_before << ", " << dba_YR_sz_after << std::endl;
        statusOFS << "LM DBA ZR: " << dba_ZR_sz_before << ", " << dba_ZR_sz_after << std::endl;
        statusOFS << "LM BJ XR: " << bj_XR_sz_before << ", " << bj_XR_sz_after << std::endl;
        statusOFS << "LM BJ YR: " << bj_YR_sz_before << ", " << bj_YR_sz_after << std::endl;
        statusOFS << "LM BJ ZR: " << bj_ZR_sz_before << ", " << bj_ZR_sz_after << std::endl;


        auto compare_distvec = []( std::string label, const auto& a, const auto& a_device ) {

          // Make sure that the maps are the same size
          assert( a.LocalMap().size() == a_device.LocalMap().size() );

          statusOFS << label << " Diffs after GetEnd" << std::endl;

          // Loop over keys in host copy
          for( const auto& [key, host_val] : a.LocalMap() ) {

            // Check that all proper keys are present in Dvice copy
            auto dev_it = a_device.LocalMap().find( key );
            assert( dev_it != a_device.LocalMap().end() );

            const auto& [dev_key, dev_val] = *dev_it;

            // Check that the members have the same values
            assert( dev_val.size() == host_val.Size() );

            std::vector< Real > dev_val_h( dev_val.size() );
            cuda::copy( dev_val, dev_val_h );

            for( auto i = 0; i < dev_val.size(); ++i )
              dev_val_h[i] = std::abs( dev_val_h[i] - host_val.Data()[i] );

            statusOFS << "  " << key << ", " << *std::max_element( dev_val_h.begin(), dev_val_h.end() ) << std::endl;
     
          }
          
        };

        compare_distvec( "DbasisAverage XR", DbasisAverage[XR], DbasisAverage_device[XR] );
        compare_distvec( "DbasisAverage YR", DbasisAverage[YR], DbasisAverage_device[YR] );
        compare_distvec( "DbasisAverage ZR", DbasisAverage[ZR], DbasisAverage_device[ZR] );
        compare_distvec( "basisJump XR", basisJump[XR], basisJump_device[XR] );
        compare_distvec( "basisJump YR", basisJump[YR], basisJump_device[YR] );
        compare_distvec( "basisJump ZR", basisJump[ZR], basisJump_device[ZR] );

      }



      MPI_Barrier( domain_.comm );
      MPI_Barrier( domain_.rowComm );
      MPI_Barrier( domain_.colComm );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for the remaining communication cost is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      GetTime( timeSta );
      // Update the inter-element boundary part of the matrix
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
                DblNumMat&  drvL = DbasisAverage[XR].LocalMap()[keyL];
                DblNumMat&  drvR = DbasisAverage[XL].LocalMap()[keyR];

                Int numBasisL = valL.n();
                Int numBasisR = valR.n();

                Int numBasisLTotal = 0;
                Int numBasisRTotal = 0;

                MPI_Allreduce( &numBasisL, &numBasisLTotal, 1, MPI_INT,
                    MPI_SUM, domain_.rowComm );
                MPI_Allreduce( &numBasisR, &numBasisRTotal, 1, MPI_INT,
                    MPI_SUM, domain_.rowComm );

                DblNumMat   localMat( numBasisLTotal, numBasisRTotal );
                SetValue( localMat, 0.0 );

                // inter-element part of the boundary term



                  DblNumMat valLTemp(numGridFace, numBasisL); 
                  DblNumMat valRTemp(numGridFace, numBasisR); 
                  DblNumMat drvLTemp(numGridFace, numBasisL); 
                  DblNumMat drvRTemp(numGridFace, numBasisR); 

                  SetValue( valLTemp, 0.0 );
                  SetValue( valRTemp, 0.0 );
                  SetValue( drvLTemp, 0.0 );
                  SetValue( drvRTemp, 0.0 );

                  for( Int g = 0; g < numBasisL; g++ ){
                    Real *ptr = sqrtLGLWeight2D[0].Data();
                    Real *ptr1 = valL.VecData(g);
                    Real *ptr2 = drvL.VecData(g);
                    Real *ptr11 = valLTemp.VecData(g);
                    Real *ptr22 = drvLTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  for( Int g = 0; g < numBasisR; g++ ){
                    Real *ptr = sqrtLGLWeight2D[0].Data();
                    Real *ptr1 = valR.VecData(g);
                    Real *ptr2 = drvR.VecData(g);
                    Real *ptr11 = valRTemp.VecData(g);
                    Real *ptr22 = drvRTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  Int height = numGridFace;
                  Int widthL = numBasisLTotal;
                  Int widthR = numBasisRTotal;

                  Int widthLBlocksize = widthL / mpisizeRow;
                  Int widthRBlocksize = widthR / mpisizeRow;
                  Int heightBlocksize = height / mpisizeRow;

                  Int widthLLocal = widthLBlocksize;
                  Int widthRLocal = widthRBlocksize;
                  Int heightLocal = heightBlocksize;

                  if(mpirankRow < (widthL % mpisizeRow)){
                    widthLLocal = widthLBlocksize + 1;
                  }

                  if(mpirankRow < (widthR % mpisizeRow)){
                    widthRLocal = widthRBlocksize + 1;
                  }

                  if(mpirankRow < (height % mpisizeRow)){
                    heightLocal = heightBlocksize + 1;
                  }

                  Int numLGLGridTotal = height;  
                  Int numLGLGridLocal = heightLocal;  

                  Int numBasisLLocal = widthLLocal;
                  Int numBasisRLocal = widthRLocal;

                  DblNumMat valLTempRow( heightLocal, widthL );
                  DblNumMat valRTempRow( heightLocal, widthR );
                  DblNumMat drvLTempRow( heightLocal, widthL );
                  DblNumMat drvRTempRow( heightLocal, widthR );

                  DblNumMat localMatTemp1( widthL, widthR );
                  DblNumMat localMatTemp2( widthL, widthR );
                  DblNumMat localMatTemp3( widthL, widthR );
                  DblNumMat localMatTemp4( widthL, widthR );
                  DblNumMat localMatTemp5( widthL, widthR );

                  AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                  AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                  AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                  AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);

                  SetValue( localMatTemp1, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, drvLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp1.Data(), numBasisLTotal );

                  SetValue( localMatTemp2, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      drvRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp2.Data(), numBasisLTotal );

                  SetValue( localMatTemp3, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp3.Data(), numBasisLTotal );

                  SetValue( localMatTemp4, 0.0 );
                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMatTemp4(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                        + penaltyAlpha_ * localMatTemp3(a,b);  
                    } 

                  SetValue( localMatTemp5, 0.0 );
                  MPI_Allreduce( localMatTemp4.Data(), localMatTemp5.Data(), numBasisLTotal * numBasisRTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMat(a,b) += localMatTemp5(a,b);  
                    } 



                // Add (keyL, keyR) to HMat_
                {
                  ElemMatKey matKey( keyL, keyR );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMat;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
                        mat.Data(), 1);
                  }
                }

                // Add (keyR, keyL) to HMat_
                {
                  DblNumMat localMatTran;
                  Transpose( localMat, localMatTran );
                  ElemMatKey matKey( keyR, keyL );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMatTran;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
                        mat.Data(), 1);
                  }
                }
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
                DblNumMat&  drvL = DbasisAverage[YR].LocalMap()[keyL];
                DblNumMat&  drvR = DbasisAverage[YL].LocalMap()[keyR];

                Int numBasisL = valL.n();
                Int numBasisR = valR.n();

                Int numBasisLTotal = 0;
                Int numBasisRTotal = 0;

                MPI_Allreduce( &numBasisL, &numBasisLTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );
                MPI_Allreduce( &numBasisR, &numBasisRTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

                DblNumMat   localMat( numBasisLTotal, numBasisRTotal );
                SetValue( localMat, 0.0 );

                // inter-element part of the boundary term





                  DblNumMat valLTemp(numGridFace, numBasisL); 
                  DblNumMat valRTemp(numGridFace, numBasisR); 
                  DblNumMat drvLTemp(numGridFace, numBasisL); 
                  DblNumMat drvRTemp(numGridFace, numBasisR); 

                  SetValue( valLTemp, 0.0 );
                  SetValue( valRTemp, 0.0 );
                  SetValue( drvLTemp, 0.0 );
                  SetValue( drvRTemp, 0.0 );

                  for( Int g = 0; g < numBasisL; g++ ){
                    Real *ptr = sqrtLGLWeight2D[1].Data();
                    Real *ptr1 = valL.VecData(g);
                    Real *ptr2 = drvL.VecData(g);
                    Real *ptr11 = valLTemp.VecData(g);
                    Real *ptr22 = drvLTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  for( Int g = 0; g < numBasisR; g++ ){
                    Real *ptr = sqrtLGLWeight2D[1].Data();
                    Real *ptr1 = valR.VecData(g);
                    Real *ptr2 = drvR.VecData(g);
                    Real *ptr11 = valRTemp.VecData(g);
                    Real *ptr22 = drvRTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  Int height = numGridFace;
                  Int widthL = numBasisLTotal;
                  Int widthR = numBasisRTotal;

                  Int widthLBlocksize = widthL / mpisizeRow;
                  Int widthRBlocksize = widthR / mpisizeRow;
                  Int heightBlocksize = height / mpisizeRow;

                  Int widthLLocal = widthLBlocksize;
                  Int widthRLocal = widthRBlocksize;
                  Int heightLocal = heightBlocksize;

                  if(mpirankRow < (widthL % mpisizeRow)){
                    widthLLocal = widthLBlocksize + 1;
                  }

                  if(mpirankRow < (widthR % mpisizeRow)){
                    widthRLocal = widthRBlocksize + 1;
                  }

                  if(mpirankRow < (height % mpisizeRow)){
                    heightLocal = heightBlocksize + 1;
                  }

                  Int numLGLGridTotal = height;  
                  Int numLGLGridLocal = heightLocal;  

                  Int numBasisLLocal = widthLLocal;
                  Int numBasisRLocal = widthRLocal;

                  DblNumMat valLTempRow( heightLocal, widthL );
                  DblNumMat valRTempRow( heightLocal, widthR );
                  DblNumMat drvLTempRow( heightLocal, widthL );
                  DblNumMat drvRTempRow( heightLocal, widthR );

                  DblNumMat localMatTemp1( widthL, widthR );
                  DblNumMat localMatTemp2( widthL, widthR );
                  DblNumMat localMatTemp3( widthL, widthR );
                  DblNumMat localMatTemp4( widthL, widthR );
                  DblNumMat localMatTemp5( widthL, widthR );

                  AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                  AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                  AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                  AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);

                  SetValue( localMatTemp1, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, drvLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp1.Data(), numBasisLTotal );

                  SetValue( localMatTemp2, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      drvRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp2.Data(), numBasisLTotal );

                  SetValue( localMatTemp3, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp3.Data(), numBasisLTotal );

                  SetValue( localMatTemp4, 0.0 );
                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMatTemp4(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                        + penaltyAlpha_ * localMatTemp3(a,b);  
                    } 

                  SetValue( localMatTemp5, 0.0 );
                  MPI_Allreduce( localMatTemp4.Data(), localMatTemp5.Data(), numBasisLTotal * numBasisRTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMat(a,b) += localMatTemp5(a,b);  
                    } 



                // Add (keyL, keyR) to HMat_
                {
                  ElemMatKey matKey( keyL, keyR );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMat;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
                        mat.Data(), 1);
                  }
                }

                // Add (keyR, keyL) to HMat_
                {
                  DblNumMat localMatTran;
                  Transpose( localMat, localMatTran );
                  ElemMatKey matKey( keyR, keyL );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMatTran;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
                        mat.Data(), 1);
                  }
                }
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
                DblNumMat&  drvL = DbasisAverage[ZR].LocalMap()[keyL];
                DblNumMat&  drvR = DbasisAverage[ZL].LocalMap()[keyR];

                Int numBasisL = valL.n();
                Int numBasisR = valR.n();

                Int numBasisLTotal = 0;
                Int numBasisRTotal = 0;

                MPI_Allreduce( &numBasisL, &numBasisLTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );
                MPI_Allreduce( &numBasisR, &numBasisRTotal, 1, MPI_INT, MPI_SUM, domain_.rowComm );

                DblNumMat   localMat( numBasisLTotal, numBasisRTotal );
                SetValue( localMat, 0.0 );

                // inter-element part of the boundary term


                  DblNumMat valLTemp(numGridFace, numBasisL); 
                  DblNumMat valRTemp(numGridFace, numBasisR); 
                  DblNumMat drvLTemp(numGridFace, numBasisL); 
                  DblNumMat drvRTemp(numGridFace, numBasisR); 

                  SetValue( valLTemp, 0.0 );
                  SetValue( valRTemp, 0.0 );
                  SetValue( drvLTemp, 0.0 );
                  SetValue( drvRTemp, 0.0 );

                  for( Int g = 0; g < numBasisL; g++ ){
                    Real *ptr = sqrtLGLWeight2D[2].Data();
                    Real *ptr1 = valL.VecData(g);
                    Real *ptr2 = drvL.VecData(g);
                    Real *ptr11 = valLTemp.VecData(g);
                    Real *ptr22 = drvLTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  for( Int g = 0; g < numBasisR; g++ ){
                    Real *ptr = sqrtLGLWeight2D[2].Data();
                    Real *ptr1 = valR.VecData(g);
                    Real *ptr2 = drvR.VecData(g);
                    Real *ptr11 = valRTemp.VecData(g);
                    Real *ptr22 = drvRTemp.VecData(g);
                    for( Int l = 0; l < numGridFace; l++ ){
                      *(ptr11++) = (*(ptr1++)) * (*ptr);
                      *(ptr22++) = (*(ptr2++)) * (*ptr);
                      ptr++;
                    }
                  }

                  Int height = numGridFace;
                  Int widthL = numBasisLTotal;
                  Int widthR = numBasisRTotal;

                  Int widthLBlocksize = widthL / mpisizeRow;
                  Int widthRBlocksize = widthR / mpisizeRow;
                  Int heightBlocksize = height / mpisizeRow;

                  Int widthLLocal = widthLBlocksize;
                  Int widthRLocal = widthRBlocksize;
                  Int heightLocal = heightBlocksize;

                  if(mpirankRow < (widthL % mpisizeRow)){
                    widthLLocal = widthLBlocksize + 1;
                  }

                  if(mpirankRow < (widthR % mpisizeRow)){
                    widthRLocal = widthRBlocksize + 1;
                  }

                  if(mpirankRow < (height % mpisizeRow)){
                    heightLocal = heightBlocksize + 1;
                  }

                  Int numLGLGridTotal = height;  
                  Int numLGLGridLocal = heightLocal;  

                  Int numBasisLLocal = widthLLocal;
                  Int numBasisRLocal = widthRLocal;

                  DblNumMat valLTempRow( heightLocal, widthL );
                  DblNumMat valRTempRow( heightLocal, widthR );
                  DblNumMat drvLTempRow( heightLocal, widthL );
                  DblNumMat drvRTempRow( heightLocal, widthR );

                  DblNumMat localMatTemp1( widthL, widthR );
                  DblNumMat localMatTemp2( widthL, widthR );
                  DblNumMat localMatTemp3( widthL, widthR );
                  DblNumMat localMatTemp4( widthL, widthR );
                  DblNumMat localMatTemp5( widthL, widthR );

                  AlltoallForward (valLTemp, valLTempRow, domain_.rowComm);
                  AlltoallForward (valRTemp, valRTempRow, domain_.rowComm);
                  AlltoallForward (drvLTemp, drvLTempRow, domain_.rowComm);
                  AlltoallForward (drvRTemp, drvRTempRow, domain_.rowComm);

                  SetValue( localMatTemp1, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, drvLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp1.Data(), numBasisLTotal );

                  SetValue( localMatTemp2, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      drvRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp2.Data(), numBasisLTotal );

                  SetValue( localMatTemp3, 0.0 );
                  blas::Gemm( 'T', 'N', numBasisLTotal, numBasisRTotal, numLGLGridLocal,
                      1.0, valLTempRow.Data(), numLGLGridLocal, 
                      valRTempRow.Data(), numLGLGridLocal, 0.0,
                      localMatTemp3.Data(), numBasisLTotal );

                  SetValue( localMatTemp4, 0.0 );
                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMatTemp4(a,b) = 0.0 - 0.5 * localMatTemp1(a,b) - 0.5 * localMatTemp2(a,b) 
                        + penaltyAlpha_ * localMatTemp3(a,b);  
                    } 

                  SetValue( localMatTemp5, 0.0 );
                  MPI_Allreduce( localMatTemp4.Data(), localMatTemp5.Data(), numBasisLTotal * numBasisRTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );

                  for( Int a = 0; a < numBasisLTotal; a++ )
                    for( Int b = 0; b < numBasisRTotal; b++ ){
                      localMat(a,b) += localMatTemp5(a,b);  
                    } 



                // Add (keyL, keyR) to HMat_
                {
                  ElemMatKey matKey( keyL, keyR );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMat;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMat.Data(), 1,
                        mat.Data(), 1);
                  }
                }

                // Add (keyR, keyL) to HMat_
                {
                  DblNumMat localMatTran;
                  Transpose( localMat, localMatTran );
                  ElemMatKey matKey( keyR, keyL );
                  std::map<ElemMatKey, DblNumMat>::iterator mi = 
                    HMat_.LocalMap().find( matKey );
                  if( mi == HMat_.LocalMap().end() ){
                    HMat_.LocalMap()[matKey] = localMatTran;
                  }
                  else{
                    DblNumMat&  mat = (*mi).second;
                    blas::Axpy( mat.Size(), 1.0, localMatTran.Data(), 1,
                        mat.Data(), 1);
                  }
                }
              } // z-direction
            }
          } // for (i)

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for the inter-element boundary calculation is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }


    // *********************************************************************
    // Collect information and combine HMat_
    // 
    // When intra-element parallelization is invoked, at this stage all
    // processors in the same processor row communicator do the same job,
    // and communication is restricted to each column communicator group.
    // *********************************************************************
    {
      GetTime( timeSta );
      std::vector<ElemMatKey>  keyIdx;
      for( std::map<ElemMatKey, DblNumMat>::iterator 
          mi  = HMat_.LocalMap().begin();
          mi != HMat_.LocalMap().end(); mi++ ){
        ElemMatKey key = (*mi).first;
        if( HMat_.Prtn().Owner(key) != (mpirank / dmRow_) ){
          keyIdx.push_back( key );
        }
      }

      // Communication
      HMat_.PutBegin( keyIdx, NO_MASK );
      HMat_.PutEnd( NO_MASK, PutMode::COMBINE );

      // Clean up
      std::vector<ElemMatKey>  eraseKey;
      for( std::map<ElemMatKey, DblNumMat>::iterator 
          mi  = HMat_.LocalMap().begin();
          mi != HMat_.LocalMap().end(); mi++ ){
        ElemMatKey key = (*mi).first;
        if( HMat_.Prtn().Owner(key) != (mpirank / dmRow_) ){
          eraseKey.push_back( key );
        }
      }
      for( std::vector<ElemMatKey>::iterator vi = eraseKey.begin();
          vi != eraseKey.end(); vi++ ){
        HMat_.LocalMap().erase( *vi );
      }


      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for combining the matrix is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
    }



    return ;
  }         // -----  end of method HamiltonianDG::CalculateDGMatrix_device  ----- 

} // namespace dgdft
