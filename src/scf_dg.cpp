/*
  Copyright (c) 2012 The Regents of the University of California,
  through Lawrence Berkeley National Laboratory.  

  Authors: Lin Lin, Wei Hu and Amartya Banerjee

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
/// @file scf_dg.cpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Add intra-element parallelization.
#include  "scf_dg.hpp"
#include	"blas.hpp"
#include	"lapack.hpp"
#include  "utility.hpp"

#define _DEBUGlevel_ 0

namespace  dgdft{

  using namespace dgdft::DensityComponent;


  // FIXME Leave the smoother function to somewhere more appropriate
  Real
  Smoother ( Real x )
  {
#ifndef _RELEASE_
    PushCallStack("Smoother");
#endif
    Real t, z;
    if( x <= 0 )
      t = 1.0;
    else if( x >= 1 )
      t = 0.0;
    else{
      z = -1.0 / x + 1.0 / (1.0 - x );
      if( z < 0 )
	t = 1.0 / ( std::exp(z) + 1.0 );
      else
	t = std::exp(-z) / ( std::exp(-z) + 1.0 );
    }
#ifndef _RELEASE_
    PopCallStack();
#endif
    return t;
  }		// -----  end of function Smoother  ----- 


  SCFDG::SCFDG	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::SCFDG");
#endif
    isPEXSIInitialized_ = false;
#ifndef _RELEASE_
    PopCallStack();
#endif
  } 		// -----  end of method SCFDG::SCFDG  ----- 


  SCFDG::~SCFDG	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::~SCFDG");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
#ifdef _USE_PEXSI_

    if( isPEXSIInitialized_ == true ){
      Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
      Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
      Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
      Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

      if( mpirankRow < numProcPEXSICommRow_ & mpirankCol < numProcPEXSICommCol_ ){
	Int info;
	PPEXSIPlanFinalize(
			   pexsiPlan_,
			   &info );
	if( info != 0 ){
	  std::ostringstream msg;
	  msg 
	    << "PEXSI finalization returns info " << info << std::endl;
	  throw std::runtime_error( msg.str().c_str() );
	}
      }

      MPI_Comm_free( &pexsiComm_ );
    }
#endif
#ifndef _RELEASE_
    PopCallStack();
#endif
  } 		// -----  end of method SCFDG::~SCFDG  ----- 

  void
  SCFDG::Setup	( 
		 const esdf::ESDFInputParam& esdfParam, 
		 HamiltonianDG&              hamDG,
		 DistVec<Index3, EigenSolver, ElemPrtn>&  distEigSol,
		 DistFourier&                distfft,
		 PeriodTable&                ptable,
		 Int                         contxt	)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::Setup");
#endif
  
    // *********************************************************************
    // Read parameters from ESDFParam
    // *********************************************************************
    // Control parameters
    {
      domain_        = esdfParam.domain;
      mixMaxDim_     = esdfParam.mixMaxDim;
      mixVariable_   = esdfParam.mixVariable;
      mixType_       = esdfParam.mixType;
      mixStepLength_ = esdfParam.mixStepLength;
      eigTolerance_  = esdfParam.eigTolerance;
      eigMinIter_    = esdfParam.eigMinIter;
      eigMaxIter_    = esdfParam.eigMaxIter;
      scfInnerTolerance_  = esdfParam.scfInnerTolerance;
      scfInnerMinIter_    = esdfParam.scfInnerMinIter;
      scfInnerMaxIter_    = esdfParam.scfInnerMaxIter;
      scfOuterTolerance_  = esdfParam.scfOuterTolerance;
      scfOuterMinIter_    = esdfParam.scfOuterMinIter;
      scfOuterMaxIter_    = esdfParam.scfOuterMaxIter;
      scfOuterEnergyTolerance_ = esdfParam.scfOuterEnergyTolerance;
      numUnusedState_ = esdfParam.numUnusedState;
      SVDBasisTolerance_  = esdfParam.SVDBasisTolerance;
      isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
      isRestartDensity_ = esdfParam.isRestartDensity;
      isRestartWfn_     = esdfParam.isRestartWfn;
      isOutputDensity_  = esdfParam.isOutputDensity;
      isOutputALBElemLGL_     = esdfParam.isOutputALBElemLGL;
      isOutputALBElemUniform_ = esdfParam.isOutputALBElemUniform;
      isOutputWfnExtElem_  = esdfParam.isOutputWfnExtElem;
      isOutputPotExtElem_  = esdfParam.isOutputPotExtElem;
      isCalculateAPosterioriEachSCF_ = esdfParam.isCalculateAPosterioriEachSCF;
      isCalculateForceEachSCF_       = esdfParam.isCalculateForceEachSCF;
      isOutputHMatrix_  = esdfParam.isOutputHMatrix;
      solutionMethod_   = esdfParam.solutionMethod;

      Tbeta_            = esdfParam.Tbeta;
      scaBlockSize_     = esdfParam.scaBlockSize;
      numElem_          = esdfParam.numElem;
      ecutWavefunction_ = esdfParam.ecutWavefunction;
      densityGridFactor_= esdfParam.densityGridFactor;
      LGLGridFactor_    = esdfParam.LGLGridFactor;
      isPeriodizePotential_ = esdfParam.isPeriodizePotential;
      distancePeriodize_= esdfParam.distancePeriodize;

      isPotentialBarrier_ = esdfParam.isPotentialBarrier;
      potentialBarrierW_  = esdfParam.potentialBarrierW;
      potentialBarrierS_  = esdfParam.potentialBarrierS;
      potentialBarrierR_  = esdfParam.potentialBarrierR;


      XCType_             = esdfParam.XCType;
      VDWType_            = esdfParam.VDWType;
    }

    // ~~**~~
    // Variables related to Chebyshev Filtered SCF iterations for DG  
    {
   
      Diag_SCFDG_by_Cheby_ = esdfParam.Diag_SCFDG_by_Cheby; // Default: 0
      SCFDG_Cheby_use_ScaLAPACK_ = esdfParam.SCFDG_Cheby_use_ScaLAPACK; // Default: 0
    
      First_SCFDG_ChebyFilterOrder_ = esdfParam.First_SCFDG_ChebyFilterOrder; // Default 60
      First_SCFDG_ChebyCycleNum_ = esdfParam.First_SCFDG_ChebyCycleNum; // Default 5
    
      Second_SCFDG_ChebyOuterIter_ = esdfParam.Second_SCFDG_ChebyOuterIter; // Default = 3
      Second_SCFDG_ChebyFilterOrder_ = esdfParam.Second_SCFDG_ChebyFilterOrder; // Default = 60
      Second_SCFDG_ChebyCycleNum_ = esdfParam.Second_SCFDG_ChebyCycleNum; // Default 3 
    
      General_SCFDG_ChebyFilterOrder_ = esdfParam.General_SCFDG_ChebyFilterOrder; // Default = 60
      General_SCFDG_ChebyCycleNum_ = esdfParam.General_SCFDG_ChebyCycleNum; // Default 1

    }
  
  
    MPI_Barrier(domain_.comm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.rowComm);
    Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
    Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
    dmRow_ = mpisize / dmCol_;
  
    numProcScaLAPACK_ = esdfParam.numProcScaLAPACK;

    // Initialize PEXSI
#ifdef _USE_PEXSI_
    if( solutionMethod_ == "pexsi" )
      {
	Int info;
	// Initialize the PEXSI options
	PPEXSISetDefaultOptions( &pexsiOptions_ );

	pexsiOptions_.temperature      = 1.0 / Tbeta_;
	pexsiOptions_.gap              = esdfParam.energyGap;
	pexsiOptions_.deltaE           = esdfParam.spectralRadius;
	pexsiOptions_.numPole          = esdfParam.numPole;
	pexsiOptions_.isInertiaCount   = 1; 
	pexsiOptions_.maxPEXSIIter     = esdfParam.maxPEXSIIter;
	pexsiOptions_.muMin0           = esdfParam.muMin;
	pexsiOptions_.muMax0           = esdfParam.muMax;
	pexsiOptions_.muInertiaTolerance = 
	  esdfParam.muInertiaTolerance;
	pexsiOptions_.muInertiaExpansion = 
	  esdfParam.muInertiaExpansion;
	pexsiOptions_.muPEXSISafeGuard   = 
	  esdfParam.muPEXSISafeGuard;
	pexsiOptions_.numElectronPEXSITolerance = 
	  esdfParam.numElectronPEXSITolerance;

	muInertiaToleranceTarget_ = esdfParam.muInertiaTolerance;
	numElectronPEXSIToleranceTarget_ = esdfParam.numElectronPEXSITolerance;

	pexsiOptions_.ordering           = esdfParam.matrixOrdering;
	pexsiOptions_.npSymbFact         = esdfParam.npSymbFact;
	pexsiOptions_.verbosity          = 1; // FIXME

	numProcRowPEXSI_     = esdfParam.numProcRowPEXSI;
	numProcColPEXSI_     = esdfParam.numProcColPEXSI;
	inertiaCountSteps_   = esdfParam.inertiaCountSteps;

	//dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
	//dmRow_ = mpisize / dmCol_;

	// Provide a communicator for PEXSI
	numProcPEXSICommCol_ = numProcRowPEXSI_ * numProcColPEXSI_;

	if( numProcPEXSICommCol_ > dmCol_ ){
	  std::ostringstream msg;
	  msg 
	    << "In the current implementation, "
	    << "the number of processors per pole = " << numProcPEXSICommCol_ 
	    << ", and cannot exceed the number of elements = " << dmCol_ 
	    << std::endl;
	  throw std::runtime_error( msg.str().c_str() );
	}
    
	numProcPEXSICommRow_ = std::min( esdfParam.numPole, dmRow_ );

	numProcTotalPEXSI_   = numProcPEXSICommRow_ * numProcPEXSICommCol_;

	Int isProcPEXSI = 0;
	if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
	  isProcPEXSI = 1;

	// Transpose way of ranking MPI processors to be consistent with PEXSI.
	Int mpirankPEXSI = mpirankCol + mpirankRow * ( numProcPEXSICommCol_ );
      
	MPI_Comm_split( domain_.comm, isProcPEXSI, mpirankPEXSI, &pexsiComm_ );


	// Initialize PEXSI
	if( isProcPEXSI ){

#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "mpirank = " << mpirank << ", mpirankPEXSI = " << mpirankPEXSI << std::endl;
#endif

	  // FIXME More versatile control of the output of the PEXSI module
	  Int outputFileIndex = mpirank;

	  pexsiPlan_        = PPEXSIPlanInitialize(
						   pexsiComm_,
						   numProcRowPEXSI_,
						   numProcColPEXSI_,
						   outputFileIndex,
						   &info );
	  if( info != 0 ){
	    std::ostringstream msg;
	    msg 
	      << "PEXSI initialization returns info " << info << std::endl;
	    throw std::runtime_error( msg.str().c_str() );
	  }
	}
      }
#endif


    // other SCFDG parameters
    {
      hamDGPtr_      = &hamDG;
      distEigSolPtr_ = &distEigSol;
      distfftPtr_    = &distfft;
      ptablePtr_     = &ptable;
      elemPrtn_      = distEigSol.Prtn();
      contxt_        = contxt;


      distDMMat_.SetComm(domain_.colComm);
      distEDMMat_.SetComm(domain_.colComm);
      distFDMMat_.SetComm(domain_.colComm);

      //distEigSolPtr_.SetComm(domain_.colComm);
      //distfftPtr_.SetComm(domain_.colComm);


      mixOuterSave_.SetComm(domain_.colComm);
      mixInnerSave_.SetComm(domain_.colComm);
      dfOuterMat_.SetComm(domain_.colComm);
      dvOuterMat_.SetComm(domain_.colComm);
      dfInnerMat_.SetComm(domain_.colComm);
      dvInnerMat_.SetComm(domain_.colComm);
      vtotLGLSave_.SetComm(domain_.colComm);


      mixOuterSave_.Prtn()  = elemPrtn_;
      mixInnerSave_.Prtn()  = elemPrtn_;
      dfOuterMat_.Prtn()    = elemPrtn_;
      dvOuterMat_.Prtn()    = elemPrtn_;
      dfInnerMat_.Prtn()    = elemPrtn_;
      dvInnerMat_.Prtn()    = elemPrtn_;
      vtotLGLSave_.Prtn()   = elemPrtn_;

#ifdef _USE_PEXSI_
      distDMMat_.Prtn()     = hamDG.HMat().Prtn();
      distEDMMat_.Prtn()     = hamDG.HMat().Prtn();
      distFDMMat_.Prtn()     = hamDG.HMat().Prtn(); 
#endif


      // The number of processors in the column communicator must be the
      // number of elements, and mpisize should be a multiple of the
      // number of elements.
      if( (mpisize % dmCol_) != 0 ){
	statusOFS << "mpisize = " << mpisize << " mpirank = " << mpirank << std::endl;
	statusOFS << "dmCol_ = " << dmCol_ << " dmRow_ = " << dmRow_ << std::endl;
	std::ostringstream msg;
	msg << "Total number of processors do not fit to the number processors per element." << std::endl;
	throw std::runtime_error( msg.str().c_str() );
      }


      // FIXME fixed ratio between the size of the extended element and
      // the element
      for( Int d = 0; d < DIM; d++ ){
	extElemRatio_[d] = ( numElem_[d]>1 ) ? 3 : 1;
      }

      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
	      SetValue( emptyVec, 0.0 );
	      mixOuterSave_.LocalMap()[key] = emptyVec;
	      mixInnerSave_.LocalMap()[key] = emptyVec;
	      DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
	      SetValue( emptyMat, 0.0 );
	      dfOuterMat_.LocalMap()[key]   = emptyMat;
	      dvOuterMat_.LocalMap()[key]   = emptyMat;
	      dfInnerMat_.LocalMap()[key]   = emptyMat;
	      dvInnerMat_.LocalMap()[key]   = emptyMat;

	      DblNumVec  emptyLGLVec( hamDG.NumLGLGridElem().prod() );
	      SetValue( emptyLGLVec, 0.0 );
	      vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
	    } // own this element
	  }  // for (i)


      // Restart the density in the global domain
      restartDensityFileName_ = "DEN";
      // Restart the wavefunctions in the extended element
      restartWfnFileName_     = "WFNEXT";
    }

    // *********************************************************************
    // Initialization
    // *********************************************************************

    // Density
    DistDblNumVec&  density = hamDGPtr_->Density();

    density.SetComm(domain_.colComm);

    if( isRestartDensity_ ) {
      // Only the first processor column reads the matrix

#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Restarting density from DEN_ files." << std::endl;
#endif

      if( mpirankRow == 0 ){
	std::istringstream rhoStream;      
	SeparateRead( restartDensityFileName_, rhoStream, mpirankCol );

	Real sumDensityLocal = 0.0, sumDensity = 0.0;

	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		std::vector<DblNumVec> gridpos(DIM);
              
		// Dummy variables and not used
		for( Int d = 0; d < DIM; d++ ){
		  deserialize( gridpos[d], rhoStream, NO_MASK );
		}

		Index3 keyRead;
		deserialize( keyRead, rhoStream, NO_MASK );
		if( keyRead[0] != key[0] ||
		    keyRead[1] != key[1] ||
		    keyRead[2] != key[2] ){
		  std::ostringstream msg;
		  msg 
		    << "Mpirank " << mpirank << " is reading the wrong file."
		    << std::endl
		    << "key     ~ " << key << std::endl
		    << "keyRead ~ " << keyRead << std::endl;
		  throw std::runtime_error( msg.str().c_str() );
		}

		DblNumVec   denVecRead;
		DblNumVec&  denVec = density.LocalMap()[key];
		deserialize( denVecRead, rhoStream, NO_MASK );
		if( denVecRead.Size() != denVec.Size() ){
		  std::ostringstream msg;
		  msg 
		    << "The size of restarting density does not match with the current setup."  
		    << std::endl
		    << "input density size   ~ " << denVecRead.Size() << std::endl
		    << "current density size ~ " << denVec.Size()     << std::endl;
		  throw std::runtime_error( msg.str().c_str() );
		}
		denVec = denVecRead;
		for( Int p = 0; p < denVec.Size(); p++ ){
		  sumDensityLocal += denVec(p);
		}
	      }
	    } // for (i)

	// Rescale the density
	mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
			domain_.colComm );

	Print( statusOFS, "Restart density. Sum of density      = ", 
	       sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
      }

      // Broadcast the density to the column
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumVec&  denVec = density.LocalMap()[key];
	      MPI_Bcast( denVec.Data(), denVec.Size(), MPI_DOUBLE, 0, domain_.rowComm );
	    }
	  }

    } // else using the zero initial guess
    else {
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Generating initial density through linear combination of pseudocharges." 
		<< std::endl;
#endif

      // Initialize the electron density using the pseudocharge
      // make sure the pseudocharge is initialized
      DistDblNumVec& pseudoCharge = hamDGPtr_->PseudoCharge();

      pseudoCharge.SetComm(domain_.colComm);

      Real sumDensityLocal = 0.0, sumPseudoChargeLocal = 0.0;
      Real sumDensity, sumPseudoCharge;
      Real EPS = 1e-6;

      // make sure that the electron density is positive
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumVec&  denVec = density.LocalMap()[key];
	      DblNumVec&  ppVec  = pseudoCharge.LocalMap()[key];
	      for( Int p = 0; p < denVec.Size(); p++ ){
		denVec(p) = ( ppVec(p) > EPS ) ? ppVec(p) : EPS;
		sumDensityLocal += denVec(p);
		sumPseudoChargeLocal += ppVec(p);
	      }
	    }
	  } // for (i)

      // Rescale the density
      mpi::Allreduce( &sumDensityLocal, &sumDensity, 1, MPI_SUM,
		      domain_.colComm );
      mpi::Allreduce( &sumPseudoChargeLocal, &sumPseudoCharge, 
		      1, MPI_SUM, domain_.colComm );

      Print( statusOFS, "Initial density. Sum of density      = ", 
	     sumDensity * domain_.Volume() / domain_.NumGridTotalFine() );
#if ( _DEBUGlevel_ >= 1 )
      Print( statusOFS, "Sum of pseudo charge        = ", 
	     sumPseudoCharge * domain_.Volume() / domain_.NumGridTotalFine() );
#endif

      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumVec&  denVec = density.LocalMap()[key];
	      blas::Scal( denVec.Size(), sumPseudoCharge / sumDensity, 
			  denVec.Data(), 1 );
	    }
	  } // for (i)
    } // Restart the density


    // Wavefunctions in the extended element
    if( isRestartWfn_ ){
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Restarting basis functions from WFNEXT_ files"
		<< std::endl;
#endif
      std::istringstream wfnStream;      
      SeparateRead( restartWfnFileName_, wfnStream, mpirank );

      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	      Spinor& psi = eigSol.Psi();

	      DblNumTns& wavefun = psi.Wavefun();
	      DblNumTns  wavefunRead;

	      std::vector<DblNumVec> gridpos(DIM);
	      for( Int d = 0; d < DIM; d++ ){
		deserialize( gridpos[d], wfnStream, NO_MASK );
	      }

	      Index3 keyRead;
	      deserialize( keyRead, wfnStream, NO_MASK );
	      if( keyRead[0] != key[0] ||
		  keyRead[1] != key[1] ||
		  keyRead[2] != key[2] ){
		std::ostringstream msg;
		msg 
		  << "Mpirank " << mpirank << " is reading the wrong file."
		  << std::endl
		  << "key     ~ " << key << std::endl
		  << "keyRead ~ " << keyRead << std::endl;
		throw std::runtime_error( msg.str().c_str() );
	      }
	      deserialize( wavefunRead, wfnStream, NO_MASK );

	      if( wavefunRead.Size() != wavefun.Size() ){
		std::ostringstream msg;
		msg 
		  << "The size of restarting basis function does not match with the current setup."  
		  << std::endl
		  << "input basis size   ~ " << wavefunRead.Size() << std::endl
		  << "current basis size ~ " << wavefun.Size()     << std::endl;
		throw std::runtime_error( msg.str().c_str() );
	      }

	      wavefun = wavefunRead;
	    }
	  } // for (i)

    } 
    else{ 
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Initial random basis functions in the extended element."
		<< std::endl;
#endif

      // Use random initial guess for basis functions in the extended element.
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	      Spinor& psi = eigSol.Psi();

	      UniformRandom( psi.Wavefun() );


	      // For debugging purpose
	      // Make sure that the initial wavefunctions in each element
	      // are the same, when different number of processors are
	      // used for intra-element parallelization.
	      if(0){ 
		Spinor  psiTemp;
		psiTemp.Setup( eigSol.FFT().domain, 1, psi.NumStateTotal(), psi.NumStateTotal(), 0.0 );

		Int mpirankp, mpisizep;
		MPI_Comm_rank( domain_.rowComm, &mpirankp );
		MPI_Comm_size( domain_.rowComm, &mpisizep );

		if (mpirankp == 0){
		  SetRandomSeed(1);
		  UniformRandom( psiTemp.Wavefun() );
		}
		MPI_Bcast(psiTemp.Wavefun().Data(), psiTemp.Wavefun().m()*psiTemp.Wavefun().n()*psiTemp.Wavefun().p(), MPI_DOUBLE, 0, domain_.rowComm);

		Int size = psi.Wavefun().m() * psi.Wavefun().n();
		Int nocc = psi.Wavefun().p();

		IntNumVec& wavefunIdx = psi.WavefunIdx();
		NumTns<Scalar>& wavefun = psi.Wavefun();

		for (Int k=0; k<nocc; k++) {
		  Scalar *ptr = psi.Wavefun().MatData(k);
		  Scalar *ptr1 = psiTemp.Wavefun().MatData(wavefunIdx(k));
		  for (Int i=0; i<size; i++) {
		    *ptr = *ptr1;
		    ptr = ptr + 1;
		    ptr1 = ptr1 + 1;
		  }
		}
	      }//if(0)
	    }
	  } // for (i)
      Print( statusOFS, "Initial basis functions with random guess." );
    } // if (isRestartWfn_)




    // Generate the transfer matrix from the periodic uniform grid on each
    // extended element to LGL grid.  
    // 05/06/2015:
    // Based on the new understanding of the dual grid treatment, the
    // interpolation must be performed through a fine Fourier grid
    // (uniform grid) and then interpolate to the LGL grid.
    {
      PeriodicUniformToLGLMat_.resize(DIM);
      PeriodicUniformFineToLGLMat_.resize(DIM);
      PeriodicGridExtElemToGridElemMat_.resize(DIM);

      EigenSolver& eigSol = (*distEigSol.LocalMap().begin()).second;
      Domain dmExtElem = eigSol.FFT().domain;
      Domain dmElem;
      for( Int d = 0; d < DIM; d++ ){
	dmElem.length[d]   = domain_.length[d] / numElem_[d];
	dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
	dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
	// PosStart relative to the extended element 
	dmExtElem.posStart[d] = 0.0;
	dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
      }

      Index3 numLGL        = hamDG.NumLGLGridElem();
      Index3 numUniform    = dmExtElem.numGrid;
      Index3 numUniformFine    = dmExtElem.numGridFine;
      Index3 numUniformFineElem    = dmElem.numGridFine;
      Point3 lengthUniform = dmExtElem.length;

      std::vector<DblNumVec>  LGLGrid(DIM);
      LGLMesh( dmElem, numLGL, LGLGrid ); 
      std::vector<DblNumVec>  UniformGrid(DIM);
      UniformMesh( dmExtElem, UniformGrid );
      std::vector<DblNumVec>  UniformGridFine(DIM);
      UniformMeshFine( dmExtElem, UniformGridFine );
      std::vector<DblNumVec>  UniformGridFineElem(DIM);
      UniformMeshFine( dmElem, UniformGridFineElem );

      for( Int d = 0; d < DIM; d++ ){
	DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
	DblNumMat&  localMatFineElem = PeriodicGridExtElemToGridElemMat_[d];
	localMat.Resize( numLGL[d], numUniform[d] );
	localMatFineElem.Resize( numUniformFineElem[d], numUniform[d] );
	SetValue( localMat, 0.0 );
	SetValue( localMatFineElem, 0.0 );
	DblNumVec KGrid( numUniform[d] );
	for( Int i = 0; i <= numUniform[d] / 2; i++ ){
	  KGrid(i) = i * 2.0 * PI / lengthUniform[d];
	}
	for( Int i = numUniform[d] / 2 + 1; i < numUniform[d]; i++ ){
	  KGrid(i) = ( i - numUniform[d] ) * 2.0 * PI / lengthUniform[d];
	}

	for( Int j = 0; j < numUniform[d]; j++ ){
     

	  for( Int i = 0; i < numLGL[d]; i++ ){
	    localMat(i, j) = 0.0;
	    for( Int k = 0; k < numUniform[d]; k++ ){
	      localMat(i,j) += std::cos( KGrid(k) * ( LGLGrid[d](i) -
						      UniformGrid[d](j) ) ) / numUniform[d];
	    } // for (k)
	  } // for (i)
   
	  for( Int i = 0; i < numUniformFineElem[d]; i++ ){
	    localMatFineElem(i, j) = 0.0;
	    for( Int k = 0; k < numUniform[d]; k++ ){
	      localMatFineElem(i,j) += std::cos( KGrid(k) * ( UniformGridFineElem[d](i) -
							      UniformGrid[d](j) ) ) / numUniform[d];
	    } // for (k)
	  } // for (i)
   
	} // for (j)
 
      } // for (d)


      for( Int d = 0; d < DIM; d++ ){
	DblNumMat&  localMatFine = PeriodicUniformFineToLGLMat_[d];
	localMatFine.Resize( numLGL[d], numUniformFine[d] );
	SetValue( localMatFine, 0.0 );
	DblNumVec KGridFine( numUniformFine[d] );
	for( Int i = 0; i <= numUniformFine[d] / 2; i++ ){
	  KGridFine(i) = i * 2.0 * PI / lengthUniform[d];
	}
	for( Int i = numUniformFine[d] / 2 + 1; i < numUniformFine[d]; i++ ){
	  KGridFine(i) = ( i - numUniformFine[d] ) * 2.0 * PI / lengthUniform[d];
	}

	for( Int j = 0; j < numUniformFine[d]; j++ ){
        
	  for( Int i = 0; i < numLGL[d]; i++ ){
	    localMatFine(i, j) = 0.0;
	    for( Int k = 0; k < numUniformFine[d]; k++ ){
	      localMatFine(i,j) += std::cos( KGridFine(k) * ( LGLGrid[d](i) -
							      UniformGridFine[d](j) ) ) / numUniformFine[d];
	    } // for (k)
	  } // for (i)
      
	} // for (j)
      } // for (d)

      // Assume the initial error is O(1)
      scfOuterNorm_ = 1.0;
      scfInnerNorm_ = 1.0;

#if ( _DEBUGlevel_ >= 2 )
      statusOFS << "PeriodicUniformToLGLMat[0] = "
		<< PeriodicUniformToLGLMat_[0] << std::endl;
      statusOFS << "PeriodicUniformToLGLMat[1] = " 
		<< PeriodicUniformToLGLMat_[1] << std::endl;
      statusOFS << "PeriodicUniformToLGLMat[2] = "
		<< PeriodicUniformToLGLMat_[2] << std::endl;
      statusOFS << "PeriodicUniformFineToLGLMat[0] = "
		<< PeriodicUniformFineToLGLMat_[0] << std::endl;
      statusOFS << "PeriodicUniformFineToLGLMat[1] = " 
		<< PeriodicUniformFineToLGLMat_[1] << std::endl;
      statusOFS << "PeriodicUniformFineToLGLMat[2] = "
		<< PeriodicUniformFineToLGLMat_[2] << std::endl;
      statusOFS << "PeriodicGridExtElemToGridElemMat[0] = "
		<< PeriodicGridExtElemToGridElemMat_[0] << std::endl;
      statusOFS << "PeriodicGridExtElemToGridElemMat[1] = "
		<< PeriodicGridExtElemToGridElemMat_[1] << std::endl;
      statusOFS << "PeriodicGridExtElemToGridElemMat[2] = "
		<< PeriodicGridExtElemToGridElemMat_[2] << std::endl;
#endif
    }
  
    // Whether to apply potential barrier in the extended element. CANNOT
    // be used together with periodization option
    if( isPotentialBarrier_ ) {
      vBarrier_.resize(DIM);
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
	      std::vector<DblNumVec> gridpos(DIM);
	      UniformMeshFine ( dmExtElem, gridpos );

	      for( Int d = 0; d < DIM; d++ ){
		Real length   = dmExtElem.length[d];
		Int numGridFine   = dmExtElem.numGridFine[d];
		Real posStart = dmExtElem.posStart[d]; 
		Real center   = posStart + length / 2.0;
              
		// FIXME
		Real EPS      = 1.0;           // For stability reason
		Real dist;

		vBarrier_[d].Resize( numGridFine );
		SetValue( vBarrier_[d], 0.0 );
		for( Int p = 0; p < numGridFine; p++ ){
		  dist = std::abs( gridpos[d][p] - center );
		  // Only apply the barrier for region outside barrierR
		  if( dist > potentialBarrierR_){
		    vBarrier_[d][p] = potentialBarrierS_* std::exp( - potentialBarrierW_ / 
								    ( dist - potentialBarrierR_ ) ) / std::pow( dist - length / 2.0 - EPS, 2.0 );
		  }
		}
	      } // for (d)

#if ( _DEBUGlevel_ >= 0  )
	      statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
	      statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
	      statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
	      statusOFS << "vBarrier[0] = " << std::endl << vBarrier_[0] << std::endl;
	      statusOFS << "vBarrier[1] = " << std::endl << vBarrier_[1] << std::endl;
	      statusOFS << "vBarrier[2] = " << std::endl << vBarrier_[2] << std::endl;
#endif
            
	    } // own this element
	  } // for (k)
    }


    // Whether to periodize the potential in the extended element. CANNOT
    // be used together with barrier option.
    if( isPeriodizePotential_ ){
      vBubble_.resize(DIM);
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      Domain& dmExtElem = distEigSolPtr_->LocalMap()[key].FFT().domain;
	      std::vector<DblNumVec> gridpos(DIM);
	      UniformMeshFine ( dmExtElem, gridpos );

	      for( Int d = 0; d < DIM; d++ ){
		Real length   = dmExtElem.length[d];
		Int numGridFine   = dmExtElem.numGridFine[d];
		Real posStart = dmExtElem.posStart[d]; 
		// FIXME
		Real EPS = 0.2; // Criterion for distancePeriodize_
		vBubble_[d].Resize( numGridFine );
		SetValue( vBubble_[d], 1.0 );

		if( distancePeriodize_[d] > EPS ){
		  Real lb = posStart + distancePeriodize_[d];
		  Real rb = posStart + length - distancePeriodize_[d];
		  for( Int p = 0; p < numGridFine; p++ ){
		    if( gridpos[d][p] > rb ){
		      vBubble_[d][p] = Smoother( (gridpos[d][p] - rb ) / 
						 (distancePeriodize_[d] - EPS) );
		    }

		    if( gridpos[d][p] < lb ){
		      vBubble_[d][p] = Smoother( (lb - gridpos[d][p] ) / 
						 (distancePeriodize_[d] - EPS) );
		    }
		  }
		}
	      } // for (d)

#if ( _DEBUGlevel_ >= 0  )
	      statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
	      statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
	      statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
	      statusOFS << "vBubble[0] = " << std::endl << vBubble_[0] << std::endl;
	      statusOFS << "vBubble[1] = " << std::endl << vBubble_[1] << std::endl;
	      statusOFS << "vBubble[2] = " << std::endl << vBubble_[2] << std::endl;
#endif
	    } // own this element
	  } // for (k)
    }



#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::Setup  ----- 

  void
  SCFDG::Update	( )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::Update");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG& hamDG = *hamDGPtr_;

    {
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumVec  emptyVec( hamDG.NumUniformGridElemFine().prod() );
	      SetValue( emptyVec, 0.0 );
	      mixOuterSave_.LocalMap()[key] = emptyVec;
	      mixInnerSave_.LocalMap()[key] = emptyVec;
	      DblNumMat  emptyMat( hamDG.NumUniformGridElemFine().prod(), mixMaxDim_ );
	      SetValue( emptyMat, 0.0 );
	      dfOuterMat_.LocalMap()[key]   = emptyMat;
	      dvOuterMat_.LocalMap()[key]   = emptyMat;
	      dfInnerMat_.LocalMap()[key]   = emptyMat;
	      dvInnerMat_.LocalMap()[key]   = emptyMat;

	      DblNumVec  emptyLGLVec( hamDG.NumLGLGridElem().prod() );
	      SetValue( emptyLGLVec, 0.0 );
	      vtotLGLSave_.LocalMap()[key] = emptyLGLVec;
	    } // own this element
	  }  // for (i)


    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::Update  ----- 


  void
  SCFDG::Iterate	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::Iterate");
#endif

    MPI_Barrier(domain_.comm);
    MPI_Barrier(domain_.colComm);
    MPI_Barrier(domain_.rowComm);

    Int mpirank; MPI_Comm_rank( domain_.comm, &mpirank );
    Int mpisize; MPI_Comm_size( domain_.comm, &mpisize );

    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);

    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    Real timeSta, timeEnd;

    Domain dmElem;
    for( Int d = 0; d < DIM; d++ ){
      dmElem.length[d]   = domain_.length[d] / numElem_[d];
      dmElem.numGrid[d]  = domain_.numGrid[d] / numElem_[d];
      dmElem.numGridFine[d]  = domain_.numGridFine[d] / numElem_[d];
      dmElem.posStart[d] = ( numElem_[d] > 1 ) ? dmElem.length[d] : 0;
    }

    HamiltonianDG&  hamDG = *hamDGPtr_;

    if( XCType_ == "XC_GGA_XC_PBE" ){
      hamDG.CalculateGradDensity(  *distfftPtr_ );
    }

    // Compute the exchange-correlation potential and energy
    GetTime( timeSta );
    hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calculating XC is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    // Compute the Hartree potential
    GetTime( timeSta );
    hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calculating Hartree is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
 
    // No external potential

    // Compute the total potential
    GetTime( timeSta );
    hamDG.CalculateVtot( hamDG.Vtot() );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for calculating Vtot is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    Real timeIterStart(0), timeIterEnd(0);
    Real timeTotalStart(0), timeTotalEnd(0);

    bool isSCFConverged = false;

    scfTotalInnerIter_  = 0;

    GetTime( timeTotalStart );

    Int iter;

    // Total number of SVD basis functions. Determined at the first
    // outer SCF and is not changed later. This facilitates the reuse of
    // symbolic factorization
    Int numSVDBasisTotal;	

    // ~~**~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // FIXME: Assuming spinor has only one component here    
    // Deque for ALBs expressed on the LGL grid
    std::deque<DblNumMat> ALB_LGL_deque;

  
  
  
    for (iter=1; iter <= scfOuterMaxIter_; iter++) {
      if ( isSCFConverged && (iter >= scfOuterMinIter_ ) ) break;
		
		
      // Performing each iteartion
      {
	std::ostringstream msg;
	msg << "Outer SCF iteration # " << iter;
	PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );
		
      // *********************************************************************
      // Update the local potential in the extended element and the element.
      //
      // NOTE: The modification of the potential on the extended element
      // to reduce the Gibbs phenomena is now in UpdateElemLocalPotential
      // *********************************************************************
      {
	GetTime(timeSta);

	UpdateElemLocalPotential();

	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for updating the local potential in the extended element and the element is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

		
   
      // *********************************************************************
      // Solve the basis functions in the extended element
      // *********************************************************************

      Real timeBasisSta, timeBasisEnd;
      GetTime(timeBasisSta);
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	      DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();
	      Index3 numGridExtElem = eigSol.FFT().domain.numGrid;
	      Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;
	      Index3 numLGLGrid     = hamDG.NumLGLGridElem();

	      Index3 numGridElemFine    = dmElem.numGridFine;
						
	      // Skip the interpoation if there is no adaptive local
	      // basis function.  
	      if( eigSol.Psi().NumState() == 0 ){
		hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
		hamDG.BasisUniformFine().LocalMap()[key].Resize( numGridElemFine.prod(), 0 );  
		continue;
	      }

	      // Solve the basis functions in the extended element
  
	      Real eigTolNow;
	      if( isEigToleranceDynamic_ ){
		// Dynamic strategy to control the tolerance
		if( iter == 1 )
		  // FIXME magic number
		  eigTolNow = 1e-2;
		else
		  eigTolNow = eigTolerance_;
	      }
	      else{
		// Static strategy to control the tolerance
		eigTolNow = eigTolerance_;
	      }

#if ( _DEBUGlevel_ >= 0 ) 
	      Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
	      statusOFS << "The current tolerance used by the eigensolver is " 
			<< eigTolNow << std::endl;
	      statusOFS << "The target number of converged eigenvectors is " 
			<< numEig << std::endl;
#endif
           
	      GetTime( timeSta );
	      // FIXME multiple choices of solvers for the extended
	      // element should be given in the input file
	      if(0){
		eigSol.Solve();
	      }
	      if(1){
		eigSol.LOBPCGSolveReal2(numEig, eigMaxIter_, eigTolNow );
	      }
	      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	      statusOFS << "Eigensolver time = " 	<< timeEnd - timeSta
			<< " [s]" << std::endl;
#endif


	      // Print out the information
	      statusOFS << std::endl 
			<< "ALB calculation in extended element " << key << std::endl;
	      Real maxRes = 0.0, avgRes = 0.0;
	      for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
		if( maxRes < eigSol.ResVal()(ii) ){
		  maxRes = eigSol.ResVal()(ii);
		}
		avgRes = avgRes + eigSol.ResVal()(ii);
#if ( _DEBUGlevel_ >= 0 )
		Print(statusOFS, 
		      "basis#   = ", ii, 
		      "eigval   = ", eigSol.EigVal()(ii),
		      "resval   = ", eigSol.ResVal()(ii));
#endif
	      }
	      avgRes = avgRes / eigSol.EigVal().m();
#if ( _DEBUGlevel_ >= 0 )
	      statusOFS << std::endl;
	      Print(statusOFS, "Max residual of basis = ", maxRes );
	      Print(statusOFS, "Avg residual of basis = ", avgRes );
	      statusOFS << std::endl;
#endif

	      GetTime( timeSta );
	      Spinor& psi = eigSol.Psi();

	      // Assuming that wavefun has only 1 component.  This should
	      // be changed when spin-polarization is added.
	      DblNumTns& wavefun = psi.Wavefun();

	      DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
	      DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );

	      Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
	      for( Int i = 0; i < numLGLGrid.prod(); i++ ){
		*(ptr2++) = std::sqrt( *(ptr1++) );
	      }

	      // Int numBasis = psi.NumState() + 1;
	      Int numBasis = psi.NumState();
	      Int numBasisTotal = psi.NumStateTotal();
	      Int numBasisLocal = numBasis;
	      Int numBasisTotalTest = 0;

	      mpi::Allreduce( &numBasis, &numBasisTotalTest, 1, MPI_SUM, domain_.rowComm );
	      if( numBasisTotalTest != numBasisTotal ){
		statusOFS << "numBasisTotal = " << numBasisTotal << std::endl;
		statusOFS << "numBasisTotalTest = " << numBasisTotalTest << std::endl;
		throw std::logic_error("Sum{numBasis} = numBasisTotal does not match.");
	      }

	      // FIXME The constant mode is now not used.
	      DblNumMat localBasis( 
				   numLGLGrid.prod(), 
				   numBasis );

	      SetValue( localBasis, 0.0 );

#ifdef _USE_OPENMP_
#pragma omp parallel
	      {
#endif
#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
		for( Int l = 0; l < psi.NumState(); l++ ){
		  InterpPeriodicUniformToLGL( 
					     numGridExtElem,
					     numLGLGrid,
					     wavefun.VecData(0, l), 
					     localBasis.VecData(l) );
		}



		//#ifdef _USE_OPENMP_
		//#pragma omp for schedule (dynamic,1) nowait
		//#endif
		//              for( Int l = 0; l < psi.NumState(); l++ ){
		//                // FIXME Temporarily removing the mean value from each
		//                // basis function and add the constant mode later
		//                Real avg = blas::Dot( numLGLGrid.prod(),
		//                    localBasis.VecData(l), 1,
		//                    LGLWeight3D.Data(), 1 );
		//                avg /= ( domain_.Volume() / numElem_.prod() );
		//                for( Int p = 0; p < numLGLGrid.prod(); p++ ){
		//                  localBasis(p, l) -= avg;
		//                }
		//              }
		//
		//              // FIXME Temporary adding the constant mode. Should be done more systematically later.
		//              for( Int p = 0; p < numLGLGrid.prod(); p++ ){
		//                localBasis(p,psi.NumState()) = 1.0 / std::sqrt( domain_.Volume() / numElem_.prod() );
		//              }

#ifdef _USE_OPENMP_
	      }
#endif
	      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	      statusOFS << "Time for interpolating basis = " 	<< timeEnd - timeSta
			<< " [s]" << std::endl;
#endif


	      // Post processing for the basis functions on the LGL grid.
	      // Perform GEMM and threshold the basis functions for the
	      // small matrix.
	      //
	      // This method might have lower numerical accuracy, but is
	      // much more scalable than other known options.
	      if(1){

		GetTime( timeSta );

		// Scale the basis functions by sqrt(weight).  This
		// allows the consequent SVD decomposition of the form
		//
		// X' * W * X
		for( Int g = 0; g < localBasis.n(); g++ ){
		  Real *ptr1 = localBasis.VecData(g);
		  Real *ptr2 = sqrtLGLWeight3D.Data();
		  for( Int l = 0; l < localBasis.m(); l++ ){
		    *(ptr1++)  *= *(ptr2++);
		  }
		}

		// Convert the column partition to row partition

		Int height = psi.NumGridTotal() * psi.NumComponent();
		Int heightLGL = numLGLGrid.prod();
		Int heightElem = numGridElemFine.prod();
		Int width = psi.NumStateTotal();

		Int widthBlocksize = width / mpisizeRow;
		Int heightBlocksize = height / mpisizeRow;
		Int heightLGLBlocksize = heightLGL / mpisizeRow;
		Int heightElemBlocksize = heightElem / mpisizeRow;

		Int widthLocal = widthBlocksize;
		Int heightLocal = heightBlocksize;
		Int heightLGLLocal = heightLGLBlocksize;
		Int heightElemLocal = heightElemBlocksize;

		if(mpirankRow < (width % mpisizeRow)){
		  widthLocal = widthBlocksize + 1;
		}

		if(mpirankRow == (mpisizeRow - 1)){
		  heightLocal = heightBlocksize + height % mpisizeRow;
		}

		if(mpirankRow == (mpisizeRow - 1)){
		  heightLGLLocal = heightLGLBlocksize + heightLGL % mpisizeRow;
		}

		if(mpirankRow == (mpisizeRow - 1)){
		  heightElemLocal = heightElemBlocksize + heightElem % mpisizeRow;
		}

		// FIXME Use AlltoallForward and AlltoallBackward
		// functions to replace below

		DblNumMat MMat( numBasisTotal, numBasisTotal );
		DblNumMat MMatTemp( numBasisTotal, numBasisTotal );
		SetValue( MMat, 0.0 );
		SetValue( MMatTemp, 0.0 );
		Int numLGLGridTotal = numLGLGrid.prod();
		Int numLGLGridLocal = heightLGLLocal;

		DblNumMat localBasisRow(heightLGLLocal, numBasisTotal );
		SetValue( localBasisRow, 0.0 );

		//DblNumMat localBasisElemRow(heightElemLocal, numBasisTotal );
		//SetValue( localBasisElemRow, 0.0 );

		AlltoallForward (localBasis, localBasisRow, domain_.rowComm);

		//DblNumMat& localBasisUniformGrid = hamDG.BasisUniformFine().LocalMap()[key];

		//AlltoallForward (localBasisUniformGrid, localBasisElemRow, domain_.rowComm);

		SetValue( MMatTemp, 0.0 );
		blas::Gemm( 'T', 'N', numBasisTotal, numBasisTotal, numLGLGridLocal,
			    1.0, localBasisRow.Data(), numLGLGridLocal, 
			    localBasisRow.Data(), numLGLGridLocal, 0.0,
			    MMatTemp.Data(), numBasisTotal );


		SetValue( MMat, 0.0 );
		MPI_Allreduce( MMatTemp.Data(), MMat.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, MPI_SUM, domain_.rowComm );


		// The following operation is only performed on the
		// master processor in the row communicator

		DblNumMat    U( numBasisTotal, numBasisTotal );
		DblNumMat   VT( numBasisTotal, numBasisTotal );
		DblNumVec    S( numBasisTotal );
		SetValue(U, 0.0);
		SetValue(VT, 0.0);
		SetValue(S, 0.0);

		MPI_Barrier( domain_.rowComm );

		if ( mpirankRow == 0) {
		  lapack::QRSVD( numBasisTotal, numBasisTotal, 
				 MMat.Data(), numBasisTotal,
				 S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );
		} 

		// Broadcast U and S
		MPI_Bcast(S.Data(), numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
		MPI_Bcast(U.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
		MPI_Bcast(VT.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);

		MPI_Barrier( domain_.rowComm );

		for( Int g = 0; g < numBasisTotal; g++ ){
		  S[g] = std::sqrt( S[g] );
		}

		// Total number of SVD basis functions. NOTE: Determined at the first
		// outer SCF and is not changed later. This facilitates the reuse of
		// symbolic factorization
		if( iter == 1 ){
		  numSVDBasisTotal = 0;	
		  for( Int g = 0; g < numBasisTotal; g++ ){
		    if( S[g] / S[0] > SVDBasisTolerance_ )
		      numSVDBasisTotal++;
		  }
		}
		else{
		  // Reuse the value saved in numSVDBasisTotal
		  statusOFS 
		    << "NOTE: The number of basis functions (after SVD) " 
		    << "is the same as the number in the first SCF iteration." << std::endl
		    << "This facilitates the reuse of symbolic factorization in PEXSI." 
		    << std::endl;
		}

		Int numSVDBasisBlocksize = numSVDBasisTotal / mpisizeRow;

		Int numSVDBasisLocal = numSVDBasisBlocksize;	

		if(mpirankRow < (numSVDBasisTotal % mpisizeRow)){
		  numSVDBasisLocal = numSVDBasisBlocksize + 1;
		}

		Int numSVDBasisTotalTest = 0;

		mpi::Allreduce( &numSVDBasisLocal, &numSVDBasisTotalTest, 1, MPI_SUM, domain_.rowComm );

		if( numSVDBasisTotal != numSVDBasisTotalTest ){
		  statusOFS << "numSVDBasisLocal = " << numSVDBasisLocal << std::endl;
		  statusOFS << "numSVDBasisTotal = " << numSVDBasisTotal << std::endl;
		  statusOFS << "numSVDBasisTotalTest = " << numSVDBasisTotalTest << std::endl;
		  throw std::logic_error("numSVDBasisTotal != numSVDBasisTotalTest");
		}

		// Multiply X <- X*U in the row-partitioned format
		// Get the first numSVDBasis which are significant.

		DblNumMat& basis = hamDG.BasisLGL().LocalMap()[key];

		basis.Resize( numLGLGridTotal, numSVDBasisLocal );
		DblNumMat basisRow( numLGLGridLocal, numSVDBasisTotal );

		SetValue( basis, 0.0 );
		SetValue( basisRow, 0.0 );


		for( Int g = 0; g < numSVDBasisTotal; g++ ){
		  blas::Scal( numBasisTotal, 1.0 / S[g], U.VecData(g), 1 );
		}


		// FIXME
		blas::Gemm( 'N', 'N', numLGLGridLocal, numSVDBasisTotal,
			    numBasisTotal, 1.0, localBasisRow.Data(), numLGLGridLocal,
			    U.Data(), numBasisTotal, 0.0, basisRow.Data(), numLGLGridLocal );

		AlltoallBackward (basisRow, basis, domain_.rowComm);

		// FIXME
		// row-partition to column partition via MPI_Alltoallv

		// Unscale the orthogonal basis functions by sqrt of
		// integration weight
		// FIXME


		for( Int g = 0; g < basis.n(); g++ ){
		  Real *ptr1 = basis.VecData(g);
		  Real *ptr2 = sqrtLGLWeight3D.Data();
		  for( Int l = 0; l < basis.m(); l++ ){
		    *(ptr1++)  /= *(ptr2++);
		  }
		}

		// FIXME
		//blas::Gemm( 'N', 'N', heightElemLocal, numSVDBasisTotal,
		//   numBasisTotal, 1.0, localBasisElemRow.Data(), heightElemLocal,
		//   U.Data(), numBasisTotal, 0.0, localBasisElemRow.Data(), heightElemLocal );

		//AlltoallBackward (localBasisElemRow, localBasisUniformGrid, domain_.rowComm);


#if ( _DEBUGlevel_ >= 1 )
		statusOFS << "Singular values of the basis = " 
			  << S << std::endl;
#endif

#if ( _DEBUGlevel_ >= 0 )
		statusOFS << "Number of significant SVD basis = " 
			  << numSVDBasisTotal << std::endl;
#endif


		MPI_Barrier( domain_.rowComm );




		GetTime( timeEnd );
		statusOFS << "Time for SVD of basis = " 	<< timeEnd - timeSta
			  << " [s]" << std::endl;


		MPI_Barrier( domain_.rowComm );

		//if(1){
		//  statusOFS << std::endl<< "All processors exit with abort in scf_dg.cpp." << std::endl;
		//  abort();
		// }


		// Transfer psi from coarse grid to fine grid with FFT
		if(1){ 

		  Int ntot  = psi.NumGridTotal();
		  Int ntotFine  = numGridExtElemFine.prod ();
		  Int ncom  = psi.NumComponent();
		  Int nocc  = psi.NumState();

		  //DblNumMat psiUniformGridFine( 
		  //ntotFine, 
		  //numBasis );

		  DblNumMat localBasisUniformGrid;

		  localBasisUniformGrid.Resize( numGridElemFine.prod(), numBasis );

		  SetValue( localBasisUniformGrid, 0.0 );

		  //Fourier& fft = eigSol.FFT();

		  for (Int k=0; k<nocc; k++) {

		    //                    for( Int i = 0; i < ntot; i++ ){
		    //                      fft.inputComplexVec(i) = Complex( psi.Wavefun(i,0,k), 0.0 );
		    //                    }
		    //
		    //                    fftw_execute( fft.forwardPlan );
		    //
		    //                    // fft Coarse to Fine 
		    //
		    //                    SetValue( fft.outputComplexVecFine, Z_ZERO );
		    //                    for( Int i = 0; i < ntot; i++ ){
		    //                      fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
		    //                    }
		    //
		    //                    fftw_execute( fft.backwardPlanFine );
		    //
		    //                    DblNumVec psiTemp (ntotFine);
		    //                    SetValue( psiTemp, 0.0 ); 
		    //
		    //                    for( Int i = 0; i < ntotFine; i++ ){
		    //
		    //                      psiTemp(i) = fft.inputComplexVecFine(i).real() / (double(ntot) * double(ntotFine));
		    //
		    //                    }
		    //

		    InterpPeriodicGridExtElemToGridElem( 
							numGridExtElem,
							numGridElemFine,
							wavefun.VecData(0, k), 
							localBasisUniformGrid.VecData(k) );


		  } // for k


		  DblNumMat localBasisElemRow( heightElemLocal, numBasisTotal );
		  SetValue( localBasisElemRow, 0.0 );

		  DblNumMat  basisUniformGridRow( heightElemLocal, numSVDBasisTotal );
		  SetValue( basisUniformGridRow, 0.0 );

		  DblNumMat& basisUniformGrid = hamDG.BasisUniformFine().LocalMap()[key];
		  basisUniformGrid.Resize( numGridElemFine.prod(), numSVDBasisLocal );
		  SetValue( basisUniformGrid, 0.0 );

		  AlltoallForward (localBasisUniformGrid, localBasisElemRow, domain_.rowComm);

		  // FIXME
		  blas::Gemm( 'N', 'N', heightElemLocal, numSVDBasisTotal,
			      numBasisTotal, 1.0, localBasisElemRow.Data(), heightElemLocal,
			      U.Data(), numBasisTotal, 0.0, basisUniformGridRow.Data(), heightElemLocal );

		  AlltoallBackward (  basisUniformGridRow, basisUniformGrid, domain_.rowComm);

		} // if (1)


		MPI_Barrier( domain_.rowComm );

	      } // if(1)


	    } // own this element
	  } // for (i)




      GetTime( timeBasisEnd );

      statusOFS << std::endl << "Time for generating ALB function is " <<
	timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;
      
      
      // ~~**~~
      // ~~~~~~~~~~~~~~~~~~~~~~~~	
      // Routine for re-orienting eigenvectors based on current basis set
      if(Diag_SCFDG_by_Cheby_ == 1)
	{
	  Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
    
	  if( iter > 1)
	    {  
	      statusOFS << std::endl << " Rotating the eigenvectors from the previous step ... ";
	      GetTime(timeSta);
	    }
	  // Figure out the element that we own using the standard loop
	  for( Int k = 0; k < numElem_[2]; k++ )
	    for( Int j = 0; j < numElem_[1]; j++ )
	      for( Int i = 0; i < numElem_[0]; i++ )
		{
		  Index3 key( i, j, k );
		
		  // If we own this element
		  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
		    {
		      EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
		      Index3 numLGLGrid    = hamDG.NumLGLGridElem();
		      Spinor& psi = eigSol.Psi();

		      // Assuming that wavefun has only 1 component, i.e., spin-unpolarized
		      // These are element specific quantities
		      DblNumTns& wavefun = psi.Wavefun();		
		      Int numBasis = psi.NumState();
		      Int numBasisTotal = psi.NumStateTotal();
	      
		      // This is the band distributed local basis
		      DblNumMat& ref_band_distrib_local_basis = hamDG.BasisLGL().LocalMap()[key];
		      DblNumMat band_distrib_local_basis(ref_band_distrib_local_basis.m(),ref_band_distrib_local_basis.n());
		  
		      blas::Copy(ref_band_distrib_local_basis.m() * ref_band_distrib_local_basis.n(), 
				 ref_band_distrib_local_basis.Data(), 1,
				 band_distrib_local_basis.Data(), 1);   
		  
		      // LGL weights and sqrt weights
		      DblNumTns&   LGLWeight3D = hamDG.LGLWeight3D();
		      DblNumTns    sqrtLGLWeight3D( numLGLGrid[0], numLGLGrid[1], numLGLGrid[2] );
	      
		      Real *ptr1 = LGLWeight3D.Data(), *ptr2 = sqrtLGLWeight3D.Data();
		      for( Int i = 0; i < numLGLGrid.prod(); i++ ){
			*(ptr2++) = std::sqrt( *(ptr1++) );
		      }
		  
		  	  
		      // Scale band_distrib_local_basis using sqrt(weights)
		      for( Int g = 0; g < band_distrib_local_basis.n(); g++ ){
			Real *ptr1 = band_distrib_local_basis.VecData(g);
			Real *ptr2 = sqrtLGLWeight3D.Data();
			for( Int l = 0; l < band_distrib_local_basis.m(); l++ ){
			  *(ptr1++)  *= *(ptr2++);
			}
		      }

		      // Figure out a few dimensions for the row-distribution
		      Int heightLGL = numLGLGrid.prod();
		      Int width = psi.NumStateTotal();
		
		      Int widthBlocksize = width / mpisizeRow;
		      Int widthLocal = widthBlocksize;
		
		      Int heightLGLBlocksize = heightLGL / mpisizeRow;
		      Int heightLGLLocal = heightLGLBlocksize;
		
		      if(mpirankRow < (width % mpisizeRow)){
			widthLocal = widthBlocksize + 1;
		      }

		      if(mpirankRow == (mpisizeRow - 1)){
			heightLGLLocal = heightLGLBlocksize + heightLGL % mpisizeRow;
		      }
		
				
		      // Convert from band distribution to row distribution
		      DblNumMat row_distrib_local_basis(heightLGLLocal, width);;
		      SetValue(row_distrib_local_basis, 0.0);  
		
		      
		      statusOFS << std::endl << " AlltoallForward: Changing distribution of local basis functions ... ";
		      GetTime(extra_timeSta);
		      AlltoallForward(band_distrib_local_basis, row_distrib_local_basis, domain_.rowComm);
		      GetTime(extra_timeEnd);
		      statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";
		
		      // Push the row-distributed matrix into the deque
		      ALB_LGL_deque.push_back( row_distrib_local_basis );
		
		
		      // If the deque has 2 elements, compute the overlap and perform a rotation of the eigenvectors
		      if( ALB_LGL_deque.size() == 2)
			{
			  GetTime(extra_timeSta);
			  statusOFS << std::endl << " Computing the overlap matrix using basis sets on LGL grid ... ";
		    
			  // Compute the local overlap matrix V2^T * V1		    
			  DblNumMat Overlap_Mat( width, width );
			  DblNumMat Overlap_Mat_Temp( width, width );
			  SetValue( Overlap_Mat, 0.0 );
			  SetValue( Overlap_Mat_Temp, 0.0 );
		    
			  double *ptr_0 = ALB_LGL_deque[0].Data();
			  double *ptr_1 = ALB_LGL_deque[1].Data();
		    
			  blas::Gemm( 'T', 'N', width, width, heightLGLLocal,
				      1.0, ptr_1, heightLGLLocal, 
				      ptr_0, heightLGLLocal, 
				      0.0, Overlap_Mat_Temp.Data(), width );

		    
			  // Reduce along rowComm (i.e., along the intra-element direction)
			  // to compute the actual overlap matrix
			  MPI_Allreduce( Overlap_Mat_Temp.Data(), 
					 Overlap_Mat.Data(), 
					 width * width, 
					 MPI_DOUBLE, 
					 MPI_SUM, 
					 domain_.rowComm );
		
			  GetTime(extra_timeEnd);
			  statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";
		    
			  // Rotate the current eigenvectors : This can also be done in parallel
			  // at the expense of an AllReduce along rowComm
			  
			  statusOFS << std::endl << " Rotating the eigenvectors using overlap matrix ... ";
		          GetTime(extra_timeSta);
			  
			  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;               
		    
			  DblNumMat temp_loc_eigvecs_buffer;
			  temp_loc_eigvecs_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
		
			  blas::Copy( eigvecs_local.m() * eigvecs_local.n(), 
				      eigvecs_local.Data(), 1, 
				      temp_loc_eigvecs_buffer.Data(), 1 ); 
		
			  blas::Gemm( 'N', 'N', Overlap_Mat.m(), eigvecs_local.n(), Overlap_Mat.n(), 
				      1.0, Overlap_Mat.Data(), Overlap_Mat.m(), 
				      temp_loc_eigvecs_buffer.Data(), temp_loc_eigvecs_buffer.m(), 
				      0.0, eigvecs_local.Data(), eigvecs_local.m());
		    
		          GetTime(extra_timeEnd);
		          statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta) << " s)";
		
		    
			  ALB_LGL_deque.pop_front();
			}		
	     
		    } // End of if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) )
	    
		} // End of loop over key indices i.e., for( Int i = 0; i < numElem_[0]; i++ )
	  
	  if( iter > 1)
	    {
	      GetTime(timeEnd);
	      statusOFS << std::endl << " Rotation completed. ( " << (timeEnd - timeSta) << " s )";
	   
	    }
	  
	} // End of if(Diag_SCFDG_by_Cheby_ == 1)
      


      // *********************************************************************
      // Inner SCF iteration 
      //
      // Assemble and diagonalize the DG matrix until convergence is
      // reached for updating the basis functions in the next step.
      // *********************************************************************

      GetTime(timeSta);

      // Save the mixing variable in the outer SCF iteration 
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      if( mixVariable_ == "density" ){
		DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
		mixOuterSave_.LocalMap()[key] = oldVec;
	      }
	      else if( mixVariable_ == "potential" ){
		DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
		mixOuterSave_.LocalMap()[key] = oldVec;
	      }
	    } // own this element
	  } // for (i)

      // Main function here
      InnerIterate( iter );

      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for all inner SCF iterations is " <<
	timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // *********************************************************************
      // Post processing 
      // *********************************************************************

      Int numAtom = hamDG.AtomList().size();
      efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

      // Compute the error of the mixing variable 
      {
	Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
	Real normMixDif, normMixOld;
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		if( mixVariable_ == "density" ){
		  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
		  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

		  for( Int p = 0; p < oldVec.m(); p++ ){
		    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
		    normMixOldLocal += pow( oldVec(p), 2.0 );
		  }
		}
		else if( mixVariable_ == "potential" ){
		  DblNumVec& oldVec = mixOuterSave_.LocalMap()[key];
		  DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

		  for( Int p = 0; p < oldVec.m(); p++ ){
		    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
		    normMixOldLocal += pow( oldVec(p), 2.0 );
		  }
		}
	      } // own this element
	    } // for (i)


	mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
			domain_.colComm );
	mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
			domain_.colComm );

	normMixDif = std::sqrt( normMixDif );
	normMixOld = std::sqrt( normMixOld );

	scfOuterNorm_    = normMixDif / normMixOld;


	Print(statusOFS, "OUTERSCF: EfreeHarris                 = ", EfreeHarris_ ); 
	//			FIXME
	//			Print(statusOFS, "OUTERSCF: EfreeSecondOrder            = ", EfreeSecondOrder_ ); 
	Print(statusOFS, "OUTERSCF: Efree                       = ", Efree_ ); 
	Print(statusOFS, "OUTERSCF: norm(out-in)/norm(in) = ", scfOuterNorm_ ); 
	Print(statusOFS, "OUTERSCF: Efree diff per atom   = ", efreeDifPerAtom_ ); 
	statusOFS << std::endl;
      }

      //		// Print out the state variables of the current iteration
      //    PrintState( );


      if( iter >= 2 & 
	  ( (scfOuterNorm_ < scfOuterTolerance_) & 
	    (efreeDifPerAtom_ < scfOuterEnergyTolerance_) ) ){
	/* converged */
	Print( statusOFS, "Outer SCF is converged!\n" );
	statusOFS << std::endl;
	isSCFConverged = true;
      }

      // Potential mixing for the outer SCF iteration. or no mixing at all anymore?
      // It seems that no mixing is the best.
	
      GetTime( timeIterEnd );
      statusOFS << "Time for this SCF iteration = " << timeIterEnd - timeIterStart
		<< " [s]" << std::endl;
    } // for( iter )

    GetTime( timeTotalEnd );

    statusOFS << std::endl;
    statusOFS << "Total time for all SCF iterations = " << 
      timeTotalEnd - timeTotalStart << " [s]" << std::endl;
    if( isSCFConverged == true ){
      statusOFS << "Total number of outer SCF steps = " <<
	iter << std::endl;
    }
    else{
      statusOFS << "Total number of outer SCF steps = " <<
	scfOuterMaxIter_ << std::endl;
    }

    //    if(0)
    //    {
    //      // Output the electron density on the LGL grid in each element
    //      std::ostringstream rhoStream;      
    //
    //      NumTns<std::vector<DblNumVec> >& LGLGridElem =
    //        hamDG.LGLGridElem();
    //
    //      for( Int k = 0; k < numElem_[2]; k++ )
    //        for( Int j = 0; j < numElem_[1]; j++ )
    //          for( Int i = 0; i < numElem_[0]; i++ ){
    //            Index3 key( i, j, k );
    //            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
    //              DblNumVec&  denVec = hamDG.DensityLGL().LocalMap()[key];
    //              std::vector<DblNumVec>& grid = LGLGridElem(i, j, k);
    //              for( Int d = 0; d < DIM; d++ ){
    //                serialize( grid[d], rhoStream, NO_MASK );
    //              }
    //              serialize( denVec, rhoStream, NO_MASK );
    //            }
    //          } // for (i)
    //      SeparateWrite( "DENLGL", rhoStream );
    //    }


    // *********************************************************************
    // Output information
    // *********************************************************************

    // Output the atomic structure, and other information for describing
    // density, basis functions etc.
    // 
    // Only mpirank == 0 works on this

    Real timeOutputSta, timeOutputEnd;
    GetTime( timeOutputSta );

    if(1){
      if( mpirank == 0 ){
	std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << std::endl 
		  << "Output the structure information" 
		  << std::endl;
#endif
	// Domain
	serialize( domain_.length, structStream, NO_MASK );
	serialize( domain_.numGrid, structStream, NO_MASK );
	serialize( domain_.numGridFine, structStream, NO_MASK );
	serialize( domain_.posStart, structStream, NO_MASK );
	serialize( numElem_, structStream, NO_MASK );

	// Atomic information
	serialize( hamDG.AtomList(), structStream, NO_MASK );
	std::string structFileName = "STRUCTURE";

	std::ofstream fout(structFileName.c_str());
	if( !fout.good() ){
	  std::ostringstream msg;
	  msg 
	    << "File " << structFileName.c_str() << " cannot be open." 
	    << std::endl;
	  throw std::runtime_error( msg.str().c_str() );
	}
	fout << structStream.str();
	fout.close();
      }
    }


    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    // Output density, and only mpirankRow == 0 does the job of
	    // for each element.
	    if( isOutputDensity_ ){
	      if( mpirankRow == 0 ){
#if ( _DEBUGlevel_ >= 0 )
		statusOFS << std::endl 
			  << "Output the electron density on the global grid" 
			  << std::endl;
#endif
		// Output the wavefunctions on the uniform grid
		{
		  std::ostringstream rhoStream;      

		  NumTns<std::vector<DblNumVec> >& uniformGridElem =
		    hamDG.UniformGridElemFine();
		  std::vector<DblNumVec>& grid = hamDG.UniformGridElemFine()(i, j, k);
		  for( Int d = 0; d < DIM; d++ ){
		    serialize( grid[d], rhoStream, NO_MASK );
		  }

		  serialize( key, rhoStream, NO_MASK );
		  serialize( hamDG.Density().LocalMap()[key], rhoStream, NO_MASK );

		  SeparateWrite( restartDensityFileName_, rhoStream, mpirankCol );
		}

		// Output the wavefunctions on the LGL grid
		{
		  std::ostringstream rhoStream;      

		  // Generate the uniform mesh on the extended element.
		  std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
		  for( Int d = 0; d < DIM; d++ ){
		    serialize( gridpos[d], rhoStream, NO_MASK );
		  }
		  serialize( key, rhoStream, NO_MASK );
		  serialize( hamDG.DensityLGL().LocalMap()[key], rhoStream, NO_MASK );
		  SeparateWrite( "DENLGL", rhoStream, mpirankCol );
		}

	      } // if( mpirankRow == 0 )
	    }

	    // Output potential in extended element, and only mpirankRow
	    // == 0 does the job of for each element.
	    if( isOutputPotExtElem_ ) {
	      if( mpirankRow == 0 ){
#if ( _DEBUGlevel_ >= 0 )
		statusOFS 
		  << std::endl 
		  << "Output the total potential and external potential in the extended element."
		  << std::endl;
#endif
		std::ostringstream potStream;      
		EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];

		// Generate the uniform mesh on the extended element.
		//              std::vector<DblNumVec> gridpos;
		//              UniformMeshFine ( eigSol.FFT().domain, gridpos );
		//              for( Int d = 0; d < DIM; d++ ){
		//                serialize( gridpos[d], potStream, NO_MASK );
		//              }


		serialize( key, potStream, NO_MASK );
		serialize( eigSol.Ham().Vtot(), potStream, NO_MASK );
		serialize( eigSol.Ham().Vext(), potStream, NO_MASK );
		SeparateWrite( "POTEXT", potStream, mpirankCol );
	      } // if( mpirankRow == 0 )
	    }

	    // Output wavefunction in the extended element.  All processors participate
	    if( isOutputWfnExtElem_ )
	      {
#if ( _DEBUGlevel_ >= 0 )
		statusOFS 
		  << std::endl 
		  << "Output the wavefunctions in the extended element."
		  << std::endl;
#endif

		EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
		std::ostringstream wavefunStream;      

		// Generate the uniform mesh on the extended element.
		// NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
		std::vector<DblNumVec> gridpos;
		UniformMesh ( eigSol.FFT().domain, gridpos );
		for( Int d = 0; d < DIM; d++ ){
		  serialize( gridpos[d], wavefunStream, NO_MASK );
		}

		serialize( key, wavefunStream, NO_MASK );
		serialize( eigSol.Psi().Wavefun(), wavefunStream, NO_MASK );
		SeparateWrite( restartWfnFileName_, wavefunStream, mpirank);
	      }

	    // Output wavefunction in the element on LGL grid. All processors participate.
	    if( isOutputALBElemLGL_ )
	      {
#if ( _DEBUGlevel_ >= 0 )
		statusOFS 
		  << std::endl 
		  << "Output the wavefunctions in the element on a LGL grid."
		  << std::endl;
#endif
		// Output the wavefunctions in the extended element.
		std::ostringstream wavefunStream;      

		// Generate the uniform mesh on the extended element.
		std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
		for( Int d = 0; d < DIM; d++ ){
		  serialize( gridpos[d], wavefunStream, NO_MASK );
		}
		serialize( key, wavefunStream, NO_MASK );
		serialize( hamDG.BasisLGL().LocalMap()[key], wavefunStream, NO_MASK );
		SeparateWrite( "ALBLGL", wavefunStream, mpirank );
	      }

	    // Output wavefunction in the element on uniform fine grid.
	    // All processors participate
	    // NOTE: 
	    // Since interpolation needs to be performed, this functionality can be slow.
	    if( isOutputALBElemUniform_ )
	      {
#if ( _DEBUGlevel_ >= 0 )
		statusOFS 
		  << std::endl 
		  << "Output the wavefunctions in the element on a fine LGL grid."
		  << std::endl;
#endif
		// Output the wavefunctions in the extended element.
		std::ostringstream wavefunStream;      

		// Generate the uniform mesh on the extended element.
		serialize( key, wavefunStream, NO_MASK );
		DblNumMat& basisLGL = hamDG.BasisLGL().LocalMap()[key];
		DblNumMat basisUniformFine( 
					   hamDG.NumUniformGridElemFine().prod(), 
					   basisLGL.n() );
		SetValue( basisUniformFine, 0.0 );
            
		for( Int g = 0; g < basisLGL.n(); g++ ){
		  hamDG.InterpLGLToUniform(
					   hamDG.NumLGLGridElem(),
					   hamDG.NumUniformGridElemFine(),
					   basisLGL.VecData(g),
					   basisUniformFine.VecData(g) );
		}
		// Generate the uniform mesh on the extended element.
		// NOTE 05/06/2015: THIS IS NOT COMPATIBLE WITH THAT OF THE ALB2DEN!!
		std::vector<DblNumVec> gridpos;
		EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
		// UniformMeshFine ( eigSol.FFT().domain, gridpos );
		// for( Int d = 0; d < DIM; d++ ){
		//   serialize( gridpos[d], wavefunStream, NO_MASK );
		// }

		serialize( key, wavefunStream, NO_MASK );
		serialize( basisUniformFine, wavefunStream, NO_MASK );
		SeparateWrite( "ALBUNIFORM", wavefunStream, mpirank );
	      }

	  } // (own this element)
	} // for (i)

    GetTime( timeOutputEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl 
	      << "Time for outputing data is = " << timeOutputEnd - timeOutputSta
	      << " [s]" << std::endl;
#endif

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::Iterate  ----- 


  // ~~**~~
  // ---------------------------------------------------------
  // ------------ Routines for Chebyshev Filtering -----------
  // ---------------------------------------------------------
  
  // Dot product for conforming distributed vectors
  // Requires that a distributed vector (and not a distributed matrix) has been sent
  double 
  SCFDG::scfdg_distvec_dot(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a,
			   DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_b)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_distvec_dot");
#endif
    
    double temp_result = 0.0, final_result = 0.0;
    DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
    DblNumMat& vec_b= (dist_vec_b.LocalMap().begin())->second;
    
    // Conformity check
    if( (dist_vec_a.LocalMap().size() != 1) ||
        (dist_vec_b.LocalMap().size() != 1) ||
        (vec_a.m() != vec_b.m()) ||
        (vec_a.n() != 1) ||
        (vec_b.n() != 1) )
      {
	statusOFS << std::endl << " Non-conforming vectors in dot product !! Aborting ... " << std::endl;
	exit(1);
      }
      
    // Local result
    temp_result = blas::Dot(vec_a.m(),vec_a.Data(), 1, vec_b.Data(), 1);
    // Global reduce  across colComm since the partion is by DG elements
    mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );
      
    
#ifndef _RELEASE_
    PopCallStack();
#endif  
     
    return final_result;
  } // End of routine scfdg_distvec_dot
  
  // L2 norm : Requires that a distributed vector (and not a distributed matrix) has been sent
  double 
  SCFDG::scfdg_distvec_nrm2(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_vec_a)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_distvec_nrm2");
#endif
    
    double temp_result = 0.0, final_result = 0.0;
    DblNumMat& vec_a= (dist_vec_a.LocalMap().begin())->second;
    double *ptr = vec_a.Data();
    
    if((dist_vec_a.LocalMap().size() != 1) ||
       (vec_a.n() != 1))
      {
	statusOFS << std::endl << " Unacceptable vector in norm routine !! Aborting ... " << std::endl;
	exit(1);
      }
      
    // Local result 
    for(Int iter = 0; iter < vec_a.m(); iter ++)
      temp_result += (ptr[iter] * ptr[iter]);
        
    // Global reduce  across colComm since the partion is by DG elements
    mpi::Allreduce( &temp_result, &final_result, 1, MPI_SUM, domain_.colComm );
    final_result = sqrt(final_result);
    
#ifndef _RELEASE_
    PopCallStack();
#endif  
    return  final_result;
    
  } // End of routine scfdg_distvec_nrm2
  
  // Computes b = scal_a * a + scal_b * b for conforming distributed vectors / matrices
  // Set scal_a = 0.0 and use any vector / matrix a to obtain blas::scal on b
  // Set scal_b = 1.0 for blas::axpy with b denoting y, i.e., b = scal_a * a + b;
  void 
  SCFDG::scfdg_distmat_update(DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_a, 
			      double scal_a,
			      DistVec<Index3, DblNumMat, ElemPrtn>  &dist_mat_b,
			      double scal_b)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_distmat_update");
#endif
    
    DblNumMat& mat_a= (dist_mat_a.LocalMap().begin())->second;
    DblNumMat& mat_b= (dist_mat_b.LocalMap().begin())->second;  
    
    double *ptr_a = mat_a.Data();
    double *ptr_b = mat_b.Data();
    
    // Conformity check
    if( (dist_mat_a.LocalMap().size() != 1) ||
        (dist_mat_b.LocalMap().size() != 1) ||
        (mat_a.m() != mat_b.m()) ||
        (mat_a.n() != mat_b.n()) )
      {
	statusOFS << std::endl << " Non-conforming distributed vectors / matrices in update routine !!" 
		  << std::endl << " Aborting ... " << std::endl;
	exit(1);
      }
      
    for(Int iter = 0; iter < (mat_a.m() * mat_a.n()) ; iter ++)
      ptr_b[iter] = scal_a * ptr_a[iter] + scal_b * ptr_b[iter];
    
#ifndef _RELEASE_
    PopCallStack();
#endif  
    
    
  } // End of routine scfdg_distvec_update
  
 
  // This routine computes Hmat_times_my_dist_mat = Hmat_times_my_dist_mat + Hmat * my_dist_mat
  void SCFDG::scfdg_hamiltonian_times_distmat(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
					      DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_mat)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_hamiltonian_times_distmat");
#endif
    
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    
    HamiltonianDG&  hamDG = *hamDGPtr_;
    std::vector<Index3>  getKeys_list;
    
    // Check that vectors provided only contain one entry in the local map
    // This is a safeguard to ensure that we are really dealing with distributed matrices
    if((my_dist_mat.LocalMap().size() != 1) ||
       (Hmat_times_my_dist_mat.LocalMap().size() != 1) ||
       ((my_dist_mat.LocalMap().begin())->first != (Hmat_times_my_dist_mat.LocalMap().begin())->first))
      {
	statusOFS << std::endl << " Vectors in Hmat * vector_block product not formatted correctly !!"
		  << std::endl << " Aborting ... " << std::endl;
	exit(1);
      }
    
    
    // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
    // based on processor number, etc.
    Index3 key = (my_dist_mat.LocalMap().begin())->first;
    
    // Obtain keys of neighbors using the Hamiltonian matrix
    // We only use these keys to minimize communication in GetBegin since other parts of the vector
    // block, even if they are non-zero will get multiplied by the zero block of the Hamiltonian
    // anyway.
    for(typename std::map<ElemMatKey, DblNumMat >::iterator 
	  get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
        get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
        get_neighbors_from_Ham_iterator ++)
      {
	Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;
      
	if(neighbor_key == key)
	  continue;
	else
	  getKeys_list.push_back(neighbor_key);
      }
    
     
    // Do the communication necessary to get the information from
    // procs holding the neighbors
    // Supposedly, independent row communicators (i.e. colComm)
    //  are being used for this
    my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
    my_dist_mat.GetEnd( NO_MASK );
     
     
    // Obtain a reference to the chunk where we want to store
    DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];
     
    // Now pluck out relevant chunks of the Hamiltonian and the vector and multiply
    for(typename std::map<Index3, DblNumMat >::iterator 
	  mat_X_iterator = my_dist_mat.LocalMap().begin();
	mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++ ){
	      
      Index3 iter_key = mat_X_iterator->first;	   
      DblNumMat& mat_X_local = mat_X_iterator->second; // Chunk from input block of vectors
      
      // Create key for looking up Hamiltonian chunk 
      ElemMatKey myelemmatkey = std::make_pair(key, iter_key);
	            
      std::map<ElemMatKey, DblNumMat >::iterator ham_iterator = hamDG.HMat().LocalMap().find(myelemmatkey);
          
      //statusOFS << std::endl << " Working on key " << key << "   " << iter_key << std::endl;

      // Now do the actual multiplication
      DblNumMat& mat_H_local = ham_iterator->second; // Chunk from Hamiltonian
		   
      Int m = mat_H_local.m(), n = mat_X_local.n(), k = mat_H_local.n();
		   
      blas::Gemm( 'N', 'N', m, n, k, 
		  1.0, mat_H_local.Data(), m, 
		  mat_X_local.Data(), k, 
		  1.0, mat_Y_local.Data(), m);


    } // End of loop using mat_X_iterator
	    
    // Matrix * vector_block product is ready now ... 
    // Need to clean up extra entries in my_dist_mat
    typename std::map<Index3, DblNumMat >::iterator it;
    for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
      {
	it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
	(my_dist_mat.LocalMap()).erase(it);
      }
 	
#ifndef _RELEASE_
    PopCallStack();
#endif  
	   
  }
 

  
  // This routine estimates the spectral bounds using the Lanczos method
  double 
  SCFDG::scfdg_Cheby_Upper_bound_estimator(DblNumVec& ritz_values, 
					   int Num_Lanczos_Steps
					   )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_Cheby_Upper_bound_estimator");
#endif
    
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    
    Real timeSta, timeEnd;
    Real timeIterStart, timeIterEnd;
    
    
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    // Declare vectors partioned according to DG elements
    // These will be used for Lanczos
    
    // Input vector
    DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v; 
    dist_vec_v.Prtn() = elemPrtn_;
    dist_vec_v.SetComm(domain_.colComm);
    
    // vector to hold f = H*v
    DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_f; 
    dist_vec_f.Prtn() = elemPrtn_;
    dist_vec_f.SetComm(domain_.colComm);
    
    // vector v0
    DistVec<Index3, DblNumMat, ElemPrtn>  dist_vec_v0; 
    dist_vec_v0.Prtn() = elemPrtn_;
    dist_vec_v0.SetComm(domain_.colComm);
    
    // Step 1 : Generate a random vector v
    // Fill up the vector v using random entries
    // Also, initialize the vectors f and v0 to 0
    
    // We use this loop for setting up the keys. 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  
	  // If the current processor owns this element
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 
	    
	    // Associate the current key with a vector that contains the stuff 
	    // that should reside on this process.
	    const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
	    
	    dist_vec_v.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now	    
	    dist_vec_f.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now
            dist_vec_v0.LocalMap()[key].Resize(idx.size(), 1); // Because of this, the LocalMap is of size 1 now
	    
	    // Initialize the local maps	
	    // Vector v is initialized randomly
	    UniformRandom(dist_vec_v.LocalMap()[key]);
	    
	    // Vector f and v0 are filled with zeros
	    SetValue(dist_vec_f.LocalMap()[key], 0.0);
	    SetValue(dist_vec_v0.LocalMap()[key], 0.0);
	   	    
	  }
	} // End of vector initializations
	

    // Normalize the vector v
    double norm_v = scfdg_distvec_nrm2(dist_vec_v);     
    scfdg_distmat_update(dist_vec_v, 0.0, dist_vec_v, (1.0 / norm_v));
     
     
    // Step 2a : f = H * v 
    //scfdg_hamiltonian_times_distvec(dist_vec_v,  dist_vec_f); // f has already been set to 0
    scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f); // f has already been set to 0
     
    // Step 2b : alpha = f^T * v
    double alpha, beta;
    alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);
     
    // Step 2c : f = f - alpha * v
    scfdg_distmat_update(dist_vec_v, (-alpha), dist_vec_f, 1.0);
     
    // Step 2d: T(1,1) = alpha
    DblNumMat matT( Num_Lanczos_Steps, Num_Lanczos_Steps );
    SetValue(matT, 0.0);    
    matT(0,0) = alpha;
     
    // Step 3: Lanczos iteration
    for(Int j = 1; j < Num_Lanczos_Steps; j ++)
      {
	// Step 4 : beta = norm(f)
	beta = scfdg_distvec_nrm2(dist_vec_f);
	
	// Step 5a : v0 = v
	scfdg_distmat_update(dist_vec_v, 1.0, dist_vec_v0,  0.0);
	
	// Step 5b : v = f / beta
	scfdg_distmat_update(dist_vec_f, (1.0 / beta), dist_vec_v, 0.0);
	
	// Step 6a : f = H * v
	SetValue((dist_vec_f.LocalMap().begin())->second, 0.0);// Set f to zero first !
	scfdg_hamiltonian_times_distmat(dist_vec_v,  dist_vec_f);
	 
	// Step 6b : f = f - beta * v0
	scfdg_distmat_update(dist_vec_v0, (-beta) , dist_vec_f,  1.0);
	 
	// Step 7a : alpha = f^T * v
	alpha = scfdg_distvec_dot(dist_vec_f, dist_vec_v);
	 
	// Step 7b : f = f - alpha * v
	scfdg_distmat_update(dist_vec_v, (-alpha) , dist_vec_f,  1.0);
	
	// Step 8 : Fill up the matrix
	matT(j, j - 1) = beta;
	matT(j - 1, j) = beta;
	matT(j, j) = alpha;
      } // End of loop over Lanczos steps 
      
    // Step 9 : Compute the Lanczos-Ritz values
    ritz_values.Resize(Num_Lanczos_Steps);
    SetValue( ritz_values, 0.0 );
              
    // Solve the eigenvalue problem for the Ritz values
    lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, matT.Data(), Num_Lanczos_Steps, ritz_values.Data() );
         
    // Step 10 : Compute the upper bound on each process
    double b_up = ritz_values(Num_Lanczos_Steps - 1) + scfdg_distvec_nrm2(dist_vec_f);
      
    //statusOFS << std::endl << " Ritz values in estimator here : " << ritz_values ;
    //statusOFS << std::endl << " Upper bound of spectrum = " << b_up << std::endl;
      
    // Need to synchronize the Ritz values and the upper bound across the processes
    MPI_Bcast(&b_up, 1, MPI_DOUBLE, 0, domain_.comm);
    MPI_Bcast(ritz_values.Data(), Num_Lanczos_Steps, MPI_DOUBLE, 0, domain_.comm);
      
	
#ifndef _RELEASE_
    PopCallStack();
#endif  
	
    return b_up;
    
  } // End of scfdg_Cheby_Upper_bound_estimator
  
  // Apply the scaled Chebyshev Filter on the Eigenvectors
  // Use a distributor to work on selected bands based on
  // number of processors per element (i.e., no. of rows in process grid).
  void 
  SCFDG::scfdg_Chebyshev_filter_scaled(int m, 
				       double a, 
				       double b, 
				       double a_L)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_Chebyshev_filter");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
    
    Real timeSta, timeEnd;
    Real extra_timeSta, extra_timeEnd;
    Real filter_total_time = 0.0;
       
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    // We need to distribute bands according to rowComm since colComm is
    // used for element-wise partition.
    if(mpisizeRow > hamDG.NumStateTotal())
      {
	statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
		  << std::endl << " Cannot continue with band-parallelization. "
		  << std::endl << " Aborting ... " << std::endl;
	exit(1);

      }
    simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);
    
    
    // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
    const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
    DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;
    
    Int local_width = band_distributor.current_proc_size;
    Int local_height = eigvecs_local.m();
    Int local_pluck_sz = local_height * local_width;
    
    
    DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
    pluck_X.Prtn() = elemPrtn_;
    pluck_X.SetComm(domain_.colComm);
    pluck_X.LocalMap()[key].Resize(local_height, local_width);
    
    DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
    pluck_Y.Prtn() = elemPrtn_;
    pluck_Y.SetComm(domain_.colComm);
    pluck_Y.LocalMap()[key].Resize(local_height, local_width);
    
    DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Yt; 
    pluck_Yt.Prtn() = elemPrtn_;
    pluck_Yt.SetComm(domain_.colComm);
    pluck_Yt.LocalMap()[key].Resize(local_height, local_width);
    
    
    // Initialize the distributed matrices
    blas::Copy(local_pluck_sz, 
	       eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
               1,
	       pluck_X.LocalMap()[key].Data(),
	       1);
    
    SetValue(pluck_Y.LocalMap()[key], 0.0);
    SetValue(pluck_Yt.LocalMap()[key], 0.0);
    
    // Filtering scalars
    double e = (b - a) / 2.0;
    double c = (a + b) / 2.0;
    double sigma = e / (c - a_L);
    double tau = 2.0 / sigma;
    double sigma_new;
    
    // Begin the filtering process
    // Y = (H * X - c * X) * (sigma / e)
    // pluck_Y has been initialized to 0 already
    
    statusOFS << std::endl << " Chebyshev filtering : Process " << mpirank << " working on " 
	      << local_width << " of " << eigvecs_local.n() << " bands.";
	     
    statusOFS << std::endl << " Chebyshev filtering : Lower bound = " << a
              << std::endl << "                     : Upper bound = " << b
              << std::endl << "                     : a_L = " << a_L;
	      
    //statusOFS << std::endl << " Chebyshev filtering step 1 of " << m << " ... ";
    GetTime( extra_timeSta );
    
    scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X
    scfdg_distmat_update(pluck_X, (-c) , pluck_Y,  1.0); // Y = -c * X + 1.0 * Y
    scfdg_distmat_update(pluck_Y, 0.0 , pluck_Y,  (sigma / e)); // Y = 0.0 * Y + (sigma / e) * Y
    
    GetTime( extra_timeEnd );
    //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
    filter_total_time += (extra_timeEnd - extra_timeSta );
    
    for(Int filter_iter = 2; filter_iter < m; filter_iter ++)
      {   
	//statusOFS << std::endl << " Chebyshev filtering step " << filter_iter << " of " << m << " ... ";
	GetTime( extra_timeSta );
     
	sigma_new = 1.0 / (tau - sigma);
      
	//Compute Yt = (H * Y - c * Y) * (2 * sigma_new / e) - (sigma * sigma_new) * X
	// Set Yt to 0
	SetValue(pluck_Yt.LocalMap()[key], 0.0);
	scfdg_hamiltonian_times_distmat(pluck_Y, pluck_Yt); // Yt = H * Y
	scfdg_distmat_update(pluck_Y, (-c) , pluck_Yt,  1.0); // Yt = -c * Y + 1.0 * Yt
	scfdg_distmat_update(pluck_Yt, 0.0 , pluck_Yt,  (2.0 * sigma_new / e)); // Yt = 0.0 * Yt + (2.0 * sigma_new / e) * Yt
	scfdg_distmat_update(pluck_X, (-sigma * sigma_new) , pluck_Yt,  1.0 ); // Yt = (-sigma * sigma_new) * X + 1.0 * Yt
      
	// Update / Re-assign : X = Y, Y = Yt, sigma = sigma_new
	scfdg_distmat_update(pluck_Y, 1.0 , pluck_X,  0.0 ); // X = 0.0 * X + 1.0 * Y
	scfdg_distmat_update(pluck_Yt, 1.0 , pluck_Y,  0.0 ); // Y = 0.0 * Y + 1.0 * Yt
     
	sigma = sigma_new;   
     
	GetTime( extra_timeEnd );
	//statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
	filter_total_time += (extra_timeEnd - extra_timeSta );
     
      }
    
    statusOFS << std::endl <<  " Total filtering time for " 
	      << m << " filter steps = " << filter_total_time << " s."
	      << std::endl <<  " Average per filter step = " << ( filter_total_time / double(m) ) << " s.";
	       
    // pluck_Y contains the results of filtering.
    // Copy back pluck_Y to the eigenvector
    // SetValue(eigvecs_local, 0.0); // All entries set to zero for All-Reduce
    GetTime( extra_timeSta );
    
    DblNumMat temp_buffer;
    temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
    SetValue(temp_buffer, 0.0);
    
    blas::Copy(local_pluck_sz, 
	       pluck_Y.LocalMap()[key].Data(),
	       1,
	       temp_buffer.Data() + local_height * band_distributor.current_proc_start,
	       1);
       
    MPI_Allreduce(temp_buffer.Data(),
		  eigvecs_local.Data(),
		  (eigvecs_local.m() * eigvecs_local.n()),
		  MPI_DOUBLE,
		  MPI_SUM,
		  domain_.rowComm);
    
    GetTime( extra_timeEnd );
    statusOFS << std::endl << " Eigenvector block rebuild time = " 
              << (extra_timeEnd - extra_timeSta ) << " s.";
    
    
  
        
#ifndef _RELEASE_
    PopCallStack();
#endif  
   
  } // End of scfdg_Chebyshev_filter
  
  
  // Apply the Hamiltonian to the Eigenvectors and place result in result_mat
  // This routine is actually not used by the Chebyshev filter routine since all 
  // the filtering can be done block-wise to reduce communication 
  // among processor rows (i.e., rowComm), i.e., we do the full filter and 
  // communicate only in the end.
  // This routine is used for the Raleigh-Ritz step.
  // Uses a distributor to work on selected bands based on
  // number of processors per element (i.e., no. of rows in process grid).  
  void 
  SCFDG::scfdg_Hamiltonian_times_eigenvectors(DistVec<Index3, DblNumMat, ElemPrtn>  &result_mat)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_Hamiltonian_times_eigenvectors");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
    
    Real timeSta, timeEnd;
    Real extra_timeSta, extra_timeEnd;
    Real filter_total_time = 0.0;
       
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    // We need to distribute bands according to rowComm since colComm is
    // used for element-wise partition.
    if(mpisizeRow > hamDG.NumStateTotal())
      {
	statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
		  << std::endl << " Cannot continue with band-parallelization. "
		  << std::endl << " Aborting ... " << std::endl;
	exit(1);

      }
    simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);
    
    // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
    const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
    DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;
    
    Int local_width = band_distributor.current_proc_size;
    Int local_height = eigvecs_local.m();
    Int local_pluck_sz = local_height * local_width;
    
    
    DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
    pluck_X.Prtn() = elemPrtn_;
    pluck_X.SetComm(domain_.colComm);
    pluck_X.LocalMap()[key].Resize(local_height, local_width);
    
    DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
    pluck_Y.Prtn() = elemPrtn_;
    pluck_Y.SetComm(domain_.colComm);
    pluck_Y.LocalMap()[key].Resize(local_height, local_width);
    
    // Initialize the distributed matrices
    blas::Copy(local_pluck_sz, 
	       eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
               1,
	       pluck_X.LocalMap()[key].Data(),
	       1);
    
    SetValue(pluck_Y.LocalMap()[key], 0.0); // pluck_Y is initialized to 0 
   
    
    GetTime( extra_timeSta );
    
    scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X
    
    GetTime( extra_timeEnd );
    
    statusOFS << std::endl << " Hamiltonian times eigenvectors calculation time = " 
              << (extra_timeEnd - extra_timeSta ) << " s.";
    
    // Copy pluck_Y to result_mat after preparing it
    GetTime( extra_timeSta );
    
    DblNumMat temp_buffer;
    temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
    SetValue(temp_buffer, 0.0);
    
    blas::Copy(local_pluck_sz, 
	       pluck_Y.LocalMap()[key].Data(),
	       1,
	       temp_buffer.Data() + local_height * band_distributor.current_proc_start,
	       1);
    
    // Empty out the results and prepare the distributed matrix
    result_mat.LocalMap().clear();
    result_mat.Prtn() = elemPrtn_;
    result_mat.SetComm(domain_.colComm);
    result_mat.LocalMap()[key].Resize(eigvecs_local.m(), eigvecs_local.n());
          
    MPI_Allreduce(temp_buffer.Data(),
		  result_mat.LocalMap()[key].Data(),
		  (eigvecs_local.m() * eigvecs_local.n()),
		  MPI_DOUBLE,
		  MPI_SUM,
		  domain_.rowComm);
    
    GetTime( extra_timeEnd );
    statusOFS << std::endl << " Eigenvector block rebuild time = " 
              << (extra_timeEnd - extra_timeSta ) << " s.";
          
#ifndef _RELEASE_
    PopCallStack();
#endif  
   
  } // End of scfdg_Hamiltonian_times_eigenvectors
  
 
  //   // Given a block of eigenvectors (size:  hamDG.NumBasisTotal() * hamDG.NumStateTotal()),
  //   // convert this to ScaLAPACK format for subsequent use with ScaLAPACK routines
  //   // Should only be called by processors for which context > 0
  //   // This is the older version which has a higher communcation load.
  //   void 
  //   SCFDG::scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_vec, 
  // 							 std::vector<int> &my_cheby_scala_info,
  // 							 dgdft::scalapack::Descriptor &my_scala_descriptor,
  // 							 dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_vec)
  //   {
  // #ifndef _RELEASE_
  //     PushCallStack("SCFDG::scfdg_Cheby_distmat_to_ScaLAPACK_conversion_old");
  // #endif
  //    
  //     HamiltonianDG&  hamDG = *hamDGPtr_;
  //     
  //     // Load up the important ScaLAPACK info
  //     int cheby_scala_num_rows = my_cheby_scala_info[0];
  //     int cheby_scala_num_cols = my_cheby_scala_info[1];
  //     int my_cheby_scala_proc_row = my_cheby_scala_info[2];
  //     int my_cheby_scala_proc_col = my_cheby_scala_info[3];
  // 
  //     // Set the descriptor for the ScaLAPACK matrix
  //     my_scala_vec.SetDescriptor( my_scala_descriptor );
  //     
  //     // Get the original key for the distributed vector
  //     Index3 my_original_key = (my_dist_vec.LocalMap().begin())->first;
  // 	    
  //     // Form the list of unique keys that we will be requiring
  //     // Use the usual map trick for this
  //     std::map<Index3, int> unique_keys_list;
  // 	    
  //     // Use the row index for figuring out the key 	    
  //     for(int iter = 0; iter < hamDG.ElemBasisInvIdx().size(); iter ++)
  //       {
  // 	int pr = 0 + int(iter / scaBlockSize_) % cheby_scala_num_rows;
  // 	if(pr == my_cheby_scala_proc_row)
  // 	  unique_keys_list[hamDG.ElemBasisInvIdx()[iter]] = 0;
  // 	      
  //       }
  // 	    
  //     
  //     std::vector<Index3>  getKeys_list;
  // 	    
  //     // Form the list for Get-Begin and Get-End
  //     for(typename std::map<Index3, int >::iterator 
  // 	  it = unique_keys_list.begin(); 
  // 	it != unique_keys_list.end(); 
  // 	it ++)
  //       { 
  // 	getKeys_list.push_back(it->first);
  //       }    
  // 	    
  //     // Communication
  //     my_dist_vec.GetBegin( getKeys_list, NO_MASK ); 
  //     my_dist_vec.GetEnd( NO_MASK ); 
  // 	    
  // 	    
  //     // Get the offset value for each of the matrices pointed to by the keys
  //     std::map<Index3, int> offset_map;
  //     for(typename std::map<Index3, DblNumMat >::iterator 
  // 	  test_iterator = my_dist_vec.LocalMap().begin();
  //         test_iterator != my_dist_vec.LocalMap().end();
  // 	test_iterator ++)
  //       {	      
  // 	Index3 key = test_iterator->first;
  // 	const std::vector<Int>&  my_idx = hamDG.ElemBasisIdx()(key[0],key[1],key[2]); 
  // 	offset_map[key] = my_idx[0];
  //       }
  //      
  //     // All the data is now available: simply use this to fill up the local
  //     // part of the ScaLAPACK matrix. We do this by looping over global indices
  //     // and computing local indices from the globals	  
  //             
  //     double *local_scala_mat_ptr = my_scala_vec.Data();
  //     int local_scala_mat_height = my_scala_vec.LocalHeight();
  // 	    
  //     for(int global_col_iter = 0; global_col_iter < hamDG.NumStateTotal(); global_col_iter ++)
  //       {
  // 	int m = int(global_col_iter / (cheby_scala_num_cols * scaBlockSize_));
  // 	int y = global_col_iter  % scaBlockSize_;
  // 	int pc = int(global_col_iter / scaBlockSize_) % cheby_scala_num_cols;
  // 		
  // 	int local_col_iter = m * scaBlockSize_ + y;
  // 	      
  // 	for(int global_row_iter = 0; global_row_iter <  hamDG.NumBasisTotal(); global_row_iter ++)
  // 	  {
  // 	    int l = int(global_row_iter / (cheby_scala_num_rows * scaBlockSize_));
  // 	    int x = global_row_iter % scaBlockSize_;
  // 	    int pr = int(global_row_iter / scaBlockSize_) % cheby_scala_num_rows;
  // 						
  // 	    int local_row_iter = l * scaBlockSize_ + x;
  // 		
  // 	    // Check if this entry resides on the current process
  // 	    if((pr == my_cheby_scala_proc_row) && (pc == my_cheby_scala_proc_col))
  // 	      {  
  // 		  
  // 		// Figure out where to read entry from
  // 		Index3 key = hamDG.ElemBasisInvIdx()[global_row_iter];
  // 		DblNumMat &mat_chunk = my_dist_vec.LocalMap()[key]; 
  // 		  
  // 		// Assignment to local part of ScaLAPACK matrix
  // 		local_scala_mat_ptr[local_col_iter * local_scala_mat_height + local_row_iter] 
  // 		  = mat_chunk(global_row_iter - offset_map[key], global_col_iter);		  
  // 	      }
  // 	  }
  //       }
  // 	    
  // 	    
  // 
  //     // Clean up extra entries from Get-Begin / Get-End
  //     typename std::map<Index3, DblNumMat >::iterator delete_it;
  //     for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
  //       {
  // 	if(getKeys_list[delete_iter] != my_original_key) // Be careful about original key
  // 	  {  
  // 	    delete_it = my_dist_vec.LocalMap().find(getKeys_list[delete_iter]);
  // 	    (my_dist_vec.LocalMap()).erase(delete_it);
  // 	  }
  //       }
  //       
  //       
  //       
  //   
  //     
  // #ifndef _RELEASE_
  //     PopCallStack();
  // #endif  
  //   } // End of scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old
  //   
  
  
  // Given a block of eigenvectors (size:  hamDG.NumBasisTotal() * hamDG.NumStateTotal()),
  // convert this to ScaLAPACK format for subsequent use with ScaLAPACK routines
  // Should only be called by processors for which context > 0
  // This routine is largely based on the Hamiltonian to ScaLAPACK conversion routine which is similar
  void 
  SCFDG::scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
							 MPI_Comm comm,
							 dgdft::scalapack::Descriptor &my_scala_descriptor,
							 dgdft::scalapack::ScaLAPACKMatrix<Real>  &my_scala_mat)
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_Cheby_distmat_to_ScaLAPACK_conversion");
#endif
   
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    // Load up the important ScaLAPACK info
    int nprow = my_scala_descriptor.NpRow();
    int npcol = my_scala_descriptor.NpCol();
    int myprow = my_scala_descriptor.MypRow();
    int mypcol = my_scala_descriptor.MypCol();

    // Set the descriptor for the ScaLAPACK matrix
    my_scala_mat.SetDescriptor( my_scala_descriptor );
    
    // Save the original key for the distributed vector
    const Index3 my_original_key = (my_dist_mat.LocalMap().begin())->first;
    
    // Get some basic information set up
    Int mpirank, mpisize;
    MPI_Comm_rank( comm, &mpirank );
    MPI_Comm_size( comm, &mpisize );
    
    Int MB = my_scala_mat.MB(); 
    Int NB = my_scala_mat.NB();
    
    
    if( MB != NB )
      {
	throw std::runtime_error("MB must be equal to NB.");
      }

    // ScaLAPACK block information
    Int numRowBlock = my_scala_mat.NumRowBlocks();
    Int numColBlock = my_scala_mat.NumColBlocks();

    // Get the processor map
    IntNumMat  procGrid( nprow, npcol );
    SetValue( procGrid, 0 );
    {
      IntNumMat  procTmp( nprow, npcol );
      SetValue( procTmp, 0 );
      procTmp( myprow, mypcol ) = mpirank;
      mpi::Allreduce( procTmp.Data(), procGrid.Data(), nprow * npcol,
		      MPI_SUM, comm );
    }


    // ScaLAPACK block partition 
    BlockMatPrtn  blockPrtn;
    
    // Fill up the owner information
    IntNumMat&    blockOwner = blockPrtn.ownerInfo;
    blockOwner.Resize( numRowBlock, numColBlock );   
    for( Int jb = 0; jb < numColBlock; jb++ ){
      for( Int ib = 0; ib < numRowBlock; ib++ ){
	blockOwner( ib, jb ) = procGrid( ib % nprow, jb % npcol );
      }
    }

    // Distributed matrix in ScaLAPACK format
    DistVec<Index2, DblNumMat, BlockMatPrtn> distScaMat;
    distScaMat.Prtn() = blockPrtn;
    distScaMat.SetComm(comm);
	    
 
    // Zero out and initialize
    DblNumMat empty_mat( MB, MB ); // As MB = NB
    SetValue( empty_mat, 0.0 );
    // We need this loop here since we are initializing the distributed 
    // ScaLAPACK matrix. Subsequently, the LocalMap().begin()->first technique can be used
    for( Int jb = 0; jb < numColBlock; jb++ )
      for( Int ib = 0; ib < numRowBlock; ib++ )
	{
	  Index2 key( ib, jb );
	  if( distScaMat.Prtn().Owner( key ) == mpirank )
	    {
	      distScaMat.LocalMap()[key] = empty_mat;
	    }
	}
 
 
 
    // Copy data from DG distributed matrix to ScaLAPACK distributed matrix
    DblNumMat& localMat = (my_dist_mat.LocalMap().begin())->second;
    const std::vector<Int>& my_idx = hamDG.ElemBasisIdx()( my_original_key(0), my_original_key(1), my_original_key(2) );
   
    {
      Int ib, jb, io, jo;
      for( Int b = 0; b < localMat.n(); b++ )
	{
	  for( Int a = 0; a < localMat.m(); a++ )
	    {
	      ib = my_idx[a] / MB;
	      io = my_idx[a] % MB;
	    
	      jb = b / MB;            
	      jo = b % MB;
	    
	      typename std::map<Index2, DblNumMat >::iterator 
		ni = distScaMat.LocalMap().find( Index2(ib, jb) );
	      if( ni == distScaMat.LocalMap().end() )
		{
		  distScaMat.LocalMap()[Index2(ib, jb)] = empty_mat;
		  ni = distScaMat.LocalMap().find( Index2(ib, jb) );
		}
            
	      DblNumMat&  localScaMat = ni->second;
	      localScaMat(io, jo) += localMat(a, b);
	    } // for (a)
        } // for (b)

    }     
        
    // Communication step
    {
      // Prepare
      std::vector<Index2>  keyIdx;
      for( typename std::map<Index2, DblNumMat >::iterator 
	     mi  = distScaMat.LocalMap().begin();
	   mi != distScaMat.LocalMap().end(); mi++ )
	{
	  Index2 key = mi->first;
			
	  // Include all keys which do not reside on current processor
	  if( distScaMat.Prtn().Owner( key ) != mpirank )
	    {
	      keyIdx.push_back( key );
	    }
	} // for (mi)

      // Communication
      distScaMat.PutBegin( keyIdx, NO_MASK );
      distScaMat.PutEnd( NO_MASK, PutMode::COMBINE );
	    
      // Clean up
      std::vector<Index2>  eraseKey;
      for( typename std::map<Index2, DblNumMat >::iterator 
	     mi  = distScaMat.LocalMap().begin();
	   mi != distScaMat.LocalMap().end(); mi++)
	{
	  Index2 key = mi->first;
	  if( distScaMat.Prtn().Owner( key ) != mpirank )
	    {
	      eraseKey.push_back( key );
	    }
	} // for (mi)

      for( std::vector<Index2>::iterator vi = eraseKey.begin();
	   vi != eraseKey.end(); vi++ )
	{
	  distScaMat.LocalMap().erase( *vi );
	}	


    } // End of communication step
	
	
    // Final step: Copy from distributed ScaLAPACK matrix to local part of actual
    // ScaLAPACK matrix
    {
      for( typename std::map<Index2, DblNumMat>::iterator 
	     mi  = distScaMat.LocalMap().begin();
	   mi != distScaMat.LocalMap().end(); mi++ )
	{
	  Index2 key = mi->first;
	  if( distScaMat.Prtn().Owner( key ) == mpirank )
	    {
	      Int ib = key(0), jb = key(1);
	      Int offset = ( jb / npcol ) * MB * my_scala_mat.LocalLDim() + 
		( ib / nprow ) * MB;
	      lapack::Lacpy( 'A', MB, MB, mi->second.Data(),
			     MB, my_scala_mat.Data() + offset, my_scala_mat.LocalLDim() );
	    } // own this block
	} // for (mi)

    }
	

#ifndef _RELEASE_
    PopCallStack();
#endif  
  } // End of scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK
  
  
 
 
  void 
  SCFDG::scfdg_FirstChebyStep(Int MaxIter,
			      Int filter_order )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_FirstChebyStep");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
    
    Real timeSta, timeEnd;
    Real extra_timeSta, extra_timeEnd;
    Real cheby_timeSta, cheby_timeEnd;
    
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    
    // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
    // using the Lanczos estimator
       
    DblNumVec Lanczos_Ritz_values;
    
    
    statusOFS << std::endl << " Estimating the spectral bounds ... "; 
    GetTime( extra_timeSta );
    const int Num_Lanczos_Steps = 6;
    double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
    
    double a_L = Lanczos_Ritz_values(0);
    double beta = 0.5; // 0.5 <= beta < 1.0
    double b_low = beta * Lanczos_Ritz_values(0) + (1.0 - beta) * Lanczos_Ritz_values(Num_Lanczos_Steps - 1);
    //b_low = 0.0; // This is probably better than the above estimate based on Lanczos
    
    // Step 2: Initialize the eigenvectors to random numbers
    statusOFS << std::endl << " Initializing eigenvectors randomly on first SCF step ... "; 
 
    hamDG.EigvecCoef().Prtn() = elemPrtn_;
    hamDG.EigvecCoef().SetComm(domain_.colComm);
    
    GetTime( extra_timeSta );
    
    // This is one of the very few places where we use this loop 
    // for setting up the keys. In other cases, we can simply access
    // the first and second elements of the LocalMap of the distributed vector
    // once it has been set up.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  
	  // If the current processor owns this element
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){ 
	    
	    // Associate the current key with a vector that contains the stuff 
	    // that should reside on this process.
	    const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
	    hamDG.EigvecCoef().LocalMap()[key].Resize(idx.size(), hamDG.NumStateTotal()); 
	    
	    DblNumMat &ref_mat =  hamDG.EigvecCoef().LocalMap()[key];
	    
	    // Only the first processor on every column does this
	    if(mpirankRow == 0)	      
	      UniformRandom(ref_mat);
	    // Now broadcast this
	    MPI_Bcast(ref_mat.Data(), (ref_mat.m() * ref_mat.n()), MPI_DOUBLE, 0, domain_.rowComm);
	    
	  }
	}// End of eigenvector initialization
    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
    
            
    // Step 3: Main loop
    const Int Iter_Max = MaxIter;
    const Int Filter_Order = filter_order ;
    
    DblNumVec eig_vals_Raleigh_Ritz;
    
    for(Int i = 1; i <= Iter_Max; i ++)
      {
	GetTime( cheby_timeSta );
	statusOFS << std::endl << std::endl << " ------------------------------- ";
	statusOFS << std::endl << " First Chebyshev step iteration " << i << " of " << Iter_Max << " . ";
	// Filter the eigenvectors
	statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
	GetTime( timeSta );
	scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
	GetTime( timeEnd );
	statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";
      
	// Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
        // This can be done serially or using ScaLAPACK	
	if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
	  {
	    // Do the subspace problem serially 
	    statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";
	
	    // Orthonormalize using Cholesky factorization  
	    statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
	    GetTime( timeSta );
      
	  
	  
	    DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
	    DblNumMat square_mat;
	    DblNumMat temp_square_mat;
      
	    Int width = local_eigvec_mat.n();
	    Int height_local = local_eigvec_mat.m();
      
	    square_mat.Resize(width, width);
	    temp_square_mat.Resize(width, width);
      
	    SetValue(temp_square_mat, 0.0);
      
	    // Compute square_mat = X^T * X for Cholesky	
	    blas::Gemm( 'T', 'N', width, width, height_local, 
			1.0, local_eigvec_mat.Data(), height_local,
			local_eigvec_mat.Data(), height_local, 
			0.0, temp_square_mat.Data(), width );
      
	    SetValue( square_mat, 0.0 );
	    MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );
      
	    // In the following, reduction happens on colComm but the result is broadcast to everyone
	    // This can probably be band-parallelized
      
	    // Make the Cholesky factorization call on proc 0
	    if ( mpirank == 0) {
	      lapack::Potrf( 'U', width, square_mat.Data(), width );
	    }
	    // Send the Cholesky factor to every process
	    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);
      
	    // Do a solve with the Cholesky factor : Band parallelization ??
	    // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
	    blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
			local_eigvec_mat.Data(), height_local );
	
	
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
	    GetTime( timeSta );
      
	    // Compute H * X
	    DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
	    scfdg_Hamiltonian_times_eigenvectors(result_mat);
	    DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;
      
	    SetValue(temp_square_mat, 0.0);
      
	    // Compute square_mat = X^T * HX 
	    blas::Gemm( 'T', 'N', width, width, height_local, 
			1.0, local_eigvec_mat.Data(), height_local,
			local_result_mat.Data(), height_local, 
			0.0, temp_square_mat.Data(), width );
      
	    SetValue( square_mat, 0.0 );
	    MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );
          
      
	    eig_vals_Raleigh_Ritz.Resize(width);
	    SetValue(eig_vals_Raleigh_Ritz, 0.0);
      
	    if ( mpirank == 0 ) {
	      lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
	   
	    }
	
	    // ~~ Send the results to every process
	    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
	    MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values
	
      
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
	    GetTime( timeSta );
      
	    // ~~ So copy X to HX 
	    lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );
	
	    // ~~ Gemm: X <-- HX (= X) * Q
	    blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
			height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );	
      
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Reset the filtering bounds using results of the Raleigh-Ritz step
	    b_low = eig_vals_Raleigh_Ritz(width - 1);
	    a_L = eig_vals_Raleigh_Ritz(0);
	    
	    // Fill up the eigen-values
	    DblNumVec& eigval = hamDG.EigVal(); 
	    eigval.Resize( hamDG.NumStateTotal() );	
	    
	    for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
	      eigval[i] =  eig_vals_Raleigh_Ritz[i];

	    
	
	  } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
	else
	  {
	    // Do the subspace problems using ScaLAPACK
	    statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";
	    
	    statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
	    GetTime( timeSta );
	     
	    // Basic ScaLAPACK setup steps
	    //Number of ScaLAPACK processors equal to number of DG elements
	    const int num_cheby_scala_procs = mpisizeCol; 
	  
	    // Figure out the process grid dimensions
	    int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
	    while(num_cheby_scala_procs % temp_factor != 0 )
	      ++temp_factor;
    
	    // temp_factor now contains the process grid height
	    const int cheby_scala_num_rows = temp_factor;	  
	    const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;
     
         	  
	    
	    // Set up the ScaLAPACK context
	    IntNumVec cheby_scala_pmap(num_cheby_scala_procs);
	  
	    // Use the first processor from every DG-element 
	    for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
	      cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 
	  
	    // Set up BLACS for subsequent ScaLAPACK operations
	    Int cheby_scala_context = -2;
	    dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
	    dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);
	  
	    // Figure out my ScaLAPACK information
	    int dummy_np_row, dummy_np_col;
	    int my_cheby_scala_proc_row, my_cheby_scala_proc_col;
	  
	    dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);
	  
	  
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	    
	    statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
	    statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;
	  
	    // Eigenvcetors in ScaLAPACK format : this will be used multiple times
	    dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked
	        
	    if(cheby_scala_context >= 0)
	      { 
		// Now setup the ScaLAPACK matrix
		statusOFS << std::endl << " Orthonormalization step:";
		statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
		GetTime( timeSta );
		
		// The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
		// But this is not verified here as far as the distributed vector is concerned
		dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
								scaBlockSize_, scaBlockSize_, 
								0, 0, 
								cheby_scala_context);
	    
		

		// Make the conversion call 		
		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
								domain_.colComm,
								cheby_eigvec_desc,
								cheby_scala_eigvecs_X);
		
		
		// 		//Older version of conversion call
		// 		// Store the important ScaLAPACK information
		// 		std::vector<int> my_cheby_scala_info;
		// 		my_cheby_scala_info.resize(4,0);
		// 		my_cheby_scala_info[0] = cheby_scala_num_rows;
		// 		my_cheby_scala_info[1] = cheby_scala_num_cols;
		// 		my_cheby_scala_info[2] = my_cheby_scala_proc_row;
		// 		my_cheby_scala_info[3] = my_cheby_scala_proc_col;
		// 	  

		// 		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
		// 								    my_cheby_scala_info,
		// 								    cheby_eigvec_desc,
		// 								    cheby_scala_eigvecs_X);
		// 		

		
		

		
		GetTime( timeEnd );
	        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
		
		statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
		GetTime( timeSta );
      
	  
		// Compute C = X^T * X
		dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
							      scaBlockSize_, scaBlockSize_, 
							      0, 0, 
							      cheby_scala_context);
		
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
		cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);
		
		dgdft::scalapack::Gemm( 'T', 'N',
					hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
					1.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
					0.0,
					cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
					cheby_scala_context);
	    
		

		
		// Compute V = Chol(C)
		dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

		// Compute  X = X * V^{-1}
		dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
					cheby_scala_chol_mat, 
					cheby_scala_eigvecs_X );
		
		GetTime( timeEnd );
	        statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;
      
		statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
		GetTime( timeSta );
		
		// Now convert this back to DG-distributed matrix format
		ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
				   elemPrtn_,
				   hamDG.EigvecCoef(),
				   hamDG.ElemBasisIdx(), 
				   domain_.colComm, 
				   hamDG.NumStateTotal() );
		
		GetTime( timeEnd );
	        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
				
	      } // End of  if(cheby_scala_context >= 0)
	    else
	      {
		statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
	      }		
		
	    // Communicate the orthonormalized eigenvectors (to other intra-element processors)
	    statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
	    GetTime( timeSta );
	   
	    DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
	    MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);
	    
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;
	    
	    // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
	    DblNumVec& eigval = hamDG.EigVal(); 
	    eigval.Resize( hamDG.NumStateTotal() );	
	    
	    
	    // Compute H * X
	    statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors: ";
	    GetTime( timeSta );
	    
	    DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
	    scfdg_Hamiltonian_times_eigenvectors(result_mat);
	    
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	  
	    // Raleigh-Ritz step
	    if(cheby_scala_context >= 0)
	      { 
		statusOFS << std::endl << std::endl << " Raleigh - Ritz step:";
	      

		// Convert HX to ScaLAPACK format
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
		dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
							    scaBlockSize_, scaBlockSize_, 
							    0, 0, 
							    cheby_scala_context);
	    
		
		statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
		GetTime( timeSta );
	      
		
		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
								domain_.colComm,
								cheby_HX_desc,
								cheby_scala_HX);
		
		
	      
	      
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
		statusOFS << std::endl << " Solving the subspace problem ... ";
		GetTime( timeSta );
       
		dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
							      scaBlockSize_, scaBlockSize_, 
							      0, 0, 
							      cheby_scala_context);
		
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
		cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);
		
		dgdft::scalapack::Gemm( 'T', 'N',
					hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
					1.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
					cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
					0.0,
					cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
					cheby_scala_context);
	    
	
		scalapack::ScaLAPACKMatrix<Real>  scaZ;

		std::vector<Real> eigen_values;
	      
		// Eigenvalue probem solution call
		scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);
	     	      
		// Copy the eigenvalues to the Hamiltonian object	      
		for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
		  eigval[i] = eigen_values[i];


		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	      
		// Subspace rotation step : X <- X * Q
		statusOFS << std::endl << " Subspace rotation step ... ";
		GetTime( timeSta );
	      
		// To save memory, copy X to HX
		blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
			   cheby_scala_eigvecs_X.Data(),
			   1,
			   cheby_scala_HX.Data(),
			   1);
	      
		// Now perform X <- HX (=X) * Q (=scaZ)
		dgdft::scalapack::Gemm( 'N', 'N',
					hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
					1.0,
					cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
					scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
					0.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
					cheby_scala_context);
	    
	      
	      
	      
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
	      
		// Convert the eigenvectors back to distributed vector format
		statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
		GetTime( timeSta );
		
		// Now convert this back to DG-distributed matrix format
		ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
				   elemPrtn_,
				   hamDG.EigvecCoef(),
				   hamDG.ElemBasisIdx(), 
				   domain_.colComm, 
				   hamDG.NumStateTotal() );
		
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	      
    
	      }  // End of if(cheby_scala_context >= 0)
	   
	    // Communicate the final eigenvectors (to other intra-element processors)
	    statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
	    GetTime( timeSta );
	   
	    DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
	    MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);
	    
	    // Communicate the final eigenvalues (to other intra-element processors)
	    MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values
	  
	  
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

	  
	  
	  
	    // Reset the filtering bounds using results of the Raleigh-Ritz step	
	    b_low = eigval(ref_mat_2.n() - 1);
	    a_L = eigval(0);

	    MPI_Barrier(domain_.rowComm);
	    MPI_Barrier(domain_.colComm);
	    MPI_Barrier(domain_.comm);
	    
	    // Clean up BLACS
	    if(cheby_scala_context >= 0) {
	      dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
	    }
    

	    
	  
	  } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 
	
	
      
	statusOFS << std::endl << " ------------------------------- ";
	GetTime( cheby_timeEnd );
	
	//statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
	//statusOFS << eig_vals_Raleigh_Ritz << std::endl;
	
	statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
	statusOFS << std::endl << " ------------------------------- " << std::endl;
            
      } // End of loop over inner iteration repeats
 
    
    
#ifndef _RELEASE_
    PopCallStack();
#endif  
    
   
  }
  
  void 
  SCFDG::scfdg_GeneralChebyStep(Int MaxIter, 
				Int filter_order )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::scfdg_GeneralChebyStep");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
    
    Real timeSta, timeEnd;
    Real extra_timeSta, extra_timeEnd;
    Real cheby_timeSta, cheby_timeEnd;
    
    HamiltonianDG&  hamDG = *hamDGPtr_;
    
    DblNumVec& eigval = hamDG.EigVal(); 
    
    
    // Step 1: Obtain the upper bound and the Ritz values (for the lower bound)
    // using the Lanczos estimator
       
    DblNumVec Lanczos_Ritz_values;
    
    
    statusOFS << std::endl << " Estimating the spectral bounds ... "; 
    GetTime( extra_timeSta );
    const int Num_Lanczos_Steps = 6;
    double b_up = scfdg_Cheby_Upper_bound_estimator(Lanczos_Ritz_values, Num_Lanczos_Steps);  
    GetTime( extra_timeEnd );
    statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
    
    double a_L = eigval[0];
    double b_low = eigval[hamDG.NumStateTotal() - 1];
    
    // Step 2: Main loop
    const Int Iter_Max = MaxIter;
    const Int Filter_Order = filter_order ;
    
    DblNumVec eig_vals_Raleigh_Ritz;
    
    for(Int i = 1; i <= Iter_Max; i ++)
      {
	GetTime( cheby_timeSta );
	statusOFS << std::endl << std::endl << " ------------------------------- ";
	statusOFS << std::endl << " General Chebyshev step iteration " << i << " of " << Iter_Max << " . ";
      
	// Filter the eigenvectors
	statusOFS << std::endl << std::endl << " Filtering the eigenvectors ... (Filter order = " << Filter_Order << ")";
	GetTime( timeSta );
	scfdg_Chebyshev_filter_scaled(Filter_Order, b_low, b_up, a_L);
	GetTime( timeEnd );
	statusOFS << std::endl << " Filtering completed. ( " << (timeEnd - timeSta ) << " s.)";
      

	// Subspace projected problems: Orthonormalization, Raleigh-Ritz and subspace rotation steps
        // This can be done serially or using ScaLAPACK	
	if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
	  {
	    // Do the subspace problem serially 
	    statusOFS << std::endl << std::endl << " Solving subspace problems serially ...";
	
	    // Orthonormalize using Cholesky factorization  
	    statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
	    GetTime( timeSta );
      
	  
	  
	    DblNumMat &local_eigvec_mat = (hamDG.EigvecCoef().LocalMap().begin())->second;
	    DblNumMat square_mat;
	    DblNumMat temp_square_mat;
      
	    Int width = local_eigvec_mat.n();
	    Int height_local = local_eigvec_mat.m();
      
	    square_mat.Resize(width, width);
	    temp_square_mat.Resize(width, width);
      
	    SetValue(temp_square_mat, 0.0);
      
	    // Compute square_mat = X^T * X for Cholesky	
	    blas::Gemm( 'T', 'N', width, width, height_local, 
			1.0, local_eigvec_mat.Data(), height_local,
			local_eigvec_mat.Data(), height_local, 
			0.0, temp_square_mat.Data(), width );
      
	    SetValue( square_mat, 0.0 );
	    MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );
      
	    // In the following, reduction happens on colComm but the result is broadcast to everyone
	    // This can probably be band-parallelized
      
	    // Make the Cholesky factorization call on proc 0
	    if ( mpirank == 0) {
	      lapack::Potrf( 'U', width, square_mat.Data(), width );
	    }
	    // Send the Cholesky factor to every process
	    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm);
      
	    // Do a solve with the Cholesky factor : Band parallelization ??
	    // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
	    blas::Trsm( 'R', 'U', 'N', 'N', height_local, width, 1.0, square_mat.Data(), width, 
			local_eigvec_mat.Data(), height_local );
	
	
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Raleigh-Ritz step: This part is non-scalable and needs to be fixed
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    statusOFS << std::endl << std::endl << " Raleigh-Ritz step ... ";
	    GetTime( timeSta );
      
	    // Compute H * X
	    DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
	    scfdg_Hamiltonian_times_eigenvectors(result_mat);
	    DblNumMat &local_result_mat = (result_mat.LocalMap().begin())->second;
      
	    SetValue(temp_square_mat, 0.0);
      
	    // Compute square_mat = X^T * HX 
	    blas::Gemm( 'T', 'N', width, width, height_local, 
			1.0, local_eigvec_mat.Data(), height_local,
			local_result_mat.Data(), height_local, 
			0.0, temp_square_mat.Data(), width );
      
	    SetValue( square_mat, 0.0 );
	    MPI_Allreduce( temp_square_mat.Data(), square_mat.Data(), width*width, MPI_DOUBLE, MPI_SUM, domain_.colComm );
          
      
	    eig_vals_Raleigh_Ritz.Resize(width);
	    SetValue(eig_vals_Raleigh_Ritz, 0.0);
      
	    if ( mpirank == 0 ) {
	      lapack::Syevd( 'V', 'U', width, square_mat.Data(), width, eig_vals_Raleigh_Ritz.Data() );
	   
	    }
	
	    // ~~ Send the results to every process
	    MPI_Bcast(square_mat.Data(), width*width, MPI_DOUBLE, 0, domain_.comm); // Eigen-vectors
	    MPI_Bcast(eig_vals_Raleigh_Ritz.Data(), width, MPI_DOUBLE, 0,  domain_.comm); // Eigen-values
	
      
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Raleigh-Ritz step completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Subspace rotation step X <- X * Q: This part is non-scalable and needs to be fixed
	    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    statusOFS << std::endl << std::endl << " Subspace rotation step ... ";
	    GetTime( timeSta );
      
	    // ~~ So copy X to HX 
	    lapack::Lacpy( 'A', height_local, width, local_eigvec_mat.Data(),  height_local, local_result_mat.Data(), height_local );
	
	    // ~~ Gemm: X <-- HX (= X) * Q
	    blas::Gemm( 'N', 'N', height_local, width, width, 1.0, local_result_mat.Data(),
			height_local, square_mat.Data(), width, 0.0, local_eigvec_mat.Data(), height_local );	
      
	    GetTime( timeEnd );
	    statusOFS << std::endl << " Subspace rotation step completed ( " << (timeEnd - timeSta ) << " s.)";
      
	    // Reset the filtering bounds using results of the Raleigh-Ritz step
	    b_low = eig_vals_Raleigh_Ritz(width - 1);
	    a_L = eig_vals_Raleigh_Ritz(0);
	    
	    // Fill up the eigen-values
	    DblNumVec& eigval = hamDG.EigVal(); 
	    eigval.Resize( hamDG.NumStateTotal() );	
	    
	    for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
	      eigval[i] =  eig_vals_Raleigh_Ritz[i];

	    
	
	  } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0)
	else
	  {
	    // Do the subspace problems using ScaLAPACK
	    statusOFS << std::endl << std::endl << " Solving subspace problems using ScaLAPACK ...";
	    
	    statusOFS << std::endl << " Setting up BLACS and ScaLAPACK Process Grid ...";
	    GetTime( timeSta );
	     
	    // Basic ScaLAPACK setup steps
	    //Number of ScaLAPACK processors equal to number of DG elements
	    const int num_cheby_scala_procs = mpisizeCol; 
	  
	    // Figure out the process grid dimensions
	    int temp_factor = int(sqrt(double(num_cheby_scala_procs)));
	    while(num_cheby_scala_procs % temp_factor != 0 )
	      ++temp_factor;
    
	    // temp_factor now contains the process grid height
	    const int cheby_scala_num_rows = temp_factor;	  
	    const int cheby_scala_num_cols = num_cheby_scala_procs / temp_factor;
     
         	  
	    
	    // Set up the ScaLAPACK context
	    IntNumVec cheby_scala_pmap(num_cheby_scala_procs);
	  
	    // Use the first processor from every DG-element 
	    for ( Int pmap_iter = 0; pmap_iter < num_cheby_scala_procs; pmap_iter++ )
	      cheby_scala_pmap[pmap_iter] = pmap_iter * mpisizeRow; 
	  
	    // Set up BLACS for subsequent ScaLAPACK operations
	    Int cheby_scala_context = -2;
	    dgdft::scalapack::Cblacs_get( 0, 0, &cheby_scala_context );
	    dgdft::scalapack::Cblacs_gridmap(&cheby_scala_context, &cheby_scala_pmap[0], cheby_scala_num_rows, cheby_scala_num_rows, cheby_scala_num_cols);
	  
	    // Figure out my ScaLAPACK information
	    int dummy_np_row, dummy_np_col;
	    int my_cheby_scala_proc_row, my_cheby_scala_proc_col;
	  
	    dgdft::scalapack::Cblacs_gridinfo(cheby_scala_context, &dummy_np_row, &dummy_np_col, &my_cheby_scala_proc_row, &my_cheby_scala_proc_col);
	  
	  
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	    
	    statusOFS << std::endl << " ScaLAPACK will use " << num_cheby_scala_procs << " processes.";
	    statusOFS << std::endl << " ScaLAPACK process grid = " << cheby_scala_num_rows << " * " << cheby_scala_num_cols << " ."  << std::endl;
	  
	    // Eigenvcetors in ScaLAPACK format : this will be used multiple times
	    dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_eigvecs_X; // Declared here for scope, empty constructor invoked
	        
	    if(cheby_scala_context >= 0)
	      { 
		// Now setup the ScaLAPACK matrix
		statusOFS << std::endl << " Orthonormalization step:";
		statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
		GetTime( timeSta );
		
		// The dimensions should be  hamDG.NumBasisTotal() * hamDG.NumStateTotal()
		// But this is not verified here as far as the distributed vector is concerned
		dgdft::scalapack::Descriptor cheby_eigvec_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
								scaBlockSize_, scaBlockSize_, 
								0, 0, 
								cheby_scala_context);
	    
		

		// Make the conversion call 		
		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(hamDG.EigvecCoef(),
								domain_.colComm,
								cheby_eigvec_desc,
								cheby_scala_eigvecs_X);
		
		
		// 		//Older version of conversion call
		// 		// Store the important ScaLAPACK information
		// 		std::vector<int> my_cheby_scala_info;
		// 		my_cheby_scala_info.resize(4,0);
		// 		my_cheby_scala_info[0] = cheby_scala_num_rows;
		// 		my_cheby_scala_info[1] = cheby_scala_num_cols;
		// 		my_cheby_scala_info[2] = my_cheby_scala_proc_row;
		// 		my_cheby_scala_info[3] = my_cheby_scala_proc_col;
		// 	  

		// 		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK_old(hamDG.EigvecCoef(),
		// 								    my_cheby_scala_info,
		// 								    cheby_eigvec_desc,
		// 								    cheby_scala_eigvecs_X);
		// 		

		
		GetTime( timeEnd );
	        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
		
		statusOFS << std::endl << " Orthonormalizing filtered vectors ... ";
		GetTime( timeSta );
      
	  
		// Compute C = X^T * X
		dgdft::scalapack::Descriptor cheby_chol_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
							      scaBlockSize_, scaBlockSize_, 
							      0, 0, 
							      cheby_scala_context);
		
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_chol_mat;
		cheby_scala_chol_mat.SetDescriptor(cheby_chol_desc);
		
		dgdft::scalapack::Gemm( 'T', 'N',
					hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
					1.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
					0.0,
					cheby_scala_chol_mat.Data(), I_ONE, I_ONE, cheby_scala_chol_mat.Desc().Values(),
					cheby_scala_context);
	    
		

		
		// Compute V = Chol(C)
		dgdft::scalapack::Potrf( 'U', cheby_scala_chol_mat);

		// Compute  X = X * V^{-1}
		dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
					cheby_scala_chol_mat, 
					cheby_scala_eigvecs_X );
		
		GetTime( timeEnd );
	        statusOFS << std::endl << " Orthonormalization completed ( " << (timeEnd - timeSta ) << " s.)" << std::endl;
      
		statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
		GetTime( timeSta );
		
		// Now convert this back to DG-distributed matrix format
		ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
				   elemPrtn_,
				   hamDG.EigvecCoef(),
				   hamDG.ElemBasisIdx(), 
				   domain_.colComm, 
				   hamDG.NumStateTotal() );
		
		GetTime( timeEnd );
	        statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
				
	      } // End of  if(cheby_scala_context >= 0)
	    else
	      {
		statusOFS << std::endl << " Waiting for ScaLAPACK solution of subspace problems ...";
	      }	
		
	    // Communicate the orthonormalized eigenvectors (to other intra-element processors)
	    statusOFS << std::endl << " Communicating orthonormalized filtered vectors ... ";
	    GetTime( timeSta );
	   
	    DblNumMat &ref_mat_1 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
	    MPI_Bcast(ref_mat_1.Data(), (ref_mat_1.m() * ref_mat_1.n()), MPI_DOUBLE, 0, domain_.rowComm);
	    
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)" << std::endl;
	    
	    // Set up space for eigenvalues in the Hamiltonian object for results of Raleigh-Ritz step
	    DblNumVec& eigval = hamDG.EigVal(); 
	    eigval.Resize( hamDG.NumStateTotal() );	
	  
	    // Compute H * X
	    statusOFS << std::endl << " Computing H * X for filtered orthonormal vectors:";
	    GetTime( timeSta );
	    
	    DistVec<Index3, DblNumMat, ElemPrtn>  result_mat;
	    scfdg_Hamiltonian_times_eigenvectors(result_mat);
	    
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	    
	    
	    // Raleigh-Ritz step
	    if(cheby_scala_context >= 0)
	      { 
		statusOFS << std::endl << std::endl << " Raleigh - Ritz step:";
	      
		// Convert HX to ScaLAPACK format
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_HX;
		dgdft::scalapack::Descriptor cheby_HX_desc( hamDG.NumBasisTotal(), hamDG.NumStateTotal(), 
							    scaBlockSize_, scaBlockSize_, 
							    0, 0, 
							    cheby_scala_context);
	    
		
		statusOFS << std::endl << " Distributed vector to ScaLAPACK format conversion ... ";
		GetTime( timeSta );
	      
		// Make the conversion call 				
		scfdg_Cheby_convert_eigvec_distmat_to_ScaLAPACK(result_mat,
								domain_.colComm,
								cheby_HX_desc,
								cheby_scala_HX);
		
	      
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
		statusOFS << std::endl << " Solving the subspace problem ... ";
		GetTime( timeSta );
       
		dgdft::scalapack::Descriptor cheby_XTHX_desc( hamDG.NumStateTotal(), hamDG.NumStateTotal(), 
							      scaBlockSize_, scaBlockSize_, 
							      0, 0, 
							      cheby_scala_context);
		
		dgdft::scalapack::ScaLAPACKMatrix<Real>  cheby_scala_XTHX_mat;
		cheby_scala_XTHX_mat.SetDescriptor(cheby_XTHX_desc);
		
		dgdft::scalapack::Gemm( 'T', 'N',
					hamDG.NumStateTotal(), hamDG.NumStateTotal(), hamDG.NumBasisTotal(),
					1.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(), 
					cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(),
					0.0,
					cheby_scala_XTHX_mat.Data(), I_ONE, I_ONE, cheby_scala_XTHX_mat.Desc().Values(),
					cheby_scala_context);
	    
	
		scalapack::ScaLAPACKMatrix<Real>  scaZ;

		std::vector<Real> eigen_values;
	      
		// Eigenvalue probem solution call
		scalapack::Syevd('U', cheby_scala_XTHX_mat, eigen_values, scaZ);
	     	      
		// Copy the eigenvalues to the Hamiltonian object	      
		for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
		  eigval[i] = eigen_values[i];


		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	      
		// Subspace rotation step : X <- X * Q
		statusOFS << std::endl << " Subspace rotation step ... ";
		GetTime( timeSta );
	      
		// To save memory, copy X to HX
		blas::Copy((cheby_scala_eigvecs_X.LocalHeight() * cheby_scala_eigvecs_X.LocalWidth()), 
			   cheby_scala_eigvecs_X.Data(),
			   1,
			   cheby_scala_HX.Data(),
			   1);
	      
		// Now perform X <- HX (=X) * Q (=scaZ)
		dgdft::scalapack::Gemm( 'N', 'N',
					hamDG.NumBasisTotal(), hamDG.NumStateTotal(), hamDG.NumStateTotal(),
					1.0,
					cheby_scala_HX.Data(), I_ONE, I_ONE, cheby_scala_HX.Desc().Values(), 
					scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
					0.0,
					cheby_scala_eigvecs_X.Data(), I_ONE, I_ONE, cheby_scala_eigvecs_X.Desc().Values(),
					cheby_scala_context);
	    
	      
	      
	      
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
		
	      
		// Convert the eigenvectors back to distributed vector format
		statusOFS << std::endl << " ScaLAPACK to Distributed vector format conversion ... ";
		GetTime( timeSta );
		
		// Now convert this back to DG-distributed matrix format
		ScaMatToDistNumMat(cheby_scala_eigvecs_X ,
				   elemPrtn_,
				   hamDG.EigvecCoef(),
				   hamDG.ElemBasisIdx(), 
				   domain_.colComm, 
				   hamDG.NumStateTotal() );
		
		GetTime( timeEnd );
		statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";
	      
    
	      } // End of if(cheby_scala_context >= 0)
	      	

	   
	    // Communicate the final eigenvectors (to other intra-element processors)
	    statusOFS << std::endl << " Communicating eigenvalues and eigenvectors ... ";
	    GetTime( timeSta );
	   
	    DblNumMat &ref_mat_2 =  (hamDG.EigvecCoef().LocalMap().begin())->second;
	    MPI_Bcast(ref_mat_2.Data(), (ref_mat_2.m() * ref_mat_2.n()), MPI_DOUBLE, 0, domain_.rowComm);
	    
	    // Communicate the final eigenvalues (to other intra-element processors)
	    MPI_Bcast(eigval.Data(), ref_mat_2.n(), MPI_DOUBLE, 0,  domain_.rowComm); // Eigen-values
	  
	  
	    GetTime( timeEnd );
	    statusOFS << " Done. ( " << (timeEnd - timeSta ) << " s.)";

	  
	  
	  
	    // Reset the filtering bounds using results of the Raleigh-Ritz step	
	    b_low = eigval(ref_mat_2.n() - 1);
	    a_L = eigval(0);

	    MPI_Barrier(domain_.rowComm);
	    MPI_Barrier(domain_.colComm);
	    MPI_Barrier(domain_.comm);
	    
	    // Clean up BLACS
	    if(cheby_scala_context >= 0) {
	      dgdft::scalapack::Cblacs_gridexit( cheby_scala_context );
	    }
    

	  
	  } // End of if(SCFDG_Cheby_use_ScaLAPACK_ == 0) ... else 
	
	// Display the eigenvalues 
	statusOFS << std::endl << " ------------------------------- ";
	GetTime( cheby_timeEnd );
	
	//statusOFS << std::endl << " Eigenvalues via Chebyshev filtering : " << std::endl;
	//statusOFS << eig_vals_Raleigh_Ritz << std::endl;
	
	statusOFS << std::endl << " This Chebyshev cycle took a total of " << (cheby_timeEnd - cheby_timeSta ) << " s.";
	statusOFS << std::endl << " ------------------------------- " << std::endl;
            
      } // End of loop over inner iteration repeats
    
#ifndef _RELEASE_
    PopCallStack();
#endif  
    
  }
  
 



  void
  SCFDG::InnerIterate	( Int outerIter )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::InnerIterate");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

    Real timeSta, timeEnd;
    Real timeIterStart, timeIterEnd;

    HamiltonianDG&  hamDG = *hamDGPtr_;

    bool isInnerSCFConverged = false;

    for( Int innerIter = 1; innerIter <= scfInnerMaxIter_; innerIter++ ){
      if ( isInnerSCFConverged ) break;
      scfTotalInnerIter_++;

      GetTime( timeIterStart );

      statusOFS << std::endl << "Inner SCF iteration #"  
		<< innerIter << " starts." << std::endl << std::endl;


      // *********************************************************************
      // Update potential and construct/update the DG matrix
      // *********************************************************************

      if( innerIter == 1 ){
	// The first inner iteration does not update the potential, and
	// construct the global Hamiltonian matrix from scratch
	GetTime(timeSta);
      

	MPI_Barrier( domain_.comm );
	MPI_Barrier( domain_.rowComm );
	MPI_Barrier( domain_.colComm );

	hamDG.CalculateDGMatrix( );
   
	MPI_Barrier( domain_.comm );
	MPI_Barrier( domain_.rowComm );
	MPI_Barrier( domain_.colComm );
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for constructing the DG matrix is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }
      else{
	// The consequent inner iterations update the potential in the
	// element, and only update the global Hamiltonian matrix
			
	// Update the potential in the element (and the extended element)

     
	GetTime(timeSta);

	// Save the old potential on the LGL grid
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		Index3 numLGLGrid     = hamDG.NumLGLGridElem();
		blas::Copy( numLGLGrid.prod(),
			    hamDG.VtotLGL().LocalMap()[key].Data(), 1,
			    vtotLGLSave_.LocalMap()[key].Data(), 1 );
	      } // if (own this element)
	    } // for (i)

      
	// Update the local potential on the extended element and on the
	// element.
	UpdateElemLocalPotential();


	// Save the difference of the potential on the LGL grid into vtotLGLSave_
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		Index3 numLGLGrid     = hamDG.NumLGLGridElem();
		Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
		Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
		for( Int p = 0; p < numLGLGrid.prod(); p++ ){
		  (*ptrDif) = (*ptrNew) - (*ptrDif);
		  ptrNew++;
		  ptrDif++;
		} 
	      } // if (own this element)
	    } // for (i)

      
     
	MPI_Barrier( domain_.comm );
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for updating the local potential in the extended element and the element is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


	// Update the DG Matrix
	GetTime(timeSta);
	hamDG.UpdateDGMatrix( vtotLGLSave_ );
	MPI_Barrier( domain_.comm );
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for updating the DG matrix is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      } // if ( innerIter == 1 )

#if ( _DEBUGlevel_ >= 2 )
      {
	statusOFS << "Owned H matrix blocks on this processor" << std::endl;
	for( std::map<ElemMatKey, DblNumMat>::iterator 
	       mi  = hamDG.HMat().LocalMap().begin();
	     mi != hamDG.HMat().LocalMap().end(); mi++ ){
	  ElemMatKey key = (*mi).first;
	  statusOFS << key.first << " -- " << key.second << std::endl;
	}
      }
#endif


      // *********************************************************************
      // Write the Hamiltonian matrix to a file (if needed) 
      // *********************************************************************

      if( isOutputHMatrix_ ){
	// Only the first processor column participates in the conversion
	if( mpirankRow == 0 ){
	  DistSparseMatrix<Real>  HSparseMat;

	  GetTime(timeSta);
	  DistElemMatToDistSparseMat( 
				     hamDG.HMat(),
				     hamDG.NumBasisTotal(),
				     HSparseMat,
				     hamDG.ElemBasisIdx(),
				     domain_.colComm );
	  GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	  GetTime(timeSta);
	  ParaWriteDistSparseMatrix( "H.csc", HSparseMat );
	  //			WriteDistSparseMatrixFormatted( "H.matrix", HSparseMat );
	  GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for writing the matrix in parallel is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	MPI_Barrier( domain_.comm );

      }


      // *********************************************************************
      // Evaluate the density matrix
      // 
      // This can be done either using diagonalization method or using PEXSI
      // *********************************************************************

      // Save the mixing variable first
      {
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		if( mixVariable_ == "density" ){
		  DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
		  DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
		  blas::Copy( oldVec.Size(), oldVec.Data(), 1,
			      newVec.Data(), 1 );
		}
		else if( mixVariable_ == "potential" ){
		  DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
		  DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
		  blas::Copy( oldVec.Size(), oldVec.Data(), 1,
			      newVec.Data(), 1 );
		}

	      } // own this element
	    } // for (i)
      }



      // Method 1: Using diagonalization method
      //
      // FIXME The diagonalization procedure is now only performed on each
      // processor column.  A better implementation should 
      //
      // 1) Convert the HMat_ to a distributed ScaLAPACK matrix involving
      // all (or a given number of) processors
      //
      // 2) Diagonalize using all (or a given number of) processors.
      //
      // 3) Convert the eigenfunction matrices to the format that is
      // distributed among all processors.
      if( solutionMethod_ == "diag" && 0 ){
	{
	  GetTime(timeSta);
	  Int sizeH = hamDG.NumBasisTotal();
        
	  DblNumVec& eigval = hamDG.EigVal(); 
	  eigval.Resize( hamDG.NumStateTotal() );		

	  for( Int k = 0; k < numElem_[2]; k++ )
	    for( Int j = 0; j < numElem_[1]; j++ )
	      for( Int i = 0; i < numElem_[0]; i++ ){
		Index3 key( i, j, k );
		if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		  const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
		  DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
		  localCoef.Resize( idx.size(), hamDG.NumStateTotal() );		
		}
	      } 
        

	  if(contxt_ >= 0){

	    scalapack::Descriptor descH( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
					 0, 0, contxt_ );

	    scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;

	    std::vector<Real> eigs;

	    DistElemMatToScaMat( hamDG.HMat(), 	descH,
				 scaH, hamDG.ElemBasisIdx(), domain_.colComm );

	    scalapack::Syevd('U', scaH, eigs, scaZ);

	    //DblNumVec& eigval = hamDG.EigVal(); 
	    //eigval.Resize( hamDG.NumStateTotal() );		
	    for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
	      eigval[i] = eigs[i];



	    ScaMatToDistNumMat( scaZ, hamDG.Density().Prtn(), 
				hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.colComm, 
				hamDG.NumStateTotal() );

	  } //if(contxt_ >= 0)

	  MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 0, domain_.rowComm);
        
	  for( Int k = 0; k < numElem_[2]; k++ )
	    for( Int j = 0; j < numElem_[1]; j++ )
	      for( Int i = 0; i < numElem_[0]; i++ ){
		Index3 key( i, j, k );
		if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		  DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
		  MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), MPI_DOUBLE, 0, domain_.rowComm);
		}
	      } 
        
       
	  MPI_Barrier( domain_.comm );
	  MPI_Barrier( domain_.rowComm );
	  MPI_Barrier( domain_.colComm );
        
	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for diagonalizing the DG matrix using ScaLAPACK is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	}

	// Post processing

	Evdw_ = 0.0;

	// Compute the occupation rate
	CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

	// Compute the Harris energy functional.  
	// NOTE: In computing the Harris energy, the density and the
	// potential must be the INPUT density and potential without ANY
	// update.
	CalculateHarrisEnergy();

	MPI_Barrier( domain_.comm );
	MPI_Barrier( domain_.rowComm );
	MPI_Barrier( domain_.colComm );

	// Compute the output electron density
	GetTime( timeSta );

	// Calculate the new electron density
	// FIXME 
	// Do not need the conversion from column to row partition as well
	hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );

	MPI_Barrier( domain_.comm );
	MPI_Barrier( domain_.rowComm );
	MPI_Barrier( domain_.colComm );
      
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for computing density in the global domain is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	// Update the output potential, and the KS and second order accurate
	// energy
	{
	  // Update the Hartree energy and the exchange correlation energy and
	  // potential for computing the KS energy and the second order
	  // energy.
	  // NOTE Vtot should not be updated until finishing the computation
	  // of the energies.

	  if( XCType_ == "XC_GGA_XC_PBE" ){
	    hamDG.CalculateGradDensity(  *distfftPtr_ );
	  }

	  GetTime( timeSta );
	  hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for computing Exc in the global domain is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
     
	  GetTime( timeSta );

	  hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for computing Vhart in the global domain is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        
       
	  // Compute the second order accurate energy functional.
	  // NOTE: In computing the second order energy, the density and the
	  // potential must be the OUTPUT density and potential without ANY
	  // MIXING.
	  CalculateSecondOrderEnergy();
   
	  // Compute the KS energy 
	  CalculateKSEnergy();

	  // Update the total potential AFTER updating the energy

	  // No external potential

	  // Compute the new total potential

	  GetTime( timeSta );
        
	  hamDG.CalculateVtot( hamDG.Vtot() );
      
	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for computing Vtot in the global domain is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
     
	}

      
	// Compute the force at every step
	if( isCalculateForceEachSCF_ ){
	  // Compute force
	  GetTime( timeSta );
        
	  hamDG.CalculateForce( *distfftPtr_ );
        
	  GetTime( timeEnd );
	  statusOFS << "Time for computing the force is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;

	  // Print out the force
	  // Only master processor output information containing all atoms
	  if( mpirank == 0 ){
	    PrintBlock( statusOFS, "Atomic Force" );
	    {
	      Point3 forceCM(0.0, 0.0, 0.0);
	      std::vector<Atom>& atomList = hamDG.AtomList();
	      Int numAtom = atomList.size();
	      for( Int a = 0; a < numAtom; a++ ){
		Print( statusOFS, "atom", a, "force", atomList[a].force );
		forceCM += atomList[a].force;
	      }
	      statusOFS << std::endl;
	      Print( statusOFS, "force for centroid: ", forceCM );
	      statusOFS << std::endl;
	    }
	  }
	}

	// Compute the a posteriori error estimator at every step
	// FIXME This is not used when intra-element parallelization is
	// used.
	if( isCalculateAPosterioriEachSCF_ && 0 )
	  {
	    GetTime( timeSta );
	    DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
	    hamDG.CalculateAPosterioriError( 
					    eta2Total, eta2Residual, eta2GradJump, eta2Jump );
	    GetTime( timeEnd );
	    statusOFS << "Time for computing the a posteriori error is " <<
	      timeEnd - timeSta << " [s]" << std::endl << std::endl;

	    // Only master processor output information containing all atoms
	    if( mpirank == 0 ){
	      PrintBlock( statusOFS, "A Posteriori error" );
	      {
		statusOFS << std::endl << "Total a posteriori error:" << std::endl;
		statusOFS << eta2Total << std::endl;
		statusOFS << std::endl << "Residual term:" << std::endl;
		statusOFS << eta2Residual << std::endl;
		statusOFS << std::endl << "Jump of gradient term:" << std::endl;
		statusOFS << eta2GradJump << std::endl;
		statusOFS << std::endl << "Jump of function value term:" << std::endl;
		statusOFS << eta2Jump << std::endl;
	      }
	    }
	  }
      }

      // -----------------------------------------------
      // Method 1_1: Using diagonalization method, but with a more
      // versatile choice of processors for using ScaLAPACK.
      // Or using Chebyshev filtering
      
      if( solutionMethod_ == "diag" ){
        {
          // ~~**~~
          if(Diag_SCFDG_by_Cheby_ == 1 )
          {
            // Chebyshev filtering based diagonalization
            GetTime(timeSta);

            if(outerIter == 1)
            {
              statusOFS << std::endl << " Calling First Chebyshev Iter  " << std::endl;
              scfdg_FirstChebyStep(First_SCFDG_ChebyCycleNum_, First_SCFDG_ChebyFilterOrder_);
            }
            else if(outerIter > 1 && 	outerIter <= Second_SCFDG_ChebyOuterIter_)
            {
              statusOFS << std::endl << " Calling Second Stage Chebyshev Iter  " << std::endl;
              scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
            }
            else
            {  
              statusOFS << std::endl << " Calling General Chebyshev Iter  " << std::endl;
              scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 

            }



            MPI_Barrier( domain_.comm );
            MPI_Barrier( domain_.rowComm );
            MPI_Barrier( domain_.colComm );

            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << std::endl << " Time for diag DG matrix via Chebyshev filtering is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;

#endif	    


            DblNumVec& eigval = hamDG.EigVal(); 	     
            //for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
            //  statusOFS << setw(8) << i << setw(20) << '\t' << eigval[i] << std::endl;

          }
          else
          {

            // ScaLAPACK based diagonalization
            GetTime(timeSta);
            Int sizeH = hamDG.NumBasisTotal();

            DblNumVec& eigval = hamDG.EigVal(); 
            eigval.Resize( hamDG.NumStateTotal() );		

            for( Int k = 0; k < numElem_[2]; k++ )
              for( Int j = 0; j < numElem_[1]; j++ )
                for( Int i = 0; i < numElem_[0]; i++ ){
                  Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
                    DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                    localCoef.Resize( idx.size(), hamDG.NumStateTotal() );		
                  }
                } 

            // All processors participate in the data conversion procedure


            scalapack::Descriptor descH;

            if( contxt_ >= 0 ){
              descH.Init( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
                  0, 0, contxt_ );
            }

            scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;

            std::vector<Int> mpirankElemVec(dmCol_);
            std::vector<Int> mpirankScaVec( numProcScaLAPACK_ );

            // The processors in the first column are the source
            for( Int i = 0; i < dmCol_; i++ ){
              mpirankElemVec[i] = i * dmRow_;
            }
            // The first numProcScaLAPACK processors are the target
            for( Int i = 0; i < numProcScaLAPACK_; i++ ){
              mpirankScaVec[i] = i;
            }

#if ( _DEBUGlevel_ >= 2 )
            statusOFS << "mpirankElemVec = " << mpirankElemVec << std::endl;
            statusOFS << "mpirankScaVec = " << mpirankScaVec << std::endl;
#endif

            Real timeConversionSta, timeConversionEnd;

            GetTime( timeConversionSta );
            DistElemMatToScaMat2( hamDG.HMat(), 	descH,
                scaH, hamDG.ElemBasisIdx(), domain_.comm,
                domain_.colComm, mpirankElemVec,
                mpirankScaVec );
            GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
            statusOFS << "Time for converting from DistElemMat to ScaMat is " <<
              timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

            if(contxt_ >= 0){

              //          statusOFS << "LocalMatrix = " << scaH.LocalMatrix() << std::endl;
              //
              //          DistElemMatToScaMat( hamDG.HMat(), 	descH,
              //              scaH, hamDG.ElemBasisIdx(), domain_.colComm );
              //
              //          statusOFS << "LocalMatrixOri = " << scaH.LocalMatrix() << std::endl;

              std::vector<Real> eigs;

              GetTime( timeConversionSta );
              scalapack::Syevd('U', scaH, eigs, scaZ);
              GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "Time for scalapack::Syevd is " <<
                timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

              for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                eigval[i] = eigs[i];

            } //if(contxt_ >= 0)

            GetTime( timeConversionSta );
            ScaMatToDistNumMat2( scaZ, hamDG.Density().Prtn(), 
                hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm,
                domain_.colComm, mpirankElemVec, mpirankScaVec, 
                hamDG.NumStateTotal() );
            GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
            statusOFS << "Time for converting from ScaMat to DistNumMat is " <<
              timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

            GetTime( timeConversionSta );

            for( Int k = 0; k < numElem_[2]; k++ )
              for( Int j = 0; j < numElem_[1]; j++ )
                for( Int i = 0; i < numElem_[0]; i++ ){
                  Index3 key( i, j, k );
                  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                    DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                    MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), MPI_DOUBLE, 0, domain_.rowComm);
                  }
                } 
            GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
            statusOFS << "Time for MPI_Bcast eigval and localCoef is " <<
              timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

            MPI_Barrier( domain_.comm );
            MPI_Barrier( domain_.rowComm );
            MPI_Barrier( domain_.colComm );

            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << "Time for diag DG matrix via ScaLAPACK is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            // Communicate the eigenvalues
            Int mpirankScaSta = mpirankScaVec[0];
            MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
                mpirankScaVec[0], domain_.comm);


          } // End of ScaLAPACK based diagonalization
        } // End of diagonalization routines

        // Post processing

        Evdw_ = 0.0;

        // Compute the occupation rate
        CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

        // Compute the Harris energy functional.  
        // NOTE: In computing the Harris energy, the density and the
        // potential must be the INPUT density and potential without ANY
        // update.
        CalculateHarrisEnergy();

        MPI_Barrier( domain_.comm );
        MPI_Barrier( domain_.rowComm );
        MPI_Barrier( domain_.colComm );

        // Compute the output electron density
        GetTime( timeSta );

        // Calculate the new electron density
        // FIXME 
        // Do not need the conversion from column to row partition as well
        hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );


        MPI_Barrier( domain_.comm );
        MPI_Barrier( domain_.rowComm );
        MPI_Barrier( domain_.colComm );

        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing density in the global domain is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Update the output potential, and the KS and second order accurate
        // energy
        {
          // Update the Hartree energy and the exchange correlation energy and
          // potential for computing the KS energy and the second order
          // energy.
          // NOTE Vtot should not be updated until finishing the computation
          // of the energies.

          if( XCType_ == "XC_GGA_XC_PBE" ){
            hamDG.CalculateGradDensity(  *distfftPtr_ );
          }

          GetTime( timeSta );
          hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing Exc in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

          GetTime( timeSta );

          hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing Vhart in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

          // Compute the second order accurate energy functional.

          // Compute the second order accurate energy functional.
          // NOTE: In computing the second order energy, the density and the
          // potential must be the OUTPUT density and potential without ANY
          // MIXING.
          CalculateSecondOrderEnergy();

          // Compute the KS energy 

          GetTime( timeSta );

          CalculateKSEnergy();

          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing KSEnergy in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

          // Update the total potential AFTER updating the energy

          // No external potential

          // Compute the new total potential

          GetTime( timeSta );

          hamDG.CalculateVtot( hamDG.Vtot() );

          GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
          statusOFS << "Time for computing Vtot in the global domain is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
        }

        // Compute the force at every step
        if( isCalculateForceEachSCF_ ){
          // Compute force
          GetTime( timeSta );
          hamDG.CalculateForce( *distfftPtr_ );
          GetTime( timeEnd );
          statusOFS << "Time for computing the force is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

          // Print out the force
          // Only master processor output information containing all atoms
          if( mpirank == 0 ){
            PrintBlock( statusOFS, "Atomic Force" );
            {
              Point3 forceCM(0.0, 0.0, 0.0);
              std::vector<Atom>& atomList = hamDG.AtomList();
              Int numAtom = atomList.size();
              for( Int a = 0; a < numAtom; a++ ){
                Print( statusOFS, "atom", a, "force", atomList[a].force );
                forceCM += atomList[a].force;
              }
              statusOFS << std::endl;
              Print( statusOFS, "force for centroid: ", forceCM );
              statusOFS << std::endl;
            }
          }
        }

        // Compute the a posteriori error estimator at every step
        // FIXME This is not used when intra-element parallelization is
        // used.
        if( isCalculateAPosterioriEachSCF_ && 0 )
        {
          GetTime( timeSta );
          DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
          hamDG.CalculateAPosterioriError( 
              eta2Total, eta2Residual, eta2GradJump, eta2Jump );
          GetTime( timeEnd );
          statusOFS << "Time for computing the a posteriori error is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

          // Only master processor output information containing all atoms
          if( mpirank == 0 ){
            PrintBlock( statusOFS, "A Posteriori error" );
            {
              statusOFS << std::endl << "Total a posteriori error:" << std::endl;
              statusOFS << eta2Total << std::endl;
              statusOFS << std::endl << "Residual term:" << std::endl;
              statusOFS << eta2Residual << std::endl;
              statusOFS << std::endl << "Jump of gradient term:" << std::endl;
              statusOFS << eta2GradJump << std::endl;
              statusOFS << std::endl << "Jump of function value term:" << std::endl;
              statusOFS << eta2Jump << std::endl;
            }
          }
        }
      }


      // Method 2: Using the pole expansion and selected inversion (PEXSI) method
      // FIXME Currently it is assumed that all processors used by DG will be used by PEXSI.
#ifdef _USE_PEXSI_
      // The following version is with intra-element parallelization
      if( solutionMethod_ == "pexsi" ){
	Real timePEXSISta, timePEXSIEnd;
	GetTime( timePEXSISta );

	Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
	Real muMinInertia, muMaxInertia;
	Real muPEXSI, numElectronPEXSI;
	Int numTotalInertiaIter, numTotalPEXSIIter;

	std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

	// FIXME 
	// Currently, only the first processor column participate in the
	// communication between PEXSI and DGDFT For the first processor
	// column involved in PEXSI, the first numProcPEXSICommCol_
	// processors are involved in the data communication between PEXSI
	// and DGDFT
      
	for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
	  mpirankSparseVec[i] = i;
	}

#if ( _DEBUGlevel_ >= 1 )
	statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

	Int info;

	// Temporary matrices 
	DistSparseMatrix<Real>  HSparseMat;
	DistSparseMatrix<Real>  DMSparseMat;
	DistSparseMatrix<Real>  EDMSparseMat;
	DistSparseMatrix<Real>  FDMSparseMat;
          
	if( mpirankRow == 0 ){

	  // Convert the DG matrix into the distributed CSC format

	  GetTime(timeSta);
	  DistElemMatToDistSparseMat3( 
				      hamDG.HMat(),
				      hamDG.NumBasisTotal(),
				      HSparseMat,
				      hamDG.ElemBasisIdx(),
				      domain_.colComm,
				      mpirankSparseVec );
	  GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
	  if( mpirankCol < numProcPEXSICommCol_ ){
	    statusOFS << "H.size = " << HSparseMat.size << std::endl;
	    statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
	    statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
	    statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
	    statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
	    statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
	  }
#endif
	}


	if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) ){
	  // Load the matrices into PEXSI.  
	  // Only the processors with mpirankCol == 0 need to carry the
	  // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
	  statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
	  statusOFS << "mpirankRow = " << mpirankRow << std::endl;
	  statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif


	  GetTime( timeSta );
	  PPEXSILoadRealSymmetricHSMatrix(
					  pexsiPlan_,
					  pexsiOptions_,
					  HSparseMat.size,
					  HSparseMat.nnz,
					  HSparseMat.nnzLocal,
					  HSparseMat.colptrLocal.m() - 1,
					  HSparseMat.colptrLocal.Data(),
					  HSparseMat.rowindLocal.Data(),
					  HSparseMat.nzvalLocal.Data(),
					  1,  // isSIdentity
					  NULL,
					  &info );
	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for loading the matrix into PEXSI is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	  if( info != 0 ){
	    std::ostringstream msg;
	    msg 
	      << "PEXSI loading H matrix returns info " << info << std::endl;
	    throw std::runtime_error( msg.str().c_str() );
	  }

	  // PEXSI solver

	  {
	    if( outerIter >= inertiaCountSteps_ ){
	      pexsiOptions_.isInertiaCount = 0;
	    }
	    // Note: Heuristics strategy for dynamically adjusting the
	    // tolerance
	    pexsiOptions_.muInertiaTolerance = 
	      std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.05 );
	    pexsiOptions_.numElectronPEXSITolerance = 
	      std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

	    // Only perform symbolic factorization for the first outer SCF. 
	    // Reuse the previous Fermi energy as the initial guess for mu.
	    if( outerIter == 1 ){
	      pexsiOptions_.isSymbolicFactorize = 1;
	      pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
	    }
	    else{
	      pexsiOptions_.isSymbolicFactorize = 0;
	      pexsiOptions_.mu0 = fermi_;
	    }

	    statusOFS << std::endl 
		      << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
		      << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
		      << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
	  }


	  GetTime( timeSta );
	  PPEXSIDFTDriver(
			  pexsiPlan_,
			  pexsiOptions_,
			  numElectronExact,
			  &muPEXSI,
			  &numElectronPEXSI,         
			  &muMinInertia,              
			  &muMaxInertia,             
			  &numTotalInertiaIter,
			  &numTotalPEXSIIter,
			  &info );
	  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	  statusOFS << "Time for the main PEXSI Driver is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	  if( info != 0 ){
	    std::ostringstream msg;
	    msg 
	      << "PEXSI main driver returns info " << info << std::endl;
	    throw std::runtime_error( msg.str().c_str() );
	  }

	  // Update the fermi level 
	  fermi_ = muPEXSI;

	  // Heuristics for the next step
	  pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
	  pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

	  // Retrieve the PEXSI data

	  if( mpirankRow == 0 ){
	    Real totalEnergyH, totalEnergyS, totalFreeEnergy;

	    GetTime( timeSta );
          
	    CopyPattern( HSparseMat, DMSparseMat );
	    CopyPattern( HSparseMat, EDMSparseMat );
	    CopyPattern( HSparseMat, FDMSparseMat );

	    PPEXSIRetrieveRealSymmetricDFTMatrix(
						 pexsiPlan_,
						 DMSparseMat.nzvalLocal.Data(),
						 EDMSparseMat.nzvalLocal.Data(),
						 FDMSparseMat.nzvalLocal.Data(),
						 &totalEnergyH,
						 &totalEnergyS,
						 &totalFreeEnergy,
						 &info );
	    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	    statusOFS << "Time for retrieving PEXSI data is " <<
	      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	    statusOFS << std::endl
		      << "Results obtained from PEXSI:" << std::endl
		      << "Total energy (H*DM)         = " << totalEnergyH << std::endl
		      << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
		      << "Total free energy           = " << totalFreeEnergy << std::endl 
		      << "InertiaIter                 = " << numTotalInertiaIter << std::endl
		      << "PEXSIIter                   = " <<  numTotalPEXSIIter << std::endl
		      << "mu                          = " << muPEXSI << std::endl
		      << "numElectron                 = " << numElectronPEXSI << std::endl 
		      << std::endl;

	    if( info != 0 ){
	      std::ostringstream msg;
	      msg 
		<< "PEXSI data retrieval returns info " << info << std::endl;
	      throw std::runtime_error( msg.str().c_str() );
	    }
	  }
	} // if( mpirank < numProcTotalPEXSI_ )

	// Broadcast the Fermi level
	MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );

	if( mpirankRow == 0 )
	  {
	    GetTime(timeSta);
	    // Convert the density matrix from DistSparseMatrix format to the
	    // DistElemMat format
	    DistSparseMatToDistElemMat3(
					DMSparseMat,
					hamDG.NumBasisTotal(),
					hamDG.HMat().Prtn(),
					distDMMat_,
					hamDG.ElemBasisIdx(),
					hamDG.ElemBasisInvIdx(),
					domain_.colComm,
					mpirankSparseVec );


	    // Convert the energy density matrix from DistSparseMatrix
	    // format to the DistElemMat format
	    DistSparseMatToDistElemMat3(
					EDMSparseMat,
					hamDG.NumBasisTotal(),
					hamDG.HMat().Prtn(),
					distEDMMat_,
					hamDG.ElemBasisIdx(),
					hamDG.ElemBasisInvIdx(),
					domain_.colComm,
					mpirankSparseVec );


	    // Convert the free energy density matrix from DistSparseMatrix
	    // format to the DistElemMat format
	    DistSparseMatToDistElemMat3(
					FDMSparseMat,
					hamDG.NumBasisTotal(),
					hamDG.HMat().Prtn(),
					distFDMMat_,
					hamDG.ElemBasisIdx(),
					hamDG.ElemBasisInvIdx(),
					domain_.colComm,
					mpirankSparseVec );
	    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	    statusOFS << "Time for converting the DistSparseMatrices to DistElemMat " << 
	      "for post-processing is " <<
	      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	  }

	// Broadcast the distElemMat matrices
	// FIXME this is not a memory efficient implementation
	GetTime(timeSta);
	{
	  Int sstrSize;
	  std::vector<char> sstr;
	  if( mpirankRow == 0 ){
	    std::stringstream distElemMatStream;
	    Int cnt = 0;
	    for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
		 mi != distDMMat_.LocalMap().end(); mi++ ){
	      cnt++;
	    } // for (mi)
	    serialize( cnt, distElemMatStream, NO_MASK );
	    for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
		 mi != distDMMat_.LocalMap().end(); mi++ ){
	      ElemMatKey key = (*mi).first;
	      serialize( key, distElemMatStream, NO_MASK );
	      serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );
	      serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
	      serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
	    } // for (mi)
	    sstr.resize( Size( distElemMatStream ) );
	    distElemMatStream.read( &sstr[0], sstr.size() );
	    sstrSize = sstr.size();
	  }
        
	  MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
	  sstr.resize( sstrSize );
	  MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

	  if( mpirankRow != 0 ){
	    std::stringstream distElemMatStream;
	    distElemMatStream.write( &sstr[0], sstrSize );
	    Int cnt;
	    deserialize( cnt, distElemMatStream, NO_MASK );
	    for( Int i = 0; i < cnt; i++ ){
	      ElemMatKey key;
	      DblNumMat mat;
	      deserialize( key, distElemMatStream, NO_MASK );
	      deserialize( mat, distElemMatStream, NO_MASK );
	      distDMMat_.LocalMap()[key] = mat;
	      deserialize( mat, distElemMatStream, NO_MASK );
	      distEDMMat_.LocalMap()[key] = mat;
	      deserialize( mat, distElemMatStream, NO_MASK );
	      distFDMMat_.LocalMap()[key] = mat;
	    } // for (mi)
	  }
	}
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for broadcasting the density matrix for post-processing is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	Evdw_ = 0.0;

	// Compute the Harris energy functional.  
	// NOTE: In computing the Harris energy, the density and the
	// potential must be the INPUT density and potential without ANY
	// update.
	GetTime( timeSta );
	CalculateHarrisEnergyDM( distFDMMat_ );
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for calculating the Harris energy is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	// Evaluate the electron density

	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	      } // own this element
	    } // for (i)


	GetTime( timeSta );
	hamDG.CalculateDensityDM2( 
				  hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
	MPI_Barrier( domain_.comm );
	GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for computing density in the global domain is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
	
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	      } // own this element
	    } // for (i)


	// Update the output potential, and the KS and second order accurate
	// energy
	GetTime(timeSta);
	{
	  // Update the Hartree energy and the exchange correlation energy and
	  // potential for computing the KS energy and the second order
	  // energy.
	  // NOTE Vtot should not be updated until finishing the computation
	  // of the energies.

	  if( XCType_ == "XC_GGA_XC_PBE" ){
	    hamDG.CalculateGradDensity(  *distfftPtr_ );
	  }

	  hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

	  hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

	  // Compute the second order accurate energy functional.
	  // NOTE: In computing the second order energy, the density and the
	  // potential must be the OUTPUT density and potential without ANY
	  // MIXING.
	  //        CalculateSecondOrderEnergy();

	  // Compute the KS energy 
	  CalculateKSEnergyDM( 
			      distEDMMat_, distFDMMat_ );

	  // Update the total potential AFTER updating the energy

	  // No external potential

	  // Compute the new total potential

	  hamDG.CalculateVtot( hamDG.Vtot() );

	}
	GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for computing the potential is " <<
	  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

	// Compute the force at every step
	if( isCalculateForceEachSCF_ ){
	  // Compute force
	  GetTime( timeSta );
	  hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
	  GetTime( timeEnd );
	  statusOFS << "Time for computing the force is " <<
	    timeEnd - timeSta << " [s]" << std::endl << std::endl;

	  // Print out the force
	  // Only master processor output information containing all atoms
	  if( mpirank == 0 ){
	    PrintBlock( statusOFS, "Atomic Force" );
	    {
	      Point3 forceCM(0.0, 0.0, 0.0);
	      std::vector<Atom>& atomList = hamDG.AtomList();
	      Int numAtom = atomList.size();
	      for( Int a = 0; a < numAtom; a++ ){
		Print( statusOFS, "atom", a, "force", atomList[a].force );
		forceCM += atomList[a].force;
	      }
	      statusOFS << std::endl;
	      Print( statusOFS, "force for centroid: ", forceCM );
	      statusOFS << std::endl;
	    }
	  }
	}

	// TODO Evaluate the a posteriori error estimator

	GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
	statusOFS << "Time for PEXSI evaluation is " <<
	  timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
      } //if( solutionMethod_ == "pexsi" )


#endif

    
      // Compute the error of the mixing variable

      GetTime(timeSta);
      {
	Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
	Real normMixDif, normMixOld;
	for( Int k = 0; k < numElem_[2]; k++ )
	  for( Int j = 0; j < numElem_[1]; j++ )
	    for( Int i = 0; i < numElem_[0]; i++ ){
	      Index3 key( i, j, k );
	      if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		if( mixVariable_ == "density" ){
		  DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
		  DblNumVec& newVec = hamDG.Density().LocalMap()[key];

		  for( Int p = 0; p < oldVec.m(); p++ ){
		    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
		    normMixOldLocal += pow( oldVec(p), 2.0 );
		  }
		}
		else if( mixVariable_ == "potential" ){
		  DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
		  DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

		  for( Int p = 0; p < oldVec.m(); p++ ){
		    normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
		    normMixOldLocal += pow( oldVec(p), 2.0 );
		  }
		}
	      } // own this element
	    } // for (i)

      


	mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
			domain_.colComm );
	mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
			domain_.colComm );

	normMixDif = std::sqrt( normMixDif );
	normMixOld = std::sqrt( normMixOld );

	scfInnerNorm_    = normMixDif / normMixOld;
#if ( _DEBUGlevel_ >= 1 )
	Print(statusOFS, "norm(MixDif)          = ", normMixDif );
	Print(statusOFS, "norm(MixOld)          = ", normMixOld );
	Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
#endif
      }

      if( scfInnerNorm_ < scfInnerTolerance_ ){
	/* converged */
	Print( statusOFS, "Inner SCF is converged!\n" );
	isInnerSCFConverged = true;
      }

      MPI_Barrier( domain_.colComm );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing the SCF residual is " <<
	timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Mixing for the inner SCF iteration.
      GetTime( timeSta );

      // The number of iterations used for Anderson mixing
      Int numAndersonIter;

      if( scfInnerMaxIter_ == 1 ){
	// Maximum inner iteration = 1 means there is no distinction of
	// inner/outer SCF.  Anderson mixing uses the global history
	numAndersonIter = scfTotalInnerIter_;
      }
      else{
	// If more than one inner iterations is used, then Anderson only
	// uses local history.  For explanation see 
	//
	// Note 04/11/2013:  
	// "Problem of Anderson mixing in inner/outer SCF loop"
	numAndersonIter = innerIter;
      }

      if( mixVariable_ == "density" ){
	if( mixType_ == "anderson" ||
	    mixType_ == "kerker+anderson"	){
	  AndersonMix(
		      numAndersonIter, 
		      mixStepLength_,
		      mixType_,
		      hamDG.Density(),
		      mixInnerSave_,
		      hamDG.Density(),
		      dfInnerMat_,
		      dvInnerMat_);
	} else{
	  throw std::runtime_error("Invalid mixing type.");
	}
      }
      else if( mixVariable_ == "potential" ){
	if( mixType_ == "anderson" ||
	    mixType_ == "kerker+anderson"	){
	  AndersonMix(
		      numAndersonIter, 
		      mixStepLength_,
		      mixType_,
		      hamDG.Vtot(),
		      mixInnerSave_,
		      hamDG.Vtot(),
		      dfInnerMat_,
		      dvInnerMat_);
	} else{
	  throw std::runtime_error("Invalid mixing type.");
	}
      }



      MPI_Barrier( domain_.comm );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for mixing is " <<
	timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Post processing for the density mixing. Make sure that the
      // density is positive, and compute the potential again. 
      // This is only used for density mixing.
      if( mixVariable_ == "density" )
	{
	  Real sumRhoLocal = 0.0;
	  Real sumRho;
	  for( Int k = 0; k < numElem_[2]; k++ )
	    for( Int j = 0; j < numElem_[1]; j++ )
	      for( Int i = 0; i < numElem_[0]; i++ ){
		Index3 key( i, j, k );
		if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		  DblNumVec&  density      = hamDG.Density().LocalMap()[key];

		  for (Int p=0; p < density.Size(); p++) {
		    density(p) = std::max( density(p), 0.0 );
		    sumRhoLocal += density(p);
		  }
		} // own this element
	      } // for (i)
	  mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
	  sumRho *= domain_.Volume() / domain_.NumGridTotal();

	  Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;

#if ( _DEBUGlevel_ >= 1 )
	  statusOFS << std::endl;
	  Print( statusOFS, "Sum Rho after mixing (raw data) = ", sumRho );
	  statusOFS << std::endl;
#endif


	  // Normalize the electron density in the global domain
	  for( Int k = 0; k < numElem_[2]; k++ )
	    for( Int j = 0; j < numElem_[1]; j++ )
	      for( Int i = 0; i < numElem_[0]; i++ ){
		Index3 key( i, j, k );
		if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
		  DblNumVec& localRho = hamDG.Density().LocalMap()[key];
		  blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
		} // own this element
	      } // for (i)


	  // Update the potential after mixing for the next iteration.  
	  // This is only used for potential mixing

	  // Compute the exchange-correlation potential and energy from the
	  // new density

	  if( XCType_ == "XC_GGA_XC_PBE" ){
	    hamDG.CalculateGradDensity(  *distfftPtr_ );
	  }

	  hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

	  hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

	  // No external potential

	  // Compute the new total potential

	  hamDG.CalculateVtot( hamDG.Vtot() );
	}

      // Print out the state variables of the current iteration

      // Only master processor output information containing all atoms
      if( mpirank == 0 ){
	PrintState( );
      }

      GetTime( timeIterEnd );
   
      statusOFS << "Time time for this inner SCF iteration = " << timeIterEnd - timeIterStart
		<< " [s]" << std::endl << std::endl;

    } // for (innerIter)

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::InnerIterate  ----- 

  void
  SCFDG::UpdateElemLocalPotential	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::UpdateElemLocalPotential");
#endif

    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

  

    // vtot gather the neighborhood
    DistDblNumVec&  vtot = hamDG.Vtot();


    std::set<Index3> neighborSet;
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner(key) == (mpirank / dmRow_) ){
	    std::vector<Index3>   idx(3);

	    for( Int d = 0; d < DIM; d++ ){
	      // Previous
	      if( key[d] == 0 ) 
		idx[0][d] = numElem_[d]-1; 
	      else 
		idx[0][d] = key[d]-1;

	      // Current
	      idx[1][d] = key[d];

	      // Next
	      if( key[d] == numElem_[d]-1) 
		idx[2][d] = 0;
	      else
		idx[2][d] = key[d] + 1;
	    } // for (d)

	    // Tensor product 
	    for( Int c = 0; c < 3; c++ )
	      for( Int b = 0; b < 3; b++ )
		for( Int a = 0; a < 3; a++ ){
		  // Not the element key itself
		  if( idx[a][0] != i || idx[b][1] != j || idx[c][2] != k ){
		    neighborSet.insert( Index3( idx[a][0], idx[b][1], idx[c][2] ) );
		  }
		} // for (a)
	  } // own this element
	} // for (i)
    std::vector<Index3>  neighborIdx;
    neighborIdx.insert( neighborIdx.begin(), neighborSet.begin(), neighborSet.end() );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "neighborIdx = " << neighborIdx << std::endl;
#endif

    // communicate
    vtot.Prtn()   = elemPrtn_;
    vtot.SetComm(domain_.colComm);
    vtot.GetBegin( neighborIdx, NO_MASK );
    vtot.GetEnd( NO_MASK );



    // Update of the local potential in each extended element locally.
    // The nonlocal potential does not need to be updated
    //
    // Also update the local potential on the LGL grid in hamDG.
    //
    // NOTE:
    //
    // 1. It is hard coded that the extended element is 1 or 3
    // times the size of the element
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	    // Skip the calculation if there is no adaptive local
	    // basis function.  
	    if( eigSol.Psi().NumState() == 0 )
	      continue;

	    Hamiltonian&  hamExtElem  = eigSol.Ham();
	    DblNumVec&    vtotExtElem = hamExtElem.Vtot();
	    SetValue( vtotExtElem, 0.0 );

	    Index3 numGridElem = hamDG.NumUniformGridElemFine();
	    Index3 numGridExtElem = eigSol.FFT().domain.numGridFine;

	    // Update the potential in the extended element
	    for(std::map<Index3, DblNumVec>::iterator 
		  mi = vtot.LocalMap().begin();
		mi != vtot.LocalMap().end(); mi++ ){
	      Index3      keyElem = (*mi).first;
	      DblNumVec&  vtotElem = (*mi).second;




	      // Determine the shiftIdx which maps the position of vtotElem to 
	      // vtotExtElem
	      Index3 shiftIdx;
	      for( Int d = 0; d < DIM; d++ ){
		shiftIdx[d] = keyElem[d] - key[d];
		shiftIdx[d] = shiftIdx[d] - IRound( Real(shiftIdx[d]) / 
						    numElem_[d] ) * numElem_[d];
		// FIXME Adjustment  
		if( numElem_[d] > 1 ) shiftIdx[d] ++;

		shiftIdx[d] *= numGridElem[d];
	      }

#if ( _DEBUGlevel_ >= 1 )
	      statusOFS << "keyExtElem         = " << key << std::endl;
	      statusOFS << "numGridExtElemFine = " << numGridExtElem << std::endl;
	      statusOFS << "numGridElemFine    = " << numGridElem << std::endl;
	      statusOFS << "keyElem            = " << keyElem << ", shiftIdx = " << shiftIdx << std::endl;
#endif
            
	      Int ptrExtElem, ptrElem;
	      for( Int k = 0; k < numGridElem[2]; k++ )
		for( Int j = 0; j < numGridElem[1]; j++ )
		  for( Int i = 0; i < numGridElem[0]; i++ ){
		    ptrExtElem = (shiftIdx[0] + i) + 
		      ( shiftIdx[1] + j ) * numGridExtElem[0] +
		      ( shiftIdx[2] + k ) * numGridExtElem[0] * numGridExtElem[1];
		    ptrElem    = i + j * numGridElem[0] + k * numGridElem[0] * numGridElem[1];
		    vtotExtElem( ptrExtElem ) = vtotElem( ptrElem );
		  } // for (i)

	    } // for (mi)

	    // Loop over the neighborhood

	  } // own this element
	} // for (i)

    // Clean up vtot not owned by this element
    std::vector<Index3>  eraseKey;
    for( std::map<Index3, DblNumVec>::iterator 
	   mi  = vtot.LocalMap().begin();
	 mi != vtot.LocalMap().end(); mi++ ){
      Index3 key = (*mi).first;
      if( vtot.Prtn().Owner(key) != (mpirank / dmRow_) ){
	eraseKey.push_back( key );
      }
    }

    for( std::vector<Index3>::iterator vi = eraseKey.begin();
	 vi != eraseKey.end(); vi++ ){
      vtot.LocalMap().erase( *vi );
    }

    // Modify the potential in the extended element.  Current options are
    //
    // 1. Add barrier
    // 2. Periodize the potential
    //
    // Numerical results indicate that option 2 seems to be better.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	    Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

	    // Add the external barrier potential. CANNOT be used
	    // together with periodization option
	    if( isPotentialBarrier_ ){
	      Domain& dmExtElem = eigSol.FFT().domain;
	      DblNumVec& vext = eigSol.Ham().Vext();
	      SetValue( vext, 0.0 );
	      for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
		for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
		  for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
		    Int idx = gi + gj * dmExtElem.numGridFine[0] + 
		      gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
		    vext[idx] = vBarrier_[0][gi] + vBarrier_[1][gj] + vBarrier_[2][gk];
		  } // for (gi)
	      // NOTE:
	      // Directly modify the vtot.  vext is not used in the
	      // matrix-vector multiplication in the eigensolver.
	      blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
			  eigSol.Ham().Vtot().Data(), 1 );

	    }

	    // Periodize the external potential. CANNOT be used together
	    // with the barrier potential option
	    if( isPeriodizePotential_ ){
	      Domain& dmExtElem = eigSol.FFT().domain;
	      // Get the potential
	      DblNumVec& vext = eigSol.Ham().Vext();
	      DblNumVec& vtot = eigSol.Ham().Vtot();

	      // Find the max of the potential in the extended element
	      Real vtotMax = *std::max_element( &vtot[0], &vtot[0] + vtot.Size() );
	      Real vtotAvg = 0.0;
	      for(Int i = 0; i < vtot.Size(); i++){
		vtotAvg += vtot[i];
	      }
	      vtotAvg /= Real(vtot.Size());
	      Real vtotMin = *std::min_element( &vtot[0], &vtot[0] + vtot.Size() );

#if ( _DEBUGlevel_ >= 0 ) 
	      Print( statusOFS, "vtotMax  = ", vtotMax );
	      Print( statusOFS, "vtotAvg  = ", vtotAvg );
	      Print( statusOFS, "vtotMin  = ", vtotMin );
#endif

	      SetValue( vext, 0.0 );
	      for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
		for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
		  for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
		    Int idx = gi + gj * dmExtElem.numGridFine[0] + 
		      gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
		    // Bring the potential to the vacuum level
		    vext[idx] = ( vtot[idx] - 0.0 ) * 
		      ( vBubble_[0][gi] * vBubble_[1][gj] * vBubble_[2][gk] - 1.0 );
		  } // for (gi)
	      // NOTE:
	      // Directly modify the vtot.  vext is not used in the
	      // matrix-vector multiplication in the eigensolver.
	      blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
			  eigSol.Ham().Vtot().Data(), 1 );

	    } // if ( isPeriodizePotential_ ) 
	  } // own this element
	} // for (i)


    // Update the potential in element on LGL grid
    //
    // The local potential on the LGL grid is done by using Fourier
    // interpolation from the extended element to the element. Gibbs
    // phenomena MAY be there but at least this is better than
    // Lagrange interpolation on a uniform grid.
    //
    // NOTE: The interpolated potential on the LGL grid is taken to be the
    // MODIFIED potential with vext on the extended element. Therefore it
    // is important that the artificial vext vanishes inside the element.
    // When periodization option is used, it can potentially reduce the
    // effect of Gibbs phenomena.
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
	    Index3 numGridExtElemFine = eigSol.FFT().domain.numGridFine;

	    DblNumVec&  vtotLGLElem = hamDG.VtotLGL().LocalMap()[key];
	    Index3 numLGLGrid       = hamDG.NumLGLGridElem();

	    DblNumVec&    vtotExtElem = eigSol.Ham().Vtot();

	    InterpPeriodicUniformFineToLGL( 
					   numGridExtElemFine,
					   numLGLGrid,
					   vtotExtElem.Data(),
					   vtotLGLElem.Data() );
	  } // own this element
	} // for (i)


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::UpdateElemLocalPotential  ----- 

  void
  SCFDG::CalculateOccupationRate	( DblNumVec& eigVal, DblNumVec& occupationRate )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateOccupationRate");
#endif
    // For a given finite temperature, update the occupation number */
    // FIXME Magic number here
    Real tol = 1e-10; 
    Int maxiter = 100;  

    Real lb, ub, flb, fub, occsum;
    Int ilb, iub, iter;

    Int npsi       = hamDGPtr_->NumStateTotal();
    Int nOccStates = hamDGPtr_->NumOccupiedState();

    if( eigVal.m() != npsi ){
      std::ostringstream msg;
      msg 
	<< "The number of eigenstates do not match."  << std::endl
	<< "eigVal         ~ " << eigVal.m() << std::endl
	<< "numStateTotal  ~ " << npsi << std::endl;
      throw std::logic_error( msg.str().c_str() );
    }


    if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

    if( npsi > nOccStates )  {
      /* use bisection to find efermi such that 
       * sum_i fermidirac(ev(i)) = nocc
       */
      ilb = nOccStates-1;
      iub = nOccStates+1;

      lb = eigVal(ilb-1);
      ub = eigVal(iub-1);

      /* Calculate Fermi-Dirac function and make sure that
       * flb < nocc and fub > nocc
       */

      flb = 0.0;
      fub = 0.0;
      for(Int j = 0; j < npsi; j++) {
	flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
	fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub))); 
      }

      while( (nOccStates-flb)*(fub-nOccStates) < 0 ) {
	if( flb > nOccStates ) {
	  if(ilb > 0){
	    ilb--;
	    lb = eigVal(ilb-1);
	    flb = 0.0;
	    for(Int j = 0; j < npsi; j++) flb += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-lb)));
	  }
	  else {
	    throw std::logic_error( "Cannot find a lower bound for efermi" );
	  }
	}

	if( fub < nOccStates ) {
	  if( iub < npsi ) {
	    iub++;
	    ub = eigVal(iub-1);
	    fub = 0.0;
	    for(Int j = 0; j < npsi; j++) fub += 1.0 / (1.0 + exp(Tbeta_*(eigVal(j)-ub)));
	  }
	  else {
	    throw std::logic_error( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
	  }
	}
      }  /* end while */

      fermi_ = (lb+ub)*0.5;
      occsum = 0.0;
      for(Int j = 0; j < npsi; j++) {
	occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
	occsum += occupationRate(j);
      }

      /* Start bisection iteration */
      iter = 1;
      while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
	if( occsum < nOccStates ) {lb = fermi_;}
	else {ub = fermi_;}

	fermi_ = (lb+ub)*0.5;
	occsum = 0.0;
	for(Int j = 0; j < npsi; j++) {
	  occupationRate(j) = 1.0 / (1.0 + exp(Tbeta_*(eigVal(j) - fermi_)));
	  occsum += occupationRate(j);
	}
	iter++;
      }
    }
    else {
      if (npsi == nOccStates ) {
	for(Int j = 0; j < npsi; j++) 
	  occupationRate(j) = 1.0;
	fermi_ = eigVal(npsi-1);
      }
      else {
	throw std::logic_error( "The number of eigenvalues in ev should be larger than nocc" );
      }
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateOccupationRate  ----- 



  void
  SCFDG::InterpPeriodicUniformToLGL	( 
					 const Index3& numUniformGrid, 
					 const Index3& numLGLGrid, 
					 const Real*   psiUniform, 
					 Real*         psiLGL )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::InterpPeriodicUniformToLGL");
#endif

    Index3 Ns1 = numUniformGrid;
    Index3 Ns2 = numLGLGrid;
	
    DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
    DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
    SetValue( tmp1, 0.0 );
    SetValue( tmp2, 0.0 );

    // x-direction, use Gemm
    {
      Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
      blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformToLGLMat_[0].Data(),
		  m, psiUniform, k, 0.0, tmp1.Data(), m );
    }
	
    // y-direction, use Gemv
    {
      Int   m = Ns2[1], n = Ns1[1];
      Int   ptrShift1, ptrShift2;
      Int   inc = Ns2[0];
      for( Int k = 0; k < Ns1[2]; k++ ){
	for( Int i = 0; i < Ns2[0]; i++ ){
	  ptrShift1 = i + k * Ns2[0] * Ns1[1];
	  ptrShift2 = i + k * Ns2[0] * Ns2[1];
	  blas::Gemv( 'N', m, n, 1.0, 
		      PeriodicUniformToLGLMat_[1].Data(), m, 
		      tmp1.Data() + ptrShift1, inc, 0.0, 
		      tmp2.Data() + ptrShift2, inc );
	} // for (i)
      } // for (k)
    }

	
    // z-direction, use Gemm
    {
      Int m = Ns2[0] * Ns2[1], n = Ns2[2], k = Ns1[2]; 
      blas::Gemm( 'N', 'T', m, n, k, 1.0, 
		  tmp2.Data(), m, 
		  PeriodicUniformToLGLMat_[2].Data(), n, 0.0, psiLGL, m );
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::InterpPeriodicUniformToLGL  ----- 


  void
  SCFDG::InterpPeriodicUniformFineToLGL	( 
					 const Index3& numUniformGridFine, 
					 const Index3& numLGLGrid, 
					 const Real*   rhoUniform, 
					 Real*         rhoLGL )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::InterpPeriodicUniformFineToLGL");
#endif

    Index3 Ns1 = numUniformGridFine;
    Index3 Ns2 = numLGLGrid;
	
    DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
    DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
    SetValue( tmp1, 0.0 );
    SetValue( tmp2, 0.0 );

    // x-direction, use Gemm
    {
      Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
      blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicUniformFineToLGLMat_[0].Data(),
		  m, rhoUniform, k, 0.0, tmp1.Data(), m );
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
		      PeriodicUniformFineToLGLMat_[1].Data(), m, 
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
		  PeriodicUniformFineToLGLMat_[2].Data(), n, 0.0, rhoLGL, m );
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::InterpPeriodicUniformFineToLGL  ----- 


  void
  SCFDG::InterpPeriodicGridExtElemToGridElem ( 
					      const Index3& numUniformGridFineExtElem, 
					      const Index3& numUniformGridFineElem, 
					      const Real*   rhoUniformExtElem, 
					      Real*         rhoUniformElem )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::InterpPeriodicGridExtElemToGridElem");
#endif

    Index3 Ns1 = numUniformGridFineExtElem;
    Index3 Ns2 = numUniformGridFineElem;
	
    DblNumVec  tmp1( Ns2[0] * Ns1[1] * Ns1[2] );
    DblNumVec  tmp2( Ns2[0] * Ns2[1] * Ns1[2] );
    SetValue( tmp1, 0.0 );
    SetValue( tmp2, 0.0 );

    // x-direction, use Gemm
    {
      Int m = Ns2[0], n = Ns1[1] * Ns1[2], k = Ns1[0];
      blas::Gemm( 'N', 'N', m, n, k, 1.0, PeriodicGridExtElemToGridElemMat_[0].Data(),
		  m, rhoUniformExtElem, k, 0.0, tmp1.Data(), m );
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
		      PeriodicGridExtElemToGridElemMat_[1].Data(), m, 
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
		  PeriodicGridExtElemToGridElemMat_[2].Data(), n, 0.0, rhoUniformElem, m );
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::InterpPeriodicGridExtElemToGridElem  ----- 


  void
  SCFDG::CalculateKSEnergy	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateKSEnergy");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // Kinetic energy
    Int numSpin = hamDG.NumSpin();
    Ekin_ = 0.0;
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin_  += numSpin * eigVal(i) * occupationRate(i);
    }

    // Self energy part
    Eself_ = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself_ +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }


    // Hartree and XC part
    Ehart_ = 0.0;
    EVxc_  = 0.0;

    Real EhartLocal = 0.0, EVxcLocal = 0.0;
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	    DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
	    DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
	    DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

	    for (Int p=0; p < density.Size(); p++) {
	      EVxcLocal  += vxc(p) * density(p);
	      EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
	    }

	  } // own this element
	} // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );

    Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();

    // Correction energy
    Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;
  
    // Total energy
    Etot_ = Ekin_ + Ecor_;

    // Helmholtz free energy
    if( hamDG.NumOccupiedState() == 
	hamDG.NumStateTotal() ){
      // Zero temperature
      Efree_ = Etot_;
    }
    else{
      // Finite temperature
      Efree_ = 0.0;
      Real fermi = fermi_;
      Real Tbeta = Tbeta_;
      for(Int l=0; l< eigVal.m(); l++) {
	Real eig = eigVal(l);
	if( eig - fermi >= 0){
	  Efree_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
	}
	else{
	  Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
	}
      }
      Efree_ += Ecor_ + fermi * hamDG.NumOccupiedState() * numSpin; 
    }


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateKSEnergy  ----- 


  void
  SCFDG::CalculateKSEnergyDM (
			      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distEDMMat,
			      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateKSEnergyDM");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // Kinetic energy
    Int numSpin = hamDG.NumSpin();

    // Self energy part
    Eself_ = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself_ +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }

    // Hartree and XC part
    Ehart_ = 0.0;
    EVxc_  = 0.0;

    Real EhartLocal = 0.0, EVxcLocal = 0.0;
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	    DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
	    DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
	    DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];
          
	    for (Int p=0; p < density.Size(); p++) {
	      EVxcLocal  += vxc(p) * density(p);
	      EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
	    }

	  } // own this element
	} // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc_, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart_, 1, MPI_SUM, domain_.colComm );

    Ehart_ *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc_  *= domain_.Volume() / domain_.NumGridTotalFine();

    // Correction energy
    Ecor_   = (Exc_ - EVxc_) - Ehart_ - Eself_;
  
    // Kinetic energy and helmholtz free energy, calculated from the
    // energy and free energy density matrices.
    // Here 
    // 
    //   Ekin = Tr[H 2/(1+exp(beta(H-mu)))] 
    // and
    //   Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
    // FIXME Put the above documentation to the proper place like the hpp
    // file

    Real Ehelm = 0.0, EhelmLocal = 0.0, EkinLocal = 0.0;
  
    if( 1 ) {
      // Compute the trace of the energy density matrix in each element
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
            
	      DblNumMat& localEDM = distEDMMat.LocalMap()[
							  ElemMatKey(key, key)];
	      DblNumMat& localFDM = distFDMMat.LocalMap()[
							  ElemMatKey(key, key)];

	      for( Int a = 0; a < localEDM.m(); a++ ){
		EkinLocal  += localEDM(a,a);
		EhelmLocal += localFDM(a,a);
	      }
	    } // own this element
	  } // for (i)

      // Reduce the results 
      mpi::Allreduce( &EkinLocal, &Ekin_, 
		      1, MPI_SUM, domain_.colComm );

      mpi::Allreduce( &EhelmLocal, &Ehelm, 
		      1, MPI_SUM, domain_.colComm );

      // Add the mu*N term for the free energy
      Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

    }

    // Total energy
    Etot_ = Ekin_ + Ecor_;

    // Free energy at finite temperature
    Efree_ = Ehelm + Ecor_;


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateKSEnergyDM  ----- 


  void
  SCFDG::CalculateHarrisEnergy	(  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateHarrisEnergy");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // NOTE: To avoid confusion, all energies in this routine are
    // temporary variables other than EfreeHarris_.
    //
    // The related energies will be computed again in the routine
    //
    // CalculateKSEnergy()
	
    Real Ekin, Eself, Ehart, EVxc, Exc, Ecor;

    // Kinetic energy from the new density matrix.
    Int numSpin = hamDG.NumSpin();
    Ekin = 0.0;
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }

    // Self energy part
    Eself = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }

    // Nonlinear correction part.  This part uses the Hartree energy and
    // XC correlation energy from the old electron density.

    Real EhartLocal = 0.0, EVxcLocal = 0.0;
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	    DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
	    DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
	    DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

	    for (Int p=0; p < density.Size(); p++) {
	      EVxcLocal  += vxc(p) * density(p);
	      EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
	    }

	  } // own this element
	} // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

    Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
    // Use the previous exchange-correlation energy
    Exc    = Exc_;

    // Correction energy.  
    Ecor   = (Exc - EVxc) - Ehart - Eself;
  
    // Harris free energy functional
    if( hamDG.NumOccupiedState() == 
	hamDG.NumStateTotal() ){
      // Zero temperature
      EfreeHarris_ = Ekin + Ecor;
    }
    else{
      // Finite temperature
      EfreeHarris_ = 0.0;
      Real fermi = fermi_;
      Real Tbeta = Tbeta_;
      for(Int l=0; l< eigVal.m(); l++) {
	Real eig = eigVal(l);
	if( eig - fermi >= 0){
	  EfreeHarris_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
	}
	else{
	  EfreeHarris_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
	}
      }
      EfreeHarris_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
    }

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateHarrisEnergy  ----- 

  void
  SCFDG::CalculateHarrisEnergyDM(
				 DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>& distFDMMat )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateHarrisEnergyDM");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    // NOTE: To avoid confusion, all energies in this routine are
    // temporary variables other than EfreeHarris_.
    //
    // The related energies will be computed again in the routine
    //
    // CalculateKSEnergy()
	
    Real Ehelm, Eself, Ehart, EVxc, Exc, Ecor;

    Int numSpin = hamDG.NumSpin();

    // Self energy part
    Eself = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }


    // Nonlinear correction part.  This part uses the Hartree energy and
    // XC correlation energy from the old electron density.

    Real EhartLocal = 0.0, EVxcLocal = 0.0;
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	    DblNumVec&  vxc          = hamDG.Vxc().LocalMap()[key];
	    DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
	    DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

	    for (Int p=0; p < density.Size(); p++) {
	      EVxcLocal  += vxc(p) * density(p);
	      EhartLocal += 0.5 * vhart(p) * ( density(p) + pseudoCharge(p) );
	    }

	  } // own this element
	} // for (i)

    mpi::Allreduce( &EVxcLocal, &EVxc, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

    Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
    EVxc  *= domain_.Volume() / domain_.NumGridTotalFine();
    // Use the previous exchange-correlation energy
    Exc    = Exc_;


    // Correction energy.  
    Ecor   = (Exc - EVxc) - Ehart - Eself;



    // The Helmholtz part of the free energy
    //   Ehelm = -2/beta Tr[log(1+exp(mu-H))] + mu*N_e
    // FIXME Put the above documentation to the proper place like the hpp
    // file
    if( 1 ) {
      Real EhelmLocal = 0.0;
      Ehelm = 0.0;

      // Compute the trace of the energy density matrix in each element
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumMat& localFDM = distFDMMat.LocalMap()[
							  ElemMatKey(key, key)];

	      for( Int a = 0; a < localFDM.m(); a++ ){
		EhelmLocal += localFDM(a,a);
	      }
	    } // own this element
	  } // for (i)

      mpi::Allreduce( &EhelmLocal, &Ehelm, 
		      1, MPI_SUM, domain_.colComm );

      // Add the mu*N term
      Ehelm += fermi_ * hamDG.NumOccupiedState() * numSpin;

    }
		

    // Harris free energy functional. This has to be the finite
    // temperature formulation

    EfreeHarris_ = Ehelm + Ecor;

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateHarrisEnergyDM  ----- 

  void
  SCFDG::CalculateSecondOrderEnergy  (  )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateSecondOrderEnergy");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    HamiltonianDG&  hamDG = *hamDGPtr_;

    DblNumVec&  eigVal         = hamDG.EigVal();
    DblNumVec&  occupationRate = hamDG.OccupationRate();

    // NOTE: To avoid confusion, all energies in this routine are
    // temporary variables other than EfreeSecondOrder_.
    // 
    // This is similar to the situation in 
    //
    // CalculateHarrisEnergy()

	
    Real Ekin, Eself, Ehart, EVtot, Exc, Ecor;

    // Kinetic energy from the new density matrix.
    Int numSpin = hamDG.NumSpin();
    Ekin = 0.0;
    for (Int i=0; i < eigVal.m(); i++) {
      Ekin  += numSpin * eigVal(i) * occupationRate(i);
    }

    // Self energy part
    Eself = 0.0;
    std::vector<Atom>&  atomList = hamDG.AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
      Int type = atomList[a].type;
      Eself +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }


    // Nonlinear correction part.  This part uses the Hartree energy and
    // XC correlation energy from the OUTPUT electron density, but the total
    // potential is the INPUT one used in the diagonalization process.
    // The density is also the OUTPUT density.
    //
    // NOTE the sign flip in Ehart, which is different from those in KS
    // energy functional and Harris energy functional.

    Real EhartLocal = 0.0, EVtotLocal = 0.0;


    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec&  density      = hamDG.Density().LocalMap()[key];
	    DblNumVec&  vext         = hamDG.Vext().LocalMap()[key];
	    DblNumVec&  vtot         = hamDG.Vtot().LocalMap()[key];
	    DblNumVec&  pseudoCharge = hamDG.PseudoCharge().LocalMap()[key];
	    DblNumVec&  vhart        = hamDG.Vhart().LocalMap()[key];

	    for (Int p=0; p < density.Size(); p++) {
	      EVtotLocal  += (vtot(p) - vext(p)) * density(p);
	      // NOTE the sign flip
	      EhartLocal  += 0.5 * vhart(p) * ( density(p) - pseudoCharge(p) );
	    }

	  } // own this element
	} // for (i)

    mpi::Allreduce( &EVtotLocal, &EVtot, 1, MPI_SUM, domain_.colComm );
    mpi::Allreduce( &EhartLocal, &Ehart, 1, MPI_SUM, domain_.colComm );

    Ehart *= domain_.Volume() / domain_.NumGridTotalFine();
    EVtot *= domain_.Volume() / domain_.NumGridTotalFine();

    // Use the exchange-correlation energy with respect to the new
    // electron density
    Exc = Exc_;
	
    // Correction energy.  
    // NOTE The correction energy in the second order method means
    // differently from that in Harris energy functional or the KS energy
    // functional.
    Ecor   = (Exc + Ehart - Eself) - EVtot;
    // FIXME
    //	statusOFS
    //		<< "Component energy for second order correction formula = " << std::endl
    //		<< "Exc     = " << Exc      << std::endl
    //		<< "Ehart   = " << Ehart    << std::endl
    //		<< "Eself   = " << Eself    << std::endl
    //		<< "EVtot   = " << EVtot    << std::endl
    //		<< "Ecor    = " << Ecor     << std::endl;
    //	


    // Second order accurate free energy functional
    if( hamDG.NumOccupiedState() == 
	hamDG.NumStateTotal() ){
      // Zero temperature
      EfreeSecondOrder_ = Ekin + Ecor;
    }
    else{
      // Finite temperature
      EfreeSecondOrder_ = 0.0;
      Real fermi = fermi_;
      Real Tbeta = Tbeta_;
      for(Int l=0; l< eigVal.m(); l++) {
	Real eig = eigVal(l);
	if( eig - fermi >= 0){
	  EfreeSecondOrder_ += -numSpin /Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
	}
	else{
	  EfreeSecondOrder_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
	}
      }
      EfreeSecondOrder_ += Ecor + fermi * hamDG.NumOccupiedState() * numSpin; 
    }


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateSecondOrderEnergy  ----- 


  void
  SCFDG::CalculateVDW	( Real& VDWEnergy, DblNumMat& VDWForce )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::CalculateVDW");
#endif

    HamiltonianDG&  hamDG = *hamDGPtr_;
    std::vector<Atom>& atomList = hamDG.AtomList();
    Evdw_ = 0.0;
    forceVdw_.Resize( atomList.size(), DIM );
    SetValue( forceVdw_, 0.0 );

    Int numAtom = atomList.size();

    Domain& dm = domain_;

    if( VDWType_ == "DFT-D2"){

      const Int vdw_nspecies = 55;
      Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
      bool need_gradient,newshell;
      const Real vdw_d = 20.0;
      const Real vdw_tol_default = 1e-10;
      const Real vdw_s_pbe = 0.75;
      Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;

      double vdw_c6_dftd2[vdw_nspecies] = 
	{  0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
	   5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
	   10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
	   16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
	   24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
	   38.44,31.74,31.50,29.99, 0.00 };

      double vdw_r0_dftd2[vdw_nspecies] =
	{ 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
	  1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
	  1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
	  1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
	  1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
	  1.881,1.892,1.892,1.881,1.000 };

      for(Int i=0; i<vdw_nspecies; i++) {
	vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
	vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
      }

      DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
      DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
      SetValue( vdw_c6, 0.0 );
      SetValue( vdw_r0, 0.0 );

      for(Int i=0; i<vdw_nspecies; i++) {
        for(Int j=0; j<vdw_nspecies; j++) {
	  vdw_c6(i,j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
	  vdw_r0(i,j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
        }
      }

      Real vdw_s;
      if (XCType_ == "XC_GGA_XC_PBE") {
	vdw_s=vdw_s_pbe;
      }
      else {
	throw std::logic_error( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE!" );
      }

      // Calculate the number of atom types.
      //    Real numAtomType = 0;   
      //    for(Int a=0; a< atomList.size() ; a++) {
      //      Int type1 = atomList[a].type;
      //      Int a1 = 0;
      //      Int a2 = 0;
      //      for(Int b=0; b<a ; b++) {
      //        a1 = a1 + 1;
      //        Int type2 = atomList[b].type;
      //        if ( type1 != type2 ) {
      //          a2 = a2 + 1;
      //        }
      //      }
      //
      //      if ( a1 == a2 ) {
      //        numAtomType = numAtomType + 1;
      //      }
      //
      //    }
      //

      //    IntNumVec  atomType ( numAtomType );
      //    SetValue( atomType, 0 );

      //    Real numAtomType1 = 0;
      //    atomType(0) = atomList[0].type;


      //    for(Int a=0; a< atomList.size() ; a++) {
      //      Int type1 = atomList[a].type;
      //      Int a1 = 0;
      //      Int a2 = 0;
      //      for(Int b=0; b<a ; b++) {
      //        a1 = a1 + 1;
      //        Int type2 = atomList[b].type;
      //        if ( type1 != type2 ) {
      //          a2 = a2 + 1;
      //        }
      //      }
      //      if ( a1 == a2 ) {
      //        numAtomType1 = numAtomType1 + 1;
      //        atomType(numAtomType1-1) = atomList[a].type;
      //      }
      //    }


      //    DblNumMat  vdw_c6 ( numAtomType, numAtomType );
      //    DblNumMat  vdw_r0 ( numAtomType, numAtomType );
      //    SetValue( vdw_c6, 0.0 );
      //    SetValue( vdw_r0, 0.0 );
      //
      //    for(Int i=0; i< numAtomType; i++) {
      //      for(Int j=0; j< numAtomType; j++) {
      //        vdw_c6(i,j)=std::sqrt(vdw_c6_dftd2[atomType(i)-1]*vdw_c6_dftd2[atomType(j)-1]);
      //        //vdw_r0(i,j)=(vdw_r0_dftd2(atomType(i))+vdw_r0_dftd2(atomType(j)))/Bohr_Ang;
      //        vdw_r0(i,j)=(vdw_r0_dftd2[atomType(i)-1]+vdw_r0_dftd2[atomType(j)-1]);
      //      }
      //    }

      //    statusOFS << "vdw_c6 = " << vdw_c6 << std::endl;
      //    statusOFS << "vdw_r0 = " << vdw_r0 << std::endl;


      for(Int ii=-1; ii<2; ii++) {
	for(Int jj=-1; jj<2; jj++) {
	  for(Int kk=-1; kk<2; kk++) {

	    for(Int i=0; i<atomList.size(); i++) {
	      Int iType = atomList[i].type;
	      for(Int j=0; j<(i+1); j++) {
		Int jType = atomList[j].type;
       
		Real rx = atomList[i].pos[0] - atomList[j].pos[0] + ii * dm.length[0];
		Real ry = atomList[i].pos[1] - atomList[j].pos[1] + jj * dm.length[1];
		Real rz = atomList[i].pos[2] - atomList[j].pos[2] + kk * dm.length[2];
		Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

		if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

		  Real sfact = vdw_s;
		  if ( i == j ) sfact = sfact * 0.5;

		  Real c6 = vdw_c6(iType-1, jType-1);
		  Real r0 = vdw_r0(iType-1, jType-1);
		  //Real c6 = std::sqrt( vdw_c6_dftd2[iType-1] * vdw_c6_dftd2[jType-1] );
		  //Real r0 = vdw_r0_dftd2[iType-1] + vdw_r0_dftd2[jType-1];

		  Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
		  Real fr = 1.0 / ( 1.0 + ex );
		  Real c6r6 = c6 / pow(rr, 6.0);

		  // Contribution to energy
		  Evdw_ = Evdw_ - sfact * fr * c6r6;

		  // Contribution to force
		  if( i != j ) {

		    Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
		    Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

		    //Real fx = grad * rx * dm.length[0];
		    //Real fy = grad * ry * dm.length[1];
		    //Real fz = grad * rz * dm.length[2];
		    Real fx = grad * rx;
		    Real fy = grad * ry;
		    Real fz = grad * rz;

		    forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
		    forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
		    forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
		    forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
		    forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
		    forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

		  } // end for i != j

		} // end if


	      } // end for j
	    } // end for i

	  } // end for ii
	} // end for jj
      } // end for kk


      //#endif 

    } // If DFT-D2


    VDWEnergy = Evdw_;
    VDWForce = forceVdw_;


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::CalculateVDW  ----- 



  void
  SCFDG::AndersonMix	( 
			 Int             iter, 
			 Real            mixStepLength,
			 std::string     mixType,
			 DistDblNumVec&  distvMix,
			 DistDblNumVec&  distvOld,
			 DistDblNumVec&  distvNew,
			 DistDblNumMat&  dfMat,
			 DistDblNumMat&  dvMat )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::AndersonMix");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );
  
    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
	
    distvMix.SetComm(domain_.colComm);
    distvOld.SetComm(domain_.colComm);
    distvNew.SetComm(domain_.colComm);
    dfMat.SetComm(domain_.colComm);
    dvMat.SetComm(domain_.colComm);
	
    // Residual 
    DistDblNumVec distRes;
    // Optimal input potential in Anderon mixing.
    DistDblNumVec distvOpt; 
    // Optimal residual in Anderson mixing
    DistDblNumVec distResOpt; 
    // Preconditioned optimal residual in Anderson mixing
    DistDblNumVec distPrecResOpt;

    distRes.SetComm(domain_.colComm);
    distvOpt.SetComm(domain_.colComm);
    distResOpt.SetComm(domain_.colComm);
    distPrecResOpt.SetComm(domain_.colComm);
	
    // *********************************************************************
    // Initialize
    // *********************************************************************
    Int ntot  = hamDGPtr_->NumUniformGridElemFine().prod();
	
    // Number of iterations used, iter should start from 1
    Int iterused = std::min( iter-1, mixMaxDim_ ); 
    // The current position of dfMat, dvMat
    Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
    // The next position of dfMat, dvMat
    Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

    distRes.Prtn()          = elemPrtn_;
    distvOpt.Prtn()         = elemPrtn_;
    distResOpt.Prtn()       = elemPrtn_;
    distPrecResOpt.Prtn()   = elemPrtn_;

	
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    DblNumVec  emptyVec( ntot );
	    SetValue( emptyVec, 0.0 );
	    distRes.LocalMap()[key]        = emptyVec;
	    distvOpt.LocalMap()[key]       = emptyVec;
	    distResOpt.LocalMap()[key]     = emptyVec;
	    distPrecResOpt.LocalMap()[key] = emptyVec;
	  } // if ( own this element )
	} // for (i)


	
    // *********************************************************************
    // Anderson mixing
    // *********************************************************************
	
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    // res(:) = vOld(:) - vNew(:) is the residual
	    distRes.LocalMap()[key] = distvOld.LocalMap()[key];
	    blas::Axpy( ntot, -1.0, distvNew.LocalMap()[key].Data(), 1, 
			distRes.LocalMap()[key].Data(), 1 );

	    distvOpt.LocalMap()[key]   = distvOld.LocalMap()[key];
	    distResOpt.LocalMap()[key] = distRes.LocalMap()[key];


	    // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
	    // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
	    if( iter > 1 ){
	      blas::Scal( ntot, -1.0, dfMat.LocalMap()[key].VecData(ipos-1), 1 );
	      blas::Axpy( ntot, 1.0,  distRes.LocalMap()[key].Data(), 1, 
			  dfMat.LocalMap()[key].VecData(ipos-1), 1 );
	      blas::Scal( ntot, -1.0, dvMat.LocalMap()[key].VecData(ipos-1), 1 );
	      blas::Axpy( ntot, 1.0,  distvOld.LocalMap()[key].Data(),  1, 
			  dvMat.LocalMap()[key].VecData(ipos-1), 1 );
	    }
	  } // own this element
	} // for (i)



    // For iter == 1, Anderson mixing is the same as simple mixing.
    if( iter > 1 ){

      Int nrow = iterused;

      // Normal matrix FTF = F^T * F
      DblNumMat FTFLocal( nrow, nrow ), FTF( nrow, nrow );
      SetValue( FTFLocal, 0.0 );
      SetValue( FTF, 0.0 );

      // Right hand side FTv = F^T * vout
      DblNumVec FTvLocal( nrow ), FTv( nrow );
      SetValue( FTvLocal, 0.0 );
      SetValue( FTv, 0.0 );

      // Local construction of FTF and FTv
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      DblNumMat& df     = dfMat.LocalMap()[key];
	      DblNumVec& res    = distRes.LocalMap()[key];
	      for( Int q = 0; q < nrow; q++ ){
		FTvLocal(q) += blas::Dot( ntot, df.VecData(q), 1,
					  res.Data(), 1 );

		for( Int p = q; p < nrow; p++ ){
		  FTFLocal(p, q) += blas::Dot( ntot, df.VecData(p), 1, 
					       df.VecData(q), 1 );
		  if( p > q )
		    FTFLocal(q,p) = FTFLocal(p,q);
		} // for (p)
	      } // for (q)

	    } // own this element
	  } // for (i)
		
      // Reduce the data
      mpi::Allreduce( FTFLocal.Data(), FTF.Data(), nrow * nrow, 
		      MPI_SUM, domain_.colComm );
      mpi::Allreduce( FTvLocal.Data(), FTv.Data(), nrow, 
		      MPI_SUM, domain_.colComm );

      // All processors solve the least square problem

      // FIXME Magic number for pseudo-inverse
      Real rcond = 1e-6;
      Int rank;

      DblNumVec  S( nrow );

      // FTv = pinv( FTF ) * res
      lapack::SVDLeastSquare( nrow, nrow, 1, 
			      FTF.Data(), nrow, FTv.Data(), nrow,
			      S.Data(), rcond, &rank );

      statusOFS << "Rank of dfmat = " << rank <<
	", rcond = " << rcond << std::endl;

      // Update vOpt, resOpt. 
      // FTv = Y^{\dagger} r as in the usual notation.
      // 
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      // vOpt   -= dv * FTv
	      blas::Gemv('N', ntot, nrow, -1.0, dvMat.LocalMap()[key].Data(),
			 ntot, FTv.Data(), 1, 1.0, 
			 distvOpt.LocalMap()[key].Data(), 1 );

	      // resOpt -= df * FTv
	      blas::Gemv('N', ntot, nrow, -1.0, dfMat.LocalMap()[key].Data(),
			 ntot, FTv.Data(), 1, 1.0, 
			 distResOpt.LocalMap()[key].Data(), 1 );
	    } // own this element
	  } // for (i)
    } // (iter > 1)

	
	
 
    if( mixType == "kerker+anderson" ){
      KerkerPrecond( distPrecResOpt, distResOpt );
    }
    else if( mixType == "anderson" ){
      for( Int k = 0; k < numElem_[2]; k++ )
	for( Int j = 0; j < numElem_[1]; j++ )
	  for( Int i = 0; i < numElem_[0]; i++ ){
	    Index3 key( i, j, k );
	    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	      distPrecResOpt.LocalMap()[key] = 
		distResOpt.LocalMap()[key];
	    } // own this element
	  } // for (i)
    }
    else{
      throw std::runtime_error("Invalid mixing type.");
    }
	
	
	

    // Update dfMat, dvMat, vMix 
    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
	for( Int i = 0; i < numElem_[0]; i++ ){
	  Index3 key( i, j, k );
	  if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
	    // dfMat(:, inext-1) = res(:)
	    // dvMat(:, inext-1) = vOld(:)
	    blas::Copy( ntot, distRes.LocalMap()[key].Data(), 1, 
			dfMat.LocalMap()[key].VecData(inext-1), 1 );
	    blas::Copy( ntot, distvOld.LocalMap()[key].Data(),  1, 
			dvMat.LocalMap()[key].VecData(inext-1), 1 );

	    // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
	    distvMix.LocalMap()[key] = distvOpt.LocalMap()[key];
	    blas::Axpy( ntot, -mixStepLength, 
			distPrecResOpt.LocalMap()[key].Data(), 1, 
			distvMix.LocalMap()[key].Data(), 1 );
	  } // own this element
	} // for (i)



#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::AndersonMix  ----- 

  void
  SCFDG::KerkerPrecond ( 
			DistDblNumVec&  distPrecResidual,
			const DistDblNumVec&  distResidual )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::KerkerPrecond");
#endif
    Int mpirank, mpisize;
    MPI_Comm_rank( domain_.comm, &mpirank );
    MPI_Comm_size( domain_.comm, &mpisize );

    Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
    Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);
	
    DistFourier& fft = *distfftPtr_;
    //DistFourier.SetComm(domain_.colComm);

    Int ntot      = fft.numGridTotal;
    Int ntotLocal = fft.numGridLocal;

    Index3 numUniformGridElem = hamDGPtr_->NumUniformGridElem();

    // Convert distResidual to tempVecLocal in distributed row vector format
    DblNumVec  tempVecLocal;

    DistNumVecToDistRowVec(
			   distResidual,
			   tempVecLocal,
			   domain_.numGridFine,
			   numElem_,
			   fft.localNzStart,
			   fft.localNz,
			   fft.isInGrid,
			   domain_.colComm );

    // NOTE Fixed KerkerB parameter
    //
    // From the point of view of the elliptic preconditioner
    //
    // (-\Delta + 4 * pi * b) r_p = -Delta r
    //
    // The Kerker preconditioner in the Fourier space is
    //
    // k^2 / (k^2 + 4 * pi * b)
    //
    // or using gkk = k^2 /2 
    //
    // gkk / ( gkk + 2 * pi * b )
    //
    // Here we choose KerkerB to be a fixed number.
    Real KerkerB = 0.1; 

    if( fft.isInGrid ){

      for( Int i = 0; i < ntotLocal; i++ ){
	fft.inputComplexVecLocal(i) = Complex( 
					      tempVecLocal(i), 0.0 );
      }
      fftw_execute( fft.forwardPlan );

      for( Int i = 0; i < ntotLocal; i++ ){
	// Do not touch the zero frequency
	if( fft.gkkLocal(i) != 0 ){
	  fft.outputComplexVecLocal(i) *= fft.gkkLocal(i) / 
	    ( fft.gkkLocal(i) + 2.0 * PI * KerkerB );
	}
      }
      fftw_execute( fft.backwardPlan );

      for( Int i = 0; i < ntotLocal; i++ ){
	tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
      }
    } // if (fft.isInGrid)

    // Convert tempVecLocal to distPrecResidual in the DistNumVec format 

    DistRowVecToDistNumVec(
			   tempVecLocal,
			   distPrecResidual,
			   domain_.numGridFine,
			   numElem_,
			   fft.localNzStart,
			   fft.localNz,
			   fft.isInGrid,
			   domain_.colComm );


#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::KerkerPrecond  ----- 


  void
  SCFDG::PrintState	( )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::PrintState");
#endif
  
    HamiltonianDG&  hamDG = *hamDGPtr_;

    statusOFS << std::endl << "Eigenvalues in the global domain." << std::endl;
    for(Int i = 0; i < hamDG.EigVal().m(); i++){
      Print(statusOFS, 
	    "band#    = ", i, 
	    "eigval   = ", hamDG.EigVal()(i),
	    "occrate  = ", hamDG.OccupationRate()(i));
    }
    statusOFS << std::endl;
    // FIXME
    //	statusOFS 
    //		<< "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself" << std::endl
    //	  << "       Etot  = Ekin + Ecor" << std::endl
    //	  << "       Efree = Etot	+ Entropy" << std::endl << std::endl;
    Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
    //			FIXME
    //	Print(statusOFS, "EfreeSecondOrder  = ",  EfreeSecondOrder_, "[au]");
    Print(statusOFS, "Etot              = ",  Etot_, "[au]");
    Print(statusOFS, "Efree             = ",  Efree_, "[au]");
    Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
    Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
    Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
    Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
    Print(statusOFS, "Evdw              = ",  Evdw_, "[au]"); 
    Print(statusOFS, "Eself             = ",  Eself_, "[au]");
    Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
    Print(statusOFS, "Fermi             = ",  fermi_, "[au]");

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::PrintState  ----- 


  void  
  SCFDG::LastSCF( Real& efreeHarris, Real& etot, Real& efree, Real& ekin, 
		  Real& ehart, Real& eVxc, Real& exc, Real& evdw, Real& eself, 
		  Real& ecor, Real& fermi, Real& scfOuterNorm, Real& efreeDifPerAtom )
  {
#ifndef _RELEASE_
    PushCallStack("SCFDG::LastSCF");
#endif
  
    efreeHarris       = EfreeHarris_;
    etot              = Etot_;
    efree             = Efree_;
    ekin              = Ekin_;
    ehart             = Ehart_;
    eVxc              = EVxc_;
    exc               = Exc_; 
    evdw              = Evdw_; 
    eself             = Eself_;
    ecor              = Ecor_;
    fermi             = fermi_;
    scfOuterNorm      = scfOuterNorm_;
    efreeDifPerAtom   = efreeDifPerAtom_;

#ifndef _RELEASE_
    PopCallStack();
#endif

    return ;
  } 		// -----  end of method SCFDG::LastSCF  ----- 

} // namespace dgdft
