/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin and Wei Hu

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
    if( mpirank < numProcTotalPEXSI_ ){
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
    eigMaxIter_    = esdfParam.eigMaxIter;
		scfInnerTolerance_  = esdfParam.scfInnerTolerance;
		scfInnerMaxIter_    = esdfParam.scfInnerMaxIter;
		scfOuterTolerance_  = esdfParam.scfOuterTolerance;
		scfOuterMaxIter_    = esdfParam.scfOuterMaxIter;
    numUnusedState_ = esdfParam.numUnusedState;
		SVDBasisTolerance_  = esdfParam.SVDBasisTolerance;
    isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
		isRestartDensity_ = esdfParam.isRestartDensity;
		isRestartWfn_     = esdfParam.isRestartWfn;
		isOutputDensity_  = esdfParam.isOutputDensity;
		isOutputWfnElem_     = esdfParam.isOutputWfnElem;
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

    // Provide a communicator for PEXSI
    Int npPerPole_ = numProcRowPEXSI_ * numProcColPEXSI_;

    if( npPerPole_ > mpisize )
      throw std::logic_error("The number of processors is not enough for one pole.");

    numProcTotalPEXSI_   = std::min( 
        ( mpisize / npPerPole_ ) * npPerPole_,
        npPerPole_ * esdfParam.numPole );
    Int isProcPEXSI = (mpirank < numProcTotalPEXSI_) ? 1 : 0;

    MPI_Comm_split( domain_.comm, isProcPEXSI, mpirank, &pexsiComm_ );

    // Initialize PEXSI
    if( mpirank < numProcTotalPEXSI_ ){
      pexsiPlan_        = PPEXSIPlanInitialize(
          pexsiComm_,
          numProcRowPEXSI_,
          numProcColPEXSI_,
          mpirank,
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
    dmCol_ = numElem_[0] * numElem_[1] * numElem_[2];
    dmRow_ = mpisize / dmCol_;
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
    std::istringstream rhoStream;      
    SeparateRead( restartDensityFileName_, rhoStream );

    Real sumDensityLocal = 0.0, sumDensity = 0.0;

    for( Int k = 0; k < numElem_[2]; k++ )
      for( Int j = 0; j < numElem_[1]; j++ )
        for( Int i = 0; i < numElem_[0]; i++ ){
          Index3 key( i, j, k );
          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){

            std::vector<DblNumVec> grid(DIM);
            for( Int d = 0; d < DIM; d++ ){
              deserialize( grid[d], rhoStream, NO_MASK );
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

  } // else using the zero initial guess
  else {
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
    std::istringstream wfnStream;      
    SeparateRead( restartWfnFileName_, wfnStream );
   
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

    Print( statusOFS, "Restart basis functions." );
  } 
  else{ 
 
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
  {
		PeriodicUniformToLGLMat_.resize(DIM);
    PeriodicUniformFineToLGLMat_.resize(DIM);

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
		Point3 lengthUniform = dmExtElem.length;

		std::vector<DblNumVec>  LGLGrid(DIM);
		LGLMesh( dmElem, numLGL, LGLGrid ); 
		std::vector<DblNumVec>  UniformGrid(DIM);
    UniformMesh( dmExtElem, UniformGrid );
    std::vector<DblNumVec>  UniformGridFine(DIM);
    UniformMeshFine( dmExtElem, UniformGridFine );

//		for( Int d = 0; d < DIM; d++ ){
//			DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
//			localMat.Resize( numLGL[d], numUniform[d] );
//			SetValue( localMat, 0.0 );
//			Int maxK;
//			if (numUniform[d] % 2 == 0)
//				maxK = numUniform[d] / 2 - 1;
//			else
//				maxK = ( numUniform[d] - 1 ) / 2;
//			for( Int j = 0; j < numUniform[d]; j++ )
//				for( Int i = 0; i < numLGL[d]; i++ ){
//					// 1.0 accounts for the k=0 mode
//					localMat(i,j) = 1.0;
//					for( Int k = 1; k < maxK; k++ ){
//						localMat(i,j) += 2.0 * std::cos( 
//								2 * PI * k / lengthUniform[d] * 
//								( LGLGrid[d](i) - j * lengthUniform[d] / numUniform[d] ) );
//					} // for (k)
//					localMat(i,j) /= numUniform[d];
//				} // for (i)
//		} // for (d)

		for( Int d = 0; d < DIM; d++ ){
			DblNumMat&  localMat = PeriodicUniformToLGLMat_[d];
			localMat.Resize( numLGL[d], numUniform[d] );
			SetValue( localMat, 0.0 );
			DblNumVec KGrid( numUniform[d] );
			for( Int i = 0; i <= numUniform[d] / 2; i++ ){
				KGrid(i) = i * 2.0 * PI / lengthUniform[d];
			}
			for( Int i = numUniform[d] / 2 + 1; i < numUniform[d]; i++ ){
				KGrid(i) = ( i - numUniform[d] ) * 2.0 * PI / lengthUniform[d];
			}

			for( Int j = 0; j < numUniform[d]; j++ )
				for( Int i = 0; i < numLGL[d]; i++ ){
					localMat(i, j) = 0.0;
					for( Int k = 0; k < numUniform[d]; k++ ){
						localMat(i,j) += std::cos( KGrid(k) * ( LGLGrid[d](i) -
									UniformGrid[d](j) ) ) / numUniform[d];
					} // for (k)
				} // for (i)
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

			for( Int j = 0; j < numUniformFine[d]; j++ )
				for( Int i = 0; i < numLGL[d]; i++ ){
					localMatFine(i, j) = 0.0;
					for( Int k = 0; k < numUniformFine[d]; k++ ){
						localMatFine(i,j) += std::cos( KGridFine(k) * ( LGLGrid[d](i) -
									UniformGridFine[d](j) ) ) / numUniformFine[d];
					} // for (k)
				} // for (i)
		} // for (d)

    // Assume the initial error is O(1)
    scfOuterNorm_ = 1.0;
    scfInnerNorm_ = 1.0;

#if ( _DEBUGlevel_ >= 1 )
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
#endif
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::Setup  ----- 


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

  HamiltonianDG&  hamDG = *hamDGPtr_;


  // Compute the exchange-correlation potential and energy
  GetTime( timeSta );
	hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );
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
  
	bool isSCFConverged = false;

	scfTotalInnerIter_  = 0;

  for (Int iter=1; iter <= scfOuterMaxIter_; iter++) {
    if ( isSCFConverged ) break;
		
		
		// Performing each iteartion
		{
			std::ostringstream msg;
			msg << "Outer SCF iteration # " << iter;
			PrintBlock( statusOFS, msg.str() );
		}

    GetTime( timeIterStart );
		
		// *********************************************************************
		// Update the local potential in the extended element and the element.
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

						// Skip the interpoation if there is no adaptive local
						// basis function.  
						if( eigSol.Psi().NumState() == 0 ){
							hamDG.BasisLGL().LocalMap()[key].Resize( numLGLGrid.prod(), 0 );  
							continue;
						}

						// Add the external barrier potential
						// In the periodized version, the external potential depends
						// on the potential V in order to result in a C^{inf}
						// potential.
						if( isPeriodizePotential_ ){

							// Compute the bubble function in the extended element.

							Domain& dmExtElem = eigSol.FFT().domain;
							std::vector<DblNumVec> gridpos(DIM);
							//UniformMesh ( dmExtElem, gridpos );
              UniformMeshFine ( dmExtElem, gridpos );
							// Bubble function along each dimension
							std::vector<DblNumVec> vBubble(DIM);

							for( Int d = 0; d < DIM; d++ ){
								Real length   = dmExtElem.length[d];
								Int numGrid   = dmExtElem.numGridFine[d];
								Real posStart = dmExtElem.posStart[d]; 
								Real EPS = 1e-10; // Criterion for distancePeriodize_
								vBubble[d].Resize( numGrid );
								SetValue( vBubble[d], 1.0 );

								if( distancePeriodize_[d] > EPS ){
									Real lb = posStart + distancePeriodize_[d];
									Real rb = posStart + length - distancePeriodize_[d];
									for( Int p = 0; p < numGrid; p++ ){
										if( gridpos[d][p] > rb ){
											vBubble[d][p] = Smoother( (gridpos[d][p] - rb ) / 
													distancePeriodize_[d]);
										}

										if( gridpos[d][p] < lb ){
											vBubble[d][p] = Smoother( (lb - gridpos[d][p] ) / 
													distancePeriodize_[d]);
										}
									}
								}
							} // for (d)

#if ( _DEBUGlevel_ >= 1  )
							statusOFS << "gridpos[0] = " << std::endl << gridpos[0] << std::endl;
							statusOFS << "vBubble[0] = " << std::endl << vBubble[0] << std::endl;
							statusOFS << "gridpos[1] = " << std::endl << gridpos[1] << std::endl;
							statusOFS << "vBubble[1] = " << std::endl << vBubble[1] << std::endl;
							statusOFS << "gridpos[2] = " << std::endl << gridpos[2] << std::endl;
							statusOFS << "vBubble[2] = " << std::endl << vBubble[2] << std::endl;
#endif

							// Get the potential
							DblNumVec& vext = eigSol.Ham().Vext();
							DblNumVec& vtot = eigSol.Ham().Vtot();

							// Find the max of the potential in the extended element
							Real vtotMax = *std::max_element( &vtot[0], &vtot[0] + vtot.Size() );

							SetValue( vext, 0.0 );
							for( Int gk = 0; gk < dmExtElem.numGridFine[2]; gk++)
								for( Int gj = 0; gj < dmExtElem.numGridFine[1]; gj++ )
									for( Int gi = 0; gi < dmExtElem.numGridFine[0]; gi++ ){
										Int idx = gi + gj * dmExtElem.numGridFine[0] + 
											gk * dmExtElem.numGridFine[0] * dmExtElem.numGridFine[1];
										vext[idx] = ( vtot[idx] - vtotMax ) * 
											( vBubble[0][gi] * vBubble[1][gj] * vBubble[2][gk] - 1.0 );
									} // for (gi)

							// NOTE:
							// Directly modify the vtot.  vext is not used in the
							// matrix-vector multiplication in the eigensolver.
							blas::Axpy( numGridExtElemFine.prod(), 1.0, eigSol.Ham().Vext().Data(), 1,
									eigSol.Ham().Vtot().Data(), 1 );
						} // if ( isPeriodizePotential_ ) 


            // VtotFine to VtotCoarse: Restricting vtot on a fine grid
            // to a coarse grid for computing the basis functions on a
            // coarse grid.

            Int ntotCoarse  = eigSol.FFT().domain.NumGridTotal();
            Int ntotFine  = eigSol.FFT().domain.NumGridTotalFine();

            DblNumVec& vtotFine = eigSol.Ham().Vtot();
            DblNumVec& vtotCoarse = eigSol.Ham().VtotCoarse();

         
            Fourier& fft = eigSol.FFT();

            for( Int ii = 0; ii < ntotFine; ii++ ){
              fft.inputComplexVecFine(ii) = Complex( vtotFine(ii), 0.0 );
            }

            fftw_execute( fft.forwardPlanFine );

            Int PtrC = 0;
            Int PtrF = 0;

            Int iF = 0;
            Int jF = 0;
            Int kF = 0;

            SetValue( fft.outputComplexVec, Z_ZERO );

            for( Int kk = 0; kk < fft.domain.numGrid[2]; kk++ ){
              for( Int jj = 0; jj <  fft.domain.numGrid[1]; jj++ ){
                for( Int ii = 0; ii <  fft.domain.numGrid[0]; ii++ ){
   
                  PtrC = ii + jj * fft.domain.numGrid[0] + kk * fft.domain.numGrid[0] * fft.domain.numGrid[1];

                  if ( (0 <= ii) && (ii <=  fft.domain.numGrid[0] / 2) ) { iF = ii; }
                  else { iF =  fft.domain.numGridFine[0] - fft.domain.numGrid[0] + ii; }

                  if ( (0 <= jj) && (jj <=  fft.domain.numGrid[1] / 2) ) { jF = jj; }
                  else { jF =  fft.domain.numGridFine[1] - fft.domain.numGrid[1] + jj; }

                  if ( (0 <= kk) && (kk <=  fft.domain.numGrid[2] / 2) ) { kF = kk; }
                  else { kF =  fft.domain.numGridFine[2] - fft.domain.numGrid[2] + kk; }

                  PtrF = iF + jF * fft.domain.numGridFine[0] + kF * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];

                  fft.outputComplexVec(PtrC) = fft.outputComplexVecFine(PtrF);

                }
              }
            }

            fftw_execute( fft.backwardPlan );

            for( Int ii = 0; ii < ntotCoarse; ii++ ){
              vtotCoarse(ii) = fft.inputComplexVec(ii).real() / ntotFine;
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

            Int numEig = (eigSol.Psi().NumStateTotal())-numUnusedState_;
            statusOFS << "The current tolerance used by the eigensolver is " 
              << eigTolNow << std::endl;
            statusOFS << "The target number of converged eigenvectors is " 
              << numEig << std::endl;
   
           
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
						statusOFS << "Eigensolver time = " 	<< timeEnd - timeSta
							<< " [s]" << std::endl;


						// Print out the information
						statusOFS << std::endl 
							<< "ALB calculation in extended element " << key << std::endl;
						for(Int ii = 0; ii < eigSol.EigVal().m(); ii++){
							Print(statusOFS, 
									"basis#   = ", ii, 
									"eigval   = ", eigSol.EigVal()(ii),
									"resval   = ", eigSol.ResVal()(ii));
						}
						statusOFS << std::endl;


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

//#ifdef _USE_OPENMP_
//#pragma omp parallel
            {
//#endif
//#ifdef _USE_OPENMP_
//#pragma omp for schedule (dynamic,1) nowait
//#endif
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

//#ifdef _USE_OPENMP_
            }
//#endif
						GetTime( timeEnd );
						statusOFS << "Time for interpolating basis = " 	<< timeEnd - timeSta
							<< " [s]" << std::endl;

						
            // Post processing for the basis functions on the LGL grid.
            // Perform GEMM and threshold the basis functions for the
            // small matrix.
            //
            // This method might have lower numerical accuracy, but is
            // much more scalable than other known options.
						if(1){
							
              GetTime( timeSta );
						
              {
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
                Int width = psi.NumStateTotal();

                Int widthBlocksize = width / mpisizeRow;
                Int heightBlocksize = height / mpisizeRow;
                Int heightLGLBlocksize = heightLGL / mpisizeRow;

                Int widthLocal = widthBlocksize;
                Int heightLocal = heightBlocksize;
                Int heightLGLLocal = heightLGLBlocksize;

                if(mpirankRow < (width % mpisizeRow)){
                  widthLocal = widthBlocksize + 1;
                }

                if(mpirankRow == (mpisizeRow - 1)){
                  heightLocal = heightBlocksize + height % mpisizeRow;
                }

                if(mpirankRow == (mpisizeRow - 1)){
                  heightLGLLocal = heightLGLBlocksize + heightLGL % mpisizeRow;
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
  
                AlltoallForward (localBasis, localBasisRow, domain_.rowComm);
               
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

                MPI_Bcast(S.Data(), numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
                MPI_Bcast(U.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
                MPI_Bcast(VT.Data(), numBasisTotal * numBasisTotal, MPI_DOUBLE, 0, domain_.rowComm);
               
                MPI_Barrier( domain_.rowComm );

                // Broadcast U and S

                Int  numSVDBasisTotal = 0;	
                for( Int g = 0; g < numBasisTotal; g++ ){
                  S[g] = std::sqrt( S[g] );
                  if( S[g] / S[0] > SVDBasisTolerance_ )
                    numSVDBasisTotal++;
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



                statusOFS << "Singular values of the basis = " 
									<< S << std::endl;

								statusOFS << "Number of significant SVD basis = " 
                  << numSVDBasisTotal << std::endl;

           
                MPI_Barrier( domain_.rowComm );

              }


            
              GetTime( timeEnd );
							statusOFS << "Time for SVD of basis = " 	<< timeEnd - timeSta
								<< " [s]" << std::endl;
						
          
              MPI_Barrier( domain_.rowComm );
              
              //if(1){
              //  statusOFS << std::endl<< "All processors exit with abort in scf_dg.cpp." << std::endl;
              //  abort();
             // }



            MPI_Barrier( domain_.rowComm );
            
            } // if(1)


					} // own this element
				} // for (i)



    
    GetTime( timeBasisEnd );



		statusOFS << "Total time for generating the adaptive local basis function is " <<
			timeBasisEnd - timeBasisSta << " [s]" << std::endl << std::endl;

		
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
		statusOFS << "Time for outer SCF iteration is " <<
			timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

		// *********************************************************************
		// Post processing 
		// *********************************************************************
		

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
			Print(statusOFS, "OUTERSCF: inner norm(out-in)/norm(in) = ", scfInnerNorm_ ); 
			Print(statusOFS, "OUTERSCF: outer norm(out-in)/norm(in) = ", scfOuterNorm_ ); 
			statusOFS << std::endl;
		}

//		// Print out the state variables of the current iteration
//    PrintState( );

    if( scfOuterNorm_ < scfOuterTolerance_ ){
      /* converged */
      Print( statusOFS, "Outer SCF is converged!\n" );
			statusOFS << std::endl;
      isSCFConverged = true;
    }

		// Potential mixing for the outer SCF iteration. or no mixing at all anymore?
		// It seems that no mixing is the best.
	
		GetTime( timeIterEnd );
		statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
			<< " [s]" << std::endl;
  }


  // Output the electron density
  if( isOutputDensity_ ){
    if(1)
    {
      statusOFS << std::endl 
        << "Output the electron density on the global grid" << std::endl;


      // Output the electron density on the uniform grid in each element
      std::ostringstream rhoStream;      

      NumTns<std::vector<DblNumVec> >& uniformGridElem =
        hamDG.UniformGridElem();

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = hamDG.Density().LocalMap()[key];
              std::vector<DblNumVec>& grid = uniformGridElem(i, j, k);
              for( Int d = 0; d < DIM; d++ ){
                serialize( grid[d], rhoStream, NO_MASK );
              }
              serialize( denVec, rhoStream, NO_MASK );
            }
          } // for (i)
      SeparateWrite( restartDensityFileName_, rhoStream );
    }

    if(0)
    {
      // Output the electron density on the LGL grid in each element
      std::ostringstream rhoStream;      

      NumTns<std::vector<DblNumVec> >& LGLGridElem =
        hamDG.LGLGridElem();

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumVec&  denVec = hamDG.DensityLGL().LocalMap()[key];
              std::vector<DblNumVec>& grid = LGLGridElem(i, j, k);
              for( Int d = 0; d < DIM; d++ ){
                serialize( grid[d], rhoStream, NO_MASK );
              }
              serialize( denVec, rhoStream, NO_MASK );
            }
          } // for (i)
      SeparateWrite( "DENLGL", rhoStream );
    }
  } // if ( output density )

  for( Int k = 0; k < numElem_[2]; k++ )
    for( Int j = 0; j < numElem_[1]; j++ )
      for( Int i = 0; i < numElem_[0]; i++ ){
        Index3 key( i, j, k );
        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
          if( isOutputPotExtElem_ )
          {
            statusOFS 
              << std::endl 
              << "Output the total potential in the extended element."
              << std::endl;
            std::ostringstream potStream;      
            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];

            // Generate the uniform mesh on the extended element.
            std::vector<DblNumVec> gridpos;
            //UniformMesh ( eigSol.FFT().domain, gridpos );
            UniformMeshFine ( eigSol.FFT().domain, gridpos );
            for( Int d = 0; d < DIM; d++ ){
              serialize( gridpos[d], potStream, NO_MASK );
            }
            serialize( eigSol.Ham().Vtot(), potStream, NO_MASK );
            serialize( eigSol.Ham().Vext(), potStream, NO_MASK );
            SeparateWrite( "POTEXT", potStream);
          }

          if( isOutputWfnExtElem_ )
          {
            statusOFS 
              << std::endl 
              << "Output the wavefunctions in the extended element."
              << std::endl;

            EigenSolver&  eigSol = distEigSolPtr_->LocalMap()[key];
            std::ostringstream wavefunStream;      

            // Generate the uniform mesh on the extended element.
            std::vector<DblNumVec> gridpos;
            UniformMesh ( eigSol.FFT().domain, gridpos );
            for( Int d = 0; d < DIM; d++ ){
              serialize( gridpos[d], wavefunStream, NO_MASK );
            }
            serialize( eigSol.Psi().Wavefun(), wavefunStream, NO_MASK );
            SeparateWrite( "WFNEXT", wavefunStream);
          }

          if( isOutputWfnElem_ )
          {
            // Output the wavefunctions in the extended element.
            std::ostringstream wavefunStream;      

            // Generate the uniform mesh on the extended element.
            std::vector<DblNumVec>& gridpos = hamDG.LGLGridElem()(i,j,k);
            for( Int d = 0; d < DIM; d++ ){
              serialize( gridpos[d], wavefunStream, NO_MASK );
            }
            serialize( hamDG.BasisLGL().LocalMap()[key], wavefunStream, NO_MASK );
            SeparateWrite( "WFNELEM", wavefunStream);
          }

        } // (own this element)
      } // for (i)

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::Iterate  ----- 


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


        hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );
       
      
        hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );


        
       
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

        hamDG.CalculateVtot( hamDG.Vtot() );
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


    // Method 1_1: Using diagonalization method, but with a more
    // versatile choice of processors for using ScaLAPACK.
    //
    if( solutionMethod_ == "diag" && 1 ){
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

#if ( _DEBUGlevel_ >= 0 )
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
#if ( _DEBUGlevel_ >= 0 )
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
#if ( _DEBUGlevel_ >= 0 )
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
#if ( _DEBUGlevel_ >= 0 )
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
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for MPI_Bcast eigval and localCoef is " <<
          timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
       
        MPI_Barrier( domain_.comm );
        MPI_Barrier( domain_.rowComm );
        MPI_Barrier( domain_.colComm );
        
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for diagonalizing the DG matrix using ScaLAPACK in total is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

        // Communicate the eigenvalues
        Int mpirankScaSta = mpirankScaVec[0];
        MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
            mpirankScaVec[0], domain_.comm);
      }



      // Post processing

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


        hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );
       
      
        hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );


        
       
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

        hamDG.CalculateVtot( hamDG.Vtot() );
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
    // The following version is without intra-element parallelization
    if( solutionMethod_ == "pexsi" & 0 ){
      Real timePEXSISta, timePEXSIEnd;
      GetTime( timePEXSISta );

      Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
      Real muMinInertia, muMaxInertia;
      Real muPEXSI, numElectronPEXSI;
      Int numTotalInertiaIter, numTotalPEXSIIter;

      // Temporary matrices 
      DistSparseMatrix<Real>  HSparseMat;
      DistSparseMatrix<Real>  DMSparseMat;
      DistSparseMatrix<Real>  EDMSparseMat;
      DistSparseMatrix<Real>  FDMSparseMat;


      Int info;
      
      // Create an MPI communicator for saving the H matrix in a
      // subgroup of processors
      Int npPerPole_ = numProcRowPEXSI_ * numProcColPEXSI_;
      MPI_Comm HCSCComm;
      Int isProcHCSC = ( mpirank < npPerPole_ ) ? 1 : 0;
      
      MPI_Comm_split( MPI_COMM_WORLD, isProcHCSC, mpirank, &HCSCComm );
      
      // Convert the DG matrix into the distributed CSC format

			GetTime(timeSta);
			DistElemMatToDistSparseMat( 
					hamDG.HMat(),
					hamDG.NumBasisTotal(),
					HSparseMat,
					hamDG.ElemBasisIdx(),
					domain_.comm, 
          npPerPole_ );
			GetTime(timeEnd);

      // FIXME The following line is NECESSARY, and is because of the
      // unmature implementation of DistElemMatToDistSparseMat
      if( isProcHCSC ){
        HSparseMat.comm = HCSCComm;
        mpi::Allreduce( &HSparseMat.nnzLocal, 
            &HSparseMat.nnz, 1, MPI_SUM, HSparseMat.comm );
      }
#if ( _DEBUGlevel_ >= 0 )
			statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
				timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 1 )
      if( mpirank < npPerPole_ ){
        statusOFS << "H.size = " << HSparseMat.size << std::endl;
        statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
        statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
        statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
        statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
        statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
      }
#endif
 


#if ( _DEBUGlevel_ >= 1 )
      // Convert matrix back and forth to test the correctness of the
      // conversion routines.
      DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn>      HMat1;
      DistSparseMatrix<Real>      HSparseMat1;
      DistSparseMatToDistElemMat( 
          HSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          HMat1,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          npPerPole_ );

			DistElemMatToDistSparseMat( 
					HMat1,
					hamDG.NumBasisTotal(),
					HSparseMat1,
					hamDG.ElemBasisIdx(),
					domain_.comm, 
          npPerPole_ );

      // FIXME The following line is NECESSARY, and is because of the
      // unmature implementation of DistElemMatToDistSparseMat
      if( mpirank < npPerPole_ ){
        HSparseMat1.comm = HCSCComm;
        mpi::Allreduce( &HSparseMat1.nnzLocal, 
            &HSparseMat1.nnz, 1, MPI_SUM, HSparseMat1.comm );

        // Check the agreement between HSparseMat and HSparseMat1
        statusOFS << "H1.size = " << HSparseMat1.size << std::endl;
        statusOFS << "H1.nnz  = " << HSparseMat1.nnz << std::endl;
        statusOFS << "H1.nnzLocal  = " << HSparseMat1.nnzLocal << std::endl;
        statusOFS << "H1.colptrLocal.m() = " << HSparseMat1.colptrLocal.m() << std::endl;
        statusOFS << "H1.rowindLocal.m() = " << HSparseMat1.rowindLocal.m() << std::endl;
        statusOFS << "H1.nzvalLocal.m() = " << HSparseMat1.nzvalLocal.m() << std::endl;

        Real nzvalErr = 0.0;
        for( Int i = 0; i < HSparseMat.nnzLocal; i++ ){
          nzvalErr += pow( std::abs( 
                HSparseMat.nzvalLocal(i) - HSparseMat1.nzvalLocal(i) ), 2.0 );
        }
        nzvalErr = std::sqrt( nzvalErr );
        statusOFS << "||H.nzvalLocal - H1.nzvalLocal||_2 = " << nzvalErr << std::endl;
      }
#endif

      if( isProcHCSC ){
        CopyPattern( HSparseMat, DMSparseMat );
        CopyPattern( HSparseMat, EDMSparseMat );
        CopyPattern( HSparseMat, FDMSparseMat );
      }


      // Load the matrices into PEXSI.  
      // Only the processors with isProcHCSC == 1 need to carry the
      // nonzero values of HSparseMat
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
        pexsiOptions_.isSymbolicFactorize = (innerIter == 1) ? 1 : 0;
        statusOFS << std::endl 
          << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
          << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
          << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
      }


//      PPEXSIDFTDriver(
//          pexsiPlan_,
//          pexsiOptions_,
//          numElectronExact,
//          &muPEXSI,
//          &numElectronPEXSI,         
//          &muMinInertia,              
//          &muMaxInertia,             
//          &numTotalInertiaIter,
//          &numTotalPEXSIIter,
//          &info );

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

      if( isProcHCSC ){
        Real totalEnergyH, totalEnergyS, totalFreeEnergy;
        PPEXSIRetrieveRealSymmetricDFTMatrix(
            pexsiPlan_,
            DMSparseMat.nzvalLocal.Data(),
            EDMSparseMat.nzvalLocal.Data(),
            FDMSparseMat.nzvalLocal.Data(),
            &totalEnergyH,
            &totalEnergyS,
            &totalFreeEnergy,
            &info );

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

      // Convert the density matrix from DistSparseMatrix format to the
      // DistElemMat format
      DistSparseMatToDistElemMat(
          DMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          npPerPole_ );

      // Convert the energy density matrix from DistSparseMatrix
      // format to the DistElemMat format
      DistSparseMatToDistElemMat( 
          EDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distEDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          npPerPole_ );


      // Convert the free energy density matrix from DistSparseMatrix
      // format to the DistElemMat format
      DistSparseMatToDistElemMat( 
          FDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distFDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          npPerPole_ );

      for( Int k = 0; k < numElem_[2]; k++ )
        for( Int j = 0; j < numElem_[1]; j++ )
          for( Int i = 0; i < numElem_[0]; i++ ){
            Index3 key( i, j, k );
            if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
              DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
              MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), MPI_DOUBLE, 0, domain_.rowComm);
            }
          } 


      // Compute the Harris energy functional.  
      // NOTE: In computing the Harris energy, the density and the
      // potential must be the INPUT density and potential without ANY
      // update.
      CalculateHarrisEnergyDM( distFDMMat_ );

      // Evaluate the electron density

      GetTime( timeSta );
      hamDG.CalculateDensityDM( 
          hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
      MPI_Barrier( domain_.comm );
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

        hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );
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

      MPI_Comm_free( &HCSCComm );
      GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for PEXSI evaluation is " <<
        timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
    } //if( solutionMethod_ == "pexsi" )

    // The following version is with intra-element parallelization
    if( solutionMethod_ == "pexsi" & 1 ){
      Real timePEXSISta, timePEXSIEnd;
      GetTime( timePEXSISta );

      Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
      Real muMinInertia, muMaxInertia;
      Real muPEXSI, numElectronPEXSI;
      Int numTotalInertiaIter, numTotalPEXSIIter;
      Int npPerPole_ = numProcRowPEXSI_ * numProcColPEXSI_;

      std::vector<Int> mpirankElemVec( dmCol_ );
      std::vector<Int> mpirankSparseVec( npPerPole_ );

      // The processors in the first column are the source
      for( Int i = 0; i < dmCol_; i++ ){
        mpirankElemVec[i] = i * dmRow_;
      }
      // The first numPerPole_ processors are the target
      for( Int i = 0; i < npPerPole_; i++ ){
        mpirankSparseVec[i] = i;
      }

#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "mpirankElemVec = " << mpirankElemVec << std::endl;
        statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif


      // Temporary matrices 
      DistSparseMatrix<Real>  HSparseMat;
      DistSparseMatrix<Real>  DMSparseMat;
      DistSparseMatrix<Real>  EDMSparseMat;
      DistSparseMatrix<Real>  FDMSparseMat;

      Int info;
      
      // Create an MPI communicator for saving the H matrix in a
      // subgroup of processors
      MPI_Comm HCSCComm;
      Int isProcHCSC = ( mpirank < npPerPole_ ) ? 1 : 0;
      
      MPI_Comm_split( domain_.comm, isProcHCSC, mpirank, &HCSCComm );
      
      // Convert the DG matrix into the distributed CSC format
      // All processors participate in this step

			GetTime(timeSta);
			DistElemMatToDistSparseMat2( 
					hamDG.HMat(),
					hamDG.NumBasisTotal(),
					HSparseMat,
					hamDG.ElemBasisIdx(),
					domain_.comm, 
          domain_.colComm,
          HCSCComm,
          mpirankElemVec,
          mpirankSparseVec );
			GetTime(timeEnd);

      // FIXME The following line is NECESSARY, and is because of the
      // immature implementation of DistElemMatToDistSparseMat
      if( isProcHCSC ){
        HSparseMat.comm = HCSCComm;
        mpi::Allreduce( &HSparseMat.nnzLocal, 
            &HSparseMat.nnz, 1, MPI_SUM, HSparseMat.comm );
      }
#if ( _DEBUGlevel_ >= 0 )
			statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
				timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
      if( mpirank < npPerPole_ ){
        statusOFS << "H.size = " << HSparseMat.size << std::endl;
        statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
        statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
        statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
        statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
        statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
      }
#endif
 
      if( isProcHCSC ){
        CopyPattern( HSparseMat, DMSparseMat );
        CopyPattern( HSparseMat, EDMSparseMat );
        CopyPattern( HSparseMat, FDMSparseMat );
      }

      if( mpirank < numProcTotalPEXSI_ ){
        // Load the matrices into PEXSI.  
        // Only the processors with isProcHCSC == 1 need to carry the
        // nonzero values of HSparseMat
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
          pexsiOptions_.isSymbolicFactorize = (innerIter == 1) ? 1 : 0;
          statusOFS << std::endl 
            << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
            << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
            << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
        }


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

        if( isProcHCSC ){
          Real totalEnergyH, totalEnergyS, totalFreeEnergy;
          PPEXSIRetrieveRealSymmetricDFTMatrix(
              pexsiPlan_,
              DMSparseMat.nzvalLocal.Data(),
              EDMSparseMat.nzvalLocal.Data(),
              FDMSparseMat.nzvalLocal.Data(),
              &totalEnergyH,
              &totalEnergyS,
              &totalFreeEnergy,
              &info );

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

      // Convert the density matrix from DistSparseMatrix format to the
      // DistElemMat format
      DistSparseMatToDistElemMat2(
          DMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          domain_.colComm,
          HCSCComm,
          mpirankElemVec,
          mpirankSparseVec );


      // Convert the energy density matrix from DistSparseMatrix
      // format to the DistElemMat format
      DistSparseMatToDistElemMat2(
          EDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distEDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          domain_.colComm,
          HCSCComm,
          mpirankElemVec,
          mpirankSparseVec );


      // Convert the free energy density matrix from DistSparseMatrix
      // format to the DistElemMat format
      DistSparseMatToDistElemMat2(
          FDMSparseMat,
          hamDG.NumBasisTotal(),
          hamDG.HMat().Prtn(),
          distFDMMat_,
					hamDG.ElemBasisIdx(),
          domain_.comm,
          domain_.colComm,
          HCSCComm,
          mpirankElemVec,
          mpirankSparseVec );

      // Broadcast the distElemMat matrices
      // FIXME this is not a memory efficient implementation
      {
        Int sstrSize;
        std::vector<char> sstr;
        if( mpirank % dmRow_ == 0 ){
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
        
        statusOFS << "Before communication. " << std::endl;

        MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
        sstr.resize( sstrSize );
        MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

        statusOFS << "Finish communication. sstrSize =" <<  sstrSize << std::endl;

        if( mpirank % dmRow_ != 0 ){
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




      // Compute the Harris energy functional.  
      // NOTE: In computing the Harris energy, the density and the
      // potential must be the INPUT density and potential without ANY
      // update.
      CalculateHarrisEnergyDM( distFDMMat_ );

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
      {
        // Update the Hartree energy and the exchange correlation energy and
        // potential for computing the KS energy and the second order
        // energy.
        // NOTE Vtot should not be updated until finishing the computation
        // of the energies.

        hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );
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

      MPI_Comm_free( &HCSCComm );
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
			Print(statusOFS, "norm(MixDif)          = ", normMixDif );
			Print(statusOFS, "norm(MixOld)          = ", normMixOld );
			Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
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

#if ( _DEBUGlevel_ >= 0 )
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
			hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc() );

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
	//
	// 2. The local potential on the LGL grid is done by using Fourier
	// interpolation from the extended element to the element. Gibbs
	// phenomena MAY be there but at least this is better than
	// Lagrange interpolation on a uniform grid.
	//  
	//
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

          // Update the potential in the element on LGL grid
					DblNumVec&  vtotLGLElem = hamDG.VtotLGL().LocalMap()[key];
					Index3 numLGLGrid       = hamDG.NumLGLGridElem();

          InterpPeriodicUniformFineToLGL( 
							numGridExtElem,
							numLGLGrid,
							vtotExtElem.Data(),
							vtotLGLElem.Data() );

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
	Print(statusOFS, "Eself             = ",  Eself_, "[au]");
	Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
	Print(statusOFS, "Fermi             = ",  fermi_, "[au]");

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method SCFDG::PrintState  ----- 



} // namespace dgdft
