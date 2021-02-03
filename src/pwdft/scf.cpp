//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

/// @file scf.cpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing
#include  "scf.hpp"
#include  <blas.hh>
#include  <lapack.hh>
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"
#include  "utility.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"
#ifdef DEVICE
#include  "device_blas.hpp"
#include "device_utils.h"
#ifdef USE_MAGMA
#include  "magma.hpp"
#else
#include "device_solver.hpp"
#endif
#endif

namespace  scales{

using namespace scales::DensityComponent;
using namespace scales::esdf;

using namespace scales::scalapack;

SCF::SCF    (  )
{
  eigSolPtr_ = NULL;
  ptablePtr_ = NULL;

}         // -----  end of method SCF::SCF  ----- 

SCF::~SCF    (  )
{

}         // -----  end of method SCF::~SCF  ----- 


void
SCF::Setup    ( EigenSolver& eigSol, PeriodTable& ptable )
{
  int mpirank;  MPI_Comm_rank(esdfParam.domain->comm, &mpirank);
  int mpisize;  MPI_Comm_size(esdfParam.domain->comm, &mpisize);
  Real timeSta, timeEnd;

  // esdf parameters
  {
    mixMaxDim_     = esdfParam.mixMaxDim;
    mixType_       = esdfParam.mixType;
    mixStepLength_ = esdfParam.mixStepLength;
    // Note: for PW SCF there is no inner loop. Use the parameter value
    // for the outer SCF loop only.
    eigTolerance_  = esdfParam.eigTolerance;
    eigMinTolerance_  = esdfParam.eigMinTolerance;
    eigMaxIter_    = esdfParam.eigMaxIter;
    scfTolerance_  = esdfParam.scfOuterTolerance;
    scfMaxIter_    = esdfParam.scfOuterMaxIter;
    scfPhiMaxIter_ = esdfParam.scfPhiMaxIter;
    scfPhiTolerance_ = esdfParam.scfPhiTolerance;
    isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
    Tbeta_         = esdfParam.Tbeta;
    BlockSizeScaLAPACK_      = esdfParam.BlockSizeScaLAPACK;

    //        numGridWavefunctionElem_ = esdfParam.numGridWavefunctionElem;
    //        numGridDensityElem_      = esdfParam.numGridDensityElem;  


    // Chebyshev Filtering related parameters
    if(esdfParam.PWSolver == "CheFSI")
      Diag_SCF_PWDFT_by_Cheby_ = 1;
    else
      Diag_SCF_PWDFT_by_Cheby_ = 0;

    First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
    First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
    General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
    PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;
    PWDFT_Cheby_apply_wfn_ecut_filt_ =  esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt;
    Cheby_iondynamics_schedule_flag_ = 0;


  }

  // other SCF parameters
  {
    eigSolPtr_ = &eigSol;
    ptablePtr_ = &ptable;

    //        Int ntot = eigSolPtr_->Psi().NumGridTotal();
    Int ntot = esdfParam.domain->NumGridTotal();
    Int ntotFine = esdfParam.domain->NumGridTotalFine();

    vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
    dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
    dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

    restartDensityFileName_ = "DEN";
    restartPotentialFileName_ = "POT";
    restartWfnFileName_     = "WFN";
  }

  // Density
  {
    Hamiltonian& ham = eigSolPtr_->Ham();
    DblNumMat&  density = ham.Density();

    if( esdfParam.isRestartDensity ) {
      std::istringstream rhoStream;      
      SharedRead( restartDensityFileName_, rhoStream);
      // TODO Error checking
      // Read the grid
      std::vector<DblNumVec> gridpos(DIM);
      for( Int d = 0; d < DIM; d++ ){
        deserialize( gridpos[d], rhoStream, NO_MASK );
      }
      DblNumVec densityVec;
      // only for restricted spin case
      deserialize( densityVec, rhoStream, NO_MASK );    
      blas::copy( densityVec.m(), densityVec.Data(), 1, 
          density.VecData(RHO), 1 );
      statusOFS << "Density restarted from file " 
        << restartDensityFileName_ << std::endl;

    } // else using the zero initial guess
    else {
      if( esdfParam.isUseAtomDensity ){
#if ( _DEBUGlevel_ >= 0 )
        statusOFS 
          << "Use superposition of atomic density as initial "
          << "guess for electron density." << std::endl;
#endif
        
        GetTime( timeSta );
        ham.CalculateAtomDensity( *ptablePtr_ );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for calculating the atomic density = " 
          << timeEnd - timeSta << " [s]" << std::endl;
#endif

        // Use the superposition of atomic density as the initial guess for density
        const Domain& dm = *esdfParam.domain;
        Int ntotFine = dm.NumGridTotalFine();

        SetValue( density, 0.0 );
        blas::copy( ntotFine, ham.AtomDensity().Data(), 1, 
            density.VecData(0), 1 );

      }
      else{
        // Start from pseudocharge, usually this is not a very good idea
        // make sure the pseudocharge is initialized
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Generating initial density through linear combination of pseudocharges." 
          << std::endl;
#endif
        DblNumVec&  pseudoCharge = ham.PseudoCharge();
        const Domain& dm = *esdfParam.domain;

        SetValue( density, 0.0 );

        Int ntotFine = dm.NumGridTotalFine();

        Real sum0 = 0.0, sum1 = 0.0;
        Real EPS = 1e-6;

        // make sure that the electron density is positive
        for (Int i=0; i<ntotFine; i++){
          density(i, RHO) = ( pseudoCharge(i) > EPS ) ? pseudoCharge(i) : EPS;
          //                density(i, RHO) = pseudoCharge(i);
          sum0 += density(i, RHO);
          sum1 += pseudoCharge(i);
        }

        Print( statusOFS, "Initial density. Sum of density      = ", 
            sum0 * dm.Volume() / dm.NumGridTotalFine() );

        // Rescale the density
        for (int i=0; i <ntotFine; i++){
          density(i, RHO) *= sum1 / sum0;
        } 

        Print( statusOFS, "Rescaled density. Sum of density      = ", 
            sum1 * dm.Volume() / dm.NumGridTotalFine() );
      }
    }
  }

  if( ! esdfParam.isRestartWfn ) {
    // Randomized input from outside
    // Setup the occupation rate by aufbau principle (needed for hybrid functional calculation)
    DblNumVec& occ = eigSolPtr_->Ham().OccupationRate();
    Int npsi = eigSolPtr_->Psi().NumStateTotal();
    occ.Resize( npsi );
    SetValue( occ, 0.0 );
    for( Int k = 0; k < npsi; k++ ){
      occ[k] = 1.0;
    }
  }
  else {
    std::istringstream wfnStream;
    SeparateRead( restartWfnFileName_, wfnStream, mpirank );
    deserialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    deserialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
    statusOFS << "Wavefunction restarted from file " 
      << restartWfnFileName_ << std::endl;
  }

  // FIXME external force when needed

  return ;
}         // -----  end of method SCF::Setup  ----- 

void
SCF::Update    ( )
{
  {
    Int ntotFine  = eigSolPtr_->FFT().domain->NumGridTotalFine();

    vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
    dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
    dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );
  }


  return ;
}         // -----  end of method SCF::Update  ----- 

void
SCF::Execute    ( )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain->comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain->comm, &mpisize);

  Real timeSta, timeEnd;
  Real timeIterStart(0), timeIterEnd(0);

  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();

  // Compute the exchange-correlation potential and energy
  if( ham.XCRequireGradDensity() ){
    ham.CalculateGradDensity( );
  }

  // Compute the total potential
  ham.CalculateXC( Exc_ ); 
  ham.CalculateHartree( );
  ham.CalculateVtot( ham.Vtot() );

  // FIXME: LL 1/4/2021 the right place?
#ifdef DEVICE
  device_init_vtot();
  DEVICE_BLAS::Init();
#ifdef USE_MAGMA
  MAGMA::Init();
#else
  device_solver::Init();
#endif
#endif

  if( ham.XCRequireIterateDensity() ){

    if( ham.IsHybrid() and ham.IsEXXActive() == false ){
      std::ostringstream msg;
      msg << "For hybrid functionals without initial guess, start with a PBE calculation.";
      PrintBlock( statusOFS, msg.str() );
      // Modify the XC functional to improve SCF convergence
      ham.SetupXC("PBE");
      IterateDensity();
      ham.SetupXC(esdfParam.XCType);
      ham.SetEXXActive(true);
    }
    else{
      std::ostringstream msg;
      msg << "Starting regular SCF iteration.";
      PrintBlock( statusOFS, msg.str() );
      IterateDensity();
    }
  }

  if( ham.XCRequireIterateWavefun() ){
    IterateWavefun();
  }

  // FIXME: LL 1/4/2021 Should the Destroy() functions be activated somewhere?
/*
#ifdef DEVICE
    DEVICE_BLAS::Destroy();
#ifdef USE_MAGMA
    MAGMA::Destroy();
#else
    device_solver::Destroy();
#endif
    device_clean_vtot();
#endif
*/
  // Calculate the Force. This includes contribution from ionic repulsion, VDW etc
  ham.CalculateForce( psi );

  // Output the information after SCF
  {
    // Energy
    Real HOMO, LUMO;
    HOMO = eigSolPtr_->EigVal()(ham.NumOccupiedState()-1);
    if( ham.NumExtraState() > 0 )
      LUMO = eigSolPtr_->EigVal()(ham.NumOccupiedState());

    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    statusOFS 
      << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
      << "       Etot  = Ekin + Ecor" << std::endl
      << "       Efree = Etot    + Entropy" << std::endl << std::endl;
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
    Print(statusOFS, "! EVdw            = ",  EVdw_, "[au]"); 
    Print(statusOFS, "! Eext            = ",  Eext_, "[au]");
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
    Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[ev]");
    if( ham.NumExtraState() > 0 ){
      Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
    }
  }

  {
    // Print out the force
    PrintBlock( statusOFS, "Atomic Force" );

    Point3 forceCM(0.0, 0.0, 0.0);
    std::vector<Atom>& atomList = ham.AtomList();
    Int numAtom = atomList.size();

    for( Int a = 0; a < numAtom; a++ ){
      Print( statusOFS, "atom", a, "force", atomList[a].force );
      forceCM += atomList[a].force;
    }
    statusOFS << std::endl;
    Print( statusOFS, "force for centroid  : ", forceCM );
    Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
    statusOFS << std::endl;
  }


  // Output the structure information
  // FIXME: Change to using hdf5 file format
  if(0){
    if( mpirank == 0 ){
      std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl 
        << "Output the structure information" 
        << std::endl;
#endif
      // Domain
      const Domain& dm =  *eigSolPtr_->FFT().domain;
      serialize( dm.length, structStream, NO_MASK );
      serialize( dm.numGrid, structStream, NO_MASK );
      serialize( dm.numGridFine, structStream, NO_MASK );
      serialize( dm.posStart, structStream, NO_MASK );

      // Atomic information
      serialize( ham.AtomList(), structStream, NO_MASK );
      std::string structFileName = "STRUCTURE";

      std::ofstream fout(structFileName.c_str());
      if( !fout.good() ){
        std::ostringstream msg;
        msg 
          << "File " << structFileName.c_str() << " cannot be open." 
          << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      fout << structStream.str();
      fout.close();
    }
  }


  // Output restarting information
  if( esdfParam.isOutputDensity ){
    if( mpirank == 0 ){
      std::ofstream rhoStream(restartDensityFileName_.c_str());
      if( !rhoStream.good() ){
        ErrorHandling( "Density file cannot be opened." );
      }

      const Domain& dm =  *eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], rhoStream, NO_MASK );
      }

      // Only work for the restricted spin case
      DblNumMat& densityMat = eigSolPtr_->Ham().Density();
      DblNumVec densityVec(densityMat.m(), false, densityMat.Data());
      serialize( densityVec, rhoStream, NO_MASK );
      rhoStream.close();
    }
  }    

  // Output the total potential
  if( esdfParam.isOutputPotential ){
    if( mpirank == 0 ){
      std::ofstream vtotStream(restartPotentialFileName_.c_str());
      if( !vtotStream.good() ){
        ErrorHandling( "Potential file cannot be opened." );
      }

      const Domain& dm =  *eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], vtotStream, NO_MASK );
      }

      serialize( eigSolPtr_->Ham().Vtot(), vtotStream, NO_MASK );
      vtotStream.close();
    }
  }

  if( esdfParam.isOutputWfn ){
    std::ostringstream wfnStream;
    serialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    serialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
    SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
  }   


  return ;
}         // -----  end of method SCF::Execute  ----- 

void
SCF::IterateDensity    ( )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain->comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain->comm, &mpisize);

  Real timeSta, timeEnd;
  Real timeIterStart(0), timeIterEnd(0);

  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();
 

  // Perform non-hybrid functional calculation first
  bool isSCFConverged = false;

  for (Int iter=1; iter <= scfMaxIter_; iter++) {
    if ( isSCFConverged ) break;
    // *********************************************************************
    // Performing each iteartion
    // *********************************************************************
    {
      std::ostringstream msg;
      msg << "SCF iteration # " << iter;
      PrintBlock( statusOFS, msg.str() );
    }

    GetTime( timeIterStart );

    // Solve eigenvalue problem
    // Update density, gradDensity, potential (stored in vtotNew_)
    InnerSolve( iter );

    Real normVtotDif = 0.0, normVtotOld = 0.0;
    DblNumVec& vtotOld_ = ham.Vtot();
    Int ntot = vtotOld_.m();
    for( Int i = 0; i < ntot; i++ ){
      normVtotDif += pow( vtotOld_(i) - vtotNew_(i), 2.0 );
      normVtotOld += pow( vtotOld_(i), 2.0 );
    }
    normVtotDif = sqrt( normVtotDif );
    normVtotOld = sqrt( normVtotOld );
    scfNorm_    = normVtotDif / normVtotOld;

    GetTime( timeSta );
    CalculateEnergy();
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing energy in PWDFT is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    PrintState( iter );

    Int numAtom = ham.AtomList().size();
    efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

    Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
    Print(statusOFS, "Efree diff per atom   = ", efreeDifPerAtom_ ); 

    if( scfNorm_ < scfTolerance_ ){
      statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
      isSCFConverged = true;
    }

    // Potential mixing
    GetTime( timeSta );
    if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
      AndersonMix(
          iter,
          mixStepLength_,
          mixType_,
          ham.Vtot(),
          vtotOld_,
          vtotNew_,
          dfMat_,
          dvMat_);
    }
    else{
      ErrorHandling("Invalid mixing type.");
    }
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing potential mixing in PWDFT is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

    GetTime( timeIterEnd );

    statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
      << " [s]" << std::endl;

  } // for (iter)

  return ;
}         // -----  end of method SCF::IterateDensity  ----- 

void
SCF::IterateWavefun    ( )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain->comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain->comm, &mpisize);

  Real timeSta, timeEnd;
  Real timeIterStart(0), timeIterEnd(0);

  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();


  // Fock energies
  Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

  bool isPhiIterConverged = false;

  // FIXME the decision whether to fix the column choices for density
  // fitting should be controled by a parameter outside
  bool isFixColumnDF = false;
  Real timePhiIterStart(0), timePhiIterEnd(0);
  Real dExx;

  // NOTE: The different mixing mode of hybrid functional calculations
  // are not compatible with each other. So each requires its own code

  // Evaluate the Fock energy
  // Update Phi <- Psi
  GetTime( timeSta );
  ham.SetPhiEXX( psi ); 

  // Construct the ACE operator
  if( esdfParam.isHybridACE ){
    if( esdfParam.isHybridDF ){
#ifdef DEVICE
      ham.CalculateVexxACEDFGPU( psi, isFixColumnDF );
#else
      ham.CalculateVexxACEDF( psi, isFixColumnDF );
#endif
      // Fix the column after the first iteraiton
      isFixColumnDF = true;
    }
    else{
#ifdef DEVICE
      ham.CalculateVexxACEGPU ( psi );
#else
      ham.CalculateVexxACE ( psi );
#endif
    }
  }

  GetTime( timeEnd );
  statusOFS << "Time for updating Phi related variable is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;

  GetTime( timeSta );
  fock2 = ham.CalculateEXXEnergy( psi ); 
  GetTime( timeEnd );
  statusOFS << "Time for computing the EXX energy is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;

  Efock_ = fock2;
  fock1  = fock2;

  if( esdfParam.hybridMixType == "nested" ){
    GetTime( timeSta );

    for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

      std::ostringstream msg;
      msg << "Phi iteration # " << phiIter;
      PrintBlock( statusOFS, msg.str() );

      // Nested SCF iteration
      GetTime( timePhiIterStart );
      IterateDensity();
      GetTime( timePhiIterEnd );

      statusOFS << "Total wall clock time for this Phi iteration = " << 
        timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

      // Update Phi <- Psi
      GetTime( timeSta );
      ham.SetPhiEXX( psi ); 

      // Update the ACE if needed
      if( esdfParam.isHybridACE ){
        if( esdfParam.isHybridDF ){
          ham.CalculateVexxACEDF( psi, isFixColumnDF );
          // Fix the column after the first iteraiton
          isFixColumnDF = true;
        }
        else{
#ifdef DEVICE
          statusOFS << " ham.CalculateVexxACEGPU1 ..... " << std::endl << std::endl;
          ham.CalculateVexxACEGPU1 ( psi );
#else
          ham.CalculateVexxACE ( psi );
#endif
        }
      }

      GetTime( timeEnd );
      statusOFS << "Time for updating Phi related variable is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      GetTime( timeSta );
      fock2 = ham.CalculateEXXEnergy( psi ); 
      GetTime( timeEnd );
      statusOFS << "Time for computing the EXX energy is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
      dExx = std::abs(fock2 - fock1) / std::abs(fock2);
      fock1 = fock2;
      Efock_ = fock2;

      Etot_        -= Efock_;
      Efree_       -= Efock_;
      EfreeHarris_ -= Efock_;

      statusOFS << std::endl;
      Print(statusOFS, "Fock energy             = ",  Efock_, "[au]");
      Print(statusOFS, "Etot(with fock)         = ",  Etot_, "[au]");
      Print(statusOFS, "Efree(with fock)        = ",  Efree_, "[au]");
      Print(statusOFS, "EfreeHarris(with fock)  = ",  EfreeHarris_, "[au]");
      Print(statusOFS, "dExx              = ",  dExx, "[au]");
      if( dExx < scfPhiTolerance_ ){
        statusOFS << "SCF for hybrid functional is converged in " 
          << phiIter << " steps !" << std::endl;
        isPhiIterConverged = true;
      }
      if ( isPhiIterConverged ) break;
    } // for(phiIter)

    GetTime( timeEnd );
    statusOFS << "Time for using nested method is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

  } // hybridMixType == "nested"


#ifdef DEVICE
  // GPU version of pc-diis. 
  // note, ACE nested can not work on GPU yet
  if( esdfParam.hybridMixType == "pcdiis" ){
    GetTime( timeSta );

    // This requires a good initial guess of wavefunctions, 
    // from one of the following
    // 1) a regular SCF calculation
    // 2) restarting wavefunction with Hybrid_Active_Init = true
    

    Int ntot      = fft.domain->NumGridTotal();
    Int ntotFine  = fft.domain->NumGridTotalFine();
    Int numStateTotal = psi.NumStateTotal();
    Int numStateLocal = psi.NumState();
    Int numOccTotal = ham.NumOccupiedState();

    MPI_Comm mpi_comm = eigSolPtr_->FFT().domain->comm;

    Int I_ONE = 1, I_ZERO = 0;
    double D_ONE = 1.0;
    double D_ZERO = 0.0;
    double D_MinusONE = -1.0;

    // FIXME: LL 1/4/2021 Cleanup the pcdiis method with ScaLAPACKMatrix structures
    Real timeSta, timeEnd, timeSta1, timeEnd1;

    Int contxt0D, contxt1DCol, contxt1DRow,  contxt2D;
    Int nprow0D, npcol0D, myrow0D, mycol0D, info0D;
    Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
    Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
    Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

    Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
    Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
    Int ncolsNgNo1DCol, nrowsNgNo1DCol, lldNgNo1DCol; 
    Int ncolsNgNo1DRow, nrowsNgNo1DRow, lldNgNo1DRow; 

    Int desc_NgNe1DCol[9];
    Int desc_NgNe1DRow[9];
    Int desc_NgNo1DCol[9];
    Int desc_NgNo1DRow[9];

    Int Ne = numStateTotal; 
    Int No = numOccTotal; 
    Int Ng = ntot;

    // 1D col MPI
    nprow1DCol = 1;
    npcol1DCol = mpisize;

    Cblacs_get(0, 0, &contxt1DCol);
    Cblacs_gridinit(&contxt1DCol, "C", nprow1DCol, npcol1DCol);
    Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);

    // 1D row MPI
    nprow1DRow = mpisize;
    npcol1DRow = 1;

    Cblacs_get(0, 0, &contxt1DRow);
    Cblacs_gridinit(&contxt1DRow, "C", nprow1DRow, npcol1DRow);
    Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);


    //desc_NgNe1DCol
    if(contxt1DCol >= 0){
      nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
        &I_ZERO, &contxt1DCol, &lldNgNe1DCol, &info1DCol);

    //desc_NgNe1DRow
    if(contxt1DRow >= 0){
      nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK_, &Ne, &I_ZERO, 
        &I_ZERO, &contxt1DRow, &lldNgNe1DRow, &info1DRow);

    //desc_NgNo1DCol
    if(contxt1DCol >= 0){
      nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO, 
        &I_ZERO, &contxt1DCol, &lldNgNo1DCol, &info1DCol);

    //desc_NgNo1DRow
    if(contxt1DRow >= 0){
      nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK_, &No, &I_ZERO, 
        &I_ZERO, &contxt1DRow, &lldNgNo1DRow, &info1DRow);

    if(numStateLocal !=  ncolsNgNe1DCol){
      statusOFS << "numStateLocal = " << numStateLocal << " ncolsNgNe1DCol = " << ncolsNgNe1DCol << std::endl;
      ErrorHandling("The size of numState is not right!");
    }

    if(nrowsNgNe1DRow !=  nrowsNgNo1DRow){
      statusOFS << "nrowsNgNe1DRow = " << nrowsNgNe1DRow << " ncolsNgNo1DRow = " << ncolsNgNo1DRow << std::endl;
      ErrorHandling("The size of nrowsNgNe1DRow and ncolsNgNo1DRow is not right!");
    }


    Int numOccLocal = ncolsNgNo1DCol;
    Int ntotLocal = nrowsNgNe1DRow;

    DblNumMat psiPcCol(ntot, numStateLocal);
    DblNumMat psiPcRow(ntotLocal, numStateTotal);
    DblNumMat HpsiCol(ntot, numStateLocal);
    DblNumMat HpsiRow(ntotLocal, numStateTotal);

    dfMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
    dvMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

    DblNumMat PcCol(ntot, numOccLocal);
    DblNumMat PcRow(ntotLocal, numOccTotal);
    DblNumMat ResCol(ntot, numOccLocal);
    DblNumMat ResRow(ntotLocal, numOccTotal);

    DblNumMat psiMuT(numOccTotal, numOccTotal);
    DblNumMat psiMuTLocal(numOccLocal, numOccTotal);
    DblNumMat HpsiMuT(numOccTotal, numOccTotal);
    DblNumMat HpsiMuTLocal(numOccLocal, numOccTotal);

    DblNumMat psiCol( ntot, numStateLocal );
    SetValue( psiCol, 0.0 );
    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
    //AlltoallForward (psiCol, psiRow, mpi_comm);
    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

    DblNumMat psiTemp(ntotLocal, numOccTotal);
    SetValue( psiTemp, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, ntotLocal, numOccTotal, psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal );
    Real one = 1.0;
    Real minus_one = -1.0;
    Real zero = 0.0;


#ifdef DEVICE
    //device_init_vtot();
#endif
    // Phi loop
    for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){
      {
        deviceDblNumMat cu_psiMuT(numOccTotal, numOccTotal);
        deviceDblNumMat cu_HpsiMuT(numOccTotal, numOccTotal);
        deviceDblNumMat cu_psiRow( ntotLocal, numStateTotal );
        deviceDblNumMat cu_psiTemp(ntotLocal, numOccTotal);
        deviceDblNumMat cu_PcRow(ntotLocal, numOccTotal);
        deviceDblNumMat cu_HpsiCol(ntot, numStateLocal);
        deviceDblNumMat cu_psi(ntot, numStateLocal);
        deviceDblNumMat cu_HpsiRow(ntotLocal, numStateTotal);
        deviceDblNumMat cu_ResRow(ntotLocal, numOccTotal);
        deviceDblNumMat cu_psiPcRow(ntotLocal, numStateTotal);
        cu_psiTemp.CopyFrom( psiTemp);

        char right  = 'R';
        char up     = 'U';
        char nondiag   = 'N';
        char cu_transT = 'T';
        char cu_transN = 'N';
        char cu_transC = 'C';

        GetTime( timePhiIterStart );

        SetValue( psiCol, 0.0 );
        lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        SetValue( psiRow, 0.0 );
        //AlltoallForward (psiCol, psiRow, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

        if(1){

          DblNumMat psiMuTTemp(numOccTotal, numOccTotal);
          SetValue( psiMuTTemp, 0.0 );

          deviceDblNumMat cu_psiMuTTemp(numOccTotal, numOccTotal);
          cu_psiRow.CopyFrom(psiRow);
          DEVICE_BLAS::Gemm( cu_transT, cu_transN, numOccTotal, numOccTotal, ntotLocal, 
              &one, cu_psiRow.Data(), ntotLocal, cu_psiTemp.Data(), ntotLocal, 
              &zero, cu_psiMuTTemp.Data(), numOccTotal );
          cu_psiMuTTemp.CopyTo( psiMuTTemp );

          /*
             blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
             psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
             0.0, psiMuTTemp.Data(), numOccTotal );
           */
          SetValue( psiMuT, 0.0 );
          MPI_Allreduce( psiMuTTemp.Data(), psiMuT.Data(), 
              numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

        }//if

#if ( _DEBUGlevel_ >= 1 )
        {
          // Monitor the singular values as a measure as the quality of
          // the selected columns
          DblNumMat tmp = psiMuT;
          DblNumVec s(numOccTotal);
          lapack::SingularValues( numOccTotal, numOccTotal, tmp.Data(), numOccTotal, s.Data() );
          statusOFS << "Spsi = " << s << std::endl;
        }
#endif
        cu_psiMuT.CopyFrom( psiMuT);
        DEVICE_BLAS::Gemm( cu_transN, cu_transN, ntotLocal, numOccTotal, numOccTotal, &one, 
            cu_psiRow.Data(), ntotLocal, cu_psiMuT.Data(), numOccTotal, 
            &zero, cu_PcRow.Data(), ntotLocal );
        cu_PcRow.CopyTo( PcRow );

        /*
           blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
           psiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
           0.0, PcRow.Data(), ntotLocal );
         */
        SetValue( PcCol, 0.0 );
        //AlltoallBackward (PcRow, PcCol, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
            PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );

        std::ostringstream msg;
        msg << "Phi iteration # " << phiIter;
        PrintBlock( statusOFS, msg.str() );

        // Compute the residual
        {
          // Compute Hpsi for all psi 
          Int ncom = psi.NumComponent();
          Int noccTotal = psi.NumStateTotal();
          Int noccLocal = psi.NumState();

          device_memcpy_HOST2DEVICE(cu_psi.Data(), psi.Wavefun().Data(), ntot*numStateLocal*sizeof(Real) );
          deviceNumTns<Real> tnsTemp(ntot, 1, numStateLocal, false, cu_HpsiCol.Data());
          // there are two sets of grid for Row parallelization. 
          // the old fashioned split and the Scalapack split.
          // note that they turns out to be different ones in that: 
          // Scalapack divide the Psi by blocks, and old way is continuous.  
          // this is error prone.
          Spinor spnTemp(fft.domain, ncom, noccTotal, noccLocal, false, cu_psi.Data(), true);
          ham.MultSpinor_old( spnTemp, tnsTemp );
          cu_HpsiCol.CopyTo(HpsiCol);

          // remember to reset the vtot
          device_reset_vtot_flag();
          //ham.MultSpinor( psi, tnsTemp );

          //SetValue( HpsiRow, 0.0 );
          //AlltoallForward (HpsiCol, HpsiRow, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &Ne, HpsiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
              HpsiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

          cu_HpsiRow.CopyFrom(HpsiRow);

          // perform the ACE operator here
          //ham.ACEOperator( cu_psiRow, cu_HpsiRow );
          //cu_HpsiRow.CopyTo(HpsiRow);

          if(1){

            DblNumMat HpsiMuTTemp(numOccTotal,numOccTotal);
            deviceDblNumMat cu_HpsiMuTTemp(numOccTotal,numOccTotal);
            //SetValue( HpsiMuTTemp, 0.0 );

            DEVICE_BLAS::Gemm( cu_transT, cu_transN, numOccTotal, numOccTotal, ntotLocal, &one, 
                cu_HpsiRow.Data(), ntotLocal, cu_psiTemp.Data(), ntotLocal, 
                &zero, cu_HpsiMuTTemp.Data(), numOccTotal );
            cu_HpsiMuTTemp.CopyTo(HpsiMuTTemp);
            /*
               blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
               HpsiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
               0.0, HpsiMuTTemp.Data(), numOccTotal );
             */
            //SetValue( HpsiMuT, 0.0 );

            MPI_Allreduce( HpsiMuTTemp.Data(), HpsiMuT.Data(), 
                numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

          }//if

          DEVICE_BLAS::Gemm( cu_transN, cu_transN, ntotLocal, numOccTotal, numOccTotal, &one, 
              cu_HpsiRow.Data(), ntotLocal, cu_psiMuT.Data(), numOccTotal, 
              &zero, cu_ResRow.Data(), ntotLocal );

          cu_HpsiMuT.CopyFrom( HpsiMuT );
          DEVICE_BLAS::Gemm( cu_transN, cu_transN, ntotLocal, numOccTotal, numOccTotal, &minus_one, 
              cu_psiRow.Data(), ntotLocal, cu_HpsiMuT.Data(), numOccTotal, 
              &one, cu_ResRow.Data(), ntotLocal );
          cu_ResRow.CopyTo(ResRow);

          /*
             blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
             HpsiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
             0.0, ResRow.Data(), ntotLocal );
             blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, -1.0, 
             psiRow.Data(), ntotLocal, HpsiMuT.Data(), numOccTotal, 
             1.0, ResRow.Data(), ntotLocal );
             SetValue( ResCol, 0.0 );
           */
          //AlltoallBackward (ResRow, ResCol, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &No, ResRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
              ResCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        }

        GetTime( timeSta );
        // Anderson mixing. Use the same mixMaxDim_ for Phi mixing
        {
          // Optimal input potential in Anderon mixing.
          DblNumVec vOpt( ntot * numOccLocal ); 

          // Number of iterations used, iter should start from 1
          Int iterused = std::min( phiIter-1, mixMaxDim_ ); 
          // The current position of dfMat, dvMat
          Int ipos = phiIter - 1 - ((phiIter-2)/ mixMaxDim_ ) * mixMaxDim_;
          // The next position of dfMat, dvMat
          Int inext = phiIter - ((phiIter-1)/ mixMaxDim_) * mixMaxDim_;

          blas::copy( ntot * numOccLocal, PcCol.Data(), 1, vOpt.Data(), 1 );

          if( phiIter > 1 ){
            // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
            // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
            blas::scal( ntot * numOccLocal, -1.0, dfMat_.VecData(ipos-1), 1 );
            blas::axpy( ntot * numOccLocal, 1.0, ResCol.Data(), 1, dfMat_.VecData(ipos-1), 1 );
            blas::scal( ntot * numOccLocal, -1.0, dvMat_.VecData(ipos-1), 1 );
            blas::axpy( ntot * numOccLocal, 1.0, PcCol.Data(), 1, dvMat_.VecData(ipos-1), 1 );

            // Calculating pseudoinverse
            DblNumMat dfMatTemp(ntot * numOccLocal, mixMaxDim_);
            DblNumVec gammas(ntot * numOccLocal), S(iterused);

            SetValue( dfMatTemp, 0.0 );
            SetValue( gammas, 0.0 );

            int64_t rank;
            // FIXME Magic number
            Real rcond = 1e-9;

            // gammas    = res;
            blas::copy( ntot * numOccLocal, ResCol.Data(), 1, gammas.Data(), 1 );
            lapack::lacpy( lapack::MatrixType::General, ntot * numOccLocal, mixMaxDim_, dfMat_.Data(), ntot * numOccLocal, 
                dfMatTemp.Data(), ntot * numOccLocal );

            // May need different strategy in a parallel setup
            if(0){  

              lapack::gelss( ntot * numOccLocal, iterused, 1, 
                  dfMatTemp.Data(), ntot * numOccLocal,
                  gammas.Data(), ntot * numOccLocal,
                  S.Data(), rcond, &rank );

              blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
                  ntot * numOccLocal, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

            }

            if(1){

              DblNumMat XTX(iterused, iterused);
              DblNumMat XTXTemp(iterused, iterused);

              SetValue( XTXTemp, 0.0 );
              blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, iterused, iterused, ntot * numOccLocal, 1.0, 
                  dfMatTemp.Data(), ntot * numOccLocal, dfMatTemp.Data(), ntot * numOccLocal, 
                  0.0, XTXTemp.Data(), iterused );

              SetValue( XTX, 0.0 );
              MPI_Allreduce( XTXTemp.Data(), XTX.Data(), 
                  iterused * iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );

              DblNumVec gammasTemp1(iterused);
              SetValue( gammasTemp1, 0.0 );
              //blas::Gemv('T', ntot * numOccLocal, iterused, 1.0, dfMatTemp.Data(),
              //    ntot * numOccLocal, gammas.Data(), 1, 0.0, gammasTemp1.Data(), 1 );

              blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, iterused, I_ONE, ntot * numOccLocal, 1.0, 
                  dfMatTemp.Data(), ntot * numOccLocal, gammas.Data(), ntot * numOccLocal, 
                  0.0, gammasTemp1.Data(), iterused );

              DblNumVec gammasTemp2(iterused);
              SetValue( gammasTemp2, 0.0 );
              MPI_Allreduce( gammasTemp1.Data(), gammasTemp2.Data(), 
                  iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );

              lapack::gelss( iterused, iterused, 1, 
                  XTX.Data(), iterused,
                  gammasTemp2.Data(), iterused,
                  S.Data(), rcond, &rank );

              //blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
              //    ntot * numOccLocal, gammasTemp2.Data(), 1, 1.0, vOpt.Data(), 1 );

              blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntot * numOccLocal, I_ONE, iterused, -1.0, 
                  dvMat_.Data(), ntot * numOccLocal, gammasTemp2.Data(), iterused, 
                  1.0, vOpt.Data(), ntot * numOccLocal );

            }

            Print( statusOFS, "  Rank of dfmat = ", rank );

          }

          // Update dfMat, dvMat, vMix 
          // dfMat(:, inext-1) = Res(:)
          // dvMat(:, inext-1) = Pc(:)
          blas::copy( ntot * numOccLocal, ResCol.Data(), 1, 
              dfMat_.VecData(inext-1), 1 );
          blas::copy( ntot * numOccLocal, PcCol.Data(),  1, 
              dvMat_.VecData(inext-1), 1 );

          // Orthogonalize vOpt to obtain psiPc. 
          // psiPc has the same size
          SetValue( psiPcCol, 0.0 );
          blas::copy( ntot * numOccLocal, vOpt.Data(), 1, psiPcCol.Data(), 1 );
          //lapack::Orth( ntot, numOccLocal, psiPcCol.Data(), ntotLocal );

          // Orthogonalization through Cholesky factorization
          if(1){ 
            SetValue( psiPcRow, 0.0 );
            //AlltoallForward (psiPcCol, psiPcRow, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
                psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

            cu_psiPcRow.CopyFrom( psiPcRow );

            DblNumMat XTX(numOccTotal, numOccTotal);
            DblNumMat XTXTemp(numOccTotal, numOccTotal);
            deviceDblNumMat cu_XTXTemp(numOccTotal, numOccTotal);

            DEVICE_BLAS::Gemm( cu_transT, cu_transN, numOccTotal, numOccTotal, ntotLocal, &one, cu_psiPcRow.Data(), 
                ntotLocal, cu_psiPcRow.Data(), ntotLocal, &zero, cu_XTXTemp.Data(), numOccTotal );
            cu_XTXTemp.CopyTo(XTXTemp);

            /*
               blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, psiPcRow.Data(), 
               ntotLocal, psiPcRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
               SetValue( XTX, 0.0 );
             */
            MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

            //if ( mpirank == 0) {
            //  lapack::potrf( lapack::Uplo::Upper, numOccTotal, XTX.Data(), numOccTotal );
            //}
            //MPI_Bcast(XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, 0, mpi_comm);

            cu_XTXTemp.CopyFrom(XTX);
#ifdef USE_MAGMA
            MAGMA::Potrf('U', numOccTotal, cu_XTXTemp.Data(), numOccTotal);
#else
            device_solver::Potrf('U', numOccTotal, cu_XTXTemp.Data(), numOccTotal);
#endif

            DEVICE_BLAS::Trsm( right, up, cu_transN, nondiag, 
                ntotLocal, numOccTotal, &one, cu_XTXTemp.Data(), numOccTotal, cu_psiPcRow.Data(),
                ntotLocal);
            cu_psiPcRow.CopyTo( psiPcRow );

            // X <- X * U^{-1} is orthogonal
            /*
               blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
               psiPcRow.Data(), ntotLocal );
               SetValue( psiPcCol, 0.0 );
             */

            //AlltoallBackward (psiPcRow, psiPcCol, mpi_comm);
            SCALAPACK(pdgemr2d)(&Ng, &No, psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
                psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
          } 

        }//Anderson mixing
      } // GPU malloc and free.

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for GPU  Anderson mixing in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Construct the new Hamiltonian operator
      Spinor spnPsiPc(fft.domain, 1, numStateTotal,
          numStateLocal, false, psiPcCol.Data());

      // Compute the electron density
      GetTime( timeSta );
      ham.CalculateDensity(
          spnPsiPc,
          ham.OccupationRate(),
          totalCharge_ );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing density in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the exchange-correlation potential and energy
      if( ham.XCRequireGradDensity() ){
        GetTime( timeSta );
        ham.CalculateGradDensity( );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing gradient density in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      GetTime( timeSta );
      ham.CalculateXC( Exc_ ); 
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing XC potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the Hartree energy
      GetTime( timeSta );
      ham.CalculateHartree( );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing Hartree potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // No external potential

      // Compute the total potential
      GetTime( timeSta );
      ham.CalculateVtot( vtotNew_ );
      blas::copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing total potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Update Phi <- Psi
      GetTime( timeSta );
      ham.SetPhiEXX( spnPsiPc ); 

      // Update the ACE if needed
      // Still use psi but phi has changed
      if( esdfParam.isHybridACE ){
        if( esdfParam.isHybridDF ){
          ham.CalculateVexxACEDFGPU( psi,  isFixColumnDF );
          // Fix the column after the first iteraiton
          isFixColumnDF = true;
        }
        else{
          ham.CalculateVexxACEGPU ( psi );
        }
      }

      GetTime( timeEnd );
      statusOFS << "GPU Time for updating Phi related variable is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      InnerSolve( phiIter );
      blas::copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );


      EVdw_ = 0.0;

      CalculateEnergy();

      PrintState( phiIter );

      GetTime( timePhiIterEnd );

      statusOFS << "Total wall clock time for this Phi iteration = " << 
        timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

      if(1){

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi ); 

        // In principle there is no need to construct ACE operator here
        // However, this makes the code more readable by directly calling 
        // the MultSpinor function later
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDFGPU( psi, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            // GPU needs to be done
            ham.CalculateVexxACEGPU ( psi );
          }
        }

        GetTime( timeEnd );
        statusOFS << " GPU Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

      }//if

      GetTime( timeSta );
      fock2 = ham.CalculateEXXEnergy( psi ); 
      GetTime( timeEnd );
      statusOFS << "Time for computing the EXX energy is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
      dExx = std::abs(fock2 - fock1) / std::abs(fock2);
      // use scfNorm to reflect dExx
      scfNorm_ = dExx;
      fock1 = fock2;
      Efock_ = fock2;

      Etot_        -= Efock_;
      Efree_       -= Efock_;
      EfreeHarris_ -= Efock_;

      statusOFS << std::endl;
      Print(statusOFS, "Fock energy             = ",  Efock_, "[au]");
      Print(statusOFS, "Etot(with fock)         = ",  Etot_, "[au]");
      Print(statusOFS, "Efree(with fock)        = ",  Efree_, "[au]");
      Print(statusOFS, "EfreeHarris(with fock)  = ",  EfreeHarris_, "[au]");

      if( dExx < scfPhiTolerance_ ){
        statusOFS << "SCF for hybrid functional is converged in " 
          << phiIter << " steps !" << std::endl;
        isPhiIterConverged = true;
      }
      if ( isPhiIterConverged ) break;
    } // for(phiIter)
    
    GetTime( timeEnd );
    statusOFS << "Time for using pcdiis method is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

  } // hybridMixType == "pcdiis"

#else

  // FIXME: LL 1/4/2021 How to reduce the amount of repetition between the CPU and GPU branches?
  if( esdfParam.hybridMixType == "pcdiis" ){
    GetTime( timeSta );

    // This requires a good initial guess of wavefunctions, 
    // from one of the following
    // 1) a regular SCF calculation
    // 2) restarting wavefunction with Hybrid_Active_Init = true

    Int ntot      = fft.domain->NumGridTotal();
    Int ntotFine  = fft.domain->NumGridTotalFine();
    Int numStateTotal = psi.NumStateTotal();
    Int numStateLocal = psi.NumState();
    Int numOccTotal = ham.NumOccupiedState();

    MPI_Comm mpi_comm = eigSolPtr_->FFT().domain->comm;

    Int I_ONE = 1, I_ZERO = 0;
    double D_ONE = 1.0;
    double D_ZERO = 0.0;
    double D_MinusONE = -1.0;

    Real timeSta, timeEnd, timeSta1, timeEnd1;

    //Int contxt0D, contxt2D;
    Int contxt1DCol, contxt1DRow;
    Int nprow0D, npcol0D, myrow0D, mycol0D, info0D;
    Int nprow1DCol, npcol1DCol, myrow1DCol, mycol1DCol, info1DCol;
    Int nprow1DRow, npcol1DRow, myrow1DRow, mycol1DRow, info1DRow;
    Int nprow2D, npcol2D, myrow2D, mycol2D, info2D;

    Int ncolsNgNe1DCol, nrowsNgNe1DCol, lldNgNe1DCol; 
    Int ncolsNgNe1DRow, nrowsNgNe1DRow, lldNgNe1DRow; 
    Int ncolsNgNo1DCol, nrowsNgNo1DCol, lldNgNo1DCol; 
    Int ncolsNgNo1DRow, nrowsNgNo1DRow, lldNgNo1DRow; 

    Int desc_NgNe1DCol[9];
    Int desc_NgNe1DRow[9];
    Int desc_NgNo1DCol[9];
    Int desc_NgNo1DRow[9];

    Int Ne = numStateTotal; 
    Int No = numOccTotal; 
    Int Ng = ntot;

    // 1D col MPI
    nprow1DCol = 1;
    npcol1DCol = mpisize;

    Cblacs_get(0, 0, &contxt1DCol);
    Cblacs_gridinit(&contxt1DCol, "C", nprow1DCol, npcol1DCol);
    Cblacs_gridinfo(contxt1DCol, &nprow1DCol, &npcol1DCol, &myrow1DCol, &mycol1DCol);

    // 1D row MPI
    nprow1DRow = mpisize;
    npcol1DRow = 1;

    Cblacs_get(0, 0, &contxt1DRow);
    Cblacs_gridinit(&contxt1DRow, "C", nprow1DRow, npcol1DRow);
    Cblacs_gridinfo(contxt1DRow, &nprow1DRow, &npcol1DRow, &myrow1DRow, &mycol1DRow);


    //desc_NgNe1DCol
    if(contxt1DCol >= 0){
      nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO, 
        &I_ZERO, &contxt1DCol, &lldNgNe1DCol, &info1DCol);

    //desc_NgNe1DRow
    if(contxt1DRow >= 0){
      nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK_, &Ne, &I_ZERO, 
        &I_ZERO, &contxt1DRow, &lldNgNe1DRow, &info1DRow);

    //desc_NgNo1DCol
    if(contxt1DCol >= 0){
      nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1DCol, &I_ZERO, &nprow1DCol);
      ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1DCol, &I_ZERO, &npcol1DCol);
      lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO, 
        &I_ZERO, &contxt1DCol, &lldNgNo1DCol, &info1DCol);

    //desc_NgNo1DRow
    if(contxt1DRow >= 0){
      nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK_, &myrow1DRow, &I_ZERO, &nprow1DRow);
      ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
      lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
    }    

    SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK_, &No, &I_ZERO, 
        &I_ZERO, &contxt1DRow, &lldNgNo1DRow, &info1DRow);

    if(numStateLocal !=  ncolsNgNe1DCol){
      statusOFS << "numStateLocal = " << numStateLocal << " ncolsNgNe1DCol = " << ncolsNgNe1DCol << std::endl;
      ErrorHandling("The size of numState is not right!");
    }

    if(nrowsNgNe1DRow !=  nrowsNgNo1DRow){
      statusOFS << "nrowsNgNe1DRow = " << nrowsNgNe1DRow << " ncolsNgNo1DRow = " << ncolsNgNo1DRow << std::endl;
      ErrorHandling("The size of nrowsNgNe1DRow and ncolsNgNo1DRow is not right!");
    }


    Int numOccLocal = ncolsNgNo1DCol;
    Int ntotLocal = nrowsNgNe1DRow;

    //DblNumMat psiPcCol(ntot, numStateLocal);
    //DblNumMat psiPcRow(ntotLocal, numStateTotal);
    DblNumMat HpsiCol(ntot, numStateLocal);
    DblNumMat HpsiRow(ntotLocal, numStateTotal);

    dfMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
    dvMat_.Resize( ntot * numOccLocal, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

    DblNumMat psiPcCol(ntot, numOccLocal);
    DblNumMat psiPcRow(ntotLocal, numOccTotal);
    DblNumMat PcCol(ntot, numOccLocal);
    DblNumMat PcRow(ntotLocal, numOccTotal);
    DblNumMat ResCol(ntot, numOccLocal);
    DblNumMat ResRow(ntotLocal, numOccTotal);

    DblNumMat psiMuT(numOccTotal, numOccTotal);
    DblNumMat psiMuTLocal(numOccLocal, numOccTotal);
    DblNumMat HpsiMuT(numOccTotal, numOccTotal);
    DblNumMat HpsiMuTLocal(numOccLocal, numOccTotal);

    DblNumMat psiCol( ntot, numStateLocal );
    SetValue( psiCol, 0.0 );
    DblNumMat psiRow( ntotLocal, numStateTotal );
    SetValue( psiRow, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
    //AlltoallForward (psiCol, psiRow, mpi_comm);
    SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

    DblNumMat psiTemp(ntotLocal, numOccTotal);
    SetValue( psiTemp, 0.0 );

    lapack::lacpy( lapack::MatrixType::General, ntotLocal, numOccTotal, psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal );

    // Phi loop
    for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

      GetTime( timePhiIterStart );

      SetValue( psiCol, 0.0 );
      lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
      SetValue( psiRow, 0.0 );
      //AlltoallForward (psiCol, psiRow, mpi_comm);
      SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
          psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

      if(1){

        DblNumMat psiMuTTemp(numOccTotal, numOccTotal);
        SetValue( psiMuTTemp, 0.0 );
        blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
            psiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
            0.0, psiMuTTemp.Data(), numOccTotal );

        SetValue( psiMuT, 0.0 );
        MPI_Allreduce( psiMuTTemp.Data(), psiMuT.Data(), 
            numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

      }//if

#if ( _DEBUGlevel_ >= 1 )
      {
        // Monitor the singular values as a measure as the quality of
        // the selected columns
        DblNumMat tmp = psiMuT;
        DblNumVec s(numOccTotal);
        lapack::SingularValues( numOccTotal, numOccTotal, tmp.Data(), numOccTotal, s.Data() );
        statusOFS << "Spsi = " << s << std::endl;
      }
#endif

      blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
          psiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
          0.0, PcRow.Data(), ntotLocal );

      SetValue( PcCol, 0.0 );
      //AlltoallBackward (PcRow, PcCol, mpi_comm);
      SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
          PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );

      std::ostringstream msg;
      msg << "Phi iteration # " << phiIter;
      PrintBlock( statusOFS, msg.str() );

      // Compute the residual
      {
        // Compute Hpsi for all psi 
        NumTns<Real> tnsTemp(ntot, 1, numStateLocal, false, 
            HpsiCol.Data());
        ham.MultSpinor( psi, tnsTemp );

        SetValue( HpsiRow, 0.0 );
        //AlltoallForward (HpsiCol, HpsiRow, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &Ne, HpsiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
            HpsiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

        if(1){

          DblNumMat HpsiMuTTemp(numOccTotal,numOccTotal);
          SetValue( HpsiMuTTemp, 0.0 );

          blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
              HpsiRow.Data(), ntotLocal, psiTemp.Data(), ntotLocal, 
              0.0, HpsiMuTTemp.Data(), numOccTotal );

          SetValue( HpsiMuT, 0.0 );

          MPI_Allreduce( HpsiMuTTemp.Data(), HpsiMuT.Data(), 
              numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

        }//if

        blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
            HpsiRow.Data(), ntotLocal, psiMuT.Data(), numOccTotal, 
            0.0, ResRow.Data(), ntotLocal );
        blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, -1.0, 
            psiRow.Data(), ntotLocal, HpsiMuT.Data(), numOccTotal, 
            1.0, ResRow.Data(), ntotLocal );

        SetValue( ResCol, 0.0 );
        //AlltoallBackward (ResRow, ResCol, mpi_comm);
        SCALAPACK(pdgemr2d)(&Ng, &No, ResRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
            ResCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
      }

      // Anderson mixing. Use the same mixMaxDim_ for Phi mixing
      {
        // Optimal input potential in Anderon mixing.
        DblNumVec vOpt( ntot * numOccLocal ); 

        // Number of iterations used, iter should start from 1
        Int iterused = std::min( phiIter-1, mixMaxDim_ ); 
        // The current position of dfMat, dvMat
        Int ipos = phiIter - 1 - ((phiIter-2)/ mixMaxDim_ ) * mixMaxDim_;
        // The next position of dfMat, dvMat
        Int inext = phiIter - ((phiIter-1)/ mixMaxDim_) * mixMaxDim_;

        blas::copy( ntot * numOccLocal, PcCol.Data(), 1, vOpt.Data(), 1 );

        if( phiIter > 1 ){
          // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
          // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
          blas::scal( ntot * numOccLocal, -1.0, dfMat_.VecData(ipos-1), 1 );
          blas::axpy( ntot * numOccLocal, 1.0, ResCol.Data(), 1, dfMat_.VecData(ipos-1), 1 );
          blas::scal( ntot * numOccLocal, -1.0, dvMat_.VecData(ipos-1), 1 );
          blas::axpy( ntot * numOccLocal, 1.0, PcCol.Data(), 1, dvMat_.VecData(ipos-1), 1 );

          // Calculating pseudoinverse
          DblNumMat dfMatTemp(ntot * numOccLocal, mixMaxDim_);
          DblNumVec gammas(ntot * numOccLocal), S(iterused);

          SetValue( dfMatTemp, 0.0 );
          SetValue( gammas, 0.0 );

          int64_t rank;
          // FIXME Magic number
          Real rcond = 1e-9;

          // gammas    = res;
          blas::copy( ntot * numOccLocal, ResCol.Data(), 1, gammas.Data(), 1 );
          lapack::lacpy( lapack::MatrixType::General, ntot * numOccLocal, mixMaxDim_, dfMat_.Data(), ntot * numOccLocal, 
              dfMatTemp.Data(), ntot * numOccLocal );

          // May need different strategy in a parallel setup
          if(0){  

            lapack::gelss( ntot * numOccLocal, iterused, 1, 
                dfMatTemp.Data(), ntot * numOccLocal,
                gammas.Data(), ntot * numOccLocal,
                S.Data(), rcond, &rank );

            blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
                ntot * numOccLocal, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

          }

          if(1){

            DblNumMat XTX(iterused, iterused);
            DblNumMat XTXTemp(iterused, iterused);

            Int lld_ntotnumOccLocal = std::max( ntot * numOccLocal, 1 );

            SetValue( XTXTemp, 0.0 );
            blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, iterused, iterused, ntot * numOccLocal, 1.0, 
                dfMatTemp.Data(), lld_ntotnumOccLocal, dfMatTemp.Data(), 
                lld_ntotnumOccLocal, 0.0, XTXTemp.Data(), iterused );

            SetValue( XTX, 0.0 );
            MPI_Allreduce( XTXTemp.Data(), XTX.Data(), 
                iterused * iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );

            DblNumVec gammasTemp1(iterused);
            SetValue( gammasTemp1, 0.0 );
            //blas::Gemv('T', ntot * numOccLocal, iterused, 1.0, dfMatTemp.Data(),
            //    ntot * numOccLocal, gammas.Data(), 1, 0.0, gammasTemp1.Data(), 1 );

            blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, iterused, I_ONE, ntot * numOccLocal, 1.0, 
                dfMatTemp.Data(), lld_ntotnumOccLocal, gammas.Data(), 
                lld_ntotnumOccLocal, 0.0, gammasTemp1.Data(), iterused );

            DblNumVec gammasTemp2(iterused);
            SetValue( gammasTemp2, 0.0 );
            MPI_Allreduce( gammasTemp1.Data(), gammasTemp2.Data(), 
                iterused, MPI_DOUBLE, MPI_SUM, mpi_comm );

            lapack::gelss( iterused, iterused, 1, 
                XTX.Data(), iterused,
                gammasTemp2.Data(), iterused,
                S.Data(), rcond, &rank );

            //blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot * numOccLocal, iterused, -1.0, dvMat_.Data(),
            //    ntot * numOccLocal, gammasTemp2.Data(), 1, 1.0, vOpt.Data(), 1 );

            blas::gemm( blas::Layout::ColMajor,  blas::Op::NoTrans, blas::Op::NoTrans, ntot * numOccLocal, I_ONE, iterused, -1.0, 
                dvMat_.Data(), lld_ntotnumOccLocal, gammasTemp2.Data(), iterused, 
                1.0, vOpt.Data(), lld_ntotnumOccLocal );

          }

          Print( statusOFS, "  Rank of dfmat = ", rank );

        }

        // Update dfMat, dvMat, vMix 
        // dfMat(:, inext-1) = Res(:)
        // dvMat(:, inext-1) = Pc(:)
        blas::copy( ntot * numOccLocal, ResCol.Data(), 1, 
            dfMat_.VecData(inext-1), 1 );
        blas::copy( ntot * numOccLocal, PcCol.Data(),  1, 
            dvMat_.VecData(inext-1), 1 );

        // Orthogonalize vOpt to obtain psiPc. 
        // psiPc has the same size
        SetValue( psiPcCol, 0.0 );
        blas::copy( ntot * numOccLocal, vOpt.Data(), 1, psiPcCol.Data(), 1 );
        //lapack::Orth( ntot, numOccLocal, psiPcCol.Data(), ntotLocal );

        // Orthogonalization through Cholesky factorization
        if(1){ 
          SetValue( psiPcRow, 0.0 );
          //AlltoallForward (psiPcCol, psiPcRow, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &No, psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
              psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

          DblNumMat XTX(numOccTotal, numOccTotal);
          DblNumMat XTXTemp(numOccTotal, numOccTotal);

          blas::gemm( blas::Layout::ColMajor,  blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, psiPcRow.Data(), 
              ntotLocal, psiPcRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
          SetValue( XTX, 0.0 );
          MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

          if ( mpirank == 0) {
            lapack::potrf( lapack::Uplo::Upper, numOccTotal, XTX.Data(), numOccTotal );
          }
          MPI_Bcast(XTX.Data(), numOccTotal*numOccTotal, MPI_DOUBLE, 0, mpi_comm);

          // X <- X * U^{-1} is orthogonal
          blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
              psiPcRow.Data(), ntotLocal );

          SetValue( psiPcCol, 0.0 );
          //AlltoallBackward (psiPcRow, psiPcCol, mpi_comm);
          SCALAPACK(pdgemr2d)(&Ng, &No, psiPcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
              psiPcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
        } 

      }//Anderson mixing

      DblNumMat psiPcColTemp(ntot, numStateLocal);
      SetValue( psiPcColTemp, 0.0 );

      lapack::lacpy( lapack::MatrixType::General, ntot, numOccLocal, psiPcCol.Data(), ntot, psiPcColTemp.Data(), ntot );

      // Construct the new Hamiltonian operator
      Spinor spnPsiPc(eigSolPtr_->fft_ptr(), 1, numStateTotal,
          numStateLocal, false, psiPcColTemp.Data());

      // Compute the electron density
      GetTime( timeSta );
      ham.CalculateDensity(
          spnPsiPc,
          ham.OccupationRate(),
          totalCharge_ ); 
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing density in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the exchange-correlation potential and energy
      if( ham.XCRequireGradDensity() ){
        GetTime( timeSta );
        ham.CalculateGradDensity( );
        GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
        statusOFS << "Time for computing gradient density in PWDFT is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      }

      GetTime( timeSta );
      ham.CalculateXC( Exc_ ); 
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing XC potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Compute the Hartree energy
      GetTime( timeSta );
      ham.CalculateHartree( );
      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing Hartree potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
      // No external potential

      // Compute the total potential
      GetTime( timeSta );
      ham.CalculateVtot( vtotNew_ );
      blas::copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );

      GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << "Time for computing total potential in PWDFT is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

      // Update Phi <- Psi
      GetTime( timeSta );
      ham.SetPhiEXX( spnPsiPc ); 

      // Update the ACE if needed
      // Still use psi but phi has changed
      if( esdfParam.isHybridACE ){
        if( esdfParam.isHybridDF ){
          ham.CalculateVexxACEDF( psi, isFixColumnDF );
          // Fix the column after the first iteraiton
          isFixColumnDF = true;
        }
        else{
          ham.CalculateVexxACE ( psi );
        }
      }

      GetTime( timeEnd );
      statusOFS << "Time for updating Phi related variable is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      InnerSolve( phiIter );
      blas::copy( ntotFine, vtotNew_.Data(), 1, ham.Vtot().Data(), 1 );


      CalculateEnergy();

      PrintState( phiIter );

      GetTime( timePhiIterEnd );

      statusOFS << "Total wall clock time for this Phi iteration = " << 
        timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

      if(esdfParam.isHybridACETwicePCDIIS == 1){

        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi ); 

        // In principle there is no need to construct ACE operator here
        // However, this makes the code more readable by directly calling 
        // the MultSpinor function later
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF( psi, isFixColumnDF );
            // Fix the column after the first iteraiton
            isFixColumnDF = true;
          }
          else{
            ham.CalculateVexxACE ( psi );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

      }//if

      GetTime( timeSta );
      fock2 = ham.CalculateEXXEnergy( psi ); 
      GetTime( timeEnd );
      statusOFS << "Time for computing the EXX energy is " <<
        timeEnd - timeSta << " [s]" << std::endl << std::endl;

      // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
      dExx = std::abs(fock2 - fock1) / std::abs(fock2);
      // use scfNorm to reflect dExx
      scfNorm_ = dExx;
      fock1 = fock2;
      Efock_ = fock2;

      Etot_        -= Efock_;
      Efree_       -= Efock_;
      EfreeHarris_ -= Efock_;

      statusOFS << std::endl;
      Print(statusOFS, "Fock energy             = ",  Efock_, "[au]");
      Print(statusOFS, "Etot(with fock)         = ",  Etot_, "[au]");
      Print(statusOFS, "Efree(with fock)        = ",  Efree_, "[au]");
      Print(statusOFS, "EfreeHarris(with fock)  = ",  EfreeHarris_, "[au]");

      if( dExx < scfPhiTolerance_ ){
        statusOFS << "SCF for hybrid functional is converged in " 
          << phiIter << " steps !" << std::endl;
        isPhiIterConverged = true;
      }
      if ( isPhiIterConverged ) break;
    } // for(phiIter)

    if(contxt1DCol >= 0) {
      Cblacs_gridexit( contxt1DCol );
    }

    if(contxt1DRow >= 0) {
      Cblacs_gridexit( contxt1DRow );
    }

    GetTime( timeEnd );
    statusOFS << "Time for using pcdiis method is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

  } // hybridMixType == "pcdiis"
#endif

  return ;
}         // -----  end of method SCF::IterateWavefun  ----- 


void
SCF::InnerSolve	( Int iter )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain->comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain->comm, &mpisize);

  Real timeSta, timeEnd;
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  Spinor&      psi = eigSolPtr_->Psi();




  // Solve the eigenvalue problem

  Real eigTolNow;
  if( isEigToleranceDynamic_ ){
    // Dynamic strategy to control the tolerance
    if( iter == 1 )
      eigTolNow = 1e-2;
    else
      eigTolNow = std::max( std::min( scfNorm_*1e-3, 1e-2 ) , eigTolerance_);
  }
  else{
    // Static strategy to control the tolerance
    eigTolNow = eigTolerance_;
  }

  Int numEig = (psi.NumStateTotal());

  if(Diag_SCF_PWDFT_by_Cheby_ == 0)
  {  
    statusOFS << "The current tolerance used by the eigensolver is " 
      << eigTolNow << std::endl;
    statusOFS << "The target number of converged eigenvectors is " 
      << numEig << std::endl;
  }

  GetTime( timeSta );

  if(Diag_SCF_PWDFT_by_Cheby_ == 1)
  {
    if(Cheby_iondynamics_schedule_flag_ == 0)
    {
      // Use static schedule
      statusOFS << std::endl << " CheFSI in PWDFT working on static schedule." << std::endl;
      // Use CheFSI or LOBPCG on first step 
#ifndef _COMPLEX_
      if(iter <= 1){
        if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
          eigSolPtr_->LOBPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
        else
          eigSolPtr_->FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
      }
      else{
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
    else
    {
      // Use ion-dynamics schedule
#ifndef _COMPLEX_
      statusOFS << std::endl << " CheFSI in PWDFT working on ion-dynamics schedule." << std::endl;
      if( iter <= 1)
      {
        for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
          eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
      else
      {
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
  }
  else
  {
    // Use LOBPCG
    if( esdfParam.PWSolver == "LOBPCG" || esdfParam.PWSolver == "LOBPCGScaLAPACK"){
      eigSolPtr_->LOBPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
    } // Use PPCG
    else if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
#ifdef DEVICE
      statusOFS << "DBWY BEFORE EIGENSOLVER PPCG" << std::endl;
      eigSolPtr_->devicePPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow, iter );
#else
      eigSolPtr_->PPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
#endif
    }
    else{
      // FIXME Merge the Chebyshev into an option of PWSolver
      ErrorHandling("Not supported PWSolver type.");
    }
  }

  GetTime( timeEnd );

  statusOFS << std::endl << "Time for the eigensolver is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;

  GetTime( timeSta );
  ham.EigVal() = eigSolPtr_->EigVal();
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for ham.EigVal() in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // No need for normalization using LOBPCG

  // Compute the occupation rate
  GetTime( timeSta );
  CalculateOccupationRate( ham.EigVal(), 
      ham.OccupationRate() );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing occupation rate in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Calculate the Harris energy before updating the density
  CalculateHarrisEnergy ();

  // Compute the electron density
  GetTime( timeSta );
  ham.CalculateDensity(
      psi,
      ham.OccupationRate(),
      totalCharge_ ); 
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing density in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the exchange-correlation potential and energy
  if( ham.XCRequireGradDensity() ){
    GetTime( timeSta );
    ham.CalculateGradDensity( );
    GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for computing gradient density in PWDFT is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  }

  GetTime( timeSta );
  ham.CalculateXC( Exc_ ); 
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing XC potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  // Compute the Hartree energy
  GetTime( timeSta );
  ham.CalculateHartree( );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing Hartree potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
  // No external potential

  // Compute the total potential
  GetTime( timeSta );
  ham.CalculateVtot( vtotNew_ );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for computing total potential in PWDFT is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


  return ;
} 		// -----  end of method SCF::InnerSolve  ----- 

void
SCF::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
  // For a given finite temperature, update the occupation number */
  // NOTE Magic number here
  Real tol = 1e-10; 
  Int maxiter = 100;  

  Real lb, ub, flb, fub, occsum;
  Int ilb, iub, iter;

  Int npsi       = eigSolPtr_->Ham().NumStateTotal();
  Int nOccStates = eigSolPtr_->Ham().NumOccupiedState();

  if( eigVal.m() != npsi ){
    std::ostringstream msg;
    msg 
      << "The number of eigenstates do not match."  << std::endl
      << "eigVal         ~ " << eigVal.m() << std::endl
      << "numStateTotal  ~ " << npsi << std::endl;
    ErrorHandling( msg.str().c_str() );
  }


  if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );

  if( npsi > nOccStates )  {
    /* use bisection to find efermi such that 
     * sum_i fermidirac(ev(i)) = nocc
     */
    ilb = nOccStates-1;
    iub = nOccStates+1;

    if( ilb <= 0 ){
      std::ostringstream msg;
      msg 
        << "The chemical potential is smaller than the lowest eigvalue."<< std::endl
        << "The chemical potential is out of range of eigVal."<< std::endl
        << "Please set Extra_States = 0 to avoid this bug."<< std::endl
        << "NumOccupiedState  ~ " << nOccStates << std::endl
        << "numStateTotal  ~ " << npsi << std::endl
        << "eigVal         ~ " << eigVal.m() << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

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
          ErrorHandling( "Cannot find a lower bound for efermi" );
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
          ErrorHandling( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
        }
      }
    }  /* end while */

    fermi_ = (lb+ub)*0.5;
    occsum = 0.0;
    for(Int j = 0; j < npsi; j++) {
      if(Tbeta_*(eigVal(j) - fermi_) > 250.0) 
          occupationRate(j) = 0.0 ;
      else
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
        if(Tbeta_*(eigVal(j) - fermi_) > 250.0) 
          occupationRate(j) = 0.0 ;
        else
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
      ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
    }
  }


  return ;
}         // -----  end of method SCF::CalculateOccupationRate  ----- 


void
SCF::CalculateEnergy    (  )
{
  Ekin_ = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  // Kinetic energy
  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin_  += numSpin * eigVal(i) * occupationRate(i);
  }

  // Hartree and xc part
  Int  ntot = eigSolPtr_->FFT().domain->NumGridTotalFine();
  Real vol  = eigSolPtr_->FFT().domain->Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
  Ehart_ = 0.0;
  EVxc_  = 0.0;
  for (Int i=0; i<ntot; i++) {
    EVxc_  += vxc(i,RHO) * density(i,RHO);
    Ehart_ += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
  }
  Ehart_ *= vol/Real(ntot);
  EVxc_  *= vol/Real(ntot);


  // Ionic repulsion related energy
  Eself_ = eigSolPtr_->Ham().Eself();

  Ecor_ = (Exc_ - EVxc_) - Ehart_ - Eself_;
  EIonSR_ = eigSolPtr_->Ham().EIonSR();
  Ecor_ += EIonSR_;

  // Van der Waals energy
  EVdw_ = eigSolPtr_->Ham().EVdw();
  Ecor_ += EVdw_;

  // External energy
  Eext_ = eigSolPtr_->Ham().Eext();
  Ecor_ += Eext_;

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz fre energy
  if( eigSolPtr_->Ham().NumOccupiedState() == 
      eigSolPtr_->Ham().NumStateTotal() ){
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
        Efree_ += -numSpin / Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
      }
      else{
        Efree_ += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
      }
    }
    Efree_ += Ecor_ + fermi * eigSolPtr_->Ham().NumOccupiedState() * numSpin; 
  }

  return ;
}         // -----  end of method SCF::CalculateEnergy  ----- 

void
SCF::CalculateHarrisEnergy ( )
{
  // These variables are temporary variables only used in this routine
  Real Ekin, Eself, Ehart, EVxc, Exc, Ecor, Efree, EIonSR, EVdw, Eext;
  
  Ekin = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  // Kinetic energy
  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin  += numSpin * eigVal(i) * occupationRate(i);
  }

  // Self energy part
  Eself = 0;
  std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself +=  ptablePtr_->SelfIonInteraction(type);
  }


  // Ionic repulsion related energy
  Eself = eigSolPtr_->Ham().Eself();

  EIonSR = eigSolPtr_->Ham().EIonSR();

  // Van der Waals energy
  EVdw = eigSolPtr_->Ham().EVdw();

  // External energy
  Eext = eigSolPtr_->Ham().Eext();


  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.
  Int  ntot = eigSolPtr_->FFT().domain->NumGridTotalFine();
  Real vol  = eigSolPtr_->FFT().domain->Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
  Ehart = 0.0;
  EVxc  = 0.0;
  for (Int i=0; i<ntot; i++) {
    EVxc  += vxc(i,RHO) * density(i,RHO);
    Ehart += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
  }
  Ehart *= vol/Real(ntot);
  EVxc  *= vol/Real(ntot);
  Exc    = Exc_;


  // Correction energy
  Ecor = (Exc - EVxc) - Ehart - Eself + EIonSR + EVdw + Eext;

  // Helmholtz free energy
  
  if( eigSolPtr_->Ham().NumOccupiedState() == 
      eigSolPtr_->Ham().NumStateTotal() ){
    // Zero temperature
    Efree = Ekin + Ecor;
  }
  else{
    // Finite temperature
    Efree = 0.0;
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;
    for(Int l=0; l< eigVal.m(); l++) {
      Real eig = eigVal(l);
      if( eig - fermi >= 0){
        Efree += -numSpin / Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
      }
      else{
        Efree += numSpin * (eig - fermi) - numSpin / Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
      }
    }
    Efree += Ecor + fermi * eigSolPtr_->Ham().NumOccupiedState() * numSpin; 
  }

  EfreeHarris_ = Efree;


  return ;
}         // -----  end of method SCF::CalculateHarrisEnergy  ----- 


void
SCF::AndersonMix    ( 
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumVec&      vMix,
    DblNumVec&      vOld,
    DblNumVec&      vNew,
    DblNumMat&      dfMat,
    DblNumMat&      dvMat ) {
  Int ntot  = eigSolPtr_->FFT().domain->NumGridTotalFine();

  // Residual 
  DblNumVec res;
  // Optimal input potential in Anderon mixing.
  DblNumVec vOpt; 
  // Optimal residual in Anderson mixing
  DblNumVec resOpt; 
  // Preconditioned optimal residual in Anderson mixing
  DblNumVec precResOpt;

  res.Resize(ntot);
  vOpt.Resize(ntot);
  resOpt.Resize(ntot);
  precResOpt.Resize(ntot);

  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ ); 
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  // The next position of dfMat, dvMat
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

  res = vOld;
  // res(:) = vOld(:) - vNew(:) is the residual
  blas::axpy( ntot, -1.0, vNew.Data(), 1, res.Data(), 1 );

  vOpt = vOld;
  resOpt = res;

  if( iter > 1 ){
    // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
    // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
    blas::scal( ntot, -1.0, dfMat.VecData(ipos-1), 1 );
    blas::axpy( ntot, 1.0, res.Data(), 1, dfMat.VecData(ipos-1), 1 );
    blas::scal( ntot, -1.0, dvMat.VecData(ipos-1), 1 );
    blas::axpy( ntot, 1.0, vOld.Data(), 1, dvMat.VecData(ipos-1), 1 );


    // Calculating pseudoinverse
    Int nrow = iterused;
    DblNumMat dfMatTemp;
    DblNumVec gammas, S;

    int64_t rank;
    // FIXME Magic number
    Real rcond = 1e-12;

    S.Resize(nrow);

    gammas    = res;
    dfMatTemp = dfMat;

    lapack::gelss( ntot, iterused, 1, 
        dfMatTemp.Data(), ntot, gammas.Data(), ntot,
        S.Data(), rcond, &rank );

    Print( statusOFS, "  Rank of dfmat = ", rank );
    Print( statusOFS, "  Rcond = ", rcond );
    // Update vOpt, resOpt. 

    blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot, nrow, -1.0, dvMat.Data(),
        ntot, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

    blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, ntot, iterused, -1.0, dfMat.Data(),
        ntot, gammas.Data(), 1, 1.0, resOpt.Data(), 1 );
  }

  if( mixType == "kerker+anderson" ){
    KerkerPrecond( precResOpt, resOpt );
  }
  else if( mixType == "anderson" ){
    precResOpt = resOpt;
  }
  else{
    ErrorHandling("Invalid mixing type.");
  }


  // Update dfMat, dvMat, vMix 
  // dfMat(:, inext-1) = res(:)
  // dvMat(:, inext-1) = vOld(:)
  blas::copy( ntot, res.Data(), 1, 
      dfMat.VecData(inext-1), 1 );
  blas::copy( ntot, vOld.Data(),  1, 
      dvMat.VecData(inext-1), 1 );

  // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
  vMix = vOpt;
  blas::axpy( ntot, -mixStepLength, precResOpt.Data(), 1, vMix.Data(), 1 );


  return ;

}         // -----  end of method SCF::AndersonMix  ----- 

void
SCF::KerkerPrecond (
    DblNumVec&  precResidual,
    const DblNumVec&  residual )
{
  Fourier& fft = eigSolPtr_->FFT();
  Int ntot  = fft.domain->NumGridTotalFine();

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

  // FIXME hard coded
  Real KerkerB = 0.08; 
  Real Amin = 0.4;

  for (Int i=0; i<ntot; i++) {
    fft.inputComplexVecFine(i) = Complex(residual(i), 0.0);
  }

  FFTWExecute( fft, fft.forwardPlanFine );

  DblNumVec&  gkkFine = fft.gkkFine;

  for(Int i=0; i<ntot; i++) {
    // Procedure taken from VASP
    if( gkkFine(i) != 0 ){
      fft.outputComplexVecFine(i) *= gkkFine(i) / 
        ( gkkFine(i) + 2.0 * PI * KerkerB );
      //            fft.outputComplexVecFine(i) *= std::min(gkkFine(i) / 
      //                    ( gkkFine(i) + 2.0 * PI * KerkerB ), Amin);
    }
  }
  FFTWExecute ( fft, fft.backwardPlanFine );

  for (Int i=0; i<ntot; i++){
    precResidual(i) = fft.inputComplexVecFine(i).real();    
  }


  return ;
}         // -----  end of method SCF::KerkerPrecond  ----- 


void
SCF::PrintState    ( const Int iter  )
{
  Real HOMO, LUMO;
  HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
  if( eigSolPtr_->Ham().NumExtraState() > 0 )
    LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());
  for(Int i = 0; i < eigSolPtr_->EigVal().m(); i++){
    Print(statusOFS, 
        "band#    = ", i, 
        "  eigval   = ", eigSolPtr_->EigVal()(i),
        "  resval   = ", eigSolPtr_->ResVal()(i),
        "  occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
  }
  statusOFS << std::endl;
  statusOFS 
    << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
    << "       Etot  = Ekin + Ecor" << std::endl
    << "       Efree = Etot    + Entropy" << std::endl << std::endl;
  Print(statusOFS, "Etot              = ",  Etot_, "[au]");
  Print(statusOFS, "Efree             = ",  Efree_, "[au]");
  Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
  Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
  Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
  Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
  Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
  Print(statusOFS, "EVdw              = ",  EVdw_, "[au]"); 
  Print(statusOFS, "Eself             = ",  Eself_, "[au]");
  Print(statusOFS, "EIonSR            = ",  EIonSR_, "[au]");
  Print(statusOFS, "Eext              = ",  Eext_, "[au]");
  Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
  Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
  Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
  Print(statusOFS, "HOMO              = ",  HOMO*au2ev, "[eV]");
  if( eigSolPtr_->Ham().NumExtraState() > 0 ){
    Print(statusOFS, "LUMO              = ",  LUMO*au2ev, "[eV]");
  }


  return ;
}         // -----  end of method SCF::PrintState  ----- 


void
SCF::UpdateMDParameters    ( )
{
  scfMaxIter_ = esdfParam.MDscfOuterMaxIter;
  scfPhiMaxIter_ = esdfParam.MDscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateMDParameters  ----- 

void
SCF::UpdateTDDFTParameters    ( )
{
  //scfMaxIter_    = esdfParam.TDDFTscfOuterMaxIter;
  //scfPhiMaxIter_ = esdfParam.TDDFTscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateTDDFTParameters  ----- 

} // namespace scales
