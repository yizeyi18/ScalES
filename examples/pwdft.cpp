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
/// @file pwdft.cpp
/// @brief Main driver for self-consistent field iteration using plane
/// wave basis set.  
///
/// The current version of pwdft is a sequential code and is used for
/// testing purpose, both for energy and for force.
/// @date 2013-10-16 Original implementation
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-07-15 Parallelization of PWDFT.
/// @date 2016-03-07 Refactoring PWDFT to include geometry optimization
/// and molecular dynamics.
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;


void Usage(){
  std::cout 
    << "pwdft -in [inFile]" << std::endl
    << "in:             Input file (default: pwdft.in)" << std::endl;
}


int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  Real timeSta, timeEnd;

  if( mpirank == 0 )
    Usage();

  try
  {
    // *********************************************************************
    // Input parameter
    // *********************************************************************

    // Initialize log file
#ifdef _RELEASE_
    // In the release mode, only the master processor outputs information
    if( mpirank == 0 ){
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#else
    // Every processor outputs information
    {
      stringstream  ss;
      ss << "statfile." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#endif

    Print( statusOFS, "mpirank = ", mpirank );
    Print( statusOFS, "mpisize = ", mpisize );

    // Initialize input parameters
    std::map<std::string,std::string> options;
    OptionsCreate(argc, argv, options);

    std::string inFile;                   
    if( options.find("-in") != options.end() ){ 
      inFile = options["-in"];
    }
    else{
      inFile = "pwdft.in";
    }


    // Read ESDF input file. Note: esdfParam is a global variable (11/25/2016)
    ESDFReadInput( inFile.c_str() );

    // Print the initial state
    ESDFPrintInput( );

    // Initialize multithreaded version of FFTW
#ifdef _USE_FFTW_OPENMP_
#ifndef _USE_OPENMP_
    ErrorHandling("Threaded FFTW must use OpenMP.");
#endif
    statusOFS << "FFTW uses " << omp_get_max_threads() << " threads." << std::endl;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif


    // *********************************************************************
    // Preparation
    // *********************************************************************
    SetRandomSeed(mpirank);

    Domain&  dm = esdfParam.domain;
    PeriodTable ptable;
    Fourier fft;
    Spinor  psi;
    KohnSham hamKS;
    EigenSolver eigSol;
    SCF  scf;

    ptable.Setup( );

    fft.Initialize( dm );

    fft.InitializeFine( dm );

    // Hamiltonian

    hamKS.Setup( dm, esdfParam.atomList );

    DblNumVec& vext = hamKS.Vext();
    SetValue( vext, 0.0 );

    GetTime( timeSta );
    hamKS.CalculatePseudoPotential( ptable );
    GetTime( timeEnd );
    statusOFS << "Time for calculating the pseudopotential for the Hamiltonian = " 
      << timeEnd - timeSta << " [s]" << std::endl;

    // DEBUG
    if(0){
      std::vector<PseudoPot>& pseudo = hamKS.Pseudo();
      if( mpirank == 1 ){
        std::stringstream vStream;
        std::vector<PseudoPot> pseudott;
        for( Int i = 0; i < 3; i++ ){
          pseudott.push_back(pseudo[i]);
        }
        serialize( pseudott, vStream, NO_MASK );
        mpi::Send( vStream, 0, 1, 2, MPI_COMM_WORLD );
      }
      else{
        std::stringstream vStream;
        MPI_Status status1, status2;
        mpi::Recv( vStream, 1, 1, 2, MPI_COMM_WORLD, status1, status2 );
        std::vector<PseudoPot> pseudott;
        deserialize(pseudott, vStream, NO_MASK);

        statusOFS << "On proc 0, pseudott[1].pseudoCharge.first = " << 
          pseudott[1].pseudoCharge.first << std::endl;
      }
    }
    

    // Wavefunctions
    int numStateTotal = hamKS.NumStateTotal();
    int numStateLocal, blocksize;

    // Safeguard for Chebyshev Filtering
    if(esdfParam.PWSolver == "CheFSI")
    { 
      if(numStateTotal % mpisize != 0)
      {
        MPI_Barrier(MPI_COMM_WORLD);  
        statusOFS << std::endl << std::endl 
          <<" Input Error ! Currently CheFSI within PWDFT requires total number of bands to be divisble by mpisize. " << std::endl << " Total No. of states = " << numStateTotal << " , mpisize = " << mpisize << " ." << std::endl <<  " Use a different value of extrastates." << endl << " Aborting ..." << std::endl << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        exit(-1);  
      }    
    }


    if ( numStateTotal <=  mpisize ) {
      blocksize = 1;

      if ( mpirank < numStateTotal ){
        numStateLocal = 1; // blocksize == 1;
      }
      else { 
        // FIXME Throw an error here.
        numStateLocal = 0;
      }
    }
    else {  // numStateTotal >  mpisize

      if ( numStateTotal % mpisize == 0 ){
        blocksize = numStateTotal / mpisize;
        numStateLocal = blocksize ;
      }
      else {
        // blocksize = ((numStateTotal - 1) / mpisize) + 1;
        blocksize = numStateTotal / mpisize;
        numStateLocal = blocksize ;
        if ( mpirank < ( numStateTotal % mpisize ) ) {
          numStateLocal = numStateLocal + 1 ;
        }
      }    
    }

    psi.Setup( dm, 1, hamKS.NumStateTotal(), numStateLocal, 0.0 );

    statusOFS << "Spinor setup finished." << std::endl;

    UniformRandom( psi.Wavefun() );

    if(0){ // For the same random values of psi in parallel

      MPI_Comm mpi_comm = dm.comm;

      Spinor  psiTemp;
      psiTemp.Setup( dm, 1, hamKS.NumStateTotal(),
          hamKS.NumStateTotal(), 0.0 );

      if (mpirank == 0){
        UniformRandom( psiTemp.Wavefun() );
      }
      MPI_Bcast(psiTemp.Wavefun().Data(),
          psiTemp.Wavefun().m()*psiTemp.Wavefun().n()*psiTemp.Wavefun().p(),
          MPI_DOUBLE, 0, mpi_comm);

      Int size = psi.Wavefun().m() * psi.Wavefun().n();
      Int nocc = psi.Wavefun().p();

      IntNumVec& wavefunIdx = psi.WavefunIdx();
      NumTns<Real>& wavefun = psi.Wavefun();

      for (Int k=0; k<nocc; k++) {
        Real *ptr = psi.Wavefun().MatData(k);
        Real *ptr1 = psiTemp.Wavefun().MatData(wavefunIdx(k));
        for (Int i=0; i<size; i++) {
          *ptr = *ptr1;
          ptr = ptr + 1;
          ptr1 = ptr1 + 1;
        }
      }

    } // if(1)


    if( hamKS.IsHybrid() ){
      GetTime( timeSta );
      hamKS.InitializeEXX( esdfParam.ecutWavefunction, fft );
      GetTime( timeEnd );
      statusOFS << "Time for setting up the exchange for the Hamiltonian part = " 
        << timeEnd - timeSta << " [s]" << std::endl;
    }


    // Eigensolver class
    eigSol.Setup( hamKS, psi, fft );

    statusOFS << "Eigensolver setup finished ." << std::endl;

    scf.Setup( eigSol, ptable );

    statusOFS << "SCF setup finished ." << std::endl;


    // *********************************************************************
    // Single shot calculation first
    // *********************************************************************

    GetTime( timeSta );
    scf.Iterate();
    GetTime( timeEnd );
    statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
      << " [s]" << std::endl;

    // *********************************************************************
    // Geometry optimization or Molecular dynamics
    // *********************************************************************

    IonDynamics ionDyn;

    ionDyn.Setup( hamKS.AtomList(), ptable ); 

    // Change the SCF parameters if necessary
    scf.UpdateMDParameters( );

    Int maxHist = ionDyn.MaxHist();
    // Need to define both but one of them may be empty
    std::vector<DblNumMat>    densityHist(maxHist);
    std::vector<DblNumTns>    wavefunHist(maxHist);
    if( esdfParam.MDExtrapolationVariable == "density" ){
      // densityHist[0] is the lastest density
      for( Int l = 0; l < maxHist; l++ ){
        densityHist[l] = hamKS.Density();
      } // for (l)
    }
    if( esdfParam.MDExtrapolationVariable == "wavefun" ){
      // wavefunHist[0] is the lastest density
      for( Int l = 0; l < maxHist; l++ ){
        wavefunHist[l] = psi.Wavefun();
      } // for (l)
    }

    // Main loop for geometry optimization or molecular dynamics
    // If ionMaxIter == 1, it is equivalent to single shot calculation
    Int ionMaxIter = esdfParam.ionMaxIter;

    Int scfPhiMaxIter = 1;
    // FIXME Do not use this for now.
    if( esdfParam.isHybridACEOutside == true ){
      scfPhiMaxIter = esdfParam.scfPhiMaxIter;
    }

    bool isPhiIterConverged = false;
    Real timePhiIterStart(0), timePhiIterEnd(0);

    for( Int phiIter = 1; phiIter <= scfPhiMaxIter; phiIter++ ){
      if( hamKS.IsHybrid() && esdfParam.isHybridACEOutside == true )
      {
        if ( isPhiIterConverged ) break;
        GetTime( timePhiIterStart );
        std::ostringstream msg;
        msg << "Phi iteration #" << phiIter << "  (Outside SCF)";
        PrintBlock( statusOFS, msg.str() );
      }

      for( Int ionIter = 1; ionIter <= ionMaxIter; ionIter++ ){
        {
          std::ostringstream msg;
          msg << "Ion move step # " << ionIter;
          PrintBlock( statusOFS, msg.str() );
        }


        if(ionIter >= 1)
          scf.set_Cheby_iondynamics_schedule_flag(1);

        // Get the new atomic coordinates
        // NOTE: ionDyn directly updates the coordinates in Hamiltonian
        ionDyn.SetEpot( scf.Efree() );
        ionDyn.MoveIons(ionIter);

        GetTime( timeSta );
        hamKS.UpdateHamiltonian( hamKS.AtomList() );
        hamKS.CalculatePseudoPotential( ptable );
        scf.Update( ); 
        GetTime( timeEnd );
        statusOFS << "Time for updating the Hamiltonian = " << timeEnd - timeSta
          << " [s]" << std::endl;


        // Update the density history through extrapolation
        if( esdfParam.MDExtrapolationVariable == "density" )
        {
          statusOFS << "Extrapolating the density." << std::endl;

          for( Int l = maxHist-1; l > 0; l-- ){
            densityHist[l]     = densityHist[l-1];
          } // for (l)
          densityHist[0] = hamKS.Density();
          // FIXME add damping factor, currently for aspc2
          // densityHist[0] = omega*hamKS.Density()+(1.0-omega)*densityHist[0];
          //                    Real omega = 4.0/7.0;
          //                    blas::Scal( densityHist[0].Size(), 1.0-omega, densityHist[0].Data(), 1 );
          //                    blas::Axpy( densityHist[0].Size(), omega, hamKS.Density().Data(),
          //                            1, densityHist[0].Data(), 1 );

          // Compute the extrapolation coefficient
          DblNumVec denCoef;
          ionDyn.ExtrapolateCoefficient( ionIter, denCoef );
          statusOFS << "Extrapolation coefficient = " << denCoef << std::endl;

          // Update the electron density
          DblNumMat& denCurVec  = hamKS.Density();
          SetValue( denCurVec, 0.0 );
          for( Int l = 0; l < maxHist; l++ ){
            blas::Axpy( denCurVec.Size(), denCoef[l], densityHist[l].Data(),
                1, denCurVec.Data(), 1 );
          } // for (l)
        } // density extrapolation

        if( esdfParam.MDExtrapolationVariable == "wavefun" )
        {
          // FIXME Parallelization
          if( mpisize > 1 )
            ErrorHandling("Wavefunction extrapolation only works for 1 proc.");

          statusOFS << "Extrapolating the Wavefunctions." << std::endl;

          // FIXME More efficient to move the pointer later.
          // Out of core is another option that might
          // necessarily need to be taken into account
          for( Int l = maxHist-1; l > 0; l-- ){
            wavefunHist[l]     = wavefunHist[l-1];
          } // for (l)

          // Use the aligned version of wavefunction
          // psi is orthonormal
          if(1)
          {
            Int ntot      = fft.domain.NumGridTotal();
            Int numStateTotal = psi.NumStateTotal();
            DblNumMat M(numStateTotal, numStateTotal);
            // Lowdin transformation based on SVD
            blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 1.0,
                psi.Wavefun().Data(), ntot, wavefunHist[0].Data(), ntot, 
                0.0, M.Data(), M.m() );
            DblNumMat  U( numStateTotal, numStateTotal );
            DblNumMat VT( numStateTotal, numStateTotal );
            DblNumVec  S( numStateTotal );

            lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
                S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );

            blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, 
                numStateTotal, 1.0, U.Data(), numStateTotal, 
                VT.Data(), numStateTotal, 0.0, M.Data(), numStateTotal );

            blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, 1.0,
                psi.Wavefun().Data(), ntot, M.Data(), numStateTotal,
                0.0, wavefunHist[0].Data(), ntot );
          }

          // Compute the extrapolation coefficient
          DblNumVec denCoef;
          ionDyn.ExtrapolateCoefficient( ionIter, denCoef );
          statusOFS << "Extrapolation coefficient = " << denCoef << std::endl;


          // Update the wavefunction
          // FIXME only works for linear mixing at this stage, which is time reversible
          // Alignment is take into account.

          DblNumTns  wavefunPre  = psi.Wavefun(); // a real copy
          SetValue( wavefunPre, 0.0 );
          // Assume alignment is already done
          for( Int l = 0; l < maxHist; l++ ){
            blas::Axpy( wavefunPre.Size(), denCoef[l], wavefunHist[l].Data(),
                1, wavefunPre.Data(), 1 );
          } // for (l)

          // Alignment. Note: wavefunPre is NOT orthonormal
          if(1)
          {
            Int ntot      = fft.domain.NumGridTotal();
            Int numStateTotal = psi.NumStateTotal();
            // Orthonormalize with SVD. Not the most efficient way 
            DblNumTns  wavefunTmp  = wavefunPre; // a real copy
            DblNumMat  U( numStateTotal, numStateTotal );
            DblNumMat VT( numStateTotal, numStateTotal );
            DblNumVec  S( numStateTotal );

            lapack::QRSVD( ntot, numStateTotal, wavefunTmp.Data(), ntot,
                S.Data(), wavefunPre.Data(), ntot, VT.Data(), numStateTotal );


            DblNumMat M(numStateTotal, numStateTotal);
            // Lowdin transformation based on SVD
            blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntot, 1.0,
                wavefunPre.Data(), ntot, wavefunHist[0].Data(), ntot, 
                0.0, M.Data(), M.m() );

            lapack::QRSVD( numStateTotal, numStateTotal, M.Data(), numStateTotal,
                S.Data(), U.Data(), U.m(), VT.Data(), VT.m() );

            blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, 
                numStateTotal, 1.0, U.Data(), numStateTotal, 
                VT.Data(), numStateTotal, 0.0, M.Data(), numStateTotal );

            blas::Gemm( 'N', 'N', ntot, numStateTotal, numStateTotal, 1.0,
                wavefunPre.Data(), ntot, M.Data(), numStateTotal,
                0.0, psi.Wavefun().Data(), ntot );
          }

          // Compute the extrapolated density
          Real totalCharge;
          hamKS.CalculateDensity(
              psi,
              hamKS.OccupationRate(),
              totalCharge, 
              fft );

        } // wavefun extrapolation



        GetTime( timeSta );
        scf.Iterate( );
        GetTime( timeEnd );
        statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
          << " [s]" << std::endl;


        // Geometry optimization
        if( ionDyn.IsGeoOpt() ){
          if( MaxForce( hamKS.AtomList() ) < esdfParam.geoOptMaxForce ){
            statusOFS << "Stopping criterion for geometry optimization has been reached." << std::endl
              << "Exit the loops for ions." << std::endl;
            break;
          }
        }
      } // ionIter


      // EXX
      if( hamKS.IsHybrid() && esdfParam.isHybridACEOutside == true ){
        Real dExx;
        Real fock0, fock1, fock2;
        if( phiIter == 1 ){
          hamKS.SetEXXActive(true);
          // Update Phi <- Psi
          GetTime( timeSta );
          hamKS.SetPhiEXX( psi, fft ); 
          if( esdfParam.isHybridACE ){
            hamKS.CalculateVexxACE ( psi, fft );
          }
          GetTime( timeEnd );
          statusOFS << "Time for updating Phi related variable is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

          GetTime( timeSta );
          fock2 = hamKS.CalculateEXXEnergy( psi, fft ); 
          GetTime( timeEnd );
          statusOFS << "Time for computing the EXX energy is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;

          // Update the energy
          scf.UpdateEfock(fock2);
          Print(statusOFS, "Fock energy       = ",  scf.Efock(), "[au]");
          Print(statusOFS, "Etot(with fock)   = ",  scf.Etot(), "[au]");
          Print(statusOFS, "Efree(with fock)  = ",  scf.Efree(), "[au]");
        }
        else{
          // Calculate first
          fock1 = hamKS.CalculateEXXEnergy( psi, fft ); 

          // Update Phi <- Psi
          GetTime( timeSta );
          hamKS.SetPhiEXX( psi, fft ); 
          if( esdfParam.isHybridACE ){
            hamKS.CalculateVexxACE ( psi, fft );
          }
          GetTime( timeEnd );
          statusOFS << "Time for updating Phi related variable is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;


          fock0 = fock2;
          // Calculate again
          GetTime( timeSta );
          fock2 = hamKS.CalculateEXXEnergy( psi, fft ); 
          GetTime( timeEnd );
          statusOFS << "Time for computing the EXX energy is " <<
            timeEnd - timeSta << " [s]" << std::endl << std::endl;
          dExx = fock1 - 0.5 * (fock0 + fock2);

          scf.UpdateEfock(fock2);
          Print(statusOFS, "dExx              = ",  dExx, "[au]");
          Print(statusOFS, "Fock energy       = ",  scf.Efock(), "[au]");
          Print(statusOFS, "Etot(with fock)   = ",  scf.Etot(), "[au]");
          Print(statusOFS, "Efree(with fock)  = ",  scf.Efree(), "[au]");

          if( dExx < esdfParam.scfPhiTolerance ){
            statusOFS << "SCF for hybrid functional is converged in " 
              << phiIter << " steps !" << std::endl;
            isPhiIterConverged = true;
          }
        }

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;
      } // if (hybrid)

    } // for(phiIter)

    //    ErrorHandling("Test");


  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
  }

  // Finalize 
#ifdef _USE_FFTW_OPENMP
  fftw_cleanup_threads();
#endif
  MPI_Finalize();

  return 0;
}
