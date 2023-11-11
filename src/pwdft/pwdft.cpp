//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

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
#include <blas.hh>
#include <lapack.hh>

#include "pwdft.h"

using namespace scales;
using namespace std;
using namespace scales::esdf;
using namespace scales::scalapack;

#ifdef DEVICE
#include "device_mpi_interf.h"
#ifdef USE_MAGMA
#include  "magma.h"
#else
#include "device_solver.h"
#endif
#endif

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
      Print( statusOFS, "mpirank = ", mpirank );
    }
#endif

    Print( statusOFS, "The number of MPI processes is ", mpisize );

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

    Print( statusOFS, "Input file = ", inFile );

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
    Hamiltonian ham;
    EigenSolver eigSol;
    SCF  scf;

    ptable.Setup( );

    fft.Initialize( dm );

    fft.InitializeFine( dm );

    // Hamiltonian

    ham.Setup( dm, esdfParam.atomList );

    SetValue( ham.vext, 0.0 );

    GetTime( timeSta );
    ham.CalculatePseudoPotential( ptable );
    GetTime( timeEnd );
    statusOFS << "Time for calculating the pseudopotential for the Hamiltonian = " 
      << timeEnd - timeSta << " [s]" << std::endl;

    // Wavefunctions
    int numStateTotal = ham.NumStateTotal();
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


    // LL: FIXME 01/04/2021  This should go to Spinor
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

    psi.Setup( dm, 1, ham.NumStateTotal(), numStateLocal, D_ZERO );

    statusOFS << "Spinor setup finished." << std::endl;

    UniformRandom( psi.Wavefun() );

    // LL: FIXME 01/04/2021 This part should go to SCF
    if( ham.IsHybrid() ){
      GetTime( timeSta );
      ham.InitializeEXX( esdfParam.ecutWavefunction, fft );
      GetTime( timeEnd );
      statusOFS << "Time for setting up the exchange for the Hamiltonian part = " 
        << timeEnd - timeSta << " [s]" << std::endl;
      if( esdfParam.isHybridActiveInit )
        ham.SetEXXActive(true);
    }

    // Eigensolver class
    eigSol.Setup( ham, psi, fft );

    statusOFS << "Eigensolver setup finished." << std::endl;

    scf.Setup( eigSol, ptable );

    statusOFS << "SCF setup finished." << std::endl;


    // *********************************************************************
    // Single shot calculation first
    // *********************************************************************
#ifdef DEVICE
    device_mpi::setDevice(MPI_COMM_WORLD);
    device_init_vtot();
    DEVICE_BLAS::Init();
    device_fft::Init(dm);
#ifdef USE_MAGMA
    MAGMA::Init();
#else
    device_solver::Init();
#endif
#endif

    {
      GetTime( timeSta );
      scf.Execute();
      GetTime( timeEnd );
      statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
        << " [s]" << std::endl;
    }

    // *********************************************************************
    // Geometry optimization or Molecular dynamics
    // *********************************************************************


    // LL: FIXME 01/04/2021 Need to organize this into the ionDynamics class. 
    // One possibility is to introduce a scfPtr in the ionDynamics
    // class, and include all routines below as 
    //
    // ionDyn.Execute( scf );
    //
    // and put densityHist / wavefunHist as variables in ionDynamics.
    //
    // We clearly need:
    //   1. a proper structure to hold the history information
    //   2. subroutines for density / wavefunction extrapolation.
    // 
    // NOTE: It is unclear that all the SCF structures, densityHist etc
    // should be put under the `ionDynamics` class. If so some
    // 'callback' routines could be needed.  Need to discuss about this
    // before making a decision.
    {
      IonDynamics ionDyn;

      ionDyn.Setup( ham.AtomList(), ptable ); 

      // Change the SCF parameters if necessary
      scf.UpdateMDParameters( );

      Int maxHist = ionDyn.MaxHist();
      // Need to define both but one of them may be empty
      std::vector<DblNumMat>    densityHist(maxHist);
      std::vector<DblNumTns>    wavefunHist(maxHist);
      DblNumTns                 wavefunPre;           // predictor
      if( esdfParam.MDExtrapolationVariable == "density" ){
        // densityHist[0] is the lastest density
        for( Int l = 0; l < maxHist; l++ ){
          densityHist[l] = ham.Density();
        } // for (l)
      }
      if( esdfParam.MDExtrapolationVariable == "wavefun" ){
        // wavefunHist[0] is the lastest density
        for( Int l = 0; l < maxHist; l++ ){
          wavefunHist[l] = psi.Wavefun();
        } // for (l)
        wavefunPre = psi.Wavefun();
      }

      // Main loop for geometry optimization or molecular dynamics
      // If ionMaxIter == 1, it is equivalent to single shot calculation
      Int ionMaxIter = esdfParam.ionMaxIter;


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
        ham.UpdateHamiltonian( ham.AtomList() );
        ham.CalculatePseudoPotential( ptable );

        // Reset wavefunctions to random values for geometry optimization
        // Except for CheFSI. 
        // LL: Not used 10/19/2020
//        if((ionDyn.IsGeoOpt() == true) && (esdfParam.PWSolver != "CheFSI")){
//          statusOFS << std::endl << " Resetting to random wavefunctions ... \n" << std::endl ; 
//          UniformRandom( psi.Wavefun() );
//        }

        scf.Update( ); 
        GetTime( timeEnd );
        statusOFS << "Time for updating the Hamiltonian = " << timeEnd - timeSta
          << " [s]" << std::endl;

        // Extrapolation of density : used for both geometry optimization and MD    
        // Update the density history through extrapolation
        if( esdfParam.MDExtrapolationVariable == "density" )
        {
          statusOFS << "Extrapolating the density." << std::endl;

          for( Int l = maxHist-1; l > 0; l-- ){
            densityHist[l]     = densityHist[l-1];
          } // for (l)
          densityHist[0] = ham.Density();
          // FIXME add damping factor, currently for aspc2
          // densityHist[0] = omega*ham.Density()+(1.0-omega)*densityHist[0];
          //                    Real omega = 4.0/7.0;
          //                    blas::Scal( densityHist[0].Size(), 1.0-omega, densityHist[0].Data(), 1 );
          //                    blas::axpy( densityHist[0].Size(), omega, ham.Density().Data(),
          //                            1, densityHist[0].Data(), 1 );

          // Compute the extrapolation coefficient
          DblNumVec denCoef;
          ionDyn.ExtrapolateCoefficient( ionIter, denCoef );
          statusOFS << "Extrapolation coefficient = " << denCoef << std::endl;

          // Update the electron density
          DblNumMat& denCurVec  = ham.Density();
          SetValue( denCurVec, 0.0 );
          for( Int l = 0; l < maxHist; l++ ){
            blas::axpy( denCurVec.Size(), denCoef[l], densityHist[l].Data(),
                1, denCurVec.Data(), 1 );
          } // for (l)
        } // density extrapolation

        if( ionDyn.IsGeoOpt() == false )
        {
          // Wavefunction extrapolation for MD , not used in geometry optimization
          if( esdfParam.MDExtrapolationVariable == "wavefun" )
          {
            //Especially for XL-BOMD wavefunction extrapolation  

            if(esdfParam.MDExtrapolationType == "xlbomd"){ 

              statusOFS << "Extrapolating the Wavefunctions for XL-BOMD." << std::endl;

              Int ntot = psi.NumGridTotal();
              Int ncom = psi.NumComponent(); 

              Int numStateTotal = psi.NumStateTotal();
              Int numStateLocal = psi.NumState();
              Int numOccTotal = ham.NumOccupiedState();

              Real dt = esdfParam.MDTimeStep;
              Real kappa = esdfParam.kappaXLBOMD;

              Real w = std::sqrt(kappa)/dt ; // 1.4 comes from sqrt(2)

              MPI_Comm mpi_comm = dm.comm;

              Int BlockSizeScaLAPACK = esdfParam.BlockSizeScaLAPACK;

              Int I_ONE = 1, I_ZERO = 0;
              double D_ONE = 1.0;
              double D_ZERO = 0.0;
              double D_MinusONE = -1.0;

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
                nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK, &myrow1DRow, &I_ZERO, &nprow1DRow);
                ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
                lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
              }    

              SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK, &Ne, &I_ZERO, 
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
                nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK, &myrow1DRow, &I_ZERO, &nprow1DRow);
                ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
                lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
              }    

              SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK, &No, &I_ZERO, 
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

              DblNumMat psiSCFCol( ntot, numStateLocal );
              SetValue( psiSCFCol, 0.0 );

              if(1){

                DblNumMat psiCol( ntot, numStateLocal );
                SetValue( psiCol, 0.0 );
                DblNumMat psiRow( ntotLocal, numStateTotal );
                SetValue( psiRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
                //AlltoallForward (psiCol, psiRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
                    psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

                DblNumMat psiRefCol( ntot, numStateLocal );
                SetValue( psiRefCol, 0.0 );
                DblNumMat psiRefRow( ntotLocal, numStateTotal );
                SetValue( psiRefRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, wavefunHist[0].Data(), ntot, psiRefCol.Data(), ntot );
                //AlltoallForward (psiRefCol, psiRefRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiRefCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
                    psiRefRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

                DblNumMat Temp1(numOccTotal, numOccTotal);
                SetValue( Temp1, 0.0 );
                blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
                    psiRow.Data(), ntotLocal, psiRefRow.Data(), ntotLocal, 
                    0.0, Temp1.Data(), numOccTotal );

                DblNumMat Temp2(numOccTotal, numOccTotal);
                SetValue( Temp2, 0.0 );
                MPI_Allreduce( Temp1.Data(), Temp2.Data(), 
                    numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

                if( mpirank == 0 ){
                  statusOFS << "(Psi'*Psi_{ref})_{0,0} = " << Temp2(0,0) << std::endl;
                }

                DblNumMat psiSCFRow( ntotLocal, numStateTotal );
                SetValue( psiSCFRow, 0.0 );

                blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
                    psiRow.Data(), ntotLocal, Temp2.Data(), numOccTotal, 
                    0.0, psiSCFRow.Data(), ntotLocal );

                //AlltoallBackward (psiSCFRow, psiSCFCol, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiSCFRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, 
                    psiSCFCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt1DCol );

              }//if


              // FIXME More efficient to move the pointer later.
              // Out of core is another option that might
              // necessarily need to be taken into account
              //maxHist >= 3; 
              for( Int l = maxHist-1; l > 0; l-- ){
                wavefunHist[l]     = wavefunHist[l-1];
              } // for (l)

              Real w2t2 = w * w * dt * dt;

              for( Int k = 0; k < numStateLocal; k++ ){
                for (Int j = 0; j < ncom; j++) {
                  Real *psiSCFPtr = psiSCFCol.VecData(k);
                  Real *psiRef0Ptr = wavefunHist[0].VecData(j,k);
                  Real *psiRef1Ptr = wavefunHist[1].VecData(j,k);
                  Real *psiRef2Ptr = wavefunHist[2].VecData(j,k);
                  for( Int r = 0; r < ntot; r++ ){
                    psiRef0Ptr[r] = 2.00 * psiRef1Ptr[r] - psiRef2Ptr[r] + w2t2 * (psiSCFPtr[r] - psiRef1Ptr[r]);
                  } // for (r)

                } // for (j)
              } // for (k)

              // Orthogonalization through Cholesky factorization
              if(1){

                DblNumMat psiCol( ntot, numOccLocal );
                SetValue( psiCol, 0.0 );
                DblNumMat psiRow( ntotLocal, numOccTotal );
                SetValue( psiRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numOccLocal, wavefunHist[0].Data(), ntot, psiCol.Data(), ntot );
                //AlltoallForward (psiCol, psiRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &No, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
                    psiRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

                DblNumMat XTX(numOccTotal, numOccTotal);
                DblNumMat XTXTemp(numOccTotal, numOccTotal);

                blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, psiRow.Data(), 
                    ntotLocal, psiRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
                SetValue( XTX, 0.0 );
                MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

                if ( mpirank == 0) {
                  lapack::potrf( lapack::Uplo::Upper, numOccTotal, XTX.Data(), numOccTotal );
                }
                MPI_Bcast(XTX.Data(), numOccTotal * numOccTotal, MPI_DOUBLE, 0, mpi_comm);

                // X <- X * U^{-1} is orthogonal
                blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
                    psiRow.Data(), ntotLocal );

                SetValue( psiCol, 0.0 );
                //AlltoallBackward (psiRow, psiCol, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &No, psiRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
                    psiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
                lapack::lacpy( lapack::MatrixType::General, ntot, numOccLocal, psiCol.Data(), ntot, psi.Wavefun().Data(), ntot );
              } // if


              // Compute the extrapolated density
              Real totalCharge;
              ham.CalculateDensity(
                  psi,
                  ham.OccupationRate(),
                  totalCharge, 
                  fft );

            } //if Extrapolating using xlbomd
            else { 

              statusOFS << "Extrapolating the Wavefunctions using ASPC." << std::endl;

              Int ntot = psi.NumGridTotal();
              Int ncom = psi.NumComponent(); 

              Int numStateTotal = psi.NumStateTotal();
              Int numStateLocal = psi.NumState();
              Int numOccTotal = ham.NumOccupiedState();

              Real dt = esdfParam.MDTimeStep;
              Real kappa = esdfParam.kappaXLBOMD;

              Real w = std::sqrt(kappa)/dt ; // 1.4 comes from sqrt(2)

              MPI_Comm mpi_comm = dm.comm;

              Int BlockSizeScaLAPACK = esdfParam.BlockSizeScaLAPACK;

              Int I_ONE = 1, I_ZERO = 0;
              double D_ONE = 1.0;
              double D_ZERO = 0.0;
              double D_MinusONE = -1.0;

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
                nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK, &myrow1DRow, &I_ZERO, &nprow1DRow);
                ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol1DRow, &I_ZERO, &npcol1DRow);
                lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
              }    

              SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &BlockSizeScaLAPACK, &Ne, &I_ZERO, 
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
                nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &BlockSizeScaLAPACK, &myrow1DRow, &I_ZERO, &nprow1DRow);
                ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol1DRow, &I_ZERO, &npcol1DRow);
                lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
              }    

              SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &BlockSizeScaLAPACK, &No, &I_ZERO, 
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

              DblNumMat psiSCFCol( ntot, numStateLocal );
              SetValue( psiSCFCol, 0.0 );

              if(1){

                DblNumMat psiCol( ntot, numStateLocal );
                SetValue( psiCol, 0.0 );
                DblNumMat psiRow( ntotLocal, numStateTotal );
                SetValue( psiRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );
                //AlltoallForward (psiCol, psiRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
                    psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

                DblNumMat psiRefCol( ntot, numStateLocal );
                SetValue( psiRefCol, 0.0 );
                DblNumMat psiRefRow( ntotLocal, numStateTotal );
                SetValue( psiRefRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numStateLocal, wavefunPre.Data(), ntot, psiRefCol.Data(), ntot );
                //AlltoallForward (psiRefCol, psiRefRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiRefCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, 
                    psiRefRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1DCol );

                DblNumMat Temp1(numOccTotal, numOccTotal);
                SetValue( Temp1, 0.0 );
                blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, 
                    psiRow.Data(), ntotLocal, psiRefRow.Data(), ntotLocal, 
                    0.0, Temp1.Data(), numOccTotal );

                DblNumMat Temp2(numOccTotal, numOccTotal);
                SetValue( Temp2, 0.0 );
                MPI_Allreduce( Temp1.Data(), Temp2.Data(), 
                    numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm );

                DblNumMat psiSCFRow( ntotLocal, numStateTotal );
                SetValue( psiSCFRow, 0.0 );

                blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, ntotLocal, numOccTotal, numOccTotal, 1.0, 
                    psiRow.Data(), ntotLocal, Temp2.Data(), numOccTotal, 
                    0.0, psiSCFRow.Data(), ntotLocal );

                //AlltoallBackward (psiSCFRow, psiSCFCol, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &Ne, psiSCFRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, 
                    psiSCFCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol, &contxt1DCol );

                // Update wavefunPre, which stores the predictor.
                // After this wavefunPre stores the mix of predictor and corrector
                Int ASPCk;
                if( esdfParam.MDExtrapolationType == "linear" )
                  ASPCk = 1;
                else if( esdfParam.MDExtrapolationType == "aspc2" )
                  ASPCk = 2;
                else if( esdfParam.MDExtrapolationType == "aspc3" )
                  ASPCk = 3;
                else
                  ErrorHandling("Cannot use ASPC extrapolation");

                Real omega = (ASPCk + 1.0) / (2.0 * ASPCk + 1.0);


                for( Int k = 0; k < numStateLocal; k++ ){
                  for (Int j = 0; j < ncom; j++) {
                    Real *psiSCFPtr = psiSCFCol.VecData(k);
                    Real *psiPrePtr = wavefunPre.VecData(j,k);
                    for( Int r = 0; r < ntot; r++ ){
                      psiPrePtr[r] = omega * psiSCFPtr[r] + ( 1.0 - omega ) * psiPrePtr[r];
                    } // for (r)
                  } // for (j)
                } // for (k)

              }//if


              // FIXME More efficient to move the pointer later.
              // Out of core is another option that might
              // necessarily need to be taken into account
              //maxHist = 3; 
              for( Int l = maxHist-1; l > 0; l-- ){
                wavefunHist[l]     = wavefunHist[l-1];
              } // for (l)
              wavefunHist[0] = wavefunPre;

              // Compute the extrapolation coefficient
              DblNumVec denCoef;
              ionDyn.ExtrapolateCoefficient( ionIter, denCoef );
              statusOFS << "Extrapolation coefficient = " << denCoef << std::endl;

              // Reevaluate the predictor
              SetValue( wavefunPre, 0.0 );
              for( Int l = 0; l < maxHist; l++ ){
                blas::axpy( wavefunPre.Size(), denCoef[l], wavefunHist[l].Data(),
                    1, wavefunPre.Data(), 1 );
              } // for (l)

              // Orthogonalization through Cholesky factorization
              if(1){

                DblNumMat psiCol( ntot, numOccLocal );
                SetValue( psiCol, 0.0 );
                DblNumMat psiRow( ntotLocal, numOccTotal );
                SetValue( psiRow, 0.0 );
                lapack::lacpy( lapack::MatrixType::General, ntot, numOccLocal, wavefunPre.Data(), ntot, psiCol.Data(), ntot );
                //AlltoallForward (psiCol, psiRow, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &No, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, 
                    psiRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1DCol );

                DblNumMat XTX(numOccTotal, numOccTotal);
                DblNumMat XTXTemp(numOccTotal, numOccTotal);

                blas::gemm( blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, numOccTotal, numOccTotal, ntotLocal, 1.0, psiRow.Data(), 
                    ntotLocal, psiRow.Data(), ntotLocal, 0.0, XTXTemp.Data(), numOccTotal );
                SetValue( XTX, 0.0 );
                MPI_Allreduce(XTXTemp.Data(), XTX.Data(), numOccTotal * numOccTotal, MPI_DOUBLE, MPI_SUM, mpi_comm);

                if ( mpirank == 0) {
                  lapack::potrf( lapack::Uplo::Upper, numOccTotal, XTX.Data(), numOccTotal );
                }
                MPI_Bcast(XTX.Data(), numOccTotal * numOccTotal, MPI_DOUBLE, 0, mpi_comm);

                // X <- X * U^{-1} is orthogonal
                blas::trsm( blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, ntotLocal, numOccTotal, 1.0, XTX.Data(), numOccTotal, 
                    psiRow.Data(), ntotLocal );

                SetValue( psiCol, 0.0 );
                //AlltoallBackward (psiRow, psiCol, mpi_comm);
                SCALAPACK(pdgemr2d)(&Ng, &No, psiRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, 
                    psiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );
                lapack::lacpy( lapack::MatrixType::General, ntot, numOccLocal, psiCol.Data(), ntot, psi.Wavefun().Data(), ntot );
              } // if


              // Compute the extrapolated density
              Real totalCharge;
              ham.CalculateDensity(
                  psi,
                  ham.OccupationRate(),
                  totalCharge, 
                  fft );

            } //if Extrapolating the Wavefunctions using ASPC

          } // wavefun extrapolation
        } // if( ionDyn.IsGeoOpt() == false )


        GetTime( timeSta );
        scf.Execute( );
        GetTime( timeEnd );
        statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
          << " [s]" << std::endl;


        // Geometry optimization
        if( ionDyn.IsGeoOpt() and ( MaxForce( ham.AtomList() ) < esdfParam.geoOptMaxForce ) ){
          statusOFS << "Stopping criterion for geometry optimization has been reached." << std::endl
            << "Exit the loops for ions." << std::endl;
          break;
        }
      } // ionIter
   }

#ifdef DEVICE
    DEVICE_BLAS::Destroy();
    device_fft::Destroy();
#ifdef USE_MAGMA
    MAGMA::Destroy();
#else
    device_solver::Destroy();
#endif
    device_clean_vtot();
#endif

    Print(statusOFS, "PWDFT has finished the calculation.");
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
