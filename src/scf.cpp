/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin, Wei Hu and Amartya Banerjee
     
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
/// @file scf.cpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing
#include  "scf.hpp"
#include    "blas.hpp"
#include    "lapack.hpp"
#include  "utility.hpp"

namespace  dgdft{

using namespace dgdft::DensityComponent;

SCF::SCF    (  )
{
    eigSolPtr_ = NULL;
    ptablePtr_ = NULL;

}         // -----  end of method SCF::SCF  ----- 

SCF::~SCF    (  )
{

}         // -----  end of method SCF::~SCF  ----- 


void
SCF::Setup    ( const esdf::ESDFInputParam& esdfParam, EigenSolver& eigSol, PeriodTable& ptable )
{
    int mpirank;  MPI_Comm_rank(esdfParam.domain.comm, &mpirank);
    int mpisize;  MPI_Comm_size(esdfParam.domain.comm, &mpisize);

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
        isRestartDensity_ = esdfParam.isRestartDensity;
        isRestartWfn_     = esdfParam.isRestartWfn;
        isOutputDensity_  = esdfParam.isOutputDensity;
        isOutputPotential_  = esdfParam.isOutputPotential;
        isOutputWfn_      = esdfParam.isOutputWfn; 
        isCalculateForceEachSCF_       = esdfParam.isCalculateForceEachSCF;
        Tbeta_         = esdfParam.Tbeta;
        mixVariable_   = esdfParam.mixVariable;

//        numGridWavefunctionElem_ = esdfParam.numGridWavefunctionElem;
//        numGridDensityElem_      = esdfParam.numGridDensityElem;  

        PWSolver_                = esdfParam.PWSolver;
        XCType_                  = esdfParam.XCType;
        VDWType_                 = esdfParam.VDWType;

        isHybridACEOutside_      = esdfParam.isHybridACEOutside;

        // Chebyshev Filtering related parameters
        if(PWSolver_ == "CheFSI")
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
        Int ntot = esdfParam.domain.NumGridTotal();
        Int ntotFine = esdfParam.domain.NumGridTotalFine();

        vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
        dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
        dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

        restartDensityFileName_ = "DEN";
        restartPotentialFileName_ = "POT";
        restartWfnFileName_     = "WFN";
    }

    // Density
    {
        DblNumMat&  density = eigSolPtr_->Ham().Density();

        if( isRestartDensity_ ) {
            std::istringstream rhoStream;      
            SharedRead( restartDensityFileName_, rhoStream);
            // TODO Error checking
            deserialize( density, rhoStream, NO_MASK );    
        } // else using the zero initial guess
        else {
            // Start from pseudocharge, usually this is not a very good idea
            if(1){
                // make sure the pseudocharge is initialized
                DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
                const Domain& dm = esdfParam.domain;

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

            // Start from superposition of Gaussians. FIXME Currently
            // the Gaussian parameter is fixed
            if(0){
                Hamiltonian& ham = eigSolPtr_->Ham();
                std::vector<Atom>& atomList = ham.AtomList();
                Int numAtom = atomList.size();
                std::vector<DblNumVec> gridpos;
                const Domain& dm = esdfParam.domain;
                UniformMeshFine ( dm, gridpos );
                Point3 Ls = dm.length;
                
                Int ntotFine = dm.NumGridTotalFine();
                SetValue( density, 0.0 );


                for (Int a=0; a<numAtom; a++) {
                    // FIXME Each atom's truncation radius and charge are the same.
                    // This only works for hydrogen
                    Real sigma2 = 1.0;
                    Real Z = 1.0;
                    Real coef = Z / std::pow(PI*sigma2, 1.5);
                    Point3 pos = atomList[a].pos;

                    std::vector<DblNumVec>  dist(DIM);

                    Point3 minDist;
                    for( Int d = 0; d < DIM; d++ ){
                        dist[d].Resize( gridpos[d].m() );

                        for( Int i = 0; i < gridpos[d].m(); i++ ){
                            dist[d](i) = gridpos[d](i) - pos[d];
                            dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
                        }
                    }
                    {
                        Int irad = 0;
                        for(Int k = 0; k < gridpos[2].m(); k++)
                            for(Int j = 0; j < gridpos[1].m(); j++)
                                for(Int i = 0; i < gridpos[0].m(); i++){
                                    Real rad2 =  dist[0](i) * dist[0](i) +
                                            dist[1](j) * dist[1](j) +
                                            dist[2](k) * dist[2](k);

                                    density(irad,RHO) += coef*std::exp(-rad2/sigma2);
                                    irad++;
                                } // for (i)
                    } 
                }

                Real sum0 = 0.0;

                // make sure that the electron density is positive
                for (Int i=0; i<ntotFine; i++){
                    sum0 += density(i, RHO);
                }
                sum0 *= dm.Volume() / dm.NumGridTotalFine();

                Print( statusOFS, "Initial density. Sum of density      = ", 
                        sum0 );

                // Rescale the density
                Real fac = ham.NumOccupiedState() * ham.NumSpin() / sum0;

                Real sum1 = 0.0;
                for (int i=0; i <ntotFine; i++){
                    density(i, RHO) *= fac;
                    sum1 += density(i, RHO);
                } 

                Print( statusOFS, "Rescaled density. Sum of density      = ", 
                        sum1 * dm.Volume() / dm.NumGridTotalFine() );
            }
        }
    }

    if( !isRestartWfn_ ) {
        // Randomized input from outside
    }
    else {
        std::istringstream wfnStream;
        SeparateRead( restartWfnFileName_, wfnStream, mpirank );
        deserialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    }

    // XC functional
    {
        isCalculateGradRho_ = false;
        if( XCType_ == "XC_GGA_XC_PBE" || 
                XCType_ == "XC_HYB_GGA_XC_HSE06" ) {
            isCalculateGradRho_ = true;
        }
    }


    return ;
}         // -----  end of method SCF::Setup  ----- 

void
SCF::Update    ( )
{
    {
        Int ntotFine  = eigSolPtr_->FFT().domain.NumGridTotalFine();

        vtotNew_.Resize(ntotFine); SetValue(vtotNew_, 0.0);
        dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
        dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );
    }

    return ;
}         // -----  end of method SCF::Update  ----- 



void
SCF::Iterate (  )
{
    int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain.comm, &mpirank);
    int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain.comm, &mpisize);

    Real timeSta, timeEnd;
    // Only works for KohnSham class
    Hamiltonian& ham = eigSolPtr_->Ham();
    Fourier&     fft = eigSolPtr_->FFT();
    Spinor&      psi = eigSolPtr_->Psi();

    // EXX: Only allow hybrid functional here

    // Compute the exchange-correlation potential and energy
    if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
    }

    // Compute the Hartree energy
    // FIXME
    if(0)
    {
        DblNumMat rho = ham.Density();
        SetValue(ham.Density(), 0.0);
        ham.CalculateHartree( fft );
        // No external potential
        ham.Density() = rho;
    }

    if(1){
        ham.CalculateXC( Exc_, fft ); 
        ham.CalculateHartree( fft );
    }

    // Compute the total potential
    ham.CalculateVtot( ham.Vtot() );

    // FIXME The following treatment of the initial density is not
    // compatible with the density extrapolation step in MD
    if(0){
        if( isRestartDensity_ ){ 
            ham.CalculateXC( Exc_, fft ); 
            // Compute the Hartree energy
            ham.CalculateHartree( fft );
            // No external potential

            // Compute the total potential
            ham.CalculateVtot( ham.Vtot() );
        }
        else{
            // Technically needed, otherwise the initial Vtot will be zero 
            // (density = sum of pseudocharge). 
            // Note that the treatment will be different if the initial
            // density is taken from linear superposition of atomic orbitals
            // 
            // In the future this might need to be changed to something else
            // (see more from QE, VASP and QBox)?
            SetValue(ham.Vtot(), 1.0 );
            statusOFS << "Density may be negative, " << 
                "Skip the calculation of XC for the initial setup. " << std::endl;
        }
    }


    Real timeIterStart(0), timeIterEnd(0);
    Real timePhiIterStart(0), timePhiIterEnd(0);

    // EXX: Run SCF::Iterate here
    bool isPhiIterConverged = false;

    // Fock energies
    Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

    // FIXME Do not use this for now
    if( ham.IsHybrid() == false || isHybridACEOutside_ == true ){
        // Let the hybrid functional be handledo outside the SCF loop
        scfPhiMaxIter_ = 1;
    }

    if( ham.IsEXXActive() == false ){
        Efock_ = 0.0;
    }


    for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

        // Update the ACE if needed
        if( ham.IsHybrid() && isHybridACEOutside_ == false ){

            if( ham.IsEXXActive() ){
                Real dExx;
                
                if( ham.IsHybridACE()){
                    // Update Phi <- Psi
                    GetTime( timeSta );
                    ham.SetPhiEXX( psi, fft ); 
                    if( ham.IsHybridACE() ){
                        if( ham.IsHybridDF() ){
                            ham.CalculateVexxACEDF( psi, fft );
                        }
                        else{
                            ham.CalculateVexxACE ( psi, fft );
                        }
                    }
                    GetTime( timeEnd );
                    statusOFS << "Time for updating Phi related variable is " <<
                        timeEnd - timeSta << " [s]" << std::endl << std::endl;
                }
           
                GetTime( timeSta );
                fock2 = ham.CalculateEXXEnergy( psi, fft ); 
                GetTime( timeEnd );
                statusOFS << "Time for computing the EXX energy is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;

                // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
                dExx = std::abs(fock2 - fock1) / std::abs(fock2);
                fock1 = fock2;
                Efock_ = fock2;
                
                Print(statusOFS, "dExx              = ",  dExx, "[au]");
                if( dExx < scfPhiTolerance_ ){
                    statusOFS << "SCF for hybrid functional is converged in " 
                        << phiIter << " steps !" << std::endl;
                    isPhiIterConverged = true;
                }
            }
            if ( isPhiIterConverged ) break;
            GetTime( timePhiIterStart );
            std::ostringstream msg;
            msg << "Phi iteration # " << phiIter;
            PrintBlock( statusOFS, msg.str() );
        }

        
        // Regular SCF iter
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

            // Solve the eigenvalue problem

            Real eigTolNow;
            if( isEigToleranceDynamic_ ){
                // Dynamic strategy to control the tolerance
                if( iter == 1 )
                    eigTolNow = 1e-2;
                else
                    eigTolNow = std::max( std::min( scfNorm_*1e-2, 1e-2 ) , eigTolerance_);
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
                    if(iter <= 1){
                        if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
                            eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                        else
                            eigSolPtr_->FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
                    }
                    else{
                        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
                    }
                }
                else
                {
                    // Use ion-dynamics schedule
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

                }
            }
            else
            {
                // Use LOBPCG
                if( PWSolver_ == "LOBPCG" ){
                    eigSolPtr_->LOBPCGSolveReal2(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                } // Use LOBPCG with ScaLAPACK
                else if ( PWSolver_ == "LOBPCGScaLAPACK" ){
                    eigSolPtr_->LOBPCGSolveReal3(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                } // Use PPCG
                else if( PWSolver_ == "PPCG" ){
                    eigSolPtr_->PPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
                }
                else{
                    // FIXME Merge the Chebyshev into an option of PWSolver
                    ErrorHandling("Not supported PWSolver type.");
                }
            }


            GetTime( timeEnd );

            ham.EigVal() = eigSolPtr_->EigVal();

            statusOFS << std::endl << " Time for the eigensolver is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;


            // No need for normalization using LOBPCG

            // Compute the occupation rate
            CalculateOccupationRate( ham.EigVal(), 
                    ham.OccupationRate() );

            // Compute the electron density
            ham.CalculateDensity(
                    psi,
                    ham.OccupationRate(),
                    totalCharge_, 
                    fft );


            // Compute the exchange-correlation potential and energy
            if( isCalculateGradRho_ ){
                ham.CalculateGradDensity( fft );
            }
            ham.CalculateXC( Exc_, fft ); 

            // Compute the Hartree energy
            ham.CalculateHartree( fft );
            // No external potential

            // Compute the total potential
            ham.CalculateVtot( vtotNew_ );

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

            // FIXME Dump out the difference of the potential to
            // investigate source of slow SCF convergence
            if(0)
            {
                std::ostringstream vStream;
                serialize( vtotOld_, vStream, NO_MASK );
                serialize( vtotNew_, vStream, NO_MASK ); 
                SharedWrite( "VOLDNEW", vStream );
            }
            

            Evdw_ = 0.0;

            CalculateEnergy();

            PrintState( iter );

            if( scfNorm_ < scfTolerance_ ){
                /* converged */
                statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
                isSCFConverged = true;
            }

            // Potential mixing
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

            GetTime( timeIterEnd );

            statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
                << " [s]" << std::endl;

        }

        if( ham.IsHybrid() && isHybridACEOutside_ == false ){
            if( ham.IsEXXActive() == false ) ham.SetEXXActive(true);

            Etot_ = Etot_ - Efock_;
            Efree_ = Efree_ - Efock_;
            Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
            Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
            Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
            GetTime( timePhiIterEnd );

            statusOFS << "Total wall clock time for this Phi iteration = " << 
                timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;
        } // if (hybrid)

    } // for(phiIter)

    // Calculate the Force
    if(0){
        ham.CalculateForce( psi, fft );
    }
    if(1){
        ham.CalculateForce2( psi, fft );
    }

    // Calculate the VDW energy
    if( VDWType_ == "DFT-D2"){
        CalculateVDW ( Evdw_, forceVdw_ );
        // Update energy
        Etot_  += Evdw_;
        Efree_ += Evdw_;
        Ecor_  += Evdw_;

        // Update force
        std::vector<Atom>& atomList = ham.AtomList();
        for( Int a = 0; a < atomList.size(); a++ ){
            atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
        }
    } 

    // Output the information after SCF
    {
        // Energy
        Real HOMO, LUMO;
        HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
        if( eigSolPtr_->Ham().NumExtraState() > 0 )
            LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());

        // Print out the energy
        PrintBlock( statusOFS, "Energy" );
        statusOFS 
            << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
            << "       Etot  = Ekin + Ecor" << std::endl
            << "       Efree = Etot    + Entropy" << std::endl << std::endl;
        Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
        Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
        Print(statusOFS, "! Evdw            = ",  Evdw_, "[au]"); 
        Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
        Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[ev]");
        if( ham.NumExtraState() > 0 ){
            Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
        }
        Print(statusOFS, "! norm(out-in)/norm(in) = ",  scfNorm_ ); 
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

    // Output restarting information
    if( isOutputDensity_ ){
        if( mpirank == 0 ){
            std::ofstream rhoStream(restartDensityFileName_.c_str());
            if( !rhoStream.good() ){
                ErrorHandling( "Density file cannot be opened." );
            }
            serialize( eigSolPtr_->Ham().Density(), rhoStream, NO_MASK );
            rhoStream.close();
        }
    }    

    // Output the total potential
    if( isOutputPotential_ ){
        if( mpirank == 0 ){
            std::ofstream vtotStream(restartPotentialFileName_.c_str());
            if( !vtotStream.good() ){
                ErrorHandling( "Potential file cannot be opened." );
            }
            serialize( eigSolPtr_->Ham().Vtot(), vtotStream, NO_MASK );
            vtotStream.close();
        }
    }

    if( isOutputWfn_ ){
        std::ostringstream wfnStream;
        serialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
        SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
    }   


    return ;
}         // -----  end of method SCF::Iterate  ----- 



void
SCF::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
    // For a given finite temperature, update the occupation number */
    // FIXME Magic number here
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
    Int  ntot = eigSolPtr_->FFT().domain.NumGridTotalFine();
    Real vol  = eigSolPtr_->FFT().domain.Volume();
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

    // Self energy part
    Eself_ = 0;
    std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
    for(Int a=0; a< atomList.size() ; a++) {
        Int type = atomList[a].type;
        Eself_ +=  ptablePtr_->ptemap()[type].params(PTParam::ESELF);
    }

    // Correction energy
    Ecor_ = (Exc_ - EVxc_) - Ehart_ - Eself_;

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
SCF::CalculateVDW    ( Real& VDWEnergy, DblNumMat& VDWForce )
{

    //Real& VDWEnergy = Evdw_;
    //DblNumMat& VDWForce = forceVdw_;

    std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
    Evdw_ = 0.0;
    forceVdw_.Resize( atomList.size(), DIM );
    SetValue( forceVdw_, 0.0 );

    Int numAtom = atomList.size();

    Domain& dm = eigSolPtr_->FFT().domain;

    // std::vector<Point3>  atompos(numAtom);
    // for( Int i = 0; i < numAtom; i++ ){
    //   atompos[i]   = atomList[i].pos;
    // }

    if( VDWType_ == "DFT-D2"){

        const Int vdw_nspecies = 55;
        Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
        bool need_gradient,newshell;
        const Real vdw_d = 20.0;
        const Real vdw_tol_default = 1e-10;
        const Real vdw_s_pbe = 0.75;
        Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;
        //real(dp) :: rcut,rcut2,rsq,rr,sfact,ucvol,vdw_s
        //character(len=500) :: msg
        //type(atomdata_t) :: atom
        //integer,allocatable :: ivdw(:)
        //real(dp) :: gmet(3,3),gprimd(3,3),rmet(3,3)
        //real(dp),allocatable :: vdw_c6(:,:),vdw_r0(:,:),xred01(:,:)
        //DblNumVec vdw_c6_dftd2(vdw_nspecies);

        double vdw_c6_dftd2[vdw_nspecies] = 
        { 0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
            5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
            10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
            16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
            24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
            38.44,31.74,31.50,29.99, 0.00 };

        // DblNumVec vdw_r0_dftd2(vdw_nspecies);
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
                vdw_c6(i, j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
                vdw_r0(i, j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
            }
        }

        Real vdw_s;

        if (XCType_ == "XC_GGA_XC_PBE") {
            vdw_s = vdw_s_pbe;
        }
        else {
            ErrorHandling( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE!" );
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

                                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                                Real fr = 1.0 / ( 1.0 + ex );
                                Real c6r6 = c6 / pow(rr, 6.0);

                                // Contribution to energy
                                Evdw_ = Evdw_ - sfact * fr * c6r6;

                                // Contribution to force
                                if( i != j ) {

                                    Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                                    Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

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



    return ;
}         // -----  end of method SCF::CalculateVDW  ----- 

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
    Int ntot  = eigSolPtr_->FFT().domain.NumGridTotalFine();

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
    blas::Axpy( ntot, -1.0, vNew.Data(), 1, res.Data(), 1 );

    vOpt = vOld;
    resOpt = res;

    if( iter > 1 ){
        // dfMat(:, ipos-1) = res(:) - dfMat(:, ipos-1);
        // dvMat(:, ipos-1) = vOld(:) - dvMat(:, ipos-1);
        blas::Scal( ntot, -1.0, dfMat.VecData(ipos-1), 1 );
        blas::Axpy( ntot, 1.0, res.Data(), 1, dfMat.VecData(ipos-1), 1 );
        blas::Scal( ntot, -1.0, dvMat.VecData(ipos-1), 1 );
        blas::Axpy( ntot, 1.0, vOld.Data(), 1, dvMat.VecData(ipos-1), 1 );


        // Calculating pseudoinverse
        Int nrow = iterused;
        DblNumMat dfMatTemp;
        DblNumVec gammas, S;

        Int rank;
        // FIXME Magic number
        Real rcond = 1e-3;

        S.Resize(nrow);

        gammas    = res;
        dfMatTemp = dfMat;

        lapack::SVDLeastSquare( ntot, iterused, 1, 
                dfMatTemp.Data(), ntot, gammas.Data(), ntot,
                S.Data(), rcond, &rank );

        Print( statusOFS, "  Rank of dfmat = ", rank );

        // Update vOpt, resOpt. 

        blas::Gemv('N', ntot, nrow, -1.0, dvMat.Data(),
                ntot, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

        blas::Gemv('N', ntot, iterused, -1.0, dfMat.Data(),
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
    blas::Copy( ntot, res.Data(), 1, 
            dfMat.VecData(inext-1), 1 );
    blas::Copy( ntot, vOld.Data(),  1, 
            dvMat.VecData(inext-1), 1 );

    // vMix(:) = vOpt(:) - mixStepLength * precRes(:)
    vMix = vOpt;
    blas::Axpy( ntot, -mixStepLength, precResOpt.Data(), 1, vMix.Data(), 1 );


    return ;

}         // -----  end of method SCF::AndersonMix  ----- 

void
SCF::KerkerPrecond (
        DblNumVec&  precResidual,
        const DblNumVec&  residual )
{
    Fourier& fft = eigSolPtr_->FFT();
    Int ntot  = fft.domain.NumGridTotalFine();

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
                "eigval   = ", eigSolPtr_->EigVal()(i),
                "resval   = ", eigSolPtr_->ResVal()(i),
                "occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
    }
    statusOFS << std::endl;
    statusOFS 
        << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + Evdw" << std::endl
        << "       Etot  = Ekin + Ecor" << std::endl
        << "       Efree = Etot    + Entropy" << std::endl << std::endl;
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
    Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
    Print(statusOFS, "HOMO              = ",  HOMO*au2ev, "[eV]");
    if( eigSolPtr_->Ham().NumExtraState() > 0 ){
        Print(statusOFS, "LUMO              = ",  LUMO*au2ev, "[eV]");
    }
    Print(statusOFS, "norm(vout-vin)/norm(vin) = ", scfNorm_ );


    return ;
}         // -----  end of method SCF::PrintState  ----- 


void
SCF::UpdateMDParameters    ( const esdf::ESDFInputParam& esdfParam )
{
    scfMaxIter_ = esdfParam.MDscfOuterMaxIter;
    scfPhiMaxIter_ = esdfParam.MDscfPhiMaxIter;
    return ;
}         // -----  end of method SCF::UpdateMDParameters  ----- 


} // namespace dgdft
