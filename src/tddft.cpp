/*
  Copyright (c) 2012 The Regents of the University of California,
  through Lawrence Berkeley National Laboratory.  

  Author: Lin Lin

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
/// @file tddft.cpp
/// @brief time dependent density functional theory with ehrenfest dynamics.
/// @date 2017-09-05 

#include "tddft.hpp"


namespace dgdft{

using namespace dgdft::DensityComponent;
using namespace dgdft::esdf;

void setDefaultEfieldOptions( eField * eF)
{
  // set the polarization to x 
  eF->pol.resize(3);
  eF->pol[0] = esdfParam.TDDFTVextPolx;
  eF->pol[1] = esdfParam.TDDFTVextPoly;
  eF->pol[2] = esdfParam.TDDFTVextPolz;

  // set frequency to 0.0 
  eF->freq = esdfParam.TDDFTVextFreq;

  // set phase to 0.0 
  eF->phase = esdfParam.TDDFTVextPhase;

  // set phase to 0.0 
  eF->env = esdfParam.TDDFTVextEnv;

  // set Amp to 0
  eF->Amp = esdfParam.TDDFTVextAmp;

  // set t0 to 0
  eF->t0 = esdfParam.TDDFTVextT0;

  // set tau to 0
  eF->tau = esdfParam.TDDFTVextTau;
}


void setEfieldPolarization( eField* eF, std::vector<Real> & pol)
{
  Real scale = std::sqrt(pol[0]*pol[0] + pol[1]*pol[1] + pol[2] * pol[2]);

  eF->pol.resize(3);
  for(int i =0; i < 3; i++)
    eF->pol[i] = pol[i]/scale;
}

void setEfieldFrequency( eField* eF, Real freq)
{
  eF->freq = freq;
}

void setEfieldPhase( eField* eF, Real phase)
{
  eF->phase = phase;
}

void setEfieldEnv( eField* eF, std::string env)
{
  if(env != "constant" && env != "gaussian" &&
      env != "erf"      && env != "sinsq"    &&
      env != "hann"     && env != "kick")
    env = "gaussian"; // set gaussian as default

  eF->env = env;
}

void setEfieldAmplitude( eField* eF, Real Amp)
{
  eF->Amp = Amp;
}

void setEfieldT0( eField* eF, Real t0)
{
  eF->t0 = t0;
}

void setEfieldTau( eField* eF, Real tau)
{
  if(tau <= 0.0){
    tau = 1.0;
    statusOFS << " Warning: Tau must be positive number; reset to 1.0 " << std::endl;
  }
  eF->tau = tau;
}

void setDefaultTDDFTOptions( TDDFTOptions * options)
{
  options->method        = esdfParam.TDDFTMethod;
  options->ehrenfest     = esdfParam.isTDDFTEhrenfest;
  options->simulateTime  = esdfParam.TDDFTTotalT;
  options->dt            = esdfParam.TDDFTDeltaT;
  options->scfMaxIter    = esdfParam.TDDFTMaxIter; 
  options->krylovTol     = esdfParam.TDDFTKrylovTol;
  options->krylovMax     = esdfParam.TDDFTKrylovMax; 
  options->scfTol        = esdfParam.TDDFTScfTol; 

  //  check check
  options->auto_save     = 0;
  options->load_save     = false;
  options->gmres_restart = 10; // not sure.
  options->adNum         = 20; 
  options->adUpdate      = 1;
  setDefaultEfieldOptions( & options->eField_);
}
void setTDDFTMethod( TDDFTOptions * options, std::string method)
{
  if(method == "RK4" || method == "TPTRAP")
    options->method = method;
  else
    statusOFS << std::endl << " Warning: Method set failed, still use " << options->method << std::endl;
}
void setTDDFTEhrenfest( TDDFTOptions * options, bool ehrenfest)
{
  options->ehrenfest = ehrenfest;
}
void setTDDFTDt( TDDFTOptions * options, Real dT)
{
  options->dt = dT;
}
void setTDDFTTime( TDDFTOptions * options, Real time)
{
  options->simulateTime = time;
}
void setTDDFTkrylovTol( TDDFTOptions * options, Real krylovTol)
{
  options->krylovTol = krylovTol;
}
void setTDDFTkrylovMax( TDDFTOptions *options, int krylovMax)
{
  options->krylovMax = krylovMax;
}
void setTDDFTScfTol( TDDFTOptions *options, Real scfTol)
{
  options->scfTol = scfTol;
}

void TDDFT::SetUp(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    std::vector<Atom>& atomList,
    PeriodTable& ptable) {

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;
  atomListPtr_ = &atomList;

  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  Int mpirank, mpisize;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  if( psi.NumStateTotal() % mpisize != 0) 
      ErrorHandling( " Band must be multiples of Np." );

  // Grab the supercell info
  supercell_x_ = esdfParam.domain.length[0];
  supercell_y_ = esdfParam.domain.length[1];
  supercell_z_ = esdfParam.domain.length[2];

  // History of atomic position
  maxHist_ = 4;  // hard coded
  atomListHist_.resize(maxHist_);
  for( Int l = 0; l < maxHist_; l++ ){
    atomListHist_[l] = atomList;
  }

  Int numAtom = atomList.size();
  atomMass_.Resize( numAtom );
  for(Int a=0; a < numAtom; a++) {
    Int atype = atomList[a].type;
    if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    atomMass_[a]=amu2au*ptable.Mass(atype);
  }

  setDefaultTDDFTOptions( & options_);// set the default options.

  // init the time list
  int size = options_.simulateTime/ options_.dt + 1;
  tlist_.resize(size);
  for(int i = 0; i < size; i++)
    tlist_[i] = i * options_.dt;

  // might need to setup others, will add here.
  k_ = 0;

  // CHECK CHECK: change this to a input parameter.
  calDipole_ = esdfParam.isTDDFTDipole;
  calVext_ = esdfParam.isTDDFTVext;

  if( calDipole_) {
    statusOFS << " ************************ WARNING ******************************** " << std::endl;
    statusOFS << " Warning: Please make sure that your atoms are centered at (0,0,0) " << std::endl;
    statusOFS << " ************************ WARNING ******************************** " << std::endl;

    Xr_.Resize( fft.domain.NumGridTotalFine() );
    Yr_.Resize( fft.domain.NumGridTotalFine() );
    Zr_.Resize( fft.domain.NumGridTotalFine() );
    D_.Resize( fft.domain.NumGridTotalFine() );

    Real * xr = Xr_.Data();
    Real * yr = Yr_.Data();
    Real * zr = Zr_.Data();

    Int  idx;
    Real Xtmp, Ytmp, Ztmp;       

    for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
      for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
        for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){

          idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
          Xtmp = (Real(i) - Real( round(Real(i)/Real(fft.domain.numGridFine[0])) * Real(fft.domain.numGridFine[0]) ) ) / Real(fft.domain.numGridFine[0]);
          Ytmp = (Real(j) - Real( round(Real(j)/Real(fft.domain.numGridFine[1])) * Real(fft.domain.numGridFine[1]) ) ) / Real(fft.domain.numGridFine[1]);
          Ztmp = (Real(k) - Real( round(Real(k)/Real(fft.domain.numGridFine[2])) * Real(fft.domain.numGridFine[2]) ) ) / Real(fft.domain.numGridFine[2]);

          // should be AL(0,0) * X + AL(0,1) * Y + AL(0,2) * Z
          // the other parts are zeros. 
          xr[idx] = Xtmp * supercell_x_ ;
          yr[idx] = Ytmp * supercell_y_ ;
          zr[idx] = Ztmp * supercell_z_ ;

          // get the p.D corresponding to the matlab KSSOLV 
          D_[idx] = xr[idx] * options_.eField_.pol[0] + yr[idx] * options_.eField_.pol[1] + zr[idx] * options_.eField_.pol[2]; 
        }
      }
    }
  } 

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  //statusOFS << " mixMaxDim_ " << mixMaxDim_ << std::endl;
  Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
  dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
  dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

  // init the sgmres solvers. 
  sgmres_solver.Setup(ham, psi, fft, fft.domain.NumGridTotal());

  if(mpirank == 0) {
    vextOFS.open( "vext.out");
    dipoleOFS.open( "dipole.out");
    if(esdfParam.isTDDFTInputV) {
      velocityOFS.open( "TDDFT_VELOCITY");
      std::vector<Point3>  atomvel(numAtom);
      Real x, y, z;
      for( int m = 0; m < numAtom; m++){
        velocityOFS >> x >> y >> z;
        atomList[m].vel = Point3( x, y, z);
      }
    }
  }
  if(esdfParam.isTDDFTInputV) {
    for(Int a = 0; a < numAtom; a++)
      MPI_Bcast( &atomList[a].vel, 3, MPI_DOUBLE, 0, mpi_comm ); 
  }

} // TDDFT::Setup function

Real TDDFT::getEfield(Real t)
{
  Real et = 0.0;
  if (options_.eField_.env == "gaussian" ) {
    Real temp = (t-options_.eField_.t0)/options_.eField_.tau;
    et = options_.eField_.Amp * exp( - temp * temp / 2.0) * cos(options_.eField_.freq * t + options_.eField_.phase);
    return et;
  }
  else{
    statusOFS<< " Wrong Efield input, should be constant/gaussian/erf/sinsq/hann/kick" << std::endl;
    exit(0);
  }
}

void TDDFT::calculateVext(Real t)
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;

  if(calVext_){
    Real et = getEfield(t);
    DblNumVec & vext = ham.Vext();

    vextOFS << "Time[as]: " << t * 24.188843 << " et " << et << std::endl;
    // then Vext = Vext0 + et * options_.D
    // Here we suppose the Vext0 are zeros.
    Int idx;
    for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
      for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
        for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){
          idx = i + j * fft.domain.numGridFine[0] + k * fft.domain.numGridFine[0] * fft.domain.numGridFine[1];
          vext[idx] = et * D_[idx];
        }
      }
    }
  }

}


void TDDFT::Update() {

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  Int ntotFine  = fftPtr_->domain.NumGridTotalFine();
  dfMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dfMat_, 0.0 );
  dvMat_.Resize( ntotFine, mixMaxDim_ ); SetValue( dvMat_, 0.0 );

  //statusOFS << " Update ....... mixMaxDim_ " << mixMaxDim_ << std::endl;
  return;
}
void TDDFT::calculateDipole(Real t)
{
  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  // Di = ∫ Rho(i, j, k) * Xr(i, j, k) *dx
  // Density is not distributed
  Real *density = ham.Density().Data();

  Real Dx = 0.0;
  Real Dy = 0.0;
  Real Dz = 0.0;
  Real sumRho = 0.0;
  Real sumDx  = 0.0;
  Real sumDy  = 0.0;
  Real sumDz  = 0.0;

  Real * xr = Xr_.Data();
  Real * yr = Yr_.Data();
  Real * zr = Zr_.Data();
  for( Int k = 0; k < fft.domain.numGridFine[2]; k++ ){
    for( Int j = 0; j < fft.domain.numGridFine[1]; j++ ){
      for( Int i = 0; i < fft.domain.numGridFine[0]; i++ ){

        sumRho += ( * density);
        sumDx  += ( *xr);
        sumDy  += ( *yr);
        sumDz  += ( *zr);

        Dx -=( *density ) * ( *xr++);
        Dy -=( *density ) * ( *yr++);
        Dz -=( *density++ ) * ( *zr++);
      }
    }
  }
  Dx *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);
  Dy *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);
  Dz *= Real(supercell_x_ * supercell_y_ * supercell_z_) / Real( fft.domain.numGridFine[0] * fft.domain.numGridFine[1]* fft.domain.numGridFine[2]);

  dipoleOFS << "Time[as]: " << t * 24.188843 <<  " " << Dx << " " << Dy << " " << Dz << std::endl;

}    // -----  end of method TDDFT::calculateDipole ---- 

void
TDDFT::VelocityVerlet    ( Int ionIter )
{
  return ;
}         // -----  end of method TDDFT::VelocityVerlet  ----- 

void
TDDFT::MoveIons    ( Int ionIter )
{
  Int mpirank, mpisize;

  return ;
}         // -----  end of method TDDFT::MoveIons  ----- 

void
TDDFT::AndersonMix    ( 
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumVec&      vMix,
    DblNumVec&      vOld, DblNumVec&      vNew,
    DblNumMat&      dfMat,
    DblNumMat&      dvMat ) {

  Int ntot  = fftPtr_->domain.NumGridTotalFine();

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

  Int mixMaxDim_ = esdfParam.mixMaxDim;
  // Number of iterations used, iter should start from 1
  Int iterused = std::min( iter-1, mixMaxDim_ ); 
  // The current position of dfMat, dvMat
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;
  // The next position of dfMat, dvMat
  Int inext = iter - ((iter-1)/ mixMaxDim_) * mixMaxDim_;

  statusOFS << " iter " << iter << " mixMaxDim_ " << mixMaxDim_ << " iterused " << iterused << " ipos  " << ipos << " inext " << inext << std::endl;
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
    Real rcond = 1e-12;

    S.Resize(nrow);

    gammas    = res;
    dfMatTemp = dfMat;

    lapack::SVDLeastSquare( ntot, iterused, 1, 
        dfMatTemp.Data(), ntot, gammas.Data(), ntot,
        S.Data(), rcond, &rank );

    Print( statusOFS, "  Rank of dfmat = ", rank );
    Print( statusOFS, "  Rcond = ", rcond );
    // Update vOpt, resOpt. 

    blas::Gemv('N', ntot, nrow, -1.0, dvMat.Data(),
        ntot, gammas.Data(), 1, 1.0, vOpt.Data(), 1 );

    blas::Gemv('N', ntot, iterused, -1.0, dfMat.Data(),
        ntot, gammas.Data(), 1, 1.0, resOpt.Data(), 1 );
  }

  if( mixType == "kerker+anderson" ){
    Print( statusOFS, " Kerker+anderson  is not supported in TDDFT ");
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
TDDFT::CalculateEnergy  ( PeriodTable& ptable  )
{

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  Ekin_ = 0.0;
  DblNumVec&  eigVal         = ham.EigVal();
  DblNumVec&  occupationRate = ham.OccupationRate();

  // Kinetic energy
  Int numSpin = ham.NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin_  += numSpin * eigVal(i) * occupationRate(i);
  }

  // Hartree and xc part
  Int  ntot = fft.domain.NumGridTotalFine();
  Real vol  = fft.domain.Volume();
  DblNumMat&  density      = ham.Density();
  DblNumMat&  vxc          = ham.Vxc();
  DblNumVec&  pseudoCharge = ham.PseudoCharge();
  DblNumVec&  vhart        = ham.Vhart();
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
  std::vector<Atom>&  atomList = ham.AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ +=  ptable.SelfIonInteraction(type);
  }

  // Correction energy
  Ecor_ = (Exc_ - EVxc_) - Ehart_ - Eself_;

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz fre energy
  if( ham.NumOccupiedState() == 
      ham.NumStateTotal() ){
    // Zero temperature
    Efree_ = Etot_;
  }
  statusOFS << " total Energy :    " << Etot_ << std::endl;

  /*
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
  Efree_ += Ecor_ + fermi * ham.NumOccupiedState() * numSpin; 
  }
   */

  return ;
}         // -----  end of method SCF::CalculateEnergy  ----- 



void TDDFT::advanceRK4( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // print the options_ when first step. 
  if(k_ == 0){
    statusOFS<< std::endl;
    statusOFS<< " ----- TDDFT RK-4 Method Print Options   ----- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.scfTol         " << options_.scfTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " --------------------------------------------- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  // do the verlocity verlet algorithm to move ion
  Int ionIter = k_;
  //VelocityVerlet( ionIter );

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomvel_temp(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  std::vector<Point3>  atompos_mid(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {
    if( mpirank == 0 ){

      Real& dt = options_.dt;
      DblNumVec& atomMass = atomMass_;

      for( Int a = 0; a < numAtom; a++ ){
        atompos[a]     = atomList[a].pos;
        atompos_mid[a] = atomList[a].pos;
        atompos_fin[a] = atomList[a].pos;
        atomvel[a]     = atomList[a].vel;
        atomforce[a]   = atomList[a].force;
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS<< "****************************************************************************" << std::endl << std::endl;
      statusOFS<< "TDDFT RK4 Method, step " << k_ << "  t = " << dt << std::endl;

      for( Int a = 0; a < numAtom; a++ ){
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " position: " << std::setprecision(12) << atompos[a]   << std::endl;
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " velocity: " << std::setprecision(12) << atomvel[a]   << std::endl;
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " Force:    " << std::setprecision(12) << atomforce[a] << std::endl;
      }
      statusOFS<< std::endl;
      statusOFS<< "****************************************************************************" << std::endl << std::endl;
#endif

      // Update velocity and position when doing ehrenfest dynamics
      if(options_.ehrenfest){
        for(Int a=0; a<numAtom; a++) {
          atomvel_temp[a] = atomvel[a]/2.0 + atomforce[a]*dt/atomMass[a]/8.0; 
          atompos_mid[a]  = atompos[a] + atomvel_temp[a] * dt;

          atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
          atompos_fin[a]  = atompos[a] + atomvel_temp[a] * dt;
        }
      }
    }
  }

  // have the atompos_mid and atompos_final
  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_mid[a][0], 3, MPI_DOUBLE, 0, mpi_comm ); 
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm ); 
    }
  }

  // k_ is the current K
  Complex i_Z_One = Complex(0.0, 1.0);
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  //statusOFS << " step " << k_ << " ti " << ti << " tf " << tf << " dT " << dT << std::endl;

  // 4-th order Runge-Kutta  Start now 
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);
  //statusOFS << " Occupation Rate: " << occupationRate << std::endl;
  if(calDipole_)  calculateDipole(tlist_[k_]);
  if(k == 0) {
    Real totalCharge_;
    ham.CalculateDensity(
        psi,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

#if ( _DEBUGlevel_ >= 2 )
    statusOFS << " total Charge init " << setw(16) << totalCharge_ << std::endl;
#endif
    //get the new V(r,t+dt) from the rho(r,t+dt)
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    calculateVext(ti);

#if ( _DEBUGlevel_ >= 2 )
    Real et0 = getEfield(0);
    Real eti = getEfield(ti);
    Real etmid = getEfield(tmid);
    Real etf = getEfield(tf);
    statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
    statusOFS << " et0: " << et0 << " eti " << eti << " etmid " << etmid << " etf " << etf << std::endl;
    statusOFS << " DEBUG INFORMATION ON THE EFILED " << std::endl;
#endif
    ham.CalculateVtot( ham.Vtot() );

  }

  // HX1 = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  CpxNumMat Xtemp(ntot, numStateLocal); // X2, X3, X4
  CpxNumMat HX1(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HX1.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  // test psi * conj(psi) 
#if ( _DEBUGlevel_ >= 2 )
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  statusOFS << " step: " << k_ << " K1 = -i1 * ( H1 * psi ) " << std::endl;
  Int width = numStateLocal;
  Int heightLocal = ntot;
  CpxNumMat  XTXtemp1( width, width );

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
      heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

  for( Int i = 0; i < width; i++)
    statusOFS << " Psi * conjg( Psi) : "  << XTXtemp1(i,i) << std::endl;
  Complex *ptr = psi.Wavefun().Data();
  DblNumVec vtot = ham.Vtot();
  statusOFS << " Psi  0 : " << ptr[0] << " HX1 " << HX1(0,0) << " vtot " << vtot[0] << std::endl;
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif

  //  1. set up Psi <-- X - i ( HX1 ) * dt/2
  Complex* dataPtr = Xtemp.Data();
  Complex* psiDataPtr = psi.Wavefun().Data();
  Complex* HpsiDataPtr = HX1.Data();
  Int numStateTotal = psi.NumStateTotal();
  Spinor psi2 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
    }

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_mid[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // 3. Update the H matrix. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi2,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    calculateVext(tmid);
    ham.CalculateVtot( ham.Vtot() );
  }

  // 4. Calculate the K2 = H2 * X2
  CpxNumMat HX2(ntot, numStateLocal);
  NumTns<Complex> tnsTemp2(ntot, 1, numStateLocal, false, HX2.Data());
  ham.MultSpinor( psi2, tnsTemp2, fft );

  // check the psi * conj(psi)
#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K2 = ( H2 * X2 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi2.Wavefun().Data(), 
        heightLocal, psi2.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi2 * conjg( Psi2) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi2.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 2   0 : " << ptr[0]  << " HX2 " << HX2(0,0)<< " vtot " << vtot[0] << std::endl;
  }
  statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
#endif 

  //  1. set up Psi <-- X - i ( HX2) * dt/2
  Spinor psi3 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  dataPtr = Xtemp.Data();
  psiDataPtr = psi.Wavefun().Data();
  HpsiDataPtr = HX2.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++)
    {
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt/2.0;
    }

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_mid[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // 3. Update the H matrix. 
  // CHECK CHECK: The Vext is zero, not updated here. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi3,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    ham.CalculateVtot( ham.Vtot() );
  }
  // 4. Calculate the K3 = H3 * X3
  CpxNumMat HX3(ntot, numStateLocal);
  NumTns<Complex> tnsTemp3(ntot, 1, numStateLocal, false, HX3.Data());
  ham.MultSpinor( psi3, tnsTemp3, fft );

  // check psi3*conj(psi3)
#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K3 = -i1 * ( H3 * X3 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi3.Wavefun().Data(), 
        heightLocal, psi3.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi3 * conjg( Psi3) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi3.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 3   0 : " << ptr[0]  << " HX3 " << HX2(0,0) << " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif


  //  1. set up Psi <-- X - i ( HX3) * dt
  Spinor psi4 (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );
  dataPtr = Xtemp.Data();
  psiDataPtr = psi.Wavefun().Data();
  HpsiDataPtr = HX3.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++)
    {
      Int index = i* ntot +j;
      dataPtr[index] = psiDataPtr[index] -  i_Z_One * HpsiDataPtr[index] * options_.dt;
    }

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // 3. Update the H matrix. 
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psi4,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    calculateVext(tf);
    ham.CalculateVtot( ham.Vtot() );
  }

  // 4. Calculate the K3 = H3 * X3
  // Now Hpsi is H2 * X2
  // K2 = -i1(H2 * psi) 
  CpxNumMat HX4(ntot, numStateLocal);
  NumTns<Complex> tnsTemp4(ntot, 1, numStateLocal, false, HX4.Data());
  ham.MultSpinor( psi4, tnsTemp4, fft );

#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    statusOFS << " step: " << k_ << " K4 = -i1 * ( H4 * X4 ) " << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi4.Wavefun().Data(), 
        heightLocal, psi4.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi4 * conjg( Psi4) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi4.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi 4   0 : " << ptr[0]  << " HX4 " << HX4(0,0)<< " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif 

  // Xf <--  X - i ( K1 + 2K2 + 2K3 + K4) * dt / 6.0
  psiDataPtr = psi.Wavefun().Data();
  Complex *XfPtr = Xtemp.Data();
  Complex *K1    = HX1.Data();
  Complex *K2    = HX2.Data();
  Complex *K3    = HX3.Data();
  Complex *K4    = HX4.Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      XfPtr[index] =  psiDataPtr[index] 
        - i_Z_One * ( K1[index] + 2.0*K2[index] 
            + 2.0*K3[index] + K4[index]) *options_.dt /6.0;
    }

  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, Xtemp.Data() );

  // 2. if ehrenfest dynamics, re-calculate the Vlocal and Vnonlocal
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  //get the new V(r,t+dt) from the rho(r,t+dt)
  {
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );

    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    DblNumVec vtot;
    Int ntotFine  = fft.domain.NumGridTotalFine();
    vtot.Resize(ntotFine);
    SetValue(vtot, 0.0);
    ham.CalculateVtot( vtot);
    Real *vtot0 = ham.Vtot().Data() ;
#if ( _DEBUGlevel_ >= 2 )
    statusOFS << "Xf delta vtot " << vtot(0) - vtot0[0] << std::endl;
#endif
    blas::Copy( ntotFine, vtot.Data(), 1, ham.Vtot().Data(), 1 );
  }

  psiDataPtr = psi.Wavefun().Data();
  Complex* psiDataPtrFinal = psiFinal.Wavefun().Data();
  for( Int i = 0; i < numStateLocal; i ++)
    for( Int j = 0; j < ntot; j ++){
      Int index = i* ntot +j;
      psiDataPtr[index] =  psiDataPtrFinal[index];
    }

#if ( _DEBUGlevel_ >= 2 )
  {
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
    Int width = numStateLocal;
    Int heightLocal = ntot;
    CpxNumMat  XTXtemp1( width, width );

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, psi.Wavefun().Data(), 
        heightLocal, psi.Wavefun().Data(), heightLocal, 0.0, XTXtemp1.Data(), width );

    for( Int i = 0; i < width; i++)
      statusOFS << " Psi End * conjg( Psi End) : " << XTXtemp1(i,i) << std::endl;
    Complex *ptr = psi.Wavefun().Data();
    DblNumVec vtot = ham.Vtot();
    statusOFS << " Psi End 0 : " << ptr[0] << " vtot " << vtot[0] << std::endl;
    statusOFS<< "***************   DEBUG INFOMATION ******************" << std::endl << std::endl;
  }
#endif

  //update Velocity
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);
    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }

  ++k_;
}

void TDDFT::advancePTTRAP( PeriodTable& ptable ) {

  Int mpirank, mpisize;
  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Comm_rank( mpi_comm, &mpirank );
  MPI_Comm_size( mpi_comm, &mpisize );

  Hamiltonian& ham = *hamPtr_;
  Fourier&     fft = *fftPtr_;
  Spinor&      psi = *psiPtr_;

  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // print the options_ when first step. 
  if(k_ == 0){
    statusOFS<< std::endl;
    statusOFS<< " -----   TDDFT PT-TRAP Print the Options  ---- "     << std::endl;
    statusOFS<< " options.auto_save      " << options_.auto_save      << std::endl;
    statusOFS<< " options.load_save      " << options_.load_save      << std::endl;
    statusOFS<< " options.method         " << options_.method         << std::endl;
    statusOFS<< " options.ehrenfest      " << options_.ehrenfest      << std::endl;
    statusOFS<< " options.simulateTime   " << options_.simulateTime   << std::endl;
    statusOFS<< " options.dt             " << options_.dt             << std::endl;
    statusOFS<< " options.gmres_restart  " << options_.gmres_restart  << std::endl;
    statusOFS<< " options.krylovTol      " << options_.krylovTol      << std::endl;
    statusOFS<< " options.scfTol         " << options_.scfTol         << std::endl;
    statusOFS<< " options.adNum          " << options_.adNum          << std::endl;
    statusOFS<< " options.adUpdate       " << options_.adUpdate       << std::endl;
    statusOFS<< " --------------------------------------------- "     << std::endl;
    statusOFS<< std::endl;
  }

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  // do the verlocity verlet algorithm to move ion
  Int ionIter = k_;
  //VelocityVerlet( ionIter );

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atompos_fin(numAtom);
  {
    std::vector<Point3>  atomvel_temp(numAtom);
    if( mpirank == 0 ){

      Real& dt = options_.dt;
      DblNumVec& atomMass = atomMass_;

      for( Int a = 0; a < numAtom; a++ ){
        atompos[a]     = atomList[a].pos;
        atompos_fin[a] = atomList[a].pos;
        atomvel[a]     = atomList[a].vel;
        atomforce[a]   = atomList[a].force;
      }

#if ( _DEBUGlevel_ >= 0 )
      statusOFS<< "***********************************************" << std::endl ;
      statusOFS<< std::endl;
      statusOFS<< "TDDFT PTTRAP Method, step " << k_ << "  t = " << dt << std::endl;

      for( Int a = 0; a < numAtom; a++ ){
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " position: " << std::setprecision(12) << atompos[a]   << std::endl;
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " velocity: " << std::setprecision(12) << atomvel[a]   << std::endl;
        statusOFS << "time: " << k_*24.188843*dt << " atom " << a << " Force:    " << std::setprecision(12) << atomforce[a] << std::endl;
      }
      statusOFS<< std::endl;
      statusOFS<< "************************************************"<< std::endl;
      statusOFS<< std::endl;
#endif

      // Update velocity and position when doing ehrenfest dynamics
      if(options_.ehrenfest){
        for(Int a=0; a<numAtom; a++) {
          atomvel_temp[a] = atomvel[a] + atomforce[a]*dt/atomMass[a]/2.0; 
          atompos_fin[a]  = atompos[a] + atomvel_temp[a] * dt;
        }
      }
    }
  }

  // have the atompos_final
  if(options_.ehrenfest){
    for(Int a = 0; a < numAtom; a++){
      MPI_Bcast( &atompos_fin[a][0], 3, MPI_DOUBLE, 0, mpi_comm); 
    }
  }

  // k_ is the current K
  Int k = k_;
  Real ti = tlist_[k];
  Real tf = tlist_[k+1];
  Real dT = tf - ti;
  Real tmid =  (ti + tf)/2.0;
  Complex i_Z_One = Complex(0.0, 1.0);
  //statusOFS << " step " << k_ << " ti " << ti << " tf " << tf << " dT " << dT << std::endl;

  // PT-TRAP Method starts, note we only use Ne bands
  DblNumVec &occupationRate = ham.OccupationRate();
  occupationRate.Resize( psi.NumStateTotal() );
  SetValue( occupationRate, 1.0);


  // update H when it is first step.
  if(k == 0) {
    Real totalCharge_;
    ham.CalculateDensity(
        psi,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
    ham.CalculateXC( Exc_, fft ); 
    ham.CalculateHartree( fft );
    calculateVext(ti); // ti is 0
    ham.CalculateVtot( ham.Vtot() );
  }


  // calculate Dipole at the beginning.
  if(calDipole_)  calculateDipole(tlist_[k_]);
  CalculateEnergy( ptable);

  // 1. Calculate Xmid which appears on the right hand of the equation
  // HPSI = (H1 * psi)
  Int ntot  = fft.domain.NumGridTotal();
  Int numStateLocal = psi.NumState();
  Int ntotLocal = ntot/mpisize;
  if(mpirank < (ntot % mpisize)) ntotLocal++;
  Int numStateTotal = psi.NumStateTotal();
  CpxNumMat HPSI(ntot, numStateLocal);
  NumTns<Complex> tnsTemp(ntot, 1, numStateLocal, false, HPSI.Data());
  ham.MultSpinor( psi, tnsTemp, fft );

  // All X's are in G-parallel
  CpxNumMat X(ntotLocal, numStateTotal); 
  CpxNumMat HX(ntotLocal, numStateTotal); 
  CpxNumMat RX(ntotLocal, numStateTotal); 
  CpxNumMat Xmid(ntotLocal, numStateTotal); 
  CpxNumMat Xfin(ntotLocal, numStateTotal); 
  CpxNumMat Ymid(ntotLocal, numStateTotal); 
  CpxNumMat Yfin(ntotLocal, numStateTotal); 
  CpxNumMat XF  (ntotLocal, numStateTotal);

  // psi and HPSI in Band parallel 
  CpxNumMat psiF  ( ntot, numStateLocal );
  CpxNumMat psiCol( ntot, numStateLocal );
  CpxNumMat psiYmid( ntot, numStateLocal );
  CpxNumMat psiYfin( ntot, numStateLocal );
  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );

  // tranfer psi and Hpsi from band-parallel to G-parallel
  AlltoallForward( HPSI,  HX, mpi_comm);
  AlltoallForward( psiCol, X, mpi_comm);
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, X.Data(), ntotLocal, XF.Data(), ntotLocal );

  // RX <-- HX - X*(X'*HX)
  Int width = numStateTotal;
  Int heightLocal = ntotLocal;
  CpxNumMat  XHXtemp( width, width );
  CpxNumMat  XHX( width, width );
  lapack::Lacpy( 'A', ntotLocal, numStateTotal, HX.Data(), ntotLocal, RX.Data(), ntotLocal );
  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );
  MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );
  blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
      X.Data(), heightLocal, XHX.Data(), width, 1.0, RX.Data(), heightLocal );

  // Xmid <-- X - li*T/2 * RX  in G-parallel
  {
    Complex * xmidPtr = Xmid.Data();
    Complex * xPtr    = X.Data();
    Complex * rxPtr   = RX.Data();
    for( Int i = 0; i < numStateTotal; i ++)
      for( Int j = 0; j < ntotLocal; j ++){
        Int index = i* ntotLocal +j;
        xmidPtr[index] = xPtr[index] -  i_Z_One * dT/2.0 * rxPtr[index];
      }
  }

  // Xf <-- X
  Spinor psiFinal (fft.domain, 1, numStateTotal, numStateLocal, false, psiF.Data() );
  lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiF.Data(), ntot );

  // move the atom from atom_begin to atom_final, then recalculate the
  if(options_.ehrenfest){
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].pos  = atompos_fin[a];
    }
    ham.UpdateHamiltonian( atomList );
    ham.CalculatePseudoPotential( ptable );
  }

  // get the charge density, but not update the H matrix
  {
    calculateVext(tf); // tf is the current step, calculate only once.

    // get the charge density of the Hf.
    Real totalCharge_;
    ham.CalculateDensity(
        psiFinal,
        ham.OccupationRate(),
        totalCharge_, 
        fft );
  }

  Int maxscfiter = options_.scfMaxIter; 
  int iscf;
  int totalHx = 0;
  for (iscf = 0; iscf < maxscfiter; iscf++){

    // update the Hf matrix, note rho is calculated before.
    {
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      DblNumVec vtot;
      Int ntotFine  = fft.domain.NumGridTotalFine();
      vtot.Resize(ntotFine);
      SetValue(vtot, 0.0);
      ham.CalculateVtot( vtot);
      Real *vtot0 = ham.Vtot().Data() ;
      blas::Copy( ntotFine, vtot.Data(), 1, ham.Vtot().Data(), 1 );
    }

    // HXf <--- Hf * Xf , Now HPSI is HXf
    ham.MultSpinor( psiFinal, tnsTemp, fft );

    // XHXtemp <--- X'HXf
    AlltoallForward( HPSI,  HX, mpi_comm);
    AlltoallForward( psiF, X, mpi_comm);

    blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, HX.Data(), heightLocal, 0.0, XHXtemp.Data(), width );

    MPI_Allreduce( XHXtemp.Data(), XHX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm );

    // XHX <-- 0.5(XHX + XHX') 
    {
      Complex * xPtr = XHXtemp.Data();
      Complex * yPtr = XHX.Data();
      for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
          xPtr[i*width + j] = 0.5 * ( yPtr[i*width+j] + std::conj(yPtr[j*width+i]) );
        }
      }
    }

    // Diag XHX for the eigen value and eigen vectors
    DblNumVec  eigValS(width);
    lapack::Syevd( 'V', 'U', width, XHXtemp.Data(), width, eigValS.Data() );

    // YpsiMid<-- XpsiMid * XHX 
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
        Xmid.Data(), heightLocal, XHXtemp.Data(), width, 0.0, Ymid.Data(), heightLocal );

    // YpsiF <-- Xf * XHX
    blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, 
        XF.Data(), heightLocal, XHXtemp.Data(), width, 0.0, Yfin.Data(), heightLocal );

    // change from G-para to Band-parallel
    AlltoallBackward( Ymid, psiYmid, mpi_comm);
    AlltoallBackward( Yfin, psiYfin, mpi_comm);


    for( int j = 0; j < numStateLocal; j++){

      int index = j * mpisize + mpirank;
      // psiYmid now is rhs
      Complex omega = eigValS(index) + 2.0 * i_Z_One / dT; 
      blas::Scal( ntot, -2.0*i_Z_One, psiYmid.Data() + j*ntot, 1 );

      // call the SGMRES here.
      sgmres_solver.Solve( psiYmid.Data() + j* ntot, psiYfin.Data() + j* ntot, omega);

    }


    // Change the Parallelization
    AlltoallForward ( psiYfin, XF, mpi_comm);

    // Xf.psi = Ypsif * Cf', where Ypsif is the psi get from GMRES, check check
    blas::Gemm( 'N', 'C', heightLocal, width, width, 1.0, 
        XF.Data(), heightLocal, XHXtemp.Data(), width, 0.0, X.Data(), heightLocal );

    // Change the Parallelization
    AlltoallBackward( X, psiF, mpi_comm);
    lapack::Lacpy( 'A', ntotLocal, numStateTotal, X.Data(), ntotLocal, XF.Data(), ntotLocal );

    // get the density
    {
      Real totalCharge_;
      ham.CalculateDensity(
          psiFinal,
          ham.OccupationRate(),
          totalCharge_, 
          fft );

      Int ntotFine  = fft.domain.NumGridTotalFine();
      DblNumVec vtotNew(ntotFine);
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      ham.CalculateVtot( vtotNew );

      Real normVtotDif = 0.0, normVtotOld = 0.0;
      DblNumVec& vtotOld_ = ham.Vtot();
      Int ntot = vtotOld_.m();
      for( Int i = 0; i < ntot; i++ ){
        normVtotDif += pow( vtotOld_(i) - vtotNew(i), 2.0 );
        normVtotOld += pow( vtotOld_(i), 2.0 );
      }
      normVtotDif = sqrt( normVtotDif );
      normVtotOld = sqrt( normVtotOld );
      Real scfNorm_    = normVtotDif / normVtotOld;

      Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
      totalHx += sgmres_solver.iter_;

      if( scfNorm_ < options_.scfTol){
        /* converged */
        statusOFS << "TDDFT step " << k_ << " SCF is converged in " << iscf << " steps !" << std::endl;
        statusOFS << "TDDFT step " << k_ << " used " << totalHx << " H * x operations!" << std::endl;
        break; // break if converged. 
      }

      statusOFS << " iscf " << iscf + 1 << " mixStepLength " << esdfParam.mixStepLength
        << " " << esdfParam.mixType << std::endl;

      AndersonMix(
          iscf+1,
          esdfParam.mixStepLength,
          esdfParam.mixType,
          ham.Vtot(),
          vtotOld_,
          vtotNew,
          dfMat_,
          dvMat_);

    }
  }

  // psi <--- psiFinal
  blas::Copy( ntot*numStateLocal, psiFinal.Wavefun().Data(), 1, psi.Wavefun().Data(), 1 );

  // Update the atomic position and the force.
  if(options_.ehrenfest){
    ham.CalculateForce( psi, fft);
    Real& dt = options_.dt;
    DblNumVec& atomMass = atomMass_;
    for( Int a = 0; a < numAtom; a++ ){
      atomList[a].vel = atomList[a].vel + (atomforce[a]/atomMass[a] + atomList[a].force/atomMass[a])*dt/2.0;
    } 
  }

  //Update the anderson mixing 
  Update();

  ++k_;
} // TDDFT:: advancePTTRAP

void TDDFT::propagate( PeriodTable& ptable ) {
  Int totalSteps = tlist_.size() - 1;
  if(options_.method == "RK4"){
    for( Int i = 0; i < totalSteps; i++)
      advanceRK4( ptable );
  }
  else if( options_.method == "PTTRAP"){
    for( Int i = 0; i < totalSteps; i++)
      advancePTTRAP( ptable );
  }
}
}
