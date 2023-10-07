//Only use in Real wavefunction

#include  "lrtddft.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"


#ifdef GPU
#include  "cu_numvec_impl.hpp"
#include  "cu_numtns_impl.hpp"
#include  "cublas.hpp"
#endif

using namespace dgdft::PseudoComponent;
using namespace dgdft::scalapack;
using namespace dgdft::esdf;

namespace dgdft
{
void LRTDDFT::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    const Domain& dm,
    int nvband,
    int ncband)
{
  //setup:

  numExtraState_ = ham.NumExtraState();               //maxncbandtol
  nocc_ = ham.NumOccupiedState();                     //maxnvbandtol
  density_ = ham.Density();                           //rho
  eigVal_ = ham.EigVal();                             //ev 

  numcomponent_ = psi.NumComponent();                 //spinor
  ntotR2C_ = fft.numGridTotalR2C;
  ntot_ = psi.NumGridTotal() ;

  vol_ = dm.Volume();
  domain_ = dm;

  //MPI Setup
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  if( mpirank == 0 ){
    std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
    statusOFS << std::endl
      << "Output the LRTDDFT information"
      << std::endl << std::endl;
#endif
  }

  return;
}         // -----  end of LRTDDFT::Setup  -----

void LRTDDFT::FFTRtoC(
    Fourier& fft,
    Hamiltonian& ham,
    DblNumMat psiphi,
    DblNumMat& temp,
    Int ncvband)
{
  DblNumVec vtot = ham.Vtot();       

  Real          facH = 2.0;

#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif
    // Int ntothalf = fftPtr->numGridTotalR2C;
    // These two are private variables in the OpenMP context

#ifdef GPU

      Int ntotR2C_ = fft.numGridTotalR2C;
      cuDblNumVec cu_psiphi(ntot_);
      cuDblNumVec cu_psiphi_out(2*ntotR2C_); 
#if 0
      cuDblNumVec cu_gkkR2C(ntotR2C_);
      Real C_temp = 4.0 * PI * facH;
#else
      CpxNumVec psiphi_out_temp(ntotR2C_);
#endif

      // cuDblNumVec cu_TeterPrecond(ntotR2C_);
      // cuda_memcpy_CPU2GPU(cu_TeterPrecond.Data(), fft.TeterPrecondR2C.Data(), sizeof(Real)*ntotR2C_);
// #ifdef _USE_OPENMP_
// #pragma omp for schedule (dynamic,1) nowait
// #endif
    for (Int mu = 0; mu < ncvband; mu++) {
      cuda_memcpy_CPU2GPU(cu_psiphi.Data(), psiphi.VecData(mu), sizeof(Real)*ntot_); 
      //statusOFS << "**************5*****" << cu_psiphi.Size() << std::endl << cu_psiphi_out.Size() << "**************" << std::endl;
      //statusOFS << cu_psiphi.Data() << std::endl;
      cuFFTExecuteForward(fft, fft.cuPlanR2C[0], 0, cu_psiphi, cu_psiphi_out);
#if 0
      cuda_memcpy_CPU2GPU(cu_gkkR2C.Data(), fft.gkkR2C.Data(), sizeof(Real)*ntotR2C_);
      cuda_teter2( reinterpret_cast<cuDoubleComplex*> (cu_psiphi_out.Data()), cu_gkkR2C.Data(), ntotR2C_);
      cublas::Scal(2*ntotR2C_, &C_temp, cu_psiphi_out.Data(), 1);
#else
      cuda_memcpy_GPU2CPU(psiphi_out_temp.Data(),  reinterpret_cast<cuDoubleComplex*>(cu_psiphi_out.Data()), sizeof(Complex)*ntotR2C_);
      for (Int i = 0; i < ntotR2C_; i++) {
        if (fft.gkkR2C(i) < 1e-8) {
          psiphi_out_temp(i) = Z_ZERO;
        }
        else {
          psiphi_out_temp(i) = psiphi_out_temp(i) * 4.0 * PI / fft.gkkR2C(i) * facH;
        }
      }
      cuda_memcpy_CPU2GPU(cu_psiphi_out.Data(), psiphi_out_temp.Data(), sizeof(Complex)*ntotR2C_);
#endif

      cuFFTExecuteInverse(fft, fft.cuPlanC2R[0], 0, cu_psiphi_out, cu_psiphi);

      cuda_memcpy_GPU2CPU(temp.VecData(mu), cu_psiphi.Data(), sizeof(Real)*ntot_);
    }
#else

#ifdef _USE_OPENMP_
#pragma omp for schedule (dynamic,1) nowait
#endif
   for (Int mu = 0; mu < ncvband; mu++) {
      SetValue(fft.inputVecR2C, 0.0);
      SetValue(fft.outputVecR2C, Z_ZERO);

      blas::Copy(ntot_, psiphi.VecData(mu), 1, fft.inputVecR2C.Data(), 1);

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      FFTWExecute(fft, fft.forwardPlanR2C);

      for (Int i = 0; i < ntotR2C_; i++) {
        if (fft.gkkR2C(i) < 1e-8) {
          fft.outputVecR2C(i) = 0;
        }
        else {
          fft.outputVecR2C(i) = fft.outputVecR2C(i) * 4.0 * PI / fft.gkkR2C(i) * facH;
        }
      }

      FFTWExecute(fft, fft.backwardPlanR2C);

      blas::Copy(ntot_, fft.inputVecR2C.Data(), 1, temp.VecData(mu), 1);
#endif



#ifdef _USE_OPENMP_
  } //#pragma omp parallel
#endif

  return;
}

void LRTDDFT::Calculatefxc(
    Fourier& fft,
    DblNumVec& fxcPz)
{
  const double  a = 0.0311;
  const double  b = -0.048;
  const double  c = 0.0020;
  const double  d = -0.0116;
  const double  gc = -0.1423;
  const double  b1 = 1.0529;
  const double  b2 = 0.3334;
  const double  falpha = -0.458165293283143;
  const double  e2 = 1;
  Real          rhoxc;
  Real          rs;
  Real          lnrs;
  Real          rs12;
  Real          ox;
  Real          dox;
  Real          drsdrho;
  Real          dvxdrho;
  Real          dvcdrho;

  for (Int i = 0; i < ntot_; i++) {
    rhoxc = density_(i, 0);
    if (rhoxc > 1e-15) {
      rs = pow(((3 / (4 * PI)) / rhoxc), 1 / 3);
      drsdrho = -pow(rs, 4.0) * 4 * PI / 9;
      dvxdrho = 16 * PI / 27 * falpha * pow(rs, 2);
      if (rs < 1) {
        lnrs = log(rs);
        dvcdrho = a / rs + 2 / 3 * c * (lnrs + 1) + (2 * d - c) / 3;
      }
      else {
        rs12 = sqrt(rs);
        ox = 1 + b1 * rs12 + 4 / 3 * b2 * rs;
        dox = 1 + 7 / 6 * b1 * rs12 + 4 / 3 * b2 * rs;
        dvcdrho = gc / (pow(ox, 2.0)) * (7 / 12 * b1 / rs12 + 4 / 3 * b2 - dox / ox * (b1 / rs12 + 2 * b2));
      }
      fxcPz(i) = e2 * (dvcdrho * drsdrho + dvxdrho);
    }
    else fxcPz(i) = 0;
  }
  return;
}


void LRTDDFT::CalculateLRTDDFT(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft,
    const Domain& dm,
    int nvband,
    int ncband) {
#ifdef GPU
  double alpha = 1.0, beta = 0.0;
  cublas::Init();
  statusOFS << std::endl << "cublas init..." << std::endl << std::endl;
#endif
  Int           ncvband = ncband * nvband;               //Matrix size

  //Final matrix
  DblNumMat  hamTDDFT(ncvband, ncvband);                    //Save all matrix
  SetValue(hamTDDFT, 0.0);
  DblNumVec  eigValS(ncvband);                           //Save energy eig
  SetValue(eigValS, 0.0);

  //main
  int kk = 0;

  //time
  Real timeSta;
  Real timeEnd;
  Real timeStaTotal;
  Real timeEndTotal;
  Real timeFFTRtoC = 0.0;
  Real timeCalculateHartree = 0.0;
  Real timeCalculatefxc = 0.0;

  //MPI init

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot = ntot_;
  Int ntotR2C = ntotR2C_;
  Int ncom = numcomponent_;
  Int numStateTotal = psi.NumStateTotal();
  Real vol = vol_;

  // Convert the column partition to row partition

  Int numStateBlocksize = numStateTotal / mpisize;
  Int ntotBlocksize = ntot / mpisize;

  Int numStateLocal = numStateBlocksize;
  Int ntotLocal = ntotBlocksize;

  Int numStateBlocksizeNcv = ncvband / mpisize;
  Int numStateLocalNcv = numStateBlocksizeNcv;

  if (mpirank < (numStateTotal % mpisize)) {
    numStateLocal = numStateBlocksize + 1;
  }

  if (mpirank < (ncvband % mpisize)) {
    numStateLocalNcv = numStateBlocksizeNcv + 1;
  }

  if (mpirank < (ntot % mpisize)) {
    ntotLocal = ntotBlocksize + 1;
  }

  GetTime(timeStaTotal);

  DblNumMat psiphiRow(ntotLocal, ncvband);                       //Save wavefunproduct Row
  SetValue(psiphiRow, 0.0);

  DblNumMat tempRow(ntotLocal, ncvband);                              //Save HartreeFock wave Row
  SetValue(tempRow, 0.0);

  {
    DblNumMat psiCol(ntot, numStateLocal);                         //Save wavefun Col
    SetValue(psiCol, 0.0);
    DblNumMat psiRow(ntotLocal, numStateTotal);                    //Save wavefun Row  
    SetValue(psiRow, 0.0);

    DblNumMat psiphiCol(ntot, numStateLocalNcv);                   //Save wavefunproduct Col
    SetValue(psiphiCol, 0.0);

    GetTime(timeSta);
    lapack::Lacpy('A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot);
    GetTime(timeEnd);
    statusOFS << "Time for Lacpy                   = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    AlltoallForward(psiCol, psiRow, domain_.comm);

    //SCALAPACK(pdgemr2d)(&Ng, &No, PcRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow,
    //	    PcCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol, &contxt1DCol );

    GetTime(timeEnd);
    statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
#ifdef _USE_OPENMP_
#pragma omp parallel
    {
#endif

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
      for (Int k = nocc_ - nvband; k < nocc_; k++) {
        for (Int j = nocc_; j < nocc_ + ncband; j++) {
          for (Int i = 0; i < ntotLocal; i++) {
            psiphiRow(i, kk) = (psiRow(i, k)) * (psiRow(i, j));
          }//for i 
          kk++; 
        }//for j
      }//for k

#ifdef _USE_OPENMP_
    } //#pragma omp parallel
#endif

    GetTime(timeEnd);
    statusOFS << "Time for psiphiRow               = " << timeEnd - timeSta << " [s]" << std::endl;

    GetTime(timeSta);
    AlltoallBackward(psiphiRow, psiphiCol, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for AlltoallBackward        = " << timeEnd - timeSta << " [s]" << std::endl;

    // Fourier transform

    DblNumMat     tempCol(ntot_, numStateLocalNcv);                  //Save HartreeFock wave
    SetValue(tempCol, 0.0);

    GetTime(timeSta);
    FFTRtoC(fft, ham, psiphiCol, tempCol, numStateLocalNcv);
    GetTime(timeEnd);
    statusOFS << "Time for FFTRtoC                 = " << timeEnd - timeSta << " [s]" << std::endl;
    timeFFTRtoC = timeFFTRtoC + (timeEnd - timeSta);

    //end of FFT

    //get Hartree potential   

    //MPI belong to Hartree Fock

    GetTime(timeSta);
    AlltoallForward(tempCol, tempRow, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "Time for AlltoallForward         = " << timeEnd - timeSta << " [s]" << std::endl;

  }
  
  //Hartree Fock setup
  //DblNumMat     HartreeFock(ncvband, ncvband);                             //Save HartreeFock matrix
  //SetValue(HartreeFock, 0.0);

  //Hartree Fock product

  if(0){
    //Hartree Fock setup
    DblNumMat     HartreeFock(ncvband, ncvband);                             //Save HartreeFock matrix
    SetValue(HartreeFock, 0.0);
    DblNumMat     HartreeFockLocal(ncvband, ncvband);                         //Save Local HartreeFock matrix
    SetValue(HartreeFockLocal, 0.0);
  
    GetTime(timeSta);
    blas::Gemm('T', 'N', ncvband, ncvband, ntotLocal, 1.0, psiphiRow.Data(), ntotLocal, tempRow.Data(), ntotLocal, 0.0, HartreeFockLocal.Data(), ncvband);
    GetTime(timeEnd);
    statusOFS << "time for Hatree product          = " << timeEnd - timeSta << " [s]" << std::endl;                

    //reduce

    GetTime(timeSta);
    MPI_Allreduce(HartreeFockLocal.Data(), HartreeFock.Data(), ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
    GetTime(timeEnd);
    statusOFS << "time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;                
  }
  //end of Hartree Fock

  //get fxc potential

  //fxc setup
  DblNumMat     fxc(ncvband, ncvband);                                   //Save fxc matrix
  SetValue(fxc, 0.0);
  DblNumMat     psiphifxcR2C(ntotLocal, ncvband);                      //Save fxc wavefun
  SetValue(psiphifxcR2C, 0.0);

  //Calculatefxc operator

  DblNumVec     fxcPz(ntot);                                       //Save fxc operator
  SetValue(fxcPz, 0.0); 

  GetTime(timeSta);
  Calculatefxc(fft, fxcPz);
  GetTime(timeEnd);
  statusOFS << "Time for Calculatefxc            = " << timeEnd - timeSta << " [s]" << std::endl;

  //fxcPz send to every MPI 

  GetTime(timeSta);
  MPI_Bcast(fxcPz.Data(), ntot, MPI_DOUBLE, 0, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for MPI_Bcast               = " << timeEnd - timeSta << " [s]" << std::endl;
    
  DblNumMat     fxcLocal(ncvband, ncvband);
  SetValue(fxcLocal, 0.0);

  //Calculate fxc wavefun;
#ifdef _USE_OPENMP_
#pragma omp parallel
  {
#endif

    double facfxc = 2.0 * double(ntot_) / vol_;

    GetTime(timeSta);

#ifdef _USE_OPENMP_
#pragma omp for schedule(dynamic,1)
#endif
    for (Int mu = 0; mu < ncvband; mu++) {
      for (Int i = 0; i < ntotLocal; i++) {
        psiphifxcR2C(i, mu) = psiphiRow(i, mu) * fxcPz(i + mpirank * ntotLocal) * facfxc + tempRow(i, mu);
      }
    }//for mu  

#ifdef _USE_OPENMP_
  } //#pragma omp parallel
#endif

  GetTime(timeEnd);
  statusOFS << "Time for psiphifxcR2C            = " << timeEnd - timeSta << " [s]" << std::endl;

  //Calculate Hatree-fxc product;

  //DblNumMat     fxcLocal(ncvband, ncvband);
  //SetValue(fxcLocal, 0.0);
  GetTime(timeSta);
#ifdef GPU
  //Real ** cu_psiphiRow, cu_psiphifxcR2C, cu_fxcLocal;
  // cudaMalloc((void**)&cu_psiphiRow, sizeof(Real*) * psiphiRow.Size());
  // cudaMalloc((void**)&cu_psiphifxcR2C, sizeof(Real*) * psiphifxcR2C.Size());
  // cudaMalloc((void**)&cu_fxcLocal, sizeof(Real*) * fxcLocal.Size());
  cuDblNumMat cu_psiphiRow(ntotLocal, ncvband);  
  cuDblNumMat cu_psiphifxcR2C(ntotLocal, ncvband); 
  cuDblNumMat cu_fxcLocal(ncvband, ncvband);

  cuda_memcpy_CPU2GPU(cu_psiphiRow.Data(), psiphiRow.Data(), sizeof(Real*) * psiphiRow.Size());
  cuda_memcpy_CPU2GPU(cu_psiphifxcR2C.Data(), psiphifxcR2C.Data(), sizeof(Real*) * psiphifxcR2C.Size());
  cuda_memcpy_CPU2GPU(cu_fxcLocal.Data(), fxcLocal.Data(), sizeof(Real*) * fxcLocal.Size());

  cublas::Gemm( HIPBLAS_OP_T, HIPBLAS_OP_N, ncvband, ncvband, ntotLocal, &alpha, cu_psiphiRow.Data(), ntotLocal, cu_psiphifxcR2C.Data(), ntotLocal, &beta, cu_fxcLocal.Data(), ncvband);
  //cublasDgemm_v2(hcublas, CUBLAS_OP_T, CUBLAS_OP_N, ncvband, ncvband, ntotLocal, &alpha, cu_psiphiRow.Data(), ntotLocal, cu_psiphifxcR2C.Data(), ntotLocal, &beta, cu_fxcLocal.Data(), ncvband);

  cuda_memcpy_GPU2CPU(fxcLocal.Data(), cu_fxcLocal.Data(), sizeof(Real*) * fxcLocal.Size());
  // cudaFree(cu_psiphiRow);
  // cudaFree(cu_psiphifxcR2C);
  // cudaFree(cu_fxcLocal);
#else

  blas::Gemm('T', 'N', ncvband, ncvband, ntotLocal, 1.0, psiphiRow.Data(), ntotLocal, psiphifxcR2C.Data(), ntotLocal, 0.0, fxcLocal.Data(), ncvband);
#endif
  GetTime(timeEnd);
  statusOFS << "Time for Hatree-fxc product      = " << timeEnd - timeSta << " [s]" << std::endl;

  //#ifdef _USE_OPENMP_
  //#pragma omp critical
  //    {
  //#endif

  // This is a reduce operation for an array, and should be
  // done in the OMP critical syntax
  GetTime(timeSta);
  MPI_Allreduce(fxcLocal.Data(), fxc.Data(), ncvband, MPI_DOUBLE, MPI_SUM, domain_.comm);
  GetTime(timeEnd);
  statusOFS << "Time for MPI_Allreduce           = " << timeEnd - timeSta << " [s]" << std::endl;
  //#ifdef _USE_OPENMP_
  //    } //#pragma omp critical
  //#endif

  //end of get fxc potential

  //Diagonal part

  //Diagonal part setup
  DblNumMat Energydiff(ncvband, ncvband);
  SetValue(Energydiff, 0.0);

  kk = 0;
  GetTime(timeSta);
  for (Int k = nocc_ - nvband; k < nocc_; k++) {
    for (Int j = nocc_; j < nocc_ + ncband; j++) {      
      fxc(kk, kk) += eigVal_(j) - eigVal_(k);
      kk++;
    }//for j
  }//for k
  GetTime(timeEnd);
  statusOFS << "Time for Energydiff              = " << timeEnd - timeSta << " [s]" << std::endl;

  GetTime(timeSta);

  lapack::Lacpy( 'A', ncvband, ncvband, fxc.Data(), ncvband, hamTDDFT.Data(), ncvband );
  //blas::Axpy( ncvband, 1.0, fxc.Data(), 1, hamTDDFT.Data(), 1 );

  if(0){
    for (Int i = 0; i < ncvband; i++) {
      for (Int j = 0; j < ncvband; j++) {
        hamTDDFT(i, j) = Energydiff(i, j) + fxc(i, j);
      }
    }
  }

  if(0){
    Real *TDDFTncv = hamTDDFT.Data();
    Real *Energydiffncv = Energydiff.Data();
    Real *fxcncv = fxc.Data();
    for( Int i = 0; i < ncvband*ncvband; i++ ){
      *(TDDFTncv++) = *(Energydiffncv++) + *(fxcncv++) ;
    }
  }

  GetTime(timeEnd);

  statusOFS << "Time for hamTDDFT                = " << timeEnd - timeSta << " [s]" << std::endl;


  if(0){	

    GetTime(timeSta);
    lapack::Syevd('V', 'U', ncvband, hamTDDFT.Data(), ncvband, eigValS.Data());
    GetTime(timeEnd);
    statusOFS << "Time for Syevd in lapack         = " << timeEnd - timeSta << " [s]" << std::endl;

    //statusOFS << "eigValS = " << eigValS << std::endl;

  } //end if(0)


  if(1){

    GetTime(timeSta);

    // Setup BLACS
    Int contxt;
    Int nprow, npcol, myrow, mycol, info;
    Int scaBlockSize      = esdfParam.scaBlockSize;
    Int numProcScaLAPACK  = esdfParam.numProcScaLAPACKPW;

    for(Int i = IRound(sqrt(double(numProcScaLAPACK))); i <= numProcScaLAPACK; i++){
      nprow = i; 
      npcol = int(numProcScaLAPACK / nprow);
      if( nprow * npcol == numProcScaLAPACK ) break;
    }

    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    Cblacs_gridinfo(contxt, &nprow, &npcol, &myrow, &mycol);

    //IntNumVec pmap(numProcScaLAPACK);
    //for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
    //  pmap[i] = i;
    //}
    //Cblacs_get(0, 0, &contxt_);
    //Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);

    Int numKeep = ncvband;
    Int lda = ncvband;
    scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
    scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;
    scalapack::Descriptor descReduceSeq, descReducePar;
    Real timeEigScala_sta, timeEigScala_end;

    // Leading dimension provided
    descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt, lda );

    // Automatically comptued Leading Dimension
    descReducePar.Init( numKeep, numKeep, scaBlockSize, scaBlockSize, I_ZERO, I_ZERO, contxt );

    square_mat_scala.SetDescriptor( descReducePar );

    eigvecs_scala.SetDescriptor( descReducePar );

    DblNumMat&  square_mat = hamTDDFT;

    // Redistribute the input matrix over the process grid
    SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, 
        descReduceSeq.Values(), &square_mat_scala.LocalMatrix()[0],
        &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt );

    // Make the ScaLAPACK call

    char uplo = 'U';
    std::vector<Real> temp_eigs(lda);
    scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

    for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
      eigValS[copy_iter] = temp_eigs[copy_iter];
    }
    
    GetTime(timeEnd);
    statusOFS << "Time for Syevd in scalapack      = " << timeEnd - timeSta << " [s]" << std::endl;
    //statusOFS << "eigValS = " << eigValS << std::endl;

    if(esdfParam.isOutputExcitationWfn) {

      SetValue(square_mat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE,
          square_mat_scala.Desc().Values(), square_mat.Data(), &I_ONE, 
          &I_ONE, descReduceSeq.Values(), &contxt );

      DblNumMat excitationWfn(ntotLocal, ncvband);                       //Save wavefunproduct Row
      SetValue(excitationWfn, 0.0);

      GetTime(timeSta);
#ifdef GPU

      cuDblNumMat cu_square_mat(ncvband, ncvband); 
      cuDblNumMat cu_excitationWfn(ntotLocal, ncvband);

      cuda_memcpy_CPU2GPU(cu_psiphiRow.Data(), psiphiRow.Data(), sizeof(Real*) * fxcLocal.Size());
      cuda_memcpy_CPU2GPU(cu_square_mat.Data(), square_mat.Data(), sizeof(Real*) * square_mat.Size());
      cuda_memcpy_CPU2GPU(cu_excitationWfn.Data(), excitationWfn.Data(), sizeof(Real*) * excitationWfn.Size());

      cublas::Gemm(HIPBLAS_OP_T, HIPBLAS_OP_N, ntotLocal, ncvband, ncvband, &alpha, cu_psiphiRow.Data(), ntotLocal, cu_square_mat.Data(), ntotLocal, &beta, cu_excitationWfn.Data(), ncvband);
      //cublasDgemm_v2(hcublas, CUBLAS_OP_T, CUBLAS_OP_N, ntotLocal, ncvband, ncvband, &alpha, cu_psiphiRow.Data(), ntotLocal, cu_square_mat.Data(), ntotLocal, &beta, cu_excitationWfn.Data(), ncvband);

      cuda_memcpy_GPU2CPU(excitationWfn.Data(), cu_excitationWfn.Data(), sizeof(Real*) * fxcLocal.Size()); 
#else
      blas::Gemm('T', 'N', ntotLocal, ncvband, ncvband, 1.0, psiphiRow.Data(), ntotLocal, 
          square_mat.Data(), ncvband, 0.0, excitationWfn.Data(), ntotLocal);
#endif
      GetTime(timeEnd);
      statusOFS << "Time for excitation Wfn          = " << timeEnd - timeSta << " [s]" << std::endl;

    }

    if(contxt >= 0) {
      Cblacs_gridexit( contxt );
    }

    GetTime(timeEndTotal);
    statusOFS << std::endl << "Total time for LDA LRTDDFT       = " << timeEndTotal - timeStaTotal << " [s]" << std::endl;

    if(esdfParam.isOutputExcitationEnergy) {
      statusOFS << std::endl << "Output LRTDDFT excitation energy" <<  std::endl <<  std::endl; 
      for(Int i = 0; i < ncvband; i++){
        statusOFS << "excitation# = " << i << "      " << "eigValS = " << eigValS[i] << std::endl; 
      }
    }

  } //end if(1)

#ifdef GPU
  cublas::Destroy();
  statusOFS << std::endl << "cublas destroy..." << std::endl;
#endif
  return;

}//CalculateLRTDDFT

} //dgdft
