/*
  Copyright (c) 2017 The Regents of the University of California,
  through Lawrence Berkeley National Laboratory.  

  Author: Amartya Banerjee

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
/// @file ex48.cpp 

#include "dgdft.hpp"


using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;



// This should be called only by prcessors sharing the context of Hmat
double test_find_comp_subspace_UB_parallel(dgdft::scalapack::ScaLAPACKMatrix<Real>& Hmat)
{
  double b_up = 0.0;  
  double alpha, beta;
  double minus_alpha, minus_beta;
    
  char uplo = 'N';
  const double scalar_one = 1.0;
  // const double scalar_minus_one = -1.0;
  const double scalar_zero = 0.0;
    
    
  const int ht = Hmat.Height();     
  const int context = Hmat.Context();
  const int scaBlockSize = Hmat.MB();
    
  // Set up v0
  dgdft::scalapack::Descriptor vec_v0_desc;
  vec_v0_desc.Init( ht, 1, 
		    scaBlockSize, scaBlockSize, 
		    0, 0, 
		    context);   
  dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_v0;
  vec_v0.SetDescriptor(vec_v0_desc);
    
  // Set up v
  dgdft::scalapack::Descriptor vec_v_desc;
  vec_v_desc.Init( ht, 1, 
		   scaBlockSize, scaBlockSize, 
		   0, 0, 
		   context);   
  dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_v;     
  vec_v.SetDescriptor(vec_v_desc);
    
  // Set up f
  dgdft::scalapack::Descriptor vec_f_desc;
  vec_f_desc.Init( ht, 1, 
		   scaBlockSize, scaBlockSize, 
		   0, 0, 
		   context);   
  dgdft::scalapack::ScaLAPACKMatrix<Real>  vec_f;    
  vec_f.SetDescriptor(vec_f_desc);

    
  // Randomly initialize vector v
  double *data_ptr =  vec_v.Data();    
  for (int ii = 0; ii < vec_v0.LocalHeight(); ii ++)
    data_ptr[ii] = UniformRandom();
    
  // Normalize this vector
  double nrm;     
  dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &nrm, vec_v.Data(), &I_ONE, &I_ONE, vec_v.Desc().Values(), &I_ONE);

  double scalar_a = 1.0 / nrm;
  dgdft::scalapack::SCALAPACK(pdscal)(&ht , &scalar_a , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values(), &I_ONE);

  // Compute f = H * v 
  dgdft::scalapack::SCALAPACK(pdgemv)(&uplo , &ht , &ht , &scalar_one , Hmat.Data() , &I_ONE , &I_ONE , Hmat.Desc().Values() , 
				      vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
				      &scalar_zero , vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE);

  // alpha = dot(f,v)
  dgdft::scalapack::SCALAPACK(pddot)(&ht , &alpha ,  vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
				     vec_v.Data() ,  &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE );
    

  // f = f - alpha * v;
  minus_alpha = -alpha;
  dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_alpha , vec_v.Data(), &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
				      vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );

    

  int Num_Lanczos_Steps = 5;
  DblNumMat mat_T(Num_Lanczos_Steps, Num_Lanczos_Steps);
  SetValue(mat_T, 0.0);

  // 0,0 entry is alpha
  mat_T(0,0) = alpha;
    
  for(Int j = 1; j < Num_Lanczos_Steps; j ++)
    {
      // beta = norm2(f)
      dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &beta, vec_f.Data(), &I_ONE, &I_ONE, vec_f.Desc().Values(), &I_ONE);
	
      // v0 = v
      dgdft::scalapack::SCALAPACK(pdcopy)(&ht , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
					  vec_v0.Data() , &I_ONE , &I_ONE , vec_v0.Desc().Values() , &I_ONE );
  
      // v = f / beta
      dgdft::scalapack::SCALAPACK(pdcopy)(&ht , vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
					  vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE ); // v = f
	
      scalar_a = (1.0 / beta);
      dgdft::scalapack::SCALAPACK(pdscal)(&ht , &scalar_a , vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values(), &I_ONE); // v <-- v (=f) / beta
	
      // f = H * v : use -H here 
      dgdft::scalapack::SCALAPACK(pdgemv)(&uplo , &ht , &ht , &scalar_one , Hmat.Data() , &I_ONE , &I_ONE , Hmat.Desc().Values() , 
					  vec_v.Data() , &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
					  &scalar_zero , vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE);


      // f = f - beta * v0
      minus_beta = -beta;
      dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_beta , vec_v0.Data(), &I_ONE , &I_ONE , vec_v0.Desc().Values() , &I_ONE , 
					  vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );
	
      // alpha = dot(f,v)
      dgdft::scalapack::SCALAPACK(pddot)(&ht , &alpha ,  vec_f.Data(), &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE , 
					 vec_v.Data() ,  &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE );

      // f = f - alpha * v;
      minus_alpha = -alpha;
      dgdft::scalapack::SCALAPACK(pdaxpy)(&ht, &minus_alpha , vec_v.Data(), &I_ONE , &I_ONE , vec_v.Desc().Values() , &I_ONE , 
					  vec_f.Data() , &I_ONE , &I_ONE , vec_f.Desc().Values() , &I_ONE );

	
      // Set up matrix entries
      mat_T(j, j - 1) = beta;
      mat_T(j - 1, j) = beta;
      mat_T(j, j) = alpha;
    
    } // End of loop over Lanczos steps 

  DblNumVec ritz_values(Num_Lanczos_Steps);
  SetValue( ritz_values, 0.0 );


  // Solve the eigenvalue problem for the Ritz values
  lapack::Syevd( 'N', 'U', Num_Lanczos_Steps, mat_T.Data(), Num_Lanczos_Steps, ritz_values.Data() );
  
  // Compute the norm of f
  dgdft::scalapack::SCALAPACK(pdnrm2)(&ht, &nrm, vec_f.Data(), &I_ONE, &I_ONE, vec_f.Desc().Values(), &I_ONE);
    
  // Finally compute upper bound
  b_up = ritz_values(Num_Lanczos_Steps - 1) + nrm;
    

  return b_up;
    
}
  
// This should be called only by prcessors sharing the context of Hmat
void  test_CheFSI_Hmat_parallel(dgdft::scalapack::ScaLAPACKMatrix<Real>& Hmat,
				dgdft::scalapack::ScaLAPACKMatrix<Real>& Xmat,
				DblNumVec& eig_vals_Xmat,
				int filter_order,
				int num_cycles,
				double lower_bound, double upper_bound, double a_L)
{
    
  const int ht = Hmat.Height();     
  const int wd = Xmat.Width();
  const int context = Hmat.Context();
  const int scaBlockSize = Hmat.MB();
    
  const int loc_sz = Xmat.LocalHeight() * Xmat.LocalWidth(); 

  double time_sta, time_end;
    
  double time_fine_sta, time_fine_end;
  double time_accu_filt = 0.0, time_accu_subspc = 0.0;
    
  double a = lower_bound;
  double b = upper_bound;
    
  // Set up Ymat
  dgdft::scalapack::Descriptor Ymat_desc;
  Ymat_desc.Init( ht, wd, 
		  scaBlockSize, scaBlockSize, 
		  0, 0, 
		  context);   
  dgdft::scalapack::ScaLAPACKMatrix<Real>  Ymat;
  Ymat.SetDescriptor(Ymat_desc);
	
  // Set up Yt_mat
  dgdft::scalapack::Descriptor Yt_mat_desc;
  Yt_mat_desc.Init( ht, wd, 
		    scaBlockSize, scaBlockSize, 
		    0, 0, 
		    context);   
  dgdft::scalapack::ScaLAPACKMatrix<Real>  Yt_mat;
  Yt_mat.SetDescriptor(Yt_mat_desc);

  statusOFS << std::endl << "  ------- " << std::endl;
    
  for(int cycle_iter = 1; cycle_iter <= num_cycles; cycle_iter ++)
    { 
      GetTime(time_sta);
	
      GetTime(time_fine_sta);
	
      statusOFS << std::endl << " Parallel CheFSI cycle iter no. " << cycle_iter << " of " << num_cycles ;
      statusOFS << std::endl << "   Filter order = " << filter_order ;
      statusOFS << std::endl << "   a = " << a << " b = " << b << " a_L = " << a_L ;
	
	
      double e = (b - a) / 2.0;
      double c = (a + b) / 2.0;
      double sigma = e / (c - a_L);
      double tau = 2.0 / sigma;

      double sigma_new;
	

      // A) Compute the filtered subspace
      // Step 1: Y = (H * X - c * X) * (sigma/e)
  
      // Compute Y = H * X 
      dgdft::scalapack::Gemm('N', 'N',
			     ht, wd, ht,
			     1.0,
			     Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
			     Xmat.Data(), I_ONE, I_ONE, Xmat.Desc().Values(),
			     0.0,
			     Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
			     context);

  
      // Compute Y = Y - c * X : Use local operations for this
      blas::Axpy( loc_sz, (-c), Xmat.Data(), 1, Ymat.Data(), 1 );
  
      // Compute Y = Y * (sigma / e) : Use local operations for this
      blas::Scal( loc_sz, (sigma / e), Ymat.Data(), 1 );
	
      // Loop over filter order
	
      for(int i = 2; i <= filter_order; i ++)
	{
	  sigma_new = 1.0 / (tau - sigma);
    
	    
	  // Step 2: Yt = (H * Y - c * Y) * (2 * sigma_new/e) - (sigma * sigma_new) * X
    
	  // Compute Yt = H * Y 
	  dgdft::scalapack::Gemm('N', 'N',
				 ht, wd, ht,
				 1.0,
				 Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
				 Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
				 0.0,
				 Yt_mat.Data(), I_ONE, I_ONE, Yt_mat.Desc().Values(),
				 context);
    
	    
	  // Compute Yt = Yt - c * Y : Use local operations for this
	  blas::Axpy( loc_sz, (-c), Ymat.Data(), 1, Yt_mat.Data(), 1 );
    
	  // Compute Yt = Yt * (2 * sigma_new / e) : Use local operations for this
	  blas::Scal( loc_sz, (2.0 * sigma_new / e), Yt_mat.Data(), 1 );
    
	  // Compute Yt = Yt - (sigma * sigma_new) * X : Use local operations for this
	  blas::Axpy( loc_sz, (-sigma * sigma_new), Xmat.Data(), 1, Yt_mat.Data(), 1 );

	  // Step 3: Update assignments
    
	    
	  // Set X = Y : Use local operations for this
	  blas::Copy( loc_sz, Ymat.Data(), 1, Xmat.Data(), 1);
    
	  // Set Y = Yt : Use local operations for this
	  blas::Copy( loc_sz, Yt_mat.Data(), 1, Ymat.Data(), 1);

	  // Set sigma = sigma_new
	  sigma = sigma_new;	   
	}
      
      GetTime(time_fine_end);
      time_accu_filt += (time_fine_end - time_fine_sta);
      
      GetTime(time_fine_sta);
	
      // B) Orthonormalize the filtered vectors
      dgdft::scalapack::Descriptor square_mat_desc;
      square_mat_desc.Init( wd, wd, 
			    scaBlockSize, scaBlockSize, 
			    0, 0, 
			    context);   
	
      dgdft::scalapack::ScaLAPACKMatrix<Real> square_mat;
      square_mat.SetDescriptor(square_mat_desc);
  
      // Compute X^T * X
      dgdft::scalapack::Syrk('U', 'T',
			     wd, ht,
			     1.0, Xmat.Data(),
			     I_ONE, I_ONE, Xmat.Desc().Values(),
			     0.0, square_mat.Data(),
			     I_ONE, I_ONE, square_mat.Desc().Values());
  
      // Compute the Cholesky factor 
      dgdft::scalapack::Potrf( 'U', square_mat);
	  
      // Solve using the Cholesky factor
      // X = X * U^{-1} is orthogonal, where U is the Cholesky factor
      dgdft::scalapack::Trsm('R', 'U', 'N', 'N', 1.0,
			     square_mat, 
			     Xmat);
	
	
      // C) Raleigh-Ritz step
      // Compute Y = H * X 
      dgdft::scalapack::Gemm('N', 'N',
			     ht, wd, ht,
			     1.0,
			     Hmat.Data(), I_ONE, I_ONE, Hmat.Desc().Values(), 
			     Xmat.Data(), I_ONE, I_ONE, Xmat.Desc().Values(),
			     0.0,
			     Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(),
			     context);

  
      // Compute X^T * HX
      dgdft::scalapack::Gemm( 'T', 'N',
			      wd, wd, ht,
			      1.0,
			      Xmat.Data(), I_ONE, I_ONE,  Xmat.Desc().Values(), 
			      Ymat.Data(), I_ONE, I_ONE,  Ymat.Desc().Values(),
			      0.0,
			      square_mat.Data(), I_ONE, I_ONE, square_mat.Desc().Values(),
			      context);


  
      // Solve the eigenvalue problem
      std::vector<Real> temp_eigen_values_vector; 	
      dgdft::scalapack::ScaLAPACKMatrix<Real>  scaZ;
  
      dgdft::scalapack::Syevd('U', square_mat, temp_eigen_values_vector, scaZ);
	
      // D) Subspace rotation step
      // Copy X to Y
      blas::Copy( loc_sz, Xmat.Data(), 1, Ymat.Data(), 1 );
	
      // X = X * Q (  Here Ymat contains X)
      dgdft::scalapack::Gemm( 'N', 'N',
			      ht, wd, wd,
			      1.0,
			      Ymat.Data(), I_ONE, I_ONE, Ymat.Desc().Values(), 
			      scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
			      0.0,
			      Xmat.Data(), I_ONE, I_ONE,  Xmat.Desc().Values(),
			      context);
	
  
      GetTime(time_fine_end);
      time_accu_subspc += (time_fine_end - time_fine_sta);
	
      // Adjust the lower filter bound for the next cycle 
      a = temp_eigen_values_vector[wd - 1];
  
      // Flip the sign of the eigenvalues
      for(int iter = 0; iter < wd; iter ++)
	eig_vals_Xmat(iter) = temp_eigen_values_vector[iter];
  
      //statusOFS << std::endl << "  CheFSI Eigenvalues in this cycle = " << eig_vals_Xmat << std::endl;
	
      GetTime(time_end);
      statusOFS << std::endl << " This CheFSI cycle finished in " << (time_end - time_sta) << " s." << std::endl;

    }
    
    
  statusOFS << std::endl << " Total time for " << filter_order * num_cycles << " filter applications = " << time_accu_filt << " s. (avg. = " 
	    << (time_accu_filt / (filter_order * num_cycles)) << " s.)";
  statusOFS << std::endl << " Total time for " << num_cycles << " subspace steps = " << time_accu_subspc << " s. (avg. = " 
	    << (time_accu_subspc / num_cycles) << " s.)";
  statusOFS << std::endl << " Standard 4 filter order cycle with 4 repeats would take " << 4 * (4 * (time_accu_filt / (filter_order * num_cycles)) + (time_accu_subspc / num_cycles)) << " s.";
    
  statusOFS << std::endl << "  ------- " << std::endl;
    
    
    
    
  return;
    
}



int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  try
    {

      Real timeSta, timeEnd;
      Real extra_timeSta, extra_timeEnd;

      // Set defaults
      Int blockSize = 32;  
    
      Int N_elem = 100;
      Int N_albs = 250;
    
      Int N = N_elem * N_albs;
    
      Int N_states =  int(N / 20); // Approx 1 states per 20 ALBs
      Int N_top =  int(N_states / 20); // Deal with only the top 5 % of states
    
    
      Int filter_order = 3;
      Int num_cycles = 3;
    
      int temp_factor = int(sqrt(double(mpisize)));
    
      while(mpisize % temp_factor != 0 )
	++ temp_factor;

      // temp_factor now contains the process grid height
      int nprow = temp_factor;      
      int npcol = mpisize / temp_factor;
    
      // By default, we choose a process grid that is long
      if(nprow < npcol)
	{
	  int temp = nprow;
	  nprow = npcol;
	  npcol = temp;
	}
    
      int print_opt = 0;	  

      // Read in some options 
      std::map<std::string,std::string> options;

      OptionsCreate(argc, argv, options);

      // Read in process grid dimensions
      if( options.find("-r") != options.end() ){
	if( options.find("-c") != options.end() ){
	  nprow= atoi(options["-r"].c_str());
	  npcol= atoi(options["-c"].c_str());
	  if(nprow*npcol > mpisize){
	    throw std::runtime_error("The number of used processors cannot be higher than the total number of available processors." );
	  } 
	}
	else{
	  throw std::runtime_error( "When using -r option, -c also needs to be provided." );
	}
      }
    
      // Read in matrix dimensions
      if( options.find("-E") != options.end() )
	N_elem = atoi(options["-E"].c_str());

      if( options.find("-A") != options.end() )
	N_albs = atoi(options["-A"].c_str());     
    
      N = N_elem * N_albs;
    
      if( options.find("-Ns") != options.end() )
	N_states = atoi(options["-Ns"].c_str());
      else
	N_states = int(N / 20);
	  
      if( options.find("-Nt") != options.end() )
	N_top = atoi(options["-Nt"].c_str());
      else
	N_top =  int(N_states / 20);
    
      // Read in ScaLAPACK block size
      if( options.find("-MB") != options.end() )
	blockSize = atoi(options["-MB"].c_str());     
    
      // CheFSI info
      if( options.find("-FO") != options.end() )
	filter_order = atoi(options["-FO"].c_str());     
    
      if( options.find("-NC") != options.end() )
	num_cycles = atoi(options["-NC"].c_str());     
    
    
      // Open output file
  
      if( mpirank == 0 ){
    
	stringstream  ss; ss << "statfile." << mpirank;
	statusOFS.open( ss.str().c_str() );
      }

      SetRandomSeed(mpirank);
    
      statusOFS << std::endl;
      statusOFS << std::endl << " mpisize = " << mpisize; 
      statusOFS << std::endl << " Process grid = " << nprow << " * " << npcol;  
      statusOFS << std::endl << " ScaLAPACK block size = " << blockSize;  

      statusOFS << std::endl;
    
      statusOFS << std::endl << " ALBs per element = " << N_albs;
      statusOFS << std::endl << " Number of elements = " << N_elem;
      statusOFS << std::endl << " Basis set size = " << N;
      statusOFS << std::endl;
    
      statusOFS << std::endl << " Number of electronic states = " << N_states;
      statusOFS << std::endl << " Number of top states = " << N_top;

      statusOFS << std::endl << std::endl << " Smallest problem size (inner subspace problem) = " << N_top << " * " << N_top;
      statusOFS << std::endl << " Note: proc_grid_rows * blocksize = " << (nprow * blockSize);
      statusOFS << std::endl << " Note: proc_grid_cols * blocksize = " << (npcol * blockSize);
    
      statusOFS << std::endl << " -----------------------" <<  std::endl;

	      
      statusOFS << std::endl << " Setting up BLACS ( " << nprow << " * " << npcol << " proc. grid ) ...";
    
      // Initialize BLACS for contxt1
      Int contxt = -1;
      Cblacs_get(0, 0, &contxt);
      Cblacs_gridinit(&contxt, "C", nprow, npcol);
    
    
      int dummy_np_row, dummy_np_col;
      int proc_grid_row, proc_grid_col;

      if(contxt >= 0)
	dgdft::scalapack::Cblacs_gridinfo(contxt, &dummy_np_row, &dummy_np_col, &proc_grid_row, &proc_grid_col);

    
      statusOFS << " Done.";
  

      // Set up ScaLAPACK matrix X
      statusOFS << std::endl << " Setting up ScaLAPACK matrix X and filling with random entries ... ";
    
      GetTime( timeSta );
    
      scalapack::Descriptor descX;
      scalapack::ScaLAPACKMatrix<Real>  scaX;
  
      if(contxt >= 0)
	{  
	  descX.Init( N, N_states, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaX.SetDescriptor(descX);
      
	  double *ptr = scaX.Data();
	  int loc_sz = scaX.LocalHeight() * scaX.LocalWidth();
    
	  for(int iter = 0; iter < loc_sz; iter ++)
	    ptr[iter] = UniformRandom();
      
	}
      
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      // Set up ScaLAPACK matrix C = X^T * X via GEMM
      statusOFS << std::endl << " Setting up matrix C via GEMM ... ";
    
      GetTime( timeSta );
    
      scalapack::Descriptor descC;
      scalapack::ScaLAPACKMatrix<Real>  scaC;
    
      if(contxt >= 0)
	{  
	  descC.Init( N_states, N_states, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaC.SetDescriptor(descC);
      
	  dgdft::scalapack::Gemm( 'T', 'N',
				  N_states, N_states, N,
				  1.0,
				  scaX.Data(), I_ONE, I_ONE,scaX.Desc().Values(), 
				  scaX.Data(), I_ONE, I_ONE,scaX.Desc().Values(),
				  0.0,
				  scaC.Data(), I_ONE, I_ONE, scaC.Desc().Values(),
				  contxt);

      
	}
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
       // Set up ScaLAPACK matrix C = X^T * X via SYRK
      statusOFS << std::endl << " Setting up matrix C via SYRK ... ";
    
      GetTime( timeSta );

      if(contxt >= 0)
	{  
	 dgdft::scalapack::Syrk('U', 'T',
				N_states, N,
				1.0, scaX.Data(),
				I_ONE, I_ONE, scaX.Desc().Values(),
				0.0, scaC.Data(),
				I_ONE, I_ONE,scaC.Desc().Values());


      
	}
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";

      
    
      // Cholesky factorization 
      statusOFS << std::endl << " Cholesky factorization of C ... ";
    
      GetTime( timeSta );
    
      if(contxt >= 0)
	{ 
	  dgdft::scalapack::Potrf( 'U', scaC);
	}
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      // TRSM
      statusOFS << std::endl << " TRSM operation ... ";
    
      GetTime( timeSta );
    
      if(contxt >= 0)
	{ 
	  dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0, 
				  scaC, 
				  scaX );
	}
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
    
      // Create a diagonal matrix
      statusOFS << std::endl << " Creating diagonal matrix ... ";
      GetTime( timeSta );
    
      scalapack::Descriptor descM;
      scalapack::ScaLAPACKMatrix<Real>  scaM;
      if(contxt >= 0)
	{  
	  descM.Init( N_states, N_states, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaM.SetDescriptor(descM);
    
	  double *ptr = scaM.Data();
	  int loc_sz = scaM.LocalHeight() * scaM.LocalWidth();
    
	  // Fill up with zeros
	  for(int iter = 0; iter < loc_sz; iter ++)
	    ptr[iter] = 0.0;
    
	  // Now get the diagonal blocks : This trick works for equal block-sizes in row and column directions
	  if(proc_grid_row == proc_grid_col)
	    {
	      int iter = 0, iter_row, iter_col;
	      for(iter_row = 0; iter_row < scaM.LocalHeight(); iter_row ++)
		{
		  for(iter_col = 0; iter_col < scaM.LocalWidth(); iter_col ++)
		    {
		      if(iter_row == iter_col)
			ptr[iter] = UniformRandom();
	  
		      iter ++;
	  
		    }
		}              
	    }     
	}
    
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
    
      // Make this diagonal matrix full
      statusOFS << std::endl << " Creating full matrix out of diagonal matrix : ";
      GetTime( timeSta );
      
       if(contxt >= 0)
	{ 
	  scalapack::Descriptor desc_small_X;
	  scalapack::ScaLAPACKMatrix<Real>  small_X;
	  
	  desc_small_X.Init( N_states, N_states, blockSize, blockSize, 
			     0, 0, contxt );
      
	  small_X.SetDescriptor(desc_small_X);
	  
	  // Fill up with random numbers
	  double *ptr = small_X.Data();
	  int loc_sz = small_X.LocalHeight() * small_X.LocalWidth();
    
	  for(int iter = 0; iter < loc_sz; iter ++)
	    ptr[iter] = UniformRandom();
      
	  // Compute the overlap
	  scalapack::Descriptor desc_alt_C;
	  scalapack::ScaLAPACKMatrix<Real>  alt_C;
	  
	  desc_alt_C.Init( N_states, N_states, blockSize, blockSize, 
			     0, 0, contxt );
      
	  alt_C.SetDescriptor(desc_alt_C);

	  dgdft::scalapack::Gemm( 'T', 'N',
				  N_states, N_states, N_states,
				  1.0,
				  small_X.Data(), I_ONE, I_ONE, small_X.Desc().Values(), 
				  small_X.Data(), I_ONE, I_ONE, small_X.Desc().Values(),
				  0.0,
				  alt_C.Data(), I_ONE, I_ONE, alt_C.Desc().Values(),
				  contxt);

	  // Cholesky factorization
	  dgdft::scalapack::Potrf( 'U', alt_C);
	  
	  // TRSM 
	  dgdft::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0, 
				  alt_C, 
				  small_X );
	  
	  
	  // Now compute small_Y = scaM * small_X 
	  scalapack::Descriptor desc_small_Y;
	  scalapack::ScaLAPACKMatrix<Real>  small_Y;
	  
	  desc_small_Y.Init( N_states, N_states, blockSize, blockSize, 
			     0, 0, contxt );
      
	  small_Y.SetDescriptor(desc_small_Y);
	  
	  dgdft::scalapack::Gemm( 'N', 'N',
				  N_states, N_states, N_states,
				  1.0,
				  scaM.Data(), I_ONE, I_ONE, scaM.Desc().Values(), 
				  small_X.Data(), I_ONE, I_ONE, small_X.Desc().Values(),
				  0.0,
				  small_Y.Data(), I_ONE, I_ONE, small_Y.Desc().Values(),
				  contxt);
	  
	  
	  // Finally, compute scaC = X^T * Y
	   dgdft::scalapack::Gemm( 'T', 'N',
				   N_states, N_states, N_states,
				   1.0,
				   small_Y.Data(), I_ONE, I_ONE, small_Y.Desc().Values(), 
				   small_X.Data(), I_ONE, I_ONE, small_X.Desc().Values(),
				   0.0,
				   scaC.Data(), I_ONE, I_ONE, scaC.Desc().Values(),
				   contxt);

	  
	  
	}
      
      
      
      GetTime( timeEnd );
      statusOFS << " Full matrix created in " << (timeEnd - timeSta) << " s.";
      
    
      // Set up a copy of matrix C 
      statusOFS << std::endl << " Setting up matrix D as copy of matrix C ... ";
    
      GetTime( timeSta );
    
      scalapack::Descriptor descD;
      scalapack::ScaLAPACKMatrix<Real>  scaD;
    
      if(contxt >= 0)
	{  
	  descD.Init( N_states, N_states, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaD.SetDescriptor(descD);
    
	  char uplo = 'A';
	  int ht = N_states;
    
	  dgdft::scalapack::SCALAPACK(pdlacpy)(&uplo, &ht, &ht,
					       scaC.Data(), &I_ONE, &I_ONE, scaC.Desc().Values(), 
					       scaD.Data(), &I_ONE, &I_ONE, scaD.Desc().Values() );

	  /*
	    statusOFS << std::endl << std::endl;
	    double *ptr = scaD.Data();
	    int iter = 0, iter_row, iter_col;
	    for(iter_row = 0; iter_row < scaD.LocalHeight(); iter_row ++)
	    {
	    for(iter_col = 0; iter_col < scaD.LocalWidth(); iter_col ++)
	    {
	    statusOFS << std::endl << " loc. row = " << iter_row << " loc. col" << iter_col << " val = " << ptr[iter];
	  
	    iter ++;
	  
	    }
	    }
	    statusOFS << std::endl << std::endl;
	  */

	}
    
    
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      statusOFS << std::endl << " -----------------------" <<  std::endl;
      GetTime( extra_timeSta );
    
      // Solve the eigenvalue problem   
      GetTime( timeSta );
      statusOFS << std::endl;
      statusOFS << std::endl << " Solving the eigenvalue problem directly for matrix C ... ";
    
      scalapack::ScaLAPACKMatrix<Real>  scaZ;
      std::vector<Real> eigen_values;
      
      if(contxt >= 0)
	{  
	  // Eigenvalue probem solution call
	  dgdft::scalapack::Syevd('U', scaC, eigen_values, scaZ);
	}
    
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      // statusOFS << std::endl << eigen_values << std::endl << std::endl;
    
      // Subspace rotation step
      GetTime( timeSta );
      statusOFS << std::endl << " Performing subspace rotation Y = X * Q ... ";
    
      scalapack::Descriptor descY;
      scalapack::ScaLAPACKMatrix<Real>  scaY;
    
      if(contxt >= 0)
	{  
          descY.Init( N, N_states, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaY.SetDescriptor(descY);
	  
	  
	  dgdft::scalapack::Gemm( 'N', 'N',
				  N, N_states, N_states,
				  1.0,
				  scaX.Data(), I_ONE, I_ONE, scaX.Desc().Values(), 
				  scaZ.Data(), I_ONE, I_ONE, scaZ.Desc().Values(),
				  0.0,
				  scaY.Data(), I_ONE, I_ONE, scaY.Desc().Values(),
				  contxt );
	}
      
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
    
      GetTime( extra_timeEnd );
      statusOFS << std::endl << std::endl << " Total time for Eig + Subspace Rotation = " << (extra_timeEnd - extra_timeSta);
    
      statusOFS << std::endl << " -----------------------" <<  std::endl;
    
      
      statusOFS << std::endl << " Setting up local blocks for timing DM calculation ... ";
      GetTime( timeSta );
      
      // Create a local matrix to represent local portion of X
      DblNumMat X_block_local(N_albs, N_states);
      UniformRandom(X_block_local);
      
      DblNumMat DM_block_local(N_albs, N_albs);
      SetValue(DM_block_local, 0.0);
      
      DblNumMat top_states_local(N_states, N_top);
      DblNumMat XC_mat_local(N_albs, N_top);
      
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      statusOFS << std::endl << " -----------------------" <<  std::endl;

    
      // Create a random block of vectors for initial guess to CheFSI 
      statusOFS << std::endl << " Setting up vector block for CheFSI and filling with random entries ... ";
      GetTime( timeSta );

      scalapack::Descriptor descV;
      scalapack::ScaLAPACKMatrix<Real>  scaV;
    
      if(contxt >= 0)
	{  
	  descV.Init( N_states, N_top, blockSize, blockSize, 
		      0, 0, contxt );
      
	  scaV.SetDescriptor(descV);
      
	  double *ptr = scaV.Data();
	  int loc_sz = scaV.LocalHeight() * scaV.LocalWidth();
    
	  for(int iter = 0; iter < loc_sz; iter ++)
	    ptr[iter] = UniformRandom();    
	}
    
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      
      // CheFSI related operations here
      GetTime( extra_timeSta );
      double ub, lb, aL;
    
      // Determine the upper bound of D
      GetTime( timeSta );
      statusOFS << std::endl << " Computing upper bound of D ... ";
    
      if(contxt >= 0)
	{
	  ub =  test_find_comp_subspace_UB_parallel(scaD);
	}
    
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      // Use CheFSI for top states
      GetTime( timeSta );
      statusOFS << std::endl << std::endl << " Computing top states using CheFSI : ";
  
      aL = eigen_values[0];
      int state_ind = N_top;        
      lb = 0.5 * (eigen_values[state_ind] + eigen_values[state_ind - 1]);
    
      DblNumVec eig_vals_Cheby;
      eig_vals_Cheby.Resize(N_top);
    
      if(contxt >= 0)
	{
	  test_CheFSI_Hmat_parallel(scaD,
				    scaV,
				    eig_vals_Cheby,
				    filter_order,
				    num_cycles,
				    lb, ub, aL);      
	}
    
    
      GetTime( timeEnd );
      statusOFS << " " << num_cycles << " CheFSI cycles completed in " << (timeEnd - timeSta) << " s." << std::endl;
      
      
      // Now do a pdgemr2d for redistributing the top states for DM computation
      GetTime( timeSta );
      statusOFS << std::endl << " Redistributing top states to single process : ";
    
      
      double time_1_sta, time_1_end;
      
      GetTime( time_1_sta );
      Int single_proc_context = -1;
      Int single_proc_pmap[1];  
      single_proc_pmap[0] = 0; // Just using proc. 0 for the job.

      // Set up BLACS for for the single proc context
      dgdft::scalapack::Cblacs_get( 0, 0, &single_proc_context );
      dgdft::scalapack::Cblacs_gridmap(&single_proc_context, &single_proc_pmap[0], 1, 1, 1);

       scalapack::Descriptor temp_single_proc_desc;
       scalapack::ScaLAPACKMatrix<Real>  temp_single_proc_scala_mat;
      
       if(single_proc_context >= 0)
        {
          temp_single_proc_desc.Init(N_states, N_top,
				     blockSize, blockSize, 
				     0, 0,  single_proc_context );              

          temp_single_proc_scala_mat.SetDescriptor( temp_single_proc_desc );
	}
	
	
	if(contxt >= 0)
	{
	  int M_ = N_states;
	  int N_ = N_top;
	  
	  int temp_context = contxt;
	  
	  SCALAPACK(pdgemr2d)(&M_, &N_, 
			      scaV.Data(), &I_ONE, &I_ONE, 
			      scaV.Desc().Values(), 
			      temp_single_proc_scala_mat.Data(), &I_ONE, &I_ONE, 
			      temp_single_proc_scala_mat.Desc().Values(), 
			      &temp_context);    	  	  
	}
	
	GetTime( time_1_end );
        statusOFS << std::endl << " pdgemr2d completed in " << (time_1_end - time_1_sta) << " s.";

	// Copy off the top states
	GetTime( time_1_sta );
	if(single_proc_context >= 0)
        {

          // Copy from the single process ScaLAPACK matrix to serial storage
          double *src_ptr, *dest_ptr; 

          // Copy in the regular order      
          for(Int copy_iter = 0; copy_iter < N_top; copy_iter ++)
          {
            src_ptr = temp_single_proc_scala_mat.Data() + copy_iter * temp_single_proc_scala_mat.LocalLDim();
            dest_ptr = top_states_local.VecData(copy_iter);

            blas::Copy( N_states, src_ptr, 1, dest_ptr, 1 );                                                 
          }

        }

        GetTime( time_1_end );
        statusOFS <<  std::endl << " Local dcopy completed in " << (time_1_end - time_1_sta) << " s.";
	
	GetTime( time_1_sta );
        // Broadcast local buffer of top vectors
        MPI_Bcast(top_states_local.Data(), N_states * N_top, 
		  MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	
	GetTime( time_1_end );
        statusOFS <<  std::endl <<" MPI Broadcast completed in " << (time_1_end - time_1_sta) << " s.";
	
	GetTime( timeEnd );
        statusOFS <<  std::endl << " Redistribution of top states completed. ( " << (timeEnd - timeSta) << " s.)"  << std::endl;

	// Now compute the extra part of DM due to CS strategy
        GetTime( timeSta );
        statusOFS << std::endl << " Computing extra part of diagonal DM blocks due to CS strategy ... ";
       
	blas::Gemm( 'N', 'N', N_albs, N_top, N_states,
                     1.0, 
                     X_block_local.Data(), X_block_local.m(), 
                     top_states_local.Data(), top_states_local.m(),
                     0.0, 
                     XC_mat_local.Data(), XC_mat_local.m());
	
	 blas::Gemm( 'N', 'T', XC_mat_local.m(), XC_mat_local.m(), XC_mat_local.n(),
                     -1.0, 
                      XC_mat_local.Data(), XC_mat_local.m(), 
                      XC_mat_local.Data(), XC_mat_local.m(),
                      1.0, 
                      DM_block_local.Data(),  DM_block_local.m());
     
              
	GetTime( timeEnd );
        statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      GetTime( extra_timeEnd );     
      statusOFS << std::endl << std::endl << " Upper bound + CheFSI + DM part = " << (extra_timeEnd - extra_timeSta);
    
    
    
      statusOFS << std::endl << " -----------------------" <<  std::endl;


      statusOFS << std::endl << std::endl << " Cleaning up BLACS ... ";
    
      if(contxt >= 0)
	{  
	  dgdft::scalapack::Cblacs_gridexit( contxt );
	}
    
  
      statusOFS << " Done." << std::endl ;

  
      statusOFS << std::endl << std::endl << " -----------------------" <<  std::endl;

    
      statusOFS << std::endl << std::endl ;

    }
  catch( std::exception& e )
    {
      std::cerr << " caught exception with message: "
		<< e.what() << std::endl;
    }

  MPI_Finalize();

  return 0;
}
