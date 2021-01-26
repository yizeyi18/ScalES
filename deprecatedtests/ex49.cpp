/*
  Copyright (c) 2017 The Regents of the University of California,
  through Lawrence Berkeley National Laboratory.  

  Author: Amartya Banerjee

  This file is part of ScalES. All rights reserved.

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
/// @file ex49.cpp 

#include "scales.hpp"

#ifdef ELSI
#include  "elsi.h"
#endif

using namespace scales;
using namespace std;
using namespace scales::esdf;
using namespace scales::scalapack;

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
      Int N_albs_per_elem = 250;
    
      Int N = N_elem * N_albs_per_elem;
    
      Int N_states =  int(N / 20); // Approx 1 states per 20 ALBs
    
    
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
	N_albs_per_elem = atoi(options["-A"].c_str());     
    
      N = N_elem * N_albs_per_elem;
    
      if( options.find("-Ns") != options.end() )
	N_states = atoi(options["-Ns"].c_str());
      else
	N_states = int(N / 20);
	  
    
      // Read in ScaLAPACK block size
      if( options.find("-MB") != options.end() )
	blockSize = atoi(options["-MB"].c_str());     
      
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
    
      statusOFS << std::endl << " ALBs per element = " << N_albs_per_elem;
      statusOFS << std::endl << " Number of elements = " << N_elem;
      statusOFS << std::endl << " Basis set size = " << N; 
      statusOFS << std::endl << " Number of electronic states = " << N_states;

      statusOFS << std::endl << " Note: proc_grid_rows * blocksize = " << (nprow * blockSize);
      statusOFS << std::endl << " Note: proc_grid_cols * blocksize = " << (npcol * blockSize);
    
      statusOFS << std::endl << " -----------------------" <<  std::endl;

	      
      statusOFS << std::endl << " Setting up BLACS ( " << nprow << " * " << npcol << " proc. grid ) ...";
    
      // Initialize BLACS on all procs
      Int contxt = -1;
      Cblacs_get(0, 0, &contxt);
      Cblacs_gridinit(&contxt, "C", nprow, npcol);
    
    
      int dummy_np_row, dummy_np_col;
      int proc_grid_row, proc_grid_col;

      if(contxt >= 0)
	scales::scalapack::Cblacs_gridinfo(contxt, &dummy_np_row, &dummy_np_col, &proc_grid_row, &proc_grid_col);

    
      statusOFS << " Done.";
      
      // Set up ELSI
      
      statusOFS << std::endl << " Setting up ELSI ...";
   
      Int Solver = 1;      // 1: ELPA, 2: LibSOMM 3: PEXSI for dense matrix, default to use ELPA
      Int parallelism = 1; // 1 for multi-MPIs 
      Int storage = 0;     // ELSI only support DENSE(0) 
      Int sizeH_elsi = N; 
      Int num_states_elsi = N_states;

      c_elsi_init(Solver, parallelism, storage, sizeH_elsi, (2.0 * num_states_elsi), num_states_elsi);

      // MPI setup for ELSI
      MPI_Comm newComm;
      MPI_Comm_split(MPI_COMM_WORLD, contxt, mpirank, &newComm);
      int comm = MPI_Comm_c2f(newComm);
      c_elsi_set_mpi(comm); 

      // BLACS for ELSI

      if(contxt >= 0)
          c_elsi_set_blacs(contxt, blockSize);   

      //  customize the ELSI interface to use identity matrix S
      c_elsi_customize(0, 1, 1.0E-8, 1, 0, 0); 


      // Use ELPA 2 stage solver
      c_elsi_customize_elpa(2);

      statusOFS << " Done.";
      statusOFS << std::endl << " -----------------------" <<  std::endl;

      // Set up the matrix H
      statusOFS << std::endl << " Setting up ScaLAPACK matrix H and filling with random entries: ";
    
      GetTime( timeSta );
    
      scalapack::Descriptor descH;
      scalapack::ScaLAPACKMatrix<Real>  scaH_temp;
      scalapack::ScaLAPACKMatrix<Real>  scaH;
  
      // Fill with random entries
      if(contxt >= 0)
      {  
	descH.Init( N, N, blockSize, blockSize, 
		      0, 0, contxt );
      
        scaH_temp.SetDescriptor(descH);   
	scaH.SetDescriptor(descH);
      
	double *ptr = scaH_temp.Data();
	int loc_sz = scaH_temp.LocalHeight() * scaH_temp.LocalWidth();
    
	for(int iter = 0; iter < loc_sz; iter ++)
	  ptr[iter] = UniformRandom();
      
       //ptr = scaH_temp.Data();
       //statusOFS << std::endl << " ScaH_temp = ";  	
       //for(int iter = 0; iter < loc_sz; iter ++)
       //  statusOFS << std::endl << iter << "  " << ptr[iter];
       
      // Try to symmetrize the matrix explicitly
      // Transpose the matrix : scaH = scaH_temp'

      int m_temp = N, n_temp = N;
      const double DBL_ONE = 1.0, DBL_ZERO = 0.0;
      scales::scalapack::SCALAPACK(pdtran)(&m_temp, &n_temp, &DBL_ONE,
                                          scaH_temp.Data(), &I_ONE, &I_ONE, scaH_temp.Desc().Values(),
                                          &DBL_ZERO,
                                          scaH.Data(),  &I_ONE, &I_ONE, scaH.Desc().Values());                                     
      
       //ptr = scaH.Data();
       //statusOFS << std::endl << " ScaH = ";        
       //for(int iter = 0; iter < loc_sz; iter ++)
       //  statusOFS << std::endl << iter << "  " << ptr[iter];
 

      // Compute scaH = scaH (=scaH_temp') + scaH_temp
       blas::Axpy( loc_sz, 1.0, scaH_temp.Data(), 1, scaH.Data(), 1 );
  
       //ptr = scaH.Data();
       //statusOFS << std::endl << " ScaH_symmetrized = ";        
       //for(int iter = 0; iter < loc_sz; iter ++)
       //  statusOFS << std::endl << iter << "  " << ptr[iter];
      
      
     GetTime( timeEnd );

     statusOFS << std::endl << " Symmetrized matrix prepared. ( " << (timeEnd - timeSta) << " s.)";
     
     // Diagonalize 
     scalapack::ScaLAPACKMatrix<Real>  scaH_scala, scaH_ELSI;
     scalapack::ScaLAPACKMatrix<Real>  scaZ_scala, scaZ_ELSI;

     // scaH_scala.SetDescriptor(descH);
     
     // // Copy the symmetrized matrix
     // blas::Copy(loc_sz, scaH.Data(), 1, scaH_scala.Data(), 1);
     
     // Reserve space for eigenvalues
     // std::vector<Real> eigs_scala(N);


     //statusOFS << std::endl << std::endl << " Diagonalizing using ScaLAPACK: ";
     //GetTime( timeSta );

     //scalapack::Syevd('U', scaH_scala, eigs_scala, scaZ_scala);
     

     //GetTime( timeEnd ); 
     //statusOFS << std::endl << " Done. ( " << (timeEnd - timeSta) << " s.)";
      
     statusOFS << std::endl << std::endl << " Diagonalizing using ELSI: ";

     std::vector<Real> eigs_ELSI(N);
     double * Smatrix = NULL;     
     
     int num_cycle = 3;
     double time_tot = 0.0;
     for(int iter = 1; iter <= num_cycle; iter++)
     {
      
 
      scaH_ELSI.SetDescriptor(descH);
      scaZ_ELSI.SetDescriptor(descH);
      Smatrix = NULL; 
      
      // Copy the symmetrized matrix
      blas::Copy(loc_sz, scaH.Data(), 1, scaH_ELSI.Data(), 1);

      GetTime( timeSta );
      c_elsi_ev_real(scaH_ELSI.Data(), Smatrix, eigs_ELSI.data(), scaZ_ELSI.Data()); 
      GetTime( timeEnd );

      statusOFS << std::endl << "  Diagonalization " << iter << " of " << num_cycle << " finished in " << (timeEnd - timeSta) << " s.";

      time_tot += (timeEnd - timeSta);  
      }

     
     statusOFS << std::endl << " Completed " << num_cycle << " diagonalizations in " << time_tot  << " s.";

     statusOFS << std::endl << " Average = " << (time_tot) / double(num_cycle) << " s.";

     //statusOFS << std::endl << std::endl << " Eigenvalues via ScaLAPACK and ELPA are:";
     //for(int iter=0; iter < N_states; iter ++)
      //statusOFS << std::endl << "  " << iter << "  " << eigs_scala[iter] << "   " << eigs_ELSI[iter];

     statusOFS << std::endl;     

     statusOFS << std::endl << " -----------------------" <<  std::endl;

}
      statusOFS << std::endl << std::endl << " Cleaning up BLACS and ELSI ... ";

      if(contxt >= 0)
	{  
          c_elsi_finalize();    
	  scales::scalapack::Cblacs_gridexit( contxt );
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
