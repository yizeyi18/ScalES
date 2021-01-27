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
/// @file ex46.cpp (formerly scala_time.cp)

#include "scales.hpp"

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

    // Set defaults
    Int blockSize = 32;  
    
    Int N = 4096 ;
    Int N_s = 512 ;
    
    Int num_rep = 5;
    
    Int print_test = 0;
    
    
    double time_gemm = 0.0, time_syrk = 0.0, time_chol = 0.0, time_trsm = 0.0;
    
    int temp_factor = int(sqrt(double(mpisize)));
    
    while(mpisize % temp_factor != 0 )
      ++ temp_factor;

    // temp_factor now contains the process grid height
    int nprow = temp_factor;      
    int npcol = mpisize / temp_factor;
	  

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
    if( options.find("-N") != options.end() )
      N = atoi(options["-N"].c_str());

    if( options.find("-Ns") != options.end() )
      N_s = atoi(options["-Ns"].c_str());     
    
    // Read in ScaLAPACK block size
    if( options.find("-MB") != options.end() )
      blockSize = atoi(options["-MB"].c_str());     
     
    // Read in number of iterations
    if( options.find("-I") != options.end() )
      num_rep = atoi(options["-I"].c_str());     
    
    // Read in print test option
    if( options.find("-P") != options.end() )
     print_test = atoi(options["-P"].c_str());     
    
    
    // Open output file
   
    if( mpirank == 0 ){
      
     stringstream  ss; ss << "statfile." << mpirank;
     statusOFS.open( ss.str().c_str() );
    }

    SetRandomSeed(mpirank);
    
    statusOFS << " mpisize = " << mpisize; 
    statusOFS << std::endl << " Matrix size = " << N << " * " << N_s 
	      << " , blockSize = " << blockSize << std::endl;
    statusOFS << std::endl << " Setting up BLACS ( " << nprow << " * " << npcol << " proc. grid ) ...";
    
    
    // Initialize BLACS
    Int contxt;
    Cblacs_get(0, 0, &contxt);
    Cblacs_gridinit(&contxt, "C", nprow, npcol);
    
    statusOFS << " Done.";
   

    // Set up ScaLAPACK
    statusOFS << std::endl << " Setting up ScaLAPACK ... ";
    
    scalapack::Descriptor descB, descX, descY;


    GetTime( timeSta );
    
    descX.Init( N, N_s, blockSize, blockSize, 
        0, 0, contxt );
    descY.Init( N, N_s, blockSize, blockSize, 
        0, 0, contxt );
    descB.Init( N_s, N_s, blockSize, blockSize, 
        0, 0, contxt );
    
    
    scalapack::ScaLAPACKMatrix<Real>  scaB, scaX, scaY;
    
    scaX.SetDescriptor(descX);
    scaY.SetDescriptor(descY);
    scaB.SetDescriptor(descB);
  
    GetTime( timeEnd );

    statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)" << std::endl;
    
    statusOFS << std::endl << " -----------------------" <<  std::endl;

    
    for(int rep_iter = 1; rep_iter <= num_rep; rep_iter ++)
    {  
      statusOFS << std::endl << " Repeat iter = " << rep_iter << std::endl;
      
      // Fill up X with some random entries
      statusOFS << std::endl << " For X : Local matrix size on proc. " << mpirank << std::endl 
		<< " = " << scaX.LocalHeight() << " * " << scaX.LocalWidth() << std::endl;
    
      statusOFS << std::endl << " Filling up X with random entries ... " ; 
	      
      GetTime( timeSta );
	      
      double *ptr = scaX.Data();
      int loc_sz = scaX.LocalHeight() * scaX.LocalWidth();
    
      for(int iter = 0; iter < loc_sz; iter ++)
	ptr[iter] = UniformRandom();
   
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
     
      // Timing the GEMM operation
      statusOFS << std::endl << " Computing X^T * X via GEMM ... " ; 
      GetTime( timeSta );

      scales::scalapack::Gemm( 'T', 'N',
		  N_s, N_s, N,
		  1.0,
		  scaX.Data(), I_ONE, I_ONE,scaX.Desc().Values(), 
		  scaX.Data(), I_ONE, I_ONE,scaX.Desc().Values(),
		  0.0,
		  scaB.Data(), I_ONE, I_ONE, scaB.Desc().Values(),
		  contxt);

    
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      
      time_gemm += (timeEnd - timeSta);

      if(print_test == 1)
      {
	statusOFS << std::endl << " GEMM : " ;
	for(int iter = 0; iter < blockSize; iter ++)
	  statusOFS << std::endl << " iter = " << iter << " data = " << scaB.Data()[iter];
      }	   
      
      // Timing the SYRK operation
      statusOFS << std::endl << " Computing X^T * X via SYRK ... " ; 
      GetTime( timeSta );

       
      scales::scalapack::Syrk( 'U', 'T',
			       N_s, N,
			       1.0, scaX.Data(),
			       I_ONE, I_ONE,scaX.Desc().Values(),
			       0.0, scaB.Data(),
			       I_ONE, I_ONE,scaB.Desc().Values());
			      
 
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
    
      if(print_test == 1)
      {
	statusOFS << std::endl << " SYRK : " ;
	for(int iter = 0; iter < blockSize; iter ++)
	  statusOFS << std::endl << " iter = " << iter << " data = " << scaB.Data()[iter];
      }	      
      
      time_syrk += (timeEnd - timeSta);
      
      
      // Timing the Cholesky operation
      statusOFS << std::endl << " Computing Cholesky factorization ... " ; 
      GetTime( timeSta );
      
      scales::scalapack::Potrf( 'U', scaB);
      
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      time_chol += (timeEnd - timeSta);
           
      
      // Timing the TRSM step
      statusOFS << std::endl << " Solving using TRSM ... " ; 
      GetTime( timeSta );
      
      scales::scalapack::Trsm( 'R', 'U', 'N', 'N', 1.0,
			      scaB, scaX );
      
      GetTime( timeEnd );
      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
      
      time_trsm += (timeEnd - timeSta);
      
      statusOFS << std::endl << std::endl << " -----------------------" <<  std::endl;
 
    }
    
    statusOFS << std::endl << std::endl << " Cleaning up BLACS ... ";
    
    scales::scalapack::Cblacs_gridexit( contxt );

    statusOFS << " Done." << std::endl ;

    statusOFS << std::endl << " Summary : "; 
    statusOFS << std::endl << " mpisize = " << mpisize << " , grid = " << nprow << " * " << npcol;
    statusOFS << std::endl << " Matrix size = " << N << " * " << N_s 
	      << " , blockSize = " << blockSize << std::endl;
    
    statusOFS << std::endl << " Average GEMM time = " << time_gemm / num_rep << " s.";
    statusOFS << std::endl << " Average SYRK time = " << time_syrk / num_rep << " s.";
    statusOFS << std::endl << " Average CHOL time = " << time_chol / num_rep << " s.";
    statusOFS << std::endl << " Average TRSM time = " << time_trsm / num_rep << " s.";
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
