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
/// @file ex47.cpp (formerly scala_gemr2d.cpp)

#include "dgdft.hpp"

using namespace dgdft;
using namespace std;
using namespace dgdft::esdf;
using namespace dgdft::scalapack;



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
    Int blockSize = 4;  
    
    Int N = 200 ;
    Int N_s = 40 ;
    
  
    
       
    int temp_factor = int(sqrt(double(mpisize)));
    
    while(mpisize % temp_factor != 0 )
      ++ temp_factor;

    // temp_factor now contains the process grid height
    int nprow = temp_factor;      
    int npcol = mpisize / temp_factor;
    
    int nprow_smaller = 1, npcol_smaller = 1;
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
    if( options.find("-N") != options.end() )
      N = atoi(options["-N"].c_str());

    if( options.find("-Ns") != options.end() )
      N_s = atoi(options["-Ns"].c_str());     
    
    // Read in ScaLAPACK block size
    if( options.find("-MB") != options.end() )
      blockSize = atoi(options["-MB"].c_str());     
     
    // Read in smaller process grid rows
    if( options.find("-SR") != options.end() )
      nprow_smaller = atoi(options["-SR"].c_str());     
    
    // Read in smaller process grid cols
    if( options.find("-SC") != options.end() )
     npcol_smaller = atoi(options["-SC"].c_str());     
    
     // Read in print option
     if( options.find("-P") != options.end() )
     print_opt = atoi(options["-P"].c_str());     
    
    
    // Open output file
   
    //if( mpirank == 0 ){
    
     stringstream  ss; ss << "statfile." << mpirank;
     statusOFS.open( ss.str().c_str() );
    

    SetRandomSeed(mpirank);
    
    statusOFS << " mpisize = " << mpisize; 
    statusOFS << std::endl << " Matrix size = " << N << " * " << N_s 
	      << " , blockSize = " << blockSize << std::endl;
    
    statusOFS << std::endl << " -----------------------" <<  std::endl;

	      
    statusOFS << std::endl << " Setting up BLACS ( " << nprow << " * " << npcol << " proc. grid ) ...";
    
    // Initialize BLACS for contxt1
    Int contxt_1 = -1;
    Cblacs_get(0, 0, &contxt_1);
    Cblacs_gridinit(&contxt_1, "C", nprow, npcol);
    
    statusOFS << " Done.";
   

    // Set up regular ScaLAPACK matrix
    statusOFS << std::endl << " Setting up regular ScaLAPACK matrix ... ";
    
    scalapack::Descriptor descX_contxt_1;

    GetTime( timeSta );
    
    if(contxt_1 >= 0)
    {  
      descX_contxt_1.Init( N, N_s, blockSize, blockSize, 
	  0, 0, contxt_1 );
    }
    
    scalapack::ScaLAPACKMatrix<Real>  scaX_contxt_1;
    
    if(contxt_1 >= 0)
    {  
      scaX_contxt_1.SetDescriptor(descX_contxt_1);
    }
  
    GetTime( timeEnd );

    statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)" << std::endl;
    
    statusOFS << std::endl << " -----------------------" <<  std::endl;

    
    // Set up new ScaLAPACK grid
    statusOFS << std::endl << " Setting up BLACS ( " << nprow_smaller << " * " << npcol_smaller << " proc. grid ) ...";
    Int contxt_2 = -1;
    Cblacs_get(0, 0, &contxt_2);
    Cblacs_gridinit(&contxt_2, "C", nprow_smaller, npcol_smaller);

    statusOFS << " Done.";
    
    statusOFS << std::endl << " Setting up new ScaLAPACK matrix ... ";
    
    scalapack::Descriptor descX_contxt_2;

    GetTime( timeSta );
    
    if(contxt_2 >= 0)
    {
      descX_contxt_2.Init( N, N_s, blockSize, blockSize, 
        0, 0, contxt_2 );
    }
    
    scalapack::ScaLAPACKMatrix<Real>  scaX_contxt_2;
    
    if(contxt_2 >= 0)
    {  
      scaX_contxt_2.SetDescriptor(descX_contxt_2);
    }
    
    GetTime( timeEnd );

    statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)" << std::endl;
    
    statusOFS << std::endl << " -----------------------" <<  std::endl;

    
   statusOFS << std::endl << " Context 1 = " << contxt_1 ;
   statusOFS << std::endl << " Values of desc 1 = " ;
   for (int ij = 0; ij < 9 ; ij ++)
     statusOFS << descX_contxt_1.Values()[ij] << "  ";
   statusOFS << std::endl;
   
   statusOFS << std::endl << " Context 2 = " << contxt_2 ;
    statusOFS << std::endl << " Values of desc 2 = " ;
   for (int ij = 0; ij < 9 ; ij ++)
     statusOFS << descX_contxt_2.Values()[ij] << "  ";
   statusOFS << std::endl;
   
   
   statusOFS << std::endl ;

   if(contxt_1 >= 0)
   {
     statusOFS << std::endl << " Here in Context 1 ..." << std::endl;
     
     statusOFS << std::endl << " Matrix size = " << scaX_contxt_1.Height() << " * " << scaX_contxt_1.Width() << std::endl;

     
     statusOFS << std::endl << " For X on original process grid : Local matrix size on proc. " << mpirank << std::endl 
		<< " = " << scaX_contxt_1.LocalHeight() << " * " << scaX_contxt_1.LocalWidth() << std::endl;
    
      statusOFS << std::endl << " Filling up X with random entries ... " ; 
	      
      GetTime( timeSta );
	      
      double *ptr = scaX_contxt_1.Data();
      int loc_sz = scaX_contxt_1.LocalHeight() * scaX_contxt_1.LocalWidth();
    
      for(int iter = 0; iter < loc_sz; iter ++)
	ptr[iter] = UniformRandom();
   
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
     
   
      statusOFS << std::endl << std::endl << " -----------------------" <<  std::endl;
     
   }
    
    if(contxt_2 >= 0)
    {  
      statusOFS << std::endl << " Here in Context 2 ..." << std::endl;
      
      statusOFS << std::endl << " Matrix size = " << scaX_contxt_2.Height() << " * " << scaX_contxt_2.Width() << std::endl;
      
      statusOFS << std::endl << " For X on newer process grid : Local matrix size on proc. " << mpirank << std::endl 
		<< " = " << scaX_contxt_2.LocalHeight() << " * " << scaX_contxt_2.LocalWidth() << std::endl;
    
      statusOFS << std::endl << " Filling up this matrix with 0 ... " ; 
	      
      GetTime( timeSta );
	      
      double *ptr = scaX_contxt_2.Data();
      int loc_sz = scaX_contxt_2.LocalHeight() * scaX_contxt_2.LocalWidth();
    
      for(int iter = 0; iter < loc_sz; iter ++)
	ptr[iter] = 0.0;
   
      GetTime( timeEnd );

      statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
     
   
      statusOFS << std::endl << std::endl << " -----------------------" <<  std::endl;
     		
   
    }
    
  GetTime( timeSta );  
  statusOFS << std::endl << " Making the pdgemr2d call ..." ;  
    
  // Make the pdgemr2d call
  const Int M_ = N;
  const Int N_ = N_s;
  const Int global_contxt_ = scaX_contxt_1.Context();

  SCALAPACK(pdgemr2d)(&M_, &N_, scaX_contxt_1.Data(), &I_ONE, &I_ONE,
      scaX_contxt_1.Desc().Values(), 
      scaX_contxt_2.Data(), &I_ONE, &I_ONE, 
      scaX_contxt_2.Desc().Values(), &global_contxt_);    
  
  GetTime( timeEnd );

  statusOFS << "Done. ( " << (timeEnd - timeSta) << " s.)";
     
   
  statusOFS << std::endl << std::endl << " -----------------------" <<  std::endl;
     
   
   
   if(contxt_1 >= 0 && print_opt == 1)
   {
     double *ptr = scaX_contxt_1.Data();
     int loc_sz = scaX_contxt_1.LocalHeight() * scaX_contxt_1.LocalWidth();
     
     statusOFS << std::endl << " Context 1 Local matrix is : " 
			    << scaX_contxt_1.LocalHeight() 
			    << " * " << scaX_contxt_1.LocalWidth() << std::endl;
			    			   
     
      for(int iter = 0; iter < loc_sz; iter ++)
	statusOFS << std::endl << ptr[iter];
      
     statusOFS << std::endl << std::endl;
   }
   
   
   if(contxt_2 >= 0 && print_opt == 1)
   {
     double *ptr = scaX_contxt_2.Data();
     int loc_sz = scaX_contxt_2.LocalHeight() * scaX_contxt_2.LocalWidth();
     
     statusOFS << std::endl << " Context 2 Local matrix is : " 
			    << scaX_contxt_2.LocalHeight() 
			    << " * " << scaX_contxt_2.LocalWidth() << std::endl;
			    			   
     
      for(int iter = 0; iter < loc_sz; iter ++)
	statusOFS << std::endl << ptr[iter];
      
     statusOFS << std::endl << std::endl;
   }
   
   
   
   
    
    statusOFS << std::endl << std::endl << " Cleaning up BLACS ... ";
    
    if(contxt_1 >= 0)
    {  
     dgdft::scalapack::Cblacs_gridexit( contxt_1 );
    }
    
    if(contxt_2 >= 0)
    {  
      dgdft::scalapack::Cblacs_gridexit( contxt_2 );
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
