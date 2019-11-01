/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lin Lin, Wei Hu and Amartya Banerjee

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
/// @file scf_dg.cpp
/// @brief Self consistent iteration using the DG method.
/// @date 2013-02-05
/// @date 2014-08-06 Add intra-element parallelization.
#include  "scf_dg.hpp"
#include  "blas.hpp"
//#include  "cuda_blas.hpp"
#include  "lapack.hpp"
#include  "utility.hpp"
#ifdef ELSI
#include  "elsi.h"
#endif
// **###**
#include  "scfdg_upper_end_of_spectrum.hpp"


#include <cublas_v2.h>

namespace  dgdft{

  using namespace dgdft::DensityComponent;
  using namespace dgdft::esdf;
  using namespace dgdft::scalapack;

	//Keys and CUDA pointers to Hamiltonian matrix
	//TODO: Move to a class + add the handle
	std::vector<ElemMatKey> hKeys;
	std::vector<ElemMatKey> hamDGKeys;	
	double **h_hamDG_ptr_d;
	//Keys and CUDA pointers to X matrix
	//std::vector<Index3> pluckXKeys;	
	double *d_local_X_data;
	double **h_pluckX_ptr_d;
	double **h_pluckY_ptr_d;
	double **h_Harr_ptr_d;
	double **d_Xarr, **d_Yarr, **d_Harr;
	#define BATCH_COUNT 30

      void SCFDG::scfdg_hamiltonian_times_distmat_device(DistVec<Index3, DblNumMat, ElemPrtn>  &my_dist_mat, 
          DistVec<Index3, DblNumMat, ElemPrtn>  &Hmat_times_my_dist_mat, bool applyBatched)
      {

        Int mpirank, mpisize;
        MPI_Comm_rank( domain_.comm, &mpirank );
        MPI_Comm_size( domain_.comm, &mpisize );

        HamiltonianDG&  hamDG = *hamDGPtr_;
        std::vector<Index3>  getKeys_list;

        // Check that vectors provided only contain one entry in the local map
        // This is a safeguard to ensure that we are really dealing with distributed matrices
        if((my_dist_mat.LocalMap().size() != 1) ||
            (Hmat_times_my_dist_mat.LocalMap().size() != 1) ||
            ((my_dist_mat.LocalMap().begin())->first != (Hmat_times_my_dist_mat.LocalMap().begin())->first))
        {
          statusOFS << std::endl << " Vectors in Hmat * vector_block product not formatted correctly !!"
            << std::endl << " Aborting ... " << std::endl;
          exit(1);
        }


        // Obtain key based on my_dist_mat : This assumes that my_dist_mat is formatted correctly
        // based on processor number, etc.
        Index3 key = (my_dist_mat.LocalMap().begin())->first;

        // Obtain keys of neighbors using the Hamiltonian matrix
        // We only use these keys to minimize communication in GetBegin since other parts of the vector
        // block, even if they are non-zero will get multiplied by the zero block of the Hamiltonian
        // anyway.
        for(typename std::map<ElemMatKey, DblNumMat >::iterator 
            get_neighbors_from_Ham_iterator = hamDG.HMat().LocalMap().begin();
            get_neighbors_from_Ham_iterator != hamDG.HMat().LocalMap().end();
            get_neighbors_from_Ham_iterator ++)
        {
          Index3 neighbor_key = (get_neighbors_from_Ham_iterator->first).second;

          if(neighbor_key == key)
            continue;
          else
            getKeys_list.push_back(neighbor_key);
        }


        // Do the communication necessary to get the information from
        // procs holding the neighbors
        // Supposedly, independent row communicators (i.e. colComm)
        //  are being used for this
        my_dist_mat.GetBegin( getKeys_list, NO_MASK ); 
        my_dist_mat.GetEnd( NO_MASK );
				//if(applyBatched) {
				/*if(applyBatched) {
					statusOFS << std::endl << "---------------------BATCHED X keys and Data---------------------------" << std::endl;
				}else
					statusOFS << std::endl << "---------------------NON-BATCHED X keys and Data---------------------------" << std::endl;
					for(typename std::map<Index3, DblNumMat >::iterator
						mat_X_iterator = my_dist_mat.LocalMap().begin();
						mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++ ) {
							statusOFS << std::endl << mat_X_iterator->first << std::endl;
							DblNumMat& mat_X_local = mat_X_iterator->second;
							statusOFS << std::endl << "*****Data: " << mat_X_local.Data() << std::endl;
					}
					statusOFS << std::endl << "------------------------------------------------" << std::endl;
				*/			
				//}

				Real XDataCopy_timeSta,XDataCopy_timeEnd; 
				GetTime(XDataCopy_timeSta);
				//Copy X data to the GPU. Very expensive. Q: looks like data coming from MPI communication? Yes
				//Pack the data and make one cudaMemcpy call!!
				if(applyBatched) {
					/*
						double *h_local_X_data = (double *) malloc((my_dist_mat.LocalMap().begin())->second.Size()*sizeof(double)*my_dist_mat.LocalMap().size());
					int ix = 0;
					for(typename std::map<Index3, DblNumMat >::iterator 
            mat_X_iterator = my_dist_mat.LocalMap().begin();
            mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++ ){
							memcpy(&h_local_X_data[ix*(*mat_X_iterator).second.Size()], (*mat_X_iterator).second.Data(), (*mat_X_iterator).second.Size() * sizeof(double));
							h_pluckX_ptr_d[ix] = d_local_X_data + (ix*(*mat_X_iterator).second.Size() );
							ix++;
						}
					//cudaMemcpy(d_local_X_data, h_local_X_data, sizeof(h_local_X_data), cudaMemcpyHostToDevice);
					cudaMemcpy(d_local_X_data, h_local_X_data, (my_dist_mat.LocalMap().begin())->second.Size()*sizeof(double)*my_dist_mat.LocalMap().size(), cudaMemcpyHostToDevice);
					*/
				//	statusOFS << std::endl << "After GetBegin: " << pluckXKeys.size() << "\t" << my_dist_mat.LocalMap().size() << "\n";
					int ix = 0;
					for(typename std::map<Index3, DblNumMat >::iterator 
            mat_X_iterator = my_dist_mat.LocalMap().begin();
            mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++ ){
							//pluckXKeys.push_back(mat_X_iterator->first);	
							cudaMemcpy(h_pluckX_ptr_d[ix], (*mat_X_iterator).second.Data(), (*mat_X_iterator).second.Size() * sizeof(double), cudaMemcpyHostToDevice);
							ix++;
					} //working version
				}
				GetTime(XDataCopy_timeEnd);

        // Obtain a reference to the chunk where we want to store
        DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];
				Real original_timeSta, original_timeEnd;
				Real G_timeSta, G_timeEnd, totalTimeGEMM = 0.0;
				GetTime( original_timeSta );
        //statusOFS << std::endl << " Hadia: .......Start Original Run......\n" ;
        // Now pluck out relevant chunks of the Hamiltonian and the vector and multiply
        for(typename std::map<Index3, DblNumMat >::iterator 
            mat_X_iterator = my_dist_mat.LocalMap().begin();
            mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++ ){

          Index3 iter_key = mat_X_iterator->first;       
          DblNumMat& mat_X_local = mat_X_iterator->second; // Chunk from input block of vectors

          // Create key for looking up Hamiltonian chunk 
          ElemMatKey myelemmatkey = std::make_pair(key, iter_key);

          std::map<ElemMatKey, DblNumMat >::iterator ham_iterator = hamDG.HMat().LocalMap().find(myelemmatkey);

          //statusOFS << std::endl << " Working on key " << key << "   " << iter_key << std::endl;

          // Now do the actual multiplication
          DblNumMat& mat_H_local = ham_iterator->second; // Chunk from Hamiltonian

          Int m = mat_H_local.m(), n = mat_X_local.n(), k = mat_H_local.n();
//Chao
//statusOFS << "calling GEMM, m, n, k = " << m << " " << n << " " << k << std::endl; 
					GetTime( G_timeSta );
          blas::Gemm( 'N', 'N', m, n, k, 
              1.0, mat_H_local.Data(), m, 
              mat_X_local.Data(), k, 
              1.0, mat_Y_local.Data(), m);
					GetTime( G_timeEnd );
					totalTimeGEMM += (G_timeEnd - G_timeSta );


        } // End of loop using mat_X_iterator
				GetTime( original_timeEnd );
        //statusOFS << std::endl << " Hadia: .......End Original Run......\n" ;
//#ifdef _USE_CUDA_
        //DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];
				DblNumMat copy_mat_Y_local(mat_Y_local.m(), mat_Y_local.n());
				if(applyBatched){
					statusOFS << std::endl << " GEMM CPU in Loop Total completed. ( " << (original_timeEnd - original_timeSta ) << " s.)";
					statusOFS << std::endl << " GEMM CPU ONLY completed. ( " << (totalTimeGEMM) << " s.)";
					Real timeSta, timeEnd;	
					Real B_timeSta, B_timeEnd, R_timeSta, R_timeEnd, CPYR_timeSta, CPYR_timeEnd, Ptr_timeSta, Ptr_timeEnd, GetPtr_timeSta, GetPtr_timeEnd;
					Real HC_timeSta, HC_timeEnd, HD_timeSta, HD_timeEnd;
					//TODO: move handle creation/destruction
					cublasStatus_t stat;
					cublasHandle_t handle;
					GetTime( HC_timeSta );
					stat = cublasCreate(&handle);
					if (stat != CUBLAS_STATUS_SUCCESS) {
							statusOFS << std::endl <<"CUBLAS Handle initialization failed\n" 
								<< std::endl << " Aborting ... " << std::endl;
							exit(1);
					}
					GetTime( HC_timeEnd );

					Int Bm = mat_Y_local.m(), Bn = mat_Y_local.n(), Bk = mat_Y_local.m();
					int Bcount = my_dist_mat.LocalMap().size();
					//Create host pointer array to device matrix storage
					GetTime( GetPtr_timeSta);
					// Loop to find pointers for the matrices
					int cur = 0;
					//Q: can be done only once? Since values of key/iter_key/pointer to data are not changing across calls.
					for(typename std::map<Index3, DblNumMat >::iterator mat_X_iterator = my_dist_mat.LocalMap().begin() ;
            mat_X_iterator != my_dist_mat.LocalMap().end(); mat_X_iterator ++, cur++ ){

						Index3 iter_key = mat_X_iterator->first;       

						// Create key for looking up Hamiltonian chunk 
						ElemMatKey myelemmatkey = std::make_pair(key, iter_key);
						//statusOFS << std::endl << "H: key: " << key << "\titer_key: " << iter_key << std::endl;
						//Hadia: Find Hamiltionian matrix element position and map it to the new pointer position
						int pos = std::distance(hamDGKeys.begin(), std::find(hamDGKeys.begin(), hamDGKeys.end(), myelemmatkey));
						h_Harr_ptr_d[cur] = h_hamDG_ptr_d[pos];
						hKeys.push_back(myelemmatkey);
						
					} // End of loop using mat_X_iterator to get pointers to matrices
					GetTime( GetPtr_timeEnd);

					//Moving this out of timing, since it can be called once only. *****Need to verify 
					GetTime( Ptr_timeSta);
					//Copy host array of device pointers to the device
					//Q: can be done only once?
					cudaMemcpy(d_Harr, h_Harr_ptr_d, Bcount*sizeof(double*), cudaMemcpyHostToDevice);
					GetTime( Ptr_timeEnd);

					double alpha = 1.0;
					double beta  = 0.0;			//Hadia: Changed from 1 to 0 (to allow independent batches)
					
					GetTime( timeSta );

					GetTime( B_timeSta);
					cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
							Bm, Bn, Bk, &alpha, d_Harr, Bm, d_Xarr, Bk, &beta, d_Yarr, Bm, Bcount);
					GetTime( B_timeEnd );
					
					//reduction on the GPUs	
					GetTime( R_timeSta);
					for(int bi = 1; bi < Bcount; bi++){
							cublasDaxpy(handle, mat_Y_local.Size(), &alpha, h_pluckY_ptr_d[bi], 1, h_pluckY_ptr_d[0], 1);
							//cublasDaxpy(handle, mat_Y_local.Size(), &alpha, d_Yarr[bi], 1, d_Yarr[0], 1);
					}
					GetTime( R_timeEnd);


					GetTime( timeEnd );

					//Moving this out of timing, since we will not copy it back here since it will be used by next step.
					GetTime( CPYR_timeSta);
					cudaMemcpy(copy_mat_Y_local.Data(), h_pluckY_ptr_d[0], mat_Y_local.Size()*sizeof(double), cudaMemcpyDeviceToHost);
					//cudaMemcpy(copy_mat_Y_local.Data(), d_Yarr[0], mat_Y_local.Size()*sizeof(double), cudaMemcpyDeviceToHost);
					GetTime( CPYR_timeEnd);

					GetTime( HD_timeSta );
					cublasDestroy(handle);
					GetTime( HD_timeEnd );

					//Hadia: Compare output from both 
					//statusOFS << std::endl << " ---------------------------------- Hadia: Compare Results of Both CUDA and CPU -------------------------------\n ";
					for(int ci = 0; ci < copy_mat_Y_local.Size(); ci++){
						//statusOFS << copy_mat_Y_local.Data()[ci] << "\t" << mat_Y_local.Data()[ci] << std::endl;
						if(abs(copy_mat_Y_local.Data()[ci]-mat_Y_local.Data()[ci]) > 0.000001 ) {
							statusOFS << std::endl << " CUDA ERROR\n"  
								<< std::endl << " GEMM Data Results not the same."
								<< std::endl << " Aborting ... " << std::endl;
							exit(1);
						}
					}
					statusOFS << std::endl << " Batched Total completed. ( " << (timeEnd - timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- 1. Move Pointers from Host to Device ONLY ( " << (Ptr_timeEnd - Ptr_timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- 2. Batched ONLY ( " << (B_timeEnd - B_timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- 3. Reduction of Results (Daxpy loop) ONLY ( " << (R_timeEnd - R_timeSta ) << " s.)";
					statusOFS << std::endl << " OTHER Timings\n ----------- Handle Creation ONLY ( " << (HC_timeEnd - HC_timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- 4. Moving X data to GPU ( " << (XDataCopy_timeEnd- XDataCopy_timeSta) << " s.)";
					statusOFS << std::endl << " ----------- Loop Get Pointers ONLY ( " << (GetPtr_timeEnd - GetPtr_timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- Handle Destruction ONLY ( " << (HD_timeEnd - HD_timeSta ) << " s.)";
					statusOFS << std::endl << " ----------- Copy Result Device to Host ONLY ( " << (CPYR_timeEnd - CPYR_timeSta ) << " s.)";
					statusOFS << std::endl << "calling GEMM, m, n, k, batchCount = " << Bm << " " << Bn << " " << Bk << " " << Bcount <<  std::endl << "--------------------------------------------------------------------\n" ; 
				} // end-applyBatched
				/*
				if(applyBatched) {
					statusOFS << std::endl << "---------------------BATCHED Hamiltonian  keys and Data---------------------------" << std::endl;
				}else
					statusOFS << std::endl << "---------------------NON-BATCHED Hamiltonian keys and Data---------------------------" << std::endl;
				for( int ih = 0; ih <  hKeys.size(); ih++){
							statusOFS << std::endl << hKeys[ih].first << hKeys[ih].second << std::endl;
							statusOFS << std::endl << "*****Data: " << h_Harr_ptr_d[ih] << std::endl;
					}
				statusOFS << std::endl << "------------------------------------------------" << std::endl;
				*/
        // Matrix * vector_block product is ready now ... 
        // Need to clean up extra entries in my_dist_mat
        typename std::map<Index3, DblNumMat >::iterator it;
        for(Int delete_iter = 0; delete_iter <  getKeys_list.size(); delete_iter ++)
        {
          it = my_dist_mat.LocalMap().find(getKeys_list[delete_iter]);
          (my_dist_mat.LocalMap()).erase(it);
        }


      }

      void 
        SCFDG::scfdg_Chebyshev_filter_scaled_device(int m, 
            double a, 
            double b, 
            double a_L)
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real extra_timeSta, extra_timeEnd;
          Real filter_total_time = 0.0;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          // We need to distribute bands according to rowComm since colComm is
          // used for element-wise partition.
          if(mpisizeRow > hamDG.NumStateTotal())
          {
            statusOFS << std::endl << " Number of processors per element exceeds number of bands !!"
              << std::endl << " Cannot continue with band-parallelization. "
              << std::endl << " Aborting ... " << std::endl;
            exit(1);

          }
          simple_distributor band_distributor(hamDG.NumStateTotal(), mpisizeRow, mpirankRow);


          // Create distributed matrices pluck_X, pluck_Y, pluck_Yt for filtering 
          const Index3 key = (hamDG.EigvecCoef().LocalMap().begin())->first; // Will use same key as eigenvectors
          DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

          Int local_width = band_distributor.current_proc_size;
          Int local_height = eigvecs_local.m();
          Int local_pluck_sz = local_height * local_width;


          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_X; 
          pluck_X.Prtn() = elemPrtn_;
          pluck_X.SetComm(domain_.colComm);
          pluck_X.LocalMap()[key].Resize(local_height, local_width);

          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Y; 
          pluck_Y.Prtn() = elemPrtn_;
          pluck_Y.SetComm(domain_.colComm);
          pluck_Y.LocalMap()[key].Resize(local_height, local_width);

          DistVec<Index3, DblNumMat, ElemPrtn>  pluck_Yt; 
          pluck_Yt.Prtn() = elemPrtn_;
          pluck_Yt.SetComm(domain_.colComm);
          pluck_Yt.LocalMap()[key].Resize(local_height, local_width);


          // Initialize the distributed matrices
          blas::Copy(local_pluck_sz, 
              eigvecs_local.Data() + local_height * band_distributor.current_proc_start, 
              1,
              pluck_X.LocalMap()[key].Data(),
              1);
						//Hadia: Move pluck_X matrix to CUDA
					
//Size of the map changes after the GetEnd/GetBegin
//Preallocate size.
					//Q: Can all of this be done once and reused with each call to this scaled function? Yes.
					//TODO: Find best place to move these.
					h_pluckX_ptr_d = (double**) malloc(BATCH_COUNT*sizeof(double*));
					h_pluckY_ptr_d = (double**) malloc(BATCH_COUNT*sizeof(double*));
					h_Harr_ptr_d = (double**) malloc(BATCH_COUNT*sizeof(double*));
					//Idea: use pinned memory. Pack all the data and malloc once!!
					//Q: size of matrices fixed? Yes.
					for( int p_cur = 0; p_cur < BATCH_COUNT; p_cur++){
						//Create device memory 
						cudaMalloc((void**)&h_pluckX_ptr_d[p_cur], local_height*local_width * sizeof(double));
						cudaMalloc((void**)&h_pluckY_ptr_d[p_cur], local_height*local_width  * sizeof(double));
					}
					cudaMalloc((void**)&d_local_X_data, BATCH_COUNT*local_height*local_width * sizeof(double));
					cudaMalloc((void**)&d_Harr, BATCH_COUNT*sizeof(double*));
					cudaMalloc((void**)&d_Xarr, BATCH_COUNT*sizeof(double*));
					cudaMalloc((void**)&d_Yarr, BATCH_COUNT*sizeof(double*));
					cudaMemcpy(d_Xarr, h_pluckX_ptr_d, BATCH_COUNT*sizeof(double*), cudaMemcpyHostToDevice);
					cudaMemcpy(d_Yarr, h_pluckY_ptr_d, BATCH_COUNT*sizeof(double*), cudaMemcpyHostToDevice);

          SetValue(pluck_Y.LocalMap()[key], 0.0);
          SetValue(pluck_Yt.LocalMap()[key], 0.0);

          // Filtering scalars
          double e = (b - a) / 2.0;
          double c = (a + b) / 2.0;
          double sigma = e / (c - a_L);
          double tau = 2.0 / sigma;
          double sigma_new;

          // Begin the filtering process
          // Y = (H * X - c * X) * (sigma / e)
          // pluck_Y has been initialized to 0 already

          statusOFS << std::endl << " Chebyshev filtering : Process " << mpirank << " working on " 
            << local_width << " of " << eigvecs_local.n() << " bands.";

          statusOFS << std::endl << " Chebyshev filtering : Lower bound = " << a
            << std::endl << "                     : Upper bound = " << b
            << std::endl << "                     : a_L = " << a_L;

          //statusOFS << std::endl << " Chebyshev filtering step 1 of " << m << " ... ";
          GetTime( extra_timeSta );

					//***Hadia: This is the function that does the many mat. operations
          //scfdg_hamiltonian_times_distmat(pluck_X, pluck_Y); // Y = H * X
          scfdg_hamiltonian_times_distmat_device(pluck_X, pluck_Y, true); // Y = H * X
					free(h_pluckX_ptr_d);
					free(h_pluckY_ptr_d);
					//pluckXKeys.clear();
          scfdg_distmat_update(pluck_X, (-c) , pluck_Y,  1.0); // Y = -c * X + 1.0 * Y
          scfdg_distmat_update(pluck_Y, 0.0 , pluck_Y,  (sigma / e)); // Y = 0.0 * Y + (sigma / e) * Y

          GetTime( extra_timeEnd );
          //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
          filter_total_time += (extra_timeEnd - extra_timeSta );

          for(Int filter_iter = 2; filter_iter < m; filter_iter ++)
          {   
            //statusOFS << std::endl << " Chebyshev filtering step " << filter_iter << " of " << m << " ... ";
            GetTime( extra_timeSta );

            sigma_new = 1.0 / (tau - sigma);

            //Compute Yt = (H * Y - c * Y) * (2 * sigma_new / e) - (sigma * sigma_new) * X
            // Set Yt to 0
            SetValue(pluck_Yt.LocalMap()[key], 0.0);
            scfdg_hamiltonian_times_distmat(pluck_Y, pluck_Yt); // Yt = H * Y
            scfdg_distmat_update(pluck_Y, (-c) , pluck_Yt,  1.0); // Yt = -c * Y + 1.0 * Yt
            scfdg_distmat_update(pluck_Yt, 0.0 , pluck_Yt,  (2.0 * sigma_new / e)); // Yt = 0.0 * Yt + (2.0 * sigma_new / e) * Yt
            scfdg_distmat_update(pluck_X, (-sigma * sigma_new) , pluck_Yt,  1.0 ); // Yt = (-sigma * sigma_new) * X + 1.0 * Yt

            // Update / Re-assign : X = Y, Y = Yt, sigma = sigma_new
            scfdg_distmat_update(pluck_Y, 1.0 , pluck_X,  0.0 ); // X = 0.0 * X + 1.0 * Y
            scfdg_distmat_update(pluck_Yt, 1.0 , pluck_Y,  0.0 ); // Y = 0.0 * Y + 1.0 * Yt

            sigma = sigma_new;   

            GetTime( extra_timeEnd );
            //statusOFS << " Done. ( " << (extra_timeEnd - extra_timeSta ) << " s.)";
            filter_total_time += (extra_timeEnd - extra_timeSta );

          }

          statusOFS << std::endl <<  " Total filtering time for " 
            << m << " filter steps = " << filter_total_time << " s."
            << std::endl <<  " Average per filter step = " << ( filter_total_time / double(m) ) << " s.";

          // pluck_Y contains the results of filtering.
          // Copy back pluck_Y to the eigenvector
          // SetValue(eigvecs_local, 0.0); // All entries set to zero for All-Reduce
          GetTime( extra_timeSta );

          DblNumMat temp_buffer;
          temp_buffer.Resize(eigvecs_local.m(), eigvecs_local.n());
          SetValue(temp_buffer, 0.0);

          blas::Copy(local_pluck_sz, 
              pluck_Y.LocalMap()[key].Data(),
              1,
              temp_buffer.Data() + local_height * band_distributor.current_proc_start,
              1);

          MPI_Allreduce(temp_buffer.Data(),
              eigvecs_local.Data(),
              (eigvecs_local.m() * eigvecs_local.n()),
              MPI_DOUBLE,
              MPI_SUM,
              domain_.rowComm);

          GetTime( extra_timeEnd );
          statusOFS << std::endl << " Eigenvector block rebuild time = " 
            << (extra_timeEnd - extra_timeSta ) << " s.";





        } // End of scfdg_Chebyshev_filter


} // namespace dgdft
