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


//#include <cublas_v2.h>
#include <cuda_api_wrappers.hpp>
#include <cuda_type_wrappers.hpp>
#include <axpby.hpp>

namespace  dgdft{

  using namespace dgdft::DensityComponent;
  using namespace dgdft::esdf;
  using namespace dgdft::scalapack;

  // Global vars: TODO REMOVE!
  bool first = true;
  std::vector<Index3> XKeys;  
  std::vector<ElemMatKey> hKeys;
  std::vector<ElemMatKey> hamDGKeys;  
  DblNumMat new_Y_mat;
  DblNumMat new_X_mat;
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

        //statusOFS << std::endl << "mpirank= " << mpirank << "\tkey= " << key << std::endl;
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

        Real XDataCopy_timeSta,XDataCopy_timeEnd; 
        GetTime(XDataCopy_timeSta);
        //Copy X data to the GPU. Very expensive. Q: looks like data coming from MPI communication? Yes
        //Pack the data and make one cudaMemcpy call!!
        size_t len_buffer = my_dist_mat.LocalMap().size() * Hmat_times_my_dist_mat.LocalMap()[key].Size();
        if(applyBatched) {
          int ix=0;
          for( const auto& x : my_dist_mat.LocalMap() ) {
            Index3 iter_key = x.first; 
            if(first)
              XKeys.push_back(iter_key);
            //std::copy( x.second.Data(), x.second.Data() + x.second.Size(), ptr );
            memcpy(hamDG.h_x_ptr.data() + ix * x.second.Size(), 
                   x.second.Data(), x.second.Size()*sizeof(double));
            //statusOFS << std::endl << x.second.Data() << "\t" << x.second.Size() << "\t" << x.second.Data()+x.second.Size() << std::endl;
            //h_x_ptr += x.second.Size();
            ix++;
          }
          first = false;

          // DBWY: Copy packed data to device
          //cuda::memcpy_h2d( hamDG.pluckX_pack_d.data(), hamDG.h_x_ptr.data(), hamDG.h_x_ptr.size() );
          //cuda::memcpy_h2d( hamDG.pluckX_pack_d, hamDG.h_x_ptr );
          cuda::copy( hamDG.h_x_ptr, hamDG.pluckX_pack_d );
        }
        GetTime(XDataCopy_timeEnd);
        
        DblNumMat& mat_Y_local = Hmat_times_my_dist_mat.LocalMap()[key];

        // Obtain a reference to the chunk where we want to store
        Real original_timeSta, original_timeEnd;
        Real G_timeSta, G_timeEnd, totalTimeGEMM = 0.0;
        GetTime( original_timeSta );
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
        //DblNumMat copy_mat_X_local(mat_Y_local.m(), mat_Y_local.n());
        if(applyBatched){
          statusOFS << std::endl << " GEMM CPU in Loop Total completed. ( " << (original_timeEnd - original_timeSta ) << " s.)" << std::flush;
          statusOFS << std::endl << " GEMM CPU ONLY completed. ( " << (totalTimeGEMM) << " s.)" << std::endl << std::flush;
          Real timeSta, timeEnd;  
          Real B_timeSta, B_timeEnd, R_timeSta, R_timeEnd, CPYR_timeSta, CPYR_timeEnd, Ptr_timeSta, Ptr_timeEnd, GetPtr_timeSta, GetPtr_timeEnd;
          Real HC_timeSta, HC_timeEnd, HD_timeSta, HD_timeEnd;

          // DBWY: Create cublas handle TODO: move to HamiltonianDG
          GetTime( HC_timeSta );
          //cublas::handle handle;
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
            //hamDG.h_Harr_ptr_d.at(cur) = hamDG.h_hamDG_ptr_d.at(pos).data();
            hamDG.h_Harr_ptr_d.at(cur) = hamDG.h_hamDG_ptr_d.at(pos);
            hKeys.push_back(myelemmatkey);
            
          } // End of loop using mat_X_iterator to get pointers to matrices
          GetTime( GetPtr_timeEnd);

          //Moving this out of timing, since it can be called once only. *****Need to verify 
          //Copy host array of device pointers to the device
          //Q: can be done only once? DBWY-A: Yes!
          GetTime( Ptr_timeSta);
          cuda::memcpy_h2d( hamDG.d_Harr.data(), hamDG.h_Harr_ptr_d.data(), Bcount );
          GetTime( Ptr_timeEnd);

          double alpha = 1.0;
          double beta  = 0.0;      //Hadia: Changed from 1 to 0 (to allow independent batches)
          
          GetTime( timeSta );

          GetTime( B_timeSta);
          cublas::blas::gemm_batched( hamDG.handle, 'N', 'N', Bm, Bn, Bk, alpha,
            hamDG.d_Harr.data(), Bm, hamDG.d_Xarr.data(), Bk, beta,
            hamDG.d_Yarr.data(), Bm, Bcount );
          GetTime( B_timeEnd );
          
          //reduction on the GPUs  
          // DBWY: This will be slow, should write a kernel that adds an arbitrary number of matrices
          // using the device ptr array (on device)
          GetTime( R_timeSta);
          for(int bi = 1; bi < Bcount; bi++){
            cublas::blas::axpy( hamDG.handle, mat_Y_local.Size(), alpha, 
              hamDG.h_pluckY_ptr_d[bi], 1, hamDG.h_pluckY_ptr_d[0], 1
            );
          }
          GetTime( R_timeEnd);


          GetTime( timeEnd );

          //Moving this out of timing, since we will not copy it back here since it will be used by next step.
          GetTime( CPYR_timeSta);
          DblNumMat copy_mat_Y_local(mat_Y_local.m(), mat_Y_local.n());
          cuda::memcpy_d2h( copy_mat_Y_local.Data(), hamDG.pluckY_pack_d.data(), mat_Y_local.Size() );

          //new_Y_mat = copy_mat_Y_local;
          //int x_pos = std::distance(XKeys.begin(), std::find(XKeys.begin(), XKeys.end(), key));
          //cudaMemcpy(copy_mat_X_local.Data(), h_pluckX_ptr_d[x_pos], mat_Y_local.Size()*sizeof(double), cudaMemcpyDeviceToHost);
          /*
          new_X_mat = copy_mat_X_local;
          for(int ci = 0; ci < copy_mat_X_local.Size(); ci++)
            statusOFS << copy_mat_X_local.Data()[ci] << std::endl;
          */
          //cudaMemcpy(copy_mat_Y_local.Data(), d_Yarr[0], mat_Y_local.Size()*sizeof(double), cudaMemcpyDeviceToHost);
          GetTime( CPYR_timeEnd);

          //Hadia: Compare output from both 
          //statusOFS << std::endl << " ---------------------------------- Hadia: Compare Results of Both CUDA and CPU -------------------------------\n ";
          for(int ci = 0; ci < copy_mat_Y_local.Size(); ci++){
          //  statusOFS << copy_mat_Y_local.Data()[ci] << "\t" << Hmat_times_my_dist_mat.LocalMap()[key].Data()[ci] << std::endl;
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
          //statusOFS << "Each key= " << key << std::endl;
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
          

          size_t total_length = BATCH_COUNT*local_height*local_width;
          hamDG.pluckX_pack_d.resize( total_length );
          hamDG.pluckY_pack_d.resize( total_length );
          hamDG.h_x_ptr.resize( total_length ); // Use Pinned

          hamDG.h_Harr_ptr_d.resize( BATCH_COUNT );
          hamDG.h_pluckX_ptr_d.clear();
          hamDG.h_pluckY_ptr_d.clear();
          for( auto i = 0; i < BATCH_COUNT; ++i ) {
            hamDG.h_pluckX_ptr_d.emplace_back( 
              hamDG.pluckX_pack_d.data() + i * local_height * local_width
            );
            hamDG.h_pluckY_ptr_d.emplace_back( 
              hamDG.pluckY_pack_d.data() + i * local_height * local_width
            );
          }


          hamDG.d_Xarr.resize( BATCH_COUNT );
          hamDG.d_Yarr.resize( BATCH_COUNT );
          hamDG.d_Harr.resize( BATCH_COUNT );

          cuda::memcpy_h2d( hamDG.d_Xarr.data(), hamDG.h_pluckX_ptr_d.data(),
                            BATCH_COUNT );
          cuda::memcpy_h2d( hamDG.d_Yarr.data(), hamDG.h_pluckY_ptr_d.data(),
                            BATCH_COUNT );

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
          scfdg_hamiltonian_times_distmat_device(pluck_X, pluck_Y, true); // Y = H * X
          scfdg_distmat_update(pluck_X, (-c) , pluck_Y,  1.0); // Y = -c * X + 1.0 * Y
          int x_pos = std::distance(XKeys.begin(), std::find(XKeys.begin(), XKeys.end(), key));
          statusOFS << std::endl << "x_pos: " << x_pos << std::endl;
//#if USE_CUDA
#if 1
          statusOFS << std::endl << "Testing AXPBY" << std::endl;
          device::axpby_device(local_height * local_width, -c, hamDG.h_pluckX_ptr_d[x_pos], 1, 1.0, hamDG.h_pluckY_ptr_d[0], 1);
          DblNumMat copy_mat_Y_local(local_height, local_width);

          cuda::memcpy_d2h( copy_mat_Y_local.Data(), hamDG.pluckY_pack_d.data(), copy_mat_Y_local.Size() );

          for(int ci = 0; ci < copy_mat_Y_local.Size(); ci++){
            statusOFS << copy_mat_Y_local.Data()[ci] << "\t" << pluck_Y.LocalMap()[key].Data()[ci] << std::endl;
            if(abs(copy_mat_Y_local.Data()[ci]-pluck_Y.LocalMap()[key].Data()[ci]) > 0.000001 ) {
              statusOFS << std::endl << " AXPBY CUDA ERROR\n"  
                << std::endl << " GEMM Data Results not the same."
                << std::endl << " Aborting ... " << std::endl;
              exit(1);
            }
          }
#endif
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
      void
        SCFDG::InnerIterate_device    ( Int outerIter )
        {
          Int mpirank, mpisize;
          MPI_Comm_rank( domain_.comm, &mpirank );
          MPI_Comm_size( domain_.comm, &mpisize );
          Int mpirankRow;  MPI_Comm_rank(domain_.rowComm, &mpirankRow);
          Int mpisizeRow;  MPI_Comm_size(domain_.rowComm, &mpisizeRow);
          Int mpirankCol;  MPI_Comm_rank(domain_.colComm, &mpirankCol);
          Int mpisizeCol;  MPI_Comm_size(domain_.colComm, &mpisizeCol);

          Real timeSta, timeEnd;
          Real timeIterStart, timeIterEnd;

          HamiltonianDG&  hamDG = *hamDGPtr_;

          bool isInnerSCFConverged = false;

          for( Int innerIter = 1; innerIter <= scfInnerMaxIter_; innerIter++ ){
            if ( isInnerSCFConverged ) break;
            scfTotalInnerIter_++;

            GetTime( timeIterStart );

            statusOFS << std::endl << " Inner SCF iteration #"  
              << innerIter << " starts." << std::endl << std::endl;


            // *********************************************************************
            // Update potential and construct/update the DG matrix
            // *********************************************************************

            if( innerIter == 1 ){
              // The first inner iteration does not update the potential, and
              // construct the global Hamiltonian matrix from scratch
              GetTime(timeSta);


              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );


              hamDG.CalculateDGMatrix( );

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for constructing the DG matrix is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
            }
            else{
              // The consequent inner iterations update the potential in the
              // element, and only update the global Hamiltonian matrix

              // Update the potential in the element (and the extended element)


              GetTime(timeSta);

              // Save the old potential on the LGL grid
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                      blas::Copy( numLGLGrid.prod(),
                          hamDG.VtotLGL().LocalMap()[key].Data(), 1,
                          vtotLGLSave_.LocalMap()[key].Data(), 1 );
                    } // if (own this element)
                  } // for (i)


              // Update the local potential on the extended element and on the
              // element.
              UpdateElemLocalPotential();


              // Save the difference of the potential on the LGL grid into vtotLGLSave_
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      Index3 numLGLGrid     = hamDG.NumLGLGridElem();
                      Real *ptrNew = hamDG.VtotLGL().LocalMap()[key].Data();
                      Real *ptrDif = vtotLGLSave_.LocalMap()[key].Data();
                      for( Int p = 0; p < numLGLGrid.prod(); p++ ){
                        (*ptrDif) = (*ptrNew) - (*ptrDif);
                        ptrNew++;
                        ptrDif++;
                      } 
                    } // if (own this element)
                  } // for (i)



              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for updating the local potential in the extended element and the element is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


              // Update the DG Matrix
              GetTime(timeSta);
              hamDG.UpdateDGMatrix( vtotLGLSave_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for updating the DG matrix is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            } // if ( innerIter == 1 )
            //Hadia: Move DG Hamiltoniam matrix to CUDA


            size_t nlocal     = hamDG.HMat().LocalMap().size();
            size_t local_size = hamDG.HMat().LocalMap().begin()->second.Size();
            hamDGPtr_->localH_pack_d.resize( nlocal * local_size );

            cuda::pinned_vector<double> H_pack_h( nlocal * local_size );
            hamDGPtr_->h_hamDG_ptr_d.clear();
            hamDGPtr_->h_hamDG_ptr_d.reserve(nlocal);

            for( auto [i, mi] = std::tuple( 0ul, hamDG.HMat().LocalMap().begin() );
                 mi != hamDG.HMat().LocalMap().end(); ++mi, ++i ) {

               ElemMatKey key = (*mi).first;
               hamDGKeys.push_back(key);
               
               hamDGPtr_->h_hamDG_ptr_d.emplace_back( 
                 hamDGPtr_->localH_pack_d.data() + i*local_size
               );

               memcpy( H_pack_h.data() + i * local_size, mi->second.Data(),  
                       local_size * sizeof(double));
             }
      
             

             //cuda::memcpy_h2d( hamDG.localH_pack_d, H_pack_h );
             cuda::copy( H_pack_h, hamDG.localH_pack_d );

            //statusOFS << std::endl << "Matrix H counter: " << cur << std::endl ; //<< "size: " << (hamDG.HMat().LocalMap().begin()).second.Size() << std::endl;

#if ( _DEBUGlevel_ >= 2 )
            {
              statusOFS << "Owned H matrix blocks on this processor" << std::endl;
              for( std::map<ElemMatKey, DblNumMat>::iterator 
                  mi  = hamDG.HMat().LocalMap().begin();
                  mi != hamDG.HMat().LocalMap().end(); mi++ ){
                ElemMatKey key = (*mi).first;
                statusOFS << key.first << " -- " << key.second << std::endl;
              }
            }
#endif


            // *********************************************************************
            // Write the Hamiltonian matrix to a file (if needed) 
            // *********************************************************************

            if( esdfParam.isOutputHMatrix ){
              // Only the first processor column participates in the conversion
              if( mpirankRow == 0 ){
                DistSparseMatrix<Real>  HSparseMat;

                GetTime(timeSta);
                DistElemMatToDistSparseMat( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm );
                GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                GetTime(timeSta);
                ParaWriteDistSparseMatrix( "H.csc", HSparseMat );
                //            WriteDistSparseMatrixFormatted( "H.matrix", HSparseMat );
                GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for writing the matrix in parallel is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              MPI_Barrier( domain_.comm );

            }


            // *********************************************************************
            // Evaluate the density matrix
            // 
            // This can be done either using diagonalization method or using PEXSI
            // *********************************************************************

            // Save the mixing variable first
            {
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      if( mixVariable_ == "density" ){
                        DblNumVec& oldVec = hamDG.Density().LocalMap()[key];
                        DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
                        blas::Copy( oldVec.Size(), oldVec.Data(), 1,
                            newVec.Data(), 1 );
                      }
                      else if( mixVariable_ == "potential" ){
                        DblNumVec& oldVec = hamDG.Vtot().LocalMap()[key];
                        DblNumVec& newVec = mixInnerSave_.LocalMap()[key];
                        blas::Copy( oldVec.Size(), oldVec.Data(), 1,
                            newVec.Data(), 1 );
                      }

                    } // own this element
                  } // for (i)
            }



            // Method 1: Using diagonalization method
            // With a versatile choice of processors for using ScaLAPACK.
            // Or using Chebyshev filtering

            if( solutionMethod_ == "diag" ){
              {
                // ~~**~~
                if(Diag_SCFDG_by_Cheby_ == 1 )
                {
                  // Chebyshev filtering based diagonalization
                  GetTime(timeSta);

                  if(scfdg_ion_dyn_iter_ != 0)
                  {
                    if(SCFDG_use_comp_subspace_ == 1)
                    {

                      if((scfdg_ion_dyn_iter_ % SCFDG_CS_ioniter_regular_cheby_freq_ == 0) && (outerIter <= Second_SCFDG_ChebyOuterIter_ / 2)) // Just some adhoc criterion used here
                      {
                        // Usual CheFSI to help corrrect drift / SCF convergence
                        statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step to improve drift / SCF convergence ..." << std::endl;    

                        scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);

                        SCFDG_comp_subspace_engaged_ = 0;
                      }
                      else
                      {  
                        // Decide serial or parallel version here
                        if(SCFDG_comp_subspace_parallel_ == 0)
                        {  
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version) ...  " << std::endl;
                          scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
                        }
                        else
                        {
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version) ...  " << std::endl;
                          scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
                        }
                        // Set the engaged flag 
                        SCFDG_comp_subspace_engaged_ = 1;
                      }

                    }
                    else
                    {
                      if(outerIter <= Second_SCFDG_ChebyOuterIter_ / 2) // Just some adhoc criterion used here
                      {
                        // Need to re-use current guess, so do not call the first Cheby step
                        statusOFS << std::endl << " Calling Second stage Chebyshev Iter in iondynamics step " << std::endl;         
                        scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
                      }
                      else
                      {     
                        // Subsequent MD Steps
                        statusOFS << std::endl << " Calling General Chebyshev Iter in iondynamics step " << std::endl;
                        scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
                      }

                    }
                  } // if (scfdg_ion_dyn_iter_ != 0)
                  else
                  {    
                    // 0th MD / Geometry Optimization step (or static calculation)        
                    if(outerIter == 1)
                    {
                      statusOFS << std::endl << " Calling First Chebyshev Iter  " << std::endl;
                      scfdg_FirstChebyStep(First_SCFDG_ChebyCycleNum_, First_SCFDG_ChebyFilterOrder_);
                    }
                    else if(outerIter > 1 &&     outerIter <= Second_SCFDG_ChebyOuterIter_)
                    {
                      statusOFS << std::endl << " Calling Second Stage Chebyshev Iter  " << std::endl;
                      scfdg_GeneralChebyStep(Second_SCFDG_ChebyCycleNum_, Second_SCFDG_ChebyFilterOrder_);
                    }
                    else
                    {  
                      if(SCFDG_use_comp_subspace_ == 1)
                      {
                        // Decide serial or parallel version here
                        if(SCFDG_comp_subspace_parallel_ == 0)
                        {  
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (serial subspace version)  " << std::endl;
                          scfdg_complementary_subspace_serial(General_SCFDG_ChebyFilterOrder_);
                        }
                        else
                        {
                          statusOFS << std::endl << " Calling Complementary Subspace Strategy (parallel subspace version)  " << std::endl;
                          scfdg_complementary_subspace_parallel(General_SCFDG_ChebyFilterOrder_);                     
                        }

                        // Now set the engaged flag 
                        SCFDG_comp_subspace_engaged_ = 1;

                      }
                      else
                      {
                        statusOFS << std::endl << " Calling General Chebyshev Iter  " << std::endl;
                        scfdg_GeneralChebyStep(General_SCFDG_ChebyCycleNum_, General_SCFDG_ChebyFilterOrder_); 
                      }


                    }
                  } // end of if(scfdg_ion_dyn_iter_ != 0)


                  MPI_Barrier( domain_.comm );
                  MPI_Barrier( domain_.rowComm );
                  MPI_Barrier( domain_.colComm );

                  GetTime( timeEnd );

                  if(SCFDG_comp_subspace_engaged_ == 1)
                    statusOFS << std::endl << " Total time for Complementary Subspace Method is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  else
                    statusOFS << std::endl << " Total time for diag DG matrix via Chebyshev filtering is " << timeEnd - timeSta << " [s]" << std::endl << std::endl;



                  DblNumVec& eigval = hamDG.EigVal();          
                  //for(Int i = 0; i < hamDG.NumStateTotal(); i ++)
                  //  statusOFS << setw(8) << i << setw(20) << '\t' << eigval[i] << std::endl;

                }
               else // call the ELSI interface and old Scalapack interface
                {
                  GetTime(timeSta);

                   Int sizeH = hamDG.NumBasisTotal(); // used for the size of Hamitonian. 
                  DblNumVec& eigval = hamDG.EigVal(); 
                  eigval.Resize( hamDG.NumStateTotal() );        

                  for( Int k = 0; k < numElem_[2]; k++ )
                    for( Int j = 0; j < numElem_[1]; j++ )
                      for( Int i = 0; i < numElem_[0]; i++ ){
                        Index3 key( i, j, k );
                        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                          const std::vector<Int>&  idx = hamDG.ElemBasisIdx()(i, j, k); 
                          DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                          localCoef.Resize( idx.size(), hamDG.NumStateTotal() );        
                        }
                      } 

                  scalapack::Descriptor descH;
                  if( contxt_ >= 0 ){
                    descH.Init( sizeH, sizeH, scaBlockSize_, scaBlockSize_, 
                        0, 0, contxt_ );
                  }

                  scalapack::ScaLAPACKMatrix<Real>  scaH, scaZ;

                  std::vector<Int> mpirankElemVec(dmCol_);
                  std::vector<Int> mpirankScaVec( numProcScaLAPACK_ );

                  // The processors in the first column are the source
                  for( Int i = 0; i < dmCol_; i++ ){
                    mpirankElemVec[i] = i * dmRow_;
                  }
                  // The first numProcScaLAPACK processors are the target
                  for( Int i = 0; i < numProcScaLAPACK_; i++ ){
                    mpirankScaVec[i] = i;
                  }

#if ( _DEBUGlevel_ >= 2 )
                  statusOFS << "mpirankElemVec = " << mpirankElemVec << std::endl;
                  statusOFS << "mpirankScaVec = " << mpirankScaVec << std::endl;
#endif

                  Real timeConversionSta, timeConversionEnd;

                  GetTime( timeConversionSta );
                  DistElemMatToScaMat2( hamDG.HMat(),     descH,
                      scaH, hamDG.ElemBasisIdx(), domain_.comm,
                      domain_.colComm, mpirankElemVec,
                      mpirankScaVec );
                  GetTime( timeConversionEnd );


#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for converting from DistElemMat to ScaMat is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif
                  if(contxt_ >= 0){

                  std::vector<Real> eigs(sizeH);
                  double * Smatrix = NULL;
                  GetTime( timeConversionSta );

                  // allocate memory for the scaZ. and call ELSI: ELPA

                  if( diagSolutionMethod_ == "scalapack"){
                     scalapack::Syevd('U', scaH, eigs, scaZ);
                  }
                  else // by default to use ELPA
                  {
#ifdef ELSI
                     scaZ.SetDescriptor(scaH.Desc());
                     c_elsi_ev_real(scaH.Data(), Smatrix, &eigs[0], scaZ.Data()); 
#endif
                  }
                  
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  if( diagSolutionMethod_ == "scalapack"){
                      statusOFS << " Time for Scalapack::diag " <<
                          timeConversionEnd - timeConversionSta << " [s]" 
                          << std::endl << std::endl;
                  }
                  else
                  {
                      statusOFS << " Time for ELSI::ELPA  Diag " <<
                          timeConversionEnd - timeConversionSta << " [s]" 
                          << std::endl << std::endl;
                  }
#endif
                  for( Int i = 0; i < hamDG.NumStateTotal(); i++ )
                    eigval[i] = eigs[i];

                  } //if(contxt_ >= -1)

                  GetTime( timeConversionSta );
                  ScaMatToDistNumMat2( scaZ, hamDG.Density().Prtn(), 
                      hamDG.EigvecCoef(), hamDG.ElemBasisIdx(), domain_.comm,
                      domain_.colComm, mpirankElemVec, mpirankScaVec, 
                      hamDG.NumStateTotal() );
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for converting from ScaMat to DistNumMat is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

                  GetTime( timeConversionSta );

                  for( Int k = 0; k < numElem_[2]; k++ )
                    for( Int j = 0; j < numElem_[1]; j++ )
                      for( Int i = 0; i < numElem_[0]; i++ ){
                        Index3 key( i, j, k );
                        if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                          DblNumMat& localCoef  = hamDG.EigvecCoef().LocalMap()[key];
                          MPI_Bcast(localCoef.Data(), localCoef.m() * localCoef.n(), MPI_DOUBLE, 0, domain_.rowComm);
                        }
                      } 
                  GetTime( timeConversionEnd );
#if ( _DEBUGlevel_ >= 1 )
                  statusOFS << " Time for MPI_Bcast eigval and localCoef is " <<
                    timeConversionEnd - timeConversionSta << " [s]" << std::endl << std::endl;
#endif

                  MPI_Barrier( domain_.comm );
                  MPI_Barrier( domain_.rowComm );
                  MPI_Barrier( domain_.colComm );

                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  if( diagSolutionMethod_ == "scalapack"){
                  statusOFS << " Time for diag DG matrix via ScaLAPACK is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  }
                  else{
                  statusOFS << " Time for diag DG matrix via ELSI:ELPA is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
                  }
#endif

                  // Communicate the eigenvalues
                  Int mpirankScaSta = mpirankScaVec[0];
                  MPI_Bcast(eigval.Data(), hamDG.NumStateTotal(), MPI_DOUBLE, 
                      mpirankScaVec[0], domain_.comm);


                } // End of ELSI

              }// End of diagonalization routines

              // Post processing

              Evdw_ = 0.0;

              if(SCFDG_comp_subspace_engaged_ == 1)
              {
                // Calculate Harris energy without computing the occupations
                CalculateHarrisEnergy();

              }        
              else
              {        


                // Compute the occupation rate - specific smearing types dealt with within this function
                CalculateOccupationRate( hamDG.EigVal(), hamDG.OccupationRate() );

                // Compute the Harris energy functional.  
                // NOTE: In computing the Harris energy, the density and the
                // potential must be the INPUT density and potential without ANY
                // update.
                CalculateHarrisEnergy();
              }

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );



              // Calculate the new electron density

              // ~~**~~
              GetTime( timeSta );

              if(SCFDG_comp_subspace_engaged_ == 1)
              {
                // Density calculation for complementary subspace method
                statusOFS << std::endl << " Using complementary subspace method for electron density ... " << std::endl;

                Real GetTime_extra_sta, GetTime_extra_end;          
    Real GetTime_fine_sta, GetTime_fine_end;
    
                GetTime(GetTime_extra_sta);
                statusOFS << std::endl << " Forming diagonal blocks of density matrix : ";
                GetTime(GetTime_fine_sta);
    
                // Compute the diagonal blocks of the density matrix
                DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
                cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
                cheby_diag_dmat.SetComm(domain_.colComm);

                // Copy eigenvectors to temp bufer
                DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

                DblNumMat temp_local_eig_vec;
                temp_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
                blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, temp_local_eig_vec.Data(), 1);

                // First compute the X*X^T portion
                // Multiply out to obtain diagonal block of density matrix
                ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
                cheby_diag_dmat.LocalMap()[diag_block_key].Resize( temp_local_eig_vec.m(),  temp_local_eig_vec.m());

                blas::Gemm( 'N', 'T', temp_local_eig_vec.m(), temp_local_eig_vec.m(), temp_local_eig_vec.n(),
                    1.0, 
                    temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
                    temp_local_eig_vec.Data(), temp_local_eig_vec.m(),
                    0.0, 
                    cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());

                GetTime(GetTime_fine_end);
    statusOFS << std::endl << " X * X^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";
    
    GetTime(GetTime_fine_sta);
                if(SCFDG_comp_subspace_N_solve_ != 0)
    {
      // Now compute the X * C portion
                 DblNumMat XC_mat;
                 XC_mat.Resize(eigvecs_local.m(), SCFDG_comp_subspace_N_solve_);

                 blas::Gemm( 'N', 'N', temp_local_eig_vec.m(), SCFDG_comp_subspace_N_solve_, temp_local_eig_vec.n(),
                             1.0, 
                             temp_local_eig_vec.Data(), temp_local_eig_vec.m(), 
                             SCFDG_comp_subspace_matC_.Data(), SCFDG_comp_subspace_matC_.m(),
                             0.0, 
                             XC_mat.Data(),  XC_mat.m());

                 // Subtract XC*XC^T from DM
                 blas::Gemm( 'N', 'T', XC_mat.m(), XC_mat.m(), XC_mat.n(),
                            -1.0, 
                             XC_mat.Data(), XC_mat.m(), 
                             XC_mat.Data(), XC_mat.m(),
                             1.0, 
                             cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  temp_local_eig_vec.m());
    }
                GetTime(GetTime_fine_end);
    statusOFS << std::endl << " X*C and XC * (XC)^T computed in " << (GetTime_fine_end - GetTime_fine_sta) << " s.";
                
                
                GetTime(GetTime_extra_end);
                statusOFS << std::endl << " Total time for computing diagonal blocks of DM = " << (GetTime_extra_end - GetTime_extra_sta)  << " s." << std::endl ;
                statusOFS << std::endl;

                // Make the call evaluate this on the real space grid 
                hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );


              }        
              else
              {        

                int temp_m = hamDG.NumBasisTotal() / (numElem_[0] * numElem_[1] * numElem_[2]); // Average no. of ALBs per element
                int temp_n = hamDG.NumStateTotal();
                if((Diag_SCFDG_by_Cheby_ == 1) && (temp_m < temp_n))
                {  
                  statusOFS << std::endl << " Using alternate routine for electron density: " << std::endl;

                  Real GetTime_extra_sta, GetTime_extra_end;                
                  GetTime(GetTime_extra_sta);
                  statusOFS << std::endl << " Forming diagonal blocks of density matrix ... ";

                  // Compute the diagonal blocks of the density matrix
                  DistVec<ElemMatKey, NumMat<Real>, ElemMatPrtn> cheby_diag_dmat;  
                  cheby_diag_dmat.Prtn()     = hamDG.HMat().Prtn();
                  cheby_diag_dmat.SetComm(domain_.colComm);

                  // Copy eigenvectors to temp bufer
                  DblNumMat &eigvecs_local = (hamDG.EigvecCoef().LocalMap().begin())->second;

                  DblNumMat scal_local_eig_vec;
                  scal_local_eig_vec.Resize(eigvecs_local.m(), eigvecs_local.n());
                  blas::Copy((eigvecs_local.m() * eigvecs_local.n()), eigvecs_local.Data(), 1, scal_local_eig_vec.Data(), 1);

                  // Scale temp buffer by occupation square root
                  for(int iter_scale = 0; iter_scale < eigvecs_local.n(); iter_scale ++)
                  {
                    blas::Scal(  scal_local_eig_vec.m(),  sqrt(hamDG.OccupationRate()[iter_scale]), scal_local_eig_vec.Data() + iter_scale * scal_local_eig_vec.m(), 1 );
                  }

                  // Multiply out to obtain diagonal block of density matrix
                  ElemMatKey diag_block_key = std::make_pair(my_cheby_eig_vec_key_, my_cheby_eig_vec_key_);
                  cheby_diag_dmat.LocalMap()[diag_block_key].Resize( scal_local_eig_vec.m(),  scal_local_eig_vec.m());

                  blas::Gemm( 'N', 'T', scal_local_eig_vec.m(), scal_local_eig_vec.m(), scal_local_eig_vec.n(),
                      1.0, 
                      scal_local_eig_vec.Data(), scal_local_eig_vec.m(), 
                      scal_local_eig_vec.Data(), scal_local_eig_vec.m(),
                      0.0, 
                      cheby_diag_dmat.LocalMap()[diag_block_key].Data(),  scal_local_eig_vec.m());

                  GetTime(GetTime_extra_end);
                  statusOFS << " Done. ( " << (GetTime_extra_end - GetTime_extra_sta)  << " s) " << std::endl ;

                  // Make the call evaluate this on the real space grid 
                  hamDG.CalculateDensityDM2(hamDG.Density(), hamDG.DensityLGL(), cheby_diag_dmat );
                }
                else
                {  

                  // FIXME 
                  // Do not need the conversion from column to row partition as well
                  hamDG.CalculateDensity( hamDG.Density(), hamDG.DensityLGL() );

                  // 2016/11/20: Add filtering of the density. Impacts
                  // convergence at the order of 1e-5 for the LiH dimer
                  // example and therefore is not activated
                  if(0){
                    DistFourier& fft = *distfftPtr_;
                    Int ntot      = fft.numGridTotal;
                    Int ntotLocal = fft.numGridLocal;

                    DblNumVec  tempVecLocal;
                    DistNumVecToDistRowVec(
                        hamDG.Density(),
                        tempVecLocal,
                        domain_.numGridFine,
                        numElem_,
                        fft.localNzStart,
                        fft.localNz,
                        fft.isInGrid,
                        domain_.colComm );

                    if( fft.isInGrid ){
                      for( Int i = 0; i < ntotLocal; i++ ){
                        fft.inputComplexVecLocal(i) = Complex( 
                            tempVecLocal(i), 0.0 );
                      }

                      fftw_execute( fft.forwardPlan );

                      // Filter out high frequency modes
                      for( Int i = 0; i < ntotLocal; i++ ){
                        if( fft.gkkLocal(i) > std::pow(densityGridFactor_,2.0) * ecutWavefunction_ ){
                          fft.outputComplexVecLocal(i) = Z_ZERO;
                        }
                      }

                      fftw_execute( fft.backwardPlan );


                      for( Int i = 0; i < ntotLocal; i++ ){
                        tempVecLocal(i) = fft.inputComplexVecLocal(i).real() / ntot;
                      }
                    }

                    DistRowVecToDistNumVec( 
                        tempVecLocal,
                        hamDG.Density(),
                        domain_.numGridFine,
                        numElem_,
                        fft.localNzStart,
                        fft.localNz,
                        fft.isInGrid,
                        domain_.colComm );


                    // Compute the sum of density and normalize again.
                    Real sumRhoLocal = 0.0, sumRho = 0.0;
                    for( Int k = 0; k < numElem_[2]; k++ )
                      for( Int j = 0; j < numElem_[1]; j++ )
                        for( Int i = 0; i < numElem_[0]; i++ ){
                          Index3 key( i, j, k );
                          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                            DblNumVec& localRho = hamDG.Density().LocalMap()[key];

                            Real* ptrRho = localRho.Data();
                            for( Int p = 0; p < localRho.Size(); p++ ){
                              sumRhoLocal += ptrRho[p];
                            }
                          }
                        }

                    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
                    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
                    statusOFS << std::endl;
                    Print( statusOFS, "Sum Rho on uniform grid (after Fourier filtering) = ", sumRho );
                    statusOFS << std::endl;
#endif
                    Real fac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;
                    sumRhoLocal = 0.0, sumRho = 0.0;
                    for( Int k = 0; k < numElem_[2]; k++ )
                      for( Int j = 0; j < numElem_[1]; j++ )
                        for( Int i = 0; i < numElem_[0]; i++ ){
                          Index3 key( i, j, k );
                          if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                            DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                            blas::Scal(  localRho.Size(),  fac, localRho.Data(), 1 );

                            Real* ptrRho = localRho.Data();
                            for( Int p = 0; p < localRho.Size(); p++ ){
                              sumRhoLocal += ptrRho[p];
                            }
                          }
                        }

                    sumRhoLocal *= domain_.Volume() / domain_.NumGridTotalFine(); 
                    mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );

#if ( _DEBUGlevel_ >= 0 )
                    statusOFS << std::endl;
                    Print( statusOFS, "Sum Rho on uniform grid (after normalization again) = ", sumRho );
                    statusOFS << std::endl;
#endif
                  }
                }

              }        

              MPI_Barrier( domain_.comm );
              MPI_Barrier( domain_.rowComm );
              MPI_Barrier( domain_.colComm );

              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Update the output potential, and the KS and second order accurate
              // energy
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                GetTime( timeSta );
                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Exc in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                GetTime( timeSta );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Vhart in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                // Compute the second order accurate energy functional.

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                CalculateSecondOrderEnergy();

                // Compute the KS energy 

                GetTime( timeSta );

                CalculateKSEnergy();

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing KSEnergy in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                GetTime( timeSta );

                hamDG.CalculateVtot( hamDG.Vtot() );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for computing Vtot in the global domain is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Compute the force at every step
              if( esdfParam.isCalculateForceEachSCF ){

                // Compute force
                GetTime( timeSta );

                if(SCFDG_comp_subspace_engaged_ == false)
                {
                  if(1)
                  {
                    statusOFS << std::endl << " Computing forces using eigenvectors ... " << std::endl;
                    hamDG.CalculateForce( *distfftPtr_ );
                  }
                  else
                  {         
                    // Alternate (highly unusual) routine for debugging purposes
                    // Compute the Full DM (from eigenvectors) and call the PEXSI force evaluator

                    double extra_timeSta, extra_timeEnd;

                    statusOFS << std::endl << " Computing forces using Density Matrix ... ";
                    statusOFS << std::endl << " Computing full Density Matrix from eigenvectors ...";
                    GetTime(extra_timeSta);

                    distDMMat_.Prtn()     = hamDG.HMat().Prtn();

                    // Compute the full DM 
                    scfdg_compute_fullDM();

                    GetTime(extra_timeEnd);

                    statusOFS << std::endl << " Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

                    // Call the PEXSI force evaluator
                    hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
                  }



                }
                else
                {
                  double extra_timeSta, extra_timeEnd;

                  statusOFS << std::endl << " Computing forces using Density Matrix ... ";

                  statusOFS << std::endl << " Computing full Density Matrix for Complementary Subspace method ...";
                  GetTime(extra_timeSta);

                  // Compute the full DM in the complementary subspace method
                  scfdg_complementary_subspace_compute_fullDM();

                  GetTime(extra_timeEnd);

                  statusOFS << std::endl << " DM Computation took " << (extra_timeEnd - extra_timeSta) << " s." << std::endl;

                  // Call the PEXSI force evaluator
                  hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );        
                }



                GetTime( timeEnd );
                statusOFS << " Time for computing the force is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;

                // Print out the force
                // Only master processor output information containing all atoms
                if( mpirank == 0 ){
                  PrintBlock( statusOFS, "Atomic Force" );
                  {
                    Point3 forceCM(0.0, 0.0, 0.0);
                    std::vector<Atom>& atomList = hamDG.AtomList();
                    Int numAtom = atomList.size();
                    for( Int a = 0; a < numAtom; a++ ){
                      Print( statusOFS, "atom", a, "force", atomList[a].force );
                      forceCM += atomList[a].force;
                    }
                    statusOFS << std::endl;
                    Print( statusOFS, "force for centroid: ", forceCM );
                    statusOFS << std::endl;
                  }
                }
              }

              // Compute the a posteriori error estimator at every step
              // FIXME This is not used when intra-element parallelization is
              // used.
              if( esdfParam.isCalculateAPosterioriEachSCF && 0 )
              {
                GetTime( timeSta );
                DblNumTns  eta2Total, eta2Residual, eta2GradJump, eta2Jump;
                hamDG.CalculateAPosterioriError( 
                    eta2Total, eta2Residual, eta2GradJump, eta2Jump );
                GetTime( timeEnd );
                statusOFS << " Time for computing the a posteriori error is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;

                // Only master processor output information containing all atoms
                if( mpirank == 0 ){
                  PrintBlock( statusOFS, "A Posteriori error" );
                  {
                    statusOFS << std::endl << "Total a posteriori error:" << std::endl;
                    statusOFS << eta2Total << std::endl;
                    statusOFS << std::endl << "Residual term:" << std::endl;
                    statusOFS << eta2Residual << std::endl;
                    statusOFS << std::endl << "Jump of gradient term:" << std::endl;
                    statusOFS << eta2GradJump << std::endl;
                    statusOFS << std::endl << "Jump of function value term:" << std::endl;
                    statusOFS << eta2Jump << std::endl;
                  }
                }
              }
            }


            // Method 2: Using the pole expansion and selected inversion (PEXSI) method
            // FIXME Currently it is assumed that all processors used by DG will be used by PEXSI.
#ifdef _USE_PEXSI_
/*
            // The following version is with intra-element parallelization
            if( solutionMethod_ == "pexsi" ){
              Real timePEXSISta, timePEXSIEnd;
              GetTime( timePEXSISta );

              Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
              Real muMinInertia, muMaxInertia;
              Real muPEXSI, numElectronPEXSI;
              Int numTotalInertiaIter = 0, numTotalPEXSIIter = 0;

              std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

              // FIXME 
              // Currently, only the first processor column participate in the
              // communication between PEXSI and DGDFT For the first processor
              // column involved in PEXSI, the first numProcPEXSICommCol_
              // processors are involved in the data communication between PEXSI
              // and DGDFT

              for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
                mpirankSparseVec[i] = i;
              }

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

              Int info;

              // Temporary matrices 
              DistSparseMatrix<Real>  HSparseMat;
              DistSparseMatrix<Real>  DMSparseMat;
              DistSparseMatrix<Real>  EDMSparseMat;
              DistSparseMatrix<Real>  FDMSparseMat;

              if( mpirankRow == 0 ){

                // Convert the DG matrix into the distributed CSC format

                GetTime(timeSta);
                DistElemMatToDistSparseMat3( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
                if( mpirankCol < numProcPEXSICommCol_ ){
                  statusOFS << "H.size = " << HSparseMat.size << std::endl;
                  statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
                  statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
                  statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
                  statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
                  statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
                }
#endif
              }


              if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) ){
                // Load the matrices into PEXSI.  
                // Only the processors with mpirankCol == 0 need to carry the
                // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
                statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
                statusOFS << "mpirankRow = " << mpirankRow << std::endl;
                statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif


                GetTime( timeSta );
                PPEXSILoadRealHSMatrix(
                    pexsiPlan_,
                    pexsiOptions_,
                    HSparseMat.size,
                    HSparseMat.nnz,
                    HSparseMat.nnzLocal,
                    HSparseMat.colptrLocal.m() - 1,
                    HSparseMat.colptrLocal.Data(),
                    HSparseMat.rowindLocal.Data(),
                    HSparseMat.nzvalLocal.Data(),
                    1,  // isSIdentity
                    NULL,
                    &info );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for loading the matrix into PEXSI is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI loading H matrix returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // PEXSI solver

                {
                  if( outerIter >= inertiaCountSteps_ ){
                    pexsiOptions_.isInertiaCount = 0;
                  }
                  // Note: Heuristics strategy for dynamically adjusting the
                  // tolerance
                  pexsiOptions_.muInertiaTolerance = 
                    std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.01 );
                  pexsiOptions_.numElectronPEXSITolerance = 
                    std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

                  // Only perform symbolic factorization for the first outer SCF. 
                  // Reuse the previous Fermi energy as the initial guess for mu.
                  if( outerIter == 1 ){
                    pexsiOptions_.isSymbolicFactorize = 1;
                    pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
                  }
                  else{
                    pexsiOptions_.isSymbolicFactorize = 0;
                    pexsiOptions_.mu0 = fermi_;
                  }

                  statusOFS << std::endl 
                    << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
                    << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
                    << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
                }


                GetTime( timeSta );
                // Old version of PEXSI driver, uses inertia counting + Newton's iteration
                if(0){
                  PPEXSIDFTDriver(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &numTotalPEXSIIter,
                      &info );
                }

                // New version of PEXSI driver, uses inertia count + pole update
                // strategy. No Newton's iteration
                if(1){
                  PPEXSIDFTDriver2(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &info );
                }
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for the main PEXSI Driver is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI main driver returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // Update the fermi level 
                fermi_ = muPEXSI;

                // Heuristics for the next step
                pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
                pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

                // Retrieve the PEXSI data

                if( mpirankRow == 0 ){
                  Real totalEnergyH, totalEnergyS, totalFreeEnergy;

                  GetTime( timeSta );

                  CopyPattern( HSparseMat, DMSparseMat );
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );

                  PPEXSIRetrieveRealDFTMatrix(
                      pexsiPlan_,
                      DMSparseMat.nzvalLocal.Data(),
                      EDMSparseMat.nzvalLocal.Data(),
                      FDMSparseMat.nzvalLocal.Data(),
                      &totalEnergyH,
                      &totalEnergyS,
                      &totalFreeEnergy,
                      &info );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << "Time for retrieving PEXSI data is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                  statusOFS << std::endl
                    << "Results obtained from PEXSI:" << std::endl
                    << "Total energy (H*DM)         = " << totalEnergyH << std::endl
                    << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
                    << "Total free energy           = " << totalFreeEnergy << std::endl 
                    << "InertiaIter                 = " << numTotalInertiaIter << std::endl
                    << "PEXSIIter                   = " <<  numTotalPEXSIIter << std::endl
                    << "mu                          = " << muPEXSI << std::endl
                    << "numElectron                 = " << numElectronPEXSI << std::endl 
                    << std::endl;

                  if( info != 0 ){
                    std::ostringstream msg;
                    msg 
                      << "PEXSI data retrieval returns info " << info << std::endl;
                    ErrorHandling( msg.str().c_str() );
                  }
                }
              } // if( mpirank < numProcTotalPEXSI_ )

              // Broadcast the Fermi level
              MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );

              if( mpirankRow == 0 )
              {
                GetTime(timeSta);
                // Convert the density matrix from DistSparseMatrix format to the
                // DistElemMat format
                DistSparseMatToDistElemMat3(
                    DMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    EDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distEDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the free energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    FDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distFDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "Time for converting the DistSparseMatrices to DistElemMat " << 
                  "for post-processing is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Broadcast the distElemMat matrices
              // FIXME this is not a memory efficient implementation
              GetTime(timeSta);
              {
                Int sstrSize;
                std::vector<char> sstr;
                if( mpirankRow == 0 ){
                  std::stringstream distElemMatStream;
                  Int cnt = 0;
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    cnt++;
                  } // for (mi)
                  serialize( cnt, distElemMatStream, NO_MASK );
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    ElemMatKey key = (*mi).first;
                    serialize( key, distElemMatStream, NO_MASK );
                    serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );
                    serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                    serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                  } // for (mi)
                  sstr.resize( Size( distElemMatStream ) );
                  distElemMatStream.read( &sstr[0], sstr.size() );
                  sstrSize = sstr.size();
                }

                MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
                sstr.resize( sstrSize );
                MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

                if( mpirankRow != 0 ){
                  std::stringstream distElemMatStream;
                  distElemMatStream.write( &sstr[0], sstrSize );
                  Int cnt;
                  deserialize( cnt, distElemMatStream, NO_MASK );
                  for( Int i = 0; i < cnt; i++ ){
                    ElemMatKey key;
                    DblNumMat mat;
                    deserialize( key, distElemMatStream, NO_MASK );
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distEDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distFDMMat_.LocalMap()[key] = mat;
                  } // for (mi)
                }
              }
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for broadcasting the density matrix for post-processing is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              Evdw_ = 0.0;

              // Compute the Harris energy functional.  
              // NOTE: In computing the Harris energy, the density and the
              // potential must be the INPUT density and potential without ANY
              // update.
              GetTime( timeSta );
              CalculateHarrisEnergyDM( distFDMMat_ );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for calculating the Harris energy is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Evaluate the electron density

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              GetTime( timeSta );
              hamDG.CalculateDensityDM2( 
                  hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              // Update the output potential, and the KS and second order accurate
              // energy
              GetTime(timeSta);
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << "Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                //        CalculateSecondOrderEnergy();

                // Compute the KS energy 
                CalculateKSEnergyDM( 
                    distEDMMat_, distFDMMat_ );

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                hamDG.CalculateVtot( hamDG.Vtot() );

              }
              GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for computing the potential is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Compute the force at every step
              //      if( esdfParam.isCalculateForceEachSCF ){
              //        // Compute force
              //        GetTime( timeSta );
              //        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
              //        GetTime( timeEnd );
              //        statusOFS << "Time for computing the force is " <<
              //          timeEnd - timeSta << " [s]" << std::endl << std::endl;
              //
              //        // Print out the force
              //        // Only master processor output information containing all atoms
              //        if( mpirank == 0 ){
              //          PrintBlock( statusOFS, "Atomic Force" );
              //          {
              //            Point3 forceCM(0.0, 0.0, 0.0);
              //            std::vector<Atom>& atomList = hamDG.AtomList();
              //            Int numAtom = atomList.size();
              //            for( Int a = 0; a < numAtom; a++ ){
              //              Print( statusOFS, "atom", a, "force", atomList[a].force );
              //              forceCM += atomList[a].force;
              //            }
              //            statusOFS << std::endl;
              //            Print( statusOFS, "force for centroid: ", forceCM );
              //            statusOFS << std::endl;
              //          }
              //        }
              //      }

              // TODO Evaluate the a posteriori error estimator

              GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << "Time for PEXSI evaluation is " <<
                timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
            } //if( solutionMethod_ == "pexsi" )
*/

            // The following version is with intra-element parallelization
            DistDblNumVec VtotHist; // check check
      // check check
      Real difNumElectron = 0.0;
            if( solutionMethod_ == "pexsi" ){
//chao
statusOFS << "Chao: pexsi starts from here! " << std::endl;

            // Initialize the history of vtot , check check
            for( Int k=0; k< numElem_[2]; k++ )
              for( Int j=0; j< numElem_[1]; j++ )
                for( Int i=0; i< numElem_[0]; i++ ) {
                  Index3 key = Index3(i,j,k);
                  if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
                    DistDblNumVec& vtotCur = hamDG.Vtot();
                    VtotHist.LocalMap()[key] = vtotCur.LocalMap()[key];
                    //VtotHist.LocalMap()[key] = mixInnerSave_.LocalMap()[key];
                  } // owns this element
                } // for (i)



              Real timePEXSISta, timePEXSIEnd;
              GetTime( timePEXSISta );

              Real numElectronExact = hamDG.NumOccupiedState() * hamDG.NumSpin();
              Real muMinInertia, muMaxInertia;
              Real muPEXSI, numElectronPEXSI;
              Int numTotalInertiaIter = 0, numTotalPEXSIIter = 0;

              std::vector<Int> mpirankSparseVec( numProcPEXSICommCol_ );

              // FIXME 
              // Currently, only the first processor column participate in the
              // communication between PEXSI and DGDFT For the first processor
              // column involved in PEXSI, the first numProcPEXSICommCol_
              // processors are involved in the data communication between PEXSI
              // and DGDFT

              for( Int i = 0; i < numProcPEXSICommCol_; i++ ){
                mpirankSparseVec[i] = i;
              }

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << "mpirankSparseVec = " << mpirankSparseVec << std::endl;
#endif

              Int info;

              // Temporary matrices 
              DistSparseMatrix<Real>  HSparseMat;
              DistSparseMatrix<Real>  DMSparseMat;
              DistSparseMatrix<Real>  EDMSparseMat;
              DistSparseMatrix<Real>  FDMSparseMat;

              if( mpirankRow == 0 ){

                // Convert the DG matrix into the distributed CSC format

                GetTime(timeSta);
                DistElemMatToDistSparseMat3( 
                    hamDG.HMat(),
                    hamDG.NumBasisTotal(),
                    HSparseMat,
                    hamDG.ElemBasisIdx(),
                    domain_.colComm,
                    mpirankSparseVec );
                GetTime(timeEnd);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DG matrix to DistSparseMatrix format is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif


#if ( _DEBUGlevel_ >= 0 )
                if( mpirankCol < numProcPEXSICommCol_ ){
                  statusOFS << "H.size = " << HSparseMat.size << std::endl;
                  statusOFS << "H.nnz  = " << HSparseMat.nnz << std::endl;
                  statusOFS << "H.nnzLocal  = " << HSparseMat.nnzLocal << std::endl;
                  statusOFS << "H.colptrLocal.m() = " << HSparseMat.colptrLocal.m() << std::endl;
                  statusOFS << "H.rowindLocal.m() = " << HSparseMat.rowindLocal.m() << std::endl;
                  statusOFS << "H.nzvalLocal.m() = " << HSparseMat.nzvalLocal.m() << std::endl;
                }
#endif
              }

              // So energy must be obtained from DM as in totalEnergyH
              // and free energy is nothing but energy..
              Real totalEnergyH, totalFreeEnergy;
              if( (mpirankRow < numProcPEXSICommRow_) && (mpirankCol < numProcPEXSICommCol_) )
              {

                // Load the matrices into PEXSI.  
                // Only the processors with mpirankCol == 0 need to carry the
                // nonzero values of HSparseMat

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << "numProcPEXSICommRow_ = " << numProcPEXSICommRow_ << std::endl;
                statusOFS << "numProcPEXSICommCol_ = " << numProcPEXSICommCol_ << std::endl;
                statusOFS << "mpirankRow = " << mpirankRow << std::endl;
                statusOFS << "mpirankCol = " << mpirankCol << std::endl;
#endif


                GetTime( timeSta );

#ifndef ELSI                
                PPEXSILoadRealHSMatrix(
                    pexsiPlan_,
                    pexsiOptions_,
                    HSparseMat.size,
                    HSparseMat.nnz,
                    HSparseMat.nnzLocal,
                    HSparseMat.colptrLocal.m() - 1,
                    HSparseMat.colptrLocal.Data(),
                    HSparseMat.rowindLocal.Data(),
                    HSparseMat.nzvalLocal.Data(),
                    1,  // isSIdentity
                    NULL,
                    &info );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for loading the matrix into PEXSI is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI loading H matrix returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }
#endif           

                // PEXSI solver

                {
                  if( outerIter >= inertiaCountSteps_ ){
                    pexsiOptions_.isInertiaCount = 0;
                  }
                  // Note: Heuristics strategy for dynamically adjusting the
                  // tolerance
                  pexsiOptions_.muInertiaTolerance = 
                    std::min( std::max( muInertiaToleranceTarget_, 0.1 * scfOuterNorm_ ), 0.01 );
                  pexsiOptions_.numElectronPEXSITolerance = 
                    std::min( std::max( numElectronPEXSIToleranceTarget_, 1.0 * scfOuterNorm_ ), 0.5 );

                  // Only perform symbolic factorization for the first outer SCF. 
                  // Reuse the previous Fermi energy as the initial guess for mu.
                  if( outerIter == 1 ){
                    pexsiOptions_.isSymbolicFactorize = 1;
                    pexsiOptions_.mu0 = 0.5 * (pexsiOptions_.muMin0 + pexsiOptions_.muMax0);
                  }
                  else{
                    pexsiOptions_.isSymbolicFactorize = 0;
                    pexsiOptions_.mu0 = fermi_;
                  }

                  statusOFS << std::endl 
                    << "muInertiaTolerance        = " << pexsiOptions_.muInertiaTolerance << std::endl
                    << "numElectronPEXSITolerance = " << pexsiOptions_.numElectronPEXSITolerance << std::endl
                    << "Symbolic factorization    =  " << pexsiOptions_.isSymbolicFactorize << std::endl;
                }
#ifdef ELSI
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << std::endl << "ELSI PEXSI set sparsity start" << std::endl<< std::flush;
#endif
#endif

                // ///////////////////////////////////////////////////////////////////////
                // ///////////////////////////////////////////////////////////////////////
#ifdef ELSI
                c_elsi_set_sparsity( HSparseMat.nnz,
                                   HSparseMat.nnzLocal,
                                   HSparseMat.colptrLocal.m() - 1,
                                   HSparseMat.rowindLocal.Data(),
                                   HSparseMat.colptrLocal.Data() );

                c_elsi_customize_pexsi(pexsiOptions_.temperature,
                                       pexsiOptions_.gap,
                                       pexsiOptions_.deltaE,
                                       pexsiOptions_.numPole,
                                       numProcPEXSICommCol_,  // # n_procs_per_pole
                                       pexsiOptions_.maxPEXSIIter,
                                       pexsiOptions_.muMin0,
                                       pexsiOptions_.muMax0,
                                       pexsiOptions_.mu0,
                                       pexsiOptions_.muInertiaTolerance,
                                       pexsiOptions_.muInertiaExpansion,
                                       pexsiOptions_.muPEXSISafeGuard,
                                       pexsiOptions_.numElectronPEXSITolerance,
                                       pexsiOptions_.matrixType,
                                       pexsiOptions_.isSymbolicFactorize,
                                       pexsiOptions_.ordering,
                                       pexsiOptions_.npSymbFact,
                                       pexsiOptions_.verbosity);

#if ( _DEBUGlevel_ >= 0 )
                statusOFS << std::endl << "ELSI PEXSI Customize Done " << std::endl;
#endif

                if( mpirankRow == 0 )
                   CopyPattern( HSparseMat, DMSparseMat );
                statusOFS << std::endl << "ELSI PEXSI Copy pattern done" << std::endl;
                c_elsi_dm_real_sparse(HSparseMat.nzvalLocal.Data(), NULL, DMSparseMat.nzvalLocal.Data());

                GetTime( timeEnd );
                statusOFS << std::endl << "ELSI PEXSI real sparse done" << std::endl;

                if( mpirankRow == 0 ){
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );
                  c_elsi_collect_pexsi(&fermi_,EDMSparseMat.nzvalLocal.Data(),FDMSparseMat.nzvalLocal.Data());
                  statusOFS << std::endl << "ELSI PEXSI collecte done " << std::endl;
                }
                statusOFS << std::endl << "Time for ELSI PEXSI = " << 
                       timeEnd - timeSta << " [s]" << std::endl << std::endl<<std::flush;

#endif

#ifndef ELSI
                GetTime( timeSta );
                // Old version of PEXSI driver, uses inertia counting + Newton's iteration
                if(1){
                  PPEXSIDFTDriver(
                      pexsiPlan_,
                      pexsiOptions_,
                      numElectronExact,
                      &muPEXSI,
                      &numElectronPEXSI,         
                      &muMinInertia,              
                      &muMaxInertia,             
                      &numTotalInertiaIter,
                      &numTotalPEXSIIter,
                      &info );
                }

                // New version of PEXSI driver, uses inertia count + pole update
                // strategy. No Newton's iteration. But this is not very stable.
//                if(0){
//                  PPEXSIDFTDriver2(
//                      pexsiPlan_,
//                      pexsiOptions_,
//                      numElectronExact,
//                      &muPEXSI,
//                      &numElectronPEXSI,         
//                      &muMinInertia,              
//                      &muMaxInertia,             
//                      &numTotalInertiaIter,
//                      &info );
//                }

                // New version of PEXSI driver, use inertia count + pole update.
                // two method of pole expansion. default is 2
                int method = esdfParam.pexsiMethod;
                int npoint = esdfParam.pexsiNpoint;

//                if(0){
//                  PPEXSIDFTDriver3(
//                      pexsiPlan_,
//                      pexsiOptions_,
//                      numElectronExact,
//                      method,
//                      npoint,
//                      &muPEXSI,
//                      &numElectronPEXSI,         
//                      &pexsiOptions_.muMin0,              
//                      &pexsiOptions_.muMax0,             
//                      &numTotalInertiaIter,
//                      &info );
//                }
 
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for the main PEXSI Driver is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                if( info != 0 ){
                  std::ostringstream msg;
                  msg 
                    << "PEXSI main driver returns info " << info << std::endl;
                  ErrorHandling( msg.str().c_str() );
                }

                // Update the fermi level 
                fermi_ = muPEXSI;
                difNumElectron = std::abs(numElectronPEXSI - numElectronExact);

                // Heuristics for the next step
                //pexsiOptions_.muMin0 = muMinInertia - 5.0 * pexsiOptions_.temperature;
                //pexsiOptions_.muMax0 = muMaxInertia + 5.0 * pexsiOptions_.temperature;

                // Retrieve the PEXSI data

                // FIXME: Hack: in PEXSIDriver3, only DM is available.

                if( mpirankRow == 0 ){
                  Real totalEnergyS;

                  GetTime( timeSta );

                  CopyPattern( HSparseMat, DMSparseMat );
                  CopyPattern( HSparseMat, EDMSparseMat );
                  CopyPattern( HSparseMat, FDMSparseMat );

                  statusOFS << "Before retrieve" << std::endl;
                  PPEXSIRetrieveRealDFTMatrix(
                      pexsiPlan_,
                      DMSparseMat.nzvalLocal.Data(),
                      EDMSparseMat.nzvalLocal.Data(),
                      FDMSparseMat.nzvalLocal.Data(),
                      &totalEnergyH,
                      &totalEnergyS,
                      &totalFreeEnergy,
                      &info );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for retrieving PEXSI data is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

                  // FIXME: Hack: there is no free energy really. totalEnergyS is to be added later
                  statusOFS << "NOTE: Free energy = Energy in PPEXSIDFTDriver3!" << std::endl;

                  statusOFS << std::endl
                    << "Results obtained from PEXSI:" << std::endl
                    << "Total energy (H*DM)         = " << totalEnergyH << std::endl
                    << "Total energy (S*EDM)        = " << totalEnergyS << std::endl
                    << "Total free energy           = " << totalFreeEnergy << std::endl 
                    << "InertiaIter                 = " << numTotalInertiaIter << std::endl
//                    << "PEXSIIter                   = " <<  numTotalPEXSIIter << std::endl
                    << "mu                          = " << muPEXSI << std::endl
                    << "numElectron                 = " << numElectronPEXSI << std::endl 
                    << std::endl;

                  if( info != 0 ){
                    std::ostringstream msg;
                    msg 
                      << "PEXSI data retrieval returns info " << info << std::endl;
                    ErrorHandling( msg.str().c_str() );
                  }
                }
#endif
              } // if( mpirank < numProcTotalPEXSI_ )

              // Broadcast the total energy Tr[H*DM] and free energy (which is energy)
              MPI_Bcast( &totalEnergyH, 1, MPI_DOUBLE, 0, domain_.comm );
              MPI_Bcast( &totalFreeEnergy, 1, MPI_DOUBLE, 0, domain_.comm );
              // Broadcast the Fermi level
              MPI_Bcast( &fermi_, 1, MPI_DOUBLE, 0, domain_.comm );
              MPI_Bcast( &difNumElectron, 1, MPI_DOUBLE, 0, domain_.comm );

              if( mpirankRow == 0 )
              {
                GetTime(timeSta);
                // Convert the density matrix from DistSparseMatrix format to the
                // DistElemMat format
                DistSparseMatToDistElemMat3(
                    DMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );


                // Convert the energy density matrix from DistSparseMatrix
                // format to the DistElemMat format

                DistSparseMatToDistElemMat3(
                    EDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distEDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );

                // Convert the free energy density matrix from DistSparseMatrix
                // format to the DistElemMat format
                DistSparseMatToDistElemMat3(
                    FDMSparseMat,
                    hamDG.NumBasisTotal(),
                    hamDG.HMat().Prtn(),
                    distFDMMat_,
                    hamDG.ElemBasisIdx(),
                    hamDG.ElemBasisInvIdx(),
                    domain_.colComm,
                    mpirankSparseVec );

                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for converting the DistSparseMatrices to DistElemMat " << 
                  "for post-processing is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              // Broadcast the distElemMat matrices
              // FIXME this is not a memory efficient implementation
              GetTime(timeSta);
              {
                Int sstrSize;
                std::vector<char> sstr;
                if( mpirankRow == 0 ){
                  std::stringstream distElemMatStream;
                  Int cnt = 0;
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    cnt++;
                  } // for (mi)
                  serialize( cnt, distElemMatStream, NO_MASK );
                  for( typename std::map<ElemMatKey, DblNumMat >::iterator mi  = distDMMat_.LocalMap().begin();
                      mi != distDMMat_.LocalMap().end(); mi++ ){
                    ElemMatKey key = (*mi).first;
                    serialize( key, distElemMatStream, NO_MASK );
                    serialize( distDMMat_.LocalMap()[key], distElemMatStream, NO_MASK );

                    serialize( distEDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 
                    serialize( distFDMMat_.LocalMap()[key], distElemMatStream, NO_MASK ); 

                  } // for (mi)
                  sstr.resize( Size( distElemMatStream ) );
                  distElemMatStream.read( &sstr[0], sstr.size() );
                  sstrSize = sstr.size();
                }

                MPI_Bcast( &sstrSize, 1, MPI_INT, 0, domain_.rowComm );
                sstr.resize( sstrSize );
                MPI_Bcast( &sstr[0], sstrSize, MPI_BYTE, 0, domain_.rowComm );

                if( mpirankRow != 0 ){
                  std::stringstream distElemMatStream;
                  distElemMatStream.write( &sstr[0], sstrSize );
                  Int cnt;
                  deserialize( cnt, distElemMatStream, NO_MASK );
                  for( Int i = 0; i < cnt; i++ ){
                    ElemMatKey key;
                    DblNumMat mat;
                    deserialize( key, distElemMatStream, NO_MASK );
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distDMMat_.LocalMap()[key] = mat;

                    deserialize( mat, distElemMatStream, NO_MASK );
                    distEDMMat_.LocalMap()[key] = mat;
                    deserialize( mat, distElemMatStream, NO_MASK );
                    distFDMMat_.LocalMap()[key] = mat;

                  } // for (mi)
                }
              }

              GetTime(timeSta);
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for broadcasting the density matrix for post-processing is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              Evdw_ = 0.0;

              // Compute the Harris energy functional.  
              // NOTE: In computing the Harris energy, the density and the
              // potential must be the INPUT density and potential without ANY
              // update.
              GetTime( timeSta );
              CalculateHarrisEnergyDM( totalFreeEnergy, distFDMMat_ );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for calculating the Harris energy is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Evaluate the electron density

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              GetTime( timeSta );
              hamDG.CalculateDensityDM2( 
                  hamDG.Density(), hamDG.DensityLGL(), distDMMat_ );
              MPI_Barrier( domain_.comm );
              GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing density in the global domain is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];
                    } // own this element
                  } // for (i)


              // Update the output potential, and the KS and second order accurate
              // energy
              GetTime(timeSta);
              {
                // Update the Hartree energy and the exchange correlation energy and
                // potential for computing the KS energy and the second order
                // energy.
                // NOTE Vtot should not be updated until finishing the computation
                // of the energies.

                if( XCType_ == "XC_GGA_XC_PBE" ){
                  GetTime( timeSta );
                  hamDG.CalculateGradDensity(  *distfftPtr_ );
                  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                  statusOFS << " Time for calculating gradient of density is " <<
                    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
                }

                hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

                hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

                // Compute the second order accurate energy functional.
                // NOTE: In computing the second order energy, the density and the
                // potential must be the OUTPUT density and potential without ANY
                // MIXING.
                //        CalculateSecondOrderEnergy();

                // Compute the KS energy 
                CalculateKSEnergyDM( totalEnergyH, distEDMMat_, distFDMMat_ );

                // Update the total potential AFTER updating the energy

                // No external potential

                // Compute the new total potential

                hamDG.CalculateVtot( hamDG.Vtot() );

             }
              GetTime(timeEnd);
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for computing the potential is " <<
                timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

              // Compute the force at every step
              //      if( esdfParam.isCalculateForceEachSCF ){
              //        // Compute force
              //        GetTime( timeSta );
              //        hamDG.CalculateForceDM( *distfftPtr_, distDMMat_ );
              //        GetTime( timeEnd );
              //        statusOFS << "Time for computing the force is " <<
              //          timeEnd - timeSta << " [s]" << std::endl << std::endl;
              //
              //        // Print out the force
              //        // Only master processor output information containing all atoms
              //        if( mpirank == 0 ){
              //          PrintBlock( statusOFS, "Atomic Force" );
              //          {
              //            Point3 forceCM(0.0, 0.0, 0.0);
              //            std::vector<Atom>& atomList = hamDG.AtomList();
              //            Int numAtom = atomList.size();
              //            for( Int a = 0; a < numAtom; a++ ){
              //              Print( statusOFS, "atom", a, "force", atomList[a].force );
              //              forceCM += atomList[a].force;
              //            }
              //            statusOFS << std::endl;
              //            Print( statusOFS, "force for centroid: ", forceCM );
              //            statusOFS << std::endl;
              //          }
              //        }
              //      }

              // TODO Evaluate the a posteriori error estimator

              GetTime( timePEXSIEnd );
#if ( _DEBUGlevel_ >= 0 )
              statusOFS << " Time for PEXSI evaluation is " <<
                timePEXSIEnd - timePEXSISta << " [s]" << std::endl << std::endl;
#endif
            } //if( solutionMethod_ == "pexsi" )


#endif


            // Compute the error of the mixing variable

            GetTime(timeSta);
            {
              Real normMixDifLocal = 0.0, normMixOldLocal = 0.0;
              Real normMixDif, normMixOld;
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      if( mixVariable_ == "density" ){
                        DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                        DblNumVec& newVec = hamDG.Density().LocalMap()[key];

                        for( Int p = 0; p < oldVec.m(); p++ ){
                          normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                          normMixOldLocal += pow( oldVec(p), 2.0 );
                        }
                      }
                      else if( mixVariable_ == "potential" ){
                        DblNumVec& oldVec = mixInnerSave_.LocalMap()[key];
                        DblNumVec& newVec = hamDG.Vtot().LocalMap()[key];

                        for( Int p = 0; p < oldVec.m(); p++ ){
                          normMixDifLocal += pow( oldVec(p) - newVec(p), 2.0 );
                          normMixOldLocal += pow( oldVec(p), 2.0 );
                        }
                      }
                    } // own this element
                  } // for (i)




              mpi::Allreduce( &normMixDifLocal, &normMixDif, 1, MPI_SUM, 
                  domain_.colComm );
              mpi::Allreduce( &normMixOldLocal, &normMixOld, 1, MPI_SUM,
                  domain_.colComm );

              normMixDif = std::sqrt( normMixDif );
              normMixOld = std::sqrt( normMixOld );

              scfInnerNorm_    = normMixDif / normMixOld;
#if ( _DEBUGlevel_ >= 1 )
              Print(statusOFS, "norm(MixDif)          = ", normMixDif );
              Print(statusOFS, "norm(MixOld)          = ", normMixOld );
              Print(statusOFS, "norm(out-in)/norm(in) = ", scfInnerNorm_ );
#endif
            }

            if( scfInnerNorm_ < scfInnerTolerance_ ){
              /* converged */
              Print( statusOFS, "Inner SCF is converged!\n" );
              isInnerSCFConverged = true;
            }

            MPI_Barrier( domain_.colComm );
            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << " Time for computing the SCF residual is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            // Mixing for the inner SCF iteration.
            GetTime( timeSta );

            // The number of iterations used for Anderson mixing
            Int numAndersonIter;

            if( scfInnerMaxIter_ == 1 ){
              // Maximum inner iteration = 1 means there is no distinction of
              // inner/outer SCF.  Anderson mixing uses the global history
              numAndersonIter = scfTotalInnerIter_;
            }
            else{
              // If more than one inner iterations is used, then Anderson only
              // uses local history.  For explanation see 
              //
              // Note 04/11/2013:  
              // "Problem of Anderson mixing in inner/outer SCF loop"
              numAndersonIter = innerIter;
            }

            if( mixVariable_ == "density" ){
              if( mixType_ == "anderson" ||
                  mixType_ == "kerker+anderson"    ){
                AndersonMix(
                    numAndersonIter, 
                    mixStepLength_,
                    mixType_,
                    hamDG.Density(),
                    mixInnerSave_,
                    hamDG.Density(),
                    dfInnerMat_,
                    dvInnerMat_);
              } else{
                ErrorHandling("Invalid mixing type.");
              }
            }
            else if( mixVariable_ == "potential" ){
              if( mixType_ == "anderson" ||
                  mixType_ == "kerker+anderson"    ){
                AndersonMix(
                    numAndersonIter, 
                    mixStepLength_,
                    mixType_,
                    hamDG.Vtot(),
                    mixInnerSave_,
                    hamDG.Vtot(),
                    dfInnerMat_,
                    dvInnerMat_);
              } else{
                ErrorHandling("Invalid mixing type.");
              }
            }



            MPI_Barrier( domain_.comm );
            GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
            statusOFS << " Time for mixing is " <<
              timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

            // Post processing for the density mixing. Make sure that the
            // density is positive, and compute the potential again. 
            // This is only used for density mixing.
            if( mixVariable_ == "density" )
            {
              Real sumRhoLocal = 0.0;
              Real sumRho;
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec&  density      = hamDG.Density().LocalMap()[key];

                      for (Int p=0; p < density.Size(); p++) {
                        density(p) = std::max( density(p), 0.0 );
                        sumRhoLocal += density(p);
                      }
                    } // own this element
                  } // for (i)
              mpi::Allreduce( &sumRhoLocal, &sumRho, 1, MPI_SUM, domain_.colComm );
              sumRho *= domain_.Volume() / domain_.NumGridTotal();

              Real rhofac = hamDG.NumSpin() * hamDG.NumOccupiedState() / sumRho;

#if ( _DEBUGlevel_ >= 1 )
              statusOFS << std::endl;
              Print( statusOFS, "Sum Rho after mixing (raw data) = ", sumRho );
              statusOFS << std::endl;
#endif


              // Normalize the electron density in the global domain
              for( Int k = 0; k < numElem_[2]; k++ )
                for( Int j = 0; j < numElem_[1]; j++ )
                  for( Int i = 0; i < numElem_[0]; i++ ){
                    Index3 key( i, j, k );
                    if( elemPrtn_.Owner( key ) == (mpirank / dmRow_) ){
                      DblNumVec& localRho = hamDG.Density().LocalMap()[key];
                      blas::Scal( localRho.Size(), rhofac, localRho.Data(), 1 );
                    } // own this element
                  } // for (i)


              // Update the potential after mixing for the next iteration.  
              // This is only used for potential mixing

              // Compute the exchange-correlation potential and energy from the
              // new density

              if( XCType_ == "XC_GGA_XC_PBE" ){
                GetTime( timeSta );
                hamDG.CalculateGradDensity(  *distfftPtr_ );
                GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
                statusOFS << " Time for calculating gradient of density is " <<
                  timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif
              }

              hamDG.CalculateXC( Exc_, hamDG.Epsxc(), hamDG.Vxc(), *distfftPtr_ );

              hamDG.CalculateHartree( hamDG.Vhart(), *distfftPtr_ );

              // No external potential

              // Compute the new total potential

              hamDG.CalculateVtot( hamDG.Vtot() );
            }
            // check check 
#ifdef _USE_PEXSI_
            if( solutionMethod_ == "pexsi" )
            {
            Real deltaVmin = 0.0;
            Real deltaVmax = 0.0;

            for( Int k=0; k< numElem_[2]; k++ )
              for( Int j=0; j< numElem_[1]; j++ )
                for( Int i=0; i< numElem_[0]; i++ ) {
                  Index3 key = Index3(i,j,k);
                  if( distEigSolPtr_->Prtn().Owner(key) == (mpirank / dmRow_) ){
                    DblNumVec vtotCur;
                    vtotCur = hamDG.Vtot().LocalMap()[key];
                    DblNumVec& oldVtot = VtotHist.LocalMap()[key];
                    blas::Axpy( vtotCur.m(), -1.0, oldVtot.Data(),
                                    1, vtotCur.Data(), 1);
                    deltaVmin = std::min( deltaVmin, findMin(vtotCur) );
                    deltaVmax = std::max( deltaVmax, findMax(vtotCur) );
                  }
                }

              {
                int color = mpirank % dmRow_;
                MPI_Comm elemComm;
                std::vector<Real> vlist(mpisize/dmRow_);
  
                MPI_Comm_split( domain_.comm, color, mpirank, &elemComm );
                MPI_Allgather( &deltaVmin, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
                deltaVmin = 0.0;
                for(int i =0; i < mpisize/dmRow_; i++)
                  if(deltaVmin > vlist[i])
                     deltaVmin = vlist[i];

                MPI_Allgather( &deltaVmax, 1, MPI_DOUBLE, &vlist[0], 1, MPI_DOUBLE, elemComm);
                deltaVmax = 0.0;
                for(int i =0; i < mpisize/dmRow_; i++)
                  if(deltaVmax < vlist[i])
                     deltaVmax = vlist[i];
 
                pexsiOptions_.muMin0 += deltaVmin;
                pexsiOptions_.muMax0 += deltaVmax;
                MPI_Comm_free( &elemComm);
              }
            }
#endif 
            // Print out the state variables of the current iteration

            // Only master processor output information containing all atoms
            if( mpirank == 0 ){
              PrintState( );
            }

            GetTime( timeIterEnd );

            statusOFS << " Time for this inner SCF iteration = " << timeIterEnd - timeIterStart
              << " [s]" << std::endl << std::endl;
          } // for (innerIter)


          return ;
        }         // -----  end of method SCFDG::InnerIterate_device  ----- 


} // namespace dgdft
