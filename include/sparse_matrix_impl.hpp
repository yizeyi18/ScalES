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
/// @file sparse_matrix_impl.hpp
/// @brief Implementation of sparse matrices.
/// @date 2012-11-28
#ifndef _SPARSE_MATRIX_IMPL_HPP_
#define _SPARSE_MATRIX_IMPL_HPP_

#include "sparse_matrix_decl.hpp"
#include "mpi_interf.hpp"

namespace  dgdft{

extern Int SharedRead(std::string name, std::istringstream& is);

//---------------------------------------------------------
template<typename F>
void ReadSparseMatrix ( const char* filename, SparseMatrix<F>& spmat )
{
#ifndef _RELEASE_
	PushCallStack("ReadSparseMatrix");
#endif
	
	std::istringstream iss;
	Int dummy;
	SharedRead( std::string(filename), iss );
	deserialize( spmat.size, iss, NO_MASK );
	deserialize( spmat.dummy, iss, NO_MASK );
	deserialize( spmat.nnz,  iss, NO_MASK );
	deserialize( spmat.colptr, iss, NO_MASK );
	deserialize( spmat.rowind, iss, NO_MASK );
	deserialize( spmat.nzval, iss, NO_MASK );
	
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function ReadSparseMatrix  ----- 


template <class F> void
ReadSparseMatrixFormatted	( const char* filename, SparseMatrix<F>& spmat )
{
#ifndef _RELEASE_
	PushCallStack("ReadSparseMatrixFormatted");
#endif
	std::ifstream fin(filename);
	Int dummy;
	fin >> spmat.size >> dummy >> spmat.nnz;

	spmat.colptr.Resize( spmat.size+1 );
	spmat.rowind.Resize( spmat.nnz );
	spmat.nzval.Resize ( spmat.nnz );

	for( Int i = 0; i < spmat.size + 1; i++ ){
		fin >> spmat.colptr(i);
	}

	for( Int i = 0; i < spmat.nnz; i++ ){
		fin >> spmat.rowind(i);
	}

	for( Int i = 0; i < spmat.nnz; i++ ){
		fin >> spmat.nzval(i);
	}

	fin.close();

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
}		// -----  end of function ReadSparseMatrixFormatted  ----- 

//---------------------------------------------------------
template<typename F>
void ReadDistSparseMatrix ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
{
#ifndef _RELEASE_
	PushCallStack("ReadDistSparseMatrix");
#endif
	// Get the processor information within the current communicator
  MPI_Barrier( comm );
  Int mpirank;  MPI_Comm_rank(comm, &mpirank);
  Int mpisize;  MPI_Comm_size(comm, &mpisize);
	MPI_Status mpistat;
	std::ifstream fin;

  // Read basic information
	if( mpirank == 0 ){
		fin.open(filename);
		if( !fin.good() ){
			throw std::logic_error( "File cannot be openeded!" );
		}
		Int dummy;
		fin.read((char*)&pspmat.size, sizeof(Int));
		fin.read((char*)&dummy, sizeof(Int));
		fin.read((char*)&pspmat.nnz,  sizeof(Int));
	}

	pspmat.comm = comm;

	MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
	MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

	// Read colptr

	IntNumVec  colptr(pspmat.size+1);
	if( mpirank == 0 ){
		Int tmp;
		fin.read((char*)&tmp, sizeof(Int));  
		if( tmp != pspmat.size+1 ){
			throw std::logic_error( "colptr is not of the right size." );
		}
		fin.read((char*)colptr.Data(), sizeof(Int)*tmp);
	}

	MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);
//	std::cout << "Proc " << mpirank << " outputs colptr[end]" << colptr[pspmat.size] << endl;

	// Compute the number of columns on each processor
	IntNumVec numColLocalVec(mpisize);
	Int numColLocal, numColFirst;
	numColFirst = pspmat.size / mpisize;
  SetValue( numColLocalVec, numColFirst );
  numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry	
	numColLocal = numColLocalVec[mpirank];

	pspmat.colptrLocal.Resize( numColLocal + 1 );
	for( Int i = 0; i < numColLocal + 1; i++ ){
		pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
	}

	// Calculate nnz_loc on each processor
	pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];

  pspmat.rowindLocal.Resize( pspmat.nnzLocal );
	pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

	// Read and distribute the row indices
	if( mpirank == 0 ){
		Int tmp;
		fin.read((char*)&tmp, sizeof(Int));  
		if( tmp != pspmat.nnz ){
			std::ostringstream msg;
			msg 
				<< "The number of nonzeros in row indices do not match." << std::endl
				<< "nnz = " << pspmat.nnz << std::endl
				<< "size of row indices = " << tmp << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		IntNumVec buf;
		Int numRead;
		for( Int ip = 0; ip < mpisize; ip++ ){
			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
				colptr[ip*numColFirst];
			buf.Resize(numRead);
			fin.read( (char*)buf.Data(), numRead*sizeof(Int) );
			if( ip > 0 ){
				MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
				MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
			}
			else{
        pspmat.rowindLocal = buf;
			}
		}
	}
	else{
		Int numRead;
		MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
		if( numRead != pspmat.nnzLocal ){
			std::ostringstream msg;
			msg << "The number of columns in row indices do not match." << std::endl
				<< "numRead  = " << numRead << std::endl
				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}

    pspmat.rowindLocal.Resize( numRead );
		MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
	}
		
//	std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
//		<< pspmat.rowindLocal.m() << endl;


	// Read and distribute the nonzero values
	if( mpirank == 0 ){
		Int tmp;
		fin.read((char*)&tmp, sizeof(Int));  
		if( tmp != pspmat.nnz ){
			std::ostringstream msg;
			msg 
				<< "The number of nonzeros in values do not match." << std::endl
				<< "nnz = " << pspmat.nnz << std::endl
				<< "size of values = " << tmp << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
		NumVec<F> buf;
		Int numRead;
		for( Int ip = 0; ip < mpisize; ip++ ){
			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
				colptr[ip*numColFirst];
			buf.Resize(numRead);
			fin.read( (char*)buf.Data(), numRead*sizeof(F) );
			if( ip > 0 ){
				std::stringstream sstm;
				serialize( buf, sstm, NO_MASK );
				mpi::Send( sstm, ip, 0, 1, comm );
			}
			else{
        pspmat.nzvalLocal = buf;
			}
		}
	}
	else{
		std::stringstream sstm;
		mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
		deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
		if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
			std::ostringstream msg;
			msg << "The number of columns in values do not match." << std::endl
				<< "numRead  = " << pspmat.nzvalLocal.m() << std::endl
				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
	}

	// Close the file
	if( mpirank == 0 ){
    fin.close();
	}



  MPI_Barrier( comm );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function ReadDistSparseMatrix  ----- 



template<typename F>
void ReadDistSparseMatrixFormatted ( const char* filename, DistSparseMatrix<F>& pspmat, MPI_Comm comm )
{
#ifndef _RELEASE_
	PushCallStack("ReadDistSparseMatrixFormatted");
#endif
	// Get the processor information within the current communicator
  MPI_Barrier( comm );
  Int mpirank;  MPI_Comm_rank(comm, &mpirank);
  Int mpisize;  MPI_Comm_size(comm, &mpisize);
	MPI_Status mpistat;
	std::ifstream fin;

  // Read basic information
	if( mpirank == 0 ){
		fin.open(filename);
		if( !fin.good() ){
			throw std::logic_error( "File cannot be openeded!" );
		}
		Int dummy;
		fin >> pspmat.size >> dummy;
		fin >> pspmat.nnz;
	}

	pspmat.comm = comm;
	
	MPI_Bcast(&pspmat.size, 1, MPI_INT, 0, comm);
	MPI_Bcast(&pspmat.nnz,  1, MPI_INT, 0, comm);

	// Read colptr

	IntNumVec  colptr(pspmat.size+1);
	if( mpirank == 0 ){
		Int* ptr = colptr.Data();
		for( Int i = 0; i < pspmat.size+1; i++ )
			fin >> *(ptr++);
	}

	MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);

	// Compute the number of columns on each processor
	IntNumVec numColLocalVec(mpisize);
	Int numColLocal, numColFirst;
	numColFirst = pspmat.size / mpisize;
  SetValue( numColLocalVec, numColFirst );
  numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry	
	numColLocal = numColLocalVec[mpirank];

	// The first column follows the 1-based (FORTRAN convention) index.
	pspmat.firstCol = mpirank * numColFirst + 1;

	pspmat.colptrLocal.Resize( numColLocal + 1 );
	for( Int i = 0; i < numColLocal + 1; i++ ){
		pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
	}

	// Calculate nnz_loc on each processor
	pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];

  pspmat.rowindLocal.Resize( pspmat.nnzLocal );
	pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );

	// Read and distribute the row indices
	if( mpirank == 0 ){
		Int tmp;
		IntNumVec buf;
		Int numRead;
		for( Int ip = 0; ip < mpisize; ip++ ){
			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
				colptr[ip*numColFirst];
			buf.Resize(numRead);
			Int *ptr = buf.Data();
			for( Int i = 0; i < numRead; i++ ){
				fin >> *(ptr++);
			}
			if( ip > 0 ){
				MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
				MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
			}
			else{
        pspmat.rowindLocal = buf;
			}
		}
	}
	else{
		Int numRead;
		MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
		if( numRead != pspmat.nnzLocal ){
			std::ostringstream msg;
			msg << "The number of columns in row indices do not match." << std::endl
				<< "numRead  = " << numRead << std::endl
				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}

    pspmat.rowindLocal.Resize( numRead );
		MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
	}
		
#if ( _DEBUGlevel_ >= 2 )
	std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
		<< pspmat.rowindLocal.m() << endl;
#endif


	// Read and distribute the nonzero values
	if( mpirank == 0 ){
		Int tmp;
		NumVec<F> buf;
		Int numRead;
		for( Int ip = 0; ip < mpisize; ip++ ){
			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
				colptr[ip*numColFirst];
			buf.Resize(numRead);
			F *ptr = buf.Data();
			for( Int i = 0; i < numRead; i++ ){
				fin >> *(ptr++);
			}
			if( ip > 0 ){
				std::stringstream sstm;
				serialize( buf, sstm, NO_MASK );
				mpi::Send( sstm, ip, 0, 1, comm );
			}
			else{
        pspmat.nzvalLocal = buf;
			}
		}
	}
	else{
		std::stringstream sstm;
		mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
		deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
		if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
			std::ostringstream msg;
			msg << "The number of columns in values do not match." << std::endl
				<< "numRead  = " << pspmat.nzvalLocal.m() << std::endl
				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
			throw std::logic_error( msg.str().c_str() );
		}
	}

	// Close the file
	if( mpirank == 0 ){
    fin.close();
	}

  MPI_Barrier( comm );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function ReadDistSparseMatrixFormatted  ----- 


template<typename F>
void WriteDistSparseMatrixFormatted ( 
		const char* filename, 
		DistSparseMatrix<F>& pspmat )
{
#ifndef _RELEASE_
	PushCallStack("WriteDistSparseMatrixFormatted");
#endif
	// Get the processor information within the current communicator
	MPI_Comm comm = pspmat.comm;
  Int mpirank;  MPI_Comm_rank(comm, &mpirank);
  Int mpisize;  MPI_Comm_size(comm, &mpisize);

	MPI_Status mpistat;
	std::ofstream ofs;

  // Write basic information
	if( mpirank == 0 ){
		ofs.open(filename, std::ios_base::out);
		if( !ofs.good() ){
			throw std::logic_error( "File cannot be openeded!" );
		}
		ofs << std::setiosflags(std::ios::left) 
			<< std::setw(LENGTH_VAR_DATA) << pspmat.size
			<< std::setw(LENGTH_VAR_DATA) << pspmat.size
			<< std::setw(LENGTH_VAR_DATA) << pspmat.nnz << std::endl;
		ofs.close();
	}

	// Write colptr information, one processor after another
	IntNumVec colptrSizeLocal(mpisize);
	SetValue( colptrSizeLocal, 0 );
	IntNumVec colptrSize(mpisize);
	SetValue( colptrSize, 0 );
	colptrSizeLocal(mpirank) = pspmat.colptrLocal[pspmat.colptrLocal.Size()-1] - 1;
	mpi::Allreduce( colptrSizeLocal.Data(), colptrSize.Data(),
			mpisize, MPI_SUM, comm );
	IntNumVec colptrStart(mpisize);
	colptrStart[0] = 1;
	for( Int l = 1; l < mpisize; l++ ){
		colptrStart[l] = colptrStart[l-1] + colptrSize[l-1];
	}
	for( Int p = 0; p < mpisize; p++ ){
		if( mpirank == p ){
			ofs.open(filename, std::ios_base::out | std::ios_base::app );
			if( !ofs.good() ){
				throw std::logic_error( "File cannot be openeded!" );
			}
			IntNumVec& colptrLocal = pspmat.colptrLocal;
			for( Int i = 0; i < colptrLocal.Size() - 1; i++ ){
				ofs << std::setiosflags(std::ios::left) 
					<< colptrLocal[i] + colptrStart[p] - 1 << "  ";
			}
			if( p == mpisize - 1 ){
				ofs << std::setiosflags(std::ios::left) 
					<< colptrLocal[colptrLocal.Size()-1] + colptrStart[p] - 1 << std::endl;
			}
			ofs.close();
		}

		MPI_Barrier( comm );
	}	

	// Write rowind information, one processor after another
	for( Int p = 0; p < mpisize; p++ ){
		if( mpirank == p ){
			ofs.open(filename, std::ios_base::out | std::ios_base::app );
			if( !ofs.good() ){
				throw std::logic_error( "File cannot be openeded!" );
			}
			IntNumVec& rowindLocal = pspmat.rowindLocal;
			for( Int i = 0; i < rowindLocal.Size(); i++ ){
				ofs << std::setiosflags(std::ios::left) 
					<< rowindLocal[i] << "  ";
			}
			if( p == mpisize - 1 ){
				ofs << std::endl;
			}
			ofs.close();
		}

		MPI_Barrier( comm );
	}	

	// Write nzval information, one processor after another
	for( Int p = 0; p < mpisize; p++ ){
		if( mpirank == p ){
			ofs.open(filename, std::ios_base::out | std::ios_base::app );
			if( !ofs.good() ){
				throw std::logic_error( "File cannot be openeded!" );
			}
			NumVec<F>& nzvalLocal = pspmat.nzvalLocal;
			for( Int i = 0; i < nzvalLocal.Size(); i++ ){
				ofs << std::setiosflags(std::ios::left) 
					<< std::setiosflags(std::ios::scientific)
					<< std::setiosflags(std::ios::showpos)
					<< std::setprecision(LENGTH_FULL_PREC)
					<< nzvalLocal[i] << "  ";
			}
			if( p == mpisize - 1 ){
				ofs << std::endl;
			}
			ofs.close();
		}

		MPI_Barrier( comm );
	}	


//	// Read colptr
//
//	IntNumVec  colptr(pspmat.size+1);
//	if( mpirank == 0 ){
//		Int* ptr = colptr.Data();
//		for( Int i = 0; i < pspmat.size+1; i++ )
//			fin >> *(ptr++);
//	}
//
//	MPI_Bcast(colptr.Data(), pspmat.size+1, MPI_INT, 0, comm);
//
//	// Compute the number of columns on each processor
//	IntNumVec numColLocalVec(mpisize);
//	Int numColLocal, numColFirst;
//	numColFirst = pspmat.size / mpisize;
//  SetValue( numColLocalVec, numColFirst );
//  numColLocalVec[mpisize-1] = pspmat.size - numColFirst * (mpisize-1);  // Modify the last entry	
//	numColLocal = numColLocalVec[mpirank];
//
//	// The first column follows the 1-based (FORTRAN convention) index.
//	pspmat.firstCol = mpirank * numColFirst + 1;
//
//	pspmat.colptrLocal.Resize( numColLocal + 1 );
//	for( Int i = 0; i < numColLocal + 1; i++ ){
//		pspmat.colptrLocal[i] = colptr[mpirank * numColFirst+i] - colptr[mpirank * numColFirst] + 1;
//	}
//
//	// Calculate nnz_loc on each processor
//	pspmat.nnzLocal = pspmat.colptrLocal[numColLocal] - pspmat.colptrLocal[0];
//
//  pspmat.rowindLocal.Resize( pspmat.nnzLocal );
//	pspmat.nzvalLocal.Resize ( pspmat.nnzLocal );
//
//	// Read and distribute the row indices
//	if( mpirank == 0 ){
//		Int tmp;
//		IntNumVec buf;
//		Int numRead;
//		for( Int ip = 0; ip < mpisize; ip++ ){
//			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
//				colptr[ip*numColFirst];
//			buf.Resize(numRead);
//			Int *ptr = buf.Data();
//			for( Int i = 0; i < numRead; i++ ){
//				fin >> *(ptr++);
//			}
//			if( ip > 0 ){
//				MPI_Send(&numRead, 1, MPI_INT, ip, 0, comm);
//				MPI_Send(buf.Data(), numRead, MPI_INT, ip, 1, comm);
//			}
//			else{
//        pspmat.rowindLocal = buf;
//			}
//		}
//	}
//	else{
//		Int numRead;
//		MPI_Recv(&numRead, 1, MPI_INT, 0, 0, comm, &mpistat);
//		if( numRead != pspmat.nnzLocal ){
//			std::ostringstream msg;
//			msg << "The number of columns in row indices do not match." << std::endl
//				<< "numRead  = " << numRead << std::endl
//				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
//			throw std::logic_error( msg.str().c_str() );
//		}
//
//    pspmat.rowindLocal.Resize( numRead );
//		MPI_Recv( pspmat.rowindLocal.Data(), numRead, MPI_INT, 0, 1, comm, &mpistat );
//	}
//		
//#if ( _DEBUGlevel_ >= 2 )
//	std::cout << "Proc " << mpirank << " outputs rowindLocal.size() = " 
//		<< pspmat.rowindLocal.m() << endl;
//#endif
//
//
//	// Read and distribute the nonzero values
//	if( mpirank == 0 ){
//		Int tmp;
//		NumVec<F> buf;
//		Int numRead;
//		for( Int ip = 0; ip < mpisize; ip++ ){
//			numRead = colptr[ip*numColFirst + numColLocalVec[ip]] - 
//				colptr[ip*numColFirst];
//			buf.Resize(numRead);
//			F *ptr = buf.Data();
//			for( Int i = 0; i < numRead; i++ ){
//				fin >> *(ptr++);
//			}
//			if( ip > 0 ){
//				std::stringstream sstm;
//				serialize( buf, sstm, NO_MASK );
//				mpi::Send( sstm, ip, 0, 1, comm );
//			}
//			else{
//        pspmat.nzvalLocal = buf;
//			}
//		}
//	}
//	else{
//		std::stringstream sstm;
//		mpi::Recv( sstm, 0, 0, 1, comm, mpistat, mpistat );
//		deserialize( pspmat.nzvalLocal, sstm, NO_MASK );
//		if( pspmat.nzvalLocal.m() != pspmat.nnzLocal ){
//			std::ostringstream msg;
//			msg << "The number of columns in values do not match." << std::endl
//				<< "numRead  = " << pspmat.nzvalLocal.m() << std::endl
//				<< "nnzLocal = " << pspmat.nnzLocal << std::endl;
//			throw std::logic_error( msg.str().c_str() );
//		}
//	}
//
//	// Close the file
//	if( mpirank == 0 ){
//    fin.close();
//	}

  MPI_Barrier( comm );

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function WriteDistSparseMatrixFormatted  ----- 


} // namespace dgdft

#endif // _SPARSE_MATRIX_IMPL_HPP_
