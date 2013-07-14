/// @file sparse_matrix_impl.hpp
/// @brief Implementation of sparse matrices.
/// @author Lin Lin
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

} // namespace dgdft

#endif // _SPARSE_MATRIX_IMPL_HPP_
