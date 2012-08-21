#ifndef _ENVIRONMENT_DECL_HPP_
#define _ENVIRONMENT_DECL_HPP_

// STL libraries
#include <iostream> 
#include <fstream>
#include <sstream>
#include <complex>
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <cfloat>
#include <cassert>
#include <cmath>
#include <ctime>
#include <stack>
#include <stdexcept>

// NON-STL libraries
#include <unistd.h>

// FFTW libraries
#include "fftw3.h"
#include "fftw3-mpi.h"

// PETSc / SLEPc libraries
#include <slepceps.h>



/***********************************************************************
 *  Data types
 **********************************************************************/

namespace dgdft{

	typedef    PetscBool             Bool;
	typedef    PetscInt              Int;
	typedef    PetscReal             Real;
	typedef    std::complex<double>  Complex;     
	typedef    PetscScalar           Scalar;      

} // namespace dgdft

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace dgdft{


#ifndef _RELEASE_
	void PushCallStack( std::string s );
	void PopCallStack();
	void DumpCallStack();
	void DumpElemCallStack();                     // Overload the elemental's DumpCallStack
#endif // ifndef _RELEASE_

	// We define an output stream that does nothing. This is done so that the 
	// root process can be used to print data to a file's ostream while all other 
	// processes use a null ostream. 
	struct NullStream : std::ostream
	{            
		struct NullStreamBuffer : std::streambuf
		{
			int overflow( int c ) { return traits_type::not_eof(c); }
		} nullStreamBuffer_;

		NullStream() 
			: std::ios(&nullStreamBuffer_), std::ostream(&nullStreamBuffer_)
		{ }
	};  
} // namespace dgdft


#endif // _ENVIRONMENT_DECL_HPP_
