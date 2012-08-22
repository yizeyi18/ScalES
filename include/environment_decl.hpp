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

// Make sure that the following definition is consistent with the
// Petsc definition.  

typedef    int                   Int;
typedef    double                Real;
typedef    std::complex<double>  Complex; // Must use elemental form of complex
#ifdef _USE_COMPLEX_
typedef    std::complex<double>  Scalar;  // Must use elemental form of complex
#else
typedef    double                Scalar;
#endif
typedef    PetscScalar           PetscScalar;      // Still use PetscScalar


// *********************************************************************
// Define constants
// *********************************************************************
const Int DIM = 3;                            // Always in 3D
const Int I_ZERO = 0;
const Int I_ONE = 1;
const Real D_ZERO = 0.0;
const Real D_ONE  = 1.0;
const Complex Z_ZERO = Complex(0.0, 0.0);
const Complex Z_ONE  = Complex(1.0, 0.0);
const char UPPER = 'U';
const char LOWER = 'L';
const Real PI = PETSC_PI;


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
