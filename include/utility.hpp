/// @file utility.hpp
/// @brief Utility subroutines.
/// @author Lin Lin
/// @date 2012-08-13
#ifndef _UTILITY_HPP_ 
#define _UTILITY_HPP_

#include  <stdlib.h>
#include  "domain.hpp"
#include  "environment.hpp"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"
#include  "numtns_impl.hpp"

namespace dgdft{

// *********************************************************************
// Global utility functions 
// These utility functions do not depend on local definitions
// *********************************************************************
inline Int IRound(Real a){ 
		Int b = 0;
		if(a>0) b = (a-Int(a)<0.5)?Int(a):(Int(a)+1);
		else b = (Int(a)-a<0.5)?Int(a):(Int(a)-1);
		return b; 
	}

inline Int IMod(int a,int b){
	assert( b > 0 );
  return ((a%b)<0)?((a%b)+b):(a%b);
}

inline Real DMod( Real a, Real b){
  assert(b>0);
  return (a>=0)?(a-Int(a/b)*b):(a-(Int(a/b)-1)*b);
}


inline Int OptionsCreate(Int argc, char** argv, std::map<std::string,std::string>& options)
{
	options.clear();
	for(Int k=1; k<argc; k=k+2) {
		options[ std::string(argv[k]) ] = std::string(argv[k+1]);
	}
	return 0;
}

// *********************************************************************
// Stringstream
// *********************************************************************

// Size information.
// Like sstm.str().length() but without making the copy
inline Int Size( std::stringstream& sstm ){
	Int length;
	sstm.seekg (0, std::ios::end);
	length = sstm.tellg();
	sstm.seekg (0, std::ios::beg);
	return length;
}


// *********************************************************************
// Data types
// *********************************************************************
// Sparse vector used for a domain with uniform grid (not for LGL grid).  
// SparseVec.first is the indices of grid points for which the sparse
// vector has nonzero value.  
// SparseVec.second contains four columns, ordered as value, and its
// derivatives along x, y, z directions, respectively. 
typedef std::pair<IntNumVec, DblNumMat > SparseVec; 

// Each nonlocal pseudopotential is written as
//   \int dy V(x,y) f(y) = w b(x) \int dy b(y) f(y).
// First: b vector with integration measure.
// Second: weight w
typedef std::pair<SparseVec, Real> NonlocalPP; 



// *********************************************************************
// Define constants
// *********************************************************************

// Write format control parameters 
const int LENGTH_VAR_NAME = 8;
const int LENGTH_DBL_DATA = 16;
const int LENGTH_INT_DATA = 5;
const int LENGTH_VAR_UNIT = 6;
const int LENGTH_DBL_PREC = 8;
const int LENGTH_VAR_DATA = 16;



// *********************************************************************
// Mesh  and other utilities FIXME
// *********************************************************************

/// @brief Check whether a point is in a sub-domain which is a part of a
/// global domain with periodic boundary conditions
inline bool IsInSubdomain( 
		const Point3& r, 
		const Domain& dm, 
		const Point3& Lsglb )
{
  bool isIn = true;
	Point3 posstart = dm.posStart;
	Point3 Lsbuf = dm.length;
  Point3 shiftstart;
  Point3 shiftr;
  for( Int i = 0; i < DIM; i++){
    shiftstart[i] = DMod(posstart[i], Lsglb[i]);
    shiftr[i]     = DMod(r[i],        Lsglb[i]);
    /* Case 1 of the buffer interval */
		if( shiftstart[i] + Lsbuf[i] > Lsglb[i] ){
			if( (shiftr[i] > shiftstart[i] + Lsbuf[i] - Lsglb[i]) &&
					(shiftr[i] < shiftstart[i]) ){
				isIn = false;
			}
		}
    /* Case 2 of the buffer interval */
		else{
			if( (shiftr[i] < shiftstart[i]) ||
					(shiftr[i] > shiftstart[i] + Lsbuf[i]) ){
				isIn = false;
			}
		}
  }
  return isIn;
}

/// @brief Converts a 1D global index to a 3D index.
inline Index3 Index1To3( const Int idx1, const Index3& numGrid ){
	Index3 idx3;
	idx3[2] = idx1 / ( numGrid[0] * numGrid[1] );
	idx3[1] = ( idx1 % ( numGrid[0] * numGrid[1] ) ) / numGrid[0];
	idx3[0] = idx1 % numGrid[0];
	return idx3;
}

/// @brief Converts a 3D index to a 1D global index.
inline Int Index3To1( const Index3& idx3, const Index3& numGrid ){
	Int idx1 = idx3[0] + idx3[1] * numGrid[0] + idx3[2] * numGrid[0] * numGrid[1];
	return idx1;
}

/// @brief Generate the unscaled LGL grid points, integration weights, polynomials and differentiation matrix.
void GenerateLGL(Real* x,  Int N);
void GenerateLGL(Real* x,  Real* w, Int N);
void GenerateLGL(std::vector<Real>& x,  Int N);
void GenerateLGL(std::vector<Real>& x,  std::vector<Real>& w, std::vector<Real>& P, Int N);
void GenerateLGL(std::vector<Real>& x,  std::vector<Real>& D, Int N);

template<class F>
void Transpose(std::vector<F>& A, std::vector<F>& B, Int m, Int n){
  Int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

template<class F>
void Transpose(Real* A, Real* B, Int m, Int n){
  Int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

/// @brief Generate 1D spline coefficients
void spline(Int, Real*, Real*, Real*, Real*, Real*);

/// @brief Evaluate the spline
void seval(Real*, Int, Real*, Int, Real*, Real*, Real*, Real*, Real*);


/// @brief Three component inner product.
inline Real Innerprod(Real* x, Real* y, Real *w, Int ntot){
  Real tmp = 0.0;
  for(Int i = 0; i < ntot; i++){
    tmp += x[i] * y[i] * w[i];
  }
  return tmp;
}

/// @brief Generate a uniform mesh from a domain.
void UniformMesh( const Domain &dm, std::vector<DblNumVec> &gridpos );

/// @brief Generate a LGL mesh from a domain.
void LGLMesh( const Domain &dm, const Index3& numGrid, std::vector<DblNumVec> &gridpos );


// *********************************************************************
// Formatted output stream
// *********************************************************************

// Bool is NOT defined due to ambiguity with Int

inline Int PrintBlock(std::ostream &os, const std::string name){

	os << std::endl<< "*********************************************************************" << std::endl;
	os << name << std::endl;
	os << "*********************************************************************" << std::endl << std::endl;
	return 0;
}

// String
inline Int Print(std::ostream &os, const std::string name) {
  os << std::setiosflags(std::ios::left) << name << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name) {
  os << std::setiosflags(std::ios::left) << std::string(name) << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, std::string val) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setw(LENGTH_VAR_DATA) << val
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const std::string name, const char* val) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setw(LENGTH_VAR_DATA) << std::string(val)
     << std::endl;
  return 0;
};


// Real

// one real number

inline Int Print(std::ostream &os, const std::string name, Real val) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char* name, Real val) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << std::string(name)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Real val, const std::string unit) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << unit 
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name, Real val, const char *unit) {
  os << std::setiosflags(std::ios::left) 
     << std::setw(LENGTH_VAR_NAME) << std::string(name)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_DBL_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit) 
     << std::endl;
  return 0;
};

// Two real numbers
inline Int Print(std::ostream &os, const std::string name1, Real val1, const std::string unit1,
  const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name1
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1
     << std::setw(LENGTH_VAR_UNIT) << unit1 
     << std::setw(LENGTH_VAR_NAME) << name2
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << unit2 
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Real val1, const char *unit1,
  char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val1
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
     << std::endl;
  return 0;
};

// Int and Real
inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
  const std::string name2, Real val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name1
     << std::setw(LENGTH_INT_DATA) << val1
     << std::setw(LENGTH_VAR_UNIT) << unit1 
     << std::setw(LENGTH_VAR_NAME) << name2
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << unit2 
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
  char *name2, Real val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_INT_DATA) << val1
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
  const char *name1, Int val1, 
  const char *name2, Real val2, 
  char *name3, Real val3) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_INT_DATA) << val1 
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, 
  const char *name1, Int val1, 
  const char *name2, Real val2, 
  const char *name3, Real val3,
  const char *name4, Real val4 ) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_INT_DATA) << val1 
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val2
     << std::setw(LENGTH_VAR_NAME) << std::string(name3) 
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val3
     << std::setw(LENGTH_VAR_NAME) << std::string(name4) 
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< val4
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};

// Int

// one integer number
inline Int Print(std::ostream &os, std::string name, Int val) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setw(LENGTH_VAR_DATA) << val
     << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char *name, Int val) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name)
     << std::setw(LENGTH_VAR_DATA) << val
     << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const std::string name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name
     << std::setw(LENGTH_VAR_DATA) << val
     << std::setw(LENGTH_VAR_UNIT) << unit 
     << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, Int val, const std::string unit) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name)
     << std::setw(LENGTH_VAR_DATA) << val
     << std::setw(LENGTH_VAR_UNIT) << unit 
     << std::endl;
  return 0;
};



// two integer numbers
inline Int Print(std::ostream &os, const std::string name1, Int val1, const std::string unit1,
  const std::string name2, Int val2, const std::string unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name1
     << std::setw(LENGTH_VAR_DATA) << val1
     << std::setw(LENGTH_VAR_UNIT) << unit1 
     << std::setw(LENGTH_VAR_NAME) << name2
     << std::setw(LENGTH_VAR_DATA) << val2
     << std::setw(LENGTH_VAR_UNIT) << unit2 
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, const char *name1, Int val1, const char *unit1,
  char *name2, Int val2, char *unit2) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_VAR_DATA) << val1
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit1) 
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setw(LENGTH_VAR_DATA) << val2
     << std::setw(LENGTH_VAR_UNIT) << std::string(unit2) 
     << std::endl;
  return 0;
};

// Bool

// one boolean number
inline Int Print(std::ostream &os, const std::string name, bool val) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << name;
	if( val == true )
     os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
	else
     os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


inline Int Print(std::ostream &os, const char* name, bool val) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name);
	if( val == true )
     os << std::setw(LENGTH_VAR_NAME) << "true" << std::endl;
	else
     os << std::setw(LENGTH_VAR_NAME) << "false" << std::endl;
  return 0;
};


// Index 3 and Point 3
inline Int Print(std::ostream &os, 
  const char *name1, Index3 val ) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_VAR_DATA) << val[0]
     << std::setw(LENGTH_VAR_DATA) << val[1]
     << std::setw(LENGTH_VAR_DATA) << val[2]
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
  const char *name1, Point3 val ) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[0]
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[1]
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << val[2]
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};

inline Int Print(std::ostream &os, 
  const char *name1, Int val1,
  const char *name2, Point3 val ) {
  os << std::setiosflags(std::ios::left)
     << std::setw(LENGTH_VAR_NAME) << std::string(name1)
     << std::setw(LENGTH_INT_DATA) << val1
     << std::setw(LENGTH_VAR_NAME) << std::string(name2)
     << std::setiosflags(std::ios::scientific)
     << std::setiosflags(std::ios::showpos)
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[0]
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[1]
     << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) <<val[2]
     << std::resetiosflags(std::ios::scientific)
     << std::resetiosflags(std::ios::showpos)
     << std::endl;
  return 0;
};

// *********************************************************************
// Overload << and >> operators for basic data types
// *********************************************************************

// std::vector
template <class F> inline std::ostream& operator<<( std::ostream& os, const std::vector<F>& vec)
{
	os<<vec.size()<<std::endl;
	os.setf(std::ios_base::scientific, std::ios_base::floatfield);
	for(Int i=0; i<vec.size(); i++)	 
		os<<" "<<vec[i];
	os<<std::endl;
	return os;
}

// NumVec
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumVec<F>& vec)
{
	os<<vec.m()<<std::endl;
	os.setf(std::ios_base::scientific, std::ios_base::floatfield);
	for(Int i=0; i<vec.m(); i++)	 
		os<<" "<<vec(i);
	os<<std::endl;
	return os;
}

//template <class F> inline std::istream& operator>>( std::istream& is, NumVec<F>& vec)
//{
//	Int m;  is>>m;  vec.resize(m);
//	for(Int i=0; i<vec.m(); i++)	 
//		is >> vec(i);
//	return is;
//}

// NumMat
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumMat<F>& mat)
{
  os<<mat.m()<<" "<<mat.n()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<mat.m(); i++) {
    for(Int j=0; j<mat.n(); j++)
      os<<" "<<mat(i,j);
    os<<std::endl;
  }
  return os;
}

// NumTns
template <class F> inline std::ostream& operator<<( std::ostream& os, const NumTns<F>& tns)
{
  os<<tns.m()<<" "<<tns.n()<<" "<<tns.p()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<tns.m(); i++) {
	 for(Int j=0; j<tns.n(); j++) {
		for(Int k=0; k<tns.p(); k++) {
		  os<<" "<<tns(i,j,k);
		}
		os<<std::endl;
	 }
	 os<<std::endl;
  }
  return os;
}

// *********************************************************************
// serialize/deserialize for basic types
// More specific serialize/deserialize will be defined in individual
// class files
// *********************************************************************

// standard case for most serialization/deserialization process.
const std::vector<Int> NO_MASK(1);

//bool
inline Int serialize(const bool& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(bool));
  return 0;
}

inline Int deserialize(bool& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(bool));
  return 0;
}

//char
inline Int serialize(const char& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(char));
  return 0;
}

inline Int deserialize(char& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(char));
  return 0;
}

inline Int combine(char& val, char& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
}

//-------------------
//Int
inline Int serialize(const Int& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Int));
  return 0;
}

inline Int deserialize(Int& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Int));
  return 0;
}

inline Int combine(Int& val, Int& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Real
inline Int serialize(const Real& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Real));
  return 0;
}

inline Int deserialize(Real& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Real));
  return 0;
}

inline Int combine(Real& val, Real& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Complex
inline Int serialize(const Complex& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&val, sizeof(Complex));
  return 0;
}

inline Int deserialize(Complex& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&val, sizeof(Complex));
  return 0;
}

inline Int combine(Complex& val, Complex& ext)
{
  val += ext;
  return 0;
}

//-------------------
//Index2  TODO
//inline Int serialize(const Index2& val, std::ostream& os, const std::vector<Int>& mask)
//{
//  os.write((char*)&(val[0]), 2*sizeof(Int));
//  return 0;
//}
//
//inline Int deserialize(Index2& val, std::istream& is, const std::vector<Int>& mask)
//{
//  is.read((char*)&(val[0]), 2*sizeof(Int));
//  return 0;
//}
//
//inline Int combine(Index2& val, Index2& ext)
//{
//	throw  std::logic_error( "Combine operation not implemented." );
//  return 0;
//}

//-------------------
//Point2  TODO
//inline Int serialize(const Point2& val, std::ostream& os, const std::vector<Int>& mask)
//{
//  os.write((char*)&(val[0]), 2*sizeof(Real));
//  return 0;
//}
//
//inline Int deserialize(Point2& val, std::istream& is, const std::vector<Int>& mask)
//{
//  is.read((char*)&(val[0]), 2*sizeof(Real));
//  return 0;
//}
//
//inline Int combine(Point2& val, Point2& ext)
//{
//	throw  std::logic_error( "Combine operation not implemented." );
//  return 0;
//}

//-------------------
//Index3
inline Int serialize(const Index3& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 3*sizeof(Int));
  return 0;
}

inline Int deserialize(Index3& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 3*sizeof(Int));
  return 0;
}

inline Int combine(Index3& val, Index3& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
}

//-------------------
//Point3
inline Int serialize(const Point3& val, std::ostream& os, const std::vector<Int>& mask)
{
  os.write((char*)&(val[0]), 3*sizeof(Real));
  return 0;
}

inline Int deserialize(Point3& val, std::istream& is, const std::vector<Int>& mask)
{
  is.read((char*)&(val[0]), 3*sizeof(Real));
  return 0;
}

inline Int combine(Point3& val, Point3& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
}

//-------------------
//std::vector
template<class T>
Int serialize(const std::vector<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++)
    serialize(val[k], os, mask);
  return 0;
}

template<class T>
Int deserialize(std::vector<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  val.Resize(sz);
  for(Int k=0; k<sz; k++)
    deserialize(val[k], is, mask);
  return 0;
}

template<class T>
Int combine(std::vector<T>& val, std::vector<T>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::set
template<class T>
Int serialize(const std::set<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(typename std::set<T>::const_iterator mi=val.begin(); mi!=val.end(); mi++) 
	serialize((*mi), os, mask);
  return 0;
}

template<class T>
Int deserialize(std::set<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  val.clear();
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++) {
	T t; deserialize(t, is, mask);
	val.insert(t);
  }
  return 0;
}

template<class T>
Int combine(std::set<T>& val, std::set<T>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::map
template<class T, class S>
Int serialize(const std::map<T,S>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int sz = val.size();
  os.write((char*)&sz, sizeof(Int));
  for(typename std::map<T,S>::const_iterator mi=val.begin(); mi!=val.end(); mi++) {
    serialize((*mi).first, os, mask);
    serialize((*mi).second, os, mask);
  }
  return 0;
}

template<class T, class S>
Int deserialize(std::map<T,S>& val, std::istream& is, const std::vector<Int>& mask)
{
  val.clear();
  Int sz;
  is.read((char*)&sz, sizeof(Int));
  for(Int k=0; k<sz; k++) {
    T t;	deserialize(t, is, mask);
    S s;	deserialize(s, is, mask);
    val[t] = s;
  }
  return 0;
}

template<class T, class S>
Int combine(std::map<T,S>& val, std::map<T,S>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

//-------------------
//std::pair
template<class T, class S>
Int serialize(const std::pair<T,S>& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.first, os, mask);
  serialize(val.second, os, mask);
  return 0;
}

template<class T, class S>
Int deserialize(std::pair<T,S>& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.first, is, mask);
  deserialize(val.second, is, mask);
  return 0;
}

template<class T, class S>
Int combine(std::pair<T,S>& val, std::pair<T,S>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

/*
//-------------------
//BolNumVec
inline Int serialize(const BolNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(bool));
  return 0;
}

inline Int deserialize(BolNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(bool));
  return 0;
}

//-------------------
//BolNumMat
inline Int serialize(const BolNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(bool));
  return 0;
}

inline Int deserialize(BolNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(bool));
  return 0;
}

//-------------------
//BolNumTns
inline Int serialize(const BolNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(bool));
  return 0;
}

inline Int deserialize(BolNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(bool));
  return 0;
}
*/

//-------------------
//IntNumVec
inline Int serialize(const IntNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Int));
  return 0;
}

inline Int combine(IntNumVec& val, IntNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}


//-------------------
//IntNumMat
inline Int serialize(const IntNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Int));
  return 0;
}

inline Int combine(IntNumMat& val, IntNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}

//-------------------
//IntNumTns
inline Int serialize(const IntNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Int));
  return 0;
}

inline Int deserialize(IntNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Int));
  return 0;
}

inline Int combine(IntNumTns& val, IntNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
	val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//DblNumVec
inline Int serialize(const DblNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Real));
  return 0;
}

inline Int combine(DblNumVec& val, DblNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}

//-------------------
//DblNumMat
inline Int serialize(const DblNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Real));
  return 0;
}

inline Int combine(DblNumMat& val, DblNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}


//-------------------
//DblNumTns
inline Int serialize(const DblNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Real));
  return 0;
}

inline Int deserialize(DblNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Real));
  return 0;
}

inline Int combine(DblNumTns& val, DblNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
	val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//CpxNumVec
inline Int serialize(const CpxNumVec& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)(val.Data()), m*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumVec& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  is.read((char*)(val.Data()), m*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumVec& val, CpxNumVec& ext)
{
  //val.resize(ext.m());
  assert(val.m()==ext.m());
  for(Int i=0; i<val.m(); i++)    val(i) += ext(i);
  return 0;
}

//-------------------
//CpxNumMat
inline Int serialize(const CpxNumMat& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)(val.Data()), m*n*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumMat& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  is.read((char*)(val.Data()), m*n*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumMat& val, CpxNumMat& ext)
{
  //val.resize(ext.m(),ext.n());
  assert(val.m()==ext.m() && val.n()==ext.n());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      val(i,j) += ext(i,j);
  return 0;
}

//-------------------
//CpxNumTns
inline Int serialize(const CpxNumTns& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();  Int n = val.n();  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  os.write((char*)(val.Data()), m*n*p*sizeof(Complex));
  return 0;
}

inline Int deserialize(CpxNumTns& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m,n,p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  is.read((char*)(val.Data()), m*n*p*sizeof(Complex));
  return 0;
}

inline Int combine(CpxNumTns& val, CpxNumTns& ext)
{
  //val.resize(ext.m(),ext.n(),ext.p());
  assert(val.m()==ext.m() && val.n()==ext.n() && val.p()==ext.p());
  for(Int i=0; i<val.m(); i++)
    for(Int j=0; j<val.n(); j++)
      for(Int k=0; k<val.p(); k++)
	val(i,j,k) += ext(i,j,k);
  return 0;
}

//-------------------
//NumVec
template<class T>
Int inline serialize(const NumVec<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  os.write((char*)&m, sizeof(Int));
  for(Int i=0; i<m; i++)
	serialize(val(i), os, mask);
  return 0;
}

template<class T>
Int inline deserialize(NumVec<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  is.read((char*)&m, sizeof(Int));
  val.Resize(m);
  for(Int i=0; i<m; i++)
    deserialize(val(i), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumVec<T>& val, NumVec<T>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

//-------------------
//NumMat
template<class T>
Int inline serialize(const NumMat<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  for(Int j=0; j<n; j++)
	for(Int i=0; i<m; i++)
	  serialize(val(i,j), os, mask);
  return 0;
}
template<class T>
Int inline deserialize(NumMat<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  val.Resize(m,n);
  for(Int j=0; j<n; j++)
	for(Int i=0; i<m; i++)
	  deserialize(val(i,j), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumMat<T>& val, NumMat<T>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}


//-------------------
//NumTns
template<class T>
Int inline serialize(const NumTns<T>& val, std::ostream& os, const std::vector<Int>& mask)
{
  Int m = val.m();
  Int n = val.n();
  Int p = val.p();
  os.write((char*)&m, sizeof(Int));
  os.write((char*)&n, sizeof(Int));
  os.write((char*)&p, sizeof(Int));
  for(Int k=0; k<p; k++)
	for(Int j=0; j<n; j++)
	  for(Int i=0; i<m; i++)
		serialize(val(i,j,k), os, mask);
  return 0;
}

template<class T>
Int inline deserialize(NumTns<T>& val, std::istream& is, const std::vector<Int>& mask)
{
  Int m;
  Int n;
  Int p;
  is.read((char*)&m, sizeof(Int));
  is.read((char*)&n, sizeof(Int));
  is.read((char*)&p, sizeof(Int));
  val.Resize(m,n,p);
  for(Int k=0; k<p; k++)
    for(Int j=0; j<n; j++)
      for(Int i=0; i<m; i++)
	deserialize(val(i,j,k), is, mask);
  return 0;
}

template<class T>
Int inline combine(NumTns<T>& val, NumTns<T>& ext)
{
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}

Int inline serialize(const Domain& dm, std::ostream& os, const std::vector<Int>& mask){
	serialize( dm.length, os, mask );
	serialize( dm.posStart, os, mask );
	serialize( dm.numGrid, os, mask );
	// Do not serialize the communicatior
	return 0;
}

Int inline deserialize(Domain& dm, std::istream& is, const std::vector<Int>& mask){
	deserialize( dm.length, is, mask );
	deserialize( dm.posStart, is, mask );
	deserialize( dm.numGrid, is, mask );
	// Do not deserialize the communicatior 
	return 0;
}

Int inline combine(Domain& dm1, Domain& dm2){
	throw  std::logic_error( "Combine operation not implemented." );
  return 0;
}


// *********************************************************************
// Parallel IO functions
// *********************************************************************

Int SeparateRead(std::string name, std::istringstream& is);

Int SeparateWrite(std::string name, std::ostringstream& os);

Int SeparateWriteAscii(std::string name, std::ostringstream& os);

Int SharedRead(std::string name, std::istringstream& is);

Int SharedWrite(std::string name, std::ostringstream& os);


// *********************************************************************
// Random numbers
// *********************************************************************
inline void SetRandomSeed(long int seed){
	srand48(seed);
}

inline Real UniformRandom(){
	return (Real)drand48();
}

inline void UniformRandom( NumVec<Real>& vec )
{
	for(Int i=0; i<vec.m(); i++)
		vec(i) = UniformRandom();
}

inline void UniformRandom( NumVec<Complex>& vec )
{
	for(Int i=0; i<vec.m(); i++)
		vec(i) = Complex(UniformRandom(), UniformRandom());
}

inline void UniformRandom( NumMat<Real>& M )
{
	Real *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
		*(ptr++) = UniformRandom(); 
}

inline void UniformRandom( NumMat<Complex>& M )
{
	Complex *ptr = M.Data();
  for(Int i=0; i < M.m() * M.n(); i++) 
		*(ptr++) = Complex(UniformRandom(), UniformRandom()); 
}


inline void UniformRandom( NumTns<Real>& T )
{
	Real *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
		*(ptr++) = UniformRandom(); 
}

inline void UniformRandom( NumTns<Complex>& T )
{
	Complex *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
		*(ptr++) = Complex(UniformRandom(), UniformRandom()); 
}

// *********************************************************************
// Timing
// *********************************************************************
inline void GetTime(Real&  t){
	t = MPI_Wtime();
}

// *********************************************************************
// Comparator
// *********************************************************************

// Real
inline bool PairLtComparator( const std::pair<Real, Int>& l, 
		const std::pair<Real, Int>& r ){
	return l.first < r.first;
}

inline bool PairGtComparator( const std::pair<Real, Int>& l, 
		const std::pair<Real, Int>& r ){
	return l.first > r.first;
}

// For sorting with indices
// Example usage:
//   std::sort(val.begin(), val.end(), IndexComp<std::vector<int>&>(indices));
template<class T> 
struct IndexComp {
private: 
	const T indices_;
public:
	IndexComp (const T indices) : indices_(indices) {}
	bool operator()(const size_t a, const size_t b) const
		{ return indices_[a] < indices_[b]; }
};


} // namespace dgdft
#endif // _UTILITY_HPP_
