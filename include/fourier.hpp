#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"

namespace dgdft{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

struct Fourier {
	Domain           domain;
  bool             isInitialized;
	Int              numGridTotal;

  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;

  unsigned  plannerFlag;

	// Laplacian operator related
  DblNumVec                gkk;
	std::vector<CpxNumVec>   ik;
  DblNumVec                TeterPrecond;

	// Temporary vectors that can also be used globally
	CpxNumVec                inputComplexVec;     
	CpxNumVec                outputComplexVec;     

	//  TODO Real Fourier transform
//  Int  halfsize;                                // For the usage of r2c and c2r
//  fftw_plan cpx2real;
//  fftw_plan real2cpx;
//  DblNumVec gkkhalf;
//  DblNumVec prechalf;
//  Real    * in_dbl;

	Fourier();
	~Fourier();

	void Initialize( const Domain& dm );

};


// *********************************************************************
// Parallel FFTW interface
// *********************************************************************

struct DistFourier {
	// The communicator in the domain may be different from the global communicator
	// This is because FFTW cannot use employ many processors. 
	Domain           domain;                      
  bool             isInitialized;
	Int              numGridTotal;
  Int              numGridLocal;	
	ptrdiff_t        localNz;
	ptrdiff_t        localNzStart;
	// numAllocLocal is the size for the FFTW vectors.
	// numAllocLocal is close but may not be exactly the same as numGridLocal, which is
	// localNz * Ny * Nx.  This is because FFTW may need some intermediate
	// space.  For more information see FFTW's manual.
	ptrdiff_t        numAllocLocal;

  // plans
  fftw_plan        backwardPlan;
  fftw_plan        forwardPlan;

  unsigned         plannerFlag;

	// Laplacian operator related
  DblNumVec                gkkLocal;
	std::vector<CpxNumVec>   ikLocal;
  DblNumVec                TeterPrecondLocal;

	// Temporary vectors that can also be used globally
	CpxNumVec                inputComplexVecLocal;     
	CpxNumVec                outputComplexVecLocal;     

	DistFourier();
	~DistFourier();

	void Initialize( const Domain& dm );
};

} // namespace dgdft


#endif // _FOURIER_HPP_
