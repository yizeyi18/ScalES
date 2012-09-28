#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment_impl.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"

namespace dgdft{

struct Fourier {
	Domain           domain;
  bool             isPrepared;
	//	TODO MPI
	Int              numGridTotal;
//	Int              localN0;
//	Int              localN0Start;

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

};

void SetupFourier( Fourier& fft, const Domain& dm );

} // namespace dgdft


#endif // _FOURIER_HPP_
