#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"

namespace dgdft{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
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

	// Real data Fourier transform
	Int       numGridTotalR2C;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;


  DblNumVec                gkkR2C;
  DblNumVec                TeterPrecondR2C;

	// Temporary vectors that can also be used globally
	DblNumVec                inputVecR2C;     
	CpxNumVec                outputVecR2C;     


	Fourier();
	~Fourier();

	void Initialize( const Domain& dm );
};


// *********************************************************************
// Parallel FFTW interface
// *********************************************************************

/// @struct DistFourier
/// @brief Distributed memory (MPI only) parallel FFTW interface.
struct DistFourier {
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
	
	/// @brief Whether the processor according to the rank in domain
	/// participate in the FFTW calculation.
	bool             isInGrid;
	/// @brief The communicator used by parallel FFTW, should be
	/// consistent with inGrid.
	MPI_Comm         comm;

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

	/// @brief Initialize the FFTW variables.
	/// 
	/// @param[in] dm Domain for the FFTW calculation. 
	/// @param[in] numProc The number of processors actually participate
	/// in the FFTW calculation.  A processor participates in the FFTW
	/// calculation if mpirank(dm.comm) < numProc.
	void Initialize( const Domain& dm, Int numProc );
};

} // namespace dgdft


#endif // _FOURIER_HPP_
