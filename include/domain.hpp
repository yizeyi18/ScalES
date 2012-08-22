#ifndef _DOMAIN_HPP_
#define _DOMAIN_HPP_

#include  "vec3t.hpp"
#include  "numvec.hpp"

namespace dgdft{

	// Type of grids
	enum {
		UNIFORM = 0,
		LGL     = 1 
	};

	struct Domain
	{
		Point3       length;                          // length
		Point3       posStart;                        // starting position
		Index3       numGrid;                         // number of grids points in each direction
		Int          typeGrid;                        // Type of grids: LGL / Uniform
		std::vector<NumVec<Real> >   grid(DIM);       // Grid points along each dimension
		std::vector<NumVec<Real> >   weight(DIM);     // Integration weight along each dimension
		MPI_Comm     comm;                            // MPI Communicator
	};

} // namespace dgdft


#endif
