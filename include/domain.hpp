#ifndef _DOMAIN_HPP_
#define _DOMAIN_HPP_

#include  "environment_impl.hpp"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"

namespace dgdft{
// TODO decide where to put grid and weight

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
	Int          typeGrid;                        // Type of grids: LGL / UNIFORM
//	std::vector<NumVec<Real> >   grid;            // Grid points along each dimension
//	std::vector<NumVec<Real> >   weight;          // Integration weight along each dimension
	MPI_Comm     comm;                            // MPI Communicator

	Domain()
	{ 
		length        = Point3( 0.0, 0.0, 0.0 );
		posStart      = Point3( 0.0, 0.0, 0.0 );
		numGrid       = Index3( 0, 0, 0 );

		typeGrid = UNIFORM; 
		comm = MPI_COMM_WORLD; 
//		grid.resize(DIM);
//		weight.resize(DIM);
	}

	~Domain(){}

	Real Volume() const { return length[0] * length[1] * length[2]; }
	Int  NumGridTotal() const { return numGrid[0] * numGrid[1] * numGrid[2]; }
};

// TODO: Serialize
//int serialize(const Domain&, ostream&, const vector<int>&);
//int deserialize(Domain&, istream&, const vector<int>&);
//int combine(Domain&, Domain&);


} // namespace dgdft


#endif // _DOMAIN_HPP
