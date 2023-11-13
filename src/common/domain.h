//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin

/// @file domain.hpp
/// @brief Computational domain.
/// @date 2012-08-01
#ifndef _DOMAIN_HPP_
#define _DOMAIN_HPP_

#include  "environment.h"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"

namespace scales{

struct Domain
{
  Point3       length;                          // length
  Point3       posStart;                        // starting position
  Index3       numGrid;                         // number of coarse grids points in each direction
  Index3       numGridFine;                     // number of fine grids points in each direction
  MPI_Comm     comm;                            // MPI Communicator
  MPI_Comm     rowComm;
  MPI_Comm     colComm;

  Domain()
  { 
    length        = Point3( 0.0, 0.0, 0.0 );
    posStart      = Point3( 0.0, 0.0, 0.0 );
    numGrid       = Index3( 0, 0, 0 );
    numGridFine   = Index3( 0, 0, 0 );

    comm    = MPI_COMM_WORLD; 
    rowComm = MPI_COMM_WORLD;
    colComm = MPI_COMM_WORLD;
  }

  ~Domain(){}

  Real Volume() const { return length[0] * length[1] * length[2]; }
  Int  NumGridTotal() const { return numGrid[0] * numGrid[1] * numGrid[2]; }
  Int  NumGridTotalFine() const { return numGridFine[0] * numGridFine[1] * numGridFine[2]; }

};


} // namespace scales


#endif // _DOMAIN_HPP
