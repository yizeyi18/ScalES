/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Author: Lin Lin
	 
   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file plobpcg++.hpp
/// @brief Interface to parallel LOBPCG.
/// @date 2014-04-25 Parallel eigensolver
#ifndef _PLOBPCG_HPP_
#define _PLOBPCG_HPP_

#include "environment.hpp"

#include "fortran_matrix.h"
#include "fortran_interpreter.h"
#include "lobpcg.h"
#include "multivector.h"
#include "interpreter.h"

#ifndef BlopexInt
#define BlopexInt   Int  
#endif

/*--------------------------------------------------------------------------
 * Accessor functions for the Multi_Vector structure
 *--------------------------------------------------------------------------*/

#define parallel_Multi_VectorData(vector)      ((vector) -> data)
#define parallel_Multi_VectorSize(vector)      ((vector) -> size)
#define parallel_Multi_VectorOwnsData(vector)  ((vector) -> owns_data)
#define parallel_Multi_VectorNumVectors(vector) ((vector) -> num_vectors)


namespace dgdft{ 
namespace LOBPCG{
  //----------------------------------------------------------------
  class parallel_Multi_Vector {
    public:
      Scalar*        data;
      BlopexInt      size;
      BlopexInt      owns_data;
      BlopexInt      num_vectors;  /* the above "size" is size of one vector */

      BlopexInt      num_active_vectors;
      BlopexInt*     active_indices;   /* indices of active vectors; 0-based notation */
  };

} // namespace LOBPCG 
} // namespace dgdft

#endif // _PLOBPCG_HPP_

