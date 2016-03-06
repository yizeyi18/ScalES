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
/// @file iondynamics.hpp
/// @brief Geometry optimization and molecular dynamics for ions
/// @date 2015-03-05 Organize previously implemented methods
#ifndef _IONDYNAMICS_HPP_
#define _IONDYNAMICS_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "periodtable.hpp"
#include  "esdf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace dgdft{
class IonDynamics
{
private:
  /// @brief Syncec with atomic position (and velocity and force) from
  /// input. Currently atomList info is stored in the Hamiltonian
  /// classes (Hamiltonian or HamiltonianDG).
  std::vector<Atom>*   atomListPtr_;

  std::vector<Atom>    atomListSave_;

  std::string          ionMove_;

  /// @brief BarzilaiBorwein method for geometry optimization
  ///
  void BarzilaiBorweinOpt( Int ionIter );

//  void FireOpt( Int ionIter );

  void VelocityVerlet( Int ionIter );

  void NoseHoover1( Int ionIter );

  void Langevin( Int ionIter );



public:

  /// @brief Initial setup from the input parameters
  void Setup( const esdf::ESDFInputParam& esdfParam, std::vector<Atom>& atomList );

  /// @brief Main program to move the ions.
  ///
  /// Will determine both geometry optimization and molecular dynamics
  void MoveIons( Int ionIter );
 

  // *********************************************************************
  // Access functions
  // *********************************************************************
  std::vector<Atom>& AtomList() { return *atomListPtr_; }

};


} // namespace dgdft


#endif // _IONDYNAMICS_HPP_
