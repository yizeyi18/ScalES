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
/// @file iondynamics.cpp
/// @brief Geometry optimization and molecular dynamics for ions
/// @date 2015-03-05 Organize previously implemented methods
#ifndef _IONDYNAMICS_CPP_
#define _IONDYNAMICS_CPP_

#include  "iondynamics.hpp"

namespace dgdft{

void
IonDynamics::Setup	( const esdf::ESDFInputParam& esdfParam, std::vector<Atom>& atomList )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::Setup");
#endif
  atomListPtr_    = &atomList;
   
  ionMove_        = esdfParam.ionMove;


  // Print out the force
//  PrintBlock( statusOFS, "Atomic Force" );
//
//  Int numAtom = atomListPtr_->size();
//
//  for( Int a = 0; a < numAtom; a++ ){
//    Print( statusOFS, "atom", a, "force", (*atomListPtr_)[a].force );
//  }
//  statusOFS << std::endl;

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::Setup  ----- 


void
IonDynamics::MoveIons	( Int ionIter )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::MoveIons");
#endif

  // *********************************************************************
  // Geometry optimization methods
  // *********************************************************************
  if( ionMove_ == "bb" ){
    BarzilaiBorweinOpt( ionIter );
  }


  // *********************************************************************
  // Molecular dynamics methods
  // *********************************************************************
//  if( ionMove_ == "verlet" ){
//    VelocityVerlet( ionIter );
//  }
//
//  if( ionMove_ == "nosehoover1" ){
//    NoseHoover1( ionIter );
//  }
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::MoveIons  ----- 


void
IonDynamics::BarzilaiBorweinOpt	( Int ionIter )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::BarzilaiBorweinOpt");
#endif

  std::vector<Atom>&   atomList = *atomListPtr_;

  Int numAtom = atomList.size();

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atomposOld(numAtom);
  std::vector<Point3>  atomforceOld(numAtom);

  for( Int i = 0; i < numAtom; i++ ){
    atompos[i]   = atomList[i].pos;
    atomforce[i] = atomList[i].force;
  }

  if( ionIter == 1 ){
    // FIXME 0.1 is a magic number
    for( Int i = 0; i < numAtom; i++ ){
      atompos[i]   = atompos[i] + 0.1 * atomforce[i];
    }
  }
  else{

    for( Int i = 0; i < numAtom; i++ ){
      atomposOld[i]   = atomListSave_[i].pos;
      atomforceOld[i] = atomListSave_[i].force;
    }

    DblNumVec sVec(DIM*numAtom), yVec(DIM*numAtom);
    SetValue( sVec, 0.0 );
    SetValue( yVec, 0.0 );

    for( Int i = 0; i < numAtom; i++ ){
      for( Int d = 0; d < DIM; d++ ){
        sVec(DIM*i+d) = atompos[i][d] - atomposOld[i][d];
        yVec(DIM*i+d) = atomforce[i][d] - atomforceOld[i][d];
      }
    }
    // Note the minus sign
    Real step = - blas::Dot( DIM*numAtom, sVec.Data(), 1, yVec.Data(), 1 ) / 
      blas::Dot( DIM*numAtom, yVec.Data(), 1, yVec.Data(), 1 );

    for( Int i = 0; i < numAtom; i++ ){
      // Update the atomic position
      atompos[i]   = atompos[i] + step * atomforce[i];
    }
  }

  // Update atomic position to store in atomListPtr_
  for(Int i = 0; i < numAtom; i++){
    atomList[i].pos = atompos[i];
  }

  {
    Print(statusOFS, ""); 
    Print(statusOFS, "Atom Type and Coordinates");
    Print(statusOFS, ""); 
    for(Int i=0; i < atomList.size(); i++) {
      Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
    }
  }

  // Overwrite the stored atom list
  atomListSave_ = atomList;   // make a copy


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::BarzilaiBorweinOpt  ----- 


} // namespace dgdft



#endif // _IONDYNAMICS_CPP_
