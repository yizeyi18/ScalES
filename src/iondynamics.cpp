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
IonDynamics::Setup	( const esdf::ESDFInputParam& esdfParam, std::vector<Atom>& atomList,
    PeriodTable& ptable )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::Setup");
#endif
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  // Read in input parameters
  atomListPtr_            = &atomList;
  ionMove_                = esdfParam.ionMove;
  ionTemperature_         = 1.0 / esdfParam.TbetaIonTemperature;
  isOutputPosition_       = esdfParam.isOutputPosition;
  isOutputVelocity_       = esdfParam.isOutputVelocity;
  isOutputXYZ_            = esdfParam.isOutputXYZ;
  MDExtrapolationType_    = esdfParam.MDExtrapolationType;
  dt_                     = esdfParam.MDTimeStep;
  Q1_                     = esdfParam.qMass;
 
  // History of atomic position
  maxHist_ = 3;  // hard coded
  atomListHist_.resize(maxHist_);
  for( Int l = 0; l < maxHist_; l++ ){
    atomListHist_[l] = atomList;
  }
 
  Int numAtom = atomList.size();
  atomMass_.Resize( numAtom );
  for(Int a=0; a < numAtom; a++) {
    Int atype = atomList[a].type;
    if (ptable.ptemap().find(atype)==ptable.ptemap().end() ){
      throw std::logic_error( "Cannot find the atom type." );
    }
    atomMass_[a]=amu2au*ptable.ptemap()[atype].params(PTParam::MASS); 
  }

  // Geometry optimization

  if( ionMove_ == "bb" ){
  }

  // Molecular dynamics
  Ekinetic_ = 0.0;
  Epot_    = 0.0;
  EconserveInit_ = 0.0;
  Econserve_ = 0.0;
  Edrift_   = 0.0;

  if( ionMove_ == "verlet" ){
    if(esdfParam.isRestartVelocity){
      statusOFS << std::endl 
        << "Read velocity information from lastVel.out. " << std::endl;

      DblNumVec atomvelRead(3*numAtom);
      if( mpirank == 0 ){
        std::fstream fin;
        fin.open("lastVel.out",std::ios::in);
        if( !fin.good() ){
          throw std::logic_error( "Cannot open lastVel.out!" );
        }
        for(Int a=0; a<numAtom; a++){
          fin>> atomvelRead[3*a+0];
          fin>> atomvelRead[3*a+1];
          fin>> atomvelRead[3*a+2];
        }
        fin.close();
      }
      // Broadcast thermostat information
      MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );

      for(Int a=0; a<numAtom; a++){
        atomList[a].vel = 
          Point3( atomvelRead[3*a], atomvelRead[3*a+1], atomvelRead[3*a+2] );
      }

      if( mpirank == 0 ){
        PrintBlock( statusOFS, "Read in Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomList[a].vel );
          }
        }
      }
    }//restart read in last velocities of atoms
//    else{
//      // Random velocity given by ion temperature
//      if( mpirank == 0 ){
//
//      }
//    }
  }

  if( ionMove_ == "nosehoover1" ){
    xi1_ = 0.0;
    vxi1_ = 0.0;
    G1_ = 0.0;
    scalefac_ = 0.0;
 
    if(esdfParam.isRestartVelocity){
      statusOFS << std::endl 
        << "Read velocity and thermostat information from lastVel.out. " << std::endl;

      DblNumVec atomvelRead(3*numAtom);
      if( mpirank == 0 ){
        std::fstream fin;
        fin.open("lastVel.out",std::ios::in);
        if( !fin.good() ){
          throw std::logic_error( "Cannot open lastVel.out!" );
        }
        for(Int a=0; a<numAtom; a++){
          fin>> atomvelRead[3*a+0];
          fin>> atomvelRead[3*a+1];
          fin>> atomvelRead[3*a+2];
        }
        fin >> vxi1_;
        fin >> xi1_;

        fin.close();
      }
      // Broadcast thermostat information
      MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &vxi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &Ekinetic_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &xi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

      for(Int a=0; a<numAtom; a++){
        atomList[a].vel = 
          Point3( atomvelRead[3*a], atomvelRead[3*a+1], atomvelRead[3*a+2] );
      }

      if( mpirank == 0 ){
        PrintBlock( statusOFS, "Read in Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomList[a].vel );
          }
        }
      }

      Print( statusOFS, "Ekinetic     = ", Ekinetic_ );
      Print( statusOFS, "vxi1         = ", vxi1_ );
      Print( statusOFS, "xi1          = ", xi1_ );

    }//restart read in last velocities of atoms
    else{
      for(Int a=0; a<numAtom; a++) 
        atomList[a].vel = Point3( 0.0, 0.0, 0.0 );
    }
 
  } // nosehoover 1


  if( ionMove_ == "nosehoover1" ){
    xi1_ = 0.0;
    vxi1_ = 0.0;
    G1_ = 0.0;
    scalefac_ = 0.0;
    Q1_ = esdfParam.qMass;
 
    if(esdfParam.isRestartVelocity){
      statusOFS << std::endl 
        << "Read velocity and thermostat information from lastVel.out, " << std::endl 
        << "overwrite the atomic position read from the input file." 
        << std::endl;

      DblNumVec atomvelRead(3*numAtom);
      if( mpirank == 0 ){
        std::fstream fin;
        fin.open("lastVel.out",std::ios::in);
        for(Int a=0; a<numAtom; a++){
          fin>> atomvelRead[3*a+0];
          fin>> atomvelRead[3*a+1];
          fin>> atomvelRead[3*a+2];
        }
        fin >> vxi1_;
        fin >> Ekinetic_;
        fin >> xi1_;

        fin.close();
      }
      // Broadcast thermostat information
      MPI_Bcast( atomvelRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &vxi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &Ekinetic_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
      MPI_Bcast( &xi1_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

      for(Int a=0; a<numAtom; a++){
        atomList[a].vel = 
          Point3( atomvelRead[3*a], atomvelRead[3*a+1], atomvelRead[3*a+2] );
      }

      if( mpirank == 0 ){
        PrintBlock( statusOFS, "Read in Atomic Velocity" );
        {
          for( Int a = 0; a < numAtom; a++ ){
            Print( statusOFS, "atom", a, "Velocity   ", atomList[a].vel );
          }
        }
      }

      Print( statusOFS, "Ekinetic     = ", Ekinetic_ );
      Print( statusOFS, "vxi1         = ", vxi1_ );
      Print( statusOFS, "xi1          = ", xi1_ );

    }//restart read in last velocities of atoms
    else{
      for(Int a=0; a<numAtom; a++) 
        atomList[a].vel = Point3( 0.0, 0.0, 0.0 );
    }
 
  } // nosehoover 1

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
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  
  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  // Update saved atomList. 0 is the latest one
  for( Int l = maxHist_-1; l > 0; l-- ){
    atomListHist_[l] = atomListHist_[l-1];
  }
  atomListHist_[0] = atomList;

  // *********************************************************************
  // Geometry optimization methods
  // *********************************************************************
  if( ionMove_ == "bb" ){
    BarzilaiBorweinOpt( ionIter );
  }


  // *********************************************************************
  // Molecular dynamics methods
  // *********************************************************************
  if( ionMove_ == "verlet" ){
    VelocityVerlet( ionIter );
  }

  if( ionMove_ == "nosehoover1" ){
    NoseHoover1( ionIter );
  }

  // Output the new coordinates
  {
    Print(statusOFS, ""); 
    Print(statusOFS, "Atom Type and Coordinates");
    Print(statusOFS, ""); 
    for(Int i=0; i < atomList.size(); i++) {
      statusOFS << std::setiosflags(std::ios::left) 
        << std::setw(LENGTH_VAR_NAME) << "Type = "
        << std::setw(6) << atomList[i].type
        << std::setiosflags(std::ios::scientific)
        << std::setiosflags(std::ios::showpos)
        << std::setw(LENGTH_VAR_NAME) << "Pos  = "
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[0]
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[1]
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].pos[2]
        << std::setw(LENGTH_VAR_NAME) << "Vel  = "
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[0]
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[1]
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC) << atomList[i].vel[2]
        << std::resetiosflags(std::ios::scientific)
        << std::resetiosflags(std::ios::showpos)
        << std::endl;
    }
  }

  // Output the position. Common to all routines
  if( mpirank == 0 ){
    if(isOutputPosition_){
      std::fstream fout;
      fout.open("lastPos.out",std::ios::out);
      if( !fout.good() ){
        throw std::logic_error( "File cannot be open!" );
      }
      for(Int i=0; i<numAtom; i++){
        fout << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomList[i].pos[2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }
      fout.close();
    }
  }


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
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  std::vector<Atom>&   atomList    = *atomListPtr_;
  // Note: atomListHist_[0] stores the same info as atomList
  std::vector<Atom>&   atomListOld = atomListHist_[1];

  Int numAtom = atomList.size();

  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomforce(numAtom);
  std::vector<Point3>  atomposOld(numAtom);
  std::vector<Point3>  atomforceOld(numAtom);

  for( Int a = 0; a < numAtom; a++ ){
    atompos[a]   = atomList[a].pos;
    atomforce[a] = atomList[a].force;
  }

  // Output the XYZ format for movie
  // Once this is written, all work associated with the current atomic
  // position is DONE.
  if( mpirank == 0 ){
    if( isOutputXYZ_ ){
      std::fstream fout;
      fout.open("MD.xyz",std::ios::out | std::ios::app) ;
      if( !fout.good() ){
        throw std::logic_error( "Cannot open MD.xyz!" );
      }
      fout << numAtom << std::endl;
      fout << "MD step # "<< ionIter << std::endl;
      for(Int a=0; a<numAtom; a++){
        fout<< std::setw(6)<< atomList[a].type
          << std::setw(16)<< atompos[a][0]*au2ang
          << std::setw(16)<< atompos[a][1]*au2ang
          << std::setw(16)<< atompos[a][2]*au2ang
          << std::endl;
      }
      fout.close();
    }
  } // if( mpirank == 0 )


  if( ionIter == 1 ){
    // FIXME 0.1 is a magic number
    for( Int a = 0; a < numAtom; a++ ){
      atompos[a]   = atompos[a] + 0.1 * atomforce[a];
    }
  }
  else{

    for( Int a = 0; a < numAtom; a++ ){
      atomposOld[a]   = atomListOld[a].pos;
      atomforceOld[a] = atomListOld[a].force;
    }

    DblNumVec sVec(DIM*numAtom), yVec(DIM*numAtom);
    SetValue( sVec, 0.0 );
    SetValue( yVec, 0.0 );

    for( Int a = 0; a < numAtom; a++ ){
      for( Int d = 0; d < DIM; d++ ){
        sVec(DIM*a+d) = atompos[a][d] - atomposOld[a][d];
        yVec(DIM*a+d) = atomforce[a][d] - atomforceOld[a][d];
      }
    }
    // Note the minus sign
    Real step = - blas::Dot( DIM*numAtom, sVec.Data(), 1, yVec.Data(), 1 ) / 
      blas::Dot( DIM*numAtom, yVec.Data(), 1, yVec.Data(), 1 );

    for( Int a = 0; a < numAtom; a++ ){
      // Update the atomic position
      atompos[a]   = atompos[a] + step * atomforce[a];
    }
  }

  // Update atomic position to store in atomListPtr_
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::BarzilaiBorweinOpt  ----- 


void
IonDynamics::VelocityVerlet	( Int ionIter )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::VelocityVerlet");
#endif
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  
  std::vector<Atom>&   atomList = *atomListPtr_;

  Int numAtom = atomList.size();


  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  // some aliasing to be compatible with implementation before
  Real& dt = dt_;
  DblNumVec& atomMass = atomMass_;
  Real  K;

  for( Int a = 0; a < numAtom; a++ ){
    atompos[a]   = atomList[a].pos;
    atomvel[a]   = atomList[a].vel;
    atomforce[a] = atomList[a].force;
  }

  
  // Propagate velocity. This is the second part of Verlet step

  for( Int a = 0; a < numAtom; a++ ){
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
  }

  // Propagate the chain. This is due to the remaining update of the
  // chain variables.
  K=0.;
  for(Int a=0; a<numAtom; a++){
    for(Int j=0; j<3; j++){
      K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
    }
  }

  // At this point, the position, velocity and thermostat variables are
  // synced at the same time step

  Ekinetic_  = K;
  Econserve_ = Ekinetic_ + Epot_;
  if(ionIter == 1)
    EconserveInit_ = Econserve_;
  Edrift_ = (Econserve_-EconserveInit_)/EconserveInit_;
  
  Print(statusOFS, "MD_Ekin    =  ", Ekinetic_);
  Print(statusOFS, "MD_Epot    =  ", Epot_);
  Print(statusOFS, "MD_Econ    =  ", Econserve_);
  Print(statusOFS, "MD_Edrift  =  ", Edrift_);

  // Output the XYZ format for movie
  // Once this is written, all work associated with the current atomic
  // position is DONE.
  if( mpirank == 0 ){
    if( isOutputXYZ_ ){
      std::fstream fout;
      fout.open("MD.xyz",std::ios::out | std::ios::app) ;
      if( !fout.good() ){
        throw std::logic_error( "Cannot open MD.xyz!" );
      }
      fout << numAtom << std::endl;
      fout << "MD step # "<< ionIter << std::endl;
      for(Int a=0; a<numAtom; a++){
        fout<< std::setw(6)<< atomList[a].type
          << std::setw(16)<< atompos[a][0]*au2ang
          << std::setw(16)<< atompos[a][1]*au2ang
          << std::setw(16)<< atompos[a][2]*au2ang
          << std::endl;
      }
      fout.close();
    }
  } // if( mpirank == 0 )
 
  // Update velocity and position
  for(Int a=0; a<numAtom; a++) {
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
    atompos[a] = atompos[a] + atomvel[a] * dt;
  }

  // Output the position and thermostat variable. 
  // These are the configuration that SCF will work on next. 
  // Hence if the job is stopped in the middle of SCF (which is most
  // likely), the MD job should continue from this configuration
  if( mpirank == 0 ){

    if(isOutputVelocity_){
      std::fstream fout_v;
      fout_v.open("lastVel.out",std::ios::out);
      if( !fout_v.good() ){
        throw std::logic_error( "File cannot be open!" );
      }
      for(Int i=0; i<numAtom; i++){
        fout_v << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }
      fout_v.close();
    }
    
  } // if( mpirank == 0 )


  // Update atomic position and velocity to store in atomListPtr_
  // NOTE: Force is NOT consistent with the position yet.
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
    atomList[a].vel = atomvel[a];
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::VelocityVerlet  ----- 

void
IonDynamics::NoseHoover1	( Int ionIter )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::NoseHoover1");
#endif
  Int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  
  std::vector<Atom>&   atomList = *atomListPtr_;

  Int numAtom = atomList.size();


  std::vector<Point3>  atompos(numAtom);
  std::vector<Point3>  atomvel(numAtom);
  std::vector<Point3>  atomforce(numAtom);

  // some aliasing to be compatible with implementation before
  Real& vxi1 = vxi1_;
  Real& xi1  = xi1_;
  Real& s = scalefac_;
  Real& dt = dt_;
  Real& T  = ionTemperature_;
  Real& Q1 = Q1_;
  DblNumVec& atomMass = atomMass_;
  Real  L =  3.0 * numAtom;  // number of degrees of freedom
  Real  K;

  for( Int a = 0; a < numAtom; a++ ){
    atompos[a]   = atomList[a].pos;
    atomvel[a]   = atomList[a].vel;
    atomforce[a] = atomList[a].force;
  }

  
  // Propagate velocity. This is the second part of Verlet step

  for( Int a = 0; a < numAtom; a++ ){
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
  }

  // Propagate the chain. This is due to the remaining update of the
  // chain variables.
  K=0.;
  for(Int a=0; a<numAtom; a++){
    for(Int j=0; j<3; j++){
      K += atomMass[a]*atomvel[a][j]*atomvel[a][j]/2.;
    }
  }

  vxi1 = vxi1+(K*2.-L*T)/Q1*dt/4.;
  xi1  = xi1+vxi1*dt/2.;
  s    = std::exp(-vxi1*dt/2.);
  for(Int a=0;a<numAtom;a++){
    atomvel[a]=s*atomvel[a];
  }
  K=K*s*s;
  vxi1 = vxi1+(2*K-L*T)/Q1*dt/4.;

  // At this point, the position, velocity and thermostat variables are
  // synced at the same time step

  Ekinetic_  = K;
  Econserve_ = Ekinetic_ + Epot_ + Q1*vxi1*vxi1/2. + L*T*xi1;
  if(ionIter == 1)
    EconserveInit_ = Econserve_;
  Edrift_ = (Econserve_-EconserveInit_)/EconserveInit_;
  
  Print(statusOFS, "MD_Ekin    =  ", Ekinetic_);
  Print(statusOFS, "MD_Epot    =  ", Epot_);
  Print(statusOFS, "MD_Econ    =  ", Econserve_);
  Print(statusOFS, "MD_Edrift  =  ", Edrift_);

  // Output the XYZ format for movie
  // Once this is written, all work associated with the current atomic
  // position is DONE.
  if( mpirank == 0 ){
    if( isOutputXYZ_ ){
      std::fstream fout;
      fout.open("MD.xyz",std::ios::out | std::ios::app) ;
      if( !fout.good() ){
        throw std::logic_error( "Cannot open MD.xyz!" );
      }
      fout << numAtom << std::endl;
      fout << "MD step # "<< ionIter << std::endl;
      for(Int a=0; a<numAtom; a++){
        fout<< std::setw(6)<< atomList[a].type
          << std::setw(16)<< atompos[a][0]*au2ang
          << std::setw(16)<< atompos[a][1]*au2ang
          << std::setw(16)<< atompos[a][2]*au2ang
          << std::endl;
      }
      fout.close();
    }
  } // if( mpirank == 0 )
 
  // Now prepare the variable for the next evalation of the force

  // Propagate the chain again
  K = Ekinetic_;
  vxi1 = vxi1+(K*2.-L*T)/Q1*dt/4.;
  xi1  = xi1+vxi1*dt/2.;
  s    = std::exp(-vxi1*dt/2.);
  for(Int a=0;a<numAtom;a++){
    atomvel[a]=s*atomvel[a];
  }
  vxi1 = vxi1+(2*K-L*T)/Q1*dt/4.;

  // Update velocity and position
  for(Int a=0; a<numAtom; a++) {
    atomvel[a] = atomvel[a] + atomforce[a]*dt*0.5/atomMass[a]; 
    atompos[a] = atompos[a] + atomvel[a] * dt;
  }

  // Output the velocity and thermostat variable. 
  // These are the configuration that SCF will work on next. 
  // Hence if the job is stopped in the middle of SCF (which is most
  // likely), the MD job should continue from this configuration
  if( mpirank == 0 ){
    
    if(isOutputVelocity_){
      std::fstream fout_v;
      fout_v.open("lastVel.out",std::ios::out);
      if( !fout_v.good() ){
        throw std::logic_error( "File cannot be open!" );
      }
      for(Int i=0; i<numAtom; i++){
        fout_v << std::setiosflags(std::ios::scientific)
          << std::setiosflags(std::ios::showpos)
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][0]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][1]
          << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< atomvel[i][2]
          << std::resetiosflags(std::ios::scientific)
          << std::resetiosflags(std::ios::showpos)
          << std::endl;
      }

      fout_v << std::setiosflags(std::ios::scientific)
        << std::setiosflags(std::ios::showpos)
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< vxi1 << std::endl
        << std::setw(LENGTH_VAR_DATA) << std::setprecision(LENGTH_DBL_PREC)<< xi1 << std::endl
        << std::resetiosflags(std::ios::scientific)
        << std::resetiosflags(std::ios::showpos)
        << std::endl;
      
      fout_v.close();
    }
  } // if( mpirank == 0 )


  // Update atomic position and velocity to store in atomListPtr_
  // NOTE: Force is NOT consistent with the position yet.
  for(Int a = 0; a < numAtom; a++){
    atomList[a].pos = atompos[a];
    atomList[a].vel = atomvel[a];
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::NoseHoover1  ----- 


void
IonDynamics::DensityExtrapolateCoefficient	( Int ionIter, DblNumVec& coef )
{
#ifndef _RELEASE_
	PushCallStack("IonDynamics::DensityExtrapolateCoefficient");
#endif
  std::vector<Atom>&   atomList = *atomListPtr_;
  Int numAtom = atomList.size();

  coef.Resize( maxHist_ );
  SetValue(coef, 0.0);
  if( MDExtrapolationType_ == "linear" ){
    coef[0] = 2.0;
    coef[1] = -1.0;
  }
  else if( MDExtrapolationType_ == "quadratic" ){
    // WHY 4 rather than 2?
    if( ionIter < 3 ){
      coef[0] = 2.0;
      coef[1] = -1.0;
    }
    else{
      coef[0] = 3.0;
      coef[1] = -3.0;
      coef[2] = 1.0;
    }
  }
  else if( MDExtrapolationType_ == "dario" ){
    if( ionIter < 3 ){
      coef[0] = 2.0;
      coef[1] = -1.0;
    }
    else{
      // FIXME
      // Dario extrapolation not working yet


      // Update the density through quadratic extrapolation
      // Dario CPC 118, 31 (1999)
      // huwei 20150923 
      // Compute the coefficient a and b
      Real a11 = 0.0;
      Real a22 = 0.0;
      Real a12 = 0.0;
      Real a21 = 0.0;
      Real b1 = 0.0;
      Real b2 = 0.0;

      std::vector<Point3>  atemp1(numAtom);
      std::vector<Point3>  atemp2(numAtom);
      std::vector<Point3>  atemp3(numAtom);

      for( Int i = 0; i < numAtom; i++ ){
        atemp1[i] = atomListHist_[0][i].pos - atomListHist_[1][i].pos;
        atemp2[i] = atomListHist_[1][i].pos - atomListHist_[2][i].pos;
        atemp3[i] = atomListHist_[0][i].pos - atomList[i].pos;
      }

      for( Int i = 0; i < numAtom; i++ ){

        a11 += atemp1[i][0]*atemp1[i][0]+atemp1[i][1]*atemp1[i][1]+atemp1[i][2]*atemp1[i][2];
        a12 += atemp1[i][0]*atemp2[i][0]+atemp1[i][1]*atemp2[i][1]+atemp1[i][2]*atemp2[i][2];
        a22 += atemp2[i][0]*atemp2[i][0]+atemp2[i][1]*atemp2[i][1]+atemp2[i][2]*atemp2[i][2];
        a21 = a12;
        b1 += 0.0-atemp3[i][0]*atemp1[i][0]-atemp3[i][1]*atemp1[i][1]-atemp3[i][2]*atemp1[i][2];
        b2 += 0.0-atemp3[i][0]*atemp2[i][0]-atemp3[i][1]*atemp2[i][1]-atemp3[i][2]*atemp2[i][2];
      }


      Real detA = a11*a22 - a12*a21;
      Real aA = (b1*a22-b2*a12)/detA;
      Real bA = (b2*a11-b2*a21)/detA;


//      statusOFS << "info"<< std::endl;
//      statusOFS << a11 << ", " << a12 << ", " << a21 << ", " << a22 << ", " << detA << ", " << b1 
//        << ", " << b2 << std::endl;

//      denCurVec(ii) = den0(ii) + aA * ( den0(ii) - den1(ii) ) + bA * ( den1(ii) - den2(ii) );
      coef[0] = 1.0 + aA;
      coef[1] = -aA + bA;
      coef[2] = -bA;
    }
  }
  else{
    throw std::logic_error( "Currently three extrapolation types are supported!" );
  }


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method IonDynamics::DensityExtrapolateCoefficient  ----- 

} // namespace dgdft



#endif // _IONDYNAMICS_CPP_
