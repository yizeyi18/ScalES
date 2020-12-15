/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Weile Jia

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
/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @date 2012-08-10
#include "periodtable.hpp"
#include "esdf.hpp"
#include "utility.hpp"

#ifdef DG_HAS_CONFIG
#include "config.hpp"
#endif

#include <sstream>

namespace  dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::esdf;

// *********************************************************************
// PTEntry
// *********************************************************************

Int serialize(const PTEntry& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.params,  os, mask);
  serialize(val.samples, os, mask);
  serialize(val.weights, os, mask);
  serialize(val.types,   os, mask);
  serialize(val.cutoffs, os, mask);
  return 0;
}

Int deserialize(PTEntry& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.params,  is, mask);
  deserialize(val.samples, is, mask);
  deserialize(val.weights, is, mask);
  deserialize(val.types,   is, mask);
  deserialize(val.cutoffs, is, mask);
  return 0;
}

Int combine(PTEntry& val, PTEntry& ext)
{
  ErrorHandling( "Combine operation is not implemented" );
  return 0;
}


// *********************************************************************
// PeriodTable
// *********************************************************************

// PTEntry / PTSample is the old format for reading the binary file, but works
// in the new format too, as long as the pseudopotential is associated
// with atoms rather than species.
// 
// There is also room for optimizing the rcps parameter for each species 
// (maybe solving a least squares problems by matlab and store the default value in a table?)
//
void PeriodTable::Setup( )
{
  std::vector<Int> all(1,1);

  std::istringstream iss;  
  
  {
    // all the readins are in the samples in the old version, 
    // now in the new version, I should readin something else. 
    // the readin are reading in a sequence of numbers, which
    // is used to construct the ptemap_ struct.
    // Read from UPF file
    MPI_Barrier(MPI_COMM_WORLD);
    int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    PTEntry tempEntry;
    if(mpirank==0) {
      for( int i = 0; i < esdfParam.pspFile.size(); i++){
        int atom;
        ReadUPF(esdfParam.pspFile[i], &tempEntry, &atom);
        std::map <Int,PTEntry> :: iterator it = ptemap_.end();
        ptemap_.insert(it, std::pair<Int, PTEntry>(atom, tempEntry));
      }
    }

    // implement the MPI Bcast of the ptemap_, now we are doing all processors readin
    std::stringstream vStream;
    std::stringstream vStreamTemp;
    int vStreamSize;

    serialize( ptemap_, vStream, all);

    if( mpirank == 0)
      vStreamSize = Size( vStream );

    MPI_Bcast( &vStreamSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> sstr;
    sstr.resize( vStreamSize );

    if( mpirank == 0)
      vStream.read( &sstr[0], vStreamSize );

    MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, 0, MPI_COMM_WORLD );
    vStreamTemp.write( &sstr[0], vStreamSize );
    deserialize( ptemap_, vStreamTemp, all);
  }


  // Setup constant private variable parameters
  /*
  if( esdfParam.pseudoType == "HGH" ){
    ptsample_.RADIAL_GRID       = 0;
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;
    ptsample_.RHOATOM           = -999;
    ptsample_.DRV_RHOATOM       = -999;
    ptsample_.NONLOCAL          = 3;
  }
  */
  if( esdfParam.pseudoType == "oncv" ){
    ptsample_.RADIAL_GRID       = 0;

    ptsample_.VLOCAL            = 1;
    ptsample_.DRV_VLOCAL        = 2;

    ptsample_.RHOATOM           = 3;
    ptsample_.DRV_RHOATOM       = 4;
    ptsample_.NONLOCAL          = 5;


    // FIXME PSEUDO_CHARGE and DRV_PSEUDO_CHARGE are not used anymore.
    // delete them in the future
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;

  }

  // Common so far for all pseudopotential
  {
    pttype_.RADIAL            = 9;
    pttype_.PSEUDO_CHARGE     = 99;
    pttype_.RHOATOM           = 999;
    pttype_.VLOCAL            = 9999;
    pttype_.L0                = 0;
    pttype_.L1                = 1;
    pttype_.L2                = 2;
    pttype_.L3                = 3;
    pttype_.SPINORBIT_L1      = -1;
    pttype_.SPINORBIT_L2      = -2;
    pttype_.SPINORBIT_L3      = -3;
  }

  {
    ptparam_.ZNUC   = 0;
    ptparam_.MASS   = 1;
    ptparam_.ZION   = 2;
    ptparam_.RGAUSSIAN = 3;
  }

  // Processing of Vlocal data
  for(std::map<Int,PTEntry>::iterator mi=ptemap_.begin(); mi!=ptemap_.end(); mi++) {
    Int type = (*mi).first;    
    PTEntry& ptcur = (*mi).second;
    DblNumVec& params = ptcur.params;
    DblNumMat& samples = ptcur.samples;
    Int nspl = samples.m();
    Real Zion = params(ptparam_.ZION);
    Real RGaussian = params(ptparam_.RGAUSSIAN);
    DblNumVec rad(nspl, false, samples.VecData(ptsample_.RADIAL_GRID));
    DblNumVec vlocal(nspl, false, samples.VecData(ptsample_.VLOCAL));
    // Remove the pseudocharge contribution
    for(Int i = 0; i < rad.m(); i++){
      if( rad[i] == 0 )
        vlocal[i] += Zion / RGaussian * 2.0 / std::sqrt(PI);
      else
        vlocal[i] += Zion / rad[i] * std::erf(rad[i] / RGaussian);
    }
    //      statusOFS << "RGaussian = " << RGaussian << std::endl;
    //      statusOFS << "VLocal SR for type " << type << " = " << vlocal << std::endl;
  }


  //create splines
  for(std::map<Int,PTEntry>::iterator mi=ptemap_.begin(); mi!=ptemap_.end(); mi++) {
    Int type = (*mi).first;    
    PTEntry& ptcur = (*mi).second;
    DblNumVec& params = ptcur.params;
    DblNumMat& samples = ptcur.samples;
    std::map< Int, std::vector<DblNumVec> > spltmp;
    for(Int g=1; g<samples.n(); g++) {
      Int nspl = samples.m();
      DblNumVec rad(nspl, true, samples.VecData(0));
      DblNumVec a(nspl, true, samples.VecData(g));
      DblNumVec b(nspl), c(nspl), d(nspl);
      //create splines
      spline(nspl, rad.Data(), a.Data(), b.Data(), c.Data(), d.Data());
      std::vector<DblNumVec> aux(5);
      aux[0] = rad;      aux[1] = a;      aux[2] = b;      aux[3] = c;      aux[4] = d;
      spltmp[g] = aux;
    }
    splmap_[type] = spltmp;
  }
}         // -----  end of method PeriodTable::Setup  ----- 

void
PeriodTable::CalculatePseudoCharge    (
    const Atom& atom, 
    const Domain& dm,
    const std::vector<DblNumVec>& gridpos,        
    SparseVec& res )
{
  Int type   = atom.type;
  Point3 pos = atom.pos;
  Point3 Ls  = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns  = dm.numGridFine;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutPseudoCharge( type );

  // Initialize
  {
    SparseVec empty;
    res = empty;
  }

  // Compute the minimal distance of the atom to this set of grid points
  // and determine whether to continue 

  std::vector<DblNumVec>  dist(DIM);

  Point3 minDist;
  for( Int d = 0; d < DIM; d++ ){
    dist[d].Resize( gridpos[d].m() );

    minDist[d] = Rzero;
    for( Int i = 0; i < gridpos[d].m(); i++ ){
      dist[d](i) = gridpos[d](i) - pos[d];
      dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
      if( std::abs( dist[d](i) ) < minDist[d] )
        minDist[d] = std::abs( dist[d](i) );
    }
  }
  if( std::sqrt( dot(minDist, minDist) ) <= Rzero ){
    // At least one grid point is within Rzero
    Int irad = 0;
    std::vector<Int>  idx;
    std::vector<Real> rad;
    std::vector<Real> xx, yy, zz;
    for(Int k = 0; k < gridpos[2].m(); k++)
      for(Int j = 0; j < gridpos[1].m(); j++)
        for(Int i = 0; i < gridpos[0].m(); i++){
          Real dtmp = std::sqrt( 
              dist[0](i) * dist[0](i) +
              dist[1](j) * dist[1](j) +
              dist[2](k) * dist[2](k) );

          if( dtmp <= Rzero ) {
            idx.push_back(irad);
            rad.push_back(dtmp);
            xx.push_back(dist[0](i));        
            yy.push_back(dist[1](j));        
            zz.push_back(dist[2](k));
          }
          irad++;
        } // for (i)



    Int idxsize = idx.size();
    //
    std::vector<DblNumVec>& valspl = spldata[ptsample_.PSEUDO_CHARGE]; 
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
        valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
    //
    std::vector<DblNumVec>& derspl = spldata[ptsample_.DRV_PSEUDO_CHARGE];
    std::vector<Real> der(idxsize,0.0);

    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
        derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
    //
    IntNumVec iv(idx.size(), true, &(idx[0])); 
    DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
    //
    for(Int g=0; g<idx.size(); g++) {
      dv(g, VAL) = val[g];
      if( rad[g]> MIN_RADIAL ) {
        dv(g, DX) = der[g] * xx[g]/rad[g];
        dv(g, DY) = der[g] * yy[g]/rad[g];
        dv(g, DZ) = der[g] * zz[g]/rad[g];
      } else {
        dv(g, DX) = 0;
        dv(g, DY) = 0;
        dv(g, DZ) = 0;
      }
    }
    res = SparseVec(iv,dv);
  } // if (norm(minDist) <= Rzero )

  return ;
}         // -----  end of method PeriodTable::CalculatePseudoCharge  ----- 


void
PeriodTable::CalculatePseudoCharge    (
    const Atom& atom, 
    const Domain& dm,
    const NumTns<std::vector<DblNumVec> >& gridposElem,
    NumTns<SparseVec>& res )
{
  Index3 numElem( gridposElem.m(), gridposElem.n(), gridposElem.p() );

  res.Resize( numElem[0], numElem[1], numElem[2] );

  for( Int elemk = 0; elemk < numElem[2]; elemk++ )
    for( Int elemj = 0; elemj < numElem[1]; elemj++ )
      for( Int elemi = 0; elemi < numElem[0]; elemi++ ){
        CalculatePseudoCharge( atom, dm, 
            gridposElem(elemi, elemj, elemk),
            res( elemi, elemj, elemk ) );
      } // for (elemi)


  return ;
}         // -----  end of method PeriodTable::CalculatePseudoCharge  ----- 


//---------------------------------------------

void
PeriodTable::CalculateNonlocalPP    ( 
    const Atom& atom, 
    const Domain& dm, 
    const std::vector<DblNumVec>& gridpos,
    std::vector<NonlocalPP>&      vnlList )
{
  Point3 Ls       = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns       = dm.numGrid;
  vnlList.clear();

#ifndef _NO_NONLOCAL_ // Nonlocal potential is used. Debug option
  Int type   = atom.type;
  Point3 pos = atom.pos;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutNonlocal( type );    

  // Initialize
  // First count all the pseudopotentials
  Int numpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }


  {
    vnlList.resize( numpp );
    SparseVec empty;
    for( Int p = 0; p < numpp; p++ ){
      vnlList[p] = NonlocalPP( empty, 0.0 );
    }
  }

  // Compute the minimal distance of the atom to this set of grid points
  // and determine whether to continue 

  std::vector<DblNumVec>  dist(DIM);

  Point3 minDist;
  for( Int d = 0; d < DIM; d++ ){
    dist[d].Resize( gridpos[d].m() );

    minDist[d] = Rzero;
    for( Int i = 0; i < gridpos[d].m(); i++ ){
      dist[d](i) = gridpos[d](i) - pos[d];
      dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
      if( std::abs( dist[d](i) ) < minDist[d] )
        minDist[d] = std::abs( dist[d](i) );
    }
  }

  if( std::sqrt( dot(minDist, minDist) ) <= Rzero ){
    // At least one grid point is within Rzero
    Int irad = 0;
    std::vector<Int>  idx;
    std::vector<Real> rad;
    std::vector<Real> xx, yy, zz;
    for(Int k = 0; k < gridpos[2].m(); k++)
      for(Int j = 0; j < gridpos[1].m(); j++)
        for(Int i = 0; i < gridpos[0].m(); i++){
          Real dtmp = std::sqrt( 
              dist[0](i) * dist[0](i) +
              dist[1](j) * dist[1](j) +
              dist[2](k) * dist[2](k) );

          if( dtmp < Rzero ) {
            idx.push_back(irad);
            rad.push_back(dtmp);
            xx.push_back(dist[0](i));        
            yy.push_back(dist[1](j));        
            zz.push_back(dist[2](k));
          }
          irad++;
        } // for (i)

    Int idxsize = idx.size();
    //process non-local pseudopotential one by one
    Int cntpp = 0;
    for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
      Real wgt = ptentry.weights(g);
      Int typ = ptentry.types(g);
      //
      std::vector<DblNumVec>& valspl = spldata[g]; 
      std::vector<Real> val(idxsize,0.0);
      seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
      //
      std::vector<DblNumVec>& derspl = spldata[g+1]; 
      std::vector<Real> der(idxsize,0.0);
      seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
      //--
      if(typ==pttype_.L0) {
        Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        //
        for(Int g=0; g<idx.size(); g++) {
          if( rad[g]>MIN_RADIAL ) {
            dv(g,VAL) = coef * val[g];
            dv(g,DX) = coef * der[g] * xx[g]/rad[g];
            dv(g,DY) = coef * der[g] * yy[g]/rad[g];
            dv(g,DZ) = coef * der[g] * zz[g]/rad[g];
          } else {
            dv(g,VAL) = coef * val[g];
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      } // if(typ==pttype_.L0);

      if(typ==pttype_.L1) {
        Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          for(Int g=0; g<idx.size(); g++) {
            if( rad[g]> MIN_RADIAL ) {
              dv(g,VAL) = coef*( (xx[g]/rad[g]) * val[g] );
              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(xx[g]/rad[g]) + val[g]/rad[g] );
              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(yy[g]/rad[g])                 );
              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(zz[g]/rad[g])                 );
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = coef*der[g];
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]> MIN_RADIAL) {
              dv(g,VAL) = coef*( (yy[g]/rad[g]) * val[g] );
              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(xx[g]/rad[g])                 );
              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(yy[g]/rad[g]) + val[g]/rad[g] );
              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(zz[g]/rad[g])                 );
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = coef*der[g];
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              dv(g,VAL) = coef*( (zz[g]/rad[g]) * val[g] );
              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(xx[g]/rad[g])                 );
              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(yy[g]/rad[g])                 );
              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(zz[g]/rad[g]) + val[g]/rad[g] );
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = coef*der[g];
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
      } // if(typ==pttype_.L1)

      if(typ==pttype_.L2) {
        // d_z2
        {
          Real coef = 1.0/4.0*sqrt(5.0/PI); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(-xx[g]*xx[g]-yy[g]*yy[g]+2.0*zz[g]*zz[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*(-6.0 * xx[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
              Ylm(2) = coef*(-6.0 * yy[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
              Ylm(3) = coef*( 6.0 * zz[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)) / pow(rad[g], 4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_yz
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives

          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(yy[g]*zz[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
              Ylm(2) = coef*(     zz[g]*(pow(zz[g],2.0)+pow(xx[g],2.0)-pow(yy[g],2.0)) / 
                  pow(rad[g],4.0));
              Ylm(3) = coef*(     yy[g]*(pow(yy[g],2.0)+pow(xx[g],2.0)-pow(zz[g],2.0)) /
                  pow(rad[g],4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_xz
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives

          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(zz[g]*xx[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*(     zz[g]*(pow(zz[g],2.0)-pow(xx[g],2.0)+pow(yy[g],2.0)) / 
                  pow(rad[g],4.0));
              Ylm(2) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
              Ylm(3) = coef*(     xx[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)-pow(zz[g],2.0)) /
                  pow(rad[g],4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];

            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_xy
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(xx[g]*yy[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*(     yy[g]*(pow(yy[g],2.0)-pow(xx[g],2.0)+pow(zz[g],2.0)) / 
                  pow(rad[g],4.0));
              Ylm(2) = coef*(     xx[g]*(pow(xx[g],2.0)-pow(yy[g],2.0)+pow(zz[g],2.0)) /
                  pow(rad[g],4.0));
              Ylm(3) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_x^2-y^2
        {
          Real coef = 1.0/4.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(xx[g]*xx[g]-yy[g]*yy[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*( 2.0*xx[g]*(2.0*pow(yy[g],2.0)+pow(zz[g],2.0)) / 
                  pow(rad[g],4.0));
              Ylm(2) = coef*(-2.0*yy[g]*(2.0*pow(xx[g],2.0)+pow(zz[g],2.0)) /
                  pow(rad[g],4.0));
              Ylm(3) = coef*(-2.0*zz[g]*(pow(xx[g],2.0) - pow(yy[g],2.0)) / pow(rad[g],4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
      } // if(typ==pttype_.L2)

      // FIXME: The derivative at r=0 for the f orbital MAY NOT BE CORRECT.
      // LLIN: 10/21/2013
      if(typ==pttype_.L3) {
        // f_z3
        {
          Real coef = 1.0/4.0*sqrt(7.0/PI); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*zz[g]*(-3.*x2 - 3.*y2 + 2.*z2) / r3;
              Ylm(1) = coef*3.*xx[g]*zz[g]*(x2 + y2 - 4.*z2) / r5;
              Ylm(2) = coef*3.*yy[g]*zz[g]*(x2 + y2 - 4.*z2) / r5;
              Ylm(3) = -coef*3.*(x2 + y2)*(x2 + y2 - 4.*z2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_y(3xx-yy)
        {
          Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*yy[g]*(3.*x2 - y2) / r3;
              Ylm(1) = -coef*3.*xx[g]*yy[g]*(x2 - 3.*y2 - 2.*z2) / r5;
              Ylm(2) = coef*3.*(x2*x2 - y2*z2 + x2*(-3.*y2+z2)) / r5;
              Ylm(3) = coef*3.*yy[g]*zz[g]*(-3.*x2 + y2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_x(xx-3yy)
        {
          Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*xx[g]*(x2 - 3.*y2) / r3;
              Ylm(1) = coef*3.*(-y2*(y2+z2) + x2*(3.*y2+z2)) / r5;
              Ylm(2) = coef*3.*xx[g]*yy[g]*(-3.*x2 + y2 - 2.*z2) / r5;
              Ylm(3) = -coef*3.*xx[g]*zz[g]*(x2 - 3.*y2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_z(xx-yy)
        {
          Real coef = 1.0/4.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*zz[g]*(x2 - y2) / r3;
              Ylm(1) = coef*xx[g]*zz[g]*(-x2 + 5.*y2 + 2.*z2) / r5;
              Ylm(2) = coef*yy[g]*zz[g]*(-5.*x2 + y2 - 2.*z2) / r5;
              Ylm(3) = coef*(x2 - y2)*(x2 + y2 - 2.*z2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_xyz
        {
          Real coef = 1.0/2.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*xx[g]*yy[g]*zz[g] / r3;
              Ylm(1) = coef*yy[g]*zz[g]*(-2.*x2 + y2 + z2) / r5;
              Ylm(2) = coef*xx[g]*zz[g]*(x2 - 2.*y2 + z2) / r5;
              Ylm(3) = coef*xx[g]*yy[g]*(x2 + y2 - 2.*z2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_yzz
        {
          Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*yy[g]*(-x2 - y2 + 4.*z2) / r3;
              Ylm(1) = coef*xx[g]*yy[g]*(x2 + y2 - 14.*z2) / r5;
              Ylm(2) = -coef*(x2*x2 + 11.*y2*z2- 4.*z2*z2 + x2*(y2-3.*z2)) / r5;
              Ylm(3) = coef*yy[g]*zz[g]*(11.*x2 + 11.*y2 - 4.*z2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_xzz
        {
          Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Real x2 = xx[g]*xx[g];
              Real y2 = yy[g]*yy[g];
              Real z2 = zz[g]*zz[g];
              Real r3 = pow(rad[g], 3.0);
              Real r5 = pow(rad[g], 5.0);

              Ylm(0) = coef*xx[g]*(-x2 - y2 + 4.*z2) / r3;
              Ylm(1) = -coef*(y2*y2 - 3.*y2*z2 - 4.*z2*z2 + x2*(y2+11.*z2)) / r5;
              Ylm(2) = coef*xx[g]*yy[g]*(x2 + y2 - 14.*z2) / r5;
              Ylm(3) = coef*xx[g]*zz[g]*(11.*x2 + 11.*y2 - 4.*z2) / r5;

              dv(g,VAL) = Ylm(0) * val[g] ;
              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
              dv(g,DX) = 0;
              dv(g,DY) = 0;
              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
      } // if(typ==pttype_.L3)
    } // for (g)

    // Check the number of pseudopotentials
    if( cntpp != numpp ){
      ErrorHandling("cntpp != numpp.  Seriously wrong with nonlocal pseudopotentials.");
    }
  } // if (norm(minDist) <= Rzero )
#endif // #ifndef _NO_NONLOCAL_

  return ;
}         // -----  end of method PeriodTable::CalculateNonlocalPP  ----- 

void
PeriodTable::CalculateNonlocalPP    ( 
    const Atom&    atom, 
    const Domain&  dm, 
    const NumTns<std::vector<DblNumVec> >&   gridposElem,
    std::vector<std::pair<NumTns<SparseVec>, Real> >& vnlList )
{
  Point3 Ls       = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns       = dm.numGridFine;

#ifndef _NO_NONLOCAL_ // Nonlocal potential is used. Debug option
  Int type   = atom.type;
  Point3 pos = atom.pos;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];

  // First count all the pseudopotentials
  Int numpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }
#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "Atom type " << type << " has " << numpp << 
    " nonlocal projectors." << std::endl;
#endif

  vnlList.clear();
  vnlList.resize( numpp );

  Index3 numElem( gridposElem.m(), gridposElem.n(), gridposElem.p() );

  // Initialize
  for( Int p = 0; p < numpp; p++ ){
    vnlList[p].first.Resize( numElem[0], numElem[1], numElem[2] );
  } // for (p)

  // Evalaute the nonlocal pseudopotential on each element
  for( Int elemk = 0; elemk < numElem[2]; elemk++ )
    for( Int elemj = 0; elemj < numElem[1]; elemj++ )
      for( Int elemi = 0; elemi < numElem[0]; elemi++ ){
        std::vector<NonlocalPP>  vnlpp;
        CalculateNonlocalPP( atom, dm,
            gridposElem(elemi, elemj, elemk),
            vnlpp );
        if( vnlList.size() != vnlpp.size() ){
          std::ostringstream msg;
          msg << "The number of pseudopotentials do not match." << std::endl;
          ErrorHandling( msg.str().c_str() );
        }
        for( Int p = 0; p < numpp; p++ ){
          NumTns<SparseVec>& res = vnlList[p].first;
          res( elemi, elemj, elemk ) = vnlpp[p].first;
          vnlList[p].second = vnlpp[p].second;
        } // for (p)
      } // for (elemi)

#endif // #ifndef _NO_NONLOCAL_

  return ;
}         // -----  end of method PeriodTable::CalculateNonlocalPP  ----- 



void PeriodTable::CalculateAtomDensity( 
    const Atom& atom, 
    const Domain& dm, 
    const std::vector<DblNumVec>& gridpos, 
    DblNumVec& atomDensity )
{
  Int type   = atom.type;
  Point3 pos = atom.pos;
  Point3 Ls  = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns  = dm.numGridFine;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutRhoAtom( type );

  SetValue(atomDensity, 0.0);

  // Compute the minimal distance of the atom to this set of grid points
  // and determine whether to continue 

  std::vector<DblNumVec>  dist(DIM);
  for( Int d = 0; d < DIM; d++ ){
    dist[d].Resize( gridpos[d].m() );

    for( Int i = 0; i < gridpos[d].m(); i++ ){
      dist[d](i) = gridpos[d](i) - pos[d];
      dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
    }
  }

  {
    Int irad = 0;
    std::vector<Int>  idx;
    std::vector<Real> rad;
    for(Int k = 0; k < gridpos[2].m(); k++)
      for(Int j = 0; j < gridpos[1].m(); j++)
        for(Int i = 0; i < gridpos[0].m(); i++){
          Real dtmp = std::sqrt( 
              dist[0](i) * dist[0](i) +
              dist[1](j) * dist[1](j) +
              dist[2](k) * dist[2](k) );

          if( dtmp <= Rzero ) {
            idx.push_back(irad);
            rad.push_back(dtmp);
          }
          irad++;
        } // for (i)

    Int idxsize = idx.size();
    if( idxsize > 0 ){
      //
      std::vector<DblNumVec>& valspl = spldata[ptsample_.RHOATOM]; 
      std::vector<Real> val(idxsize,0.0);
      seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
          valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());

      for(Int g=0; g<idx.size(); g++) {
        atomDensity[idx[g]] = val[g];
      }
    }  // if( idxsize > 0 )
  }

  return ;
}         // -----  end of method PeriodTable::CalculateAtomDensity  ----- 


void
PeriodTable::CalculateVLocal(
    const Atom& atom, 
    const Domain& dm,
    const std::vector<DblNumVec>& gridpos,        
    SparseVec& resVLocalSR, 
    SparseVec& resGaussianPseudoCharge )
{
  Int type   = atom.type;
  Point3 pos = atom.pos;
  Point3 Ls  = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns  = dm.numGridFine;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  // Use the pseudocharge cutoff for Gaussian compensation charge and
  // short range potential
  Real Rzero = this->RcutPseudoCharge( type );
  Real RGaussian = this->RGaussian( type );
  Real Zion = this->Zion( type );

  // Initialize
  {
    SparseVec empty;
    resVLocalSR = empty;
    resGaussianPseudoCharge = empty;
  }

  // Compute the minimal distance of the atom to this set of grid points
  // and determine whether to continue 

  std::vector<DblNumVec>  dist(DIM);

  Point3 minDist;
  for( Int d = 0; d < DIM; d++ ){
    dist[d].Resize( gridpos[d].m() );

    minDist[d] = Rzero;
    for( Int i = 0; i < gridpos[d].m(); i++ ){
      dist[d](i) = gridpos[d](i) - pos[d];
      dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
      if( std::abs( dist[d](i) ) < minDist[d] )
        minDist[d] = std::abs( dist[d](i) );
    }
  }
  if( std::sqrt( dot(minDist, minDist) ) <= Rzero ){
    // At least one grid point is within Rzero
    Int irad = 0;
    std::vector<Int>  idx;
    std::vector<Real> rad;
    std::vector<Real> xx, yy, zz;
    for(Int k = 0; k < gridpos[2].m(); k++)
      for(Int j = 0; j < gridpos[1].m(); j++)
        for(Int i = 0; i < gridpos[0].m(); i++){
          Real dtmp = std::sqrt( 
              dist[0](i) * dist[0](i) +
              dist[1](j) * dist[1](j) +
              dist[2](k) * dist[2](k) );

          if( dtmp <= Rzero ) {
            idx.push_back(irad);
            rad.push_back(dtmp);
            xx.push_back(dist[0](i));        
            yy.push_back(dist[1](j));        
            zz.push_back(dist[2](k));
          }
          irad++;
        } // for (i)



    Int idxsize = idx.size();
    // Short range pseudopotential

    std::vector<DblNumVec>& valspl = spldata[ptsample_.VLOCAL]; 
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
        valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());

    std::vector<DblNumVec>& derspl = spldata[ptsample_.DRV_VLOCAL];
    std::vector<Real> der(idxsize,0.0);
    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
        derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());

    IntNumVec iv(idx.size(), true, &(idx[0])); 
    DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
    for(Int g=0; g<idx.size(); g++) {
      dv(g, VAL) = val[g];
      // FIXME derivatives later
//      if( rad[g]> MIN_RADIAL ) {
//        dv(g, DX) = der[g] * xx[g]/rad[g];
//        dv(g, DY) = der[g] * yy[g]/rad[g];
//        dv(g, DZ) = der[g] * zz[g]/rad[g];
//      } else {
//        dv(g, DX) = 0;
//        dv(g, DY) = 0;
//        dv(g, DZ) = 0;
//      }
    }
    resVLocalSR = SparseVec(iv,dv);

    // Gaussian pseudocharge
    SetValue(dv, D_ZERO);
    Real fac = Zion / std::pow(std::sqrt(PI) * RGaussian,3);
    for(Int g=0; g<idx.size(); g++) {
      dv(g, VAL) = fac * std::exp(-(rad[g]/RGaussian)*(rad[g]/RGaussian)) ;
      // FIXME derivatives later
//      if( rad[g]> MIN_RADIAL ) {
//        dv(g, DX) = der[g] * xx[g]/rad[g];
//        dv(g, DY) = der[g] * yy[g]/rad[g];
//        dv(g, DZ) = der[g] * zz[g]/rad[g];
//      } else {
//        dv(g, DX) = 0;
//        dv(g, DY) = 0;
//        dv(g, DZ) = 0;
//      }
    }
    resGaussianPseudoCharge = SparseVec(iv,dv);
  } // if (norm(minDist) <= Rzero )

  return ;
}         // -----  end of method PeriodTable::CalculateVLocal  ----- 


Real PeriodTable::SelfIonInteraction(Int type) 
{
  Real eself;
  Real Rzero = this->RcutPseudoCharge( type );
  Real RGaussian = this->RGaussian( type );
  Real Zion = this->Zion( type );
  eself = Zion * Zion / ( std::sqrt(2.0 * PI) * RGaussian );
  
  return eself;
}         // -----  end of method PeriodTable::CalculateVLocal  ----- 




//---------------------------------------------
// TODO SpinOrbit from RelDFT

// Serialization / Deserialization
Int serialize(const Atom& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.type, os, mask);
  serialize(val.pos,  os, mask);
  serialize(val.vel,  os, mask);
  serialize(val.force,  os, mask);
  return 0;
}

Int deserialize(Atom& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.type, is, mask);
  deserialize(val.pos,  is, mask);
  deserialize(val.vel,  is, mask);
  deserialize(val.force,  is, mask);
  return 0;
}

Real MaxForce( const std::vector<Atom>& atomList ){
  Int numAtom = atomList.size();
  Real maxForce = 0.0;
  for( Int i = 0; i < numAtom; i++ ){
    Real forceMag = atomList[i].force.l2();
    maxForce = ( maxForce < forceMag ) ? forceMag : maxForce;
  }
  return maxForce;
}



// *********************************************************************
// The following comes from the UPF2QSO subroutine.
// *********************************************************************
struct Element
{
  int z;
  std::string symbol;
  std::string config;
  double mass;
  Element(int zz, std::string s, std::string c, double m) : z(zz), symbol(s), config(c),
    mass(m) {}
};

class PeriodicTable
{
  private:

  std::vector<Element> ptable;
  std::map<std::string,int> zmap;

  public:

  PeriodicTable(void);
  int z(std::string symbol) const;
  std::string symbol(int zval) const;
  std::string configuration(int zval) const;
  std::string configuration(std::string symbol) const;
  double mass(int zval) const;
  double mass(std::string symbol) const;
  int size(void) const;

};


/// the following code are merged from the UPF2QSO package
int PeriodicTable::z(std::string symbol) const
{
  std::map<std::string,int>::const_iterator i = zmap.find(symbol);
  assert( i != zmap.end() );
  return (*i).second;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::symbol(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].symbol;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::configuration(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].config;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::configuration(std::string symbol) const
{
  return ptable[z(symbol)-1].config;
}

////////////////////////////////////////////////////////////////////////////////
double PeriodicTable::mass(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].mass;
}

////////////////////////////////////////////////////////////////////////////////
double PeriodicTable::mass(std::string symbol) const
{
  return ptable[z(symbol)-1].mass;
}

////////////////////////////////////////////////////////////////////////////////
int PeriodicTable::size(void) const
{
  return ptable.size();
}

////////////////////////////////////////////////////////////////////////////////
PeriodicTable::PeriodicTable(void)
{
  ptable.push_back(Element(1,"H","1s1",1.00794));
  ptable.push_back(Element(2,"He","1s2",4.00260));
  ptable.push_back(Element(3, "Li","1s2 2s1",     6.941));
  ptable.push_back(Element(4, "Be","1s2 2s2",     9.01218));
  ptable.push_back(Element(5, "B", "1s2 2s2 2p1",10.811));
  ptable.push_back(Element(6, "C", "1s2 2s2 2p2",12.0107));
  ptable.push_back(Element(7, "N", "1s2 2s2 2p3",14.00674));
  ptable.push_back(Element(8, "O", "1s2 2s2 2p4",15.9994));
  ptable.push_back(Element(9, "F", "1s2 2s2 2p5",18.9884));
  ptable.push_back(Element(10,"Ne","1s2 2s2 2p6",20.1797));

  ptable.push_back(Element(11,"Na","[Ne] 3s1",    22.98977));
  ptable.push_back(Element(12,"Mg","[Ne] 3s2",    24.3050));
  ptable.push_back(Element(13,"Al","[Ne] 3s2 3p1",26.98154));
  ptable.push_back(Element(14,"Si","[Ne] 3s2 3p2",28.0855));
  ptable.push_back(Element(15,"P", "[Ne] 3s2 3p3",30.97376));
  ptable.push_back(Element(16,"S", "[Ne] 3s2 3p4",32.066));
  ptable.push_back(Element(17,"Cl","[Ne] 3s2 3p5",35.4527));
  ptable.push_back(Element(18,"Ar","[Ne] 3s2 3p6",39.948));

  ptable.push_back(Element(19,"K", "[Ar] 4s1",39.0983));
  ptable.push_back(Element(20,"Ca","[Ar] 4s2",40.078));
  ptable.push_back(Element(21,"Sc","[Ar] 3d1 4s2",44.95591));
  ptable.push_back(Element(22,"Ti","[Ar] 3d2 4s2",47.867));
  ptable.push_back(Element(23,"V", "[Ar] 3d3 4s2",50.9415));
  ptable.push_back(Element(24,"Cr","[Ar] 3d5 4s1",51.9961));
  ptable.push_back(Element(25,"Mn","[Ar] 3d5 4s2",54.93805));
  ptable.push_back(Element(26,"Fe","[Ar] 3d6 4s2",55.845));
  ptable.push_back(Element(27,"Co","[Ar] 3d7 4s2",58.9332));
  ptable.push_back(Element(28,"Ni","[Ar] 3d8 4s2",58.6934));
  ptable.push_back(Element(29,"Cu","[Ar] 3d10 4s1",63.546));
  ptable.push_back(Element(30,"Zn","[Ar] 3d10 4s2",65.39));
  ptable.push_back(Element(31,"Ga","[Ar] 3d10 4s2 4p1",69.723));
  ptable.push_back(Element(32,"Ge","[Ar] 3d10 4s2 4p2",72.61));
  ptable.push_back(Element(33,"As","[Ar] 3d10 4s2 4p3",74.9216));
  ptable.push_back(Element(34,"Se","[Ar] 3d10 4s2 4p4",78.96));
  ptable.push_back(Element(35,"Br","[Ar] 3d10 4s2 4p5",79.904));
  ptable.push_back(Element(36,"Kr","[Ar] 3d10 4s2 4p6",83.80));

  ptable.push_back(Element(37,"Rb","[Kr] 5s1",85.4678));
  ptable.push_back(Element(38,"Sr","[Kr] 5s2",87.62));
  ptable.push_back(Element(39,"Y" ,"[Kr] 4d1 5s2",88.90585));
  ptable.push_back(Element(40,"Zr","[Kr] 4d2 5s2",91.224));
  ptable.push_back(Element(41,"Nb","[Kr] 4d4 5s1",92.90638));
  ptable.push_back(Element(42,"Mo","[Kr] 4d5 5s1",95.94));
  ptable.push_back(Element(43,"Tc","[Kr] 4d5 5s2",98.0));
  ptable.push_back(Element(44,"Ru","[Kr] 4d7 5s1",101.07));
  ptable.push_back(Element(45,"Rh","[Kr] 4d8 5s1",102.9055));
  ptable.push_back(Element(46,"Pd","[Kr] 4d10",106.42));
  ptable.push_back(Element(47,"Ag","[Kr] 4d10 5s1",107.8682));
  ptable.push_back(Element(48,"Cd","[Kr] 4d10 5s2",112.411));
  ptable.push_back(Element(49,"In","[Kr] 4d10 5s2 5p1",114.818));
  ptable.push_back(Element(50,"Sn","[Kr] 4d10 5s2 5p2",118.710));
  ptable.push_back(Element(51,"Sb","[Kr] 4d10 5s2 5p3",121.760));
  ptable.push_back(Element(52,"Te","[Kr] 4d10 5s2 5p4",127.60));
  ptable.push_back(Element(53,"I" ,"[Kr] 4d10 5s2 5p5",126.90447));
  ptable.push_back(Element(54,"Xe","[Kr] 4d10 5s2 5p6",131.29));

  ptable.push_back(Element(55,"Cs","[Xe] 6s1",132.90545));
  ptable.push_back(Element(56,"Ba","[Xe] 6s2",137.327));
  ptable.push_back(Element(57,"La","[Xe] 5d1 6s2",138.9055));
  ptable.push_back(Element(58,"Ce","[Xe] 4f1 5d1 6s2",140.116));
  ptable.push_back(Element(59,"Pr","[Xe] 4f3 6s2",140.90765));
  ptable.push_back(Element(60,"Nd","[Xe] 4f4 6s2",144.24));
  ptable.push_back(Element(61,"Pm","[Xe] 4f5 6s2",145.0));
  ptable.push_back(Element(62,"Sm","[Xe] 4f6 6s2",150.36));
  ptable.push_back(Element(63,"Eu","[Xe] 4f7 6s2",151.964));
  ptable.push_back(Element(64,"Gd","[Xe] 4f7 5d1 6s2",157.25));
  ptable.push_back(Element(65,"Tb","[Xe] 4f9 6s2",158.92534));
  ptable.push_back(Element(66,"Dy","[Xe] 4f10 6s2",162.50));
  ptable.push_back(Element(67,"Ho","[Xe] 4f11 6s2",164.93032));
  ptable.push_back(Element(68,"Er","[Xe] 4f12 6s2",167.26));
  ptable.push_back(Element(69,"Tm","[Xe] 4f13 6s2",168.93421));
  ptable.push_back(Element(70,"Yb","[Xe] 4f14 6s2",173.04));
  ptable.push_back(Element(71,"Lu","[Xe] 4f14 5d1 6s2",174.967));
  ptable.push_back(Element(72,"Hf","[Xe] 4f14 5d2 6s2",178.49));
  ptable.push_back(Element(73,"Ta","[Xe] 4f14 5d3 6s2",180.9479));
  ptable.push_back(Element(74,"W" ,"[Xe] 4f14 5d4 6s2",183.84));
  ptable.push_back(Element(75,"Re","[Xe] 4f14 5d5 6s2",186.207));
  ptable.push_back(Element(76,"Os","[Xe] 4f14 5d6 6s2",190.23));
  ptable.push_back(Element(77,"Ir","[Xe] 4f14 5d7 6s2",192.217));
  ptable.push_back(Element(78,"Pt","[Xe] 4f14 5d9 6s1",195.078));
  ptable.push_back(Element(79,"Au","[Xe] 4f14 5d10 6s1",196.96655));
  ptable.push_back(Element(80,"Hg","[Xe] 4f14 5d10 6s2",200.59));
  ptable.push_back(Element(81,"Tl","[Xe] 4f14 5d10 6s2 6p1",204.3833));
  ptable.push_back(Element(82,"Pb","[Xe] 4f14 5d10 6s2 6p2",207.2));
  ptable.push_back(Element(83,"Bi","[Xe] 4f14 5d10 6s2 6p3",208.98038));
  ptable.push_back(Element(84,"Po","[Xe] 4f14 5d10 6s2 6p4",209.0));
  ptable.push_back(Element(85,"At","[Xe] 4f14 5d10 6s2 6p5",210.0));
  ptable.push_back(Element(86,"Rn","[Xe] 4f14 5d10 6s2 6p6",222.0));

  ptable.push_back(Element(87,"Fr","[Rn] 7s1",223.0));
  ptable.push_back(Element(88,"Ra","[Rn] 7s2",226.0));
  ptable.push_back(Element(89,"Ac","[Rn] 6d1 7s2",227.0));
  ptable.push_back(Element(90,"Th","[Rn] 6d2 7s2",232.0381));
  ptable.push_back(Element(91,"Pa","[Rn] 5f2 6d1 7s2",231.03588));
  ptable.push_back(Element(92,"U" ,"[Rn] 5f3 6d1 7s2",238.0289));
  ptable.push_back(Element(93,"Np","[Rn] 5f4 6d1 7s2",237.0));
  ptable.push_back(Element(94,"Pu","[Rn] 5f5 6d1 7s2",244.0));

  for ( int i = 0; i < ptable.size(); i++ )
    zmap[ptable[i].symbol] = i+1;
}

// change the main subroutine to a readin function.
Int ReadUPF( std::string file_name, PTEntry * tempEntry, Int * atom)
{

  DblNumVec & params  = (*tempEntry).params;
  DblNumMat & samples = (*tempEntry).samples;
  DblNumVec & weights = (*tempEntry).weights;
  IntNumVec & types   = (*tempEntry).types;
  DblNumVec & cutoffs = (*tempEntry).cutoffs;

  params.Resize(5); // in the order of the ParamPT
  

  PeriodicTable pt;

  std::string buf,s;
  std::istringstream is;

  // Determine if UPF file is specified as a KW
  auto kw_pos = file_name.find("KW:");
  if( kw_pos != std::string::npos ) {
  #ifdef DG_PP_ONCV_PATH
    // Correct file name
    file_name.erase(0,kw_pos);
    file_name.erase(0,3);
    file_name = std::string(DG_PP_ONCV_PATH "/") + file_name + std::string(".upf");

    statusOFS << " * Reading from standard UPF file located at "
              << file_name << std::endl;
  #else
    ErrorHandling(" KW for UPF Files Requires -DDG_PP_ONCV_PATH=...");
  #endif
  }


  // determine UPF version
  int upf_version = 0;

  // The first line of the UPF potential file contains either of the following:
  // <PP_INFO>  (for UPF version 1)
  // <UPF version="2.0.1"> (for UPF version 2)

  std::string::size_type p;
  std::ifstream upfin( file_name );
  if( upfin.good() ) {
    getline(upfin,buf);
    p = buf.find("<PP_INFO>");
    if ( p != std::string::npos )
      upf_version = 1;
    else
    {
      p = buf.find("<UPF version=\"2.0.1\">");
      if ( p != std::string::npos )
        upf_version = 2;
    }
    if ( upf_version == 0 )
    {
      statusOFS << " Format of UPF file not recognized " << std::endl;
      statusOFS << " First line of file: " << buf << std::endl;
      ErrorHandling( " Format of UPF file not recognized " );
      return 1;
    }
  } else {
    std::ostringstream msg;
    msg << "UPF File " << file_name << " not found";
    ErrorHandling( msg.str() );
  }

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << " UPF version: " << upf_version << std::endl;
#endif

  if ( upf_version == 1 )
  {
    ErrorHandling( " Format of UPF file 1.0 not supported" );
#if 0
    // process UPF version 1 potential
    std::string upf_pp_info;
    bool done = false;
    while (!done)
    {
      getline(upfin,buf);
      is.clear();
      is.str(buf);
      is >> s;
      done = ( s == "</PP_INFO>" );
      if ( !done )
      {
        upf_pp_info += buf + '\n';
      }
    }

    // remove all '<' and '>' characters from the PP_INFO field
    // for XML compatibility
    p = upf_pp_info.find_first_of("<>");
    while ( p != std::string::npos )
    {
      upf_pp_info[p] = ' ';
      p = upf_pp_info.find_first_of("<>");
    }

    seek_str("<PP_HEADER>", upfin);
    skipln(upfin);

    // version number (ignore)
    getline(upfin,buf);

    // element symbol
    std::string upf_symbol;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_symbol;

    // get atomic number and mass
    const int atomic_number = pt.z(upf_symbol);
    const double mass = pt.mass(upf_symbol);
    *atom = atomic_number;

    // NC flag
    std::string upf_ncflag;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_ncflag;
    if ( upf_ncflag != "NC" )
    {
      statusOFS << " not a Norm-conserving potential" << std::endl;
      statusOFS << " NC flag: " << upf_ncflag << std::endl;
      return 1;
    }

    // NLCC flag
    std::string upf_nlcc_flag;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_nlcc_flag;
    if ( upf_nlcc_flag == "T" )
    {
      statusOFS << " Potential includes a non-linear core correction" << std::endl;
    }

    // XC functional (add in description)
    std::string upf_xcf[4];
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_xcf[0] >> upf_xcf[1] >> upf_xcf[2] >> upf_xcf[3];

    // add XC functional information to description
    upf_pp_info += upf_xcf[0] + ' ' + upf_xcf[1] + ' ' +
                   upf_xcf[2] + ' ' + upf_xcf[3] + '\n';

    // Z valence
    double upf_zval;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_zval;

    // Total energy (ignore)
    getline(upfin,buf);

    // suggested cutoff (ignore)
    getline(upfin,buf);

    // max angular momentum
    int upf_lmax;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_lmax;

    // number of points in mesh
    int upf_mesh_size;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_mesh_size;

    // number of wavefunctions, number of projectors
    int upf_nwf, upf_nproj;
    getline(upfin,buf);
    is.clear();
    is.str(buf);
    is >> upf_nwf >> upf_nproj;

    // Wavefunctions
    std::vector<std::string> upf_shell(upf_nwf);
    std::vector<int> upf_l(upf_nwf);
    std::vector<double> upf_occ(upf_nwf);
    // skip header
    getline(upfin,buf);
    for ( int ip = 0; ip < upf_nwf; ip++ )
    {
      getline(upfin,buf);
      is.clear();
      is.str(buf);
      is >> upf_shell[ip] >> upf_l[ip] >> upf_occ[ip];
    }
    seek_str("</PP_HEADER>", upfin);

    // read mesh
    seek_str("<PP_MESH>", upfin);
    seek_str("<PP_R>", upfin);
    skipln(upfin);
    std::vector<double> upf_r(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
     upfin >> upf_r[i];
    seek_str("</PP_R>", upfin);
    seek_str("<PP_RAB>", upfin);
    skipln(upfin);
    std::vector<double> upf_rab(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
     upfin >> upf_rab[i];
    seek_str("</PP_RAB>", upfin);
    seek_str("</PP_MESH>", upfin);

    std::vector<double> upf_nlcc;
    if ( upf_nlcc_flag == "T" )
    {
      upf_nlcc.resize(upf_mesh_size);
      seek_str("<PP_NLCC>", upfin);
      skipln(upfin);
      std::vector<double> upf_nlcc(upf_mesh_size);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_nlcc[i];
      seek_str("</PP_NLCC>", upfin);
    }

    seek_str("<PP_LOCAL>", upfin);
    skipln(upfin);
    std::vector<double> upf_vloc(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_vloc[i];
    seek_str("</PP_LOCAL>", upfin);

    seek_str("<PP_NONLOCAL>", upfin);
    skipln(upfin);
    std::vector<std::vector<double> > upf_vnl;
    upf_vnl.resize(upf_nproj);
    std::vector<int> upf_proj_l(upf_nproj);
    for ( int j = 0; j < upf_nproj; j++ )
    {
      seek_str("<PP_BETA>", upfin);
      skipln(upfin);
      int ip, l, np;
      upfin >> ip >> l;
      skipln(upfin);
      assert(ip-1 < upf_nproj);
      assert(l <= upf_lmax);
      upf_proj_l[ip-1] = l;
      upfin >> np;
      upf_vnl[j].resize(upf_mesh_size);
      for ( int i = 0; i < np; i++ )
        upfin >> upf_vnl[j][i];
      seek_str("</PP_BETA>", upfin);
      skipln(upfin);
    }
    seek_str("<PP_DIJ>", upfin);
    skipln(upfin);
    int upf_ndij;
    upfin >> upf_ndij;
    skipln(upfin);
    if ( upf_ndij != upf_nproj )
    {
      statusOFS << " Number of non-zero Dij differs from number of projectors"
           << std::endl;
      return 1;
    }

    std::vector<double> upf_d(upf_ndij);
    for ( int i = 0; i < upf_ndij; i++ )
    {
      int m,n;
      upfin >> m >> n >> upf_d[i];
      if ( m != n )
      {
        statusOFS << " Non-local Dij has off-diagonal elements" << std::endl;
        statusOFS << " m=" << m << " n=" << n << std::endl;
        return 1;
      }
    }
    seek_str("</PP_DIJ>", upfin);

    seek_str("</PP_NONLOCAL>", upfin);

    // make table iproj[l] mapping l to iproj
    // vnl(l) is in vnl[iproj[l]] if iproj[l] > -1
    // vlocal if iproj[llocal] = -1
    std::vector<int> iproj(upf_lmax+2);
    for ( int l = 0; l <= upf_lmax+1; l++ )
      iproj[l] = -1;
    for ( int j = 0; j < upf_nproj; j++ )
      iproj[upf_proj_l[j]] = j;

    // determine angular momentum of local potential in UPF file
    int upf_llocal;
    // reverse loop to get correct upf_llocal when upf_nproj < upf_lmax
    for ( int l = upf_lmax+1; l >= 0; l-- )
      if ( iproj[l] == -1 )
        upf_llocal = l;
    // increase lmax if there are more projectors than wavefunctions
    int qso_lmax = upf_lmax;
    if (upf_lmax < upf_llocal)
    {
      qso_lmax = upf_lmax+1;
    }

    seek_str("<PP_PSWFC>", upfin);
    skipln(upfin);
    std::vector<std::vector<double> > upf_wf;
    std::vector<int> upf_wf_l(upf_nwf);
    std::vector<double> upf_wf_occ(upf_nwf);
    upf_wf.resize(upf_nwf);
    for ( int j = 0; j < upf_nwf; j++ )
    {
      upf_wf[j].resize(upf_mesh_size);
      std::string label;
      upfin >> label >> upf_wf_l[j] >> upf_wf_occ[j];
      skipln(upfin);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_wf[j][i];
    }
    seek_str("</PP_PSWFC>", upfin);

    // output original data in file upf.dat
    std::ofstream upf("upf.dat");
    upf << "# vloc" << std::endl;
    for ( int i = 0; i < upf_vloc.size(); i++ )
      upf << upf_r[i] << " " << upf_vloc[i] << std::endl;
    upf << std::endl << std::endl;
    for ( int j = 0; j < upf_nproj; j++ )
    {
      upf << "# proj j=" << j << std::endl;
      for ( int i = 0; i < upf_vnl[j].size(); i++ )
        upf << upf_r[i] << " " << upf_vnl[j][i] << std::endl;
      upf << std::endl << std::endl;
    }
    for ( int j = 0; j < upf_nwf; j++ )
    {
      upf << "# wf j=" << j << std::endl;
      for ( int i = 0; i < upf_wf[j].size(); i++ )
        upf << upf_r[i] << " " << upf_wf[j][i] << std::endl;
      upf << std::endl << std::endl;
    }
    upf.close();


    // print summary
    statusOFS << "PP_INFO:" << std::endl << upf_pp_info << std::endl;
    statusOFS << "Element: " << upf_symbol << std::endl;
    statusOFS << "NC: " << upf_ncflag << std::endl;
    statusOFS << "NLCC: " << upf_nlcc_flag << std::endl;
    statusOFS << "XC: " << upf_xcf[0] << " " << upf_xcf[1] << " "
         << upf_xcf[2] << " " << upf_xcf[3] << std::endl;
    statusOFS << "Zv: " << upf_zval << std::endl;
    statusOFS << "lmax: " << qso_lmax << std::endl;
    statusOFS << "llocal: " << upf_llocal << std::endl;
    statusOFS << "nwf: " << upf_nwf << std::endl;
    statusOFS << "mesh_size: " << upf_mesh_size << std::endl;

    // compute delta_vnl[l][i] on the upf log mesh

    // divide the projector function by the wavefunction, except if
    // the wavefunction amplitude is smaller than tol, outside of rcut_divide.
    const double tol = 1.e-5;
    const double rcut_divide = 1.0;
    std::vector<std::vector<double> > delta_vnl;
    delta_vnl.resize(upf_nproj);
    for ( int j = 0; j < upf_nproj; j++ )
    {
      delta_vnl[j].resize(upf_wf[j].size());
      for ( int i = 0; i < delta_vnl[j].size(); i++ )
      {
        double num = upf_vnl[j][i];
        double den = upf_wf[upf_proj_l[j]][i];

        delta_vnl[j][i] = 0.0;
        if ( upf_r[i] < rcut_divide )
        {
          // near the origin
          if ( i == 0 && fabs(den) < tol )
          {
            // i = 0 for linear mesh, r = 0.0: use nearest value
            delta_vnl[j][i] = upf_vnl[j][1] / upf_wf[upf_proj_l[j]][1];
          }
          else
          {
            // other points near the origin
            delta_vnl[j][i] = num / den;
          }
        }
        else
        {
          // wavefunction gets small at large r.
          // Assume that delta_vnl is zero when that happens
          if ( fabs(den) > tol )
            delta_vnl[j][i] = num / den;
        }
      }
    }

    std::vector<std::vector<double> > vps;
    vps.resize(upf_nproj+1);
    for ( int j = 0; j < upf_nproj; j++ )
    {
      vps[j].resize(upf_mesh_size);
      for ( int i = 0; i < delta_vnl[j].size(); i++ )
        vps[j][i] = upf_vloc[i] + delta_vnl[j][i];
    }

    // interpolate functions on linear mesh
    const double mesh_spacing = 0.01;
    const double rcut = 6.0;  // FIXME seems to be only useful for nonlinear core correction
    int nplin = (int) (rcut / mesh_spacing);
    std::vector<double> f(upf_mesh_size), fspl(upf_mesh_size);

    std::vector<double> nlcc_lin(nplin);
    // interpolate NLCC
    if ( upf_nlcc_flag == "T" )
    {
      assert(upf_mesh_size==upf_nlcc.size());
      for ( int i = 0; i < upf_nlcc.size(); i++ )
      f[i] = upf_nlcc[i];
      int n = upf_nlcc.size();
      int bcnat_left = 0;
      double yp_left = 0.0;
      int bcnat_right = 1;
      double yp_right = 0.0;
      spline(n,&upf_r[0],&f[0],yp_left,yp_right,
             bcnat_left,bcnat_right,&fspl[0]);

      for ( int i = 0; i < nplin; i++ )
      {
        double r = i * mesh_spacing;
        if ( r >= upf_r[0] )
          splint(n,&upf_r[0],&f[0],&fspl[0],r,&nlcc_lin[i]);
        else
          // use value closest to the origin for r=0
          nlcc_lin[i] = upf_nlcc[0];
      }
    }

    // interpolate vloc
    // factor 0.5: convert from Ry in UPF to Hartree in QSO
    for ( int i = 0; i < upf_vloc.size(); i++ )
      f[i] = 0.5 * upf_vloc[i];

    int n = upf_vloc.size();
    int bcnat_left = 0;
    double yp_left = 0.0;
    int bcnat_right = 1;
    double yp_right = 0.0;
    spline(n,&upf_r[0],&f[0],yp_left,yp_right,
           bcnat_left,bcnat_right,&fspl[0]);

    std::vector<double> vloc_lin(nplin);
    for ( int i = 0; i < nplin; i++ )
    {
      double r = i * mesh_spacing;
      if ( r >= upf_r[0] )
        splint(n,&upf_r[0],&f[0],&fspl[0],r,&vloc_lin[i]);
      else
        // use value closest to the origin for r=0
        vloc_lin[i] = 0.5 * upf_vloc[0];
    }

    // interpolate vps[j], j=0, nproj-1
    std::vector<std::vector<double> > vps_lin;
    vps_lin.resize(vps.size());
    for ( int j = 0; j < vps.size(); j++ )
    {
      vps_lin[j].resize(nplin);
    }

    for ( int j = 0; j < upf_nproj; j++ )
    {
      // factor 0.5: convert from Ry in UPF to Hartree in QSO
      for ( int i = 0; i < upf_vloc.size(); i++ )
        f[i] = 0.5 * vps[j][i];

      int n = upf_vloc.size();
      int bcnat_left = 0;
      double yp_left = 0.0;
      int bcnat_right = 1;
      double yp_right = 0.0;
      spline(n,&upf_r[0],&f[0],yp_left,yp_right,
             bcnat_left,bcnat_right,&fspl[0]);

      for ( int i = 0; i < nplin; i++ )
      {
        double r = i * mesh_spacing;
        if ( r >= upf_r[0] )
          splint(n,&upf_r[0],&f[0],&fspl[0],r,&vps_lin[j][i]);
        else
          vps_lin[j][i] = 0.5 * vps[j][0];
      }
    }

    // write potentials in gnuplot format on file vlin.dat
    std::ofstream vlin("vlin.dat");
    for ( int l = 0; l <= qso_lmax; l++ )
    {
      vlin << "# v, l=" << l << std::endl;
      if ( iproj[l] == -1 )
      {
        // l == llocal
        for ( int i = 0; i < nplin; i++ )
          vlin << i*mesh_spacing << " " << vloc_lin[i] << std::endl;
        vlin << std::endl << std::endl;
      }
      else
      {
        for ( int i = 0; i < nplin; i++ )
          vlin << i*mesh_spacing << " " << vps_lin[iproj[l]][i] << std::endl;
        vlin << std::endl << std::endl;
      }
    }

    // interpolate wavefunctions on the linear mesh

    std::vector<std::vector<double> > wf_lin;
    wf_lin.resize(upf_nwf);
    for ( int j = 0; j < upf_nwf; j++ )
    {
      wf_lin[j].resize(nplin);
      assert(upf_wf[j].size()<=f.size());
      for ( int i = 0; i < upf_wf[j].size(); i++ )
      {
        if ( upf_r[i] > 0.0 )
          f[i] = upf_wf[j][i] / upf_r[i];
        else
        {
          // value at origin, depending on angular momentum
          if ( upf_wf_l[j] == 0 )
          {
            // l=0: take value closest to r=0
            f[i] = upf_wf[j][1]/upf_r[1];
          }
          else
          {
            // l>0:
            f[i] = 0.0;
          }
        }
      }

      int n = upf_wf[j].size();
      // choose boundary condition at origin depending on angular momentum
      int bcnat_left = 0;
      double yp_left = 0.0;
      if ( upf_wf_l[j] == 1 )
      {
        bcnat_left = 1; // use natural bc
        double yp_left = 0.0; // not used
      }
      int bcnat_right = 1;
      double yp_right = 0.0;
      spline(n,&upf_r[0],&f[0],yp_left,yp_right,
             bcnat_left,bcnat_right,&fspl[0]);

      for ( int i = 0; i < nplin; i++ )
      {
        double r = i * mesh_spacing;
        if ( r >= upf_r[0] )
          splint(n,&upf_r[0],&f[0],&fspl[0],r,&wf_lin[j][i]);
        else
        {
          // r < upf_r[0]
          assert(upf_r[0]>0.0);
          // compute value near origin, depending on angular momentum
          if ( upf_wf_l[j] == 0 )
          {
            // l=0: take value closest to r=0
            wf_lin[j][i] = upf_wf[j][0]/upf_r[0];
          }
          else
          {
            // l>0:
            wf_lin[j][i] = upf_wf[j][0] * r / ( upf_r[0] * upf_r[0] );
          }
        }
      }

      vlin << "# phi, l=" << upf_l[j] << std::endl;
      for ( int i = 0; i < nplin; i++ )
        vlin << i*mesh_spacing << " " << wf_lin[j][i] << std::endl;
      vlin << std::endl << std::endl;
    }

    statusOFS << " interpolation done" << std::endl;

#if 0
    // output potential on log mesh
    std::ofstream vout("v.dat");
    for ( int l = 0; l <= qso_lmax; l++ )
    {
      vout << "# v, l=" << l << std::endl;
      if ( iproj[l] == -1 )
      {
        // l == llocal
        for ( int i = 0; i < upf_vloc.size(); i++ )
          vout << upf_r[i] << " " << 0.5*upf_vloc[i] << std::endl;
        vout << std::endl << std::endl;
      }
      else
      {
        for ( int i = 0; i < vps[iproj[l]].size(); i++ )
          vout << upf_r[i] << " " << 0.5*vps[iproj[l]][i] << std::endl;
        vout << std::endl << std::endl;
      }
    }
    vout << std::endl << std::endl;
    vout.close();
#endif
#endif

  }
  else if ( upf_version == 2 )
  { 
    // process UPF version 2 potential
    seek_str("<PP_INFO>", upfin);
    std::string upf_pp_info;
    bool done = false;
    while (!done)
    {
      getline(upfin,buf);
      is.clear();
      is.str(buf);
      is >> s;
      done = ( s == "</PP_INFO>" );
      if ( !done )
      {
        upf_pp_info += buf + '\n';
      }
    }

    // remove all '<' and '>' characters from the PP_INFO field
    // for XML compatibility
    p = upf_pp_info.find_first_of("<>");
    while ( p != std::string::npos )
    {
      upf_pp_info[p] = ' ';
      p = upf_pp_info.find_first_of("<>");
    }

    std::string tag = find_start_element("PP_HEADER", upfin);
#if 0
    std::cout << "  pp header " << tag << " upfin " <<  upfin << std::endl;
#endif

    // get attribute "element"
    std::string upf_symbol = get_attr(tag,"element");

#if 0
    std::cout << " get attribute " << upf_symbol << std::endl;
#endif

    upf_symbol.erase(remove_if(upf_symbol.begin(), upf_symbol.end(), isspace), upf_symbol.end());

    // get atomic number and mass
    const int atomic_number = pt.z(upf_symbol);
    const double mass = pt.mass(upf_symbol);

    *atom = atomic_number;
    params[0] = atomic_number;
    params[1] = mass;

    // check if potential is norm-conserving or semi-local
    std::string pseudo_type = get_attr(tag,"pseudo_type");
#if 0
    statusOFS << " pseudo_type = " << pseudo_type << std::endl;
    if ( pseudo_type!="NC" && pseudo_type!="SL" )
    {
      statusOFS << " pseudo_type must be NC or SL" << std::endl;
      return 1;
    }
#endif

    // NLCC flag
    std::string upf_nlcc_flag = get_attr(tag,"core_correction");
    if ( upf_nlcc_flag == "T" )
    {
      statusOFS << " Potential includes a non-linear core correction" << std::endl;
    }
#if 0
    statusOFS << " upf_nlcc_flag = " << upf_nlcc_flag << std::endl;
#endif

    // XC functional (add in description)
    std::string upf_functional = get_attr(tag,"functional");
    // add XC functional information to description
    upf_pp_info += "functional = " + upf_functional + '\n';
#if 0
    statusOFS << " upf_functional = " << upf_functional << std::endl;
#endif

    // valence charge
    double upf_zval = 0.0;
    std::string buf = get_attr(tag,"z_valence");
    is.clear();
    is.str(buf);
    is >> upf_zval;
#if 0
    statusOFS << " upf_zval = " << upf_zval << std::endl;
#endif

    // FIXME rhocut readin from the PSP file. 
    // for SG15, the default value is 6.01.
    // but may change for other user defined PSP files. 
    double rhocut = 6.01; 
    buf = get_attr(tag,"rho_cutoff");
    is.clear();
    is.str(buf);
    is >> rhocut;

#if 0
    statusOFS << " rhocut = " << rhocut << std::endl;
#endif

    // FIXME labels
    const Int ZION = 2;
    const Int RGAUSSIAN = 3;
    params[ZION] = upf_zval;
    // FIXME RGaussian should be given by a table according to the element type.
    params[RGAUSSIAN] = 1.0; 


    // max angular momentum
    int upf_lmax;
    buf = get_attr(tag,"l_max");
    is.clear();
    is.str(buf);
    is >> upf_lmax;
#if 0
    statusOFS << " upf_lmax = " << upf_lmax << std::endl;
#endif

    // local angular momentum
    int upf_llocal;
    buf = get_attr(tag,"l_local");
    is.clear();
    is.str(buf);
    is >> upf_llocal;
#if 0
    statusOFS << " upf_llocal = " << upf_llocal << std::endl;
#endif
    // number of points in mesh
    int upf_mesh_size;
    buf = get_attr(tag,"mesh_size");
    is.clear();
    is.str(buf);
    is >> upf_mesh_size;
#if 0
    statusOFS << " upf_mesh_size = " << upf_mesh_size << std::endl;
#endif
    // number of wavefunctions
    int upf_nwf;
    buf = get_attr(tag,"number_of_wfc");
    is.clear();
    is.str(buf);
    is >> upf_nwf;
#if 0
    statusOFS << " upf_nwf = " << upf_nwf << std::endl;
#endif

    // number of projectors
    int upf_nproj;
    buf = get_attr(tag,"number_of_proj");
    is.clear();
    is.str(buf);
    is >> upf_nproj;
#if 0
    statusOFS << " upf_nproj = " << upf_nproj << std::endl;
#endif

    samples.Resize( upf_mesh_size, 5 + 2*upf_nproj);
    weights.Resize(5+2*upf_nproj);
    cutoffs.Resize(5+2*upf_nproj);
    types.Resize(5+2*upf_nproj);
    SetValue( samples, 0.0);
    SetValue( weights, 0.0);
    SetValue( types, 0);
    SetValue( cutoffs, 0.0);

    std::vector<int> upf_l(upf_nwf);

    // read mesh
    find_start_element("PP_MESH", upfin);
    find_start_element("PP_R", upfin);
    std::vector<double> upf_r(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_r[i];
    find_end_element("PP_R", upfin);
    find_start_element("PP_RAB", upfin);
    std::vector<double> upf_rab(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_rab[i];
    find_end_element("PP_RAB", upfin);
    find_end_element("PP_MESH", upfin);

    // add the mesh into samples.
    for( int i = 0; i < upf_mesh_size; i++)
      samples(i, 0) = upf_r[i];
    weights[0] = -1;
    types[0] = 9;
    // FIXME labels
    const Int RADIAL_GRID = 0;
    const Int VLOCAL = 1;
    const Int DRV_VLOCAL = 2;
    const Int RHOATOM = 3;
    const Int DRV_RHOATOM = 4;
    // FIXME rhoatomcut should be given by a table according to the element type.
    double rhoatomcut = 4.0;
    // nonlocal potential cutoff read from the pseduopotential file below. 4.0 is just initial value.
    double nlcut = 4.0;

    cutoffs[RADIAL_GRID] = rhocut;
    cutoffs[VLOCAL] = rhocut;
    cutoffs[DRV_VLOCAL] = rhocut;

    cutoffs[RHOATOM] = rhoatomcut;
    cutoffs[DRV_RHOATOM] = rhoatomcut;

    // NLCC not used
    std::vector<double> upf_nlcc;
    if ( upf_nlcc_flag == "T" )
    {
      find_start_element("PP_NLCC", upfin);
      upf_nlcc.resize(upf_mesh_size);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_nlcc[i];
      find_end_element("PP_NLCC", upfin);
    }

    find_start_element("PP_LOCAL", upfin);
    std::vector<double> upf_vloc(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_vloc[i];
    find_end_element("PP_LOCAL",upfin);

    // add the vlocal into samples.
    // vlocal derivative is 0.0
    {
       for( int i = 0; i < upf_mesh_size; i++)
         upf_vloc[i] = 0.5*upf_vloc[i];
       std::vector < double > r; 
       std::vector < double > vr; 
       splinerad( upf_r, upf_vloc, r, vr, 1);

       Int n = r.size();
       DblNumVec spla(n,true,&vr[0]); 
       DblNumVec splb(n), splc(n), spld(n);
       spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

       seval(&upf_vloc[0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
           splc.Data(), spld.Data());

//       tk::spline s;
//       s.set_points(r, vr);
//       for( int i = 0; i < upf_r.size(); i ++)
//         upf_vloc[i] = s( upf_r[i] );
    }
    for( int i = 0; i < upf_mesh_size; i++)
       samples(i, 1) = upf_vloc[i];


    // set weights 0-4 to -1
    weights[1] = -1;
    weights[2] = -1;
    weights[3] = -1;
    weights[4] = -1;
    types[1] = 99;
    types[2] = 99;
    types[3] = 999;
    types[4] = 999;


    find_start_element("PP_NONLOCAL", upfin);
    std::vector<std::vector<double> > upf_vnl;
    upf_vnl.resize(upf_nproj);
    std::vector<int> upf_proj_l(upf_nproj);

    std::ostringstream os;
    for ( int j = 0; j < upf_nproj; j++ )
    {
      int index, angular_momentum;
      double cutoff_radius = 4.0; // 4.0 is big enough for Si as default

      os.str("");
      os << j+1;
      std::string element_name = "PP_BETA." + os.str();
      tag = find_start_element(element_name, upfin);
#if 0
      statusOFS << tag << std::endl;
#endif

      buf = get_attr(tag,"index");
      is.clear();
      is.str(buf);
      is >> index;
#if 0
      statusOFS << " index = " << index << std::endl;
#endif

      //reset nlcut
      buf = get_attr(tag,"cutoff_radius");
      is.clear();
      is.str(buf);
      is >> cutoff_radius;
      nlcut = cutoff_radius;

#if 0
      statusOFS << " cutoff_radius = " << cutoff_radius << std::endl;
#endif

      buf = get_attr(tag,"angular_momentum");
      is.clear();
      is.str(buf);
      is >> angular_momentum;
#if 0
      statusOFS << " angular_momentum = " << angular_momentum << std::endl;
#endif

      assert(angular_momentum <= upf_lmax);
      upf_proj_l[index-1] = angular_momentum;

      upf_vnl[j].resize(upf_mesh_size);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_vnl[j][i];

      find_end_element(element_name, upfin);

      /// take the element nonlocal part.
      {
         std::vector < double > r; 
         std::vector < double > vr; 
         if(angular_momentum % 2 == 0) 
           splinerad( upf_r, upf_vnl[j], r, vr, 0);
         else
           splinerad( upf_r, upf_vnl[j], r, vr, 1);

         for( int i = 0; i < r.size(); i++)
           vr[i] = vr[i] / r[i];

//         tk::spline s;
//         s.set_points(r, vr);
//
//         for( int i = 0; i < upf_r.size(); i ++)
//           upf_vnl[j][i] = s( upf_r[i] );

         Int n = r.size();
         DblNumVec spla(n,true,&vr[0]); 
         DblNumVec splb(n), splc(n), spld(n);
         spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

         seval(&upf_vnl[j][0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
             splc.Data(), spld.Data());

      }
 
      // nonlocal is written.
      // nonlocal derivative is 0.0
      for( int i = 0; i < upf_mesh_size; i++)
        samples(i, 5+j*2) = upf_vnl[j][i];
    }

    // compute number of projectors for each l
    // nproj_l[l] is the number of projectors having angular momentum l
    std::vector<int> nproj_l(upf_lmax+1);
    for ( int l = 0; l <= upf_lmax; l++ )
    {
      nproj_l[l] = 0;
      for ( int ip = 0; ip < upf_nproj; ip++ )
        if ( upf_proj_l[ip] == l ) nproj_l[l]++;
    }

    tag = find_start_element("PP_DIJ", upfin);
    int size;
    buf = get_attr(tag,"size");
    is.clear();
    is.str(buf);
    is >> size;
#if 0
    statusOFS << "PP_DIJ size = " << size << std::endl;
#endif

    if ( size != upf_nproj*upf_nproj )
    {
      statusOFS << " Number of non-zero Dij differs from number of projectors"
           << std::endl;
      return 1;
    }
    int upf_ndij = size;

    std::vector<double> upf_d(upf_ndij);
    for ( int i = 0; i < upf_ndij; i++ )
    {
      upfin >> upf_d[i];
    }
    int imax = sqrt(size+1.e-5);
    assert(imax*imax==size);

    // Check if Dij has non-diagonal elements
    // non-diagonal elements are not supported
    for ( int m = 0; m < imax; m++ )
      for ( int n = 0; n < imax; n++ )
        if ( (m != n) && (upf_d[n*imax+m] != 0.0) )
        {
          statusOFS << " Non-local Dij has off-diagonal elements" << std::endl;
          statusOFS << " m=" << m << " n=" << n << std::endl;
          return 1;
        }

    find_end_element("PP_DIJ", upfin);

    find_end_element("PP_NONLOCAL", upfin);

    // add the weights to the Dij
    for ( int j = 0; j < upf_nproj; j++ )
    {
      weights[5+2*j] = 0.5 * upf_d[j*imax+j];
      weights[6+2*j] = 0.5 * upf_d[j*imax+j];

      //weights[5+2*j] = upf_d[j*imax+j];
      //weights[6+2*j] = upf_d[j*imax+j];

      types[5+ 2*j] = upf_proj_l[j];
      types[6+ 2*j] = upf_proj_l[j];
      // FIXME nonlocal cutoff should be given by a table according to the element type.
      cutoffs[5+2*j] = nlcut;
      cutoffs[6+2*j] = nlcut;
    }

    find_start_element("PP_RHOATOM", upfin);
    std::vector<double> upf_rho_atom(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_rho_atom[i];

    //add the spline part
    {
       std::vector < double > r; 
       std::vector < double > vr; 
       splinerad( upf_r, upf_rho_atom, r, vr, 1);
       for( int i = 0; i < r.size(); i++)
         vr[i] = vr[i] / ( 4.0 * PI * r[i] * r[i] );

       Int n = r.size();
       DblNumVec spla(n,true,&vr[0]); 
       DblNumVec splb(n), splc(n), spld(n);
       spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

       seval(&upf_rho_atom[0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
           splc.Data(), spld.Data());

//       tk::spline s;
//       s.set_points(r, vr);
//       for( int i = 0; i < upf_r.size(); i ++)
//         upf_rho_atom[i] = s( upf_r[i] );
    }
 
    // add the rho_atom to the samples
    // rho_atom derivative is 0.0
    for( int i = 0; i < upf_mesh_size; i++)
      samples(i, 3) = upf_rho_atom[i] ;

    find_end_element("PP_RHOATOM", upfin);

    // make table iproj[l] mapping l to iproj
    // vnl(l) is in vnl[iproj[l]] if iproj[l] > -1
    // vlocal if iproj[llocal] = -1
    std::vector<int> iproj(upf_lmax+2);
    for ( int l = 0; l <= upf_lmax+1; l++ )
      iproj[l] = -1;
    for ( int j = 0; j < upf_nproj; j++ )
      iproj[upf_proj_l[j]] = j;

    // determine angular momentum of local potential in UPF file
    // upf_llocal is the angular momentum of the local potential
    // increase lmax if there are more projectors than wavefunctions
    int qso_lmax = upf_lmax;
    if (upf_lmax < upf_llocal)
    {
      qso_lmax = upf_lmax+1;
    }

#if 0
    if ( pseudo_type == "SL" )
    {
      find_start_element("PP_PSWFC", upfin);
      std::vector<std::vector<double> > upf_wf;
      std::vector<int> upf_wf_l(upf_nwf);
      upf_wf.resize(upf_nwf);
      for ( int j = 0; j < upf_nwf; j++ )
      {
        int index, l;
        os.str("");
        os << j+1;
        std::string element_name = "PP_CHI." + os.str();
        tag = find_start_element(element_name, upfin);
        statusOFS << tag << std::endl;

        buf = get_attr(tag,"index");
        is.clear();
        is.str(buf);
        is >> index;
        statusOFS << " index = " << index << std::endl;

        buf = get_attr(tag,"l");
        is.clear();
        is.str(buf);
        is >> l;
        statusOFS << " l = " << l << std::endl;

        assert(l <= upf_lmax);
        upf_proj_l[index-1] = l;
        upf_wf[j].resize(upf_mesh_size);
        for ( int i = 0; i < upf_mesh_size; i++ )
          upfin >> upf_wf[j][i];
      }
      find_end_element("PP_PSWFC", upfin);

      // output original data in file upf.dat
      std::ofstream upf("upf.dat");
      upf << "# vloc" << std::endl;
      for ( int i = 0; i < upf_vloc.size(); i++ )
        upf << upf_r[i] << " " << upf_vloc[i] << std::endl;
      upf << std::endl << std::endl;
      for ( int j = 0; j < upf_nproj; j++ )
      {
        upf << "# proj j=" << j << std::endl;
        for ( int i = 0; i < upf_vnl[j].size(); i++ )
          upf << upf_r[i] << " " << upf_vnl[j][i] << std::endl;
        upf << std::endl << std::endl;
      }
      for ( int j = 0; j < upf_nwf; j++ )
      {
        upf << "# wf j=" << j << std::endl;
        for ( int i = 0; i < upf_wf[j].size(); i++ )
          upf << upf_r[i] << " " << upf_wf[j][i] << std::endl;
        upf << std::endl << std::endl;
      }
      upf.close();

      // print summary
      statusOFS << "PP_INFO:" << std::endl << upf_pp_info << std::endl;
      statusOFS << "Element: " << upf_symbol << std::endl;
       statusOFS << "NLCC: " << upf_nlcc_flag << std::endl;
      //statusOFS << "XC: " << upf_xcf[0] << " " << upf_xcf[1] << " "
      //     << upf_xcf[2] << " " << upf_xcf[3] << std::endl;
      statusOFS << "Zv: " << upf_zval << std::endl;
      statusOFS << "lmax: " << qso_lmax << std::endl;
      statusOFS << "llocal: " << upf_llocal << std::endl;
      statusOFS << "nwf: " << upf_nwf << std::endl;
      statusOFS << "mesh_size: " << upf_mesh_size << std::endl;

      // compute delta_vnl[l][i] on the upf log mesh

      // divide the projector function by the wavefunction, except if
      // the wavefunction amplitude is smaller than tol, outside of rcut_divide.
      const double tol = 1.e-5;
      const double rcut_divide = 1.0;
      std::vector<std::vector<double> > delta_vnl;
      delta_vnl.resize(upf_nproj);
      for ( int j = 0; j < upf_nproj; j++ )
      {
        delta_vnl[j].resize(upf_wf[j].size());
        for ( int i = 0; i < delta_vnl[j].size(); i++ )
        {
          double num = upf_vnl[j][i];
          double den = upf_wf[upf_proj_l[j]][i];

          delta_vnl[j][i] = 0.0;
          if ( upf_r[i] < rcut_divide )
          {
            // near the origin
            if ( i == 0 && fabs(den) < tol )
            {
              // i = 0 for linear mesh, r = 0.0: use nearest value
              delta_vnl[j][i] = upf_vnl[j][1] / upf_wf[upf_proj_l[j]][1];
            }
            else
            {
              // other points near the origin
              delta_vnl[j][i] = num / den;
            }
          }
          else
          {
            // wavefunction gets small at large r.
            // Assume that delta_vnl is zero when that happens
            if ( fabs(den) > tol )
              delta_vnl[j][i] = num / den;
          }
        }
      }

      std::vector<std::vector<double> > vps;
      vps.resize(upf_nproj+1);
      for ( int j = 0; j < upf_nproj; j++ )
      {
        vps[j].resize(upf_mesh_size);
        for ( int i = 0; i < delta_vnl[j].size(); i++ )
          vps[j][i] = upf_vloc[i] + delta_vnl[j][i];
      }

      // interpolate functions on linear mesh
      const double mesh_spacing = 0.01;
      int nplin = (int) (rcut / mesh_spacing);
      std::vector<double> f(upf_mesh_size), fspl(upf_mesh_size);

      // interpolate NLCC
      std::vector<double> nlcc_lin(nplin);
      if ( upf_nlcc_flag == "T" )
      {
        assert(upf_mesh_size==upf_nlcc.size());
        for ( int i = 0; i < upf_nlcc.size(); i++ )
          f[i] = upf_nlcc[i];
        int n = upf_nlcc.size();
        int bcnat_left = 0;
        double yp_left = 0.0;
        int bcnat_right = 1;
        double yp_right = 0.0;
        spline(n,&upf_r[0],&f[0],yp_left,yp_right,
               bcnat_left,bcnat_right,&fspl[0]);

        for ( int i = 0; i < nplin; i++ )
        {
          double r = i * mesh_spacing;
          if ( r >= upf_r[0] )
            splint(n,&upf_r[0],&f[0],&fspl[0],r,&nlcc_lin[i]);
          else
            // use value closest to the origin for r=0
            nlcc_lin[i] = upf_nlcc[0];
        }
      }

      // interpolate vloc
      // factor 0.5: convert from Ry in UPF to Hartree in QSO
      for ( int i = 0; i < upf_vloc.size(); i++ )
        f[i] = 0.5 * upf_vloc[i];

      int n = upf_vloc.size();
      int bcnat_left = 0;
      double yp_left = 0.0;
      int bcnat_right = 1;
      double yp_right = 0.0;
      spline(n,&upf_r[0],&f[0],yp_left,yp_right,
             bcnat_left,bcnat_right,&fspl[0]);

      std::vector<double> vloc_lin(nplin);
      for ( int i = 0; i < nplin; i++ )
      {
        double r = i * mesh_spacing;
        if ( r >= upf_r[0] )
          splint(n,&upf_r[0],&f[0],&fspl[0],r,&vloc_lin[i]);
        else
          // use value closest to the origin for r=0
          vloc_lin[i] = 0.5 * upf_vloc[0];
      }

      // interpolate vps[j], j=0, nproj-1
      std::vector<std::vector<double> > vps_lin;
      vps_lin.resize(vps.size());
      for ( int j = 0; j < vps.size(); j++ )
      {
        vps_lin[j].resize(nplin);
      }

      for ( int j = 0; j < upf_nproj; j++ )
      {
        // factor 0.5: convert from Ry in UPF to Hartree in QSO
        for ( int i = 0; i < upf_vloc.size(); i++ )
          f[i] = 0.5 * vps[j][i];

        int n = upf_vloc.size();
        int bcnat_left = 0;
        double yp_left = 0.0;
        int bcnat_right = 1;
        double yp_right = 0.0;
        spline(n,&upf_r[0],&f[0],yp_left,yp_right,
               bcnat_left,bcnat_right,&fspl[0]);

        for ( int i = 0; i < nplin; i++ )
        {
          double r = i * mesh_spacing;
          if ( r >= upf_r[0] )
            splint(n,&upf_r[0],&f[0],&fspl[0],r,&vps_lin[j][i]);
          else
            vps_lin[j][i] = 0.5 * vps[j][0];
        }
      }

      // write potentials in gnuplot format on file vlin.dat
      std::ofstream vlin("vlin.dat");
      for ( int l = 0; l <= qso_lmax; l++ )
      {
        vlin << "# v, l=" << l << std::endl;
        if ( iproj[l] == -1 )
        {
          // l == llocal
          for ( int i = 0; i < nplin; i++ )
            vlin << i*mesh_spacing << " " << vloc_lin[i] << std::endl;
          vlin << std::endl << std::endl;
        }
        else
        {
          for ( int i = 0; i < nplin; i++ )
            vlin << i*mesh_spacing << " " << vps_lin[iproj[l]][i] << std::endl;
          vlin << std::endl << std::endl;
        }
      }

      // interpolate wavefunctions on the linear mesh

      std::vector<std::vector<double> > wf_lin;
      wf_lin.resize(upf_nwf);
      for ( int j = 0; j < upf_nwf; j++ )
      {
        wf_lin[j].resize(nplin);
        assert(upf_wf[j].size()<=f.size());
        for ( int i = 0; i < upf_wf[j].size(); i++ )
        {
          if ( upf_r[i] > 0.0 )
            f[i] = upf_wf[j][i] / upf_r[i];
          else
          {
            // value at origin, depending on angular momentum
            if ( upf_wf_l[j] == 0 )
            {
              // l=0: take value closest to r=0
              f[i] = upf_wf[j][1]/upf_r[1];
            }
            else
            {
              // l>0:
              f[i] = 0.0;
            }
          }
        }

        int n = upf_wf[j].size();
        // choose boundary condition at origin depending on angular momentum
        int bcnat_left = 0;
        double yp_left = 0.0;
        if ( upf_wf_l[j] == 1 )
        {
          bcnat_left = 1; // use natural bc
          double yp_left = 0.0; // not used
        }
        int bcnat_right = 1;
        double yp_right = 0.0;
        spline(n,&upf_r[0],&f[0],yp_left,yp_right,
               bcnat_left,bcnat_right,&fspl[0]);

        for ( int i = 0; i < nplin; i++ )
        {
          double r = i * mesh_spacing;
          if ( r >= upf_r[0] )
            splint(n,&upf_r[0],&f[0],&fspl[0],r,&wf_lin[j][i]);
          else
          {
            // r < upf_r[0]
            assert(upf_r[0]>0.0);
            // compute value near origin, depending on angular momentum
            if ( upf_wf_l[j] == 0 )
            {
              // l=0: take value closest to r=0
              wf_lin[j][i] = upf_wf[j][0]/upf_r[0];
            }
            else
            {
              // l>0:
              wf_lin[j][i] = upf_wf[j][0] * r / ( upf_r[0] * upf_r[0] );
            }
          }
        }

        vlin << "# phi, l=" << upf_l[j] << std::endl;
        for ( int i = 0; i < nplin; i++ )
          vlin << i*mesh_spacing << " " << wf_lin[j][i] << std::endl;
        vlin << std::endl << std::endl;
      }

      statusOFS << " interpolation done" << std::endl;
#endif
  #if 0
      // output potential on log mesh
      std::ofstream vout("v.dat");
      for ( int l = 0; l <= qso_lmax; l++ )
      {
        vout << "# v, l=" << l << std::endl;
        if ( iproj[l] == -1 )
        {
          // l == llocal
          for ( int i = 0; i < upf_vloc.size(); i++ )
            vout << upf_r[i] << " " << 0.5*upf_vloc[i] << std::endl;
          vout << std::endl << std::endl;
        }
        else
        {
          for ( int i = 0; i < vps[iproj[l]].size(); i++ )
            vout << upf_r[i] << " " << 0.5*vps[iproj[l]][i] << std::endl;
          vout << std::endl << std::endl;
        }
      }
      vout << std::endl << std::endl;
      vout.close();
  #endif
#if  0
      // Generate QSO file

      // output potential in QSO format
      // Weile, comment out the QSO format output
      if(0)
      {
        statusOFS << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
        statusOFS << "<fpmd:species xmlns:fpmd=\"http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0\"" << std::endl;
        statusOFS << "  xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" << std::endl;
        statusOFS << "  xsi:schemaLocation=\"http://www.quantum-simulation.org/ns/fpmd/fpmd-1.0"  << std::endl;
        statusOFS << "  species.xsd\">" << std::endl;
        statusOFS << "<description>" << std::endl;
        statusOFS << "Translated from UPF format by upf2qso" << std::endl;
        statusOFS << upf_pp_info;
        statusOFS << "</description>" << std::endl;
        statusOFS << "<symbol>" << upf_symbol << "</symbol>" << std::endl;
        statusOFS << "<atomic_number>" << atomic_number << "</atomic_number>" << std::endl;
        statusOFS << "<mass>" << mass << "</mass>" << std::endl;
        statusOFS << "<norm_conserving_pseudopotential>" << std::endl;
        statusOFS << "<valence_charge>" << upf_zval << "</valence_charge>" << std::endl;
        statusOFS << "<lmax>" << qso_lmax << "</lmax>" << std::endl;
        statusOFS << "<llocal>" << upf_llocal << "</llocal>" << std::endl;
        statusOFS << "<nquad>0</nquad>" << std::endl;
        statusOFS << "<rquad>0.0</rquad>" << std::endl;
        statusOFS << "<mesh_spacing>" << mesh_spacing << "</mesh_spacing>" << std::endl;
  
        statusOFS.setf(std::ios::scientific,std::ios::floatfield);
        if ( upf_nlcc_flag == "T" )
        {
          statusOFS << "<core_density size=\"" << nplin << "\">" << std::endl;
          for ( int i = 0; i < nplin; i++ )
            statusOFS << std::setprecision(10) << nlcc_lin[i] << std::endl;
          statusOFS << "</core_density>" << std::endl;
        }
  
        for ( int l = 0; l <= qso_lmax; l++ )
        {
          statusOFS << "<projector l=\"" << l << "\" size=\"" << nplin << "\">"
               << std::endl;
          statusOFS << "<radial_potential>" << std::endl;
          if ( iproj[l] == -1 )
          {
            // l == llocal
            for ( int i = 0; i < nplin; i++ )
              statusOFS << std::setprecision(10) << vloc_lin[i] << std::endl;
          }
          else
          {
            for ( int i = 0; i < nplin; i++ )
              statusOFS << std::setprecision(10) << vps_lin[iproj[l]][i] << std::endl;
          }
          statusOFS << "</radial_potential>" << std::endl;
          // find index j corresponding to angular momentum l
          int j = 0;
          while ( upf_wf_l[j] != l && j < upf_nwf ) j++;
          // check if found
          const bool found = ( j != upf_nwf );
          // print wf only if found
          if ( found )
          {
            statusOFS << "<radial_function>" << std::endl;
            for ( int i = 0; i < nplin; i++ )
              statusOFS << std::setprecision(10) << wf_lin[j][i] << std::endl;
            statusOFS << "</radial_function>" << std::endl;
          }
          statusOFS << "</projector>" << std::endl;
        }
        statusOFS << "</norm_conserving_pseudopotential>" << std::endl;
        statusOFS << "</fpmd:species>" << std::endl;
      }
    } // if SL
#endif
#if 0
    if ( pseudo_type == "NC" )
    {
      statusOFS << " NC potential" << std::endl;
      // output original data in file upf.dat
      std::ofstream upf("upf.dat");
      upf << "# vloc" << std::endl;
      for ( int i = 0; i < upf_vloc.size(); i++ )
        upf << upf_r[i] << " " << upf_vloc[i] << std::endl;
      upf << std::endl << std::endl;
      for ( int j = 0; j < upf_nproj; j++ )
      {
        upf << "# proj j=" << j << std::endl;
        for ( int i = 0; i < upf_vnl[j].size(); i++ )
          upf << upf_r[i] << " " << upf_vnl[j][i] << std::endl;
        upf << std::endl << std::endl;
      }

      upf << "# dij " << std::endl;
      for ( int j = 0; j < upf_d.size(); j++ )
      {
        upf << j << " " << upf_d[j] << std::endl;
      }
      upf.close();

      // print summary
      statusOFS << "PP_INFO:" << std::endl << upf_pp_info << std::endl;
      statusOFS << "Element: " << upf_symbol << std::endl;
      statusOFS << "NLCC: " << upf_nlcc_flag << std::endl;
      // statusOFS << "XC: " << upf_xcf[0] << " " << upf_xcf[1] << " "
      //      << upf_xcf[2] << " " << upf_xcf[3] << std::endl;
      statusOFS << "Zv: " << upf_zval << std::endl;
      statusOFS << "lmax: " << qso_lmax << std::endl;
      statusOFS << "nproj: " << upf_nproj << std::endl;
      statusOFS << "mesh_size: " << upf_mesh_size << std::endl;

      // interpolate functions on linear mesh
      const double mesh_spacing = 0.01;
      int nplin = (int) (rcut / mesh_spacing);
      std::vector<double> f(upf_mesh_size), fspl(upf_mesh_size);

      // interpolate NLCC
      std::vector<double> nlcc_lin(nplin);
      if ( upf_nlcc_flag == "T" )
      {
        assert(upf_mesh_size==upf_nlcc.size());
        for ( int i = 0; i < upf_nlcc.size(); i++ )
          f[i] = upf_nlcc[i];
        int n = upf_nlcc.size();
        int bcnat_left = 0;
        double yp_left = 0.0;
        int bcnat_right = 1;
        double yp_right = 0.0;
        spline(n,&upf_r[0],&f[0],yp_left,yp_right,
               bcnat_left,bcnat_right,&fspl[0]);

        for ( int i = 0; i < nplin; i++ )
        {
          double r = i * mesh_spacing;
          if ( r >= upf_r[0] )
            splint(n,&upf_r[0],&f[0],&fspl[0],r,&nlcc_lin[i]);
          else
            // use value closest to the origin for r=0
            nlcc_lin[i] = upf_nlcc[0];
        }
      }

      // interpolate vloc
      // factor 0.5: convert from Ry in UPF to Hartree in QSO
      for ( int i = 0; i < upf_vloc.size(); i++ )
        f[i] = 0.5 * upf_vloc[i];

      int n = upf_vloc.size();
      int bcnat_left = 0;
      double yp_left = 0.0;
      int bcnat_right = 1;
      double yp_right = 0.0;
      spline(n,&upf_r[0],&f[0],yp_left,yp_right,
             bcnat_left,bcnat_right,&fspl[0]);

      std::vector<double> vloc_lin(nplin);
      for ( int i = 0; i < nplin; i++ )
      {
        double r = i * mesh_spacing;
        if ( r >= upf_r[0] )
          splint(n,&upf_r[0],&f[0],&fspl[0],r,&vloc_lin[i]);
        else
          // use value closest to the origin for r=0
          vloc_lin[i] = 0.5 * upf_vloc[0];
      }


      // interpolate vnl[j], j=0, nproj-1
      std::vector<std::vector<double> > vnl_lin;
      vnl_lin.resize(upf_nproj);
      for ( int j = 0; j < vnl_lin.size(); j++ )
      {
        vnl_lin[j].resize(nplin);
      }

      for ( int j = 0; j < upf_nproj; j++ )
      {
        // factor 0.5: convert from Ry in UPF to Hartree in QSO
        for ( int i = 0; i < upf_vnl.size(); i++ )
          f[i] = 0.5 * upf_vnl[j][i];

        int n = upf_vloc.size();
        int bcnat_left = 0;
        double yp_left = 0.0;
        int bcnat_right = 1;
        double yp_right = 0.0;
        spline(n,&upf_r[0],&f[0],yp_left,yp_right,
               bcnat_left,bcnat_right,&fspl[0]);

        for ( int i = 0; i < nplin; i++ )
        {
          double r = i * mesh_spacing;
          if ( r >= upf_r[0] )
            splint(n,&upf_r[0],&f[0],&fspl[0],r,&vnl_lin[j][i]);
          else
            vnl_lin[j][i] = 0.5 * upf_vnl[j][0];
        }
      }

      // write local potential and projectors in gnuplot format on file vlin.dat
      std::ofstream vlin("vlin.dat");
      vlin << "# vlocal" << std::endl;
      for ( int i = 0; i < nplin; i++ )
        vlin << vloc_lin[i] << std::endl;
      vlin << std::endl << std::endl;
      for ( int iproj = 0; iproj < vnl_lin.size(); iproj++ )
      {
        vlin << "# projector, l=" << upf_proj_l[iproj] << std::endl;
        for ( int i = 0; i < nplin; i++ )
          vlin << i*mesh_spacing << " " << vnl_lin[iproj][i] << std::endl;
        vlin << std::endl << std::endl;
      }

    }
#endif
  } // version 1 or 2
  return 0;
}

} // namespace dgdft

