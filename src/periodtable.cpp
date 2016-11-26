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
/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @date 2012-08-10
#include "periodtable.hpp"
#include "esdf.hpp"
#include "utility.hpp"


namespace  dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::esdf;

// *********************************************************************
// PTEntry
// *********************************************************************

Int serialize(const PTEntry& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.params, os, mask);
  serialize(val.samples, os, mask);
  serialize(val.weights, os, mask);
  serialize(val.types, os, mask);
  serialize(val.cutoffs, os, mask);
  return 0;
}

Int deserialize(PTEntry& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.params, is, mask);
  deserialize(val.samples, is, mask);
  deserialize(val.weights, is, mask);
  deserialize(val.types, is, mask);
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

void PeriodTable::Setup( )
{
  std::vector<Int> all(1,1);

  std::istringstream iss;  
  SharedRead( esdfParam.periodTableFile, iss );
  deserialize(ptemap_, iss, all);

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

  // Setup constant private variable parameters
  if( esdfParam.pseudoType == "HGH" ){
    ptsample_.RADIAL_GRID       = 0;
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;
    ptsample_.RHOATOM           = -999;
    ptsample_.DRV_RHOATOM       = -999;
    ptsample_.NONLOCAL          = 3;
  }
  if( esdfParam.pseudoType == "ONCV" ){
    ptsample_.RADIAL_GRID       = 0;
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;
    ptsample_.RHOATOM           = 3;
    ptsample_.DRV_RHOATOM       = 4;
    ptsample_.NONLOCAL          = 5;
  }

  // Common so far for all pseudopotential

  {
    pttype_.RADIAL            = 9;
    pttype_.PSEUDO_CHARGE     = 99;
    pttype_.RHOATOM           = 999;
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
    ptparam_.ESELF  = 3;
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
    //
    std::vector<DblNumVec>& valspl = spldata[ptsample_.RHOATOM]; 
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
        valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
    
    for(Int g=0; g<idx.size(); g++) {
      atomDensity[idx[g]] = val[g];
    }
  } 

  return ;
}         // -----  end of method PeriodTable::CalculateAtomDensity  ----- 


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



} // namespace dgdft

