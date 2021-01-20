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
#include "blas.hpp"
#include "lapack.hpp"

#ifdef DG_HAS_CONFIG
#include "config.hpp"
#endif

#include <sstream>

namespace  dgdft{

using namespace dgdft::PseudoComponent;
using namespace dgdft::esdf;



// *********************************************************************
// The following are modified based on the UPF2QSO subroutine in Qbox
// So the format is slightly different from the rest of PeriodTable
// Some properties like configurations are not used
// *********************************************************************
struct Element
{
  int z;
  std::string symbol;
  std::string config;
  double mass;
  double rgaussian;
  double vsrcut;
  double vnlcut;
  double rhoatomcut;
  Element(int znuc, std::string s, std::string c, double m, 
      double sr = 4.0, double rgauss = 0.5, 
      double nl = 3.0, double rhoatom = 6.0 ) : 
    z(znuc), symbol(s), config(c), mass(m), 
    vsrcut(sr), rgaussian(rgauss), vnlcut(nl), rhoatomcut(rhoatom) {}
};

class ElementTable
{
  private:

  std::vector<Element> etable;
  std::map<std::string,int> zmap;

  public:

  ElementTable(void);
  int z(std::string symbol) const;
  std::string symbol(int zval) const;
  std::string configuration(int zval) const;
  std::string configuration(std::string symbol) const;
  double mass(int zval) const;
  double mass(std::string symbol) const;
  int size(void) const;


  double rgaussian(std::string symbol) const { return etable[z(symbol)-1].rgaussian; };
  double vsrcut(std::string symbol) const { return etable[z(symbol)-1].vsrcut; };
  double vnlcut(std::string symbol) const { return etable[z(symbol)-1].vnlcut; };
  double rhoatomcut(std::string symbol) const { return etable[z(symbol)-1].rhoatomcut; };
};

// LL: FIXME 01/14/2021 The cutoff values should be carefully tested and
// added to the table to override the default values, at least for elements of interest
// this require plotting the pseudopotential 
ElementTable::ElementTable(void)
{
  etable.push_back(Element(1,"H","1s1",1.00794));
  etable.push_back(Element(2,"He","1s2",4.00260));
  etable.push_back(Element(3, "Li","1s2 2s1",     6.941));
  etable.push_back(Element(4, "Be","1s2 2s2",     9.01218));
  etable.push_back(Element(5, "B", "1s2 2s2 2p1",10.811));
  etable.push_back(Element(6, "C", "1s2 2s2 2p2",12.0107));
  etable.push_back(Element(7, "N", "1s2 2s2 2p3",14.00674));
  etable.push_back(Element(8, "O", "1s2 2s2 2p4",15.9994));
  etable.push_back(Element(9, "F", "1s2 2s2 2p5",18.9884));
  etable.push_back(Element(10,"Ne","1s2 2s2 2p6",20.1797));

  etable.push_back(Element(11,"Na","[Ne] 3s1",    22.98977));
  etable.push_back(Element(12,"Mg","[Ne] 3s2",    24.3050));
  etable.push_back(Element(13,"Al","[Ne] 3s2 3p1",26.98154));
  etable.push_back(Element(14,"Si","[Ne] 3s2 3p2",28.0855));
  etable.push_back(Element(15,"P", "[Ne] 3s2 3p3",30.97376));
  etable.push_back(Element(16,"S", "[Ne] 3s2 3p4",32.066));
  etable.push_back(Element(17,"Cl","[Ne] 3s2 3p5",35.4527));
  etable.push_back(Element(18,"Ar","[Ne] 3s2 3p6",39.948));

  etable.push_back(Element(19,"K", "[Ar] 4s1",39.0983));
  etable.push_back(Element(20,"Ca","[Ar] 4s2",40.078));
  etable.push_back(Element(21,"Sc","[Ar] 3d1 4s2",44.95591));
  etable.push_back(Element(22,"Ti","[Ar] 3d2 4s2",47.867));
  etable.push_back(Element(23,"V", "[Ar] 3d3 4s2",50.9415));
  etable.push_back(Element(24,"Cr","[Ar] 3d5 4s1",51.9961));
  etable.push_back(Element(25,"Mn","[Ar] 3d5 4s2",54.93805));
  etable.push_back(Element(26,"Fe","[Ar] 3d6 4s2",55.845));
  etable.push_back(Element(27,"Co","[Ar] 3d7 4s2",58.9332));
  etable.push_back(Element(28,"Ni","[Ar] 3d8 4s2",58.6934));
  etable.push_back(Element(29,"Cu","[Ar] 3d10 4s1",63.546));
  etable.push_back(Element(30,"Zn","[Ar] 3d10 4s2",65.39));
  etable.push_back(Element(31,"Ga","[Ar] 3d10 4s2 4p1",69.723));
  etable.push_back(Element(32,"Ge","[Ar] 3d10 4s2 4p2",72.61));
  etable.push_back(Element(33,"As","[Ar] 3d10 4s2 4p3",74.9216));
  etable.push_back(Element(34,"Se","[Ar] 3d10 4s2 4p4",78.96));
  etable.push_back(Element(35,"Br","[Ar] 3d10 4s2 4p5",79.904));
  etable.push_back(Element(36,"Kr","[Ar] 3d10 4s2 4p6",83.80));

  etable.push_back(Element(37,"Rb","[Kr] 5s1",85.4678));
  etable.push_back(Element(38,"Sr","[Kr] 5s2",87.62));
  etable.push_back(Element(39,"Y" ,"[Kr] 4d1 5s2",88.90585));
  etable.push_back(Element(40,"Zr","[Kr] 4d2 5s2",91.224));
  etable.push_back(Element(41,"Nb","[Kr] 4d4 5s1",92.90638));
  etable.push_back(Element(42,"Mo","[Kr] 4d5 5s1",95.94));
  etable.push_back(Element(43,"Tc","[Kr] 4d5 5s2",98.0));
  etable.push_back(Element(44,"Ru","[Kr] 4d7 5s1",101.07));
  etable.push_back(Element(45,"Rh","[Kr] 4d8 5s1",102.9055));
  etable.push_back(Element(46,"Pd","[Kr] 4d10",106.42));
  etable.push_back(Element(47,"Ag","[Kr] 4d10 5s1",107.8682));
  etable.push_back(Element(48,"Cd","[Kr] 4d10 5s2",112.411));
  etable.push_back(Element(49,"In","[Kr] 4d10 5s2 5p1",114.818));
  etable.push_back(Element(50,"Sn","[Kr] 4d10 5s2 5p2",118.710));
  etable.push_back(Element(51,"Sb","[Kr] 4d10 5s2 5p3",121.760));
  etable.push_back(Element(52,"Te","[Kr] 4d10 5s2 5p4",127.60));
  etable.push_back(Element(53,"I" ,"[Kr] 4d10 5s2 5p5",126.90447));
  etable.push_back(Element(54,"Xe","[Kr] 4d10 5s2 5p6",131.29));

  etable.push_back(Element(55,"Cs","[Xe] 6s1",132.90545));
  etable.push_back(Element(56,"Ba","[Xe] 6s2",137.327));
  etable.push_back(Element(57,"La","[Xe] 5d1 6s2",138.9055));
  etable.push_back(Element(58,"Ce","[Xe] 4f1 5d1 6s2",140.116));
  etable.push_back(Element(59,"Pr","[Xe] 4f3 6s2",140.90765));
  etable.push_back(Element(60,"Nd","[Xe] 4f4 6s2",144.24));
  etable.push_back(Element(61,"Pm","[Xe] 4f5 6s2",145.0));
  etable.push_back(Element(62,"Sm","[Xe] 4f6 6s2",150.36));
  etable.push_back(Element(63,"Eu","[Xe] 4f7 6s2",151.964));
  etable.push_back(Element(64,"Gd","[Xe] 4f7 5d1 6s2",157.25));
  etable.push_back(Element(65,"Tb","[Xe] 4f9 6s2",158.92534));
  etable.push_back(Element(66,"Dy","[Xe] 4f10 6s2",162.50));
  etable.push_back(Element(67,"Ho","[Xe] 4f11 6s2",164.93032));
  etable.push_back(Element(68,"Er","[Xe] 4f12 6s2",167.26));
  etable.push_back(Element(69,"Tm","[Xe] 4f13 6s2",168.93421));
  etable.push_back(Element(70,"Yb","[Xe] 4f14 6s2",173.04));
  etable.push_back(Element(71,"Lu","[Xe] 4f14 5d1 6s2",174.967));
  etable.push_back(Element(72,"Hf","[Xe] 4f14 5d2 6s2",178.49));
  etable.push_back(Element(73,"Ta","[Xe] 4f14 5d3 6s2",180.9479));
  etable.push_back(Element(74,"W" ,"[Xe] 4f14 5d4 6s2",183.84));
  etable.push_back(Element(75,"Re","[Xe] 4f14 5d5 6s2",186.207));
  etable.push_back(Element(76,"Os","[Xe] 4f14 5d6 6s2",190.23));
  etable.push_back(Element(77,"Ir","[Xe] 4f14 5d7 6s2",192.217));
  etable.push_back(Element(78,"Pt","[Xe] 4f14 5d9 6s1",195.078));
  etable.push_back(Element(79,"Au","[Xe] 4f14 5d10 6s1",196.96655));
  etable.push_back(Element(80,"Hg","[Xe] 4f14 5d10 6s2",200.59));
  etable.push_back(Element(81,"Tl","[Xe] 4f14 5d10 6s2 6p1",204.3833));
  etable.push_back(Element(82,"Pb","[Xe] 4f14 5d10 6s2 6p2",207.2));
  etable.push_back(Element(83,"Bi","[Xe] 4f14 5d10 6s2 6p3",208.98038));
  etable.push_back(Element(84,"Po","[Xe] 4f14 5d10 6s2 6p4",209.0));
  etable.push_back(Element(85,"At","[Xe] 4f14 5d10 6s2 6p5",210.0));
  etable.push_back(Element(86,"Rn","[Xe] 4f14 5d10 6s2 6p6",222.0));

  etable.push_back(Element(87,"Fr","[Rn] 7s1",223.0));
  etable.push_back(Element(88,"Ra","[Rn] 7s2",226.0));
  etable.push_back(Element(89,"Ac","[Rn] 6d1 7s2",227.0));
  etable.push_back(Element(90,"Th","[Rn] 6d2 7s2",232.0381));
  etable.push_back(Element(91,"Pa","[Rn] 5f2 6d1 7s2",231.03588));
  etable.push_back(Element(92,"U" ,"[Rn] 5f3 6d1 7s2",238.0289));
  etable.push_back(Element(93,"Np","[Rn] 5f4 6d1 7s2",237.0));
  etable.push_back(Element(94,"Pu","[Rn] 5f5 6d1 7s2",244.0));

  for ( int i = 0; i < etable.size(); i++ )
    zmap[etable[i].symbol] = i+1;
}


int ElementTable::z(std::string symbol) const
{
  std::map<std::string,int>::const_iterator i = zmap.find(symbol);
  assert( i != zmap.end() );
  return (*i).second;
}

std::string ElementTable::symbol(int z) const
{
  assert(z>0 && z<=etable.size());
  return etable[z-1].symbol;
}

std::string ElementTable::configuration(int z) const
{
  assert(z>0 && z<=etable.size());
  return etable[z-1].config;
}

std::string ElementTable::configuration(std::string symbol) const
{
  return etable[z(symbol)-1].config;
}

double ElementTable::mass(int z) const
{
  assert(z>0 && z<=etable.size());
  return etable[z-1].mass;
}

double ElementTable::mass(std::string symbol) const
{
  return etable[z(symbol)-1].mass;
}

int ElementTable::size(void) const
{
  return etable.size();
}



// *********************************************************************
// Some utility routines
// *********************************************************************

// The potential due to a Gaussian compensation charge
inline Real VGaussian(Real rad, Real Zval, Real RGaussian){
  Real EPS = 1e-12;

  if( rad < EPS )
    return Zval / RGaussian * 2.0 / std::sqrt(PI);
  else
    return Zval / rad * std::erf(rad / RGaussian);
}

// A smooth transition function 
// \[
// g(x)=\frac{f((b-x)/c)}{f((x-a)/c)+f((b-x)/c)}
// \]
// where
// f(x)=e^{-5/x}, x>0
// f(x)=0,        x<=0
// c=(b-a)/2
// 
// this function satisfies
// g(x) = 1, x <= a
// g(x) = 0, x >= b 
// 0<g(x)<1, a<x<b
// g((a+b)/2) = 1/2
// g'((a+b)/2) \propto -1/(b-a)
inline Real SmoothTransition(Real x, Real a, Real b){
  if( a >= b )
    ErrorHandling("Smooth transition function must have a<b");

  Real c = (b-a) * 0.5;
  auto f = [](auto x){ return ( x > 0.0 ) ? std::exp(-2.5/x) : 0.0; };
  Real g = f((b-x)/ c) / (f((b-x)/ c) + f((x-a)/ c));
  return g;
}


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
  // Setup constant private variable parameters
  if( esdfParam.pseudoType == "oncv" or esdfParam.pseudoType == "hgh"){
    ptsample_.RADIAL_GRID       = 0;

    ptsample_.VLOCAL            = 1;
    ptsample_.DRV_VLOCAL        = 2;

    ptsample_.RHOATOM           = 3;
    ptsample_.DRV_RHOATOM       = 4;
    ptsample_.NONLOCAL          = 5;
  }
  else{
    ErrorHandling("Unsupported pseudopotential type.");
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
    ptparam_.ZVAL   = 2;
    ptparam_.RGAUSSIAN = 3;
  }

  {
    // all the readins are in the samples in the old version, 
    // now in the new version, I should readin something else. 
    // the readin are reading in a sequence of numbers, which
    // is used to construct the ptemap_ struct.
    // Read from UPF file
    MPI_Barrier(MPI_COMM_WORLD);
    int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    if(mpirank==0) {
      for( int i = 0; i < esdfParam.pspFile.size(); i++){
        PTEntry tempEntry;
        int atom;
        ReadUPF(esdfParam.pspFile[i], tempEntry, atom);
        std::map <Int,PTEntry> :: iterator it = ptemap_.end();
        ptemap_.insert(it, std::pair<Int, PTEntry>(atom, tempEntry));
      }
    }

    // implement the MPI Bcast of the ptemap_, now we are doing all processors readin
    std::stringstream vStream;
    std::stringstream vStreamTemp;
    int vStreamSize;

    std::vector<Int> all(1,1);
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

  // Post-processing Vlocal data 
  // 1. Remove the contribution from the Gaussian compensation charge so
  //    that vlocal is indeed short ranged
  // 2. LL: FIXME 01/14/2021  Add a smooth transition function to make vlocal compactly supported.
  //    If this works the smooth transition function should be applied to other functions
  //    with real space cutoffs as well.
  for(auto& mi: ptemap_) {
    Int type = mi.first;    
    PTEntry& ptcur = mi.second;
    DblNumVec& params = ptcur.params;
    DblNumMat& samples = ptcur.samples;
    Int nspl = samples.m();
    Real Zval = params(ptparam_.ZVAL);
    Real RGaussian = params(ptparam_.RGAUSSIAN);
    DblNumVec rad(nspl, false, samples.VecData(ptsample_.RADIAL_GRID));
    DblNumVec vlocal(nspl, false, samples.VecData(ptsample_.VLOCAL));

    for(Int i = 0; i < rad.m(); i++){
      vlocal[i] += VGaussian(rad[i], Zval, RGaussian);
    }
    // Multiply the smooth transition function
    Real Rzero = this->RcutVLocal( type );
    assert(Rzero > 1.0);
    for(Int i = 0; i < rad.m(); i++){
      vlocal[i] *= SmoothTransition(rad[i], Rzero-1.0, Rzero);
    }
#if 0
    statusOFS << "radial grid = " << rad << std::endl;
    statusOFS << "VLocal SR for type " << type << " = " << vlocal << std::endl;
#endif
  }


  // Create splines
  for(auto& mi: ptemap_) {
    Int type = mi.first;    
    PTEntry& ptcur = mi.second;
    DblNumVec& params = ptcur.params;
    DblNumMat& samples = ptcur.samples;
    std::map< Int, std::vector<DblNumVec> > spltmp;
    for(Int g=1; g<samples.n(); g++) {
      Int nspl = samples.m();
      DblNumVec rad(nspl, true, samples.VecData(ptsample_.RADIAL_GRID));
      DblNumVec a(nspl, true, samples.VecData(g));
      DblNumVec b(nspl), c(nspl), d(nspl);
      spline(nspl, rad.Data(), a.Data(), b.Data(), c.Data(), d.Data());
      std::vector<DblNumVec> aux(5);
      aux[0] = rad;      aux[1] = a;      aux[2] = b;      aux[3] = c;      aux[4] = d;
      spltmp[g] = aux;
    }
    splmap_[type] = spltmp;
  }
}         // -----  end of method PeriodTable::Setup  ----- 

// LL: FIXME 01/12/2021  This is currently not used
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


//void
//PeriodTable::CalculatePseudoCharge    (
//    const Atom& atom, 
//    const Domain& dm,
//    const NumTns<std::vector<DblNumVec> >& gridposElem,
//    NumTns<SparseVec>& res )
//{
//  Index3 numElem( gridposElem.m(), gridposElem.n(), gridposElem.p() );
//
//  res.Resize( numElem[0], numElem[1], numElem[2] );
//
//  for( Int elemk = 0; elemk < numElem[2]; elemk++ )
//    for( Int elemj = 0; elemj < numElem[1]; elemj++ )
//      for( Int elemi = 0; elemi < numElem[0]; elemi++ ){
//        CalculatePseudoCharge( atom, dm, 
//            gridposElem(elemi, elemj, elemk),
//            res( elemi, elemj, elemk ) );
//      } // for (elemi)
//
//
//  return ;
//}         // -----  end of method PeriodTable::CalculatePseudoCharge  ----- 


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

  Int type   = atom.type;
  Point3 pos = atom.pos;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutNonlocal( type );    

  // Initialize
  // First count all the pseudopotentials
  Int numpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g++) {
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
    // Process non-local pseudopotential one by one
    Int cntpp = 0;
    for(Int j=ptsample_.NONLOCAL; j<ptentry.samples.n(); j++) {
      Real wgt = ptentry.weights(j);
      Int typ = ptentry.types(j);
      //
      std::vector<DblNumVec>& valspl = spldata[j]; 
      std::vector<Real> val(idxsize,0.0);
      seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
      //
      // std::vector<DblNumVec>& derspl = spldata[j+1]; 
      // std::vector<Real> der(idxsize,0.0);
      // seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
      //--
      if(typ==pttype_.L0) {
        Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        SetValue(dv, 0.0);
        //
        for(Int g=0; g<idx.size(); g++) {
          if( rad[g]>MIN_RADIAL ) {
            dv(g,VAL) = coef * val[g];
//            dv(g,DX) = coef * der[g] * xx[g]/rad[g];
//            dv(g,DY) = coef * der[g] * yy[g]/rad[g];
//            dv(g,DZ) = coef * der[g] * zz[g]/rad[g];
          } else {
            dv(g,VAL) = coef * val[g];
//            dv(g,DX) = 0;
//            dv(g,DY) = 0;
//            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      } // if(typ==pttype_.L0);

      if(typ==pttype_.L1) {
        Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
          for(Int g=0; g<idx.size(); g++) {
            if( rad[g]> MIN_RADIAL ) {
              dv(g,VAL) = coef*( (xx[g]/rad[g]) * val[g] );
//              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(xx[g]/rad[g]) + val[g]/rad[g] );
//              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(yy[g]/rad[g])                 );
//              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(zz[g]/rad[g])                 );
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = coef*der[g];
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]> MIN_RADIAL) {
              dv(g,VAL) = coef*( (yy[g]/rad[g]) * val[g] );
//              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(xx[g]/rad[g])                 );
//              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(yy[g]/rad[g]) + val[g]/rad[g] );
//              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(zz[g]/rad[g])                 );
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = coef*der[g];
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        {
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              dv(g,VAL) = coef*( (zz[g]/rad[g]) * val[g] );
//              dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(xx[g]/rad[g])                 );
//              dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(yy[g]/rad[g])                 );
//              dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(zz[g]/rad[g]) + val[g]/rad[g] );
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = coef*der[g];
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
          SetValue(dv, 0.0);
          DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
          for(Int g=0; g<idx.size(); g++) {
            if(rad[g]>MIN_RADIAL) {
              Ylm(0) = coef*(-xx[g]*xx[g]-yy[g]*yy[g]+2.0*zz[g]*zz[g]) / (rad[g]*rad[g]);
              Ylm(1) = coef*(-6.0 * xx[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
              Ylm(2) = coef*(-6.0 * yy[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
              Ylm(3) = coef*( 6.0 * zz[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)) / pow(rad[g], 4.0));

              dv(g,VAL) = Ylm(0) * val[g] ;
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_yz
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);

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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_xz
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);

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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];

            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_xy
        {
          Real coef = 1.0/2.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // d_x^2-y^2
        {
          Real coef = 1.0/4.0*sqrt(15.0/PI);
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
      } // if(typ==pttype_.L2)

      // LL FIXME: 10/21/2013 The derivative at r=0 for the f orbital
      // was not set properly. But this does not matter for now
      if(typ==pttype_.L3) {
        // f_z3
        {
          Real coef = 1.0/4.0*sqrt(7.0/PI); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_y(3xx-yy)
        {
          Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_x(xx-3yy)
        {
          Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_z(xx-yy)
        {
          Real coef = 1.0/4.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_xyz
        {
          Real coef = 1.0/2.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_yzz
        {
          Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
        // f_xzz
        {
          Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
          IntNumVec iv(idx.size(), true, &(idx[0]));
          DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
          SetValue(dv, 0.0);
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
//              dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
//              dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
//              dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
            } else {
              dv(g,VAL) = 0;
//              dv(g,DX) = 0;
//              dv(g,DY) = 0;
//              dv(g,DZ) = 0;
            }
          }
          vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
        }
      } // if(typ==pttype_.L3)
    } // for (j)

    // Check the number of pseudopotentials
    if( cntpp != numpp ){
      ErrorHandling("cntpp != numpp.  Seriously wrong with nonlocal pseudopotentials.");
    }
  } // if (norm(minDist) <= Rzero )

  return ;
}         // -----  end of method PeriodTable::CalculateNonlocalPP  ----- 

//void
//PeriodTable::CalculateNonlocalPP    ( 
//    const Atom&    atom, 
//    const Domain&  dm, 
//    const NumTns<std::vector<DblNumVec> >&   gridposElem,
//    std::vector<std::pair<NumTns<SparseVec>, Real> >& vnlList )
//{
//  Point3 Ls       = dm.length;
//  Point3 posStart = dm.posStart;
//  Index3 Ns       = dm.numGridFine;
//
//#ifndef _NO_NONLOCAL_ // Nonlocal potential is used. Debug option
//  Int type   = atom.type;
//  Point3 pos = atom.pos;
//
//  //get entry data and spline data
//  PTEntry& ptentry = ptemap_[type];
//
//  // First count all the pseudopotentials
//  Int numpp = 0;
//  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g++) {
//    Int typ = ptentry.types(g);
//
//    if(typ==0)
//      numpp=numpp+1;
//    if(typ==1)
//      numpp=numpp+3;
//    if(typ==2)
//      numpp=numpp+5;
//    if(typ==3)
//      numpp=numpp+7;
//  }
//#if ( _DEBUGlevel_ >= 1 )
//  statusOFS << "Atom type " << type << " has " << numpp << 
//    " nonlocal projectors." << std::endl;
//#endif
//
//  vnlList.clear();
//  vnlList.resize( numpp );
//
//  Index3 numElem( gridposElem.m(), gridposElem.n(), gridposElem.p() );
//
//  // Initialize
//  for( Int p = 0; p < numpp; p++ ){
//    vnlList[p].first.Resize( numElem[0], numElem[1], numElem[2] );
//  } // for (p)
//
//  // Evalaute the nonlocal pseudopotential on each element
//  for( Int elemk = 0; elemk < numElem[2]; elemk++ )
//    for( Int elemj = 0; elemj < numElem[1]; elemj++ )
//      for( Int elemi = 0; elemi < numElem[0]; elemi++ ){
//        std::vector<NonlocalPP>  vnlpp;
//        CalculateNonlocalPP( atom, dm,
//            gridposElem(elemi, elemj, elemk),
//            vnlpp );
//        if( vnlList.size() != vnlpp.size() ){
//          std::ostringstream msg;
//          msg << "The number of pseudopotentials do not match." << std::endl;
//          ErrorHandling( msg.str().c_str() );
//        }
//        for( Int p = 0; p < numpp; p++ ){
//          NumTns<SparseVec>& res = vnlList[p].first;
//          res( elemi, elemj, elemk ) = vnlpp[p].first;
//          vnlList[p].second = vnlpp[p].second;
//        } // for (p)
//      } // for (elemi)
//
//#endif // #ifndef _NO_NONLOCAL_
//
//  return ;
//}         // -----  end of method PeriodTable::CalculateNonlocalPP  ----- 



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
  Real Rzero = this->RcutVLocal( type );
  Real RGaussian = this->RGaussian( type );
  Real Zval = this->Zval( type );

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

//    std::vector<DblNumVec>& derspl = spldata[ptsample_.DRV_VLOCAL];
//    std::vector<Real> der(idxsize,0.0);
//    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
//        derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());

    IntNumVec iv(idx.size(), true, &(idx[0])); 
    DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
    for(Int g=0; g<idx.size(); g++) {
      // FIXME derivatives later
      if( rad[g]> MIN_RADIAL ) {
        dv(g, VAL) = val[g];
//        dv(g, DX) = der[g] * xx[g]/rad[g];
//        dv(g, DY) = der[g] * yy[g]/rad[g];
//        dv(g, DZ) = der[g] * zz[g]/rad[g];
      } else {
        dv(g, VAL) = val[g];
//        dv(g, DX) = 0;
//        dv(g, DY) = 0;
//        dv(g, DZ) = 0;
      }
    }
    resVLocalSR = SparseVec(iv,dv);

    // Gaussian pseudocharge
    SetValue(dv, D_ZERO);
    Real fac = Zval / std::pow(std::sqrt(PI) * RGaussian,3);
    for(Int g=0; g<idx.size(); g++) {
      // FIXME derivatives later
      if( rad[g]> MIN_RADIAL ) {
        dv(g, VAL) = fac * std::exp(-(rad[g]/RGaussian)*(rad[g]/RGaussian)) ;
//        dv(g, DX) = der[g] * xx[g]/rad[g];
//        dv(g, DY) = der[g] * yy[g]/rad[g];
//        dv(g, DZ) = der[g] * zz[g]/rad[g];
      } else {
        dv(g, VAL) = fac * std::exp(-(rad[g]/RGaussian)*(rad[g]/RGaussian)) ;
//        dv(g, DX) = 0;
//        dv(g, DY) = 0;
//        dv(g, DZ) = 0;
      }
    }
    resGaussianPseudoCharge = SparseVec(iv,dv);
  } // if (norm(minDist) <= Rzero )

  return ;
}         // -----  end of method PeriodTable::CalculateVLocal  ----- 


// LL: FIXME 01/12/2021 Need to consider the correction due to the
// overlap of Gaussian charges 
Real PeriodTable::SelfIonInteraction(Int type) 
{
  Real eself;
  Real RGaussian = this->RGaussian( type );
  Real Zval = this->Zval( type );
  eself = Zval * Zval / ( std::sqrt(2.0 * PI) * RGaussian );
  
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

// LL: FIXME 01/14/2021 After Reading UPF, should print out relevant
// information for each pseudopotential for debugging purposes. This can
// also be useful information for the user
void PeriodTable::ReadUPF( std::string file_name, PTEntry& tempEntry, Int& atom)
{

  DblNumVec & params    = tempEntry.params;
  DblNumMat & samples   = tempEntry.samples;
  DblNumVec & weights   = tempEntry.weights;
  IntNumVec & types     = tempEntry.types;
  DblNumVec & cutoffs   = tempEntry.cutoffs;

  params.Resize(5); // in the order of the ParamPT
  
  ElementTable etable;

  std::string buf,s;
  std::istringstream is;

  // Determine if UPF file is specified as a KW
  // LL: FIXME 01/14/2021 Change the name KW?
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
  else{
    statusOFS << " * Reading from standard UPF file " << file_name << std::endl;
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
    }
    if ( upf_version != 2 )
      ErrorHandling( " Only UPF format 2.0 is supported." );
  
  } else {
    std::ostringstream msg;
    msg << "UPF File " << file_name << " not found";
    ErrorHandling( msg.str() );
  }

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
    // check if potential is norm-conserving or semi-local
    std::string pseudo_type = get_attr(tag,"pseudo_type");
#if 0
    statusOFS << " pseudo_type = " << pseudo_type << std::endl;
#endif

    // NLCC flag
    std::string upf_nlcc_flag = get_attr(tag,"core_correction");
    if ( upf_nlcc_flag == "T" )
    {
      statusOFS << " Potential includes a non-linear core correction" << std::endl;
      statusOFS << " WARNING: this is currently not supported." << std::endl;
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
    
    atom = etable.z(upf_symbol);
    params[ptparam_.ZNUC] = atom;
    params[ptparam_.MASS] = etable.mass(upf_symbol);
    params[ptparam_.ZVAL] = upf_zval;
    // RGaussian determines the radius of the Gaussian compensation
    // charge. 
    // TODO The correction due to the overlap of Gaussian // charges
    // should also be implemented .
    params[ptparam_.RGAUSSIAN] = etable.rgaussian(upf_symbol); 




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


    // LL: 01/12/2021 
    // 1. the size of samples used to be 
    //    ( upf_mesh_size, ptsample_.NONLOCAL + 2 * upf_nproj);
    //    where 2* is due to the storage of both the values of nonlocal
    //    pseudopotentials and the derivatives. 
    //    Currently the derivatives of the nonlocal pseudopotential is
    //    no longer explicitly stored. so the size becomes
    //    ( upf_mesh_size, ptsample_.NONLOCAL + upf_nproj);
    //
    //    In the future, we should decide whether the keep the
    //    derivative in the SparseVec structure, as well as whether to
    //    keep the derivatives for the pseudocharge etc.
    //
    // 2. This code used to support spin-orbit coupling in the HGH
    //    pseudopotentials.  However, this option is not available anyway
    //    in recent UPF files such as
    //    https://www.quantum-espresso.org/upf_files/Bi.pbe-hgh.UPF
    //
    //    So we will remove this for now, and think about how to consistently support
    //    spin-orbit coupling when needed in the future.
    //    For now we only support the scalar relativistic pseudopotential. 
    //    Compared to
    //    Hartwigsen, C., Goedecker, S., & Hutter, J. (1998). Relativistic
    //    separable dual-space Gaussian pseudopotentials from H to Rn.
    //    Physical Review B, 58(7), 3641.
    //    this means that we neglect the contribution k_ij in Eq. (19)
    samples.Resize( upf_mesh_size, ptsample_.NONLOCAL + upf_nproj);
    // weights[0] to weights[ptsample_.NONLOCAL-1] are dummies
    weights.Resize(ptsample_.NONLOCAL + upf_nproj);
    cutoffs.Resize(ptsample_.NONLOCAL + upf_nproj);
    types.Resize  (ptsample_.NONLOCAL + upf_nproj);

    SetValue( samples, 0.0);
    SetValue( weights, 0.0);
    SetValue( types, 0);
    SetValue( cutoffs, 0.0);


    // Set cutoff values from etable
    // NOTE that in ONCV, rho_cutoff is simply the largest grid point in
    // rad and does not have more information. In HGH, rho_cutoff is not
    // even provided. So do not read it here.

    cutoffs[ptsample_.VLOCAL] = etable.vsrcut(upf_symbol);
    cutoffs[ptsample_.RHOATOM] = etable.rhoatomcut(upf_symbol);
    // cutoffs[ptsample_.DRV_VLOCAL] = vsrcut;
    // cutoffs[ptsample_.DRV_RHOATOM] = rhoatomcut;


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
      samples(i, ptsample_.RADIAL_GRID) = upf_r[i];
    types[0] = 9;

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
       for( int i = 0; i < upf_mesh_size; i++){
         // 0.5 is because the unit of UPF is Ry
         upf_vloc[i] = 0.5*upf_vloc[i];
       }
       std::vector<double> r; 
       std::vector<double> vr; 
       splinerad( upf_r, upf_vloc, r, vr, 1);

       Int n = r.size();
       DblNumVec spla(n,true,&vr[0]); 
       DblNumVec splb(n), splc(n), spld(n);
       spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

       seval(samples.VecData(ptsample_.VLOCAL), upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
           splc.Data(), spld.Data());
    }

    // FIXME Magic number
    types[1] = 99;
    types[2] = 99;
    types[3] = 999;
    types[4] = 999;

    // Nonlocal pseudopotential
    {
      Real EPS = 1e-12;

      find_start_element("PP_NONLOCAL", upfin);
      DblNumMat upf_vnl(upf_mesh_size, upf_nproj);
      SetValue(upf_vnl, 0.0);
      std::vector<int> upf_proj_l(upf_nproj);

      // nonlocal potential cutoff. 
      // The value in etable provides the default value. 
      // if the value read from the UPF file is larger, will use the value 
      // from the UPF file instead
      double vnlcut = etable.vnlcut(upf_symbol);

      for ( int j = 0; j < upf_nproj; j++ )
      {
#if 0
        statusOFS << "#proj = " << j << std::endl;
#endif
        int index, angular_momentum;

        std::string element_name = "PP_BETA." + std::to_string(j+1);
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

        // define vnlcut (cutoff radius for nonlocal pseudopotential) as
        // the largest cutoff among all pseudopotentials
        buf = get_attr(tag,"cutoff_radius");
        is.clear();
        is.str(buf);
        Real cutoff_radius;
        is >> cutoff_radius;
        vnlcut = std::max(vnlcut, cutoff_radius);

#if 0
        statusOFS << " current cutoff radius for nonlocal = " << vnlcut << std::endl;
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

       
        std::vector<double> upf_vnl_read(upf_mesh_size);
        for ( int i = 0; i < upf_mesh_size; i++ )
          upfin >> upf_vnl_read[i];

        find_end_element(element_name, upfin);

        // spline for the nonlocal part.
        {
          std::vector < double > r; 
          std::vector < double > vr; 
          int parity = angular_momentum % 2;
          splinerad( upf_r, upf_vnl_read, r, vr, parity);

          for( int i = 0; i < r.size(); i++)
            vr[i] = vr[i] / r[i];

          Int n = r.size();
          DblNumVec spla(n, true, &vr[0]); 
          DblNumVec splb(n), splc(n), spld(n);
          spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

          seval(upf_vnl.VecData(j), upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
              splc.Data(), spld.Data());
        }
      } // for j


      tag = find_start_element("PP_DIJ", upfin);
      int upf_ndij;
      buf = get_attr(tag,"size");
      is.clear();
      is.str(buf);
      is >> upf_ndij;
#if 0
      statusOFS << "PP_DIJ size = " << upf_ndij << std::endl;
#endif

      if ( upf_ndij != upf_nproj*upf_nproj )
        ErrorHandling(" Number of non-zero Dij differs from number of projectors");

      std::vector<double> upf_d(upf_ndij);
      for ( int i = 0; i < upf_ndij; i++ )
      {
        upfin >> upf_d[i];
      }

      // Check if Dij has non-diagonal elements
      // non-diagonal elements are not supported
      find_end_element("PP_DIJ", upfin);

      find_end_element("PP_NONLOCAL", upfin);

      for ( int j = 0; j < upf_nproj; j++ )
      {
        types[ptsample_.NONLOCAL + j] = upf_proj_l[j];
        cutoffs[ptsample_.NONLOCAL+j] = vnlcut;
      }
      // Convert DIJ to a matrix
      DblNumMat nonlocalD(upf_nproj, upf_nproj);
      for ( int i = 0; i < upf_nproj; i++ )
        for ( int j = 0; j < upf_nproj; j++ ){
          // 0.5 is because the unit of UPF is Ry
          nonlocalD(i,j) = 0.5 * upf_d[i*upf_nproj+j];
          if( (upf_proj_l[i] != upf_proj_l[j]) and
              std::abs(nonlocalD(i,j)) > EPS ){
            ErrorHandling("DIJ cannot have off-diagonal entries for different channels.");
          }
        }


      // Post processing nonlocal pseudopotential by diagonalizing
      // nonlocalD and combine the vectors in upf_vnl. This is done by
      // diagonalizing each angular momentum block

      for ( int l = 0; l <= upf_lmax; l++ ){
        // extract the relevant index
        std::vector<Int> idx;
        for ( int j = 0; j < upf_nproj; j++ )
          if( upf_proj_l[j] == l )
            idx.push_back(j);

#if 0
        statusOFS << "for l = " << l << "idx = " << idx << std::endl;
#endif

        int nproj_l = idx.size();
        DblNumMat D_block(nproj_l, nproj_l);
        bool isDiagonal = true;
        for( int j = 0; j < nproj_l; j++ )
          for( int i = 0; i < nproj_l; i++ ){
            D_block(i,j) = nonlocalD(idx[i], idx[j]);
            if( i != j and std::abs(D_block(i,j))>EPS )
              isDiagonal = false;
          }

        if( isDiagonal ){
          // if D_block is a diagonal matrix, no need for diagonalization.
          // just copy the vectors to the right place
          for( int j = 0; j < nproj_l; j++ ){
            weights[ptsample_.NONLOCAL+idx[j]] = D_block(j,j);
            blas::Copy( upf_mesh_size, upf_vnl.VecData(idx[j]), 1, 
                samples.VecData(ptsample_.NONLOCAL+idx[j]), 1);
          }
        }
        else{
          // Diagonalize the subblock to obtain projectors
          DblNumVec weights_block(nproj_l);
          DblNumMat upf_vnl_block(upf_mesh_size, nproj_l);
          DblNumMat res_block(upf_mesh_size, nproj_l);
          for( int j = 0; j < nproj_l; j++ ){
            blas::Copy( upf_mesh_size, upf_vnl.VecData(idx[j]), 1, 
                upf_vnl_block.VecData(j), 1);
          }
          lapack::Syevd( 'V', 'U', nproj_l, D_block.Data(), nproj_l,
              weights_block.Data() );
          // Now D_block stores the eigenvectors 
          blas::Gemm( 'N', 'N', upf_mesh_size, nproj_l, nproj_l, 1.0,
              upf_vnl_block.Data(), upf_mesh_size, D_block.Data(), nproj_l,
              0.0, res_block.Data(), upf_mesh_size );
          for( int j = 0; j < nproj_l; j++ ){
            weights[ptsample_.NONLOCAL+idx[j]] = weights_block[j];
            blas::Copy( upf_mesh_size, res_block.VecData(j), 1, 
                samples.VecData(ptsample_.NONLOCAL+idx[j]), 1);
          }
        }
      }

#if 0
      statusOFS << "weights (nonlocal part) = " << 
        DblNumVec(upf_nproj, false, &weights[ptsample_.NONLOCAL]) << std::endl;
#endif 

    } // Nonlocal pseudopotential

    find_start_element("PP_RHOATOM", upfin);
    std::vector<double> upf_rho_atom(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_rho_atom[i];
 
    find_end_element("PP_RHOATOM", upfin);

    //add the spline part
    {
       std::vector < double > r; 
       std::vector < double > vr; 
       splinerad( upf_r, upf_rho_atom, r, vr, 1);
       for( int i = 0; i < r.size(); i++)
         vr[i] = vr[i] / ( 4.0 * PI * r[i] * r[i] );

       Int n = r.size();
       DblNumVec spla(n, true, &vr[0]); 
       DblNumVec splb(n), splc(n), spld(n);
       spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

       // rho_atom derivative is 0.0
       seval(samples.VecData(ptsample_.RHOATOM), upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
           splc.Data(), spld.Data());
    }
  }

  // Output information of the pseudopotential
  {
    Print( statusOFS, "zval                = ", params[ptparam_.ZVAL] );
    Print( statusOFS, "mass                = ", params[ptparam_.MASS] );
    Print( statusOFS, "RGaussian           = ", params[ptparam_.RGAUSSIAN] );
    Print( statusOFS, "vsrcut              = ", cutoffs[ptsample_.VLOCAL] );
    Print( statusOFS, "rhoatomcut          = ", cutoffs[ptsample_.RHOATOM] );
    Print( statusOFS, "vnlcut              = ", cutoffs[ptsample_.NONLOCAL] );
    Print( statusOFS, "" );
  }
}




} // namespace dgdft

