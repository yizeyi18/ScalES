#include "periodtable.hpp"
#include "parallel.hpp"
#include "serialize.hpp"


//----------------------------------------------------------------
int serialize(const Domain& val, ostream& os, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) serialize(val._Ls, os, mask);  i++;
  if(mask[i]==1) serialize(val._Ns, os, mask);  i++;
  if(mask[i]==1) serialize(val._pos, os, mask);  i++;
  iA(i==Domain_Number);
  return 0;
}

int deserialize(Domain& val, istream& is, const vector<int>& mask)
{
  int i = 0;
  if(mask[i]==1) deserialize(val._Ls, is, mask);  i++;
  if(mask[i]==1) deserialize(val._Ns, is, mask);  i++;
  if(mask[i]==1) deserialize(val._pos, is, mask);  i++;
  iA(i==Domain_Number);
  return 0;
}

int combine(Domain& val, Domain& ext)
{
  iA(0);
  return 0;
}


//----------------------------------------------------------------
int serialize(const PTEntry& val, ostream& os, const vector<int>& mask)
{
  serialize(val._params, os, mask);
  serialize(val._samples, os, mask);
  serialize(val._wgts, os, mask);
  serialize(val._typs, os, mask);
  serialize(val._cuts, os, mask);
  return 0;
}

int deserialize(PTEntry& val, istream& is, const vector<int>& mask)
{
  deserialize(val._params, is, mask);
  deserialize(val._samples, is, mask);
  deserialize(val._wgts, is, mask);
  deserialize(val._typs, is, mask);
  deserialize(val._cuts, is, mask);
  return 0;
}

int combine(PTEntry& val, PTEntry& ext)
{
  iA(0);
  return 0;
}


//---------------------------------------------
int PeriodTable::setup(string strptable)
{
  //int myid, nprocs;
  //MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  //MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  istringstream fid;  iC( Shared_Read(strptable, fid) );
  vector<int> all(1,1);
  iC( deserialize(_ptemap, fid, all) );
  //create splines
  for(map<int,PTEntry>::iterator mi=_ptemap.begin(); mi!=_ptemap.end(); mi++) {
    int type = (*mi).first;    
//    fprintf(stderr, "%d\n", type);
    PTEntry& ptcur = (*mi).second;
    DblNumVec& params = ptcur.params();
    DblNumMat& samples = ptcur.samples();
    iA(samples.n()%2 == 1); //an odd number of samples
   map< int, vector<DblNumVec> > spltmp;
    for(int g=1; g<samples.n(); g++) {
      int nspl = samples.m();
      DblNumVec rad(nspl, true, samples.clmdata(0));
      DblNumVec a(nspl, true, samples.clmdata(g));
      DblNumVec b(nspl), c(nspl), d(nspl);
      //create splines
      spline(nspl, rad.data(), a.data(), b.data(), c.data(), d.data());
      vector<DblNumVec> aux(5);
      aux[0] = rad;      aux[1] = a;      aux[2] = b;      aux[3] = c;      aux[4] = d;
      spltmp[g] = aux;
    }
    _splmap[type] = spltmp;
  }
  return 0;
}

//---------------------------------------------
//Generate the pseudo-charge and its derivatives, and saved in the
//sparse veector res
//  res[0]         : pseudo-charge values
//  res[1]--res[3] : x,y,z components of the derivatives of the
//		     pseudo-charge
int PeriodTable::pseudoRho0(Atom atom, Point3 Ls, Point3 pos, 
			    Index3 Ns,
			    SparseVec& res)
{
  if(1) {
    int type = atom.type();
    Point3 coord = atom.coord();
    
    //get entry data and spline data
    PTEntry& ptentry = _ptemap[type];
    map< int, vector<DblNumVec> >& spldata = _splmap[type];
    
    double Rzero = ptentry.cuts()(i_rho0); //CUTOFF VALUE FOR rho0
    
    DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
    double hx, hy, hz;
    double dtmp;
    hx = Ls(0) / Ns(0);
    hy = Ls(1) / Ns(1);
    hz = Ls(2) / Ns(2); 
    for(int i = 0; i < Ns(0); i++){
      dtmp = pos(0) + i * hx - coord(0);      dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);      dx(i) = dtmp;
    }
    for(int j = 0; j < Ns(1); j++){
      dtmp = pos(1) + j * hy - coord(1);      dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);      dy(j) = dtmp;
    }
    for(int k = 0; k < Ns(2); k++){
      dtmp = pos(2) + k * hz - coord(2);      dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);      dz(k) = dtmp;
    }
    int irad = 0;
    vector<int> idx;
    vector<double> rad;
    vector<double> xx, yy, zz;
    for(int k = 0; k < Ns(2); k++){
      for(int j = 0; j < Ns(1); j++){
	for(int i = 0; i < Ns(0); i++){
	  dtmp = sqrt(dx(i)*dx(i) + dy(j)*dy(j) + dz(k)*dz(k));
	  if(dtmp < Rzero) {
	    idx.push_back(irad);
	    rad.push_back(dtmp);
	    xx.push_back(dx(i));	    yy.push_back(dy(j));	    zz.push_back(dz(k));
	  }
	  irad++;
	}
      }
    }
    int idxsize = idx.size();
    double eps = 1e-8;
    //
    vector<DblNumVec>& valspl = spldata[i_rho0]; //LEXING: IMPORTANT
    vector<double> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].data(), valspl[1].data(), valspl[2].data(), valspl[3].data(), valspl[4].data());
    //
    vector<DblNumVec>& derspl = spldata[i_drho0]; //LEXING: IMPORTANT
    vector<double> der(idxsize,0.0);
    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].data(), derspl[1].data(), derspl[2].data(), derspl[3].data(), derspl[4].data());
    //
    IntNumVec iv(idx.size(), true, &(idx[0])); 
    DblNumMat dv(4, idx.size());
    //
    for(int g=0; g<idx.size(); g++) {
      dv(0,g) = val[g];
      if(rad[g]>eps) {
	dv(1,g) = der[g] * xx[g]/rad[g];
	dv(2,g) = der[g] * yy[g]/rad[g];
	dv(3,g) = der[g] * zz[g]/rad[g];
      } else {
	dv(1,g) = 0;
	dv(2,g) = 0;
	dv(3,g) = 0;
      }
    }
    res = SparseVec(iv,dv);
  }
  return 0;
}

//---------------------------------------------
int PeriodTable::pseudoNL(  Atom atom, Point3 Ls, Point3 pos, 
			    vector<DblNumVec> gridpos, 
			    vector< pair<SparseVec,double> >& vnls )
{
  vnls.clear();
  if(1) {
    int type = atom.type();
    Point3 coord = atom.coord();
    
    //get entry data and spline data
    PTEntry& ptentry = _ptemap[type];
    map< int, vector<DblNumVec> >& spldata = _splmap[type];
    
    double Rzero = 0;    if(ptentry.cuts().m()>3)      Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones
    
    //LEXING: VERY IMPRORTANT
    Index3 Ns(gridpos[0].m(),gridpos[1].m(),gridpos[2].m());
    
    double dtmp;
    DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
    double *posdata = NULL;
    posdata = gridpos[0].data();
    for(int i = 0; i < Ns(0); i++){
      dtmp = posdata[i] - coord(0);      dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);      dx(i) = dtmp;
    }
    posdata = gridpos[1].data();
    for(int j = 0; j < Ns(1); j++){
      dtmp = posdata[j] - coord(1);      dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);      dy(j) = dtmp;
    }
    posdata = gridpos[2].data();
    for(int k = 0; k < Ns(2); k++){
      dtmp = posdata[k] - coord(2);      dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);      dz(k) = dtmp;
    }
    int irad = 0;
    vector<int> idx;
    vector<double> rad;
    vector<double> xx, yy, zz;
    for(int k = 0; k < Ns(2); k++){
      for(int j = 0; j < Ns(1); j++){
	for(int i = 0; i < Ns(0); i++){
	  dtmp = sqrt(dx(i)*dx(i) + dy(j)*dy(j) + dz(k)*dz(k));
	  if( dtmp < Rzero ){
	    idx.push_back(irad);
	    rad.push_back(dtmp);
	    xx.push_back(dx(i));	    yy.push_back(dy(j));	    zz.push_back(dz(k));
	  }
	  irad++;
	}
      }
    }
    int idxsize = idx.size();
    //
    double eps = 1e-8;
    //process non-local pseudopotential one by one
    for(int g=3; g<ptentry.samples().n(); g=g+2) {
      double wgt = ptentry.wgts()(g);
      int typ = ptentry.typs()(g);
      // iA( abs(wgt)>eps );  LL: IMPORTANT: wgt might be zero if h_11
      // or h_22 is 0 (say for C) in the table.
      //
      vector<DblNumVec>& valspl = spldata[g]; //LEXING: IMPORTANT
      vector<double> val(idxsize,0.0);
      seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].data(), valspl[1].data(), valspl[2].data(), valspl[3].data(), valspl[4].data());
      //
      vector<DblNumVec>& derspl = spldata[g+1]; //LEXING: IMPORTANT
      vector<double> der(idxsize,0.0);
      seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].data(), derspl[1].data(), derspl[2].data(), derspl[3].data(), derspl[4].data());
      //--
      if(typ==0) {
	double coef = sqrt(1.0/(4.0*M_PI)); //spherical harmonics
	IntNumVec iv(idx.size(), true, &(idx[0]));
	DblNumMat dv(4, idx.size());
	//
	for(int g=0; g<idx.size(); g++) {
	  if(rad[g]>eps) {
	    dv(0,g) = coef * val[g];
	    dv(1,g) = coef * der[g] * xx[g]/rad[g];
	    dv(2,g) = coef * der[g] * yy[g]/rad[g];
	    dv(3,g) = coef * der[g] * zz[g]/rad[g];
	  } else {
	    dv(0,g) = coef * val[g];
	    dv(1,g) = 0;
	    dv(2,g) = 0;
	    dv(3,g) = 0;
	  }
	}
	SparseVec res(iv,dv);
	vnls.push_back( pair<SparseVec,double>(res,wgt) );
      } // if(typ == 0);
      //--
      if(typ==1) {
	double coef = sqrt(3.0/(4.0*M_PI)); //spherical harmonics
	{
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      dv(0,g) = coef*( (xx[g]/rad[g]) * val[g] );
	      dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(xx[g]/rad[g]) + val[g]/rad[g] );
	      dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(yy[g]/rad[g])                 );
	      dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(zz[g]/rad[g])                 );
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = coef*der[g];
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	{
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      dv(0,g) = coef*( (yy[g]/rad[g]) * val[g] );
	      dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(xx[g]/rad[g])                 );
	      dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(yy[g]/rad[g]) + val[g]/rad[g] );
	      dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(zz[g]/rad[g])                 );
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = coef*der[g];
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	{
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      dv(0,g) = coef*( (zz[g]/rad[g]) * val[g] );
	      dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(xx[g]/rad[g])                 );
	      dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(yy[g]/rad[g])                 );
	      dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(zz[g]/rad[g]) + val[g]/rad[g] );
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = coef*der[g];
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
      } // if(typ==1)

      if(typ==2) {
	// d_z2
	{
	  double coef = 1.0/4.0*sqrt(5.0/M_PI); // Coefficients for spherical harmonics
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      Ylm(0) = coef*(-xx[g]*xx[g]-yy[g]*yy[g]+2.0*zz[g]*zz[g]) / (rad[g]*rad[g]);
	      Ylm(1) = coef*(-6.0 * xx[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
	      Ylm(2) = coef*(-6.0 * yy[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
	      Ylm(3) = coef*( 6.0 * zz[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)) / pow(rad[g], 4.0));

	      dv(0,g) = Ylm(0) * val[g] ;
	      dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
	      dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
	      dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	// d_yz
	{
	  double coef = 1.0/2.0*sqrt(15.0/M_PI);
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());

	  DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      Ylm(0) = coef*(yy[g]*zz[g]) / (rad[g]*rad[g]);
	      Ylm(1) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
	      Ylm(2) = coef*(     zz[g]*(pow(zz[g],2.0)+pow(xx[g],2.0)-pow(yy[g],2.0)) / 
			     pow(rad[g],4.0));
	      Ylm(3) = coef*(     yy[g]*(pow(yy[g],2.0)+pow(xx[g],2.0)-pow(zz[g],2.0)) /
			     pow(rad[g],4.0));

	      dv(0,g) = Ylm(0) * val[g] ;
	      dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
	      dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
	      dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	// d_xz
	{
	  double coef = 1.0/2.0*sqrt(15.0/M_PI);
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  
	  DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      Ylm(0) = coef*(zz[g]*xx[g]) / (rad[g]*rad[g]);
	      Ylm(1) = coef*(     zz[g]*(pow(zz[g],2.0)-pow(xx[g],2.0)+pow(yy[g],2.0)) / 
			     pow(rad[g],4.0));
	      Ylm(2) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
	      Ylm(3) = coef*(     xx[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)-pow(zz[g],2.0)) /
			     pow(rad[g],4.0));
	    
	      dv(0,g) = Ylm(0) * val[g] ;
	      dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
	      dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
	      dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
	    
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	// d_xy
	{
	  double coef = 1.0/2.0*sqrt(15.0/M_PI);
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      Ylm(0) = coef*(xx[g]*yy[g]) / (rad[g]*rad[g]);
	      Ylm(1) = coef*(     yy[g]*(pow(yy[g],2.0)-pow(xx[g],2.0)+pow(zz[g],2.0)) / 
			     pow(rad[g],4.0));
	      Ylm(2) = coef*(     xx[g]*(pow(xx[g],2.0)-pow(yy[g],2.0)+pow(zz[g],2.0)) /
			     pow(rad[g],4.0));
	      Ylm(3) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
	    
	      dv(0,g) = Ylm(0) * val[g] ;
	      dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
	      dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
	      dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
	// d_x^2-y^2
	{
	  double coef = 1.0/4.0*sqrt(15.0/M_PI);
	  IntNumVec iv(idx.size(), true, &(idx[0]));
	  DblNumMat dv(4, idx.size());
	  DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
	  for(int g=0; g<idx.size(); g++) {
	    if(rad[g]>eps) {
	      Ylm(0) = coef*(xx[g]*xx[g]-yy[g]*yy[g]) / (rad[g]*rad[g]);
	      Ylm(1) = coef*( 2.0*xx[g]*(2.0*pow(yy[g],2.0)+pow(zz[g],2.0)) / 
			     pow(rad[g],4.0));
	      Ylm(2) = coef*(-2.0*yy[g]*(2.0*pow(xx[g],2.0)+pow(zz[g],2.0)) /
			     pow(rad[g],4.0));
	      Ylm(3) = coef*(-2.0*zz[g]*(pow(xx[g],2.0) - pow(yy[g],2.0)) / pow(rad[g],4.0));
	    
	      dv(0,g) = Ylm(0) * val[g] ;
	      dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
	      dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
	      dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
	    } else {
	      dv(0,g) = 0;
	      dv(1,g) = 0;
	      dv(2,g) = 0;
	      dv(3,g) = 0;
	    }
	  }
	  SparseVec res(iv,dv);
	  vnls.push_back( pair<SparseVec,double>(res,wgt) );
	}
      } // if(typ==2)
    }
  }
  return 0;
}


//---------------------------------------------
int PeriodTable::pseudoNL(  Atom atom, Point3 Ls, Point3 pos, 
			    NumTns< vector<DblNumVec> > gridpostns,
			    vector< pair<NumTns<SparseVec>,double> >& vnls)
{
  vnls.clear();
  if(1) {
    int type = atom.type();
    Point3 coord = atom.coord();
    
    //get entry data and spline data
    PTEntry& ptentry = _ptemap[type];
    map< int, vector<DblNumVec> >& spldata = _splmap[type];

    //double Rzero = ptentry.params()(i_cutoff);
    double Rzero = 0;    if(ptentry.cuts().m()>3)      Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones
    //
    double eps = 1e-8;
    
    int Gm = gridpostns.m();
    int Gn = gridpostns.n();
    int Gp = gridpostns.p();
    
    int numpp = 0;
    for(int g=3; g<ptentry.samples().n(); g=g+2) {
      double wgt = ptentry.wgts()(g);
      int typ = ptentry.typs()(g);
//      int mpirank;
//      MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
//      if(mpirank == 0)
//	cerr << "typ = " << typ << endl;
      
      if(typ==0)
	numpp=numpp+1;
      if(typ==1)
	numpp=numpp+3;
      if(typ==2)
	numpp=numpp+5;
    }
    vector< NumTns<SparseVec> > pptmp(numpp);
    for(int a=0; a<numpp; a++)
      pptmp[a].resize(Gm,Gn,Gp);
    
    for(int gi=0; gi<Gm; gi++)
      for(int gj=0; gj<Gn; gj++)
	for(int gk=0; gk<Gp; gk++) {
	  vector<DblNumVec>& gridpos = gridpostns(gi,gj,gk);
	  //
	  Index3 Ns(gridpos[0].m(), gridpos[1].m(), gridpos[2].m());
	  double dtmp;
	  DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
	  double *posdata=NULL;
	  posdata = gridpos[0].data();
	  for(int i = 0; i < Ns(0); i++){
	    dtmp = posdata[i] - coord(0);	    dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);	    dx(i) = dtmp;
	  }
	  posdata = gridpos[1].data();
	  for(int j = 0; j < Ns(1); j++){
	    dtmp = posdata[j] - coord(1);	    dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);	    dy(j) = dtmp;
	  }
	  posdata = gridpos[2].data();
	  for(int k = 0; k < Ns(2); k++){
	    dtmp = posdata[k] - coord(2);	    dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);	    dz(k) = dtmp;
	  }
	  int irad = 0;
	  vector<int> idx;
	  vector<double> rad;
	  vector<double> xx, yy, zz;
	  for(int k = 0; k < Ns(2); k++){
	    for(int j = 0; j < Ns(1); j++){
	      for(int i = 0; i < Ns(0); i++){
		dtmp = sqrt(dx(i)*dx(i) + dy(j)*dy(j) + dz(k)*dz(k));
		if( dtmp < Rzero ){
		  idx.push_back(irad);
		  rad.push_back(dtmp);
		  xx.push_back(dx(i));	    yy.push_back(dy(j));	    zz.push_back(dz(k));
		}
		irad++;
	      }
	    }
	  }
	  int idxsize = idx.size();
	  //
	  int cntpp = 0;
	  for(int g=3; g<ptentry.samples().n(); g=g+2) {
	    double wgt = ptentry.wgts()(g);
	    int typ = ptentry.typs()(g);
	    // iA( abs(wgt)>eps );  LL: IMPORTANT: wgt might be zero if h_11
	    // or h_22 is 0 (say for C) in the table.
	    //
	    //
	    vector<DblNumVec>& valspl = spldata[g]; //LEXING: IMPORTANT
	    vector<double> val(idxsize,0.0);
	    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].data(), valspl[1].data(), valspl[2].data(), valspl[3].data(), valspl[4].data());
	    //
	    vector<DblNumVec>& derspl = spldata[g+1]; //LEXING: IMPORTANT
	    vector<double> der(idxsize,0.0);
	    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].data(), derspl[1].data(), derspl[2].data(), derspl[3].data(), derspl[4].data());
	    //
	    if(typ==0) {
	      double coef = sqrt(1.0/(4.0*M_PI)); //spherical harmonics
	      IntNumVec iv(idx.size(), true, &(idx[0]));
	      DblNumMat dv(4, idx.size());
	      //
	      for(int g=0; g<idx.size(); g++) {
		if(rad[g]>eps) {
		  dv(0,g) = coef * val[g];
		  dv(1,g) = coef * der[g] * xx[g]/rad[g];
		  dv(2,g) = coef * der[g] * yy[g]/rad[g];
		  dv(3,g) = coef * der[g] * zz[g]/rad[g];
		} else {
		  dv(0,g) = coef * val[g];
		  dv(1,g) = 0;
		  dv(2,g) = 0;
		  dv(3,g) = 0;
		}
	      }
	      SparseVec res(iv,dv);
	      pptmp[cntpp](gi,gj,gk) = res;
	      cntpp++;
	    } //if(typ==0)
	    //-------
	    if(typ==1) {
	      double coef = sqrt(3.0/(4.0*M_PI)); //spherical harmonics
	      {
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    dv(0,g) = coef*( (xx[g]/rad[g]) * val[g] );
		    dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(xx[g]/rad[g]) + val[g]/rad[g] );
		    dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(yy[g]/rad[g])                 );
		    dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(zz[g]/rad[g])                 );
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = coef*der[g];
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      {
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    dv(0,g) = coef*( (yy[g]/rad[g]) * val[g] );
		    dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(xx[g]/rad[g])                 );
		    dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(yy[g]/rad[g]) + val[g]/rad[g] );
		    dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(zz[g]/rad[g])                 );
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = coef*der[g];
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      {
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    dv(0,g) = coef*( (zz[g]/rad[g]) * val[g] );
		    dv(1,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(xx[g]/rad[g])                 );
		    dv(2,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(yy[g]/rad[g])                 );
		    dv(3,g) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(zz[g]/rad[g]) + val[g]/rad[g] );
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = coef*der[g];
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	    } //if(typ==1)
	    
	  
	    if(typ==2) {
	      // d_z2
	      {
		double coef = 1.0/4.0*sqrt(5.0/M_PI); // Coefficients for spherical harmonics
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    Ylm(0) = coef*(-xx[g]*xx[g]-yy[g]*yy[g]+2.0*zz[g]*zz[g]) / (rad[g]*rad[g]);
		    Ylm(1) = coef*(-6.0 * xx[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
		    Ylm(2) = coef*(-6.0 * yy[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
		    Ylm(3) = coef*( 6.0 * zz[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)) / pow(rad[g], 4.0));

		    dv(0,g) = Ylm(0) * val[g] ;
		    dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
		    dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
		    dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      // d_yz
	      {
		double coef = 1.0/2.0*sqrt(15.0/M_PI);
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());

		DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    Ylm(0) = coef*(yy[g]*zz[g]) / (rad[g]*rad[g]);
		    Ylm(1) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
		    Ylm(2) = coef*(     zz[g]*(pow(zz[g],2.0)+pow(xx[g],2.0)-pow(yy[g],2.0)) / 
					pow(rad[g],4.0));
		    Ylm(3) = coef*(     yy[g]*(pow(yy[g],2.0)+pow(xx[g],2.0)-pow(zz[g],2.0)) /
					pow(rad[g],4.0));

		    dv(0,g) = Ylm(0) * val[g] ;
		    dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
		    dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
		    dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      // d_xz
	      {
		double coef = 1.0/2.0*sqrt(15.0/M_PI);
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());

		DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    Ylm(0) = coef*(zz[g]*xx[g]) / (rad[g]*rad[g]);
		    Ylm(1) = coef*(     zz[g]*(pow(zz[g],2.0)-pow(xx[g],2.0)+pow(yy[g],2.0)) / 
					pow(rad[g],4.0));
		    Ylm(2) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
		    Ylm(3) = coef*(     xx[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)-pow(zz[g],2.0)) /
					pow(rad[g],4.0));

		    dv(0,g) = Ylm(0) * val[g] ;
		    dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
		    dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
		    dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];

		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      // d_xy
	      {
		double coef = 1.0/2.0*sqrt(15.0/M_PI);
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    Ylm(0) = coef*(xx[g]*yy[g]) / (rad[g]*rad[g]);
		    Ylm(1) = coef*(     yy[g]*(pow(yy[g],2.0)-pow(xx[g],2.0)+pow(zz[g],2.0)) / 
					pow(rad[g],4.0));
		    Ylm(2) = coef*(     xx[g]*(pow(xx[g],2.0)-pow(yy[g],2.0)+pow(zz[g],2.0)) /
					pow(rad[g],4.0));
		    Ylm(3) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));

		    dv(0,g) = Ylm(0) * val[g] ;
		    dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
		    dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
		    dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	      // d_x^2-y^2
	      {
		double coef = 1.0/4.0*sqrt(15.0/M_PI);
		IntNumVec iv(idx.size(), true, &(idx[0]));
		DblNumMat dv(4, idx.size());
		DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
		for(int g=0; g<idx.size(); g++) {
		  if(rad[g]>eps) {
		    Ylm(0) = coef*(xx[g]*xx[g]-yy[g]*yy[g]) / (rad[g]*rad[g]);
		    Ylm(1) = coef*( 2.0*xx[g]*(2.0*pow(yy[g],2.0)+pow(zz[g],2.0)) / 
				    pow(rad[g],4.0));
		    Ylm(2) = coef*(-2.0*yy[g]*(2.0*pow(xx[g],2.0)+pow(zz[g],2.0)) /
				   pow(rad[g],4.0));
		    Ylm(3) = coef*(-2.0*zz[g]*(pow(xx[g],2.0) - pow(yy[g],2.0)) / pow(rad[g],4.0));

		    dv(0,g) = Ylm(0) * val[g] ;
		    dv(1,g) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
		    dv(2,g) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
		    dv(3,g) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
		  } else {
		    dv(0,g) = 0;
		    dv(1,g) = 0;
		    dv(2,g) = 0;
		    dv(3,g) = 0;
		  }
		}
		SparseVec res(iv,dv);
		pptmp[cntpp](gi,gj,gk) = res;
		cntpp++;
	      }
	    } // if(typ==2)


	  }// for(g)
	  cerr << cntpp << ", " << numpp  << endl;
	  iA(cntpp==numpp);
	} //for(gk)
    
    //
    int cntpp = 0;
    for(int g=3; g<ptentry.samples().n(); g=g+2) {
      double wgt = ptentry.wgts()(g);
      int typ = ptentry.typs()(g);
      if(typ==0) {
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
      }
      if(typ==1) {
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
      }
      if(typ==2) {
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
	vnls.push_back( pair<NumTns<SparseVec>,double>(pptmp[cntpp], wgt) );	cntpp++;
      }
    }
    iA(cntpp==numpp);
  }
  return 0;
}

