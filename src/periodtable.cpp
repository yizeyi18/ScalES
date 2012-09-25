#include "periodtable.hpp"

// TODO Change the names of variables
// TODO Add the error handling stack
// TODO Change the argument of PseoduCharge etc into domain

namespace  dgdft{

// *********************************************************************
// PTEntry
// *********************************************************************

Int serialize(const PTEntry& val, std::ostream& os, const std::vector<Int>& mask)
{
	serialize(val._params, os, mask);
	serialize(val._samples, os, mask);
	serialize(val._wgts, os, mask);
	serialize(val._typs, os, mask);
	serialize(val._cuts, os, mask);
	return 0;
}

Int deserialize(PTEntry& val, std::istream& is, const std::vector<Int>& mask)
{
	deserialize(val._params, is, mask);
	deserialize(val._samples, is, mask);
	deserialize(val._wgts, is, mask);
	deserialize(val._typs, is, mask);
	deserialize(val._cuts, is, mask);
	return 0;
}

Int combine(PTEntry& val, PTEntry& ext)
{
	throw  std::logic_error( "Combine operation is not implemented" );
	return 0;
}


// *********************************************************************
// PeriodTable
// *********************************************************************

void PeriodTable::Setup( const std::string strptable )
{
#ifndef _RELEASE_
	PushCallStack("PeriodTable::Setup");
#endif
	std::vector<Int> all(1,1);

	std::istringstream iss;  
	SharedRead( strptable, iss );
	deserialize(_ptemap, iss, all);

	//create splines
	for(std::map<Int,PTEntry>::iterator mi=_ptemap.begin(); mi!=_ptemap.end(); mi++) {
		Int type = (*mi).first;    
		PTEntry& ptcur = (*mi).second;
		DblNumVec& params = ptcur.params();
		DblNumMat& samples = ptcur.samples();
		if( samples.n() % 2 == 0 ){
			throw std::logic_error( "Wrong number of samples" );
		}
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
		_splmap[type] = spltmp;
	}
#ifndef _RELEASE_
	PopCallStack();
#endif
} 		// -----  end of method PeriodTable::Setup  ----- 


void
PeriodTable::CalculatePseudoCharge	(
		const Atom& atom, 
		const Domain& dm,
		SparseVec& res )
{
#ifndef _RELEASE_
	PushCallStack("PeriodTable::CalculatePseudoCharge");
#endif
	Int type   = atom.type;
	Point3 pos = atom.pos;
	Point3 Ls  = dm.length;
	Point3 posStart = dm.posStart;
	Index3 Ns  = dm.numGrid;

	//get entry data and spline data
	PTEntry& ptentry = _ptemap[type];
	std::map< Int, std::vector<DblNumVec> >& spldata = _splmap[type];

	Real Rzero = ptentry.cuts()(i_rho0); //CUTOFF VALUE FOR rho0

	DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
	Real hx, hy, hz;
	Real dtmp;
	hx = Ls(0) / Ns(0);
	hy = Ls(1) / Ns(1);
	hz = Ls(2) / Ns(2); 
	for(Int i = 0; i < Ns(0); i++){
		dtmp = posStart(0) + i * hx - pos(0); // move the center to atom      
		dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);      
		dx(i) = dtmp;
	}
	for(Int j = 0; j < Ns(1); j++){
		dtmp = posStart(1) + j * hy - pos(1);      
		dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);      
		dy(j) = dtmp;
	}
	for(Int k = 0; k < Ns(2); k++){
		dtmp = posStart(2) + k * hz - pos(2);      
		dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);      
		dz(k) = dtmp;
	}
	Int irad = 0;
	std::vector<Int>  idx;
	std::vector<Real> rad;
	std::vector<Real> xx, yy, zz;
	for(Int k = 0; k < Ns(2); k++){
		for(Int j = 0; j < Ns(1); j++){
			for(Int i = 0; i < Ns(0); i++){
				dtmp = sqrt(dx(i)*dx(i) + dy(j)*dy(j) + dz(k)*dz(k));
				if(dtmp < Rzero) {
					idx.push_back(irad);
					rad.push_back(dtmp);
					xx.push_back(dx(i));	    
					yy.push_back(dy(j));	    
					zz.push_back(dz(k));
				}
				irad++;
			}
		}
	}
	Int idxsize = idx.size();
	// FIXME magic number here
	Real eps = 1e-8;
	//
	std::vector<DblNumVec>& valspl = spldata[i_rho0]; 
	std::vector<Real> val(idxsize,0.0);
	seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
			valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
	//
	std::vector<DblNumVec>& derspl = spldata[i_drho0];
	std::vector<Real> der(idxsize,0.0);
	
	
	seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
			derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
	//
	IntNumVec iv(idx.size(), true, &(idx[0])); 
	DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
	//
	for(Int g=0; g<idx.size(); g++) {
		dv(g, VAL) = val[g];
		if(rad[g]>eps) {
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

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method PeriodTable::CalculatePseudoCharge  ----- 

//---------------------------------------------

void
PeriodTable::CalculateNonlocalPP	( const Atom& atom, 
		const Domain& dm, 
		const std::vector<DblNumVec>& gridpos,
		std::vector<NonlocalPP>& vnlList )
{
#ifndef _RELEASE_
	PushCallStack("PeriodTable::CalculateNonlocalPP");
#endif
	Point3 Ls  = dm.length;
	Point3 posStart = dm.posStart;
	Index3 Ns  = dm.numGrid;
	vnlList.clear();

#ifndef _NO_NONLOCAL_ // Nonlocal potential is used
	Int type = atom.type;
	Point3 pos = atom.pos;

	//get entry data and spline data
	PTEntry& ptentry = _ptemap[type];
	std::map< Int, std::vector<DblNumVec> >& spldata = _splmap[type];

	Real Rzero = 0;    if(ptentry.cuts().m()>3)      Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones


	Real dtmp;
	DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
	Real *posdata = NULL;
	posdata = gridpos[0].Data();
	for(Int i = 0; i < Ns(0); i++){
		dtmp = posdata[i] - pos(0);      dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);      dx(i) = dtmp;
	}
	posdata = gridpos[1].Data();
	for(Int j = 0; j < Ns(1); j++){
		dtmp = posdata[j] - pos(1);      dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);      dy(j) = dtmp;
	}
	posdata = gridpos[2].Data();
	for(Int k = 0; k < Ns(2); k++){
		dtmp = posdata[k] - pos(2);      dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);      dz(k) = dtmp;
	}
	Int irad = 0;
	std::vector<Int> idx;
	std::vector<Real> rad;
	std::vector<Real> xx, yy, zz;
	for(Int k = 0; k < Ns(2); k++){
		for(Int j = 0; j < Ns(1); j++){
			for(Int i = 0; i < Ns(0); i++){
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
	Int idxsize = idx.size();
	// FIXME the magic number of eps
	Real eps = 1e-8;
	//process non-local pseudopotential one by one
	for(Int g=3; g<ptentry.samples().n(); g=g+2) {
		Real wgt = ptentry.wgts()(g);
		Int typ = ptentry.typs()(g);
		// iA( abs(wgt)>eps );  LL: IMPORTANT: wgt might be zero if h_11
		// or h_22 is 0 (say for C) in the table.
		//
		std::vector<DblNumVec>& valspl = spldata[g]; 
		std::vector<Real> val(idxsize,0.0);
		seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
		//
		std::vector<DblNumVec>& derspl = spldata[g+1]; 
		std::vector<Real> der(idxsize,0.0);
		seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
		//--
		if(typ==0) {
			Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
			IntNumVec iv(idx.size(), true, &(idx[0]));
			DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
			//
			for(Int g=0; g<idx.size(); g++) {
				if(rad[g]>eps) {
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
			SparseVec res(iv,dv);
			vnlList.push_back( NonlocalPP(res,wgt) );
		} // if(typ == 0);
		//--
		if(typ==1) {
			Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
		} // if(typ==1)

		if(typ==2) {
			// d_z2
			{
				Real coef = 1.0/4.0*sqrt(5.0/PI); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// d_yz
			{
				Real coef = 1.0/2.0*sqrt(15.0/PI);
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());

				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// d_xz
			{
				Real coef = 1.0/2.0*sqrt(15.0/PI);
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());

				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// d_xy
			{
				Real coef = 1.0/2.0*sqrt(15.0/PI);
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// d_x^2-y^2
			{
				Real coef = 1.0/4.0*sqrt(15.0/PI);
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
		} // if(typ==2)

		if(typ==3) {
			// f_z3
			{
				Real coef = 1.0/4.0*sqrt(7.0/PI); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_y(3xx-yy)
			{
				Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_x(xx-3yy)
			{
				Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_z(xx-yy)
			{
				Real coef = 1.0/4.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_xyz
			{
				Real coef = 1.0/2.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_yzz
			{
				Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
			// f_xzz
			{
				Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				DblNumVec Ylm(4); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
				for(Int g=0; g<idx.size(); g++) {
					if(rad[g]>eps) {
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
				SparseVec res(iv,dv);
				vnlList.push_back( NonlocalPP(res,wgt) );
			}
		} // if(typ==3)
	}
#endif // #ifndef _NO_NONLOCAL_

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method PeriodTable::CalculateNonlocalPP  ----- 


//---------------------------------------------
// TODO SpinOrbit from RelDFT


//---------------------------------------------
// TODO: DG pseudopotential from DGDFT

} // namespace dgdft
