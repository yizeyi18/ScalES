#include "periodtable.hpp"

// TODO Change the names of variables
// TODO Add the error handling stack
// TODO Define a struct for pseudoPotential type rather than
//   pair<SparseVec, double> or std::pair<SparseVec, double>
//   Vec?  Fix the nonlocal pseudopotential later by a new struct
// TODO Error handling when the names of the subroutines are fixed
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

Int PeriodTable::Setup( const std::string strptable )
{
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
			DblNumVec rad(nspl, true, samples.ColData(0));
			DblNumVec a(nspl, true, samples.ColData(g));
			DblNumVec b(nspl), c(nspl), d(nspl);
			//create splines
			spline(nspl, rad.Data(), a.Data(), b.Data(), c.Data(), d.Data());
			std::vector<DblNumVec> aux(5);
			aux[0] = rad;      aux[1] = a;      aux[2] = b;      aux[3] = c;      aux[4] = d;
			spltmp[g] = aux;
		}
		_splmap[type] = spltmp;
	}
	return 0;
}

Int PeriodTable::SetPseudoCharge( 
		const Atom& atom, 
		const Point3 Ls, 
		const Point3 posStart, 
		const Index3 Ns, 
		SparseVec& res)
{
	Int type   = atom.type;
	Point3 pos = atom.pos;

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
	std::vector<Int> idx;
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
	std::vector<DblNumVec>& valspl = spldata[i_rho0]; //LEXING: IMPORTANT
	std::vector<Real> val(idxsize,0.0);
	seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
			valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
	//
	std::vector<DblNumVec>& derspl = spldata[i_drho0]; //LEXING: IMPORTANT
	std::vector<Real> der(idxsize,0.0);
	
	
	seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
			derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
	//
	IntNumVec iv(idx.size(), true, &(idx[0])); 
	DblNumMat dv(4, idx.size());
	//
	for(Int g=0; g<idx.size(); g++) {
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
	return 0;
}

//---------------------------------------------
Int PeriodTable::SetNonlocalPseudoPotential(  
		const Atom& atom, 
		const Point3 Ls, 
		const Point3 posStart, 
		const std::vector<DblNumVec>& gridpos, 
		std::vector< std::pair<SparseVec,Real> >& vnls )
{
	vnls.clear();

#ifdef _NO_NONLOCAL_
	Int type = atom.type;
	Point3 pos = atom.pos;

	//get entry data and spline data
	PTEntry& ptentry = _ptemap[type];
	std::map< Int, std::vector<DblNumVec> >& spldata = _splmap[type];

	Real Rzero = 0;    if(ptentry.cuts().m()>3)      Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones

	//LEXING: VERY IMPRORTANT
	Index3 Ns(gridpos[0].m(),gridpos[1].m(),gridpos[2].m());

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
		std::vector<DblNumVec>& valspl = spldata[g]; //LEXING: IMPORTANT
		std::vector<Real> val(idxsize,0.0);
		seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
		//
		std::vector<DblNumVec>& derspl = spldata[g+1]; //LEXING: IMPORTANT
		std::vector<Real> der(idxsize,0.0);
		seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
		//--
		if(typ==0) {
			Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
			IntNumVec iv(idx.size(), true, &(idx[0]));
			DblNumMat dv(4, idx.size());
			//
			for(Int g=0; g<idx.size(); g++) {
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
			vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
		} // if(typ == 0);
		//--
		if(typ==1) {
			Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
			}
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
			}
			{
				IntNumVec iv(idx.size(), true, &(idx[0]));
				DblNumMat dv(4, idx.size());
				for(Int g=0; g<idx.size(); g++) {
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
				vnls.push_back( std::pair<SparseVec,Real>(res,wgt) );
			}
		} // if(typ==3)
	}
#endif // #ifndef _NO_NONLOCAL_
	return 0;
}



//---------------------------------------------
// TODO SpinOrbit
/*
Int PeriodTable::pseudoNLSpinOrbit(  Atom atom, Point3 Ls, Point3 pos, 
		std::vector<DblNumVec> gridpos, 
		std::vector< std::pair<SparseVec,Real> >& vnlSO )
{
	vnlSO.clear();
	if(1) {
		Int type = atom.type;
		Point3 pos = atom.pos;

		//get entry data and spline data
		PTEntry& ptentry = _ptemap[type];
		std::map< Int, std::vector<DblNumVec> >& spldata = _splmap[type];

		Real Rzero = 0;    
		if(ptentry.cuts().m()>3)      
			Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones

		//LEXING: VERY IMPRORTANT
		Index3 Ns(gridpos[0].m(),gridpos[1].m(),gridpos[2].m());

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
		//
		Real eps = 1e-8;
		//process non-local pseudopotential one by one
		for(Int g=3; g<ptentry.samples().n(); g=g+2) {
			Real wgt = ptentry.wgts()(g);
			Int typ = ptentry.typs()(g);
			// iA( abs(wgt)>eps );  LL: IMPORTANT: wgt might be zero if h_11
			// or h_22 is 0 (say for C) in the table.
			//
			std::vector<DblNumVec>& valspl = spldata[g]; //LEXING: IMPORTANT
			std::vector<Real> val(idxsize,0.0);
			seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
			//
			std::vector<DblNumVec>& derspl = spldata[g+1]; //LEXING: IMPORTANT
			std::vector<Real> der(idxsize,0.0);
			seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
			// Spin orbit coupling term with l=1
			if(typ==-1) {
				Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
				{
					IntNumVec iv(idx.size(), true, &(idx[0]));
					DblNumMat dv(4, idx.size());
					for(Int g=0; g<idx.size(); g++) {
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
				}
				{
					IntNumVec iv(idx.size(), true, &(idx[0]));
					DblNumMat dv(4, idx.size());
					for(Int g=0; g<idx.size(); g++) {
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
				}
				{
					IntNumVec iv(idx.size(), true, &(idx[0]));
					DblNumMat dv(4, idx.size());
					for(Int g=0; g<idx.size(); g++) {
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
				}
			} // if(typ==-1)

			// Spin orbit coupling term with l=2
			if(typ==-2) {
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
				}
			} // if(typ==-2)
			if(typ==-3) {
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
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
					vnlSO.push_back( std::pair<SparseVec,Real>(res,wgt) );
				}
			} // if(typ==-3)


		}
	}
	return 0;
}
*/


//---------------------------------------------
// TODO: DG pseudopotential
/*
Int PeriodTable::pseudoNL(  Atom atom, Point3 Ls, Point3 pos, 
		NumTns< std::vector<DblNumVec> > gridpostns,
		std::vector< std::pair<NumTns<SparseVec>,Real> >& vnls)
	// Not updated for SO yet
{
	vnls.clear();
	if(1) {
		Int type = atom.type;
		Point3 pos = atom.pos;

		//get entry data and spline data
		PTEntry& ptentry = _ptemap[type];
		std::map< Int, std::vector<DblNumVec> >& spldata = _splmap[type];

		//Real Rzero = ptentry.params()(i_cutoff);
		Real Rzero = 0;    if(ptentry.cuts().m()>3)      Rzero = ptentry.cuts()(3); //CUTOFF VALUE FOR nonlocal ones
		//
		Real eps = 1e-8;

		Int Gm = gridpostns.m();
		Int Gn = gridpostns.n();
		Int Gp = gridpostns.p();

		Int numpp = 0;
		for(Int g=3; g<ptentry.samples().n(); g=g+2) {
			Real wgt = ptentry.wgts()(g);
			Int typ = ptentry.typs()(g);
			if(typ==0)
				numpp=numpp+1;
			if(typ==1)
				numpp=numpp+3;
			if(typ==2)
				numpp=numpp+5;
		}
		std::vector< NumTns<SparseVec> > pptmp(numpp);
		for(Int a=0; a<numpp; a++)
			pptmp[a].resize(Gm,Gn,Gp);

		for(Int gi=0; gi<Gm; gi++)
			for(Int gj=0; gj<Gn; gj++)
				for(Int gk=0; gk<Gp; gk++) {
					std::vector<DblNumVec>& gridpos = gridpostns(gi,gj,gk);
					//
					Index3 Ns(gridpos[0].m(), gridpos[1].m(), gridpos[2].m());
					Real dtmp;
					DblNumVec dx(Ns(0)), dy(Ns(1)), dz(Ns(2));
					Real *posdata=NULL;
					posdata = gridpos[0].Data();
					for(Int i = 0; i < Ns(0); i++){
						dtmp = posdata[i] - pos(0);	    dtmp = dtmp - iround(dtmp/Ls(0))*Ls(0);	    dx(i) = dtmp;
					}
					posdata = gridpos[1].Data();
					for(Int j = 0; j < Ns(1); j++){
						dtmp = posdata[j] - pos(1);	    dtmp = dtmp - iround(dtmp/Ls(1))*Ls(1);	    dy(j) = dtmp;
					}
					posdata = gridpos[2].Data();
					for(Int k = 0; k < Ns(2); k++){
						dtmp = posdata[k] - pos(2);	    dtmp = dtmp - iround(dtmp/Ls(2))*Ls(2);	    dz(k) = dtmp;
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
					//
					Int cntpp = 0;
					for(Int g=3; g<ptentry.samples().n(); g=g+2) {
						Real wgt = ptentry.wgts()(g);
						Int typ = ptentry.typs()(g);
						// iA( abs(wgt)>eps );  LL: IMPORTANT: wgt might be zero if h_11
						// or h_22 is 0 (say for C) in the table.
						//
						//
						std::vector<DblNumVec>& valspl = spldata[g]; //LEXING: IMPORTANT
						std::vector<Real> val(idxsize,0.0);
						seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
						//
						std::vector<DblNumVec>& derspl = spldata[g+1]; //LEXING: IMPORTANT
						std::vector<Real> der(idxsize,0.0);
						seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
						//
						if(typ==0) {
							Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
							IntNumVec iv(idx.size(), true, &(idx[0]));
							DblNumMat dv(4, idx.size());
							//
							for(Int g=0; g<idx.size(); g++) {
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
							Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
							{
								IntNumVec iv(idx.size(), true, &(idx[0]));
								DblNumMat dv(4, idx.size());
								for(Int g=0; g<idx.size(); g++) {
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
								for(Int g=0; g<idx.size(); g++) {
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
								for(Int g=0; g<idx.size(); g++) {
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
						}//if(typ==1)


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

					}
					cerr << cntpp << ", " << numpp  << endl;
					iA(cntpp==numpp);
				}

		//
		Int cntpp = 0;
		for(Int g=3; g<ptentry.samples().n(); g=g+2) {
			Real wgt = ptentry.wgts()(g);
			Int typ = ptentry.typs()(g);
			if(typ==0) {
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
			}
			if(typ==1) {
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
			}
			if(typ==2) {
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
				vnls.push_back( std::pair<NumTns<SparseVec>,Real>(pptmp[cntpp], wgt) );	cntpp++;
			}
		}
		iA(cntpp==numpp);
	}
	return 0;
}
*/

} // namespace dgdft

