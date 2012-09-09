#include  "spinor.hpp"

namespace dgdft{

Spinor::Spinor 
	() : numGridLocal_(0), numComponent_(0),
	numState_(0), isNormalized_(false) { } 		
// -----  end of method Spinor::Spinor  ----- 

Spinor::Spinor 
	( const Domain &dm, 
		const Int     numGridLocal,
		const Int     numComponent,
		const Int     numState,
		const Scalar  val )
	{
#ifndef _RELEASE_
		PushCallStack("Spinor::Spinor");
#endif  // ifndef _RELEASE_
		PetscErrorCode ierr;
		domain_       = dm;
		numGridLocal_ = numGridLocal;
		numComponent_ = numComponent;
		numState_     = numState;

		// TODO Make sure that the sum of local grids equal to the number
		// of grid points in the domain

		localWavefun_.Resize( numGridLocal_ * numComponent_ * numState_ );
		SetValue( localWavefun_, val );
		isNormalized_ = false;

		wavefun_.resize( numComponent_ * numState_ );
		for( Int k = 0; k < numState_; k++ ){
			for( Int j = 0; j < numComponent_; j++ ){
				ierr = VecCreateMPIWithArray( dm.comm, 1, numGridLocal_, dm.NumGridTotal(),
						reinterpret_cast<const PetscScalar*>(this->LocalWavefunData( j, k )), 
						&(this->Wavefun( j, k )) );
				if( ierr ) throw ierr;
			}
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method Spinor::Spinor  ----- 


Spinor::Spinor
	( const Domain &dm, 
		const Int numGridLocal,
		const Int numComponent, 
		const Int numState,
		const bool owndata, 
		Scalar* data )
	{
#ifndef _RELEASE_
		PushCallStack("Spinor::Spinor");
#endif  // ifndef _RELEASE_
		PetscErrorCode ierr;
		domain_       = dm;
		numGridLocal_ = numGridLocal;
		numComponent_ = numComponent;
		numState_     = numState;

		// TODO Make sure that the sum of local grids equal to the number
		// of grid points in the domain

		if( owndata == true ){
			// Make a copy
			localWavefun_.Resize( numGridLocal_ * numComponent_ * numState_ );

			for( Int i = 0; i < numGridLocal_; i++ ){
				localWavefun_[i] = data[i];
			}
		}
		else{
			// Just view the array obtained from data
			localWavefun_ = NumVec<Scalar>
				( numGridLocal_ * numComponent_ * numState_, false, 
					data );
		}

		isNormalized_ = false;

		wavefun_.resize( numComponent_ * numState_ );
		for( Int k = 0; k < numState_; k++ ){
			for( Int j = 0; j < numComponent_; j++ ){
				ierr = VecCreateMPIWithArray( dm.comm, 1, numGridLocal_, dm.NumGridTotal(),
						reinterpret_cast<const PetscScalar*>(this->LocalWavefunData( j, k )), 
						&(this->Wavefun( j, k )) );
				if( ierr ) throw ierr;
			}
		}

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_

	} 		// -----  end of method Spinor::Spinor  ----- 


Spinor::~Spinor	
	() {
#ifndef _RELEASE_
		PushCallStack("Spinor::~Spinor");
#endif  // ifndef _RELEASE_
		PetscErrorCode ierr;
		for( Int i = 0; i < numComponent_ * numState_; i++ ){
			ierr = VecDestroy( &(wavefun_[i]) ); 
			if( ierr ) throw ierr;
		}
#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
	} 		// -----  end of method Spinor::~Spinor  ----- 


Vec& Spinor::Wavefun	
	( Int j, Int k ) {
#ifndef _RELEASE_
		PushCallStack("Spinor::Wavefun");
#endif  // ifndef _RELEASE_
		if( j < 0 || j >= numComponent_ ) 
			throw std::logic_error( "Component index is out of bound" );
		if( k < 0 || k >= numState_ )
			throw std::logic_error( "State index is out of bound" );

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return wavefun_[ j + k * numComponent_ ];
	} 		// -----  end of method Spinor::Wavefun  ----- 

const Vec& Spinor::LockedWavefun	
	( Int j, Int k ) const {
#ifndef _RELEASE_
		PushCallStack("Spinor::LockedWavefun");
#endif  // ifndef _RELEASE_
		if( j < 0 || j >= numComponent_ ) 
			throw std::logic_error( "Component index is out of bound" );
		if( k < 0 || k >= numState_ )
			throw std::logic_error( "State index is out of bound" );

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_
		return wavefun_[ j + k * numComponent_ ];
	} 		// -----  end of method Spinor::LockedWavefun  ----- 


Scalar* 
	Spinor::LocalWavefunData	( Int j, Int k  )
	{
#ifndef _RELEASE_
		PushCallStack("Spinor::LocalWavefunData");
#endif  // ifndef _RELEASE_
		if( j < 0 || j >= numComponent_ ) 
			throw std::logic_error( "Component index is out of bound" );
		if( k < 0 || k >= numState_ )
			throw std::logic_error( "State index is out of bound" );

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_

		return localWavefun_.Data() + ( j + k * numComponent_ ) * numGridLocal_; 
	} 		// -----  end of method Spinor::LocalWavefunData  ----- 

const Scalar* 
	Spinor::LockedLocalWavefunData	( Int j, Int k  ) const
	{
#ifndef _RELEASE_
		PushCallStack("Spinor::LockedLocalWavefunData");
#endif  // ifndef _RELEASE_
		if( j < 0 || j >= numComponent_ ) 
			throw std::logic_error( "Component index is out of bound" );
		if( k < 0 || k >= numState_ )
			throw std::logic_error( "State index is out of bound" );

#ifndef _RELEASE_
		PopCallStack();
#endif  // ifndef _RELEASE_

		return localWavefun_.Data() + ( j + k * numComponent_ ) * numGridLocal_;
	} 		// -----  end of method Spinor::LockedLocalWavefunData  ----- 


//int Spinor::normalize() { // not used in practice, already normalized in eigensolver	
//	if (_is_normalized) { return 0; }
//	else {
//		int size = _wavefun._m * _wavefun._n;
//		int nocc = _wavefun._p;
//
//		for (int k=0; k<nocc; k++) {
//			cpx    *ptr = _wavefun.matdata(k);
//			double sum = 0.0;
//			for (int i=0; i<size; i++) {
//				sum += pow(abs(*ptr++), 2.0);
//			}
//			sum = sqrt(sum);
//			if (sum != 0.0) {
//				ptr = _wavefun.matdata(k);
//				for (int i=0; i<size; i++) *(ptr++) /= sum;
//			}
//		}
//		_is_normalized = true;
//		return 0;
//	}
//};

//int Spinor::add_scalardiag (DblNumVec &val, CpxNumTns &a3) {
//	iA ((val._m != 0) && (val._m == _wavefun._m));
//	// beforing call this subroutine
//	// make sure that all the needed data is initialized
//
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *p1 = _wavefun.clmdata(j,k);
//			double *p2 = val.Data();
//			cpx    *p3 = a3.clmdata(j,k);
//			for (int i=0; i<ntot; i++) { *(p3) += (*p1) * (*p2); p3++; p1++; p2++; }
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_nonlocalPS 
//(vector< vector< pair<SparseVec,double> > > &val, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			cpx    *ptr1 = a3.clmdata(j,k);
//			int natm = val.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				int nobt = val[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = val[iatm][iobt].first;
//					double vnlwgt = val[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//
//
//					cpx    weight = cpx(0.0, 0.0); 
//					int dvm = dv.m(); 
//					// dvm = 4 here to represent the function value, 
//					// and its derivatives along x,y,z directions.
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						weight += (*dvptr) * ptr0[*ivptr];
//						ivptr++; dvptr += dvm;
//					}
//					weight *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						ptr1[*ivptr] += (*dvptr) * weight;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}
//		}
//	}
//	return 0;
//};
//
//
//int Spinor::add_D2_c2c (CpxNumTns &a3, FFTPrepare &fp) {
//	// should be noted here D2 == \nabla^2/2.
//	// make sure the FFTW is ready
//	if (!fp._is_prepared) {
//		fp.setup_xyz(_domain._Ns[0], _domain._Ns[1], _domain._Ns[2], 
//				_domain._Ls[0], _domain._Ls[1], _domain._Ls[2]);
//	}
//
//	int ntot     = fp._size;
//	int nocc     = _wavefun._p;
//	int ncom     = _wavefun._n;
//
//	iA (ntot == _wavefun._m);
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//
//			double *ptr1d = fp._gkk.Data();
//			ptr0 = fp._out_cpx;
//			for (int i=0; i<ntot; i++) *(ptr0++) *= *(ptr1d++);
//
//			//      ptr0 = fp._out_cpx;
//			//      fftw_execute_dft(fp._backward, reinterpret_cast<fftw_complex*>(ptr0), 
//			//	reinterpret_cast<fftw_complex*>(fp._in_cpx));
//
//			fftw_execute(fp._backward);
//			cpx   *ptr1 = a3.clmdata(j,k);
//			ptr0 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) *(ptr1++) += *(ptr0++) / double(ntot);
//
//		}
//	}
//	return 0;
//};
//
//int Spinor::get_D2_c2c (CpxNumTns &a3, FFTPrepare &fp) {
//	// should be noted here D2 == \nabla^2/2.
//	// make sure the FFTW is ready
//	if (!fp._is_prepared) {
//		fp.setup_xyz(_domain._Ns[0], _domain._Ns[1], _domain._Ns[2], 
//				_domain._Ls[0], _domain._Ls[1], _domain._Ls[2]);
//	}
//
//	int ntot     = fp._size;
//	int nocc     = _wavefun._p;
//	int ncom     = _wavefun._n;
//
//	iA (ntot == _wavefun._m);
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//			cpx    *ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//
//			double *ptr1d = fp._gkk.Data();
//			ptr0 = fp._out_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) *= *(ptr1d++) / ntot; //divide by ntot
//			}
//
//			ptr0 = a3.clmdata(j,k);
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//
//		}
//	}
//
//	//  int size = ntot * ncom * nocc;
//	//  cpx *ptr0 = a3.Data();
//	//  for (int i=0; i<size; i++) *(ptr0++) /= ntot;
//
//	return 0;
//};
//
//int Spinor::add_sigma_x (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j+1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//			cpx    *ptr0 = a3.clmdata(j,k);
//
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j-1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_sigma_y (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j+1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) -= cpx(0.0, 1.0) * (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j-1,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += cpx(0.0, 1.0) * (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_sigma_z (DblNumVec &a1, CpxNumTns &a3) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//		for (int j=1; j<ncom; j+=2) {
//			double tsig = - 1.0;
//			if (j<2) tsig = 1.0;
//
//			cpx    *ptr0 = a3.clmdata(j,k);
//			double *ptrd = a1.Data();
//			cpx    *ptr1 = _wavefun.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) -= (*(ptr1++)) * (*(ptrd++)) * tsig;
//			}
//		}
//	}
//	return 0;
//};
//
//int Spinor::add_nonlocalPS_SOC 
//(vector< vector< pair< SparseVec,double> > > &val, 
// vector<Atom> &atomvec, vector<DblNumVec> &grid, 
// CpxNumTns &a3, FFTPrepare &fp) {
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	int nx = _domain._Ns(0);
//	int ny = _domain._Ns(1);
//	int nz = _domain._Ns(2);
//
//	CpxNumVec psix, psiy, psiz;
//	psix.resize(ntot);
//	psiy.resize(ntot);
//	psiz.resize(ntot);
//	cpx *ptr0, *ptr1, *ptr2;
//
//	CpxNumVec lx, ly, lz;
//	lx.resize(ntot); 
//	ly.resize(ntot); 
//	lz.resize(ntot); 
//	cpx *ptr_lx, *ptr_ly, *ptr_lz;
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//
//			// get the moment
//			ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, 
//					reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//			// px
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(0);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) { // divide by ntot
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psix.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// py
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(1);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiy.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// pz
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(2);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiz.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// end of geting the moment
//
//			ptr0 = psix.Data();
//			ptr1 = psiy.Data();
//			ptr2 = psiz.Data();
//
//			setvalue(lx, cpx(0.0, 0.0));
//			setvalue(ly, cpx(0.0, 0.0));
//			setvalue(lz, cpx(0.0, 0.0));
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			ptr_lz = lz.Data();
//
//			int natm = val.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				Point3 coord = atomvec[iatm].coord();
//				int nobt = val[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = val[iatm][iobt].first;
//					double vnlwgt = val[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//					cpx weight0 = cpx(0.0, 0.0);
//					cpx weight1 = cpx(0.0, 0.0);
//					cpx weight2 = cpx(0.0, 0.0);
//					int dvm = dv.m(); 
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					//	  for (int i=0; i<iv.m(); i++) {
//					//	    weight0 += (*dvptr) * ptr0[*ivptr];
//					//	    weight1 += (*dvptr) * ptr1[*ivptr];
//					//	    weight2 += (*dvptr) * ptr2[*ivptr];
//					//	    ivptr++; dvptr += dvm;
//					//	  }
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						Index3 icoord;
//						icoord(2) = itmp/(nx*ny);
//						icoord(1) = (itmp - icoord(2)*nx*ny)/nx;
//						icoord(0) = (itmp - icoord(2)*nx*ny - icoord(1)*nx);
//						double dtmp = *dvptr;
//						double rel0 = grid[0](icoord(0))-coord(0);
//						double rel1 = grid[1](icoord(1))-coord(1);
//						double rel2 = grid[2](icoord(2))-coord(2);
//						// Shifting: VERY IMPORTANT
//						rel0 = rel0 - iround(rel0 / _domain._Ls[0]) * _domain._Ls[0];
//						rel1 = rel1 - iround(rel1 / _domain._Ls[1]) * _domain._Ls[1];
//						rel2 = rel2 - iround(rel2 / _domain._Ls[2]) * _domain._Ls[2];
//
//						weight0 += dtmp * (rel1*ptr2[itmp]-rel2*ptr1[itmp]);
//						weight1 += dtmp * (rel2*ptr0[itmp]-rel0*ptr2[itmp]);
//						weight2 += dtmp * (rel0*ptr1[itmp]-rel1*ptr0[itmp]);
//						ivptr++; dvptr += dvm;
//					}
//					weight0 *= vol/double(ntot)*vnlwgt;
//					weight1 *= vol/double(ntot)*vnlwgt;
//					weight2 *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						double dtmp = *dvptr;
//						ptr_lx[itmp] += dtmp * weight0;
//						ptr_ly[itmp] += dtmp * weight1;
//						ptr_lz[itmp] += dtmp * weight2;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}
//
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			cpx sign0;
//			double sign1;
//			if (j%2 == 0) { 
//				ptr0 = a3.clmdata(j+1,k); 
//				sign0 = cpx(0.0, +1.0); 
//				sign1 = +1.0; 
//			}
//			else          { 
//				ptr0 = a3.clmdata(j-1,k); 
//				sign0 = cpx(0.0, -1.0); 
//				sign1 = -1.0; 
//			}
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = lz.Data();
//			ptr0   = a3.clmdata(j,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lz++) * sign1 );
//			}
//		}
//	}
//
//	return 0;
//};
//
//int Spinor::add_matrix_ij(int ir, int jc, DblNumVec &a1, CpxNumTns &a3) {
//
//	int ntot = _wavefun._m;
//	// int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//
//	for (int k=0; k<nocc; k++) {
//		cpx    *ptr0 = a3.clmdata(ir,k);
//		double *ptr1 = a1.Data();
//		cpx    *ptr2 = _wavefun.clmdata(jc,k);
//		for (int i=0; i<ntot; i++) {
//			*(ptr0++) += *(ptr1++) * (*(ptr2++));
//		}
//	}
//	return 0;
//};
//int Spinor::add_matrix_ij(int ir, int jc, double  *ptr1, CpxNumTns &a3) {
//
//	int ntot = _wavefun._m;
//	int nocc = _wavefun._p;
//
//	for (int k=0; k<nocc; k++) {
//		cpx    *ptr0 = a3.clmdata(ir,k);
//		// double *ptr1 = a1.Data();
//		cpx    *ptr2 = _wavefun.clmdata(jc,k);
//		for (int i=0; i<ntot; i++) {
//			*(ptr0++) += *(ptr1++) * (*(ptr2++));
//		}
//	}
//	return 0;
//};
//
//int Spinor::get_DKS (DblNumVec &vtot, DblNumMat &vxc,
//		vector< vector< pair<SparseVec,double> > > &vnlss,
//		vector< vector< pair<SparseVec,double> > > &vnlso,
//		vector<Atom> &atomvec, CpxNumTns &a3, FFTPrepare &fp,  
//		vector<DblNumVec> &grid) {
//
//	int ntot = _wavefun._m;
//	int ncom = _wavefun._n;
//	int nocc = _wavefun._p;
//	double vol = _domain._Ls[0] * _domain._Ls[1] * _domain._Ls[2];
//
//	int nx = _domain._Ns(0);
//	int ny = _domain._Ns(1);
//	int nz = _domain._Ns(2);
//
//	// get vtot only for four component
//	double energyShift = 2.0 * pow(SPEED_OF_LIGHT, 2.0);
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<2; j++) {
//			cpx    *p1 = _wavefun.clmdata(j,k);
//			cpx    *p3 =      a3.clmdata(j,k);
//			double *p2 =    vtot.Data();
//			for (int i=0; i<ntot; i++) { 
//				*(p3) = (*p1) * (*p2); 
//				p3++; p1++; p2++; 
//			}
//		}
//		for (int j=2; j<ncom; j++) {
//			cpx    *p1 = _wavefun.clmdata(j,k);
//			cpx    *p3 =      a3.clmdata(j,k);
//			double *p2 =    vtot.Data();
//			for (int i=0; i<ntot; i++) { 
//				*(p3) = (*p1) * (*p2 - energyShift); 
//				p3++; p1++; p2++; 
//			}
//		}
//	} // end of vtot multiplication
//
//	CpxNumVec psix, psiy, psiz;
//	psix.resize(ntot);
//	psiy.resize(ntot);
//	psiz.resize(ntot);
//	cpx *ptr0, *ptr1, *ptr2;
//
//	CpxNumVec lx, ly, lz;
//	lx.resize(ntot); 
//	ly.resize(ntot); 
//	lz.resize(ntot); 
//	cpx *ptr_lx, *ptr_ly, *ptr_lz;
//
//	for (int k=0; k<nocc; k++) {
//		for (int j=0; j<ncom; j++) {
//
//			// get the moment
//			ptr0 = _wavefun.clmdata(j,k);
//			fftw_execute_dft(fp._forward, 
//					reinterpret_cast<fftw_complex*>(ptr0), 
//					reinterpret_cast<fftw_complex*>(fp._out_cpx));
//			// px
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(0);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) { // divide by ntot
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psix.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// py
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(1);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiy.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// pz
//			ptr0 = fp._out_cpx;
//			ptr1 = fp._ik.clmdata(2);
//			ptr2 = fp._in_cpx;
//			for (int i=0; i<ntot; i++) {
//				*(ptr2++) = *(ptr0++) * (*(ptr1++)/double(ntot)) * cpx(0.0, -1.0);
//			}
//			ptr0 = psiz.Data();
//			fftw_execute_dft(fp._backward, 
//					reinterpret_cast<fftw_complex*>(fp._in_cpx), 
//					reinterpret_cast<fftw_complex*>(ptr0));
//			// end of geting the moment
//
//			cpx sign0;
//			double sign1;
//			int ito;
//
//			if (j%2 == 0) {
//				sign0 = cpx(0.0, +1.0); 
//				sign1 = +1.0; 
//			}
//			else          { 
//				sign0 = cpx(0.0, -1.0); 
//				sign1 = -1.0; 
//			}
//
//			// start $c \vec{\sigma}\cdot\vec{p}$ 
//			ptr_lx = psix.Data();
//			ptr_ly = psiy.Data();
//			ito = ncom-1-j;
//			ptr0 = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += SPEED_OF_LIGHT * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = psiz.Data();
//			if (j<2) { ito = (j+2); } // %ncom for two component
//			else     { ito = (j-2); }
//			ptr0   = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += SPEED_OF_LIGHT * ( *(ptr_lz++) * sign1 );
//			}// end of $c \vec{\sigma} \cdot \vec{p}$
//
//			// start PS-SOC
//			ptr0 = psix.Data();
//			ptr1 = psiy.Data();
//			ptr2 = psiz.Data();
//
//			setvalue(lx, cpx(0.0, 0.0));
//			setvalue(ly, cpx(0.0, 0.0));
//			setvalue(lz, cpx(0.0, 0.0));
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			ptr_lz = lz.Data();
//
//			// get the angular momentum $L_{x,y,z}$
//			int natm = vnlso.size();
//			for (int iatm=0; iatm<natm; iatm++) {
//				Point3 coord = atomvec[iatm].coord();
//				int nobt = vnlso[iatm].size();
//				for (int iobt=0; iobt<nobt; iobt++) {
//					SparseVec &vnlvec = vnlso[iatm][iobt].first;
//					double vnlwgt = vnlso[iatm][iobt].second;
//					IntNumVec &iv = vnlvec.first;
//					DblNumMat &dv = vnlvec.second;
//					cpx weight0 = cpx(0.0, 0.0);
//					cpx weight1 = cpx(0.0, 0.0);
//					cpx weight2 = cpx(0.0, 0.0);
//					int dvm = dv.m(); 
//					int *ivptr = iv.Data();
//					double *dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						Index3 icoord;
//						icoord(2) = itmp/(nx*ny);
//						icoord(1) = (itmp - icoord(2)*nx*ny)/nx;
//						icoord(0) = (itmp - icoord(2)*nx*ny - icoord(1)*nx);
//						double dtmp = *dvptr;
//						double rel0 = grid[0](icoord(0))-coord(0);
//						double rel1 = grid[1](icoord(1))-coord(1);
//						double rel2 = grid[2](icoord(2))-coord(2);
//						// Shifting: VERY IMPORTANT
//						rel0 = rel0 - iround(rel0 / _domain._Ls[0]) * _domain._Ls[0];
//						rel1 = rel1 - iround(rel1 / _domain._Ls[1]) * _domain._Ls[1];
//						rel2 = rel2 - iround(rel2 / _domain._Ls[2]) * _domain._Ls[2];
//
//						weight0 += dtmp * (rel1*ptr2[itmp]-rel2*ptr1[itmp]);
//						weight1 += dtmp * (rel2*ptr0[itmp]-rel0*ptr2[itmp]);
//						weight2 += dtmp * (rel0*ptr1[itmp]-rel1*ptr0[itmp]);
//						ivptr++; dvptr += dvm;
//					}
//					weight0 *= vol/double(ntot)*vnlwgt;
//					weight1 *= vol/double(ntot)*vnlwgt;
//					weight2 *= vol/double(ntot)*vnlwgt;
//
//					ivptr = iv.Data();
//					dvptr = dv.Data();
//					for (int i=0; i<iv.m(); i++) {
//						int itmp = *ivptr;
//						double dtmp = *dvptr;
//						ptr_lx[itmp] += dtmp * weight0;
//						ptr_ly[itmp] += dtmp * weight1;
//						ptr_lz[itmp] += dtmp * weight2;
//						ivptr++; dvptr +=dvm;
//					}
//				}
//			}// end of $L_{x,y,z}$
//
//			ptr_lx = lx.Data();
//			ptr_ly = ly.Data();
//			if (j<2) { ito = 1-j; } // %ncom for two component
//			else     { ito = 5-j; }
//			ptr0 = a3.clmdata(ito,k); 
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lx++) + sign0 * (*(ptr_ly++)) );
//			}
//
//			ptr_lz = lz.Data();
//			ito = j;
//			ptr0   = a3.clmdata(ito,k);
//			for (int i=0; i<ntot; i++) {
//				*(ptr0++) += 0.5 * ( *(ptr_lz++) * sign1 );
//			}//end of the PS-SOC part
//		}
//	}
//
//	// add nonlocal PS
//	add_nonlocalPS(vnlss, a3);
//
//	// Magnetic part of Vxc
//	DblNumVec BxcX = DblNumVec(ntot, false, vxc.clmdata(MAGX));
//	DblNumVec BxcY = DblNumVec(ntot, false, vxc.clmdata(MAGY));
//	DblNumVec BxcZ = DblNumVec(ntot, false, vxc.clmdata(MAGZ));
//	add_sigma_x(BxcX, a3);
//	add_sigma_y(BxcY, a3);
//	add_sigma_z(BxcZ, a3);
//
//	return 0;
//};
}  // namespace dgdft
