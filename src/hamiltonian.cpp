#include  "hamiltonian.hpp"
#include  "blas.hpp"

namespace dgdft{


// *********************************************************************
// Hamiltonian class (base class)
// *********************************************************************

Hamiltonian::Hamiltonian	( 
			const Domain                   &dm, 
			const std::vector<Atom>        &atomList, 
			const std::string               pseudoType,
			const Int                       XCId, 
			const Int                       numExtraState, 
      const Int                       numDensityComponent):
  domain_(dm), atomList_(atomList), pseudoType_(pseudoType),
	XCId_(XCId), numExtraState_(numExtraState)
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::Hamiltonian");
#endif
  Int ntot = domain_.NumGridTotal();

	density_.Resize( ntot, numDensityComponent );   
	SetValue( density_, 0.0 );

	pseudoCharge_.Resize( ntot );
	SetValue( pseudoCharge_, 0.0 );
	
	vext_.Resize( ntot );
	SetValue( vext_, 0.0 );

	vhart_.Resize( ntot );
	SetValue( vhart_, 0.0 );

	vtot_.Resize( ntot );
	SetValue( vtot_, 0.0 );

	epsxc_.Resize( ntot );
	SetValue( epsxc_, 0.0 );

	vxc_.Resize( ntot, numDensityComponent );
	SetValue( vxc_, 0.0 );

#ifndef _RELEASE_
	PopCallStack();
#endif
	return ;
} 		// -----  end of method Hamiltonian::Hamiltonian  ----- 


// TODO
void
Hamiltonian::CalculateOccupationRate	( const Real Tbeta )
{
#ifndef _RELEASE_
	PushCallStack("Hamiltonian::CalculateOccupationRate");
#endif
	/* For a given finite temperature, update the occupation number */
	// FIXME Magic number here
	Real tol = 1e-10; 
	Int maxiter = 100;  

	Real lb, ub, flb, fub, occsum;
	Int ilb, iub, iter;

	Int npsi       = this->NumStateTotal();
	Int nOccStates = this->NumOccupiedState();

	if( npsi > nOccStates)  {
		/* use bisection to find efermi such that 
		 * sum_i fermidirac(ev(i)) = nocc
		 */
		ilb = nOccStates-1;
		iub = nOccStates+1;

		lb = eigVal_(ilb-1);
		ub = eigVal_(iub-1);

		/* Calculate Fermi-Dirac function and make sure that
		 * flb < nocc and fub > nocc
		 */

		flb = 0.0;
		fub = 0.0;
		for(Int j = 0; j < npsi; j++) {
			flb += 1.0 / (1.0 + exp(Tbeta*(eigVal_(j)-lb)));
			fub += 1.0 / (1.0 + exp(Tbeta*(eigVal_(j)-ub))); 
		}

		while( (nOccStates-flb)*(fub-nOccStates) < 0 ) {
			if( flb > nOccStates ) {
				if(ilb > 0){
					ilb--;
					lb = eigVal_(ilb-1);
					flb = 0.0;
					for(Int j = 0; j < npsi; j++) flb += 1.0 / (1.0 + exp(Tbeta*(eigVal_(j)-lb)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi" );
				}
			}

			if( fub < nOccStates ) {
				if( iub < npsi ) {
					iub++;
					ub = eigVal_(iub-1);
					fub = 0.0;
					for(Int j = 0; j < npsi; j++) fub += 1.0 / (1.0 + exp(Tbeta*(eigVal_(j)-ub)));
				}
				else {
					throw std::logic_error( "Cannot find a lower bound for efermi, try to increase the number of wavefunctions" );
				}
			}
		}  /* end while */

		fermi_ = (lb+ub)*0.5;
		occsum = 0.0;
		for(Int j = 0; j < npsi; j++) {
			occupationRate_(j) = 1.0 / (1.0 + exp(Tbeta*(eigVal_(j) - fermi_)));
			occsum += occupationRate_(j);
		}

		/* Start bisection iteration */
		iter = 1;
		while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
			if( occsum < nOccStates ) {lb = fermi_;}
			else {ub = fermi_;}

			fermi_ = (lb+ub)*0.5;
			occsum = 0.0;
			for(Int j = 0; j < npsi; j++) {
				occupationRate_(j) = 1.0 / (1.0 + exp(Tbeta*(eigVal_(j) - fermi_)));
				occsum += occupationRate_(j);
			}
			iter++;
		}
	}
	else {
		if (npsi == nOccStates ) {
			for(Int j = 0; j < npsi; j++) 
				occupationRate_(j) = 1.0;
			fermi_ = eigVal_(npsi-1);
		}
		else {
			throw std::logic_error( "The number of eigenvalues in ev should be larger than nocc" );
		}
	}

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method Hamiltonian::CalculateOccupationRate  ----- 


// *********************************************************************
// KohnSham class
// *********************************************************************

KohnSham::KohnSham() {}

KohnSham::~KohnSham() {
	xc_func_end(&XCFuncType_);
}

KohnSham::KohnSham( 
			const Domain                   &dm, 
			const std::vector<Atom>        &atomList, 
			const std::string               pseudoType,
			const Int                       XCId,
			const Int                       numExtraState, 
      const Int                       numDensityComponent ) : Hamiltonian(
				dm, atomList, pseudoType, XCId, numExtraState, numDensityComponent ) 
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::KohnSham");
#endif
	// Initialize the XC functional.  
	// Spin-unpolarized functional is used here
	if( xc_func_init(&XCFuncType_, XCId_, XC_UNPOLARIZED) != 0 ){
    throw std::runtime_error( "XC functional initialization error." );
	} 
#ifndef _RELEASE_
	PopCallStack();
#endif
};


void
KohnSham::CalculatePseudoCharge	( PeriodTable &ptable ){
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculatePseudoCharge");
#endif
	Int ntot = domain_.NumGridTotal();
	Int numAtom = atomList_.size();

	Real vol = domain_.Volume();

	SetValue( pseudoCharge_, 0.0 );
  // calculate the number of occupied states
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
		if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
			throw std::logic_error( "Cannot find the atom type." );
		}
    nelec = nelec + ptable.ptemap()[atype].params()(PeriodTable::i_Zion);
  }
	// FIXME This is the spin-restricted calculation
	if( nelec % 2 != 0 ){
		throw std::runtime_error( "This is spin-restricted calculation. nelec should be even." );
	}
	numOccupiedState_ = ceil( Real(nelec) / 2 );

  pseudoChargeList_.resize( numAtom );
  for (Int a=0; a<numAtom; a++) {
    ptable.CalculatePseudoCharge( atomList_[a], domain_, pseudoChargeList_[a] );
    //accumulate
    IntNumVec &idx = pseudoChargeList_[a].first;
    DblNumMat &val = pseudoChargeList_[a].second;
    for (Int k=0; k<idx.m(); k++) 
			pseudoCharge_[idx(k)] += val(k, VAL);
  }

  Real sumrho = 0.0;
  for (Int i=0; i<ntot; i++) 
		sumrho += pseudoCharge_[i]; 
  sumrho *= vol / Real(ntot);

	// FIXME
	std::cerr<<"sum of pseudoCharge"<<sumrho<<" total state"
		<< numOccupiedState_<<std::endl;
  
  Real diff = (numOccupiedState_ - sumrho) / vol;
  for (Int i=0; i<ntot; i++) 
		pseudoCharge_[i] += diff; 


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculatePseudoCharge  ----- 




void
KohnSham::CalculateNonlocalPP	( PeriodTable &ptable )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateNonlocalPP");
#endif
  vnlDoubleList_.resize( atomList_.size() );

	std::vector<DblNumVec> gridpos;
  UniformMesh ( domain_, gridpos );

  Int cnt = 0; // the total number of PS used
  for ( Int a=0; a < atomList_.size(); a++ ) {
		ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
				vnlDoubleList_[a]); 
		cnt = cnt + vnlDoubleList_[a].size();
  }
	// FIXME
	std::cerr << "Total number of nonlocal pseudopotential = " << cnt << std::endl;

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateNonlocalPP  ----- 




void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateDensity");
#endif
	Int ntot  = psi.NumGridTotal();
	Int ncom  = psi.NumComponent();
	Int nocc  = psi.NumState();
	Real vol  = domain_.Volume();

	SetValue( density_, 0.0 );
	for (Int k=0; k<nocc; k++) {
		for (Int j=0; j<ncom; j++) {
			for (Int i=0; i<ntot; i++) {
				density_(i,RHO) += occrate(k) * 
					pow( abs(psi.Wavefun(i,j,k)), 2.0 );
			}
		}
	}

	// Scale the density
	blas::Scal( ntot, ntot / vol, density_.VecData(RHO), 1 );
  val = 0.0; // sum of density
  for (Int i=0; i<ntot; i++) {
    val  += density_(i, RHO) * vol / ntot;
  }
	
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateDensity  ----- 


void
KohnSham::CalculateXC	( Real &val )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateXC");
#endif
	Int ntot = domain_.NumGridTotal();
	Real vol = domain_.Volume();

  switch( XCFuncType_.info->family ){
    case XC_FAMILY_LDA:
       xc_lda_exc_vxc( &XCFuncType_, ntot, density_.VecData(RHO), 
		      epsxc_.Data(), vxc_.Data() );
      break;
    default:
			throw std::logic_error( "Unsupported XC family!" );
      break;
  }

  // Compute the total exchange-correlation energy
  val = 0.0;
  for(Int i = 0; i < ntot; i++){
    val += density_(i, RHO) * epsxc_(i) * vol / (Real) ntot;
  }


#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateXC  ----- 


void KohnSham::CalculateHartree( Fourier& fft ) {
#ifndef _RELEASE_ 
	PushCallStack("KohnSham::CalculateHartree");
#endif
	if( !fft.isPrepared ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 	
  Int ntot = domain_.NumGridTotal();
	if( fft.domain.NumGridTotal() != ntot ){
		throw std::logic_error( "Grid size does not match!" );
	}

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.
	for( Int i = 0; i < ntot; i++ ){
		fft.inputComplexVec(i) = Complex( 
				density_(i,RHO) - pseudoCharge_(i), 0.0 );
	}
	fftw_execute( fft.forwardPlan );

	for( Int i = 0; i < ntot; i++ ){
		if( fft.gkk(i) == 0 ){
			fft.outputComplexVec(i) = Z_ZERO;
		}
		else{
			// NOTE: gkk already contains the factor 1/2.
			fft.outputComplexVec(i) *= 2.0 * PI / fft.gkk(i);
		}
	}
	fftw_execute( fft.backwardPlan );

	for( Int i = 0; i < ntot; i++ ){
		vhart_(i) = fft.inputComplexVec(i).real() / ntot;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 


void
KohnSham::CalculateVtot	( DblNumVec& vtot  )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::CalculateVtot");
#endif
	Int ntot = domain_.NumGridTotal();
  for (int i=0; i<ntot; i++) {
		vtot(i) = vext_(i) + vhart_(i) + vxc_(i, RHO);
  }

#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::CalculateVtot  ----- 


void
KohnSham::MultSpinor	( Spinor& psi, NumTns<Scalar>& a3, Fourier& fft )
{
#ifndef _RELEASE_
	PushCallStack("KohnSham::MultSpinor");
#endif
	SetValue( a3, SCALAR_ZERO );
	psi.AddScalarDiag( vtot_, a3 );
	psi.AddLaplacian( a3, &fft );
  psi.AddNonlocalPP( vnlDoubleList_, a3 );
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
} 		// -----  end of method KohnSham::MultSpinor  ----- 



} // namespace dgdft
