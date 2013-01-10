#include  "hamiltonian_dg.hpp"

namespace dgdft{


// *********************************************************************
// Hamiltonian class for DG
// *********************************************************************

//void
//HamiltonianDG::CalculatePseudoCharge	( PeriodTable &ptable ){
//#ifndef _RELEASE_
//	PushCallStack("HamiltonianDG::CalculatePseudoCharge");
//#endif
//	Int ntot = domain_.NumGridTotal();
//	Int numAtom = atomList_.size();
//
//	Real vol = domain_.Volume();
//
//	SetValue( pseudoCharge_, 0.0 );
//  // Calculate the number of occupied states
//  Int nelec = 0;
//  for (Int a=0; a<numAtom; a++) {
//    Int atype  = atomList_[a].type;
//		if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
//			throw std::logic_error( "Cannot find the atom type." );
//		}
//    nelec = nelec + ptable.ptemap()[atype].params(PTParam::ZION);
//  }
//
//	if( nelec % 2 != 0 ){
//		throw std::runtime_error( "This is a spin-restricted calculation. nelec should be even." );
//	}
//	numOccupiedState_ = nelec;
//
//  pseudoChargeList_.resize( numAtom );
//  for (Int a=0; a<numAtom; a++) {
//    ptable.CalculatePseudoCharge( atomList_[a], domain_, pseudoChargeList_[a] );
//    //accumulate
//    IntNumVec &idx = pseudoChargeList_[a].first;
//    DblNumMat &val = pseudoChargeList_[a].second;
//    for (Int k=0; k<idx.m(); k++) 
//			pseudoCharge_[idx(k)] += val(k, VAL);
//  }
//
//  Real sumrho = 0.0;
//  for (Int i=0; i<ntot; i++) 
//		sumrho += pseudoCharge_[i]; 
//  sumrho *= vol / Real(ntot);
//
//	Print( statusOFS, "Sum of Pseudocharge        = ", sumrho );
//	Print( statusOFS, "Number of Occupied States  = ", numOccupiedState_ );
//  
//  Real diff = (numOccupiedState_ - sumrho) / vol;
//  for (Int i=0; i<ntot; i++) 
//		pseudoCharge_(i) += diff; 
//
//#ifndef _RELEASE_
//	PopCallStack();
//#endif
//
//	return ;
//} 		// -----  end of method HamiltonianDG::CalculatePseudoCharge  ----- 


void HamiltonianDG::CalculateHartree( DistFourier& fft ) {
#ifndef _RELEASE_ 
	PushCallStack("HamiltonianDG::CalculateHartree");
#endif
	if( !fft.isInitialized ){
		throw std::runtime_error("Fourier is not prepared.");
	}
 	
  Int ntot      = fft.numGridTotal;
	Int ntotLocal = fft.numGridLocal;

	// TODO Convert density to densityLocal;

	// The contribution of the pseudoCharge is subtracted. So the Poisson
	// equation is well defined for neutral system.

	for( Int i = 0; i < ntotLocal; i++ ){
		fft.inputComplexVecLocal(i) = Complex( 
				densityLocal_(i) - pseudoChargeLocal_(i), 0.0 );
	}
	fftw_execute( fft.forwardPlan );

	for( Int i = 0; i < ntotLocal; i++ ){
		if( fft.gkkLocal(i) == 0 ){
			fft.outputComplexVecLocal(i) = Z_ZERO;
		}
		else{
			// NOTE: gkk already contains the factor 1/2.
			fft.outputComplexVecLocal(i) *= 2.0 * PI / fft.gkkLocal(i);
		}
	}
	fftw_execute( fft.backwardPlan );

	for( Int i = 0; i < ntotLocal; i++ ){
		vhartLocal_(i) = fft.inputComplexVecLocal(i).real() / ntot;
	}

#ifndef _RELEASE_
	PopCallStack();
#endif
	return; 
}  // -----  end of method HamiltonianDG::CalculateHartree ----- 

} // namespace dgdft
