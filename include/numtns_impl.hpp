#ifndef _NUMTNS_IMPL_HPP_
#define _NUMTNS_IMPL_HPP_

#include  "numtns_decl.hpp"

namespace  dgdft{

// TODO Move the things from decl to impl

template <class F> inline std::ostream& operator<<( std::ostream& os, const NumTns<F>& tns)
{
  os<<tns.m()<<" "<<tns.n()<<" "<<tns.p()<<std::endl;
  os.setf(std::ios_base::scientific, std::ios_base::floatfield);
  for(Int i=0; i<tns.m(); i++) {
	 for(Int j=0; j<tns.n(); j++) {
		for(Int k=0; k<tns.p(); k++) {
		  os<<" "<<tns(i,j,k);
		}
		os<<std::endl;
	 }
	 os<<std::endl;
  }
  return os;
}

template <class F> inline void SetValue(NumTns<F>& T, F val)
{
	F *ptr = T.data_;
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) *(ptr++) = val; 

	return;
}

template <class F> inline Real Energy(const NumTns<F>& T)
{
  Real sum = 0;

	F *ptr = T.Data();
  for(Int i=0; i < T.m() * T.n() * T.p(); i++) 
		sum += abs(ptr[i]) * abs(ptr[i]);

	return sum;
}




} // namespace dgdft

#endif // _NUMTNS_IMPL_HPP_
