#ifndef _EIGPW_HPP_
#define _EIGPW_HPP_

#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "util.hpp"

#include "fortran_matrix.h"
#include "fortran_interpreter.h"
#include "lobpcg.h"
#include "multi_vector.h"
#include "multivector.h"
#include "pcg_multi.h"
#include "interpreter.h"

#include "periodtable.hpp"

using std::vector;
using std::pair;
using std::map;
using std::set;
using std::cerr;
using std::cout;
using std::ostream;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;

//----------------------------------------------------------------
class EigPW
{
public:
  Point3 _Ls;
  Index3 _Ns;
  int _eigmaxiter;
  double _eigtol;
  //
  //local stuff
  double _vol;
  int _ntot;
  vector<double>* _vtotptr;
  vector< pair<SparseVec,double> >* _vnlptr;
  //
  fftw_plan _planpsibackward;
  fftw_plan _planpsiforward;
  fftw_plan _planpsic2r;
  fftw_plan _planpsir2c;
  vector<double> _gkk;
  vector<double> _gkkhalf;
  /* Preconditioner */
  vector<double> _prec;
  vector<double> _prechalf;
  
public:
  EigPW();
  ~EigPW();
  int setup(); //fftwplan ...
  int solve(vector<double>& vtot, vector< pair<SparseVec,double> >& vnl,
	    int npsi, vector<double>& _psi, vector<double>& _ev, 
	    int& nactive, vector<int>& active_indices); //change vtot and solve
  static void solve_MatMultiVecWrapper(void * A, void * X, void * AX);
  static void solve_ApplyPrecWrapper(  void * A, void * X, void * AX);
  BlopexInt solve_MatMultiVec(serial_Multi_Vector* x, serial_Multi_Vector* y);
  BlopexInt solve_ApplyPrec(  serial_Multi_Vector* X, serial_Multi_Vector * AX);
};


#endif
