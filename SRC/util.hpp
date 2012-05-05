#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "commoninc.hpp"
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "numvec.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "cblacs.h"
//#include "mkl_scalapack.h"

//-------------------------------------
//Physical constants

const double au2K = 315774.67;
const double amu2au = 1822.8885;


//-------------------------------------
//Utility functions

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
inline int iround(double a){
  int b = 0;
  if(a>0)
    b = (a-int(a)<0.5)?int(a):(int(a)+1);
  else
    b = (int(a)-a<0.5)?int(a):(int(a)-1);
  return b; 
}

inline int IMOD(int a,int b){
  return ((a%b)<0)?((a%b)+b):(a%b);
}

inline double DMOD(double a,double b){
  assert(b>0);
  return (a>=0)?(a-int(a/b)*b):(a-(int(a/b)-1)*b);
}

inline double dunirand(void){
  return (double)rand()/(double(RAND_MAX)+1.0);
}

#define ABORT(err_msg,err_code)\
{\
    fprintf(stdout,"%s at line %d in file %s\n",err_msg,__LINE__,__FILE__);\
    fflush(stdout); \
    MPI_Finalize(); \
    exit(err_code);\
}

using namespace std;

typedef pair<IntNumVec,DblNumMat> SparseVec; //LY: contains four components, (val, dx, dy, dz in each column)

inline int optionsCreate(int argc, char** argv, map<string,string>& options)
{
  options.clear();
  for(int k=1; k<argc; k=k+2) {
    options[ string(argv[k]) ] = string(argv[k+1]);
  }
  return 0;
}

inline void gettime(double *t){
  *t = clock()/(double)CLOCKS_PER_SEC;
  return;
}

inline double norm(double* x, int l){
  int i;
  double nm = 0.0;
  for(i = 0; i < l; i++)
    nm += x[i]*x[i];
  nm = pow(nm,1.0/2.0);
  return nm;
}

extern void lglnodes(double* x,  int N);
extern void lglnodes(double* x,  double* w, int N);
extern void lglnodes(vector<double>& x,  int N);
extern void lglnodes(vector<double>& x,  vector<double>& w, vector<double>& P, int N);
extern void lglnodes(vector<double>& x, vector<double>& D, int N);
extern void lag1dm(double*, int, int, double*, double*);

template<class F>
void Transpose(vector<F>& A, vector<F>& B, int m, int n){
  int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

template<class F>
void Transpose(double* A, double* B, int m, int n){
  int i, j;
  for(i = 0; i < m; i++){
    for(j = 0; j < n; j++){
      B[j+n*i] = A[i+j*m];
    }
  }
}

void spline(int, double*, double*, double*, double*, double*);
void seval(double*, int, double*, int, double*, double*, double*, double*, double*);


inline double Innerprod(double* x, double* y, double *w, int ntot){
  double tmp = 0.0;
  for(int i = 0; i < ntot; i++){
    tmp += x[i] * y[i] * w[i];
  }
  return tmp;
}

extern void XScaleByY(double* x, double* y, int ntot);


#endif

