#include "util.hpp"

void spline(int n, double* x, double* y, double* b, double* c, double* d){
  /* 
     the coefficients b(i), c(i), and d(i), i=1,2,...,n are computed
     for a cubic interpolating spline

     s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3

     for  x(i) .le. x .le. x(i+1)

     input..

     n = the number of data points or knots (n.ge.2)
     x = the abscissas of the knots in strictly increasing order
     y = the ordinates of the knots

     output..

     b, c, d  = arrays of spline coefficients as defined above.

     using  p  to denote differentiation,

     y(i) = s(x(i))
     b(i) = sp(x(i))
     c(i) = spp(x(i))/2
     d(i) = sppp(x(i))/6  (derivative from the right)

     the accompanying function subprogram  seval  can be used
     to evaluate the spline.
     */
  int nm1, ib, i;
  double t;

  for(i = 0; i < n; i++){
    b[i] = 0.0;
    c[i] = 0.0;
    d[i] = 0.0;
  }
  nm1 = n-1;
  if ( n < 2 ) {
    ABORT("SPLINE REQUIRES N >= 2!", 1);
    return;
  }
  if ( n < 3 ){
    b[0] = (y[1]-y[0])/(x[1]-x[0]);
    c[0] = 0;
    d[0] = 0;
    b[1] = b[0];
    c[1] = 0;
    d[1] = 0;
    return;
  }

  /*
     set up tridiagonal system

     b = diagonal, d = offdiagonal, c = right hand side.
     */ 

  d[0] = x[1] - x[0];
  c[1] = (y[1] - y[0])/d[0];
  for(i = 1; i <  nm1; i++){
    d[i] = x[i+1] - x[i];
    b[i] = 2.*(d[i-1] + d[i]);
    c[i+1] = (y[i+1] - y[i])/d[i];
    c[i] = c[i+1] - c[i];
  }

  /* 
     end  onditions.  third derivatives at  x(1)  and  x(n)
     obtained from divided differences.
     */ 
  b[0] = -d[0];
  b[n-1] = -d[n-2];
  c[0] = 0.;
  c[n-1] = 0.;
  if ( n > 3 ){
    c[0] = c[2]/(x[3]-x[1]) - c[1]/(x[2]-x[0]);
    c[n-1] = c[n-2]/(x[n-1]-x[n-3]) - c[n-3]/(x[n-2]-x[n-4]);
    c[0] = c[0]*d[0]*d[0]/(x[3]-x[0]);
    c[n-1] = -c[n-1]*d[n-2]*d[n-2]/(x[n-1]-x[n-4]);
  }

  /* forward elimination */

  for( i = 1; i < n; i++ ){
    t = d[i-1] / b[i-1];
    b[i] = b[i] - t * d[i-1];
    c[i] = c[i] - t * c[i-1];
  }

  /* backward substitution */
  c[n-1] = c[n-1] / b[n-1];
  for( i = n-2; i >= 0; i-- ){
    c[i] = (c[i] - d[i]*c[i+1]) / b[i];
  }

  /* compute polynomial coefficients */
  b[n-1] = (y[n-1] - y[nm1-1])/d[nm1-1] + d[nm1-1]*(c[nm1-1] + 2.*c[n-1]);
  for(i = 0; i < nm1; i++){
    b[i] = (y[i+1] - y[i])/d[i] - d[i]*(c[i+1] + 2.*c[i]);
    d[i] = (c[i+1] - c[i])/d[i];
    c[i] = 3.*c[i];
  }
  c[n-1] = 3.*c[n-1];
  d[n-1] = d[n-2];

}

//void seval(double* v, int m, double* u, int n, double* x, 
//	   double* y, double* b, double* c, double* d){
//
//  /* ***************************************************
//   * CAUTION! NOTE BEFORE USE:
//   *
//   * This SPLINE function is designed specifically for the interpolation
//   * part for pseudopotential generation in the electronic structure
//   * calculation.  Therefore if u is outside the range [min(x), max(x)],
//   * the corresponding v value will be ZERO!
//   * ***************************************************
//   
//   this subroutine evaluates the  spline function
//
//   seval = y(i) + b(i)*(u-x(i)) +  (i)*(u-x(i))**2 + d(i)*(u-x(i))**3
//
//   where  x(i) .lt. u .lt. x(i+1), using horner's rule
//
//   if  u .lt. x(1) then  i = 1  is used.
//   if  u .ge. x(n) then  i = n  is used.
//
//   input..
//
//   m = the number of output data points
//   n = the number of input data points
//   u = the abs issa at which the spline is to be evaluated
//   v = the value of the spline function at u
//   x,y = the arrays of data abs issas and ordinates
//   b,c,d = arrays of spline coefficients  omputed by spline
//
//   if  u  is not in the same interval as the previous  all, then a
//   binary sear h is performed to determine the proper interval.
//   */
//
//  int i, j, k, l;
//  double dx;
//  if( n < 2 ){
//    ABORT("SPLINE REQUIRES N >= 2!", 1);
//    return;
//  }
//
//  for(l = 0; l < m; l++){
//    v[l] = 0.0;
//  }
//
//  for(l = 0; l < m; l++){
//    i = 0;
//    // for u[l] outside [min(x), max(x)], v[l] = 0.  No extrapolation is
//    // used.
//    if( (u[l] < x[0]) || (u[l] > x[n-1]) ){
//      v[l] = 0.0;
//    }
//    else{
//      /* calculate the index of u[l] */
//      if( u[l] >= x[0] ){
//	i = 0;
//	j = n;
//	while( j > i+1 ) {
//	  k = (i+j)/2;
//	  if( u[l] < x[k] ) j = k;
//	  if( u[l] >= x[k] ) i = k;
//	}
//      }
//      /* evaluate spline */
//      dx = u[l] - x[i];
//      v[l] = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]));
//    }
//  }
//  return;
//}


void seval(double* v, int m, double* u, int n, double* x, 
	   double* y, double* b, double* c, double* d){

  /* ***************************************************
   * This SPLINE function is designed specifically for the interpolation
   * part for pseudopotential generation in the electronic structure
   * calculation.  Therefore if u is outside the range [min(x), max(x)],
   * the corresponding v value will be an extrapolation.
   * ***************************************************
   
   this subroutine evaluates the  spline function

   seval = y(i) + b(i)*(u-x(i)) +  (i)*(u-x(i))**2 + d(i)*(u-x(i))**3

   where  x(i) .lt. u .lt. x(i+1), using horner's rule

   if  u .lt. x(1) then  i = 1  is used.
   if  u .ge. x(n) then  i = n  is used.

   input..

   m = the number of output data points
   n = the number of input data points
   u = the abs issa at which the spline is to be evaluated
   v = the value of the spline function at u
   x,y = the arrays of data abs issas and ordinates
   b,c,d = arrays of spline coefficients  omputed by spline

   if  u  is not in the same interval as the previous  all, then a
   binary sear h is performed to determine the proper interval.
   */

  int i, j, k, l;
  double dx;
  if( n < 2 ){
    ABORT("SPLINE REQUIRES N >= 2!", 1);
    return;
  }

  for(l = 0; l < m; l++){
    v[l] = 0.0;
  }

  for(l = 0; l < m; l++){
    i = 0;
    if( u[l] < x[0] ){
      i = 0;
    }
    else if( u[l] > x[n-1] ){
      i = n-1;
    }
    else{
      /* calculate the index of u[l] */
      i = 0;
      j = n;
      while( j > i+1 ) {
	k = (i+j)/2;
	if( u[l] < x[k] ) j = k;
	if( u[l] >= x[k] ) i = k;
      }
    }
    /* evaluate spline */
    dx = u[l] - x[i];
    v[l] = y[i] + dx*(b[i] + dx*(c[i] + dx*d[i]));
  }
  return;
}
