#include "utility.hpp"

// TODO Change int->Int, double->Real etc.
// TODO Error handlding
// FIXME using namespace
using namespace std;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::cerr;

namespace dgdft{

// *********************************************************************
// Spline functions
// *********************************************************************

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
		throw  std::logic_error(" SPLINE REQUIRES N >= 2!" );
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
		throw  std::logic_error(" SPLINE REQUIRES N >= 2!" );
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

// *********************************************************************
// Generating grids in a domain
// *********************************************************************

void lglnodes(double* x, int N)
{
  int i, j, k;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  vector<double> xold;
  int N1 = N + 1;
  vector<double> P;

  xold.resize(N1);
  P.resize(N1*N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }
   
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  return;  
}

void lglnodes(double* x, double* w, int N)
{
  int i, j, k;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  vector<double> xold;
  int N1 = N + 1;
  vector<double> P;

  xold.resize(N1);
  P.resize(N1*N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }
   
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2/(N*N1*P[N*N1+i]*P[N*N1+i]);
  }
}

void lglnodes(vector<double>& x, int N)
{
  int i, j, k;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  vector<double> xold;
  int N1 = N + 1;
  vector<double> P;

  xold.resize(N1);
  P.resize(N1*N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }
   
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  return;  
}

void lglnodes(vector<double>& x, vector<double>& w, vector<double>& P, int N)
{
  int i, j, k;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  vector<double> xold;
  int N1 = N + 1;

  xold.resize(N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }
   
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  for (i=0; i<N1; i++){
    w[i] = 2/(N*N1*P[N*N1+i]*P[N*N1+i]);
  }
}

void lglnodes(vector<double>& x, vector<double>& D, int N)
{
  int i, j, k;
  double pi = 4.0 * atan(1.0);
  double err, tol = 1e-15;
  vector<double> xold;
  int N1 = N + 1;
  vector<double> P;

  xold.resize(N1);
  P.resize(N1*N1);

  for (i=0; i<N1; i++){
    x[i] = cos(pi*(N1-i-1)/(double)N);
  }
   
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      P[j*N1+i] = 0;
    }
  }

  do{
    for (i=0; i<N1; i++){
      xold[i] = x[i]; 
      P[i] = 1; 
      P[N1+i] = x[i];
    }
    for (j=2; j<N1; j++){
      for (i=0; i<N1; i++){
        P[j*N1+i] = ((2*j-1)*x[i]*P[(j-1)*N1+i] - (j-1)*P[(j-2)*N1+i])/j;
      }
    }

    for (i=0; i<N1; i++){
      x[i] = xold[i] - (x[i]*P[N*N1+i] - P[(N-1)*N1+i])/(N1*P[N*N1+i]);
    }

    err = 0;
    for (i=0; i<N1; i++){
      if (err < fabs(xold[i] - x[i])){
        err = fabs(xold[i] - x[i]);
      }
    }
  } while(err>tol);

  
  for (j=0; j<N1; j++){
    for (i=0; i<N1; i++){
      if (i!=j) {
        D[j*N1+i] = P[N*N1+i]/P[N*N1+j]/(x[i] - x[j]);
      }
      else if (i==0){
        D[j*N1+i] = - N*N1/4.;
      }
      else if (i==N1-1){
        D[j*N1+i] = N*N1/4;
      }
      else{
        D[j*N1+i] = 0;      
      }
    }
  }


}


void
UniformMesh ( const Domain &dm, std::vector<DblNumVec> &gridpos )
{
#ifndef _RELEASE_
	PushCallStack("UniformMesh");
#endif
  gridpos.resize(DIM);
  for (int d=0; d<DIM; d++) {
    gridpos[d].Resize(dm.numGrid[d]);
    double h = dm.length[d] / dm.numGrid[d];
    for (int i=0; i < dm.numGrid[d]; i++) {
      gridpos[d](i) = dm.posStart[d] + double(i)*h;
    }
  }
#ifndef _RELEASE_
	PopCallStack();
#endif

	return ;
}		// -----  end of function UniformMesh  ----- 



// *********************************************************************
// IO functions
// *********************************************************************
//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);  //cerr<<filename<<endl;
  ifstream fin(filename);
  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os)
{
   MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  ofstream fout(filename);
  fout<<os.str();
  fout.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SharedRead(std::string name, std::istringstream& is)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  vector<char> tmpstr;
  if(mpirank==0) {
    ifstream fin(name.c_str());
    //std::string str(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    //tmpstr.insert(tmpstr.end(), str.begin(), str.end());
    tmpstr.insert(tmpstr.end(), std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    fin.close();
    int size = tmpstr.size();	//cerr<<size<<endl;
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    int size;
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tmpstr.resize(size);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  is.str( std::string(tmpstr.begin(), tmpstr.end()) );
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SharedWrite(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  if(mpirank==0) {
    ofstream fout(name.c_str());
    fout<<os.str();
    fout.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


//---------------------------------------------------------
Int SeparateWriteAscii(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  ofstream fout(filename, ios::trunc);
  fout<<os;
  fout.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

}  // namespace dgdft
