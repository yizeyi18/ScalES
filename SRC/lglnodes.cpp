#include "util.hpp"

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

