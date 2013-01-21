#include "interp.hpp"
#include "util.hpp"

//--------------------------------
void xinterp(double* Vx2, double* Vx1, Buff& Mol2, ScfDG& Mol1)
{
  int N11 = Mol1._Ns[0]; int N21 = Mol1._Ns[1]; int N31 = Mol1._Ns[2];
  int N12 = Mol2._Ns[0]; int N22 = Mol2._Ns[1]; int N32 = Mol2._Ns[2];
  
  int shift1 = Mol2._posidx[0] - Mol1._posidx[0];
  int shift2 = Mol2._posidx[1] - Mol1._posidx[1];
  int shift3 = Mol2._posidx[2] - Mol1._posidx[2];

  int inew, jnew, knew;

  for (int k=0; k<N32; k++){
    knew = IMOD(k+shift3, N31);
    for (int j=0; j<N22; j++){
      jnew = IMOD(j+shift2, N21);
      for (int i=0; i<N12; i++){
	inew = IMOD(i+shift1, N11);
        Vx2[i+j*N12+k*N12*N22] \
          = Vx1[inew+jnew*N11+knew*N11*N21];
      }
    }
  }
  return;
}

//--------------------------------
void klinterp(double *Vl2, cpx* Vk1, Elem& Mol2, Buff& Mol1)
{
  char Nchar[] = "n";
  Index3 Ns1 = Mol1._Ns,  Ns2 = Mol2._Ns;


  vector<cpx> tmp1, tmp2, tmp3;
  vector<cpx> transtmp1, transtmp2, transtmp3;
    
 
  int mi, ki, ni;
  cpx alpha(1.0, 0.0), beta(0.0, 0.0);

  tmp1.resize(Ns2[0]*Ns1[1]*Ns1[2]);
  transtmp1.resize(Ns1[1]*Ns1[2]*Ns2[0]);
  tmp2.resize(Ns2[1]*Ns1[2]*Ns2[0]);
  transtmp2.resize(Ns1[2]*Ns2[0]*Ns2[1]);
  tmp3.resize(Ns2[2]*Ns2[0]*Ns2[1]);
  transtmp3.resize(Ns2[0]*Ns2[1]*Ns2[2]);

  mi = Ns2[0]; ki = Ns1[0]; ni = Ns1[1]*Ns1[2];
  zgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol2._TransBufkl[0][0], 
	 &mi, &Vk1[0], &ki, &beta, &tmp1[0], &mi);
  
  Transpose<cpx>(tmp1, transtmp1, Ns2[0], Ns1[1]*Ns1[2]);
  
  mi = Ns2[1]; ki = Ns1[1]; ni = Ns1[2]*Ns2[0]; 
  zgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol2._TransBufkl[1][0], 
	 &mi, &transtmp1[0], &ki, &beta, &tmp2[0], &mi);
 
  Transpose<cpx>(tmp2, transtmp2, Ns2[1], Ns1[2]*Ns2[0]);

  mi = Ns2[2]; ki = Ns1[2]; ni = Ns2[0]*Ns2[1]; 
  zgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol2._TransBufkl[2][0], 
	 &mi, &transtmp2[0], &ki, &beta, &tmp3[0], &mi);
  
  Transpose<cpx>(tmp3, transtmp3, Ns2[2], Ns2[0]*Ns2[1]);
 
  /* Get rid of the extra phase factor */
  double maxabs = abs(transtmp3[0]);
  int imaxabs = 0;
  for(int i = 1; i < Ns2[0]*Ns2[1]*Ns2[2]; i++){
    if( abs(transtmp3[i]) > maxabs ){
      maxabs = abs(transtmp3[i]);
      imaxabs = i;
    }
  }
  
  cpx phase;
  phase = transtmp3[imaxabs] / maxabs;
  if( phase.real() < 0 ) phase = -phase;

  for(int i = 0; i < Ns2[0]*Ns2[1]*Ns2[2]; i++){
    transtmp3[i] = transtmp3[i] / phase;
  }

  for(int i = 0; i < Ns2[0]*Ns2[1]*Ns2[2]; i++){
    Vl2[i] = transtmp3[i].real() / (Ns1[0]*Ns1[1]*Ns1[2]);
  }
  return;
}

//--------------------------------
void xlinterp(double* Vl2, double* Vx1, Elem& Mol2, Buff& Mol1)
{
  // Old code without forcing alignment
  if(0){
    vector<cpx> Vk1;
    vector<cpx> fftwin; 
    int NTOT1 = Mol1._ntot, NTOT2 = Mol2._ntot;
    Index3 Ns1 = Mol1._Ns,  Ns2 = Mol2._Ns;

    Vk1.resize(NTOT1);
    fftwin.resize(NTOT1);

    /* Different from previous version.  No division of ntot here, by
     * add it in to klinterp for consistency */
    for(int i = 0; i < NTOT1; i++){
      fftwin[i] = cpx(Vx1[i], 0.0);
    }
    fftw_execute_dft(Mol1._planpsiforward, 
	reinterpret_cast<fftw_complex*>(&fftwin[0]), 
	reinterpret_cast<fftw_complex*>(&Vk1[0]));

    klinterp(&Vl2[0], (&Vk1[0]), Mol2, Mol1);
    return;
  }
  //
  // New code forcing alignment
  if(1){
    fftw_complex *Vk1, *fftwin;

    int NTOT1 = Mol1._ntot, NTOT2 = Mol2._ntot;
    Index3 Ns1 = Mol1._Ns,  Ns2 = Mol2._Ns;

    Vk1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NTOT1);
    fftwin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NTOT1);

    /* Different from previous version.  No division of ntot here, by
     * add it in to klinterp for consistency */
    for(int i = 0; i < NTOT1; i++){
      fftwin[i][0] = Vx1[i];
      fftwin[i][1] = 0.0;
    }
    fftw_execute_dft(Mol1._planpsiforward, (&fftwin[0]), (&Vk1[0])); 

    klinterp(&Vl2[0], (cpx*)(&Vk1[0]), Mol2, Mol1);
    
    fftw_free(Vk1);
    fftw_free(fftwin);
    
    return;
  }
  
}

//--------------------------------
void lxinterp(double *Vx2, double *Vl1, ScfDG& Mol2, Elem& Mol1)
{
  /* This function is only used for interpolating rho from element to
   * global molecule */
  vector<double> tmp1, tmp2, tmp3;
  vector<double> transtmp1, transtmp2, transtmp3;
  char Nchar[] = "n";
  Index3 Ns1 = Mol1._Ns,  Ns2 = Mol1._Nsglb;

  int mi, ki, ni;
  double alpha(1.0), beta(0.0);

  tmp1.resize(Ns2[0]*Ns1[1]*Ns1[2]);
  transtmp1.resize(Ns1[1]*Ns1[2]*Ns2[0]);
  tmp2.resize(Ns2[1]*Ns1[2]*Ns2[0]);
  transtmp2.resize(Ns1[2]*Ns2[0]*Ns2[1]);
  tmp3.resize(Ns2[2]*Ns2[0]*Ns2[1]);
  transtmp3.resize(Ns2[0]*Ns2[1]*Ns2[2]);
  
  mi = Ns2[0]; ki = Ns1[0]; ni = Ns1[1]*Ns1[2];

  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[0][0], 
	 &mi, &Vl1[0], &ki, &beta, &tmp1[0], &mi);
  
  Transpose<double>(tmp1, transtmp1, Ns2[0], Ns1[1]*Ns1[2]);
  
  mi = Ns2[1]; ki = Ns1[1]; ni = Ns1[2]*Ns2[0]; 
  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[1][0], 
	 &mi, &transtmp1[0], &ki, &beta, &tmp2[0], &mi);
 
  Transpose<double>(tmp2, transtmp2, Ns2[1], Ns1[2]*Ns2[0]);

  mi = Ns2[2]; ki = Ns1[2]; ni = Ns2[0]*Ns2[1]; 
  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[2][0], 
	 &mi, &transtmp2[0], &ki, &beta, &tmp3[0], &mi);
  
  Transpose<double>(tmp3, transtmp3, Ns2[2], Ns2[0]*Ns2[1]);

  int i1, j1, k1;
  Index3 Nsglb = Mol2._Ns;
  for(int k = 0; k < Ns2[2]; k++){
    k1 = Mol1._posidx[2]+ k; 
    for(int j = 0; j < Ns2[1]; j++){
      j1 = Mol1._posidx[1] + j; 
      for(int i = 0; i < Ns2[0]; i++){
	i1 = Mol1._posidx[0] + i;
        Vx2[i1+j1*Nsglb[0]+k1*Nsglb[0]*Nsglb[1]] =  
	  transtmp3[i+j*Ns2[0]+k*Ns2[0]*Ns2[1]];
      }
    }
  }
  return;
}

//--------------------------------
void lxinterp_local(double *Vx2, double *Vl1, Elem& Mol1)
{
  /* This function is only used for interpolating rho from element to
   * global molecule */
  vector<double> tmp1, tmp2, tmp3;
  vector<double> transtmp1, transtmp2, transtmp3;
  char Nchar[] = "n";
  Index3 Ns1 = Mol1._Ns,  Ns2 = Mol1._Nsglb;

  int mi, ki, ni;
  double alpha(1.0), beta(0.0);

  tmp1.resize(Ns2[0]*Ns1[1]*Ns1[2]);
  transtmp1.resize(Ns1[1]*Ns1[2]*Ns2[0]);
  tmp2.resize(Ns2[1]*Ns1[2]*Ns2[0]);
  transtmp2.resize(Ns1[2]*Ns2[0]*Ns2[1]);
  tmp3.resize(Ns2[2]*Ns2[0]*Ns2[1]);
  transtmp3.resize(Ns2[0]*Ns2[1]*Ns2[2]);

  mi = Ns2[0]; ki = Ns1[0]; ni = Ns1[1]*Ns1[2];

  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[0][0], 
	 &mi, &Vl1[0], &ki, &beta, &tmp1[0], &mi);
  
  Transpose<double>(tmp1, transtmp1, Ns2[0], Ns1[1]*Ns1[2]);
  
  mi = Ns2[1]; ki = Ns1[1]; ni = Ns1[2]*Ns2[0]; 
  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[1][0], 
	 &mi, &transtmp1[0], &ki, &beta, &tmp2[0], &mi);
 
  Transpose<double>(tmp2, transtmp2, Ns2[1], Ns1[2]*Ns2[0]);

  mi = Ns2[2]; ki = Ns1[2]; ni = Ns2[0]*Ns2[1]; 
  dgemm_(Nchar, Nchar, &mi, &ni, &ki, &alpha, &Mol1._TransGlblx[2][0], 
	 &mi, &transtmp2[0], &ki, &beta, &tmp3[0], &mi);
  
  Transpose<double>(tmp3, transtmp3, Ns2[2], Ns2[0]*Ns2[1]);

  for(int i = 0; i < Ns2[0]*Ns2[1]*Ns2[2]; i++){
    Vx2[i] = transtmp3[i];
  }
  return;
}



//--------------------------------
void DiffPsi(Index3 Ns, Point3 Ls, double* psi, double* diffxpsi, double* diffypsi, double* diffzpsi)
{
	int mi, ni, ki;
	int ndim = 3;
	//int ntot = molelem._ntot;
	//Index3 Ns = molelem._Ns;
	//Point3 Ls = molelem._Ls;
	int ntot = Ns(0)*Ns(1)*Ns(2);

	/* Calculate Trans matrices for Buffer to element kl-interpolation */
	vector<double> lglmesh;
	vector<vector<double> > Dp;
	double alpha = 1.0, beta = 0.0;
	vector<double> tmp1, tmp2;
	char Nchar = 'n';

	tmp1.resize(ntot);
	tmp2.resize(ntot);

	Dp.resize(ndim);
	for(int d = 0; d < ndim; d++){
		/* Calculate lgl mesh at element level*/
		lglmesh.resize(Ns[d]);
		Dp[d].resize(Ns[d]*Ns[d]);
		lglnodes(lglmesh, Dp[d], Ns[d]-1);

		for (int j=0; j<Ns[d]; j++){
			for (int i=0; i<Ns[d]; i++){
				Dp[d][j*Ns[d]+i] *= 2.0/Ls[d];
			}
		}
	}

	double *enrichptr, *dxenrichptr, *dyenrichptr, *dzenrichptr;
	enrichptr   = &psi[0];
	dxenrichptr = &diffxpsi[0];
	dyenrichptr = &diffypsi[0];
	dzenrichptr = &diffzpsi[0];

	mi = Ns[0]; ni = Ns[1]*Ns[2]; ki = Ns[0];
	dgemm_(&Nchar, &Nchar, &mi, &ni, &ki, &alpha, &Dp[0][0], 
				 &mi, &enrichptr[0], &ki, &beta, &dxenrichptr[0], &mi);

	Transpose<double>(&enrichptr[0], &tmp1[0], Ns[0], Ns[1]*Ns[2]);
	mi = Ns[1]; ni = Ns[2]*Ns[0]; ki = Ns[1];
	dgemm_(&Nchar, &Nchar, &mi, &ni, &ki, &alpha, &Dp[1][0], 
				 &mi, &tmp1[0], &ki, &beta, &tmp2[0], &mi);
	Transpose<double>(&tmp2[0], &dyenrichptr[0], Ns[1]*Ns[2], Ns[0]);

	Transpose<double>(&enrichptr[0], &tmp1[0], Ns[0]*Ns[1], Ns[2]);
	mi = Ns[2]; ni = Ns[0]*Ns[1]; ki = Ns[2];
	dgemm_(&Nchar, &Nchar, &mi, &ni, &ki, &alpha, &Dp[2][0],
				 &mi, &tmp1[0], &ki, &beta, &tmp2[0], &mi);
	Transpose<double>(&tmp2[0], &dzenrichptr[0], Ns[2], Ns[0]*Ns[1]);

	return;
}


//--------------------------------
bool CheckInterval(const Point3& r, const Point3& posstart, const Point3& Lsbuf, const Point3& Lsglb)
{
  bool is_in = true;
  int ndim = 3;
  Point3 shiftstart;
  Point3 shiftr;
  for(int i = 0; i < ndim; i++){
    shiftstart[i] = DMOD(posstart[i], Lsglb[i]);
    shiftr[i]     = DMOD(r[i], Lsglb[i]);
    /* Case 1 of the buffer interval */
    if( shiftstart[i] + Lsbuf[i] > Lsglb[i] ){
      if( (shiftr[i] > shiftstart[i] + Lsbuf[i] - Lsglb[i]) &&
	  (shiftr[i] < shiftstart[i]) ){
	is_in = false;
      }
    }
    /* Case 2 of the buffer interval */
    else{
      if( (shiftr[i] < shiftstart[i]) ||
	  (shiftr[i] > shiftstart[i] + Lsbuf[i]) ){
	is_in = false;
      }
    }
  }
  return is_in;
}


//-----------------------------------
void XScaleByY(double* x, double* y, int ntot)
{
  double *xptr, *yptr;
  xptr = x;  yptr = y;
  for(int i = 0; i < ntot; i++){
    (*xptr) *= (*yptr);
    xptr++; yptr++;
  }
}


//-----------------------------------
// Real space interpolation for dual grid calculation in the extended
// element.  Relevant only when _bufdual option is 1.
void xinterp_dual(double* Vx2, double* Vx1, Buff& Mol2, ScfDG& Mol1)
{
  int N11 = Mol1._Ns[0]; int N21 = Mol1._Ns[1]; int N31 = Mol1._Ns[2];
  int N12 = Mol2._Ns[0]; int N22 = Mol2._Ns[1]; int N32 = Mol2._Ns[2];
  
  int shift1 = Mol2._posidx[0] - Mol1._posidx[0];
  int shift2 = Mol2._posidx[1] - Mol1._posidx[1];
  int shift3 = Mol2._posidx[2] - Mol1._posidx[2];

  int inew, jnew, knew;

  for (int k=0; k<N32; k++){
    knew = IMOD(2*k+shift3, N31);
    for (int j=0; j<N22; j++){
      jnew = IMOD(2*j+shift2, N21);
      for (int i=0; i<N12; i++){
	inew = IMOD(2*i+shift1, N11);
        Vx2[i+j*N12+k*N12*N22] \
          = Vx1[inew+jnew*N11+knew*N11*N21];
      }
    }
  }
  return;
}
