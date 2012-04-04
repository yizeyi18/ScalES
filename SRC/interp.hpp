#ifndef _INTERP_HPP_
#define _INTERP_HPP_

#include "scfdg.hpp"

void lglnodes(vector<double>& x, int N);

//buffer <= global
void xinterp(double* Vx2, double* Vx1, Buff& Mol2, ScfDG& Mol1);

//buffer <= global, relevant when _bufdual = 1
void xinterp_dual(double* Vx2, double* Vx1, Buff& Mol2, ScfDG& Mol1);

void klinterp(double* Vl2, cpx* Vk1, Elem& Mol2, Buff& Mol1);

//element <= buffer (use klinterp)
void xlinterp(double* Vl2, double* Vx1, Elem& Mol2, Buff& Mol1);

//global <= element (not used anymore)
void lxinterp(double *Vx2, double *Vl1, ScfDG& Mol2, Elem& Mol1);

//global(within emement) <= element
void lxinterp_local(double *Vx2, double *Vl1, Elem& Mol1);

void DiffPsi(Index3 Ns, Point3 hs, double* psi, double* diffxpsi, double* diffypsi, double* diffzpsi);

bool CheckInterval(const Point3& r, const Point3& posstart, const Point3& Lsbuf, const Point3& Lsglb);

void XScaleByY(double* x, double* y, int ntot);

#endif
