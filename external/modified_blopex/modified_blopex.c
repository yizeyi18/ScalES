/*
       This file implements a wrapper to the BLOPEX solver

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/


// *********************************************************************
// This modification allows one to directly apply preconditioner, rather
// than going through the KSP approach.
//
// This modification uses the matrix obtained from STPrecondSetMatForPC,
// rather than  using PETSc by calling STAssociatedKSPSolve.
//
//
// Lin Lin
// 8/22/2012
// *********************************************************************

#include <slepc-private/epsimpl.h>      /*I "slepceps.h" I*/
#include "slepc-interface.h"
#include <blopex_lobpcg.h>
#include <blopex_interpreter.h>
#include <blopex_multivector.h>
#include <blopex_temp_multivector.h>

PetscErrorCode EPSSolve_MODIFIED_BLOPEX(EPS);

typedef struct {
  lobpcg_Tolerance           tol;
  lobpcg_BLASLAPACKFunctions blap_fn;
  mv_MultiVectorPtr          eigenvectors;
  mv_MultiVectorPtr          Y;
  mv_InterfaceInterpreter    ii;
  ST                         st;
  Vec                        w;
} EPS_MODIFIED_BLOPEX;

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnSingleVector"
static void Precond_FnSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  EPS_MODIFIED_BLOPEX     *blopex = (EPS_MODIFIED_BLOPEX*)eps->data;
      
  PetscFunctionBegin;
	// LLIN: all the modification is here.
	Mat    P;
	ierr = STPrecondGetMatForPC( blopex->st, &P ); 
	ierr = MatMult( P, (Vec)x, (Vec)y ); 
  // ierr = STAssociatedKSPSolve(blopex->st,(Vec)x,(Vec)y); CHKERRABORT(((PetscObject)eps)->comm,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "Precond_FnMultiVector"
static void Precond_FnMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_MODIFIED_BLOPEX *blopex = (EPS_MODIFIED_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(Precond_FnSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorASingleVector"
static void OperatorASingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  EPS_MODIFIED_BLOPEX     *blopex = (EPS_MODIFIED_BLOPEX*)eps->data;
  Mat            A,B;
  PetscScalar    sigma;
 
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  ierr = MatMult(A,(Vec)x,(Vec)y);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  ierr = STGetShift(eps->OP,&sigma);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  if (sigma != 0.0) {
    if (B) { ierr = MatMult(B,(Vec)x,blopex->w);CHKERRABORT(((PetscObject)eps)->comm,ierr); }
    else { ierr = VecCopy((Vec)x,blopex->w);CHKERRABORT(((PetscObject)eps)->comm,ierr); }
    ierr = VecAXPY((Vec)y,-sigma,blopex->w);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorAMultiVector"
static void OperatorAMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_MODIFIED_BLOPEX *blopex = (EPS_MODIFIED_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorASingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorBSingleVector"
static void OperatorBSingleVector(void *data,void *x,void *y)
{
  PetscErrorCode ierr;
  EPS            eps = (EPS)data;
  Mat            B;
  
  PetscFunctionBegin;
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  ierr = MatMult(B,(Vec)x,(Vec)y);CHKERRABORT(((PetscObject)eps)->comm,ierr);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "OperatorBMultiVector"
static void OperatorBMultiVector(void *data,void *x,void *y)
{
  EPS        eps = (EPS)data;
  EPS_MODIFIED_BLOPEX *blopex = (EPS_MODIFIED_BLOPEX*)eps->data;

  PetscFunctionBegin;
  blopex->ii.Eval(OperatorBSingleVector,data,x,y);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_MODIFIED_BLOPEX"
PetscErrorCode EPSSetUp_MODIFIED_BLOPEX(EPS eps)
{
#if defined(PETSC_MISSING_LAPACK_POTRF) || defined(PETSC_MISSING_LAPACK_SYGV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF/SYGV - Lapack routine is unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i;
  EPS_MODIFIED_BLOPEX     *blopex = (EPS_MODIFIED_BLOPEX *)eps->data;
  PetscBool      isPrecond;

  PetscFunctionBegin;
  if (!eps->ishermitian) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"blopex only works for hermitian problems"); 
  if (!eps->which) eps->which = EPS_SMALLEST_REAL;
  if (eps->which!=EPS_SMALLEST_REAL) SETERRQ(((PetscObject)eps)->comm,1,"Wrong value of eps->which");

  /* Change the default sigma to inf if necessary */
  if (eps->which == EPS_LARGEST_MAGNITUDE || eps->which == EPS_LARGEST_REAL ||
      eps->which == EPS_LARGEST_IMAGINARY) {
    ierr = STSetDefaultShift(eps->OP,3e300);CHKERRQ(ierr);
  }

  ierr = STSetUp(eps->OP);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)eps->OP,STPRECOND,&isPrecond);CHKERRQ(ierr);
  if (!isPrecond) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"blopex only works with STPRECOND");
  blopex->st = eps->OP;

  eps->ncv = eps->nev = PetscMin(eps->nev,eps->n);
  if (eps->mpd) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (!eps->max_it) eps->max_it = PetscMax(100,2*eps->n/eps->ncv);
  if (eps->arbit_func) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Arbitrary selection of eigenpairs not supported in this solver");

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,1);CHKERRQ(ierr);
  
  blopex->tol.absolute = eps->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:eps->tol;
  blopex->tol.relative = 1e-50;
  
  SLEPCSetupInterpreter(&blopex->ii);
  blopex->eigenvectors = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->ncv,eps->V);
  for (i=0;i<eps->ncv;i++) { ierr = PetscObjectReference((PetscObject)eps->V[i]);CHKERRQ(ierr); }
  mv_MultiVectorSetRandom(blopex->eigenvectors,1234);
  ierr = VecDuplicate(eps->V[0],&blopex->w);CHKERRQ(ierr);

  if (eps->nds > 0) {
    blopex->Y = mv_MultiVectorCreateFromSampleVector(&blopex->ii,eps->nds,eps->defl);
    for (i=0;i<eps->nds;i++) { ierr = PetscObjectReference((PetscObject)eps->defl[i]);CHKERRQ(ierr); }
  } else
    blopex->Y = PETSC_NULL;

#if defined(PETSC_USE_COMPLEX)
  blopex->blap_fn.zpotrf = PETSC_zpotrf_interface;
  blopex->blap_fn.zhegv = PETSC_zsygv_interface;
#else
  blopex->blap_fn.dpotrf = PETSC_dpotrf_interface;
  blopex->blap_fn.dsygv = PETSC_dsygv_interface;
#endif

  if (eps->extraction) { ierr = PetscInfo(eps,"Warning: extraction type ignored\n");CHKERRQ(ierr); }

  /* dispatch solve method */
  if (eps->leftvecs) SETERRQ(((PetscObject)eps)->comm,PETSC_ERR_SUP,"Left vectors not supported in this solver");
  eps->ops->solve = EPSSolve_MODIFIED_BLOPEX;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_MODIFIED_BLOPEX"
PetscErrorCode EPSSolve_MODIFIED_BLOPEX(EPS eps)
{
  EPS_MODIFIED_BLOPEX     *blopex = (EPS_MODIFIED_BLOPEX *)eps->data;
  PetscScalar    sigma;
  int            i,j,info,its,nconv;
  double         *residhist=PETSC_NULL;
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  komplex        *lambdahist=PETSC_NULL;
#else
  double         *lambdahist=PETSC_NULL;
#endif
  
  PetscFunctionBegin;
  if (eps->numbermonitors>0) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(eps->ncv*(eps->max_it+1)*sizeof(komplex),&lambdahist);CHKERRQ(ierr);
#else
    ierr = PetscMalloc(eps->ncv*(eps->max_it+1)*sizeof(double),&lambdahist);CHKERRQ(ierr);
#endif
    ierr = PetscMalloc(eps->ncv*(eps->max_it+1)*sizeof(double),&residhist);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_COMPLEX)
  info = lobpcg_solve_complex(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:PETSC_NULL,eps->isgeneralized?OperatorBMultiVector:PETSC_NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        (komplex*)eps->eigr,lambdahist,eps->ncv,eps->errest,residhist,eps->ncv);
#else
  info = lobpcg_solve_double(blopex->eigenvectors,eps,OperatorAMultiVector,
        eps->isgeneralized?eps:PETSC_NULL,eps->isgeneralized?OperatorBMultiVector:PETSC_NULL,
        eps,Precond_FnMultiVector,blopex->Y,
        blopex->blap_fn,blopex->tol,eps->max_it,0,&its,
        eps->eigr,lambdahist,eps->ncv,eps->errest,residhist,eps->ncv);
#endif
  if (info>0) SETERRQ1(((PetscObject)eps)->comm,PETSC_ERR_LIB,"Error in blopex (code=%d)",info); 

  if (eps->numbermonitors>0) {
    for (i=0;i<its;i++) {
      nconv = 0;
      for (j=0;j<eps->ncv;j++) { if (residhist[j+i*eps->ncv]>eps->tol) break; else nconv++; }
      ierr = EPSMonitor(eps,i,nconv,(PetscScalar*)lambdahist+i*eps->ncv,eps->eigi,residhist+i*eps->ncv,eps->ncv);CHKERRQ(ierr);
    }
    ierr = PetscFree(lambdahist);CHKERRQ(ierr); 
    ierr = PetscFree(residhist);CHKERRQ(ierr); 
  }

  eps->its = its;
  eps->nconv = eps->ncv;
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);
  if (sigma != 0.0) {
    for (i=0;i<eps->nconv;i++) eps->eigr[i]+=sigma;
  }
  if (info==-1) eps->reason = EPS_DIVERGED_ITS;
  else eps->reason = EPS_CONVERGED_TOL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReset_MODIFIED_BLOPEX"
PetscErrorCode EPSReset_MODIFIED_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;
  EPS_MODIFIED_BLOPEX     *blopex = (EPS_MODIFIED_BLOPEX *)eps->data;

  PetscFunctionBegin;
  mv_MultiVectorDestroy(blopex->eigenvectors);
  mv_MultiVectorDestroy(blopex->Y);
  ierr = VecDestroy(&blopex->w);CHKERRQ(ierr);
  ierr = EPSReset_Default(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy_MODIFIED_BLOPEX"
PetscErrorCode EPSDestroy_MODIFIED_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LOBPCG_DestroyRandomContext();
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetFromOptions_MODIFIED_BLOPEX"
PetscErrorCode EPSSetFromOptions_MODIFIED_BLOPEX(EPS eps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("EPS MODIFIED_BLOPEX Options");CHKERRQ(ierr);
  LOBPCG_SetFromOptionsRandomContext();
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_MODIFIED_BLOPEX"
PetscErrorCode EPSCreate_MODIFIED_BLOPEX(EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,EPS_MODIFIED_BLOPEX,&eps->data);CHKERRQ(ierr);
  eps->ops->setup                = EPSSetUp_MODIFIED_BLOPEX;
  eps->ops->setfromoptions       = EPSSetFromOptions_MODIFIED_BLOPEX;
  eps->ops->destroy              = EPSDestroy_MODIFIED_BLOPEX;
  eps->ops->reset                = EPSReset_MODIFIED_BLOPEX;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->ops->computevectors       = EPSComputeVectors_Default;
  ierr = STSetType(eps->OP,STPRECOND);CHKERRQ(ierr);
  ierr = STPrecondSetKSPHasMat(eps->OP,PETSC_TRUE);CHKERRQ(ierr);
  LOBPCG_InitRandomContext(((PetscObject)eps)->comm,eps->rand);
  PetscFunctionReturn(0);
}
EXTERN_C_END
