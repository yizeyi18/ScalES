#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include <stdlib.h>

#define NX 4
#define NY 4
int main()
{
        float *vx = (float*) malloc( NX * NY * sizeof(hipfftComplex));
        hipfftComplex *d_vx, *d_vx2;
     	int i;
    	for(i =0; i < NX *NY ; i++)
    		vx[i] = 1.0;
	assert(hipSetDevice(0) == hipSuccess);
	printf("NX NY sizoef(hipfftComplex): %d %d %d \n", NX, NY, sizeof(hipfftComplex));
        assert(hipMalloc(&d_vx,  NX*NY*sizeof(hipfftComplex)) == hipSuccess);
        assert(hipMalloc(&d_vx2, NX*NY*sizeof(hipfftComplex)) == hipSuccess);
        assert(hipMemcpy(d_vx, vx, NX*NY*sizeof(hipfftComplex), hipMemcpyHostToDevice) == hipSuccess);
    	for(i =0; i < NX *NY ; i++)
    		vx[i] = 0.0;
        cufftHandle planr2c;
        cufftHandle planc2r;
        (cufftPlan2d(&planr2c, NY, NX, CUFFT_R2C));
        (cufftPlan2d(&planc2r, NY, NX, CUFFT_C2R));
        //(cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_NATIVE));
        //(cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_NATIVE));
        assert(cufftExecR2C(planr2c, (cufftReal *)d_vx, d_vx2) == CUFFT_SUCCESS);
        //cufftExecC2R(planc2r, d_vx, (cufftReal *)d_vx2);
        assert(hipMemcpy(vx, d_vx2, NX*NY*sizeof(hipfftComplex), hipMemcpyDeviceToHost) == hipSuccess);
	for(i =0; i < NX *NY; i++)
		printf(" vx[%d]: %f \n" , i, vx[i]);
        hipFree(d_vx);
        hipFree(d_vx2);
        free(vx);
}
