#include "cuda_blas.hpp"
#include <iostream>


extern "C" {

  //------------------------------------------------------------------------//
  // Level 3 BLAS                                                           //
  //------------------------------------------------------------------------//
  void cuda_dgemm
    ( const char* transA, const char* transB,
      const Int* m, const Int* n, const Int* k,
      const double* alpha, const double* A, const Int* lda,
      const double* B, const Int* ldb,
      const double* beta,        double* C, const Int* ldc );
  void cuda_dgemm_batched 
    //( char transA, char transB, Int m, Int n, Int k,
    ( const char* transA, const char* transB,
      const Int* m, const Int* n, const Int* k,
      const double* alpha, const double* Aarr[], const Int* lda,
      const double* Barr[], const Int* ldb,
      const double* beta,        double* Carr[], const Int* ldc, int Bcount );
	//cublasOperation_t cublasOpFromChar(char op);


} // extern "C"

//----------------------------------------------------------------------------//
// Level 3 BLAS                                                               //
//----------------------------------------------------------------------------//
void cuda_dgemm 
( char transA, char transB,
  Int m, Int n, Int k, 
  double alpha, const double* A, Int lda, const double* B, Int ldb,
  double beta,        double* C, Int ldc)
{
    //:
		//1-Copy data to GPU (A, B, C)
		int size_A = m * k;
		int size_B = k * n;
		int size_C = m * n;
		double *d_A;
		double *d_B;
		double *d_C;
		cudaMalloc((void**)&d_A, size_A *sizeof(double));
		cudaMalloc((void**)&d_B, size_B *sizeof(double));
		cudaMalloc((void**)&d_C, size_C *sizeof(double));
		//cublasSetMatrix(
		cudaMemcpy(d_A, A, size_A * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, size_B * sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_C, C, size_C * sizeof(double), cudaMemcpyHostToDevice);
		cublasHandle_t handle;
		cublasCreate(&handle);

    //const char fixedTransA = ( transA == 'C' ? 'T' : transA );
    //const char fixedTransB = ( transB == 'C' ? 'T' : transB );
    const cublasOperation_t fixedTransA= ( transA == 'C' ? CUBLAS_OP_T : cublasOpFromChar(transA) ); //'T' : transA );
    const cublasOperation_t fixedTransB = ( transB == 'C' ? CUBLAS_OP_T : cublasOpFromChar(transB) );

		cublasDgemm(handle, fixedTransA, fixedTransB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
		cudaMemcpy(C, d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

}
  void cuda_dgemm_batched 
    ( char transA, char transB, Int m, Int n, Int k,
      double alpha, double* d_Aarr[], Int lda,
      double* d_Barr[], Int ldb,
      double beta,        double* d_Carr[], Int ldc, int Bcount )
{
	
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cout << "\nCUBLAS initialization failed\n";
		exit(1);
	}
  //1- Allocate and copy data from host to device Done in scf_dg TODO: Think of ways to move it here
  //2- call 

    const cublasOperation_t fixedTransA= ( transA == 'C' ? CUBLAS_OP_T : cublasOpFromChar(transA) ); //'T' : transA );
    const cublasOperation_t fixedTransB = ( transB == 'C' ? CUBLAS_OP_T : cublasOpFromChar(transB) );
  cublasDgemmBatched(handle, fixedTransA, fixedTransB, 
      m, n, k, &alpha, d_Aarr, lda, d_Barr, ldb, &beta, d_Carr, ldc, Bcount);
  //3- copy result back to host Now it's in the caller TODO: here
  cublasDestroy(handle);
}
	cublasOperation_t cublasOpFromChar(char op){
		switch (op) {
			case 'n':
			case 'N':
				return CUBLAS_OP_N;
			case 't':
			case 'T':
				return CUBLAS_OP_T;
			case 'c':
			case 'C':
				return CUBLAS_OP_C;
		}
	}


