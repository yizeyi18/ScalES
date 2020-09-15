#ifdef DEVICE
#include "device_error.hpp"

char *deviceBLASGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "DEVICE_BLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "DEVICE_BLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "DEVICE_BLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "DEVICE_BLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "DEVICE_BLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "DEVICE_BLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "DEVICE_BLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "DEVICE_BLAS_STATUS_INTERNAL_ERROR";

	default:
    	    return "<unknown>";
    }
}

// returns string for CUFFT API error
char *deviceFFTGetErrorString(cufftResult error)
{
    switch (error)
    {
	case CUFFT_SUCCESS:
            return "DEVICE_FFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "DEVICE_FFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "DEVICE_FFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "DEVICE_FFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "DEVICE_FFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "DEVICE_FFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "DEVICE_FFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "DEVICE_FFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "DEVICE_FFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "DEVICE_FFT_UNALIGNED_DATA";

	default:
    	    return "<unknown>";
    }
}

#endif
