#pragma once

#include <exception>
#include <cassert>
#include <string>

//#define CUDA_THROW_AS_ASSERT

#define CUDA_ASSERT(err)  { assert( err == cudaSuccess );           }
#define CUBLAS_ASSERT(err){ assert( err == CUBLAS_STATUS_SUCCESS ); }


#ifdef CUDA_THROW_AS_ASSERT
  #define CUDA_THROW(err) CUDA_ASSERT(err);
  #define CUBLAS_THROW(err) CUBLAS_ASSERT(err);
#else
  #define CUDA_THROW(err)  { if(err != cudaSuccess) throw cuda::exception( std::string(cudaGetErrorString(err)) + std::string(" @ Line: " + std::string(__FILE__)) + std::to_string(__LINE__) );           }
  #define CUBLAS_THROW(err){ if(err != CUBLAS_STATUS_SUCCESS) throw cublas::exception( std::string(cublasGetErrorString(err)) + std::string(" @ Line: " + std::string(__FILE__)) + std::to_string(__LINE__) ); } 
#endif

namespace cuda {

class exception : public std::exception {

  std::string message;

  virtual const char* what() const throw() {
    return message.c_str();
  }

public:

  exception( const char* msg ) : std::exception(), message( msg ) { };
  exception( std::string msg ) : std::exception(), message( msg ) { };

};

}

namespace cublas {

class exception : public std::exception {

  std::string message;

  virtual const char* what() const throw() {
    return message.c_str();
  }

public:

  exception( const char* msg ) : std::exception(), message( msg ) { };
  exception( std::string msg ) : std::exception(), message( msg ) { };

};

}
