/// @file ex45.cpp
/// @brief Testing core dumper
/// @date 2016-11-06
#include "dgdft.hpp"
using namespace dgdft;
using namespace std;


int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  // FIXME
  //    int provided, threads_ok;
  //    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //    threads_ok = (provided >= MPI_THREAD_FUNNELED);
  //    std::cout << "threads_ok = " << threads_ok << std::endl;

  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  double timeSta, timeEnd;

  ErrorHandling("test");


  // Finalize 
  MPI_Finalize();

  return 0;
}
