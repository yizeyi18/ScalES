// *********************************************************************
// Test the error handling of PETSc with throw/catch
// *********************************************************************
#include "dgdft.hpp"

using namespace dgdft;
using namespace std;

static char help[] = "Test the error handling of PETSc with throw/catch";

int main(int argc, char **argv) 
{

  try
  {
    PushCallStack("Main");
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
    Vec v;
    ierr = VecDestroy( &v );
    if( ierr ) throw ierr;
    ierr = PetscFinalize();
    PopCallStack();
  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
#ifndef _RELEASE_
    DumpCallStack();
#endif
  }
  catch ( PetscErrorCode ierr ){
    std::cerr << "Caught PETSc error message " << std::endl;
    DumpCallStack();
    CHKERRQ(ierr);
  }

  return 0;
}
