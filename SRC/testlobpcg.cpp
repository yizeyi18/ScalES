/*Test of the new lobpcg interface

  Lin Lin  
  Revision: 5/12/2012
*/
  
#include "lobpcg.hpp"
FILE * fhstat;
int CONTXT;

BlopexInt Print(serial_Multi_Vector* X, serial_Multi_Vector* Y){
  Y->size = X->size;
  std::cout << Y->size << std::endl;
}

int main(int argc, char **argv) 
{  
  serial_Multi_Vector X, Y;
  X.size = 3;
  Y.size = 2;
  LOBPCG::MatMultiVecWrapper((void*)Print, (void*)&X, (void*)&Y);

  return 0;
}
