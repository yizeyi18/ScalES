// *********************************************************************
// Test Tiny vectors
// *********************************************************************
#include "dgdft.hpp"
#include "tinyvec_impl.hpp"

using namespace dgdft;
using namespace std;

int main(int argc, char **argv) 
{
  try
  {
    Point3  p1(1.0, 2.0, 3.0);
    Point3  p2( p1.Data() );
    Point3  p3( p1[0], p1[1], p1[2] );
    Point3  p4( p1 );
    Point3  p5; p5 = p1;

    cout << "p1 = " << p1 << endl;
    cout << "p2 = " << p2 << endl;
    cout << "p3 = " << p3 << endl;
    cout << "p4 = " << p4 << endl;
    cout << "p5 = " << p5 << endl;

    cout << "p1 + p2 = " << p1 + p2 << endl;
    cout << "p1 - p2 = " << p1 - p2 << endl;
    cout << "p1 * 2.0 = " << p1 * 2.0 << endl;
    cout << "p1 / 2.0 = " << p1 / 2.0 << endl;

    cout << "p1.l1() = " << p1.l1() << endl;
    cout << "p1.linfty() = " << p1.linfty() << endl;
    cout << "p1.l2() = " << p1.l2() << endl;

  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
#ifndef _RELEASE_
    DumpCallStack();
#endif
  }

  return 0;
}
