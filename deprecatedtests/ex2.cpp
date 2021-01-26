// *********************************************************************
// Test NumVec, NumMat and NumTns
// *********************************************************************
#include "scales.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"

using namespace scales;
using namespace std;

int main(int argc, char **argv) 
{
  try
  {
#ifndef _USE_COMPLEX_
    throw std::runtime_error("This test program require the usage of complex");
#endif
    PushCallStack("Testing NumVec");

    NumVec<Scalar>  v1(3);
    v1(0) = Complex(1.0, 2.0);
    v1(1) = Complex(0.0, 1.0);
    v1[2] = Complex(-1.0, 0.0);

    NumVec<Scalar>  v2( 3, true, v1.Data() );
    NumVec<Scalar>  v3( 3, false, v1.Data() );
    NumVec<Scalar>  v4( v1 );

    cout << "v1.m() = " << v1.m() << endl;
    cout << "v1 = " << v1 << endl;
    cout << "v2 = " << v2 << endl;
    cout << "v3 = " << v3 << endl;
    cout << "v4 = " << v4 << endl;

    cout << "v2[0] = v2[0] + (0.0, 1.0)" << endl;
    v2[0] = v2[0] + Complex(0.0, 1.0);
    cout << "v1 = " << v1 << endl;
    cout << "v2 = " << v2 << endl;

    cout << "v3[0] = v3[0] + (0.0, 1.0)" << endl;
    v3[0] = v3[0] + Complex(0.0, 1.0);
    cout << "v1 = " << v1 << endl;
    cout << "v3 = " << v3 << endl;

    cout << "Energy of v1 " << Energy(v1) << endl;

    PopCallStack();

    PushCallStack("Testing NumMat");
    NumMat<Scalar>   m1(3,2);
    SetValue( m1, Complex(1.0, 1.0) );

    cout << "m1 = " << endl << m1 << endl;

    NumMat<Scalar>   m2( 3, 2, true, m1.Data() );
    NumMat<Scalar>   m3( 3, 2, false, m1.Data() );
    NumMat<Scalar>   m4( m1 );

    m2(0,0) += Complex(1.0, 0.0);
    m3(0,0) += Complex(1.0, 0.0);


    cout << "m2 = " << endl << m2 << endl;
    cout << "m3 = " << endl << m3 << endl;
    cout << "m4 = " << endl << m4 << endl;
    cout << "m1 = " << endl << m1 << endl;
    cout << "Energy of m1 = " << Energy(m1) << endl;

    PopCallStack();

    PushCallStack("Testing NumTns");
    NumTns<Scalar>   t1(3,2,2);
    SetValue( t1, Complex(1.0, 1.0) );

    cout << "t1 = " << endl << t1 << endl;

    NumTns<Scalar>   t2( 3, 2, 2, true, t1.Data() );
    NumTns<Scalar>   t3( 3, 2, 2, false, t1.Data() );
    NumTns<Scalar>   t4( t1 );

    t2(0,0,0) += Complex(1.0, 0.0);
    t3(0,0,0) += Complex(1.0, 0.0);


    cout << "t2 = " << endl << t2 << endl;
    cout << "t3 = " << endl << t3 << endl;
    cout << "t4 = " << endl << t4 << endl;
    cout << "t1 = " << endl << t1 << endl;

    cout << "Energy of t1 " << Energy(t1) << endl;

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

  return 0;
}
