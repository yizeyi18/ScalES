// *********************************************************************
// Test NumVec
// *********************************************************************
#include "dgdft.hpp"
#include "numvec_impl.hpp"

using namespace dgdft;
using namespace std;

int main(int argc, char **argv) 
{
	try
	{
#ifndef _USE_COMPLEX_
		throw std::runtime_error("This test program require the usage of complex");
#endif
		NumVec<Scalar>  v1(3);
		v1(0) = Complex(1.0, 2.0);
		v1(1) = Complex(0.0, 1.0);
		v1[2] = Complex(-1.0, 0.0);

		NumVec<Scalar>  v2( 3, true, v1.data() );
		NumVec<Scalar>  v3( 3, false, v1.data() );
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
		PushCallStack("asdf");

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
