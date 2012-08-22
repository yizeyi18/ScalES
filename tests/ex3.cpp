// *********************************************************************
// Test domain
// *********************************************************************
#include "dgdft.hpp"
#include "domain.hpp"

using namespace dgdft;
using namespace std;

int main(int argc, char **argv) 
{
	try
	{
#ifndef _USE_COMPLEX_
		throw std::runtime_error("This test program require the usage of complex");
#endif
		Domain dm1;
		Domain dm2;

		dm1.length = Point3( 1.0, 2.0, 3.0 );
		dm1.numGrid = Index3( 10, 20, 30 );
		dm1.typeGrid = LGL;
		dm1.posStart = Point3( 1.0, 0.1, 0.0 );
		dm2 = dm1;
		cout << dm2.length << endl;
		cout << dm2.numGrid << endl;
		cout << dm2.posStart << endl;

		cout << dm2.typeGrid << endl;
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
