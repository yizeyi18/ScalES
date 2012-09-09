#include "dgdft.hpp"
#include "tinyvec_impl.hpp"
#include "numvec_impl.hpp"
//#include "domain.hpp"

typedef std::complex<double> cpx;
using namespace dgdft;

int main(int argc, char **argv) 
{
  fftw_plan plan;
  std::vector<cpx> in, out;
  in.resize(2);
  out.resize(2);
  in[0] = cpx(1.0, 1.0);
  in[1] = cpx(2.0, 2.0);

	try
	{
		plan = 
			fftw_plan_dft_1d(2, reinterpret_cast<fftw_complex*>(&in[0]), reinterpret_cast<fftw_complex*>(&out[0]), FFTW_FORWARD, FFTW_ESTIMATE);

		fftw_execute(plan);

		std::cout << out[0] << "; " << out[1] << std::endl;
		fftw_destroy_plan(plan);

		std::cout << out[0] << "; " << out[1] << std::endl;
		Point3  a(1.0, 2.0, 3.0);
		
		NumVec<Scalar>  v(4);
		SetValue<Scalar>( v, Scalar(1.0, 0.0) );
		v[2] = Complex(1.0, 2.0);
		std::cout << v ;
		std::cout << v(2) << " " << v[3] << " " << std::endl;
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
