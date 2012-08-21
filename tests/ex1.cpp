#include "dgdft.hpp"
typedef std::complex<double> cpx;

int main(int argc, char **argv) 
{
  fftw_plan plan;
  std::vector<cpx> in, out;
  in.resize(2);
  out.resize(2);
  in[0] = cpx(1.0, 1.0);
  in[1] = cpx(2.0, 2.0);

  plan = 
    fftw_plan_dft_1d(2, reinterpret_cast<fftw_complex*>(&in[0]), reinterpret_cast<fftw_complex*>(&out[0]), FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(plan);

  std::cout << out[0] << "; " << out[1] << std::endl;
  fftw_destroy_plan(plan);

  std::cout << out[0] << "; " << out[1] << std::endl;
  
  return 0;
}
