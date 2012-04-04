#ifndef _SCFPW_HPP_
#define _SCFPW_HPP_

#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "periodtable.hpp"

using std::vector;
using std::pair;
using std::map;
using std::set;
using std::cerr;
using std::cout;
using std::ostream;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;

//----------------------------------------------------------------
class ScfPW
{
public:
  //--------------------------------
  //SETS, NEEDS TO BE REMOVED LATER
  //set<string> _solvemodeset;
  set<string> _mixtypeset;
  set<string> _pseudotypeset;
  //set<string> _mappingmodeset;
  //set<string> _bufferatommodeset;
  //set<string> _restartmodeset;
  //--------------------------------------------------
  //EXTERNAL PARAMETERS
  Domain _dm;
  vector<Atom> _atomvec;
  PeriodTable _ptable;
  Index3 _posidx;
  //
  int _mixdim;            // Maximum dimension of mixing history (for Anderson) 
  string _mixtype;        // Mixing type, (anderson) | kerker
  double _alpha;          // mixing weight
  double _Tbeta;          // inverse temperature, unit: inverse hartree 
  string _pseudotype;     // Pseudopotential type, (GTH) | TM
  string _ptfilename;     // Input file for pseudopotential
  
  
  double _scftol; //LY: eigtol for both global and buffer
  int _scfmaxiter;
  
  double _eigtol;
  int _eigmaxiter; //LY: eigmaxiter=loppcg max iterationu
  
  int _nExtraStates; //LY:
  
  //-----------------------------------
  //LOCAL VARIABLES
  Point3 _Ls; //LY: lenght, width and height
  Index3 _Ns; //LY: # of grid points in each dir, uniform or LGL
  Point3 _pos; //LY: lower left front point location
  double _vol;  //LY: product of these three
  int _ntot; //LY: # of grid points
  vector<DblNumVec> _gridpos; //LY: grid in each direction, vector_size=3, not necessarily useful here
  
  vector<double> _rho0; //LY: move with atom, fixed
  vector<double> _rho; //LY: density
  
  vector<double> _vtot; //LY: total potential
  vector<double> _vhart; //LY: hartree pot.
  vector<double> _vxc; //LY:  exchange-correlation
  
  vector<double> _vext; //LLIN: External potential
  //vector<double> _vloc; //LY: local pseudopotential
  //vector<double> _vhart; //LY: hatree = vhart + vhart0
  //vector<double> _vhart0; //LY: move with atom, fixed
  //vector<double> _vlochart0;
  
  double _Efree;   //LL: Free energy
  double _Etot;    //LL: Total energy
  double _Ekin;
  double _Ecor;
  double _Exc;
  //debugging data
  double _Evxcrho;
  double _Ehalfmm;
  double _Ehalfmp;
  double _Es;
  
  double _Fermi; //LY: Fermi level mu
  int _npsi; //LY: number of eigenfns needed
  vector<double> _ev; //LY: eigvals
  
  int _nOccStates; //LY: nExtra + nOcc = npsi
  vector<double> _occ; //LY: occ rate
  
  fftw_plan _planpsibackward; //LY: fftw plans for psi
  fftw_plan _planpsiforward;
  vector<double> _gkk; //LY: k^2, k contains 2pi already
  CpxNumMat _ik; //LL: ik, size: _ntot*3. For derivaitve purpose. ik contains 2pi already
  
  //-----------------------------------
  //PW specific part
  vector<SparseVec> _rho0s;
  vector< vector< pair<SparseVec,double> > > _vnlss; //LY: each atom, each pseudopot, one sparsevec 
  vector<double> _psi; //LY: all eigen fns in one vec, eqv. to vector<DblNumTNs>
  
public:
  ScfPW();
  ~ScfPW();
  //-------------------------------
  int setup(); // Initial setup of scf calculation
  int update(); // Update the potentials once the atomic configurations are changed
  //-------------------------------
  int scf(vector<double>& rhoinput, vector<double>& psiinput); //
  int scf_CalOcc(double Tbeta); //LY: compute Fermi level and decide occ
  int scf_CalCharge(); //LY: cal charge _rho
  
  int scf_CalXC();  //LY: _vxc, _Exc
  int scf_CalHartree();  //LY: _vhart
  int scf_CalVtot(double* vtot);  //LY: _vtot
  int scf_CalEnergy(); //LY:  _Ekin, _Ecor, _Etot
  
  int scf_Print(FILE *fh);
  int scf_PrintState(FILE *fh);  //int scf_BcastInfo(int master); //TODO
  
  int scf_AndersonMix(vector<double>& vtotnew, vector<double>& df, vector<double>& dv, int iter);
  int scf_KerkerMix(vector<double>& vtotnew, double alpha);
  //-------------------------------
  int force();
  int force_innerprod_val(SparseVec& cur, double* ful, double wgt, double& res); //inner product with VL only
  int force_innerprod(SparseVec& cur, double* ful, double wgt, double* res); //inner product with VL, DX, DY, DZ
  
  //-------------------------------
  //aux functions
  int mpirank() const { int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank); return rank; }
  int mpisize() const { int size; MPI_Comm_size(MPI_COMM_WORLD, &size); return size; }
  // below are funcs from Molecule
  int ntot() const {return _ntot;}
  double vol() const {return _vol;}
  //int Ns(int i) const {assert(i>=0 && i<3); return _Ns(i);}
  //double Ls(int i) const {assert(i>=0 && i<3); return _Ls(i);}
  //double pos(int i) const {assert(i>=0 && i<3); return _pos(i);}
  //int posidx(int i) const {assert(i>=0 && i<3); return _posidx(i);}
  //AtomType& atoms(int i) {assert(i >= 0 && i < _nAtomType); return _atoms[i];}
  //void BcastInfo(int master, MPI_Comm comm); //LY: old code
  
};





#endif
