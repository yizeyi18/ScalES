#ifndef _SCFDG_HPP_
#define _SCFDG_HPP_

#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "periodtable.hpp"
#include "parvec.hpp"

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
class Buff
{
public:
  Domain _dm;
  vector<Atom> _atomvec;
  vector<DblNumVec> _gridpos;
  Index3 _posidx;
  //
  vector<double> _vtot;
  vector< pair<SparseVec,double> > _vnls; //LY: NONLOCAL PSEUDOPOT OF ALL ATOMS GROUPED TOGETHER
 
  //
  int _nactive;   //LLIN: Number of active indices 
  vector<int> _active_indices;  // LLIN: active indices for LOBPCG

  //
  int _npsi;
  vector<double> _ev;
  vector<double> _psi;
  //
  //
  fftw_plan _planpsibackward; //LY: fftw plans for psi
  fftw_plan _planpsiforward;
  //local
  Point3 _Ls; //LY: LENGTH
  Index3 _Ns; //LY: NUMBER OF GRIDPOINTS
  Point3 _pos; //LY: STARTING POS
  double _vol;
  int _ntot;
public:
  Buff();
  ~Buff();
  int setup();
};


//----------------------------------------------------------------
class ElemPtn
{
public:
  IntNumTns _ownerinfo;
public:
  ElemPtn() {;}
  ~ElemPtn() {;}
  IntNumTns& ownerinfo() { return _ownerinfo; }
  int owner(Index3 key) {
    return _ownerinfo(key(0), key(1), key(2));
  }
};

//----------------------------------------------------------------
class Elem //REPLACE THE OLD ELEM DATA STRUCTURE
{
public:
  Domain _dm;  //vector<Atom> _atomvec;
  vector<DblNumVec> _gridpos;
  Index3 _posidx;
  Index3 _Nsglb; //LY: how many global points in this element in each dimension
  vector<vector<double> > _TransGlblx; //LY: 3 times matrix, glb by element
  vector<vector<cpx> >    _TransBufkl; //LY: 3 times matrix element by buffer (k)
  //vector<double> _vtot; //local vtot
  //vector<DblNumTns> _bases; //basis, add polynomial to it afterwards
  //vector<int> _index;
  //DblNumMat _eigvecs;
  //local
  Point3 _Ls; //LY: LENGTH
  Index3 _Ns; //LY: NUMBER OF GRIDPOINTS
  Point3 _pos; //LY: STARTING POS
  double _vol;
  int _ntot;
public:
  Elem() {;}
  ~Elem() {;}
  //
  Domain& dm() { return _dm; }  //vector<Atom>& atomvec() { return _atomvec; }
  vector<DblNumVec>& gridpos() { return _gridpos; }
  Index3& posidx() { return _posidx; }  // LLIN: Starting index of the global points 
                                        // in the current element in each dimension.
  Index3& Nsglb() { return _Nsglb; } //LY: how many global points in this element in each dimension
  vector<vector<double> >& TransGlblx() { return _TransGlblx; } //LY: 3 times matrix, glb by element
  vector<vector<cpx> >&    TransBufkl() { return _TransBufkl; }//LY: 3 times matrix element by buffer (k)
  Point3& Ls() { return _Ls; } //LY: LENGTH
  Index3& Ns() { return _Ns; }//LY: NUMBER OF GRIDPOINTS
  Point3& pos() { return _pos; }//LY: STARTING POS
  double& vol() { return _vol; }
  int& ntot() { return _ntot; }
  //
  int setup();
  int CalTransMatGlb(Domain glb);
  int CalTransMatBuf(Domain buf);
  //---
  //vector<double>& vtot() { return _vtot; } //local vtot
  //vector<DblNumTns>& bases() { return _bases; } //basis, add polynomial to it afterwards
  //vector<int>& index() { return _index; } //basis, add polynomial to it afterwards
  //DblNumMat& eigvecs() { return _eigvecs; }
};

/*
#define Elem_Number 15
enum{
  Elem_dm = 0,  //Elem_atomvec = 1,
  Elem_gridpos = 1,
  Elem_posidx = 2,
  Elem_Nsglb = 3,
  Elem_vtot = 4,
  Elem_TransGlblx = 5,
  Elem_TransBufkl = 6,
  Elem_bases = 7,
  Elem_index = 8,
  Elem_eigvecs = 9,
  Elem_Ls = 10,
  Elem_Ns = 11,
  Elem_pos = 12,
  Elem_vol = 13,
  Elem_ntot = 14,
};
int serialize(const Elem&, ostream&, const vector<int>&);
int deserialize(Elem&, istream&, const vector<int>&);
int combine(Elem&, Elem&);
*/

//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
class PsdoPtn
{
public:
  vector<int> _ownerinfo; //this tells the owner of each atom
public:
  PsdoPtn() {;}
  ~PsdoPtn() {;}
  vector<int>& ownerinfo() { return _ownerinfo; }
  int owner(int key) {
    return _ownerinfo[key];
  }
};

//----------------------------------------------------------------
//typedef pair<NumTns<SparseVec>,double> Psdo;
class Psdo //pseudopot data of an atom
{
public:
  SparseVec _rho0; //rho0 of an atom, sampled on the whole grid
  vector< pair<NumTns<SparseVec>,double> > _vnls; //sampled on LBL grid
public:
  Psdo() {;}
  ~Psdo() {;}
  //
  SparseVec& rho0() { return _rho0; }
  vector< pair<NumTns<SparseVec>,double> >& vnls() { return _vnls; }
};

#define Psdo_Number 2
enum{
  Psdo_rho0 = 0,
  Psdo_vnls = 1,
};

int serialize(const Psdo&, ostream&, const vector<int>&);
int deserialize(Psdo&, istream&, const vector<int>&);
int combine(Psdo&, Psdo&);

//----------------------------------------------------------------
class ScfDG
{
public:
  string _inputformat;
  //--------------------------------
  //SETS, NEEDS TO BE REMOVED LATER
  //set<string> _solvemodeset;
  set<string> _mixtypeset;
  //set<string> _mappingmodeset;
  //set<string> _bufferatommodeset;
  //set<string> _restartmodeset;
  //--------------------------------------------------
  //CONTROL PARAMETERS
  int _isOutputBases;    // Whether to output the adaptive local basis and element orbitals
  int _isOutputDensity; 
  int _isOutputWfn;
  int _isOutputVtot;

  //--------------------------------------------------
  //EXTERNAL PARAMETERS
  Domain _dm;
  vector<Atom> _atomvec;
  PeriodTable _ptable;
  Index3 _posidx;
  //
  int _mixdim;            // Maximum dimension of mixing history (for Anderson) 
  string _mixtype;        // Mixing type, anderson | kerker
  double _alpha;          // mixing weight
  double _Tbeta;          // inverse temperature, unit: inverse hartree 
  
  double _scftol; //LY: eigtol for both global and buffer
  int _scfmaxiter;
  
  int _nExtraStates; //LY: extra number of states to be calculated besides the
                     //occupied states determined by the number of electrons
  //pw stuff
  double _eigtol;
  int _eigmaxiter; //LY: eigmaxiter=loppcg max iterationu
  
  //dg stuff
  int _nBuffUpdate;     //Basis functions are updated every _nBuffUpdate SCF steps.
  int _dgalpha; //DG penalty term
  int _dgndeg;  //LY: poly deg
  int _nenrich; //LY: number of enriches besides poly
  int _nbufextra; //LLIN: extra number of basis functions solved in buffer to guarantee stability (NOT USEFUL)
  int _MB;      //LY: Scalapack MB by MB, 128 or 256
  Index3 _NElems; //LY: numb of elements in 3 directions
  Point3 _ExtRatio; //LLIN: Length of the extended element, in the unit of element size
  NumTns<Domain> _bufftns;
  NumTns<Domain> _elemtns;
  IntNumTns _elemptninfo;


  //LLIN: Nonorthogonal adaptive local basis functions
  string _dgsolver; //LLIN: Standard solver (std) or Nonorthogonal (nonorth)
  double _delta;    //LLIN: Wall width for the Frobenius penalty. 
  double _basisradius; //LLIN: Radius of nonlocal basis function that is not penalized. 
  double _gamma;    //LLIN: Weight ratio between eigenvalue and penalty. Use the default 0.0 value
  int    _Neigperele; //LLIN: Number of eigenvalues to be solved per element in the buffer
  int    _Norbperele; //LLIN: Number of nonorthogonal oritals in the element
  double _DeltaFermi; //LLIN: The increase of the Fermi energy to control the number of candidate functions
  int    _bufdual;    //LLIN: Whether to use dual grid for buffer solve (1/2 grid number along each direction)
  //-----------------------------------
  //LOCAL VARIABLES
  Point3 _Ls; //LY: length, width and height
  Index3 _Ns; //LY: # of grid points in each dir, uniform or LGL
  Point3 _pos; //LY: lower left front point location  
  double _vol;  //LY: product of these three
  int _ntot; //LY: # of grid points
  vector<DblNumVec> _gridpos; //LY: grid in each direction, vector_size=3, not necessarily useful here
  
  vector<double> _rho0; //LLIN: Guess density
  vector<double> _rho; //LY: density
  
  vector<double> _vtot; //LY: total potential
  vector<double> _vext; //LLIn: external potential
  vector<double> _vhart; //LY: hatree = vhart + vhart0
  vector<double> _vxc; //LY: 
  
  double _Efree;   //LL: Free energy
  double _Etot;    //LL: Total energy
  double _Ekin;
  double _Ecor;
  double _Exc;
  
  double _Fermi; //LY: Fermi level mu
  int _npsi; //LY: number of eigenfns needed
  vector<double> _ev; //LY: eigvals
  
  int _nOccStates; //LY: nExtra + nOcc = npsi
  vector<double> _occ; //LY: occ rate  //int _nvnl; //LY: nb of nonlocal pseudopot
  
  fftw_plan _planpsibackward; //LY: fftw plans for psi
  fftw_plan _planpsiforward;
  vector<double> _gkk; //LY: k^2, k contains 2pi already
  CpxNumMat _ik; //LL: ik, size: _ntot*3. For derivaitve purpose. ik contains 2pi already
  
  //-----------------------------------
  //DG specific part
  ElemPtn _elemptn; //partition the elements among processors
  NumTns<Buff> _buffvec;//ParVec<Index3, Buff, ElemPtn> _buffvec;
  NumTns<Elem> _elemvec;  //LEXING: elemvec is local now (stored a copy at all processor) ParVec<Index3, Elem, ElemPtn> _elemvec;
  ParVec<Index3, vector<double>, ElemPtn> _vtotvec; //LEXING: global vector 
  ParVec<Index3, vector<DblNumTns>, ElemPtn> _basesvec; //LEXING: basesvec global
  ParVec<Index3, DblNumMat, ElemPtn> _eigvecsvec; //LEXING: eigvec global now
  ParVec<Index3,DblNumMat,ElemPtn> _EOcoef; //LLIN: The coefficients for element orbitals using ALB
  
  //
  PsdoPtn _psdoptn; //partition the pseudopotentials among processors
  ParVec<int, Psdo, PsdoPtn> _psdovec;
  
public:
  ScfDG();
  ~ScfDG();
  //-------------------------------
  int setup(); // Initial setup of scf calculation
  int update(); // Update the potentials once the atomic configurations are changed
  //-------------------------------
  int scf(); //
  int scf_CalOcc(double Tbeta); //LY: compute Fermi level and decide occ
  //int scf_CalCharge();
  
  int scf_CalXC();
  int scf_CalHartree();
  int scf_CalVtot(double* vtot); // Used here
  int scf_CalEnergy(); //Obsolete
  
  int scf_Print(FILE *fh);
  int scf_PrintState(FILE *fh);  //int scf_BcastInfo(int master); //TODO
  
  // LLIN: Parallel Anderson mixing.
  int scf_PAndersonMix(vector<double>& vtotnew, ParVec<Index3,
		       DblNumMat, ElemPtn>& df, ParVec<Index3,
		       DblNumMat, ElemPtn>& dv, int iter);
  int scf_AndersonMix(vector<double>& vtotnew, vector<double>& df, vector<double>& dv, int iter);
  int scf_KerkerMix(vector<double>& vtotnew, double alpha);
  //-----------------------------------
  int force();
  int force_innerprod_val(SparseVec& cur, double* ful, double wgt, double& res); //inner product with VL only
  int force_innerprod(SparseVec& cur, double* ful, double wgt, double* res); //inner product with VL, DX, DY, DZ
  int force_innerprod(SparseVec& cur, double* ful, double* www, double* res);
  // DG part
  //...
  //-------------------------------
  //aux functions
  int mpirank() const { int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank); return rank; }
  int mpisize() const { int size; MPI_Comm_size(MPI_COMM_WORLD, &size); return size; }
  // below are funcs from Molecule
  int ntot() const {return _ntot;}
  double vol() const {return _vol;}
  //int posidx(int i) const {assert(i>=0 && i<3); return _posidx(i);}
  //AtomType& atoms(int i) {assert(i >= 0 && i < _nAtomType); return _atoms[i];}
  //void BcastInfo(int master, MPI_Comm comm); //LY: old code
  
};





#endif
