#ifndef _PERIODTABLE_HPP_
#define _PERIODTABLE_HPP_

#include "util.hpp"

//---------------------------------------------
class Atom
{
public:
  int _type;
  double _mass;
  Point3 _coord;
  Point3 _vel;
  Point3 _force;
public:
  Atom(int t, double m, Point3 c, Point3 v, Point3 f): _type(t), _mass(m), _coord(c),  
    _vel(v), _force(f) {;}
  ~Atom() {;}
  int& type() { return _type; }
  double& mass() { return _mass; }
  Point3& coord() { return _coord; }
  Point3& vel() { return _vel; }
  Point3& force() { return _force; }
  double coord(int k) {return _coord[k];}
  void set_coord(const Point3& c) {_coord = c;}
};

//---------------------------------------------
class Domain
{
public:
  Point3 _Ls; //length
  Index3 _Ns; //number of grids points in each direction
  Point3 _pos; //starting position
public:
  Domain() {;}
  ~Domain() {;}
  Point3& Ls() { return _Ls; }
  Index3& Ns() { return _Ns; }
  Point3& pos() { return _pos; }
};

#define Domain_Number 3
enum {
  Domain_Ls = 0,
  Domain_Ns = 1,
  Domain_pos = 2,
};

int serialize(const Domain&, ostream&, const vector<int>&);
int deserialize(Domain&, istream&, const vector<int>&);
int combine(Domain&, Domain&);

//---------------------------------------------
class PTEntry
{
public:
  DblNumVec _params; //size 5
  DblNumMat _samples; //ns by nb
  DblNumVec _wgts; //nb
  IntNumVec _typs; //nb
  DblNumVec _cuts; //cutoff value for different mode
  //map<int, vector<DblNumVec> > _spldata; //data to be generated
public:
  DblNumVec& params() { return _params; }
  DblNumMat& samples() { return _samples; }
  DblNumVec& wgts() { return _wgts; }
  IntNumVec& typs() { return _typs; }
  DblNumVec& cuts() { return _cuts; }
  //map<int, vector<DblNumVec> >& spldata() { return _spldata; } //data to be generated
};

int serialize(const PTEntry&, ostream&, const vector<int>&);
int deserialize(PTEntry&, istream&, const vector<int>&);
int combine(PTEntry&, PTEntry&);


//---------------------------------------------
class PeriodTable
{
public:
  enum {
    i_Znuc = 0,
    i_mass = 1,
    i_Zion = 2,
    i_Es = 3,
  };
  enum {
    i_rad = 0,
    i_rho0 = 1,
    i_drho0 = 2,
    //the following ones are all pseudopotentials
  };
public:
  map<int, PTEntry> _ptemap; //map from atom_id to PTEntry
  map<int, map< int,vector<DblNumVec> > > _splmap;
public:
  PeriodTable() {;}
  ~PeriodTable() {;}
  map<int, PTEntry>& ptemap() { return _ptemap; }
  map<int, map< int,vector<DblNumVec> > > splmap() { return _splmap; }
  //
  int setup(string);
  //----------------------
  //evaluate val, dx, dy, dz
  int pseudoRho0(Atom, Point3 Ls, Point3 pos, 
		 Index3 Ns,
		 SparseVec&);
  
  //----------------------
  int pseudoNL(  Atom, Point3 Ls, Point3 pos, 
		 vector<DblNumVec> gridpos,
		 vector< pair<SparseVec,double> >& vnls);
  
  //----------------------
  int pseudoNL(  Atom, Point3 Ls, Point3 pos, 
		 NumTns< vector<DblNumVec> > gridpostns,
		 vector< pair<NumTns<SparseVec>,double> >& vnls);
  
  //TODO: DG version
};


#endif
