/*Test of parallel matrix vector multiplication subroutine.

  ttmatvec1 calculates
    Y = A * X
  
  There are three phases involved:
    1) Distribute X.
    2) Calculate A*X locally.
    3) Reduce  Y.
  
  Lin Lin  
  Revision: 09/10/2011
*/
  
#include "esdf.h"
#include "debug.hpp"
#include "scfdg.hpp"
#include "parallel.hpp"

class VecPtn
{
public:
  IntNumVec _ownerinfo;
public:
  VecPtn() {;}
  ~VecPtn() {;}
  IntNumVec& ownerinfo() { return _ownerinfo; }
  int owner(int key){ return _ownerinfo(key); }
};

typedef Vec2T<int> SpBlckKey;

class SpBlckPtn{
  public:
    map<SpBlckKey, int> _ownerinfo;
  public:
    SpBlckPtn() {;}
    ~SpBlckPtn() {;}
    map<SpBlckKey, int>& ownerinfo(){ return _ownerinfo;}
    int owner(SpBlckKey key) { return _ownerinfo[key];}
};


int main(int argc, char **argv) 
{ MPI_Init(&argc, &argv);
  int myid;  
  int nprocs;  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  int mpirank = myid;
  int mpisize = nprocs;

  SpBlckPtn sbptn;
  {
    map<SpBlckKey, int>& ownerinfo = sbptn.ownerinfo();
    ownerinfo[SpBlckKey(0,0)] = 0;
    ownerinfo[SpBlckKey(0,1)] = 0;
    ownerinfo[SpBlckKey(1,0)] = 1;
    ownerinfo[SpBlckKey(1,1)] = 1;
  }
  
  ParVec<SpBlckKey, DblNumMat, SpBlckPtn> Avec;
  
  Avec.prtn() = sbptn;

//  for(map<SpBlckKey, int>::iterator mi=Avec.prtn().ownerinfo().begin(); mi!=Avec.prtn().ownerinfo().end(); mi++){
//    if(mpirank == 1){
//      cout << (*mi).first << ", " << (*mi).second << endl;
//    }
//  }

  int szblck = 2;
  int nbblck = 4;
  DblNumMat tt1(szblck, szblck), tt2(szblck, szblck), tt3(szblck, szblck);
  {
    DblNumMat blank(szblck, szblck);
    setvalue(blank, 0.0);
    tt1 = blank;  tt2 = blank;  tt3 = blank;
    tt1(0,0) = 2.0;  tt1(0,1) = -1.0; tt1(1,0) = -1.0; tt1(1,1) = 2.0;
    tt2(1,0) = -1.0;
    tt3(0,1) = -1.0;
  }

  if(mpirank == 0){
    Avec.lclmap()[SpBlckKey(0,0)] = tt1;
    Avec.lclmap()[SpBlckKey(0,1)] = tt2;
    Avec.lclmap()[SpBlckKey(1,0)] = tt3;
  }
  if(mpirank == 1){
    Avec.lclmap()[SpBlckKey(1,1)] = tt1;
    Avec.lclmap()[SpBlckKey(1,2)] = tt2;
  }
  if(mpirank == 2){
    Avec.lclmap()[SpBlckKey(2,1)] = tt3;
    Avec.lclmap()[SpBlckKey(2,2)] = tt1;
    Avec.lclmap()[SpBlckKey(2,3)] = tt2;
  }
  if(mpirank == 3){
    Avec.lclmap()[SpBlckKey(3,2)] = tt3;
    Avec.lclmap()[SpBlckKey(3,3)] = tt1;
  }
  
//  for(map<SpBlckKey, DblNumMat>::iterator mi = Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++){
//    if( mpirank == 1 ){
//      cout << (*mi).first<< endl << (*mi).second << endl;
//    }
//  }

 
  VecPtn vptn;
  {
    IntNumVec& ownerinfo = vptn.ownerinfo();
    ownerinfo.resize(4);
    ownerinfo[0] = 0;
    ownerinfo[1] = 1;
    ownerinfo[2] = 2;
    ownerinfo[3] = 3;
  }
   
  ParVec<int, DblNumVec, VecPtn> Xvec;
  Xvec.prtn() = vptn;
  DblNumVec evec(2);
  if(mpirank == 0){
    setvalue(evec, 1.0);
    Xvec.lclmap()[0] = evec;
  }
  if(mpirank == 1){
    setvalue(evec, 2.0);
    Xvec.lclmap()[1] = evec;
  }
  if(mpirank == 2){
    setvalue(evec, 3.0);
    Xvec.lclmap()[2] = evec;
  }
  if(mpirank == 3){
    setvalue(evec, 4.0);
    Xvec.lclmap()[3] = evec;
  }
  
//  for(map<int, DblNumVec>::iterator mi = Xvec.lclmap().begin(); mi!=Xvec.lclmap().end(); mi++){
//    cout << "mpirank = " << mpirank << " " << (*mi).first<< ", " << (*mi).second << endl;
//  }

  /*Distribute Xvec*/
  
  set<int> col_keyset;
  for(map<SpBlckKey, DblNumMat>::iterator mi = Avec.lclmap().begin(); mi != Avec.lclmap().end(); mi++){
    int curkey = (*mi).first[1];
    if(Xvec.prtn().owner(curkey) != mpirank){
      col_keyset.insert(curkey);
    }
  }
  vector<int> col_keyvec;   col_keyvec.insert(col_keyvec.begin(), 
					      col_keyset.begin(), col_keyset.end());

//  for(vector<int>::iterator mi = col_keyvec.begin(); mi != col_keyvec.end(); mi++)
//    cout << "mpirank = " << mpirank << " " << (*mi) << endl;
  
  {
    vector<int> all(1,1);
    iC( Xvec.getBegin(col_keyvec, all) ); 
    iC( Xvec.getEnd(all) );
  }
  
//  for(map<int, DblNumVec>::iterator mi = Xvec.lclmap().begin(); mi!=Xvec.lclmap().end(); mi++){
//    if(mpirank == 1)
//      cout << "mpirank = " << mpirank << " " << (*mi).first<< endl << (*mi).second << endl;
//  }

  ParVec<int, DblNumVec, VecPtn> Yvec;
  Yvec.prtn() = vptn;
  /* Perform matrix vector multiplication*/
  int idx;
  DblNumVec blank_vec(szblck);
  setvalue(blank_vec, 0.0);
  for(map<SpBlckKey, DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++){
    idx = (*mi).first[0];
    Yvec.lclmap()[idx] = blank_vec;
  }
  for(map<SpBlckKey, DblNumMat>::iterator mi=Avec.lclmap().begin(); mi!=Avec.lclmap().end(); mi++){
    idx = (*mi).first[0];
    SpBlckKey  curidx = (*mi).first;
    DblNumMat& curmat = (*mi).second;
    DblNumVec& curxvec = Xvec.lclmap()[curidx[1]];
    DblNumVec& curyvec = Yvec.lclmap()[curidx[0]];
    { char Ntrans='N';
      double D_ONE=1.0;
      int I_ONE = 1;
      dgemv_(&Ntrans, &szblck, &szblck, &D_ONE, curmat.data(), &szblck,
	     curxvec.data(), &I_ONE, &D_ONE, curyvec.data(), &I_ONE);
    }
  }
   
//  for(map<int, DblNumVec>::iterator mi = Yvec.lclmap().begin(); mi!=Yvec.lclmap().end(); mi++){
//      cout << "mpirank = " << mpirank << " " << (*mi).first<< endl << (*mi).second << endl;
//  }
  
  /*Reduce Yvec*/
  set<int> row_keyset;
  for(map<int, DblNumVec>::iterator mi = Yvec.lclmap().begin(); mi != Yvec.lclmap().end(); mi++){
    int curkey = (*mi).first;
    if(Yvec.prtn().owner(curkey) != mpirank){
      row_keyset.insert(curkey);
    }
  }
  vector<int> row_keyvec;   row_keyvec.insert(row_keyvec.begin(), 
					      row_keyset.begin(), row_keyset.end());

  
  {
    vector<int> all(1,1);
    iC( Yvec.putBegin(row_keyvec, all) ); 
    iC( Yvec.putEnd(all, PARVEC_CMB) );
  }
   
  for(map<int, DblNumVec>::iterator mi = Yvec.lclmap().begin(); mi!=Yvec.lclmap().end(); mi++){
      cout << "mpirank = " << mpirank << " " << (*mi).first<< endl << (*mi).second << endl;
  }


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
