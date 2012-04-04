#ifndef __DEBUG_HPP
#define __DEBUG_HPP
#include "scfpw.hpp"

template<class F>
void DumpVec(F* vec, int size)
{
  ofstream fh("dump.dat", ios::trunc | ios::binary);
  if( !fh.is_open() ) ABORT("FILE IS NOT OPENED", 1);
  cout << "size = " << size << endl;
  fh.write((char*)&size, sizeof(int));
  fh.write((char*)vec, sizeof(F)*size); 
  fh.close();
}


template<class F>
void DumpVec(char *filename, F* vec, int size)
{
  ofstream fh(filename, ios::trunc | ios::binary);
  if( !fh.is_open() ) ABORT("FILE IS NOT OPENED", 1);
  cout << "size = " << size << endl;
  fh.write((char*)&size, sizeof(int));
  fh.write((char*)vec, sizeof(F)*size); 
  fh.close();
}

template<class F>
void DumpAsciiVec(char *filename, F* vec, int size)
{
  ofstream fh(filename, ios::trunc);
  if( !fh.is_open() ) ABORT("FILE IS NOT OPENED", 1);
  cout << "size = " << size << endl;
  for(int i = 0; i < size; i++){
    fh << vec[i] << endl;
  }
  fh.close();
}

template<class F>
void DumpAsciiVec(F* vec, int size)
{
  ofstream fh("dump.dat", ios::trunc);
  if( !fh.is_open() ) ABORT("FILE IS NOT OPENED", 1);
  cout << "size = " << size << endl;
  for(int i = 0; i < size; i++){
    fh << vec[i] << endl;
  }
  fh.close();
}


#endif
