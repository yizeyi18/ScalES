//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Lin Lin

/// @file den2cubepw.cpp
/// @brief Utility routine to convert the density files to Gaussian cube
/// format.
/// 
/// This is a sequential code reading the output from PWDFT.
///
/// The input takes the format of DEN.
///
/// NOTE: all units are atomic units (Bohr) here.  
///
///            Example of Gaussian cube format
///            from http://paulbourke.net/dataformats/cube/ 
///
///            CPMD CUBE FILE.
///            OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
///            3    0.000000    0.000000    0.000000
///            40    0.283459    0.000000    0.000000
///            40    0.000000    0.283459    0.000000
///            40    0.000000    0.000000    0.283459
///            8    0.000000    5.570575    5.669178    5.593517
///            1    0.000000    5.562867    5.669178    7.428055
///            1    0.000000    7.340606    5.669178    5.111259
///            -0.25568E-04  0.59213E-05  0.81068E-05  0.10868E-04  0.11313E-04  0.35999E-05
///            xxx
///
/// @date 2016-08-12 Original version
#include "scales.hpp"

#define _DEBUGlevel_ 0

using namespace scales;
using namespace std;
using namespace scales::esdf;
using namespace scales::scalapack;

using std::vector;
using std::pair;
using std::map;
using std::set;
using std::cin;
using std::cerr;
using std::cout;
using std::ostream;
using std::istream;
using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::setw;
using std::setprecision;
using std::scientific;
using std::fixed;

int main(int argc, char **argv){
  char infoFileName[100];
  char densityFileName[100];
  string outputFileName;
  Point3 domainSizeGlobal;
  Index3 numGridGlobal;
  Index3 numGridFineGlobal;
  Point3 posStartGlobal;
  Int numAtom;
  DblNumVec  densityVec;
  std::vector<Atom> atomList;


  // Initialization
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  if( mpisize != 1 ){
    if( mpirank == 0 ){
      cerr << "The current code only supports serial mode." << endl;
    }
    MPI_Finalize();
    return 0;
  }

  // Structure information 
  {
#if ( _DEBUGlevel_ >= 0 )
    cout << "Reading structure file " << endl;
#endif
    sprintf(infoFileName, "STRUCTURE");
    ifstream inputFileStream(infoFileName); iA( inputFileStream.good());


    // Domain
    deserialize(domainSizeGlobal, inputFileStream, NO_MASK);
    deserialize(numGridGlobal, inputFileStream, NO_MASK); 
    deserialize(numGridFineGlobal, inputFileStream, NO_MASK);
    deserialize(posStartGlobal, inputFileStream, NO_MASK);

    // Atom
    deserialize( atomList, inputFileStream, NO_MASK );

    numAtom = atomList.size();

    inputFileStream.close();

#if ( _DEBUGlevel_ >= 0 )
    cout << "number of fine grids in the global domain " << endl << 
      numGridFineGlobal << endl;
#endif
  }

  // Read information
  {
    outputFileName = "DEN.cub";
    //    cout << "Output file name?" << endl;
    //    cin >> outputFileName;
  }


  Real timeStaTotal, timeEndTotal;

  GetTime( timeStaTotal );

  // Construct the density in the global domain
  {
    // Only works for the case without magnetization
    densityVec.Resize(numGridFineGlobal[0]*numGridFineGlobal[1]*numGridFineGlobal[2]);
    SetValue( densityVec, 0.0 );


    ifstream inputFileStream("DEN");
    iA( inputFileStream.good() );

    // Read the gridpos
    std::vector<DblNumVec> gridpos(DIM);
    for( Int d = 0; d < DIM; d++ ){
      deserialize( gridpos[d], inputFileStream, NO_MASK );
    }

    deserialize(densityVec, inputFileStream, NO_MASK);

    if( inputFileStream.is_open() ) inputFileStream.close();
  }

  // Output the density in the Gaussian cube format
  {
    Point3 gridSizeFineGlobal;
    gridSizeFineGlobal[0] = domainSizeGlobal[0] / numGridFineGlobal[0];
    gridSizeFineGlobal[1] = domainSizeGlobal[1] / numGridFineGlobal[1];
    gridSizeFineGlobal[2] = domainSizeGlobal[2] / numGridFineGlobal[2];

    ofstream outputFileStream(outputFileName.c_str());
    iA( outputFileStream.good() );
    // Header
    outputFileStream << "Gaussian cube format, created by den2cube" << endl;
    outputFileStream << "X: Outer loop Y: Middle loop Z: Inner loop" << endl;
    outputFileStream << fixed << setw(9) << numAtom << " " 
      << setw(12) << setprecision(5) 
      << posStartGlobal[0] << " " 
      << posStartGlobal[1] << " " 
      << posStartGlobal[2] << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[0] << " " << setw(12) 
      << setprecision(5) << gridSizeFineGlobal[0]  << " " 
      << 0.0 << " " << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[1] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << gridSizeFineGlobal[1]  << " " 
      << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridFineGlobal[2] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << 0.0 << " " << gridSizeFineGlobal[2]  << 
      endl;
    for( Int a = 0; a < numAtom; a++ ){
      outputFileStream << fixed << setw(9) << atomList[a].type << " " << setw(12) 
        << setprecision(5) << 0.0 << " " 
        << atomList[a].pos[0]  << " " 
        << atomList[a].pos[1]  << " " 
        << atomList[a].pos[2]  << endl;
    }

    DblNumTns densityGlobal(
        numGridFineGlobal[0],
        numGridFineGlobal[1],
        numGridFineGlobal[2], 
        false, densityVec.Data() );


    //NOTE the special Z-Y-X order here in the Gaussian cube format.
    Int count = 0;
    for(int i = 0; i < numGridFineGlobal[0]; i++)
      for(int j = 0; j < numGridFineGlobal[1]; j++)
        for(int k = 0; k < numGridFineGlobal[2]; k++){
          outputFileStream << scientific << setw(12) << setprecision(5) <<
            densityGlobal(i,j,k);
          count++;
          if( count % 6 == 0 )
            outputFileStream << endl;
        }
    outputFileStream.close();
  }

  GetTime( timeEndTotal );
  cout << "Total processing time " << timeEndTotal - timeStaTotal 
    << " [s]" << endl;

  MPI_Finalize();

  return 0;
}



