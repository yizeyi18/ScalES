/********************************************************************************
 * GENMAT generates the DG matrix in the binary form that can be read by
 * sparse matrix subroutines in triplet form.
 *
 * Lin Lin
 * Date: 7/24/2012
 ********************************************************************************/
#include "comobject.hpp"
#include "vec3t.hpp"
#include "numtns.hpp"
#include "serialize.hpp"
#include <iomanip>


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
  int mpisize;
  vector<int> noMask(1);
  char inputFileName[100];
  char outputFileName[100];
  FILE* outputFileHandle;
  
  int sizeA, sizeTriplet;
  vector<int> RowVec;
  vector<int> ColVec;
  vector<double> ValVec;

  RowVec.clear();
  ColVec.clear();
  ValVec.clear();
  sizeA = 0;
  sizeTriplet = 0;

  cout << "MPISIZE?" << endl;
  cin >> mpisize;

  for(int iproc = 0; iproc < mpisize; iproc++)
  {
    cout << "Processing file # " << iproc << " (" << mpisize << " in total)" << endl;

    sprintf(inputFileName, "DGMAT_%d_%d", 
	    iproc, mpisize);
    ifstream inputFileStream(inputFileName); iA( inputFileStream.good());

    int tempSizeA, tempSizeTriplet;
    vector<int> tempRowVec;
    vector<int> tempColVec;
    vector<double> tempValVec;
    tempRowVec.clear();
    tempColVec.clear();
    tempValVec.clear();

    deserialize(tempSizeA, inputFileStream, noMask);
    deserialize(tempSizeTriplet, inputFileStream, noMask);
    deserialize(tempRowVec, inputFileStream, noMask);
    deserialize(tempColVec, inputFileStream, noMask);
    deserialize(tempValVec, inputFileStream, noMask);

    if( sizeA != 0 ) iA( tempSizeA == sizeA );

    sizeA = tempSizeA;
    sizeTriplet += tempSizeTriplet;
    RowVec.insert(RowVec.end(), tempRowVec.begin(), tempRowVec.end());
    ColVec.insert(ColVec.end(), tempColVec.begin(), tempColVec.end());
    ValVec.insert(ValVec.end(), tempValVec.begin(), tempValVec.end());
    
    inputFileStream.close();
  }

  iA( RowVec.size() == sizeTriplet );
  iA( ColVec.size() == sizeTriplet );
  iA( ValVec.size() == sizeTriplet );

  cout << "SizeA       = " << sizeA << endl;
  cout << "SizeTriplet = " << sizeTriplet << endl;
  
 
  // Screening according to some criterion

  vector<int> newRowVec;
  vector<int> newColVec;
  vector<double> newValVec;
  newRowVec.clear();
  newColVec.clear();
  newValVec.clear();
  // Dump the upper part only
  // +1 is used to fit the FORTRAN (and MUMPS) format.
  {
    const int TOL = 1e-14;
    for(int i = 0; i < sizeTriplet; i++){
      if( RowVec[i] <= ColVec[i] && abs(ValVec[i]) > TOL ){
	newRowVec.push_back(RowVec[i]+1);
	newColVec.push_back(ColVec[i]+1);
	newValVec.push_back(ValVec[i]);
      }
    }
    sizeTriplet = newRowVec.size();
  }



  cout << "After screening for upper matrix only, SizeTriplet = " << sizeTriplet << endl;
  cout << "Writing to file... " << endl;


  // Output the global triplet format in binary file in traditional C
  // format that can be read easily by other programs
  sprintf(outputFileName, "DGMAT_FULL_UPPER");
  outputFileHandle = fopen(outputFileName, "wb");

  fwrite(&sizeA, sizeof(int), 1, outputFileHandle);
  fwrite(&sizeTriplet, sizeof(int), 1, outputFileHandle);
  fwrite(&newRowVec[0], sizeof(int), newRowVec.size(), outputFileHandle);
  fwrite(&newColVec[0], sizeof(int), newColVec.size(), outputFileHandle);
  fwrite(&newValVec[0], sizeof(double), newValVec.size(), outputFileHandle);

  fclose(outputFileHandle);

  return 0;
}



