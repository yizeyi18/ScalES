/********************************************************************************
 * CONVCUBE converts the output of the adaptive local basis functions
 * and the coefficients of the element oribtals into element orbitals
 * using the Gaussian Cube format
 *
 * Lin Lin
 * Date: 4/9/2012
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
  char infoFileName[100];
  char elementOrbitalCoefficientFileName[100];
  char adaptiveLocalBasisFileName[100];
  string outputFileName;
  vector<int> noMask(1);
  int mpisize;
  Index3 numElements;
  IntNumVec elementToProcessorMap;
  int elementOrbitalLocalOrbitalIndex;
  int elementOrbitalLocalElementIndex;
  int elementOrbitalLocalProcessorIndex;
  Index3 elementOrbitalLocalElement3DIndex;
  DblNumMat elementOrbitalCoefficientMatrix;
  DblNumVec elementOrbitalCoefficientVector;
  DblNumTns elementOrbitalWaveFunction;
  IntNumVec adaptiveLocalBasisToProcessorMap;
  IntNumVec adaptiveLocalBasisGlobalToLocalMap;
  Index3 numGridGlobal;
  Point3 domainSizeGlobal;
  int numAtoms;
  IntNumVec atomTypeVector;
  DblNumMat atomPositionMatrix;


  const int TOLERANCE_COEFFICIENT = 1e-6;

  // information for basis functions
  {
    sprintf(infoFileName, "BASISINFO");
    ifstream inputFileStream(infoFileName); iA( inputFileStream.good());
    
    deserialize(mpisize, inputFileStream, noMask);
    deserialize(numElements, inputFileStream, noMask);
    deserialize(numGridGlobal, inputFileStream, noMask);
    deserialize(domainSizeGlobal, inputFileStream, noMask);
    deserialize(elementToProcessorMap, inputFileStream, noMask);
    deserialize(adaptiveLocalBasisToProcessorMap, inputFileStream, noMask);
    deserialize(adaptiveLocalBasisGlobalToLocalMap, inputFileStream, noMask);

    deserialize(numAtoms, inputFileStream, noMask);
    deserialize(atomTypeVector, inputFileStream, noMask);
    deserialize(atomPositionMatrix, inputFileStream, noMask);
#ifdef __DEBUG
    cout << "Element to Processor map " << endl << elementToProcessorMap << endl;
    cout << "Adaptive local basis to processor map " << endl << 
      adaptiveLocalBasisToProcessorMap << endl;
#endif

    inputFileStream.close();
  }


  // Read information for element orbitals
  {
    int i1, i2, i3;
    cout << "Which element? (Ex: 1 2 3)" << endl;
    cin >> i1 >> i2 >> i3;
    iA( i1 >= 0 && i1 < numElements[0] &&
	i2 >= 0 && i2 < numElements[1] &&
	i3 >= 0 && i3 < numElements[2] ); 
    elementOrbitalLocalElement3DIndex = Index3(i1, i2, i3);
    elementOrbitalLocalElementIndex = i1 + i2 * numElements[0] + 
      i3 * numElements[0] * numElements[1];

    cout << "Which orbital? (Ex: 1)" << endl;
    cin >> elementOrbitalLocalOrbitalIndex;

    cout << "Output file name?" << endl;
    cin >> outputFileName;
  }
  
  time_t timeStaTotal, timeEndTotal;

  timeStaTotal = time(0);

  // Find the correct EOcoef file to get the coefficient matrix.
  {
    elementOrbitalLocalProcessorIndex = 
      elementToProcessorMap[elementOrbitalLocalElementIndex];
    sprintf(elementOrbitalCoefficientFileName, "EOcoef_%d_%d", 
	    elementOrbitalLocalProcessorIndex, mpisize);
#ifdef __DEBUG
    cout << "The element orbital coefficient file name is " <<
      elementOrbitalCoefficientFileName << endl;
#endif
    ifstream inputFileStream(elementOrbitalCoefficientFileName); 
    iA( inputFileStream.good());
    Index3 localIndex;
    deserialize(localIndex, inputFileStream, noMask);
    iA(localIndex == elementOrbitalLocalElement3DIndex);
    deserialize(elementOrbitalCoefficientMatrix, inputFileStream, noMask);
    inputFileStream.close();
    
    iA(elementOrbitalLocalOrbitalIndex < 
       elementOrbitalCoefficientMatrix.n() );

    elementOrbitalCoefficientVector = 
      DblNumVec(elementOrbitalCoefficientMatrix.m(), true,
		elementOrbitalCoefficientMatrix.clmdata(elementOrbitalLocalOrbitalIndex));
#ifdef __DEBUG
    cout << "The coefficients are " << endl << elementOrbitalCoefficientMatrix;
    cout << "The chosen coefficients are " << endl << elementOrbitalCoefficientVector;
#endif

  }
 

  // Loop over the coefficient and construct the element orbital in the
  // global domain
  {
    elementOrbitalWaveFunction.resize(numGridGlobal[0], 
				      numGridGlobal[1],
				      numGridGlobal[2]);
#ifdef __DEBUG 
    cout << "number of grids in the global domain " << endl << 
      numGridGlobal << endl;
#endif
    setvalue(elementOrbitalWaveFunction, 0.0); // VERY IMPORTANT
    int openFileIndex = -1; // Do not open file
    ifstream inputFileStream; 
    Index3 originIndex;
    Index3 numGridLocal;
    vector<DblNumTns> adaptiveLocalBasisWaveFunction;

    time_t timeSta, timeEnd;
    timeSta = time(0);
    for(int iCoef = 0; iCoef < elementOrbitalCoefficientVector.m(); iCoef++){
      // Check whether the current coefficient is nonzero  
      if( abs(elementOrbitalCoefficientVector[iCoef]) < TOLERANCE_COEFFICIENT )
        continue;	
#ifdef __DEBUG
      cout << iCoef << endl;
#endif
      
      // if the coefficient is nonzero, then check whether the
      // corresponding file for saving adaptive local basis functions is
      // open.  If not, close the previous file and open a new file.

      int localProcessorIndex = adaptiveLocalBasisToProcessorMap[iCoef];

      if( localProcessorIndex != openFileIndex ){
	if( inputFileStream.is_open() ) inputFileStream.close();
        adaptiveLocalBasisWaveFunction.clear();	
	openFileIndex = localProcessorIndex;
	sprintf(adaptiveLocalBasisFileName, "ALB_%d_%d", 
		openFileIndex, mpisize); 
	cout << "Opening file " << adaptiveLocalBasisFileName << endl;
	timeEnd = time(0);
	cout << "Last file processing time " << difftime(timeEnd,timeSta) << " sec" << endl;
	inputFileStream.open(adaptiveLocalBasisFileName);
	iA( inputFileStream.good() );

	// Read the local index and the dimension.
	deserialize(numGridLocal, inputFileStream, noMask);
	deserialize(originIndex,  inputFileStream, noMask);
	deserialize(adaptiveLocalBasisWaveFunction, 
		    inputFileStream, noMask);
#ifdef __DEBUG
	cout << "number of local grids " << numGridLocal << endl;
#endif
      }

      // Update the element orbital wave vector
      {
	DblNumTns& basisTns = 
	  adaptiveLocalBasisWaveFunction[adaptiveLocalBasisGlobalToLocalMap(iCoef)];
        for(int k = 0; k < numGridLocal[2]; k++)
	  for(int j = 0; j < numGridLocal[1]; j++)
	    for(int i = 0; i < numGridLocal[0]; i++){
	      int iGlobal = originIndex[0] + i;
	      int jGlobal = originIndex[1] + j;
	      int kGlobal = originIndex[2] + k;
	      elementOrbitalWaveFunction(iGlobal, jGlobal, kGlobal) += 
		basisTns(i, j, k) * elementOrbitalCoefficientVector(iCoef);
	    }
      }
    }
    if( inputFileStream.is_open() ) inputFileStream.close();
  }
  
  // Output the element orbital in the Gaussian cube format
  //
  //            EXAMPLE FROM THE WEB
  //            CPMD CUBE FILE.
  //            OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
  //            3    0.000000    0.000000    0.000000
  //            40    0.283459    0.000000    0.000000
  //            40    0.000000    0.283459    0.000000
  //            40    0.000000    0.000000    0.283459
  //            8    0.000000    5.570575    5.669178    5.593517
  //            1    0.000000    5.562867    5.669178    7.428055
  //            1    0.000000    7.340606    5.669178    5.111259
  //            -0.25568E-04  0.59213E-05  0.81068E-05  0.10868E-04  0.11313E-04  0.35999E-05
  {
    Point3 gridSizeGlobal;
    gridSizeGlobal[0] = domainSizeGlobal[0] / numGridGlobal[0];
    gridSizeGlobal[1] = domainSizeGlobal[1] / numGridGlobal[1];
    gridSizeGlobal[2] = domainSizeGlobal[2] / numGridGlobal[2];

    ofstream outputFileStream(outputFileName.c_str());
    iA( outputFileStream.good() );
    // Header
    outputFileStream << "Gaussian cube format produced by convcube" << endl;
    outputFileStream << "X: Outer loop Y: Middle loop Z: Inner loop" << endl;
    outputFileStream << fixed << setw(9) << numAtoms << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << 0.0 << " " << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridGlobal[0] << " " << setw(12) 
      << setprecision(5) << gridSizeGlobal[0] << " " << 0.0 << " " << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridGlobal[1] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << gridSizeGlobal[1] << " " << 0.0 << endl;
    outputFileStream << fixed << setw(9) << numGridGlobal[2] << " " << setw(12) 
      << setprecision(5) << 0.0 << " " << 0.0 << " " << gridSizeGlobal[2]  << endl;
    for(int i = 0; i < numAtoms; i++){
      outputFileStream << fixed << setw(9) << atomTypeVector(i) << " " << setw(12) 
	<< setprecision(5) << 0.0 << " " 
	<< atomPositionMatrix(i,0) << " " << atomPositionMatrix(i,1) << " " 
	<< atomPositionMatrix(i,2) << endl;
    }


    //LLIN: NOTE the special Z-Y-X order here in the Gaussian cube
    //format.
    for(int i = 0; i < numGridGlobal[0]; i++)
      for(int j = 0; j < numGridGlobal[1]; j++)
	for(int k = 0; k < numGridGlobal[2]; k++){
	  outputFileStream << scientific << setw(12) << setprecision(5) <<
	    elementOrbitalWaveFunction(i,j,k) << endl;
	}
    outputFileStream.close();
  }

  timeEndTotal = time(0);
  cout << "Total processing time " << difftime(timeEndTotal, timeStaTotal) << " sec" << endl;

  return 0;
}



