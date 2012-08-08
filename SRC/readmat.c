/********************************************************************************
 * READMAT reads the binary sparse matrix in triplet form.
 *
 * Lin Lin
 * Date: 7/24/2012
 ********************************************************************************/
#include<stdio.h>
#include<stdlib.h>

int ReadTripletMatrixHeader(int* ptrSizeA, int* ptrSizeTriplet, char* fileName){
  FILE* fileHandle;
  fileHandle = fopen(fileName, "rb");  
  if(!fileHandle) {printf("ERROR reading matrix!\n");};
  fread(ptrSizeA, sizeof(int), 1, fileHandle);
  fread(ptrSizeTriplet, sizeof(int), 1, fileHandle);
  fclose(fileHandle);
  return 0;
}

int ReadTripletMatrix(int* ptrSizeA, int* ptrSizeTriplet, int* RowVec,
		      int* ColVec, double* ValVec, char* fileName){
  FILE* fileHandle;
  fileHandle = fopen(fileName, "rb");  
  if(!fileHandle) {printf("ERROR reading matrix!\n");};
  fread(ptrSizeA, sizeof(int), 1, fileHandle);
  fread(ptrSizeTriplet, sizeof(int), 1, fileHandle);
  fread(RowVec, sizeof(int), *ptrSizeTriplet, fileHandle);
  fread(ColVec, sizeof(int), *ptrSizeTriplet, fileHandle);
  fread(ValVec, sizeof(double), *ptrSizeTriplet, fileHandle);
  fclose(fileHandle);
  return 0;
}


int main(int argc, char **argv){
  int sizeA, sizeTriplet;
  int *RowVec, *ColVec;
  double *ValVec;
  char* inputFileName;

  if( argc < 2 ){
    printf("Usage:\n");
    printf("  readmat <filename>\n");
    return 1;
  }

  inputFileName = argv[1];

  ReadTripletMatrixHeader(&sizeA, &sizeTriplet, inputFileName);
  RowVec = (int*)malloc(sizeof(int)*sizeTriplet);
  ColVec = (int*)malloc(sizeof(int)*sizeTriplet);
  ValVec = (double*)malloc(sizeof(double)*sizeTriplet);
  ReadTripletMatrix(&sizeA, &sizeTriplet, RowVec, ColVec, ValVec, inputFileName);
  
  printf("sizeA = %d, sizeTriplet = %d\n", sizeA, sizeTriplet);
  free(RowVec);
  free(ColVec);
  free(ValVec);

  return 0;
}



