/// @file esdf.cpp
/// @brief Electronic structure data format for reading the input data.
/// @author Chris J. Pickard and Lin Lin
/// @date 2012-08-10
#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
#include "domain.hpp"
#include "periodtable.hpp"
#include "tinyvec_impl.hpp"

namespace dgdft{


// *********************************************************************
// Electronic structure data format
// *********************************************************************
namespace esdf{


/************************************************************ 
 * Main routines
 ************************************************************/
void esdf_bcast();
void esdf_key();
void esdf_init(const char *);
void esdf_string(const char *, const char *, char *);
int esdf_integer(const char *, int);
float esdf_single(const char *, float);
double esdf_double(const char *, double);
double esdf_physical(const char *, double, char *);
bool esdf_defined(const char *);
bool esdf_boolean(const char *, bool);
bool esdf_block(const char *, int *);
char *esdf_reduce(char *);
double esdf_convfac(char *, char *);
int esdf_unit(int *);
void esdf_file(int *, char *, int *);
void esdf_lablchk(char *, char *, int *);
void esdf_die(char *);
void esdf_warn(char *);
void esdf_close();

/************************************************************ 
 * Utilities
 ************************************************************/
void getaline(FILE *, char *);
void getlines(FILE *fp, char **);
char *trim(char *);
void adjustl(char *,char *);
int len_trim(char *);
int indexstr(char *, char *);
int indexch(char *, char);
int countw(char *, char **, int);
char *strlwr(char *);
char *strupr(char *);


// *********************************************************************
// Input interface
// *********************************************************************
struct ESDFInputParam{
	Domain              domain;
	std::vector<Atom>   atomList;

	Int                 mixMaxDim;
	std::string         mixType;
	std::string         mixVariable;
	Real                mixStepLength;            
	Real                scfInnerTolerance;
	Real                scfOuterTolerance;
	Int                 scfInnerMaxIter;
	Int                 scfOuterMaxIter;
	Real                eigTolerance;
	Int                 eigMaxIter;
	Real                SVDBasisTolerance;
	bool                isRestartDensity;
	bool                isRestartWfn;
	bool                isOutputDensity;
	bool                isOutputWfnExtElem;
	bool                isOutputPotExtElem;
	bool                isCalculateAPosterioriEachSCF; 
	bool                isCalculateForceEachSCF; 
	bool                isOutputHMatrix;


	Real                Tbeta;                    // Inverse of temperature in atomic unit
  Int                 numExtraState;
	std::string         periodTableFile;
	std::string         pseudoType;
	std::string         PWSolver;                 // Type of exchange-correlation functional
	std::string         XCType;
	Int                 XCId;

	// DG related
	Index3              numElem;
	Index3              numGridWavefunctionElem;
	Index3              numGridLGL;
	Real                penaltyAlpha;
	IntNumTns           numALBElem;
	Int                 scaBlockSize;

	// The parameters related to potential barrier is now obsolete.
	Real                potentialBarrierW;
	Real                potentialBarrierS;
	Real                potentialBarrierR;

	// Periodization of the potential in the extended element
	Real                isPeriodizePotential;
  Point3              distancePeriodize;	

	Real                ecutWavefunction;
	Real                densityGridFactor;
	Real                LGLGridFactor;
};

void ESDFReadInput( ESDFInputParam& esdfParam, const std::string filename );

void ESDFReadInput( ESDFInputParam& esdfParam, const char* filename );

} // namespace esdf
} // namespace dgdft
#endif // _ESDF_HPP_
