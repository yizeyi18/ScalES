#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
#include "domain.hpp"
#include "tinyvec_impl.hpp"
#include "periodtable.hpp"
#include <xc.h>

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
bool esdf_boolean(const char *, bool *);
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
	Real                mixStepLength;            
	Real                scfTolerance;
	Int                 scfMaxIter;
	Real                eigTolerance;
	Int                 eigMaxIter;
	bool                isRestartDensity;
	bool                isRestartWfn;
	bool                isOutputDensity;
	bool                isOutputWfn;

	Real                Tbeta;                    // Inverse of temperature in atomic unit
  Int                 numExtraState;
	std::string         periodTableFile;
	std::string         pseudoType;
	std::string         PWSolver;                 // Type of exchange-correlation functional
	std::string         XCType;
	Int                 XCId;

};

void ESDFReadInput( ESDFInputParam& esdfParam, const std::string filename );

void ESDFReadInput( ESDFInputParam& esdfParam, const char* filename );


} // namespace esdf
} // namespace dgdft
#endif // _ESDF_HPP_
