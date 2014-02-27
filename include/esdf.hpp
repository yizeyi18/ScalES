/*
	 Copyright (c) 2012 The Regents of the University of California,
	 through Lawrence Berkeley National Laboratory.  

   Authors: Chris J. Pickard and Lin Lin

   This file is part of DGDFT. All rights reserved.

	 Redistribution and use in source and binary forms, with or without
	 modification, are permitted provided that the following conditions are met:

	 (1) Redistributions of source code must retain the above copyright notice, this
	 list of conditions and the following disclaimer.
	 (2) Redistributions in binary form must reproduce the above copyright notice,
	 this list of conditions and the following disclaimer in the documentation
	 and/or other materials provided with the distribution.
	 (3) Neither the name of the University of California, Lawrence Berkeley
	 National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
	 be used to endorse or promote products derived from this software without
	 specific prior written permission.

	 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
	 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	 You are under no obligation whatsoever to provide any bug fixes, patches, or
	 upgrades to the features, functionality or performance of the source code
	 ("Enhancements") to anyone; however, if you choose to make your Enhancements
	 available either publicly, or directly to Lawrence Berkeley National
	 Laboratory, without imposing a separate written license agreement for such
	 Enhancements, then you hereby grant the following license: a non-exclusive,
	 royalty-free perpetual license to install, use, modify, prepare derivative
	 works, incorporate into other computer software, distribute, and sublicense
	 such enhancements or derivative works thereof, in binary and source code form.
*/
/// @file esdf.cpp
/// @brief Electronic structure data format for reading the input data.
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

	Int									nsw; //ZG: MD
	Int									dt; //ZG: MD
	Real								qmass; //ZG: MD								
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
	bool                isOutputWfnElem;
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
  Index3              numGridDensityElem;
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
