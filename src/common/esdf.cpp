//  This file is a part of ScalES (see LICENSE). All Right Reserved
//
//  Copyright (c) 2012-2021 The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory.  
//
//  Authors: Chris J. Pickard and Lin Lin

/// @file esdf.cpp
/// @brief Electronic structure data format for reading the input data.
/// @date 2012-08-10
/// @date 2020-09-19
#include "esdf.hpp"
#include "utility.hpp" 
#include "domain.hpp"
#include "periodtable.hpp"
#include <xc.h>

namespace scales{


// *********************************************************************
// Electronic structure data format
// *********************************************************************

//===============================================================
//
// Electronic structure data format
// ---------------------------------------------------------------
//
//                            e s d f
//                            =======
//
// Author: Chris J. Pickard (c)
// Email : cp@min.uni-kiel.de
// Place : kiel, Germany
// Date  : 5/6th august 1999
//
// Summary
// -------
//
// This module is designed to simplify and enhance the input of data into
// electronic structure codes (for example, castep). It works from a
// fashion, and input is independent of the ordering of the input
// file. An important feature is the requirement that most inputs require
// default settings to be supplied within the main program calling
// esdf. This means that rarely used variables will not clutter everyday
// input files, and, even more usefully, "intelligence" may be built into
// the main code as the defaults may be dependent of other set
// variables. Block data may also be read in. Another important feature
// is the ability to define "physical" values. This means that the input
// files need not depend on the internal physical units used by the main
// program.
//
// History
// -------
//
// Esdf has been written from scratch in f90, but is heavily based
// (especially for the concept) on the fdf package developed by alberto
// garcia and jose soler. It is not as "flexible" as fdf - there is no
// provision for branching to further input files. This simplifies the
// code, and I hope that it is still useful without this feature. Also,
// the input and defaults are not dumped to a output file currently. I've
// not found this a hindrance as of now.
//
// Future
// ------
//
// My intention is to make this release available to alberto garcia and
// jose soler for their comments. It might be a good idea to use this as
// a base for fully converting the fdf package to f90. Or it may remain
// as a cut down version of fdf. I certainly hope that a package of the
// fdf sort becomes widely used in the electronic structure community. My
// experience has been very positive.
//
// Usage
// -----
//
// First, "use esdf" wherever you wish to make use of its features. In
// the main program call the initialisation routine: call
// esdf_init('input.esdf'). "input.esdf" is the name of the input file -
// it could be anything. This routine opens the input file, and reads
// into a dynamically allocated storage array. The comments and blank
// lines are stripped out. You are now ready to use the
// esdf_functions. For example, if you want to read in the number of
// atoms in your calculation, you would use: natom =
// esdf_integer('numberofatoms',1), where 'numberofatoms' is the label to
// search for in the input file, and '1' is the default. Call esdf_close to
// deallocate the data arrays. You may then open another input file using
// esdf_init. It is not currently possible to open more that on input
// file at one time.
//
// Syntax
// ------
//
// The input file can contain comments. These are defined as anything to
// the right of, and including, '#', ';', or '!'. It is straightforward
// to modify the routine to accept further characters. Blank lines are
// ignored -- use comments and blank lines to make you input file
// readable.
//
// The "labels" are case insensitive (e.g. unitcell is equivalent to
// unitcell) and punctuation insensitive (unit.cell is equivalent to
// unit_cell is equivalent to unitcell). Punctuation characters are '.'
// and '-' at the moment. Again - use this feature to improve
// readability.
//
// The following are equivalent ways of defining a physical quantity:
//
// "Ageofuniverse = 24.d0 s" or "ageofuniverse : 24.d0 s" or
// "ageofuniverse 24.d0 s"
//
// It would be read in by the main program in the following way:
//
// Aou = esdf_physical('ageofuniverse',77.d0,'ns')
//
// "Aou" is the double precision variable, 77.d0 is the default number of
// "ns" or nanoseconds. 24s will be converted automatically to its
// equivalent number of nanoseconds.
//
// Block data should be placed in the input file as follows:
//
// Begin cellvectors
// 1.0 1.0 0.0
// 0.0 1.0 1.0
// 1.0 0.0 1.0
// end cellvectors
//
// And it may be read:
//
//   If(esdf_block('cellvectors',nlines))
//     if(nlines /= 3) then (... break out here if the incorrect number
// of lines)
//     do i=1,nlines
//       read(block_data(i),*) x,y,z
//     end do
//   endif
//
// List of functions
// -----------------
//
// Self explanatory:
//
// Esdf_string(label,default)
// esdf_integer(label,default)
// esdf_single(label,default)
// esdf_double(label,default)
// esdf_physical(label,default,unit)
//
// a little more explanation:
//
// Esdf_defined(label) is true if "label" found, false otherwise
//
// Esdf_boolean(label,default) is true if "label yes/true/t (case/punct.insens)
//                             is false if"label no/false/f (case/punct.insens)
//
// The help feature
// ----------------
//
// The routine "esdf_help(helpword,searchword)" can be used to access the
// information contained within the "esdf_key_mod" module.
//
// If "helpword" is "search" (case insensitive), then all variables whose
// description contains "searchword" will be output.
//
// If "helpword" is "basic", "inter", "expert" or "dummy" the varibles of
// that type will be displayed.
//
// If "helpword" is one of the valid labels, then a description of this
// label will be output.
//
// Finishing off
// -------------
//
// Two routines, "esdf_warnout" and "esdf_close", can be used to finish
// the use of esdf. "esdf_warnout" outputs esdf warnings to screen, and
// "esdf_close" deallocates the allocated esdf arrays.
//
// Contact the author
// ------------------
//
// This code is under development, and the author would be very happy to
// receive comments by email. Any use in a commercial software package is
// forbidden without prior arrangement with the author (Chris J. Pickard).
//---------------------------------------------------------------

namespace esdf{

// *********************************************************************
// Constants
// *********************************************************************
const int nphys = 57;
const int llength = 4096;  /* length of the lines */
const int numkw = 500;   /* maximum number of keywords */



char **block_data;
int nrecords;
int nwarns;
char **llist;
char **warns;
char ***tlist;
char phy_d[nphys][11];          /* D - dimension */
char phy_n[nphys][11];          /* N - name */
double phy_u[nphys];            /* U - unit */

char kw_label[numkw][100];
int kw_index[numkw];
char kw_typ[numkw][4];

FILE *fileunit;


/************************************************************ 
 * Main routines
 ************************************************************/
void esdf_key() {
  /*  ===============================================================
   *
   *   Module to hold keyword list. this must be updated as
   *   new keywords are brought into existence.
   *
   *   The 'label' is the label as used in calling the esdf routines
   *   'typ' defines the type, with the following syntax. it is 3
   *   characters long.
   *   the first indicates:
   *        i - integer
   *        s - single
   *        d - double
   *        p - physical
   *        t - string (text)
   *        e - defined (exists)
   *        l - boolean (logical)
   *        b - block
   *   the second is always a colon (:)
   *   the third indicates the "level" of the keyword
   *        b - basic
   *        i - intermediate
   *        e - expert
   *        d - dummy
   *
   *   'Dscrpt' is a description of the variable. it should contain a
   *   (short) title enclosed between *! ... !*, and then a more detailed
   *   description of the variable.
   *
   *  ===============================================================
   */

  int i=0;

  strcpy(kw_label[i],"atom_types_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"atom_type");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"core_cutoff_radius");
  strcpy(kw_typ[i],"P:E");

  i++;
  strcpy(kw_label[i],"potential_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"local_component");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"move_flag");
  strcpy(kw_typ[i],"I:I");

  i++;
  strcpy(kw_label[i],"correlation_type");
  strcpy(kw_typ[i],"T:D");

  i++;
  strcpy(kw_label[i],"ion_energy_diff");
  strcpy(kw_typ[i],"P:I");

  i++;
  strcpy(kw_label[i],"atom_coord");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"grid_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"super_cell");
  strcpy(kw_typ[i],"B:E");
  
  i++;
  strcpy(kw_label[i],"super_cell_angstrom");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"upf_file");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"grid_spacing");
  strcpy(kw_typ[i],"P:E");

  i++;
  strcpy(kw_label[i],"miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"cheb_deg");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mixing_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"mixing_variable");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"mixing_param");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"mixing_maxdim");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"log_files");
  strcpy(kw_typ[i],"L:B");

  /* LL: Keywords added below for ScalES */
  i++;
  strcpy(kw_label[i],"mixing_steplength");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"penalty_alpha");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_min_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"eig_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_inner_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_outer_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_outer_energy_tolerance");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"svd_basis_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"temperature");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"ion_temperature");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_inner_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_inner_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_outer_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_outer_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_energy_criteria_engage_ioniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_outer_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_etot_diff");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"md_scf_eband_diff");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"dg_degree");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"alb_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"alb_num_element");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"scalapack_block_size");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"extra_electron");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mapping_mode");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"element_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"extra_states");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pseudo_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"output_dir");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"restart_mode");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"restart_density");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_wfn");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_density");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_potential");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_wfn");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_alb_elem_lgl");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_alb_elem_uniform");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"output_wfn_extelem");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_pot_extelem");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_eigvec_coef");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"output_hmatrix");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"element_position_start");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"element_grid_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"element_cell");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"position_start");
  strcpy(kw_typ[i],"B:E");



  i++;
  strcpy(kw_label[i],"max_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"buff_update");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"statfile");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"dg_solver");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"wallwidth");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"weightratio");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eigperele");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"orbperele");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"input_format");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"extended_element_ratio");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"output_bases");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"basis_radius");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"atom_bohr");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"atom_angstrom");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"atom_crystal");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"buf_dual");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"vext");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"output_vtot");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"deltafermi");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"pw_solver");
  strcpy(kw_typ[i],"T:E");
  
  i++;
  strcpy(kw_label[i],"ppcg_sbsize");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"xc_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"vdw_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"calculate_aposteriori_each_scf");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"calculate_force_each_scf");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"potential_barrier");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_w");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_s");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_r");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"ecut_wavefunction");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"density_grid_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"lgl_grid_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"gauss_interp_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"gauss_sigma");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"periodize_potential");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"distance_periodize");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"solution_method");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"diag_solution_method");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"smearing_scheme");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"num_pole");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_distfft");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_scalapack");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_scalapack_pw");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_row_pexsi");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_col_pexsi");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_symb_fact");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"energy_gap");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"spectral_radius");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"matrix_ordering");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"inertia_count");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"inertia_count_steps");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"max_pexsi_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pexsi_method");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pexsi_npoint");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mu_min");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_max");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"num_electron_pexsi_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_inertia_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_inertia_expansion");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_pexsi_safeguard");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"unused_states");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"eig_tolerance_dynamic");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"ion_max_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"ion_move");
  strcpy(kw_typ[i],"T:E");


  i++;
  strcpy(kw_label[i],"geo_opt_max_force");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"geo_opt_nlcg_sigma");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"fire_nmin");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"fire_time_step");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"fire_atomic_mass");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"md_max_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_time_step");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_variable");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_wavefunction");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"thermostat_mass");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"langevin_damping");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"kappa_xlbomd");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"restart_position");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_velocity");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_position");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_velocity");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"output_xyz");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"scf_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_phi_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_ace");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"hybrid_df_kmeans_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_kmeans_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_mu");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_gaussianrandom");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_proc_scalapack");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"fftw_mpi_size");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"block_size_scalapack");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_active_init");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_mixing_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"hybrid_ace_twice_pcdiis");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"exx_divergence_type");
  strcpy(kw_typ[i],"I:E");


  // Inputs related to Chebyshev polynomial Filtered SCF iterations for PWDFT    
  i++;
  strcpy(kw_label[i],"first_scf_pwdft_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"first_scf_pwdft_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"general_scf_pwdft_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_ppcg_use_scala");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_cheby_use_scala");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_cheby_use_wfn_ecut_filt");
  strcpy(kw_typ[i],"I:E");



  // ~~**~~
  // Inputs related to Chebyshev ploynomial Filtered SCF iterations for DG
  i++;
  strcpy(kw_label[i],"diag_scfdg_by_cheby");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cheby_use_scalapack");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"first_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"first_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"second_scfdg_chebyouteriter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"second_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"second_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"general_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"general_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  // **###**
  // Inputs related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in ScalES
  i++;
  strcpy(kw_label[i],"scfdg_use_chefsi_complementary_subspace");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_chefsi_complementary_subspace_syrk");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_chefsi_complementary_subspace_syr2k");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_nstates");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cs_ioniter_regular_cheby_freq");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cs_bigger_grid_dim_fac");
  strcpy(kw_typ[i],"I:E");


  // Inner LOBPCG related options
  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_lobpcgtol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_lobpcgiter");
  strcpy(kw_typ[i],"I:E");

  // Inner CheFSI related options
  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_use_inner_cheby");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"use_atom_density");
  strcpy(kw_typ[i],"I:E");

  // The following are for TDDFT
  i++;
  strcpy(kw_label[i],"tddft");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_auto_save_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_tddft_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_ehrenfest");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_vext");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_dipole");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"tddft_vext_polx");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_poly");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_polz");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_freq");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_phase");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_amp");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_t0");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_tau");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_env");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"tddft_method");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"tddft_max_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_delta_t");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_total_t");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_krylov_max");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_krylov_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_diis_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_phi_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_diis_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"program");
  strcpy(kw_typ[i],"T:B");
}

void esdf() {
  /*  we allow case variations in the units. this could be dangerous
   *  (mev --> mev !!) in real life, but not in this restricted field.
   *
   *  m - mass l - length t - time e - energy f - force p - pressure
   *  c - charge d - dipole mom - mom inert ef - efield
   */
  struct {
    char d[11];
    char n[11];
    double u;
  } phy[nphys]={
    {"m",   "kg",           1.0},
    {"m",   "g",            1.0e-3},
    {"m",   "amu",          1.66054e-27},
    {"l",   "m",            1.0},
    {"l",   "nm",           1.0e-9},
    {"l",   "ang",          1.0e-10},
    {"l",   "bohr",         0.529177e-10},
    {"t",   "s",            1.0},
    {"t",   "ns",           1.0e-9},
    {"t",   "ps",           1.0e-12},
    {"t",   "fs",           1.0e-15},
    {"e",   "j",            1.0},
    {"e",   "erg",          1.0e-7},
    {"e",   "ev",           1.60219e-19},
    {"e",   "mev",          1.60219e-22},
    {"e",   "ry",           2.17991e-18},
    {"e",   "mry",          2.17991e-21},
    {"e",   "hartree",      4.35982e-18},
    {"e",   "kcal/mol",     6.94780e-21},
    {"e",   "mhartree",     4.35982e-21},
    {"e",   "kj/mol",       1.6606e-21},
    {"e",   "hz",           6.6262e-34},
    {"e",   "thz",          6.6262e-22},
    {"e",   "cm-1",         1.986e-23},
    {"e",   "cm^-1",        1.986e-23},
    {"e",   "cm**-1",       1.986e-23},
    {"f",   "N",            1.0},
    {"f",   "ev/ang",       1.60219e-9},
    {"f",   "ry/bohr",      4.11943e-8},
    {"l",   "cm",           1.0e-2},
    {"p",   "pa",           1.0},
    {"p",   "mpa",          1.0e6},
    {"p",   "gpa",          1.0e9},
    {"p",   "atm",          1.01325e5},
    {"p",   "bar",          1.0e5},
    {"p",   "mbar",         1.0e11},
    {"p",   "ry/bohr**3",   1.47108e13},
    {"p",   "ev/ang**3",    1.60219e11},
    {"c",   "c",            1.0},
    {"c",   "e",            1.602177e-19},
    {"d",   "C*m",          1.0},
    {"d",   "D",            3.33564e-30},
    {"d",   "debye",        3.33564e-30},
    {"d",   "e*bohr",       8.47835e-30},
    {"d",   "e*ang",        1.602177e-29},
    {"mom", "kg*m**2",      1.0},
    {"mom", "ry*fs**2",     2.1799e-48},
    {"ef",  "v/m",          1.0},
    {"ef",  "v/nm",         1.0e9},
    {"ef",  "v/ang",        1.0e10},
    {"ef",  "v/bohr",       1.8897268e10},
    {"ef",  "ry/bohr/e",    2.5711273e11},
    {"ef",  "har/bohr/e",   5.1422546e11},
    {"e",   "k",            1.38066e-23},
    {"b",   "t",            1.0},
    {"b",   "ry/mu_bohr",   2.350499e5},
    {"b",   "g",            1.0e4}
  };
  int i;

  for (i=0;i<nphys;i++) {
    strcpy(phy_d[i],phy[i].d);
    strcpy(phy_n[i],phy[i].n);
    phy_u[i]=phy[i].u;
  }
}

/*   --------------  esdf_bcast ----------------------  */
/*   Modified by Lin Lin, Nov 9, 2010                   */
void esdf_bcast(){
  int  mpirank;  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank ); 
  int  mpisize;  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  const int MASTER = 0;
  int i, j;
  MPI_Bcast(&nrecords, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&nwarns,   1, MPI_INT, MASTER, MPI_COMM_WORLD);
  if( mpirank != MASTER ){
    esdf();
    esdf_key();
    block_data=(char **)malloc(sizeof(char*)*nrecords);
    llist=(char **)malloc(sizeof(char*)*nrecords);
    warns=(char **)malloc(sizeof(char*)*nrecords);
    tlist=(char ***)malloc(sizeof(char**)*nrecords);
    for (i=0;i<nrecords;i++) {
      block_data[i]=(char *)malloc(sizeof(char)*llength);
      llist[i]=(char *)malloc(sizeof(char)*llength);
      warns[i]=(char *)malloc(sizeof(char)*(llength+1));
      tlist[i]=(char **)malloc(sizeof(char*)*llength);
      for (j=0;j<llength;j++)
        tlist[i][j]=(char *)malloc(sizeof(char)*llength);
    }
  }
  for (i=0;i<nrecords;i++) {
    MPI_Bcast(block_data[i], llength, MPI_CHAR, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(llist[i],      llength, MPI_CHAR, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(warns[i],    llength+1, MPI_CHAR, MASTER, MPI_COMM_WORLD);
    for (j=0;j<llength;j++)
      MPI_Bcast(tlist[i][j],    llength, MPI_CHAR, MASTER, MPI_COMM_WORLD);
  }
}


/*   --------------  esdf_init  ----------------------  */
void esdf_init(const char *fname) {
  /* Initialize */
  const int ncomm=3;
  const int ndiv=3;
  int unit,ierr,i,j,k,ic,nt,ndef,nread,itemp,itemp2;
  char cjunk[llength],ctemp[llength];
  char comment[ncomm],divide[ndiv];
  bool inblock;
  char filename[llength];

  strcpy(filename,fname);

  /* Define comment characters */
  comment[0]='#';  divide[0]=' ';
  comment[1]=';';  divide[1]='=';
  comment[2]='!';  divide[2]=':';

  esdf();
  esdf_key();

  /* "Reduce" the keyword list for comparison */
  for (i=0;i<numkw;i++) {
    strcpy(ctemp,kw_label[i]);
    strcpy(kw_label[i],esdf_reduce(strlwr(ctemp)));
  }

  /* initializing the array kw_index */
  for (i=0;i<numkw;i++) kw_index[i]=0;

  /* Open the esdf file */
  esdf_file(&unit,filename,&ierr);
  strcpy(cjunk,"Unable to open main input file \"");
  strcat(cjunk,trim(filename));
  strcat(cjunk,"\"");

  if (ierr==1) {
    printf("ESDF WARNING: %s - using defaults",trim(cjunk));
    nread=0;
  } 
  else
    nread=INT_MAX;

  /* Count the number of records (excluding blank and commented lines) */
  nrecords=0;

  for (i=0;i<nread;i++) {
    getaline(fileunit,cjunk);
    for (j=0;j<ncomm;j++) {
      ic=indexch(cjunk,comment[j]);
      if (ic>0) {
        for (k=ic-1;k<llength;k++) cjunk[k]=' ';
        cjunk[llength-1]='\0';
      }
    }
    if (len_trim(cjunk)>0) nrecords++;
    if (feof(fileunit)) break;
  }
  rewind(fileunit);

  /* Allocate the array to hold the records and tokens */
  block_data=(char **)malloc(sizeof(char*)*nrecords);
  llist=(char **)malloc(sizeof(char*)*nrecords);
  warns=(char **)malloc(sizeof(char*)*nrecords);
  tlist=(char ***)malloc(sizeof(char**)*nrecords);
  for (i=0;i<nrecords;i++) {
    block_data[i]=(char *)malloc(sizeof(char)*llength);
    llist[i]=(char *)malloc(sizeof(char)*llength);
    warns[i]=(char *)malloc(sizeof(char)*(llength+1));
    tlist[i]=(char **)malloc(sizeof(char*)*llength);
    for (j=0;j<llength;j++)
      tlist[i][j]=(char *)malloc(sizeof(char)*llength);
  }

  /* Set the number of warnings to zero */
  nwarns=0;
  for (i=0;i<nrecords;i++) {
    for (j=0;j<llength;j++)
      warns[i][j]=' ';
    warns[i][llength] ='\0';
  }

  /* Read in the records */
  nrecords=0;

  for (i=0;i<nread;i++) {
    getaline(fileunit,cjunk);
    for (j=0;j<ncomm;j++) {
      ic=indexch(cjunk,comment[j]);
      if (ic>0) {
        for (k=ic-1;k<llength;k++) cjunk[k]=' ';
        cjunk[llength-1]='\0';
      }
    }
    if (len_trim(cjunk)>0) {
      adjustl(cjunk,llist[nrecords]);
      nrecords++;
    }
    if (feof(fileunit)) break;
  }

  /* Now read in the tokens from llist */
  for (i=0;i<nrecords;i++)
    for (j=0;j<llength;j++) {
      for (k=0;k<llength-1;k++)
        tlist[i][j][k]=' ';
      tlist[i][j][llength-1]='\0';
    }

  for (i=0;i<nrecords;i++) {
    strcpy(ctemp,llist[i]);
    nt=0;
    while (len_trim(ctemp)>0) {
      ic=len_trim(ctemp)+1;
      for (itemp=0;itemp<ndiv;itemp++) {
        itemp2=indexch(ctemp,divide[itemp]);
        if (itemp2==0) itemp2=len_trim(ctemp)+1;
        if (itemp2<ic) ic=itemp2;
      }
      if (ic>1) {
        for (itemp=0;itemp<ic-1;itemp++)
          cjunk[itemp]=ctemp[itemp];
        cjunk[ic-1]='\0';
        adjustl(cjunk,tlist[i][nt]);
        nt++;
      }
      for (itemp=ic;itemp<strlen(ctemp)+1;itemp++)
        cjunk[itemp-ic]=ctemp[itemp];
      cjunk[strlen(ctemp)-ic+1]='\0';
      adjustl(cjunk,ctemp);        
    }
  }

  /* Check if any of the "labels" in the input file are unrecognized */
  inblock=0; /* false */

  for (i=0;i<nrecords;i++) {
    /* Check if we are in a block */
    strcpy(ctemp,tlist[i][0]);
    if (strcmp(esdf_reduce(strlwr(ctemp)),"begin")==0) {
      inblock=1; /* true */

      /* Check if block label is recognized */
      strcpy(ctemp,tlist[i][1]); esdf_reduce(strlwr(ctemp));
      k=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) k++;
      if (k==0) {
        strcpy(ctemp,"Label \"");
        strcat(ctemp,esdf_reduce(tlist[i][1]));
        strcat(ctemp,"\" not in keyword list");
        if (countw(ctemp,warns,nwarns)==0) esdf_warn(ctemp);
      }

      /* Check if "label" is multiply defined in the input file */
      ndef=0;
      for (j=0;j<nrecords;j++)
        if (strcmp(esdf_reduce(tlist[i][1]), esdf_reduce(tlist[j][1]))==0) {
          ndef++;
        }
      strcpy(ctemp,"Label \"");
      strcat(ctemp,esdf_reduce(tlist[i][1]));
      strcat(ctemp,"\" is multiply defined in the input file. ");

      if ((ndef>2)&&(countw(ctemp,warns,nwarns)==0)) esdf_warn(ctemp);
    }

    /* Check it is in the list of keywords */
    strcpy(ctemp,tlist[i][0]); esdf_reduce(strlwr(ctemp));
    k=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) k++;
    if ((k==0)&&(!inblock)) {
      strcpy(ctemp,"Label \"");
      strcat(ctemp,esdf_reduce(tlist[i][0]));
      strcat(ctemp,"\" not in keyword list");
      if (countw(ctemp,warns,nwarns)==0) esdf_warn(ctemp);
    }

    /* Check if "label" is multiply defined in the input file */
    if (!inblock) {
      ndef=0;
      for (j=0;j<nrecords;j++)
        if (strcmp(esdf_reduce(tlist[i][0]),
              esdf_reduce(tlist[j][0]))==0) ndef++;
      strcpy(ctemp,"Label \"");
      strcat(ctemp,esdf_reduce(tlist[i][0]));
      strcat(ctemp,"\" is multiply defined in the input file. ");
      if ((ndef>1)&&(countw(ctemp,warns,nwarns)==0)) esdf_warn(ctemp);
    }
  }
}

/*   --------------  esdf_string  ----------------------  */
void esdf_string(const char *labl, const char *def, char *out) {
  /* Return the string attached to the "label" */
  int i,kw_number;
  int ind;
  char *pout;
  char label[llength];
  char strT[] = "T";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strT,&kw_number);

  /* Set to default */
  strcpy(out,def);
  pout=out;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      ind=indexstr(llist[i],trim(tlist[i][1]))-1;
      while (llist[i][ind]!='\0') {
        *pout=llist[i][ind];
        pout++; ind++;
      }
      kw_index[kw_number]=i+1;
      break;
    }
  }
  if (pout!=out) *pout='\0';
}

/*   --------------  esdf_integer  ----------------------  */
int esdf_integer(const char *labl,int def) {
  /* Return the integer value attached to the "label" */
  int i;
  char ctemp[llength];
  char ctemp1[llength];
  int kw_number=0;
  int out;
  char label[llength];
  char strI[] = "I";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strI,&kw_number);

  /* Set to default */
  out=def;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=atoi(tlist[i][1]);
      if ((out==0)&&(atoi(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_integer");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i+2;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_single  ----------------------  */
float esdf_single(const char *labl,float def) {
  /* Return the single precisioned value attached to the "label" */
  float out;
  int i;
  char ctemp[llength];
  char ctemp1[llength];
  int kw_number;
  char label[llength];
  char strS[] = "S";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strS,&kw_number);

  /* Set to default */
  out=def;
  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_single");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i + 1;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_double  ----------------------  */
double esdf_double(const char *labl,double def) {
  /* Return the double precisioned value attached to the "label" */
  int i;
  char ctemp[llength];
  char ctemp1[llength];
  int kw_number;     
  double out;
  char label[llength];
  char strD[] = "D";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label,strD,&kw_number);

  /* Set to default */
  out=def;
  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_double");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i+1;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_physical  ----------------------  */
double esdf_physical(const char *labl,double def,char *dunit) {
  /* Return the double precisioned physical value attached to the "label"
     units converted to "dunit"
   */
  int i,j;
  char ctemp[llength],iunit[llength];
  char ctemp1[llength];
  int kw_number;
  double out;
  char label[llength];
  char strP[] = "P";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strP,&kw_number);

  /* Set to default */
  out=def;

  for (i=0;i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_physical");
        esdf_die(ctemp);
        continue;
      }
      strcpy(iunit,dunit);
      for (j=0;j<llength-strlen(dunit)-1;j++) strcat(iunit," ");
      if (len_trim(tlist[i][2])!=0)
        strcat(iunit,tlist[i][2]);
      out=esdf_convfac(iunit,dunit) * out;
      kw_index[kw_number]=i + 1;
    }
  }
  return out;
}

/*   --------------  esdf_defined  ----------------------  */
bool esdf_defined(const char *labl) {
  /* Is the "label" defined in the input file */
  int i;
  int kw_number;
  bool out;
  char label[llength];
  char ctemp1[llength];
  char strE[] = "E";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strE,&kw_number);

  /* Set to default */
  out=0; /* false */

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=1; /* true */
      kw_index[kw_number]=i+1;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_boolean  ----------------------  */
bool esdf_boolean(const char *labl, bool def) {
  /* Is the "label" defined in the input file */
  int i;
  char positive[3][llength],negative[3][llength];
  char ctemp[llength];
  char ctemp1[llength];
  int kw_number;
  bool out;
  char label[llength];
  char strL[] = "L";

  strcpy(label,labl);
  strlwr(label);
  strcpy(positive[0],"yes");
  strcpy(positive[1],"true");
  strcpy(positive[2],"t");
  strcpy(negative[0],"no");
  strcpy(negative[1],"false");
  strcpy(negative[2],"f");

  /* Check "label" is defined */
  esdf_lablchk(label, strL,&kw_number);

  /* Set to default */
  out=def;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][0]);
    strlwr(ctemp1);
    if (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0) {
      out=1; /* true */
      kw_index[kw_number]=i+2;
      if (len_trim(tlist[i][1])==0) {
        out=1; /* true */
        break;
      }
      if ((indexstr(positive[0],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(positive[1],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(positive[2],esdf_reduce(tlist[i][1]))>0)) {
        out=1; /* true */
        break;
      }
      if ((indexstr(negative[0],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(negative[1],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(negative[2],esdf_reduce(tlist[i][1]))>0)) {
        out=0; /* false */
        break;
      }
      strcpy(ctemp, "Unable to parse boolean value");
      esdf_die(ctemp);
    }
  }

  return out;
}

/*   --------------  esdf_block  ----------------------  */
bool esdf_block(const char *labl,int *nlines) {
  int i;
  char ctemp[llength];
  char ctemp1[llength];
  int kw_number;
  bool out;
  char label[llength];
  char strB[] = "B";

  strcpy(label,labl);
  strlwr(label);
  /* Check "label" is defined */
  esdf_lablchk(label, strB,&kw_number);
  strcpy(ctemp,"Block \"");
  strcat(ctemp,esdf_reduce(label));
  strcat(ctemp,"\" not closed correctly");

  out=0; /* false */
  (*nlines)=0;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    strcpy(ctemp1,tlist[i][1]);
    strlwr(ctemp1);
    if ((strcmp(esdf_reduce(tlist[i][0]),"begin")==0) &&
        (strcmp(esdf_reduce(ctemp1),esdf_reduce(label))==0)) {
      out=1; /* true */
      kw_index[kw_number]=i+1;
      while (strcmp(esdf_reduce(tlist[i+(*nlines)+1][0]),"end")!=0) {
        (*nlines)++;
        if ((*nlines)+i>nrecords) esdf_die(ctemp);
        strcpy(block_data[(*nlines)-1],llist[i+(*nlines)]);
      }
      strcpy(ctemp1,tlist[i+(*nlines)+1][1]);
      strlwr(ctemp1);
      if (strcmp(esdf_reduce(ctemp1),
            esdf_reduce(label))!=0)
        esdf_die(ctemp);
      break;
    }
  }

  return out;
}

/*   --------------  esdf_reduce  ----------------------  */
char *esdf_reduce(char *in) {
  /* Reduce the string to lower case and remove punctuation */
  const int npunct=2;
  char *end;

  /* Define the punctuation to be removed */
  char punct[npunct];

  punct[0]='.'; punct[1]='-';
  if (in) {
    while (((*in)==' ')||((*in)=='\t')||
        ((*in)==punct[0])||((*in)==punct[1])) in++;
    if (*in) {
      end=in+strlen(in);
      while ((end[-1]==' ')||(end[-1]=='\t')||
          (end[-1]==punct[0])||(end[-1]==punct[1])) end--;
      if (end<in+strlen(in)) (*end)='\0';
    }
  }

  return in;
}

/*   --------------  esdf_convfac  ----------------------  */
double esdf_convfac(char *from,char *to) {
  /* Find the conversion factor between physical units */
  int i,ifrom,ito;
  char ctemp[llength];
  double out;

  /* Find the index numbers of the from and to units */
  ifrom=-1; ito=-1;
  for (i=0;i<nphys;i++) {
    if (strcmp(esdf_reduce(from),esdf_reduce(phy_n[i]))==0) ifrom=i;
    if (strcmp(esdf_reduce(to),esdf_reduce(phy_n[i]))==0) ito=i;
  }

  /* Check that the units were recognized */
  if (ifrom==-1) {
    strcpy(ctemp,"Units not recognized in input file :");
    strcat(ctemp,esdf_reduce(from));
    esdf_die(ctemp);
  }

  if (ito==-1) {
    strcpy(ctemp,"Units not recognized in Program :");
    strcat(ctemp,esdf_reduce(to));
    esdf_die(ctemp);
  }

  /* Check that from and to are of the same dimensions */
  if (phy_d[ifrom]!=phy_d[ito]) {
    strcpy(ctemp,"Dimensions do not match :");
    strcat(ctemp,esdf_reduce(from));
    strcat(ctemp," vs ");
    strcat(ctemp,esdf_reduce(to));
    esdf_die(ctemp);
  }

  /* Set the conversion factor */
  out=phy_u[ifrom]/phy_u[ito];

  return out;
}

/*   --------------  esdf_file  ----------------------  */
void esdf_file(int *unit,char *filename,int *ierr) {
  /* Open an old file */
  (*ierr)=0;

  if ((fileunit=fopen(trim(filename),"r"))==NULL){
    ErrorHandling( "Input file cannot be open" );
    (*ierr)=1;
  }
}

/*   --------------  esdf_lablchk  ----------------------  */
void esdf_lablchk(char *str,char *typ,int *index) {
  /* Check if label is recognized */
  char ctemp[llength];
  char tp[2];
  int i,j;

  strcpy(ctemp,str); esdf_reduce(strlwr(ctemp));
  i=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) i++;
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" not recognized in keyword list");
  if (i==0) esdf_die(ctemp);
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" is multiply defined");
  if (i>1) esdf_die(ctemp);
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" has been used with the wrong type");

  strcpy(tp," ");
  i=0;
  while (strcmp(tp," ")==0) {
    strcpy(ctemp,str);
    if (strcmp(esdf_reduce(strlwr(ctemp)),kw_label[i])==0)
      strncpy(tp,kw_typ[i],1);
    i++;
  }

  (*index)=i-1;
  if (strcmp(typ,tp)!=0) esdf_die(ctemp);
}

/*   --------------  esdf_die  ----------------------  */
void esdf_die(char *str) {
  /* Stop execution due to an error cause by esdf */
  char error[llength]="ESDF ERROR: ";

  printf("%s", strcat(error,trim(str)));    
  printf("\nStopping now\n");

  exit(0);
}

/*   --------------  esdf_warn  ----------------------  */
void esdf_warn(char *str) {
  /* Warning due to an error cause by esdf */

  strcpy(warns[nwarns],str);
  nwarns++;
}

/*   --------------  esdf_close  ----------------------  */
void esdf_close() {
  /* Deallocate the data arrays --- call this before re-initializing */
  int i,j;

  for (i=0;i<nrecords;i++) {
    for (j=0;j<llength;j++)
      free(tlist[i][j]);
    free(tlist[i]);
    free(warns[i]);
    free(llist[i]);
    free(block_data[i]);
  }
  free(tlist);
  free(warns);
  free(llist);
  free(block_data);
}


/************************************************************ 
 * Utilities
 ************************************************************/

/* ------------ GETALINE ---------- */
void getaline(FILE *fp, char *aline) {
  /*
   *  Purpose
   *  =======
   *
   *  Read a line from a file.
   *
   *  Arguments
   *  =========
   *
   *  fp        (input) FILE *
   *            handle of the file
   *  aline     (output) char *
   *            output buffer
   *
   */

  char ch='\0';
  int i=0;

  while (!feof(fp)) {
    ch=fgetc(fp);
    if ((ch=='\n')||(ch=='\r'))
      break;
    aline[i++]=ch;
  }

  if (aline[i-1]==(char)EOF) aline[i-1]='\0';
  else aline[i]='\0';
}

/* ------------ GETLINES ---------- */
void getlines(FILE *fp, char **buffer) {
  /*
   *  Purpose
   *  =======
   *
   *  Load a file to memory.
   *
   *  Arguments
   *  =========
   *
   *  fp        (input) FILE *
   *            handle of the file
   *  buffer    (output) char *
   *            output buffer
   *
   */

  int i=0;

  while (!feof(fp))
    getaline(fp,buffer[i++]);
}

/* ------------ TRIM --------------- */
char *trim(char *in) {
  /*
   *  Purpose
   *  =======
   *
   *  Delete blank characters on both ends of a string.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  in        (input/output) char *
   *            pointer to the original string, changed after the call
   *
   *  Return Value
   *  ============
   *
   *  char *
   *  pointer to the trimmed string
   *
   */

  char *end;

  if (in) {
    while (*in==' '||*in=='\t') in++;
    if (*in) {
      end=in+strlen(in);
      while (end[-1]==' '||end[-1]=='\t') end--;
      (*end)='\0';
    }
  }

  return in;
}

/* ------------ ADJUSTL ------------ */
void adjustl(char *in,char *out) {
  /*
   *  Purpose
   *  =======
   *
   *  Move blank characters from the beginning of the string to the end.
   *
   *  Arguments
   *  =========
   *
   *  in        (input) char *
   *            pointer to the original string
   *  out       (output) char *
   *            pointer to the new string
   *
   */

  char *pin,*pout;
  int i;

  for (i=0;in[i]==' '||in[i]=='\t';i++);
  for (pin=in+i,pout=out;(*pout=*pin);pin++,pout++);
  for (;i>0;i--,pout++)
    *pout=' ';
  *pout='\0';
}

/* ------------ LEN_TRIM ----------- */
int len_trim(char *in) {
  /*
   *  Purpose
   *  =======
   *
   *  Trim a string and calculate its length.
   *  Delete blank characters on both ends of a string.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  in        (input/output) char *
   *            pointer to the original string, changed after the call
   *
   *  Return Value
   *  ============
   *
   *  int
   *  length of the trimmed string
   *
   */

  return strlen(trim(in));
}

/* ------------ INDEX -------------- */
int indexstr(char *string, char *substring) {
  /*
   *  Purpose
   *  =======
   *
   *  Find the first occurence of a substring.
   *
   *  Arguments
   *  =========
   *
   *  string    (input) char *
   *            pointer to the string
   *  substring (input) char *
   *            pointer to the substring
   *
   *  Return Value
   *  ============
   *
   *  int
   *  >0 index of the substring (1 based indexing)
   *  <0 the substring is not found
   *
   */

  char *p,*q;

  p=string;
  q=strstr(string,substring);

  return q-p+1;
}

/* ------------ INDEXCH ------------ */
int indexch(char *str, char ch) {
  /*
   *  Purpose
   *  =======
   *
   *  Find the first occurence of a character.
   *
   *  Arguments
   *  =========
   *
   *  str       (input) char *
   *            pointer to the string
   *  ch        (input) char
   *            the character to be found
   *
   *  Return Value
   *  ============
   *
   *  int
   *  >0 index of the character (1 based indexing)
   *  =0 the character is not found
   *
   */

  char *p,*q;

  p=str;
  q=strchr(str,ch);

  if (q-p+1>0)
    return q-p+1;
  else
    return 0;
}

/* ------------ COUNTW -------------- */
int countw(char *str, char **pool, int nrecords) {
  /*
   *  Purpose
   *  =======
   *
   *  Count the time of occurences of a string.
   *
   *  Arguments
   *  =========
   *
   *  str       (input) char *
   *            the string to be counted
   *  pool      (input) char *[]
   *            the whole string list
   *  nrecords  (input) int
   *            number of strings in the list
   *
   *  Return Value
   *  ============
   *
   *  int
   *  time of occurences of the string
   *
   */

  int i,n=0;

  for (i=0;i<nrecords;i++)
    if (strcmp(str,pool[i])==0)
      n++;

  return n;
}

/* ------------ STRLWR -------------- */
char *strlwr(char *str) {
  /*
   *  Purpose
   *  =======
   *
   *  Convert a string to lower case, if possible.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  str       (input/output) char *
   *            pointer to the original string, keep unchanged
   *
   *  Return Value
   *  ============
   * 
   *  char *
   *  pointer to the new string
   *
   */

  char *p;

  for (p=str;*p;p++)
    if (isupper(*p))
      *p=tolower(*p);

  return str;
}

/* ------------ STRUPR -------------- */
char *strupr(char *str) {
  /*
   *  Purpose
   *  =======
   *
   *  Convert a string of the first letter to upper case,
   *     others to lower case, if possible.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  str       (input/output) char *
   *            pointer to the original string, keep unchanged
   *
   *  Return Value
   *  ============
   * 
   *  char *
   *  pointer to the new string
   *
   */

  char *p;

  p=str;
  if (islower(*p)) *p=toupper(*p);
  p++;
  for (;*p;p++)
    if (isupper(*p))
      *p=tolower(*p);

  return str;
}

// *********************************************************************
// Input interface
// *********************************************************************

void ESDFReadInput( const std::string filename ){
  ESDFReadInput( filename.c_str() );
  return ;
}

void
ESDFReadInput ( const char* filename )
{
  Int  mpirank;  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  Int  mpisize;  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  Int  nlines;
  const Int MAX_CHAR = 2048;
  char  strtmp[MAX_CHAR];

  // Read and distribute the input file
  if( mpirank == 0 )  
    esdf_init( filename );
  esdf_bcast( );

  // Now each processor can read parameters independently
  // Program type. All options below assume pwdft is used unless otherwise specified. 
  // Many options are shared by all programs
  {
    std::vector<std::string> program_list = { "pwdft", "scales", "tddft" };
    esdf_string("Program", "pwdft", strtmp); 
    esdfParam.program         = strtmp;
    if( not InArray(esdfParam.program, program_list) ){
      ErrorHandling("Invalid program mode.");
    }
  }

  // Domain
  {
    Domain& dm = esdfParam.domain;
    if( esdf_block("Super_Cell", &nlines) ){
      sscanf(block_data[0],"%lf %lf %lf",
          &dm.length[0],&dm.length[1],&dm.length[2]);
    }
    else if( esdf_block("Super_Cell_Angstrom", &nlines) ){
      sscanf(block_data[0],"%lf %lf %lf",
          &dm.length[0],&dm.length[1],&dm.length[2]);
      dm.length /= au2ang;
    }
    else{
      ErrorHandling("Super_Cell or Super_Cell_Angstrom must be defined.");
    }

    dm.posStart = Point3( 0.0, 0.0, 0.0 );
  }

  {
    Domain& dm = esdfParam.domain;
    if( esdf_block("UPF_File", &nlines) ){
      esdfParam.pspFile.resize(nlines);
      int m;
      for( int i = 0; i < nlines; i++){
        esdfParam.pspFile[i] = block_data[i];
        esdfParam.pspFile[i].erase( remove_if( esdfParam.pspFile[i].begin(), esdfParam.pspFile[i].end(), isspace), esdfParam.pspFile[i].end());
      }
    }
    else{
      ErrorHandling("UPF_File must be defined.");
    }
  }

  // Atoms
  {
    std::vector<Atom>&  atomList = esdfParam.atomList;
    atomList.clear();

    esdfParam.numAtomType = esdf_integer("Atom_Types_Num", 0);
    if( esdfParam.numAtomType == 0 ){
      ErrorHandling("Atom_Types_Num cannot be found.");
    }

    for( Int ityp = 0; ityp < esdfParam.numAtomType; ityp++ ){
      Int type = esdf_integer( "Atom_Type", 0 );
      if( type == 0 ){
        ErrorHandling( "Atom_Type cannot be found.");

      }
      Int  numAtom;

      if( esdf_block("Atom_Bohr", &numAtom ) ){
        // Cartesian coordinate (in the unit of Bohr) 
        Point3 pos;
        Domain& dm = esdfParam.domain;

        for( Int j = 0; j < numAtom; j++ ){
          sscanf(block_data[j],"%lf %lf %lf", 
              &pos[0], &pos[1], &pos[2]);


          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );
        }
      }
      else if( esdf_block("Atom_Angstrom", &numAtom ) ){
        // Cartesian coordinate (in the unit of angstrom) 
        Point3 pos;
        Domain& dm = esdfParam.domain;

        for( Int j = 0; j < numAtom; j++ ){
          sscanf(block_data[j],"%lf %lf %lf", 
              &pos[0], &pos[1], &pos[2]);
           
          pos /= au2ang;
          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );

        }
      }
      else if ( esdf_block("Atom_Crystal", &numAtom) ){
        // atomic positions are in crystal coordinates, i.e.  
        // in relative coordinates of the supercell lattice vectors
        Point3 pos;
        Point3 length = esdfParam.domain.length;
        Domain& dm = esdfParam.domain;

        for( Int j = 0; j < numAtom; j++ ){
          sscanf(block_data[j],"%lf %lf %lf", 
              &pos[0], &pos[1], &pos[2]);
          pos[0] *= dm.length[0];
          pos[1] *= dm.length[1];
          pos[2] *= dm.length[2];

          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );

        }
      }
      else{
        std::ostringstream msg;
        msg << "Atomic coordinates cannot found for atom type "  << type;
        ErrorHandling( msg.str().c_str() );
      } // Read atomic coordinates
    } // for(ityp)
  }

  // System parameters
  {
    esdfParam.mixMaxDim       = esdf_integer("Mixing_MaxDim", 10);

    esdf_string("Mixing_Type", "anderson", strtmp); 
    esdfParam.mixType         = strtmp;
    if( esdfParam.mixType != "anderson" &&
        esdfParam.mixType != "kerker+anderson" ){
      ErrorHandling("Invalid mixing type.");
    }

    esdf_string("Mixing_Variable", "potential", strtmp); 
    esdfParam.mixVariable     = strtmp;
    if( esdfParam.mixVariable != "density" &&
        esdfParam.mixVariable != "potential" ){
      ErrorHandling("Invalid mixing variable.");
    }

    esdfParam.fftwMPISize          = esdf_integer( "FFTW_MPI_Size",  1 );

    esdfParam.mixStepLength        = esdf_double( "Mixing_StepLength", 0.5 );
    esdfParam.scfOuterTolerance    = esdf_double( "SCF_Tolerance", 1e-6 );
    esdfParam.scfOuterMaxIter      = esdf_integer( "SCF_MaxIter",   100 );
    esdfParam.scfPhiMaxIter        = esdf_integer( "SCF_Phi_MaxIter",   30 );
    esdfParam.scfPhiTolerance      = esdf_double( "SCF_Phi_Tolerance",   1e-6 );

    esdf_string("Hybrid_Mixing_Type", "pcdiis", strtmp); 
    esdfParam.hybridMixType         = strtmp;
    if( esdfParam.hybridMixType != "nested" &&
        esdfParam.hybridMixType != "pcdiis" ){
      ErrorHandling("Invalid hybrid mixing type.");
    }

    esdfParam.isHybridACETwicePCDIIS           = esdf_integer( "Hybrid_ACE_Twice_PCDIIS", 1 );

    esdfParam.isHybridACE                      = esdf_integer( "Hybrid_ACE", 1 );
    esdfParam.isHybridActiveInit               = esdf_integer( "Hybrid_Active_Init", 0 );

    esdfParam.isHybridDF                       = esdf_integer( "Hybrid_DF", 0 );
    //esdfParam.numMuHybridDF                    = esdf_double( "Num_Mu_Hybrid_DF", 6.0 );
    //esdfParam.numGaussianRandomHybridDF        = esdf_double( "Num_GaussianRandom_Hybrid_DF", 2.0 );
    //esdfParam.numProcScaLAPACKHybridDF         = esdf_integer( "Num_Proc_ScaLAPACK_Hybrid_DF", mpisize );
    //esdfParam.isHybridDFQRCP                   = esdf_integer( "Hybrid_DF_QRCP", 1 );
    //esdfParam.isHybridDFKmeans                 = esdf_integer( "Hybrid_DF_Kmeans", 0 );

    esdf_string("Hybrid_DF_Type", "Kmeans+QRCP", strtmp); 
    esdfParam.hybridDFType         = strtmp;
    // LL: FIXME 01/08/2021 the code only supports Kmeans+QRCP?
    if( esdfParam.hybridDFType != "QRCP" &&
        esdfParam.hybridDFType != "Kmeans" &&
        esdfParam.hybridDFType != "Kmeans+QRCP"){
      ErrorHandling("Invalid ISDF type.");
    }

    esdfParam.hybridDFKmeansTolerance          = esdf_double( "Hybrid_DF_Kmeans_Tolerance", 1e-3 );
    esdfParam.hybridDFKmeansMaxIter            = esdf_integer( "Hybrid_DF_Kmeans_MaxIter", 99 );
    esdfParam.hybridDFNumMu                    = esdf_double( "Hybrid_DF_Num_Mu", 6.0 );
    esdfParam.hybridDFNumGaussianRandom        = esdf_double( "Hybrid_DF_Num_GaussianRandom", 2.0 );
    esdfParam.hybridDFNumProcScaLAPACK         = esdf_integer( "Hybrid_DF_Num_Proc_ScaLAPACK", mpisize );
    esdfParam.hybridDFTolerance                = esdf_double( "Hybrid_DF_Tolerance", 1e-20 );
    esdfParam.BlockSizeScaLAPACK               = esdf_integer( "Block_Size_ScaLAPACK", 32 );

    esdfParam.MDscfPhiMaxIter      = esdf_integer( "MD_SCF_Phi_MaxIter", esdfParam.scfPhiMaxIter  );
    esdfParam.MDscfOuterMaxIter    = esdf_integer( "MD_SCF_Outer_MaxIter",  esdfParam.scfOuterMaxIter ); // This is used in ScalES for energy based SCF

    esdfParam.exxDivergenceType    = esdf_integer( "EXX_Divergence_Type", 1 );

    esdfParam.eigTolerance         = esdf_double( "Eig_Tolerance", 1e-8 );
    esdfParam.eigMinTolerance      = esdf_double( "Eig_Min_Tolerance", 1e-3 );
    esdfParam.eigMinIter           = esdf_integer( "Eig_MinIter",  2 );
    esdfParam.eigMaxIter           = esdf_integer( "Eig_MaxIter",  10 );
    esdfParam.SVDBasisTolerance    = esdf_double( "SVD_Basis_Tolerance", 1e-6 );
    esdfParam.isUseAtomDensity = esdf_integer( "Use_Atom_Density", 1 );
    esdfParam.isRestartDensity = esdf_integer( "Restart_Density", 0 );
    esdfParam.isRestartWfn     = esdf_integer( "Restart_Wfn", 0 );
    esdfParam.isOutputDensity  = esdf_integer( "Output_Density", 0 );
    esdfParam.isOutputPotential  = esdf_integer( "Output_Potential", 0 );
    esdfParam.isOutputWfn      = esdf_integer( "Output_Wfn", 0 );
    esdfParam.isOutputALBElemLGL      = esdf_integer( "Output_ALB_Elem_LGL", 0 );
    esdfParam.isOutputALBElemUniform  = esdf_integer( "Output_ALB_Elem_Uniform", 0 );
    esdfParam.isOutputWfnExtElem      = esdf_integer( "Output_Wfn_ExtElem", 0 );
    esdfParam.isOutputPotExtElem      = esdf_integer( "Output_Pot_ExtElem", 0 );
    esdfParam.isOutputEigvecCoef      = esdf_integer( "Output_Eigvec_Coef", 0 );
    esdfParam.isCalculateAPosterioriEachSCF = esdf_integer( "Calculate_APosteriori_Each_SCF", 0 );
    esdfParam.isCalculateForceEachSCF       = esdf_integer( "Calculate_Force_Each_SCF", 0 );
    esdfParam.isOutputHMatrix  = esdf_integer( "Output_HMatrix", 0 );

    // Parameters related to Chebyshev Filtering in PWDFT
    // ~~**~~
    {
      esdfParam.First_SCF_PWDFT_ChebyFilterOrder = esdf_integer("First_SCF_PWDFT_ChebyFilterOrder", 40 );
      esdfParam.First_SCF_PWDFT_ChebyCycleNum =  esdf_integer("First_SCF_PWDFT_ChebyCycleNum", 5);
      esdfParam.General_SCF_PWDFT_ChebyFilterOrder = esdf_integer("General_SCF_PWDFT_ChebyFilterOrder", 35);
      esdfParam.PWDFT_Cheby_use_scala = esdf_integer("PWDFT_Cheby_use_scala", 1);
      esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt = esdf_integer("PWDFT_Cheby_use_wfn_ecut_filt",1);
    }



    esdfParam.ecutWavefunction     = esdf_double( "Ecut_Wavefunction", 40.0 );
    esdfParam.densityGridFactor    = esdf_double( "Density_Grid_Factor", 2.0 );

    // Choose the number of grid points based on ecut
    // The formula for the number of grid points along each dimension with
    // length L is
    //
    // 1/2 K_max^2 = Ecut,   with K_max = pi N_max / L, 
    //
    // i.e.
    //
    // N_max = \frac{\sqrt{2 E_cut} * L}{pi}.
    {
      Domain&  dm       = esdfParam.domain;

      for( Int d = 0; d < DIM; d++ ){
        // dm.numGrid[d] = AdjustNumGridOdd(std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) * 
        //      dm.length[d] / PI));
        dm.numGrid[d] = AdjustNumGrid(std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) * 
              dm.length[d] / PI));
        dm.numGridFine[d] = AdjustNumGrid(std::ceil(dm.numGrid[d] * esdfParam.densityGridFactor));
      } // for (d)
    }


    // The density grid factor must be an integer
    // esdfParam.densityGridFactor    = std::ceil( esdfParam.densityGridFactor );

    Real temperature;
    temperature               = esdf_double( "Temperature", 300.0 );
    esdfParam.Tbeta           = au2K / temperature;


    esdf_string("Smearing_Scheme", "FD", strtmp); 
    esdfParam.smearing_scheme = strtmp;

    esdfParam.numExtraState   = esdf_integer( "Extra_States",  0 );
    esdfParam.numUnusedState  = esdf_integer( "Unused_States",  0 );
    esdfParam.isEigToleranceDynamic = esdf_integer( "Eig_Tolerance_Dynamic", 1 );


    esdf_string("Pseudo_Type", "oncv", strtmp); 
    esdfParam.pseudoType      = strtmp;
    esdf_string("PW_Solver", "LOBPCG", strtmp); 
    esdfParam.PWSolver        = strtmp;
    esdf_string("XC_Type", "Teter", strtmp); 
    esdfParam.XCType          = strtmp;
    esdf_string("VDW_Type", "None", strtmp); 
    esdfParam.VDWType          = strtmp;
    esdfParam.numProcScaLAPACKPW  = esdf_integer( "Num_Proc_ScaLAPACK_PW", mpisize );
    esdfParam.scaBlockSize  = esdf_integer( "ScaLAPACK_Block_Size", 32 );
    esdfParam.extraElectron = esdf_integer( "Extra_Electron", 0);

    // PPCG 
    {
      esdfParam.PPCGsbSize = esdf_integer( "PPCG_sbSize", 1);
    }
  
  }




  // TDDFT
//  if( esdfParam.program == "tddft" ){
//    esdfParam.restartTDDFTStep   = esdf_integer( "Restart_TDDFT_Step", 0 );
//    esdfParam.TDDFTautoSaveSteps = esdf_integer( "TDDFT_AUTO_SAVE_STEP", 20);
//    esdfParam.isTDDFTEhrenfest   = esdf_integer( "TDDFT_EHRENFEST", 1); 
//    esdfParam.isTDDFTVext        = esdf_integer( "TDDFT_VEXT",   1); 
//    esdfParam.isTDDFTDipole      = esdf_integer( "TDDFT_DIPOLE",   1); 
//    esdfParam.TDDFTVextPolx      = esdf_double( "TDDFT_VEXT_POLX", 1.0);
//    esdfParam.TDDFTVextPoly      = esdf_double( "TDDFT_VEXT_POLY", 0.0);
//    esdfParam.TDDFTVextPolz      = esdf_double( "TDDFT_VEXT_POLZ", 0.0);
//    esdfParam.TDDFTVextFreq      = esdf_double( "TDDFT_VEXT_FREQ", 18.0/27.211385);
//    esdfParam.TDDFTVextPhase     = esdf_double( "TDDFT_VEXT_PHASE",0.0);
//    esdfParam.TDDFTVextAmp       = esdf_double( "TDDFT_VEXT_AMP",  0.0194);
//    esdfParam.TDDFTVextT0        = esdf_double( "TDDFT_VEXT_T0",   13.6056925);
//    esdfParam.TDDFTVextTau       = esdf_double( "TDDFT_VEXT_TAU",  13.6056925);
//
//    esdf_string("TDDFT_VEXT_ENV", "gaussian", strtmp); 
//    esdfParam.TDDFTVextEnv       = strtmp;
//    if(esdfParam.TDDFTVextEnv != "gaussian" &&
//        esdfParam.TDDFTVextEnv != "constant" &&
//        esdfParam.TDDFTVextEnv != "sinsq" &&
//        esdfParam.TDDFTVextEnv != "erf" &&
//        esdfParam.TDDFTVextEnv != "kick"){
//      ErrorHandling("Invalid VEXT Environment .");
//    }
//
//    esdf_string("TDDFT_Method", "PTTRAP", strtmp); 
//    esdfParam.TDDFTMethod        = strtmp;
//    if(esdfParam.TDDFTMethod != "PTTRAP" &&
//        esdfParam.TDDFTMethod != "RK4"   &&
//        esdfParam.TDDFTMethod != "PTTRAPDIIS" ) {
//      ErrorHandling("Invalid TDDFT method.");
//    }
//
//    esdfParam.TDDFTDeltaT            = esdf_double("TDDFT_DELTA_T",  1.0);
//    esdfParam.TDDFTTotalT            = esdf_double("TDDFT_TOTAL_T",  40.0);
//    esdfParam.TDDFTKrylovMax         = esdf_integer("TDDFT_KRYLOV_MAX", 30);
//    esdfParam.TDDFTKrylovTol         = esdf_double("TDDFT_KRYLOV_TOL",  1.0E-7);
//    esdfParam.TDDFTPhiTol            = esdf_double("TDDFT_PHI_TOL",  1.0E-7);
//    esdfParam.TDDFTDiisTol           = esdf_double("TDDFT_DIIS_TOL",  1.0E-5);
//    esdfParam.TDDFTPhiMaxIter        = esdf_integer("TDDFT_PHI_MAXITER", 20);
//    esdfParam.TDDFTDiisMaxIter       = esdf_integer("TDDFT_DIIS_MAXITER", 50);
//  }


  // DG

//  if( esdfParam.program == "dgdft" ){
//    Index3& numElem = esdfParam.numElem;
//    if (esdf_block("Element_Size",&nlines)) {
//      sscanf(block_data[0],"%d %d %d", 
//          &numElem[0],&numElem[1],&numElem[2]);
//    }
//    else{
//      ErrorHandling("Element_Size must be defined.");
//    }
//
//
//    // ScalES specific parameters for controlling convergence
//    esdfParam.scfInnerTolerance    = esdf_double( "SCF_Inner_Tolerance", 1e-4 );
//    esdfParam.scfInnerMinIter      = esdf_integer( "SCF_Inner_MinIter",   1 ); 
//    esdfParam.scfInnerMaxIter      = esdf_integer( "SCF_Inner_MaxIter",   1 );
//    esdfParam.scfOuterEnergyTolerance    = esdf_double( "SCF_Outer_Energy_Tolerance", 1e-4 );
//    esdfParam.scfOuterMinIter      = esdf_integer( "SCF_Outer_MinIter",   3 );
//
//    // Instead of grid size, use ecut to determine the number of grid
//    // points in the local LGL domain.
//    // The LGL grid factor does not need to be an integer.
//    esdfParam.LGLGridFactor = esdf_double( "LGL_Grid_Factor", 2.0 );
//
//
//    esdfParam.GaussInterpFactor = esdf_double( "Gauss_Interp_Factor", 4.0 );
//
//    esdfParam.GaussSigma = esdf_double( "Gauss_Sigma", 0.001 );
//
//    esdfParam.penaltyAlpha  = esdf_double( "Penalty_Alpha", 20.0 );
//
//
//
//    // ScalES requires the number of grids to be readjusted, so that the
//    // wavefunction grid and the density grid sizes are a multiple of
//    // the number of elements along each dimension.
//    //
//    {
//      Domain&  dm       = esdfParam.domain;
//      Index3&  numGridWavefunctionElem = esdfParam.numGridWavefunctionElem;
//      Index3&  numGridDensityElem = esdfParam.numGridDensityElem;
//      Index3&  numGridLGL = esdfParam.numGridLGL;
//      Index3   numElem = esdfParam.numElem;
//
//      Point3  elemLength;
//
//
//      for( Int d = 0; d < DIM; d++ ){
//        elemLength[d] = dm.length[d] / numElem[d];
//        // the number of grid is assumed to be at least an even number
//
//        numGridWavefunctionElem[d] = AdjustNumGrid(std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) * 
//              elemLength[d] / PI));
//
//        numGridDensityElem[d] = AdjustNumGrid(std::ceil(numGridWavefunctionElem[d] * esdfParam.densityGridFactor));
//
//        dm.numGrid[d] = numGridWavefunctionElem[d] * numElem[d];  // Coarse Grid
//        dm.numGridFine[d] = numGridDensityElem[d] * numElem[d]; // Fine Frid
//
//        numGridLGL[d] = std::ceil( numGridWavefunctionElem[d] * esdfParam.LGLGridFactor );
//      } // for (d)
//    }
//
//    // Get the number of basis functions per element
//    // NOTE: ALB_Num_Element overwrites the parameter numALB later        
//    {
//      esdfParam.numALBElem.Resize( numElem[0], numElem[1], numElem[2] );
//
//      Int sizeALBElem;
//
//      Int numALB        = esdf_integer( "ALB_Num", 4 );
//
//      if (esdf_block((char*)("ALB_Num_Element"),&sizeALBElem)) {
//        // Use different number of ALB functions for each element.
//        if( sizeALBElem != numElem.prod() ){
//          ErrorHandling(
//              "The size of the number of ALB does not match the number of elements.");
//        }
//        for( Int k = 0; k < numElem[2]; k++ )
//          for( Int j = 0; j < numElem[1]; j++ )
//            for( Int i = 0; i < numElem[0]; i++ ){
//              sscanf( block_data[i+j*numElem[0]+k*numElem[0]*numElem[1]],
//                  "%d", &esdfParam.numALBElem(i,j,k) );
//            }
//      }
//      else{
//        // Use the same number of ALB functions for each element.
//        for( Int k = 0; k < numElem[2]; k++ )
//          for( Int j = 0; j < numElem[1]; j++ )
//            for( Int i = 0; i < numElem[0]; i++ ){
//              esdfParam.numALBElem(i,j,k) = numALB;
//            }
//      }
//    }
//
//
//    // Modification of the potential in the extended element
//    {
//
//      // FIXME The potential barrier is now obsolete.
//      esdfParam.isPotentialBarrier   = esdf_integer( "Potential_Barrier",  0 );
//      esdfParam.potentialBarrierW    = esdf_double( "Potential_Barrier_W", 2.0 );
//      esdfParam.potentialBarrierS    = esdf_double( "Potential_Barrier_S", 0.0 );
//      esdfParam.potentialBarrierR    = esdf_double( "Potential_Barrier_R", 5.0 );
//
//      // Periodization of the external potential
//      esdfParam.isPeriodizePotential = esdf_integer( "Periodize_Potential", 0 );
//
//      esdfParam.distancePeriodize[0] = 0.0;
//      esdfParam.distancePeriodize[1] = 0.0;
//      esdfParam.distancePeriodize[2] = 0.0;
//
//      if( esdfParam.isPeriodizePotential ){
//        if( esdf_block("Distance_Periodize", &nlines) ){
//          sscanf(block_data[0],"%lf %lf %lf",
//              &esdfParam.distancePeriodize[0],
//              &esdfParam.distancePeriodize[1],
//              &esdfParam.distancePeriodize[2]);
//        }
//        else{
//          // Default value for DistancePeriodize
//          for( Int d = 0; d < DIM; d++ ){
//            if( esdfParam.numElem[d] == 1 ){
//              esdfParam.distancePeriodize[d] = 0.0;
//            }
//            else{
//              esdfParam.distancePeriodize[d] = 
//                esdfParam.domain.length[d] / esdfParam.numElem[d] * 0.5;
//            }
//          }
//        }
//      }
//    } // Modify the potential
//
//    esdf_string("Solution_Method", "diag", strtmp); 
//    esdfParam.solutionMethod  = strtmp;
//    if( esdfParam.solutionMethod != "diag" &&
//        esdfParam.solutionMethod != "pexsi" ){
//      ErrorHandling("Invalid solution method for the projected problem.");
//    }
//    if( esdfParam.solutionMethod == "pexsi" ){
//#ifndef _USE_PEXSI_
//      ErrorHandling("Usage of PEXSI requires -DPEXSI to be defined in make.inc.");
//#endif
//    }
//
//    esdf_string("Diag_Solution_Method", "scalapack", strtmp); 
//    esdfParam.diagSolutionMethod  = strtmp;
//    if( esdfParam.diagSolutionMethod != "elpa" &&
//        esdfParam.diagSolutionMethod != "scalapack" ){
//      ErrorHandling("Invalid Diag solution method for the projected problem.");
//    }
//
//    // FFT
//    // esdfParam.numProcDistFFT  = esdf_integer( "Num_Proc_DistFFT", mpisize );
//
//
//    // PEXSI parameters
//    esdfParam.numPole           = esdf_integer( "Num_Pole", 60 );
//    esdfParam.numProcRowPEXSI   = esdf_integer( "Num_Proc_Row_PEXSI", 1 );
//    esdfParam.numProcColPEXSI   = esdf_integer( "Num_Proc_Col_PEXSI", 1 );
//    esdfParam.npSymbFact        = esdf_integer( "Num_Proc_Symb_Fact", 
//        std::min( 4, esdfParam.numProcRowPEXSI * esdfParam.numProcColPEXSI ) );
//    esdfParam.energyGap         = esdf_double( "Energy_Gap", 0.0 );
//    esdfParam.spectralRadius    = esdf_double( "Spectral_Radius", 100.0 );
//    esdfParam.matrixOrdering    = esdf_integer( "Matrix_Ordering", 0 );
//    esdfParam.inertiaCountSteps = esdf_integer( "Inertia_Count_Steps", 10 );
//    esdfParam.maxPEXSIIter         = esdf_integer( "Max_PEXSI_Iter", 5 );
//    esdfParam.numElectronPEXSITolerance =
//      esdf_double( "Num_Electron_PEXSI_Tolerance", 1e-3 );
//    esdfParam.muInertiaTolerance =
//      esdf_double( "Mu_Inertia_Tolerance", 0.05 );
//    esdfParam.muInertiaExpansion =
//      esdf_double( "Mu_Inertia_Expansion", 0.3 );
//    esdfParam.muPEXSISafeGuard =
//      esdf_double( "Mu_PEXSI_SafeGuard", 0.05 );
//    esdfParam.muMin             = esdf_double( "Mu_Min", -2.0 );
//    esdfParam.muMax             = esdf_double( "Mu_Max", +2.0 );
//    esdfParam.pexsiMethod       = esdf_integer( "PEXSI_Method", 2);
//    esdfParam.pexsiNpoint       = esdf_integer( "PEXSI_Npoint", 2);
//
//    // Split MPI communicators into row and column communicators
//
//    Domain& dm = esdfParam.domain;
//
//    int dmCol = numElem[0] * numElem[1] * numElem[2];
//    int dmRow = mpisize / dmCol;
//
//    if(mpisize == 1){
//      dmCol = 1;
//      dmRow = 1;
//    }
//
//    if( (mpisize % dmCol) != 0 ){
//      std::ostringstream msg;
//      msg
//        << "(mpisize % dmCol) != 0" << std::endl;
//      ErrorHandling( msg.str().c_str() );
//    }
//
//    dm.comm    = MPI_COMM_WORLD;
//    MPI_Comm_split( dm.comm, mpirank / dmRow, mpirank, &dm.rowComm );
//    MPI_Comm_split( dm.comm, mpirank % dmRow, mpirank, &dm.colComm );
//
//    MPI_Barrier(dm.rowComm);
//    Int mpirankRow;  MPI_Comm_rank(dm.rowComm, &mpirankRow);
//    Int mpisizeRow;  MPI_Comm_size(dm.rowComm, &mpisizeRow);
//
//    MPI_Barrier(dm.colComm);
//    Int mpirankCol;  MPI_Comm_rank(dm.colComm, &mpirankCol);
//    Int mpisizeCol;  MPI_Comm_size(dm.colComm, &mpisizeCol);
//
//    // FFT
//    esdfParam.numProcDistFFT  = esdf_integer( "Num_Proc_DistFFT", mpisizeCol );
//
//    // ScaLAPACK parameter
//    esdfParam.numProcScaLAPACK    = esdf_integer( "Num_Proc_ScaLAPACK", mpisize );
//    if( esdfParam.numProcScaLAPACKPW > mpisizeRow )
//      esdfParam.numProcScaLAPACKPW = mpisizeRow;
//  } // DG



  // Ionic motion
  {
    // Both for geometry optimization and molecular dynamics
    // The default is 0, which means that only static calculation.
    esdfParam.ionMaxIter     = esdf_integer("Ion_Max_Iter", 0);
    esdf_string("Ion_Move", "", strtmp); 
    esdfParam.ionMove        = strtmp;
    std::vector<std::string> GeoOpt_list = { "bb", "pgbb", "nlcg", "lbfgs", "fire" };
    std::vector<std::string> MD_list = { "verlet", "nosehoover1", "langevin" };

    esdfParam.isGeoOpt_ = InArray(esdfParam.ionMove, GeoOpt_list);
    esdfParam.isMD_     = InArray(esdfParam.ionMove, MD_list);

    // Geometry optimization
    esdfParam.geoOptMaxForce = esdf_double( "Geo_Opt_Max_Force", 0.001 );

    // NLCG related parameters
    esdfParam.geoOpt_NLCG_sigma = esdf_double( "Geo_Opt_NLCG_Sigma", 0.02 );

    // FIRE related parameters
    esdfParam.FIRE_Nmin = esdf_integer( "FIRE_Nmin", 5 );		// Compare with LAMMPS
    esdfParam.FIRE_dt = esdf_double("FIRE_Time_Step", 41.3413745758); 	// usually between 0.1-1fs 
    esdfParam.FIRE_atomicmass = esdf_double("FIRE_Atomic_Mass", 4.0); 	// Compare with LAMMPS

    // Molecular dynamics
    Real ionTemperature;
    ionTemperature            = esdf_double( "Ion_Temperature", 300.0 );
    esdfParam.ionTemperature  = ionTemperature;
    esdfParam.TbetaIonTemperature   = au2K / ionTemperature;

    esdfParam.MDTimeStep  = esdf_double("MD_Time_Step", 40.0);
    esdf_string("MD_Extrapolation_Type", "linear", strtmp); 
    esdfParam.MDExtrapolationType          = strtmp;
    esdf_string("MD_Extrapolation_Variable", "density", strtmp); 
    esdfParam.MDExtrapolationVariable      = strtmp;
    esdfParam.qMass       = esdf_double("Thermostat_Mass", 85000.0);
    esdfParam.langevinDamping       = esdf_double("Langevin_Damping", 0.01);
    esdfParam.kappaXLBOMD           = esdf_double("kappa_XLBOMD", 1.70);
    esdfParam.isRestartPosition     = esdf_integer( "Restart_Position", 0 );
    esdfParam.isRestartVelocity     = esdf_integer( "Restart_Velocity", 0 );
    esdfParam.isOutputPosition      = esdf_integer( "Output_Position", 1 );
    esdfParam.isOutputVelocity      = esdf_integer( "Output_Velocity", 1 );
    esdfParam.isOutputXYZ           = esdf_integer( "Output_XYZ", 1 );

    // Energy based SCF convergence for MD: currently used in ScalES only
    esdfParam.MDscfEnergyCriteriaEngageIonIter = esdf_integer( "MD_SCF_energy_criteria_engage_ioniter", esdfParam.ionMaxIter + 1); 
    esdfParam.MDscfEtotdiff = esdf_double("MD_SCF_Etot_diff", esdfParam.scfOuterEnergyTolerance);
    esdfParam.MDscfEbanddiff = esdf_double("MD_SCF_Eband_diff", esdfParam.scfOuterEnergyTolerance);
    // Restart position / thermostat
  }

  // Inputs related to Chebyshev Filtered SCF iterations for DG
  // ~~**~~
  {
    // Basic parameters
    esdfParam.Diag_SCFDG_by_Cheby = esdf_integer( "Diag_SCFDG_by_Cheby", 0 );
    esdfParam.SCFDG_Cheby_use_ScaLAPACK = esdf_integer( "SCFDG_Cheby_use_ScaLAPACK", 0 );

    // First SCF step parameters
    esdfParam.First_SCFDG_ChebyFilterOrder = esdf_integer( "First_SCFDG_ChebyFilterOrder", 60 );
    esdfParam.First_SCFDG_ChebyCycleNum = esdf_integer( "First_SCFDG_ChebyCycleNum", 5 );

    // Second stage parameters
    esdfParam.Second_SCFDG_ChebyOuterIter = esdf_integer( "Second_SCFDG_ChebyOuterIter", 3 );
    esdfParam.Second_SCFDG_ChebyFilterOrder = esdf_integer( "Second_SCFDG_ChebyFilterOrder", 60 );
    esdfParam.Second_SCFDG_ChebyCycleNum = esdf_integer( "Second_SCFDG_ChebyCycleNum", 3);

    // General SCF step parameters
    esdfParam.General_SCFDG_ChebyFilterOrder = esdf_integer( "General_SCFDG_ChebyFilterOrder", 60);
    esdfParam.General_SCFDG_ChebyCycleNum = esdf_integer( "General_SCFDG_ChebyCycleNum", 1);

  }

  // **###**
  // Inputs related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in ScalES
  // ~~
  {
    esdfParam.scfdg_use_chefsi_complementary_subspace = esdf_integer("SCFDG_use_CheFSI_complementary_subspace", 0);
    esdfParam.scfdg_chefsi_complementary_subspace_syrk = esdf_integer("SCFDG_CheFSI_complementary_subspace_syrk", 0);
    esdfParam.scfdg_chefsi_complementary_subspace_syr2k = esdf_integer("SCFDG_CheFSI_complementary_subspace_syr2k", esdfParam.scfdg_chefsi_complementary_subspace_syrk);

    esdfParam.scfdg_complementary_subspace_nstates = esdf_integer("SCFDG_complementary_subspace_nstates", int(double(esdfParam.numExtraState)/20.0 + 0.5) );
    esdfParam.scfdg_cs_ioniter_regular_cheby_freq = esdf_integer("SCFDG_CS_ioniter_regular_Cheby_freq", 20 );

    esdfParam.scfdg_cs_bigger_grid_dim_fac = esdf_integer("SCFDG_CS_bigger_grid_dim_fac", 1 );

    // Inner LOBPCG related options
    esdfParam.scfdg_complementary_subspace_lobpcg_iter = esdf_integer("SCFDG_complementary_subspace_inner_LOBPCGiter", 15);
    esdfParam.scfdg_complementary_subspace_lobpcg_tol = esdf_double("SCFDG_complementary_subspace_inner_LOBPCGtol", 1e-8);

    // Inner CheFSI related options
    esdfParam.Hmat_top_states_use_Cheby = esdf_integer("SCFDG_complementary_subspace_use_inner_Cheby", 1);
    esdfParam.Hmat_top_states_ChebyFilterOrder =  esdf_integer("SCFDG_complementary_subspace_inner_Chebyfilterorder", 5);
    esdfParam.Hmat_top_states_ChebyCycleNum = esdf_integer("SCFDG_complementary_subspace_inner_Chebycyclenum", 3);
  }

  // Read position from lastPos.out into esdfParam.atomList[i].pos if isRestartPosition=1
  if(esdfParam.isRestartPosition){
    statusOFS << std::endl 
      << "Read in atomic position from lastPos.out, " << std::endl 
      << "override the atomic positions read from the input file." 
      << std::endl;

    std::vector<Atom>&  atomList = esdfParam.atomList;
    Int numAtom = atomList.size();
    DblNumVec atomposRead(3*numAtom);
    // Only master processor read and then distribute
    if( mpirank == 0 ){
      std::fstream fin;
      fin.open("lastPos.out", std::ios::in);
      if( !fin.good() ){
        ErrorHandling( "Cannot open lastPos.out!" );
      }
      for(Int a=0; a<numAtom; a++){
        fin>> atomposRead[3*a];
        fin>> atomposRead[3*a+1];
        fin>> atomposRead[3*a+2];
      }
      fin.close();
    }
    // Broadcast the atomic position
    MPI_Bcast( atomposRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    Point3 pos;
    for(Int a=0; a<numAtom; a++){
      pos = Point3( atomposRead[3*a], atomposRead[3*a+1], atomposRead[3*a+2] );
      atomList[a].pos = pos;
    }
  } //position read in for restart


  return ;
}        // -----  end of function ESDFReadInput  ----- 

void ESDFPrintInput( ){
  int  mpirank;  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank ); 
  int  mpisize;  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  // If the product of the number of elements is 1, recognize this as a PWDFT calculation

  PrintBlock(statusOFS, "System information");

  Print(statusOFS, "Program                              = ",  esdfParam.program );
  Print(statusOFS, "");
  Print(statusOFS, "Super cell                           = ",  esdfParam.domain.length );
  Print(statusOFS, "Grid Wavefunction                    = ",  esdfParam.domain.numGrid ); 
  Print(statusOFS, "Grid Density                         = ",  esdfParam.domain.numGridFine );
  Print(statusOFS, "");

  Print(statusOFS, "Temperature                          = ",  au2K / esdfParam.Tbeta, "[K]");
  Print(statusOFS, "Smearing scheme                      = ",  esdfParam.smearing_scheme );
  Print(statusOFS, "Extra states                         = ",  esdfParam.numExtraState  );
  Print(statusOFS, "Number of Extra Electron             = ",  esdfParam.extraElectron);
  Print(statusOFS, "");

  Print(statusOFS, "EcutWavefunction                     = ",  esdfParam.ecutWavefunction, "[au]");
  Print(statusOFS, "Density GridFactor                   = ",  esdfParam.densityGridFactor);
  Print(statusOFS, "");

  Print(statusOFS, "Pseudo Type                          = ",  esdfParam.pseudoType );
  Print(statusOFS, "XC Type                              = ",  esdfParam.XCType );
  Print(statusOFS, "Use Atom Density initially           = ",  esdfParam.isUseAtomDensity);
  Print(statusOFS, "Van der Waals type                   = ",  esdfParam.VDWType );
  Print(statusOFS, "");
  
  Print(statusOFS, "RestartDensity                       = ",  esdfParam.isRestartDensity);
  Print(statusOFS, "RestartWfn                           = ",  esdfParam.isRestartWfn);
  Print(statusOFS, "OutputDensity                        = ",  esdfParam.isOutputDensity);
  Print(statusOFS, "OutputPotential                      = ",  esdfParam.isOutputPotential);
  Print(statusOFS, "");
  Print(statusOFS, "FFTW  MPI Size                       = ",  esdfParam.fftwMPISize);
  Print(statusOFS, "");

  
  if( esdfParam.program == "pwdft" ){
    PrintBlock(statusOFS, "PWDFT information");
    
    Print(statusOFS, "Mixing dimension                     = ",  esdfParam.mixMaxDim );
    Print(statusOFS, "Mixing variable                      = ",  esdfParam.mixVariable );
    Print(statusOFS, "Mixing type                          = ",  esdfParam.mixType );
    Print(statusOFS, "Mixing Steplength                    = ",  esdfParam.mixStepLength);

    Print(statusOFS, "PW Solver                            = ",  esdfParam.PWSolver );

    Print(statusOFS, "SCF Tolerance                        = ",  esdfParam.scfOuterTolerance);
    Print(statusOFS, "SCF MaxIter                          = ",  esdfParam.scfOuterMaxIter);
    Print(statusOFS, "Eig Tolerence                        = ",  esdfParam.eigTolerance);
    Print(statusOFS, "Eig MaxIter                          = ",  esdfParam.eigMaxIter);
    Print(statusOFS, "Eig Min Tolerence                    = ",  esdfParam.eigMinTolerance);
    Print(statusOFS, "Eig Tolerance Dynamic                = ",  esdfParam.isEigToleranceDynamic);

    // Hybrid functional only
    Print(statusOFS, "");
    Print(statusOFS, "SCF Phi MaxIter                      = ",  esdfParam.scfPhiMaxIter);
    Print(statusOFS, "SCF Phi Tol                          = ",  esdfParam.scfPhiTolerance);
    Print(statusOFS, "Hybrid ACE                           = ",  esdfParam.isHybridACE);
    Print(statusOFS, "Hybrid DF                            = ",  esdfParam.isHybridDF);
    Print(statusOFS, "Hybrid Active Init                   = ",  esdfParam.isHybridActiveInit);
    Print(statusOFS, "Hybrid Mixing Type                   = ",  esdfParam.hybridMixType);
    Print(statusOFS, "EXX div type                         = ",  esdfParam.exxDivergenceType);

    if( esdfParam.isHybridDF ){
      Print(statusOFS, "Hybrid DF Num Mu                     = ",  esdfParam.hybridDFNumMu);
      Print(statusOFS, "Hybrid DF Num GaussianRandom         = ",  esdfParam.hybridDFNumGaussianRandom);
      Print(statusOFS, "Hybrid DF Tolerance                  = ",  esdfParam.hybridDFTolerance);
    }

    if( esdfParam.PWSolver == "LOBPCGScaLAPACK" ){
      Print(statusOFS, "Number of procs for ScaLAPACK (PW)   = ",  esdfParam.numProcScaLAPACKPW); 
      Print(statusOFS, "ScaLAPACK block                      = ",  esdfParam.scaBlockSize); 
    }
  } // PW


//  if(esdfParam.program == "tddft") {
//    PrintBlock(statusOFS, "TDDFT information");
//    Print(statusOFS, "TDDFT Method                         = ",  esdfParam.TDDFTMethod   );
//    Print(statusOFS, "TDDFT Ehrenfest dynamics             = ",  esdfParam.isTDDFTEhrenfest);
//    Print(statusOFS, "TDDFT Delta T                        = ",  esdfParam.TDDFTDeltaT   );
//    Print(statusOFS, "TDDFT Total T                        = ",  esdfParam.TDDFTTotalT   );
//    Print(statusOFS, "TDDFT Restart Step                   = ",  esdfParam.restartTDDFTStep);
//    Print(statusOFS, "TDDFT auto save for Restart          = ",  esdfParam.TDDFTautoSaveSteps);
//    Print(statusOFS, "TDDFT KRYLOV Iteration Max           = ",  esdfParam.TDDFTKrylovMax);
//    Print(statusOFS, "TDDFT KRYLOV Tolerance               = ",  esdfParam.TDDFTKrylovTol);
//    Print(statusOFS, "TDDFT V external                     = ",  esdfParam.isTDDFTVext   );
//    Print(statusOFS, "TDDFT Calculate Dipole               = ",  esdfParam.isTDDFTDipole );
//    Print(statusOFS, "TDDFT Environment                    = ",  esdfParam.TDDFTVextEnv  );
//    Print(statusOFS, "TDDFT Polarization X                 = ",  esdfParam.TDDFTVextPolx );
//    Print(statusOFS, "TDDFT Polarization Y                 = ",  esdfParam.TDDFTVextPoly );
//    Print(statusOFS, "TDDFT Polarization Z                 = ",  esdfParam.TDDFTVextPolz );
//    Print(statusOFS, "TDDFT V external Frequencey          = ",  esdfParam.TDDFTVextFreq );
//    Print(statusOFS, "TDDFT V external Phase               = ",  esdfParam.TDDFTVextPhase);
//    Print(statusOFS, "TDDFT V external Amplitude           = ",  esdfParam.TDDFTVextAmp  );
//    Print(statusOFS, "TDDFT V external T0                  = ",  esdfParam.TDDFTVextT0   );
//    Print(statusOFS, "TDDFT V external Tau                 = ",  esdfParam.TDDFTVextTau  );
//    Print(statusOFS, "TDDFT DIIS Tolerance                 = ",  esdfParam.TDDFTDiisTol  );
//    Print(statusOFS, "TDDFT Phi Tolerance                  = ",  esdfParam.TDDFTPhiTol   );
//    Print(statusOFS, "TDDFT DIIS MaxIter                   = ",  esdfParam.TDDFTDiisMaxIter);
//    Print(statusOFS, "TDDFT Phi MaxIter                    = ",  esdfParam.TDDFTPhiMaxIter);
//  }

//  if( esdfParam.program == "dgdft" ){
//    PrintBlock(statusOFS, "ScalES information");
//    Print(statusOFS, "Mixing dimension                     = ",  esdfParam.mixMaxDim );
//    Print(statusOFS, "Mixing variable                      = ",  esdfParam.mixVariable );
//    Print(statusOFS, "Mixing type                          = ",  esdfParam.mixType );
//    Print(statusOFS, "Mixing Steplength                    = ",  esdfParam.mixStepLength);
//    
//    Print(statusOFS, "PW Solver                            = ",  esdfParam.PWSolver );
//   
//    Print(statusOFS, "SCF Outer Tol                        = ",  esdfParam.scfOuterTolerance);
//    Print(statusOFS, "SCF Outer MaxIter                    = ",  esdfParam.scfOuterMaxIter);
//    Print(statusOFS, "SCF Free Energy Per Atom Tol         = ",  esdfParam.scfOuterEnergyTolerance);
//    Print(statusOFS, "Eig Min Tolerence                    = ",  esdfParam.eigMinTolerance);
//    Print(statusOFS, "Eig Tolerence                        = ",  esdfParam.eigTolerance);
//    Print(statusOFS, "Eig MaxIter                          = ",  esdfParam.eigMaxIter);
//    Print(statusOFS, "Eig Tolerance Dyn                    = ",  esdfParam.isEigToleranceDynamic);
//
//    
//    // FIXME Potentially obsolete potential barriers
//    Print(statusOFS, "Penalty Alpha                        = ",  esdfParam.penaltyAlpha );
//    Print(statusOFS, "Element size                         = ",  esdfParam.numElem ); 
//    Print(statusOFS, "Wfn Elem GridSize                    = ",  esdfParam.numGridWavefunctionElem );
//    Print(statusOFS, "Rho Elem GridSize                    = ",  esdfParam.numGridDensityElem ); 
//    Print(statusOFS, "LGL Grid size                        = ",  esdfParam.numGridLGL ); 
//    Print(statusOFS, "LGL GridFactor                       = ",  esdfParam.LGLGridFactor);
//
//    Print(statusOFS, "SVD Basis Tol                        = ",  esdfParam.SVDBasisTolerance);
//    Print(statusOFS, "SCF Inner Tol                        = ",  esdfParam.scfInnerTolerance);
//    Print(statusOFS, "SCF Inner MaxIter                    = ",  esdfParam.scfInnerMaxIter);
//    Print(statusOFS, "Num unused state                     = ",  esdfParam.numUnusedState);
//
//    statusOFS << "Number of ALB for each element: " << std::endl 
//      << esdfParam.numALBElem << std::endl;
//    Print(statusOFS, "Number of procs for DistFFT          = ",  esdfParam.numProcDistFFT ); 
//
//    Print(statusOFS, "Solution Method   = ",  esdfParam.solutionMethod );
//    if( esdfParam.solutionMethod == "diag" ){
//      Print(statusOFS, "Number of procs for ScaLAPACK        = ",  esdfParam.numProcScaLAPACK); 
//      Print(statusOFS, "Number of procs for ScaLAPACK (PW)   = ",  esdfParam.numProcScaLAPACKPW); 
//      Print(statusOFS, "ScaLAPACK block                      = ",  esdfParam.scaBlockSize); 
//    }
//    if( esdfParam.solutionMethod == "pexsi" ){
//      Print(statusOFS, "");
//      Print(statusOFS, "Number of poles                    = ",  esdfParam.numPole); 
//      Print(statusOFS, "Nproc row PEXSI                    = ",  esdfParam.numProcRowPEXSI); 
//      Print(statusOFS, "Nproc col PEXSI                    = ",  esdfParam.numProcColPEXSI); 
//      Print(statusOFS, "Nproc for symbfact                 = ",  esdfParam.npSymbFact); 
//      Print(statusOFS, "Energy gap                         = ",  esdfParam.energyGap); 
//      Print(statusOFS, "Spectral radius                    = ",  esdfParam.spectralRadius); 
//      Print(statusOFS, "Matrix ordering                    = ",  esdfParam.matrixOrdering); 
//      Print(statusOFS, "Inertia before SCF                 = ",  esdfParam.inertiaCountSteps);
//      Print(statusOFS, "MuMin0                             = ",  esdfParam.muMin); 
//      Print(statusOFS, "MuMax0                             = ",  esdfParam.muMax); 
//      Print(statusOFS, "NumElectron tol                    = ",  esdfParam.numElectronPEXSITolerance); 
//      Print(statusOFS, "mu Inertia tol                     = ",  esdfParam.muInertiaTolerance); 
//      Print(statusOFS, "mu Inertia expand                  = ",  esdfParam.muInertiaExpansion); 
//      Print(statusOFS, "Max PEXSI iter (deprecated)        = ",  esdfParam.maxPEXSIIter); 
//      Print(statusOFS, "mu PEXSI safeguard (deprecated)    = ",  esdfParam.muPEXSISafeGuard); 
//    }
//    // TODO Chebyshev
//
//    Print(statusOFS, "OutputALBElemLGL                   = ",  esdfParam.isOutputALBElemLGL);
//    Print(statusOFS, "OutputALBElemUniform               = ",  esdfParam.isOutputALBElemUniform);
//    Print(statusOFS, "OutputWfnExtElem                   = ",  esdfParam.isOutputWfnExtElem);
//    Print(statusOFS, "OutputPotExtElem                   = ",  esdfParam.isOutputPotExtElem);
//    Print(statusOFS, "OutputHMatrix                      = ",  esdfParam.isOutputHMatrix );
//
//
//    Print(statusOFS, "Force each step (deprecated)       = ",  
//        esdfParam.isCalculateForceEachSCF );
//
//    // FIXME A posteriori
//    Print(statusOFS, "A Posteriori error each step       = ",  
//        esdfParam.isCalculateAPosterioriEachSCF);
//
//
//    if( esdfParam.isPeriodizePotential ){
//      Print(statusOFS, "PeriodizePotential                 = ",  esdfParam.isPeriodizePotential);
//      Print(statusOFS, "DistancePeriodize                  = ",  esdfParam.distancePeriodize);
//    }
//
//    if( esdfParam.isPotentialBarrier ){
//      Print(statusOFS, "Potential Barrier = ",  esdfParam.isPotentialBarrier);
//      Print(statusOFS, "Barrier W         = ",  esdfParam.potentialBarrierW);
//      Print(statusOFS, "Barrier S         = ",  esdfParam.potentialBarrierS);
//      Print(statusOFS, "Barrier R         = ",  esdfParam.potentialBarrierR);
//    }
//    
//    // DG specific information on ion Move
//    
//    Print(statusOFS, "MD SCF Energy Criteria Engage Iter   = ",  esdfParam.MDscfEnergyCriteriaEngageIonIter);
//    Print(statusOFS, "MD SCF Etot diff                     = ",  esdfParam.MDscfEtotdiff);
//    Print(statusOFS, "MD SCF Eband diff                    = ",  esdfParam.MDscfEbanddiff);
//  } // DG
  
  // Ion move: geometry optimization or molecular dynamics
  if( esdfParam.isGeoOpt_ ){
    PrintBlock(statusOFS, "Geometry optimization information");
    Print(statusOFS, "Ion move mode                        = ",  esdfParam.ionMove);
    Print(statusOFS, "Output last position                 = ",  esdfParam.isOutputPosition );
    Print(statusOFS, "Output traj in XYZ format            = ",  esdfParam.isOutputXYZ );
    Print(statusOFS, "Max geometry optimization iter       = ",  esdfParam.ionMaxIter);
    Print(statusOFS, "RestartPosition                      = ",  esdfParam.isRestartPosition);
    Print(statusOFS, "GeoOpt force tol                     = ",  esdfParam.geoOptMaxForce );
    Print(statusOFS, "GeoOpt SCF MaxIter                   = ",  esdfParam.MDscfOuterMaxIter);
    Print(statusOFS, "GeoOpt extrapolation type            = ",  esdfParam.MDExtrapolationType);
    Print(statusOFS, "GeoOpt extrapolation variable        = ",  esdfParam.MDExtrapolationVariable);
    Print(statusOFS, "GeoOpt SCF Phi MaxIter               = ",  esdfParam.MDscfPhiMaxIter);
    if( esdfParam.ionMove == "fire" ){
      Print(statusOFS, "MD time step                         = ",  esdfParam.MDTimeStep * au2fs, "[fs]");
    }
    Print(statusOFS, "");
  }

  if( esdfParam.isMD_ ){
    PrintBlock(statusOFS, "Molecular dynamics information");
    Print(statusOFS, "Ion move mode                        = ",  esdfParam.ionMove);
    Print(statusOFS, "Number of MD steps                   = ",  esdfParam.ionMaxIter);
    Print(statusOFS, "MD time step                         = ",  esdfParam.MDTimeStep * au2fs, "[fs]");
    Print(statusOFS, "Output last position                 = ",  esdfParam.isOutputPosition );
    Print(statusOFS, "Output last velocity                 = ",  esdfParam.isOutputVelocity   );
    Print(statusOFS, "Output traj in XYZ format            = ",  esdfParam.isOutputXYZ );
    Print(statusOFS, "RestartPosition                      = ",  esdfParam.isRestartPosition);
    Print(statusOFS, "RestartVelocity                      = ",  esdfParam.isRestartVelocity);
    Print(statusOFS, "Ion Temperature                      = ",  esdfParam.ionTemperature, "[K]");
    Print(statusOFS, "MD extrapolation type                = ",  esdfParam.MDExtrapolationType);
    Print(statusOFS, "MD extrapolation variable            = ",  esdfParam.MDExtrapolationVariable);
    Print(statusOFS, "MD SCF MaxIter                       = ",  esdfParam.MDscfOuterMaxIter);
    Print(statusOFS, "MD SCF Phi MaxIter                   = ",  esdfParam.MDscfPhiMaxIter);
    
    if( esdfParam.ionMove == "nosehoover1" ){
      Print(statusOFS, "Thermostat mass                      = ",  esdfParam.qMass);
    }
    if( esdfParam.ionMove == "langevin" ){
      Print(statusOFS, "Langevin damping                     = ",  esdfParam.langevinDamping);
    }
    Print(statusOFS, "");
  }
  
  // Only master processor output information containing all atoms
  if( mpirank == 0 ){
    const std::vector<Atom>&  atomList = esdfParam.atomList;

    Print(statusOFS, ""); 
    Print(statusOFS, "NumAtom = ", (int)atomList.size()); 
    Print(statusOFS, "Atom Type and Coordinates (unit: au)");
    Print(statusOFS, ""); 

    for(Int i=0; i < atomList.size(); i++) {
      Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
    }
  }

  Print(statusOFS, ""); 

  return ;
}        // -----  end of function ESDFPrintInput  ----- 

Int AdjustNumGridOdd( Int numGrid ){
  Int numGridNew;
  Int nnn[100];
  nnn[ 0] = 1;
  nnn[ 1] = 3;
  nnn[ 2] = 5;
  nnn[ 3] = 7;
  nnn[ 4] = 9;
  nnn[ 5] = 15;
  nnn[ 6] = 21;
  nnn[ 7] = 25;
  nnn[ 8] = 27;
  nnn[ 9] = 35;
  nnn[ 10] = 45;
  nnn[ 11] = 49;
  nnn[ 12] = 63;
  nnn[ 13] = 75;
  nnn[ 14] = 81;
  nnn[ 15] = 105;
  nnn[ 16] = 125;
  nnn[ 17] = 135;
  nnn[ 18] = 147;
  nnn[ 19] = 175;
  nnn[ 20] = 189;
  nnn[ 21] = 225;
  nnn[ 22] = 245;
  nnn[ 23] = 315;
  nnn[ 24] = 343;
  nnn[ 25] = 375;
  nnn[ 26] = 441;
  nnn[ 27] = 525;
  nnn[ 28] = 625;
  nnn[ 29] = 735;
  nnn[ 30] = 875;
  nnn[ 31] = 1029;
  nnn[ 32] = 1225;
  nnn[ 33] = 1715;
  nnn[ 34] = 2401;

  for( int i = 1; i <= 33; i++)
  {
    if( nnn[i] >= numGrid && nnn[i+1] > numGrid ) {
//      numGridNew = IRound(nnn[i]/2.0)*2;
      numGridNew = nnn[i];
      break;
    }
  }

  return numGridNew;
}

Int AdjustNumGrid( Int numGrid ){
  Int numGridNew;
  Int nnn[300];
  nnn[0] =0;
  nnn[1] =1;
  nnn[2] =2;
  nnn[3] =3;
  nnn[4] =4;
  nnn[5] =5;
  nnn[6] =6;
  nnn[7] =7;
  nnn[8] =8;
  nnn[9] =9;
  nnn[10] =10;
  nnn[11] =12;
  nnn[12] =14;
  nnn[13] =15;
  nnn[14] =16;
  nnn[15] =18;
  nnn[16] =20;
  nnn[17] =21;
  nnn[18] =24;
  nnn[19] =25;
  nnn[20] =27;
  nnn[21] =28;
  nnn[22] =30;
  nnn[23] =32;
  nnn[24] =35;
  nnn[25] =36;
  nnn[26] =40;
  nnn[27] =42;
  nnn[28] =45;
  nnn[29] =48;
  nnn[30] =49;
  nnn[31] =50;
  nnn[32] =54;
  nnn[33] =56;
  nnn[34] =60;
  nnn[35] =63;
  nnn[36] =64;
  nnn[37] =70;
  nnn[38] =72;
  nnn[39] =75;
  nnn[40] =80;
  nnn[41] =81;
  nnn[42] =84;
  nnn[43] =90;
  nnn[44] =96;
  nnn[45] =98;
  nnn[46] =100;
  nnn[47] =105;
  nnn[48] =108;
  nnn[49] =112;
  nnn[50] =120;
  nnn[51] =125;
  nnn[52] =126;
  nnn[53] =128;
  nnn[54] =135;
  nnn[55] =140;
  nnn[56] =144;
  nnn[57] =147;
  nnn[58] =150;
  nnn[59] =160;
  nnn[60] =162;
  nnn[61] =168;
  nnn[62] =175;
  nnn[63] =180;
  nnn[64] =189;
  nnn[65] =192;
  nnn[66] =196;
  nnn[67] =200;
  nnn[68] =210;
  nnn[69] =216;
  nnn[70] =224;
  nnn[71] =225;
  nnn[72] =240;
  nnn[73] =243;
  nnn[74] =245;
  nnn[75] =250;
  nnn[76] =252;
  nnn[77] =256;
  nnn[78] =270;
  nnn[79] =280;
  nnn[80] =288;
  nnn[81] =294;
  nnn[82] =300;
  nnn[83] =315;
  nnn[84] =320;
  nnn[85] =324;
  nnn[86] =336;
  nnn[87] =343;
  nnn[88] =350;
  nnn[89] =360;
  nnn[90] =375;
  nnn[91] =378;
  nnn[92] =384;
  nnn[93] =392;
  nnn[94] =400;
  nnn[95] =405;
  nnn[96] =420;
  nnn[97] =432;
  nnn[98] =441;
  nnn[99] =448;
  nnn[100] =450;
  nnn[101] =480;
  nnn[102] =486;
  nnn[103] =490;
  nnn[104] =500;
  nnn[105] =504;
  nnn[106] =512;
  nnn[107] =525;
  nnn[108] =540;
  nnn[109] =560;
  nnn[110] =567;
  nnn[111] =576;
  nnn[112] =588;
  nnn[113] =600;
  nnn[114] =625;
  nnn[115] =630;
  nnn[116] =640;
  nnn[117] =648;
  nnn[118] =672;
  nnn[119] =675;
  nnn[120] =686;
  nnn[121] =700;
  nnn[122] =720;
  nnn[123] =729;
  nnn[124] =735;
  nnn[125] =750;
  nnn[126] =756;
  nnn[127] =768;
  nnn[128] =784;
  nnn[129] =800;
  nnn[130] =810;
  nnn[131] =840;
  nnn[132] =864;
  nnn[133] =875;
  nnn[134] =882;
  nnn[135] =896;
  nnn[136] =900;
  nnn[137] =945;
  nnn[138] =960;
  nnn[139] =972;
  nnn[140] =980;
  nnn[141] =1000;
  nnn[142] =1008;
  nnn[143] =1024;
  nnn[144] =1029;
  nnn[145] =1050;
  nnn[146] =1080;
  nnn[147] =1120;
  nnn[148] =1125;
  nnn[149] =1134;
  nnn[150] =1152;
  nnn[151] =1176;
  nnn[152] =1200;

  for( int i = 1; i <= 151; i++)
  {
    if( nnn[i] >= numGrid && nnn[i+1] > numGrid ) {
//      numGridNew = IRound(nnn[i]/2.0)*2;
      numGridNew = nnn[i];
      break;
    }
  }

  return numGridNew;
}        // -----  end of function AdjustNumGrid      ----- 



} // namespace esdf
} // namespace scales
