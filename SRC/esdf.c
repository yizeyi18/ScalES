/* ===============================================================
   E l e c t r o n i c   S t r u c t u r e   D a t a   F o r m a t
   =============================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "esdf.h"
#include "mpi.h"

#define llength 80              /* length of the lines */
#define numkw 200               /* maximum number of keywords */

char **block_data;
int nrecords;
int nwarns;
char **llist;
char **warns;
char ***tlist;
char phy_d[nphys][11];          /* D - dimension */
char phy_n[nphys][11];          /* N - name */
double phy_u[nphys];            /* U - unit */

char kw_label[numkw][80];
int kw_index[numkw];
char kw_typ[numkw][4];
char kw_dscrpt[numkw][3000];

FILE *fileunit;

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
    strcpy(kw_dscrpt[i],"*! Number of atom types !*");

    i++;
    strcpy(kw_label[i],"atom_type");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Type of the atom !*");

    i++;
    strcpy(kw_label[i],"core_cutoff_radius");
    strcpy(kw_typ[i],"P:E");
    strcpy(kw_dscrpt[i],"*! Pseudopotential core cutoff radius !");

    i++;
    strcpy(kw_label[i],"potential_num");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of potentials !*");

    i++;
    strcpy(kw_label[i],"local_component");
    strcpy(kw_typ[i],"T:B");
    strcpy(kw_dscrpt[i],"*! The local component (s,p or d) !*");

    i++;
    strcpy(kw_label[i],"move_flag");
    strcpy(kw_typ[i],"I:I");
    strcpy(kw_dscrpt[i],"*! Which atom to move(all,some,first n) !*");

    i++;
    strcpy(kw_label[i],"lattice_vector_scale");
    strcpy(kw_typ[i],"P:D");
    strcpy(kw_dscrpt[i],"*! Unit for lattice vectors !*");

    i++;
    strcpy(kw_label[i],"correlation_type");
    strcpy(kw_typ[i],"T:D");
    strcpy(kw_dscrpt[i],"*! Correlation type !*");

    i++;
    strcpy(kw_label[i],"ion_energy_diff");
    strcpy(kw_typ[i],"P:I");
    strcpy(kw_dscrpt[i],"*! Ion energy diff. (only for 'cc') !*");

    i++;
    strcpy(kw_label[i],"atom_coord");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Coordinates of atom !*");

    i++;
    strcpy(kw_label[i],"grid_size");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Number of grids on each direction !*");

    i++;
    strcpy(kw_label[i],"super_cell");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Length of sides of periodic box !*");

    i++;
    strcpy(kw_label[i],"grid_spacing");
    strcpy(kw_typ[i],"P:E");
    strcpy(kw_dscrpt[i],"*! Grid spacing (h) !*");


    i++;
    strcpy(kw_label[i],"maxiter");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum iterations !*");

    i++;
    strcpy(kw_label[i],"tolerance");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Tolerance !*");

    i++;
    strcpy(kw_label[i],"cheb_deg");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Degree of Chebyshev polynomial !*");

    i++;
    strcpy(kw_label[i],"mixing_type");
    strcpy(kw_typ[i],"T:B");
    strcpy(kw_dscrpt[i],"*! Simple, Anderson, Broyden, Purlay !*");

    i++;
    strcpy(kw_label[i],"mixing_param");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Coefficients parameter (simple) !*");


    i++;
    strcpy(kw_label[i],"max_mixing");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum mixing number (Broyden, Pulay) !*");

    i++;
    strcpy(kw_label[i],"log_files");
    strcpy(kw_typ[i],"L:B");
    strcpy(kw_dscrpt[i],"*! Output data in different files !*");

    /* LL: Keywords added below for DGDFT */
    i++;
    strcpy(kw_label[i],"mixing_alpha");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Coefficients parameter !*");

    i++;
    strcpy(kw_label[i],"dg_alpha");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! DG penalty parameter !*");

    i++;
    strcpy(kw_label[i],"eig_tolerance");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Inner loop tolerance!*");

    i++;
    strcpy(kw_label[i],"scf_tolerance");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Outer loop tolerance!*");

    i++;
    strcpy(kw_label[i],"temperature");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! temperature (in Kelvin)!*");

    i++;
    strcpy(kw_label[i],"eig_maxiter");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum iteration number for inner loop !*");

    i++;
    strcpy(kw_label[i],"scf_maxiter");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum iteration number for outer loop !*");
    
    i++;
    strcpy(kw_label[i],"dg_degree");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum polynomial degrees in DG solver !*");

    i++;
    strcpy(kw_label[i],"enrich_number");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of enrichment DOF per element !*");
    
    i++;
    strcpy(kw_label[i],"mb");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of block size used in scalapack!*");

    i++;
    strcpy(kw_label[i],"mapping_mode");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Mapping Mode - self or uniform!*");

    i++;
    strcpy(kw_label[i],"element_size");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! number of elements per dimension !*");

    i++;
    strcpy(kw_label[i],"extra_states");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Extra number of states beyond occupied states !*");

    i++;
    strcpy(kw_label[i],"periodtable");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! file name for periodic table!*");

    i++;
    strcpy(kw_label[i],"pseudo_type");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Type of pseudopotential!*");

    i++;
    strcpy(kw_label[i],"output_dir");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Output Directory!*");

    i++;
    strcpy(kw_label[i],"restart_mode");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Restart mode !*");

    i++;
    strcpy(kw_label[i],"restart_density");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Density file to restart!*");


    i++;
    strcpy(kw_label[i],"output_density");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! whether density is outputed !*");

    i++;
    strcpy(kw_label[i],"restart_wave_mode");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Restart mode !*");

    i++;
    strcpy(kw_label[i],"restart_wave");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Wave function file to restart!*");


    i++;
    strcpy(kw_label[i],"output_wfn");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! whether wavefunctions and occupations numbers are outputed !*");


    i++;
    strcpy(kw_label[i],"element_position_start");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule element !*");

    i++;
    strcpy(kw_label[i],"element_grid_size");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule element !*");

    i++;
    strcpy(kw_label[i],"element_cell");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule element !*");

    i++;
    strcpy(kw_label[i],"buffer_position_start");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule buffer !*");

    i++;
    strcpy(kw_label[i],"buffer_grid_size");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule buffer !*");

    i++;
    strcpy(kw_label[i],"buffer_cell");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of each molecule buffer !*");

    i++;
    strcpy(kw_label[i],"buffer_atom_types_num");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of atom types !*");

    i++;
    strcpy(kw_label[i],"buffer_atom_type");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Type of the atom !*");

    i++;
    strcpy(kw_label[i],"buffer_atom_coord");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Coordinates of atom !*");

    i++;
    strcpy(kw_label[i],"position_start");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Starting position of molecule !*");

    i++;
    strcpy(kw_label[i],"buffer_atom_mode");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! Mode for generating atom positions in the buffer!*");

    i++;
    strcpy(kw_label[i],"barriera");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Barrier height in the buffer !*");

    i++;
    strcpy(kw_label[i],"barrierd");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Barrier starting position in the buffer !*");

    i++;
    strcpy(kw_label[i],"time_step");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Time step for MD simulation!*");
   
    i++;
    strcpy(kw_label[i],"max_step");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Maximum steps for MD simulation!*");
   
    i++;
    strcpy(kw_label[i],"buff_update");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Basis functions are updated every buff_update steps !*");

    i++;
    strcpy(kw_label[i],"statfile");
    strcpy(kw_typ[i],"T:B");
    strcpy(kw_dscrpt[i],"*! Filename of statfile!*");

    i++;
    strcpy(kw_label[i],"dg_solver");
    strcpy(kw_typ[i],"T:B");
    strcpy(kw_dscrpt[i],"*! standard (std) or Nonorthogonal (nonorth) !*");

    i++;
    strcpy(kw_label[i],"wallwidth");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Wall width for the Frobenius penalty !*");

    i++;
    strcpy(kw_label[i],"weightratio");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Weight ratio between eigenvalue and penalty!*");
    
    i++;
    strcpy(kw_label[i],"eigperele");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of eigenvalues to be solved per element in the buffer!*");

    i++;
    strcpy(kw_label[i],"orbperele");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Number of nonorthogonal oritals in the element!*");
    
    i++;
    strcpy(kw_label[i],"input_format");
    strcpy(kw_typ[i],"T:B");
    strcpy(kw_dscrpt[i],"*!Input format for md.in !*");
   
    i++;
    strcpy(kw_label[i],"extended_element_ratio");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*!Size of the extended element !*");
    
    i++;
    strcpy(kw_label[i],"output_bases");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! whether bases functions are outputed !*");
    
    i++;
    strcpy(kw_label[i],"basis_radius");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! Radius of nonlocal adaptive local basis functions !*");

    i++;
    strcpy(kw_label[i],"atom_cart");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Coordinates of atom in Cartesian unit (Bohr)!*");
    
    i++;
    strcpy(kw_label[i],"atom_red");
    strcpy(kw_typ[i],"B:E");
    strcpy(kw_dscrpt[i],"*! Coordinates of atom in reduced unit!*");

    i++;
    strcpy(kw_label[i],"buf_dual");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! Use dual grid for buffer solve !*");

    i++;
    strcpy(kw_label[i],"vext");
    strcpy(kw_typ[i],"T:E");
    strcpy(kw_dscrpt[i],"*! file name for external potential!*");

    i++;
    strcpy(kw_label[i],"output_vtot");
    strcpy(kw_typ[i],"I:E");
    strcpy(kw_dscrpt[i],"*! whether the total potential are outputed !*");

    i++;
    strcpy(kw_label[i],"deltafermi");
    strcpy(kw_typ[i],"D:E");
    strcpy(kw_dscrpt[i],"*! increase of the Fermi energy to control the number of candidate functions!*"); 
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
void esdf_bcast(int myid, int MASTER){
  int i, j;
  MPI_Bcast(&nrecords, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&nwarns,   1, MPI_INT, MASTER, MPI_COMM_WORLD);
  if( myid != MASTER ){
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
void esdf_init(char *fname) {
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
                if (strcmp(esdf_reduce(tlist[i][1]),
                    esdf_reduce(tlist[j][1]))==0) ndef++;
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
void esdf_string(char *labl,char *def,char *out) {
    /* Return the string attached to the "label" */
    int i,kw_number;
    int ind;
    char *pout;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"T",&kw_number);

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
int esdf_integer(char *labl,int def) {
    /* Return the integer value attached to the "label" */
    int i;
    char ctemp[llength];
    int kw_number=0;
    int out;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"I",&kw_number);
        
    /* Set to default */
    out=def;

    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
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
float esdf_single(char *labl,float def) {
    /* Return the single precisioned value attached to the "label" */
    float out;
    int i;
    char ctemp[llength];
    int kw_number;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"S",&kw_number);

    /* Set to default */
    out=def;
    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
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
double esdf_double(char *labl,double def) {
    /* Return the double precisioned value attached to the "label" */
    int i;
    char ctemp[llength];
    int kw_number;     
    double out;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"D",&kw_number);

    /* Set to default */
    out=def;
    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
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
double esdf_physical(char *labl,double def,char *dunit) {
    /* Return the double precisioned physical value attached to the "label"
       units converted to "dunit"
     */
    int i,j;
    char ctemp[llength],iunit[llength];
    int kw_number;
    double out;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"P",&kw_number);

    /* Set to default */
    out=def;

    for (i=0;i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
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
bool esdf_defined(char *labl) {
    /* Is the "label" defined in the input file */
    int i;
    int kw_number;
    bool out;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"E",&kw_number);
    
    /* Set to default */
    out=0; /* false */

    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
            out=1; /* true */
            kw_index[kw_number]=i+1;
            break;
        }
    }

    return out;
}

/*   --------------  esdf_boolean  ----------------------  */
bool esdf_boolean(char *labl,bool *def) {
    /* Is the "label" defined in the input file */
    int i;
    char positive[3][llength],negative[3][llength];
    int kw_number;
    bool out;
    char label[llength];

    strcpy(label,labl);
    strcpy(positive[0],"yes");
    strcpy(positive[1],"true");
    strcpy(positive[2],"t");
    strcpy(negative[0],"no");
    strcpy(negative[1],"false");
    strcpy(negative[2],"f");

    /* Check "label" is defined */
    esdf_lablchk(label,"L",&kw_number);

    /* Set to default */
    out=*def;

    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
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
            esdf_die("Unable to parse boolean value");
        }
    }

    return out;
}

/*   --------------  esdf_block  ----------------------  */
bool esdf_block(char *labl,int *nlines) {
    int i;
    char ctemp[llength];
    int kw_number;
    bool out;
    char label[llength];

    strcpy(label,labl);
    /* Check "label" is defined */
    esdf_lablchk(label,"B",&kw_number);
    strcpy(ctemp,"Block \"");
    strcat(ctemp,esdf_reduce(label));
    strcat(ctemp,"\" not closed correctly");

    out=0; /* false */
    (*nlines)=0;

    for (i=kw_index[kw_number];i<nrecords;i++) {
        /* Search in the first token for "label"
           the first instance is returned */
        if ((strcmp(esdf_reduce(tlist[i][0]),"begin")==0) &&
            (strcmp(esdf_reduce(tlist[i][1]),esdf_reduce(label))==0)) {
            out=1; /* true */
            kw_index[kw_number]=i+1;
            while (strcmp(esdf_reduce(tlist[i+(*nlines)+1][0]),"end")!=0) {
                (*nlines)++;
                if ((*nlines)+i>nrecords) esdf_die(ctemp);
                strcpy(block_data[(*nlines)-1],llist[i+(*nlines)]);
            }
            if (strcmp(esdf_reduce(tlist[i+(*nlines)+1][1]),
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

    if ((fileunit=fopen(trim(filename),"r"))==NULL)
        (*ierr)=1;
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
