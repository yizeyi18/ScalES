#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
namespace dgdft{
namespace esdf{

  const int nphys = 57;
  const int llength = 80;  /* length of the lines */
  const int numkw = 200;   /* maximum number of keywords */


  
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
  
  /************************************************************ 
   * Main routines
   ************************************************************/
  void esdf_bcast(int, int);
  void esdf_key();
  void esdf_init(char *);
  void esdf_string(char *, char *, char *);
  int esdf_integer(char *, int);
  float esdf_single(char *, float);
  double esdf_double(char *, double);
  double esdf_physical(char *, double, char *);
  bool esdf_defined(char *);
  bool esdf_boolean(char *, bool *);
  bool esdf_block(char *, int *);
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

} // namespace esdf
} // namespace dgdft
#endif // _ESDF_HPP_
