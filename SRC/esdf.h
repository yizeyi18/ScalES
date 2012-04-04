#ifndef __ESDF_2008_05_12
#define __ESDF_2008_05_12

#ifdef __cplusplus
extern "C"{
#endif

#include "esdfutil.h"

#define nphys 57

#ifndef __cplusplus
typedef int bool;
#endif

char **block_data;
char phy_d[nphys][11];
char phy_n[nphys][11];
double phy_u[nphys];

/*   Modified by Lin Lin, Nov 9, 2010                   */
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

#ifdef __cplusplus
}
#endif
#endif
