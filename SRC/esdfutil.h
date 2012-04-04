#ifndef __FORTRAN_UTIL_2008_05_12
#define __FORTRAN_UTIL_2008_05_12

#ifdef __cplusplus
extern "C"{
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


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

#ifdef __cplusplus
}
#endif
#endif
