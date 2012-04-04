#include "esdfutil.h"

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
    for (pin=in+i,pout=out;*pout=*pin;pin++,pout++);
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
