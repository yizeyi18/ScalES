#!/bin/bash

# This is a template to generate a series of RUN directories to test
# various combinations of parameters.
#
# This is for the case with 1 tuning parameters. 
#
# Lin Lin
# 12/05/2011

source headgen.sh  #Read parameters
for (( i=1; i<=${nvarptr[1]}; i++ )); do
  for (( j=1; j<=${nvarptr[2]}; j++ )); do
    rundir="RUN"${runcnt}
    outfname=${rundir}"/"${fname}
    mkdir ${rundir}
    touch ${outfname}
    echo "#Tuning parameters" >> ${outfname}
    if [[ ${isblock[1]} == 1 ]]; then
      echo "begin "${attrb[1]} >> ${outfname}
      echo "  " ${val1[i]} >> ${outfname}
      echo "end "${attrb[1]} >> ${outfname}
    else
      echo ${attrb[1]} " " ${val1[i]} >> ${outfname}
    fi
    
    echo >> ${outfname}
    
    if [[ ${isblock[2]} == 1 ]]; then
      echo "begin "${attrb[2]} >> ${outfname}
      echo "  " ${val2[j]} >> ${outfname}
      echo "end "${attrb[2]} >> ${outfname}
    else
      echo ${attrb[2]} " " ${val2[j]} >> ${outfname}
    fi
    
    echo >> ${outfname}
    echo "#From the template file" >> ${outfname}
    cat ${fname} >> ${outfname}
    cp ${relfile} ${rundir}
    runcnt=$((${runcnt}+1));
  done
done
