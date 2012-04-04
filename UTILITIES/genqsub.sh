#/bin/bash
# GENQSUB generates the qsub script for a series of RUNxx directories.
#
# The first and second parameter in the command line input gives the
# starting and ending number for the directories
#
# Lin Lin
# 12/04/2011

source headqsub.sh
for (( i=$1; i<=$2; i++ )); do
  outfname="run"${i}".pbs"
  touch ${outfname}
  echo "#!/bin/bash" >> ${outfname};
  echo "#PBS -N "${projhead}"_RUN"${i} >> ${outfname};
  cat ${fname} >> ${outfname}
  echo "cd RUN"${i}  >> ${outfname};
  echo "aprun "${procopt} >> ${outfname}; 
  echo "cd .." >> ${outfname};
done
