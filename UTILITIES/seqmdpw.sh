#!/bin/bash
iproc=0;
for i in `ls mdpw.*.in`; do
  if [[ $(( iproc % 8 )) == 0 ]] ; then
    wait
    let iproc=0;
  fi
  mpirun -n 1 ../../mdpw.x $i </dev/null > $i".out" 2>&1 &
  let iproc=$iproc+1;
done
wait
cat statfile.*.100000 > shortstat_pw

