#!/bin/bash
for (( i=$1; i<=$2; i++ )); do
  qsub run$i.pbs
done
