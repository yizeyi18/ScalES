#/bin/bash

# Print out the total energy of the system for a bunch of RUN
# directories, from RUNs to RUNe, (s and e given by input variable)
# The format is for ABINIT

for (( i=$1; i<=$2; i++ )); do
  echo RUN$i
  awk '/Internal/{egy=$4;}END{print egy;}' RUN$i/*.out
done
