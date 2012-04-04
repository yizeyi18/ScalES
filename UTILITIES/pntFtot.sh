#/bin/bash

# Print out the total energy of the system for a bunch of RUN
# directories, from RUNs to RUNe, (s and e given by input variable)

for (( i=$1; i<=$2; i++ )); do
  echo RUN$i
  awk '/Helmholtz/{egy=$3;}END{print egy;}' RUN$i/statfile.100000
done
