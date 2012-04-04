#/bin/bash

# Print out the Helmholtz free energy of the system for a bunch of RUN
# directories, from RUNs to RUNe, (s and e given by input variable) The
# format is for ABINIT

for (( i=$1; i<=$2; i++ )); do
  echo RUN$i
  awk '/Etotal/{etot=$3;}END{print etot;}' RUN$i/*.out
done
