#!/bin/bash
module load esslurm
salloc -C gpu -N 1 --exclusive -t 01:30:30 --gres=gpu:8 -A m1759
