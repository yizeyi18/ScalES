#! /bin/bash
cd ../src; make -j 12; cd ../examples; rm $1; rm $1.o; make $1 
