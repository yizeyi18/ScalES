cd external/lbfgs
make cleanall && make
cd ../rqrcp
make cleanall && make
cd ../blopex/blopex_abstract
make clean && make
cd ../../../src
make cleanall && make -j
cd ../examples
make cleanall && make pwdft && make dgdft
