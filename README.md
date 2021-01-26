Scalable Electronic Structure (ScalES)
=====================================

Installation       
============


Edit make.inc
-------------

Configuration of ScalES is controlled by a single `make.inc` file.
Examples of the `make.inc` file are given under the `config/` directory.

Find `make.inc` with the most similar architecture, and copy to the main
PEXSI directory (using Edison for example, the latest Intel computer
at NERSC).  `${ScalES_DIR}` stands for the main directory of PEXSI.

    cd ${ScalES_DIR}
    cp config/make.inc.edison make.inc

Edit the variables in make.inc. 
    
    ScalES_DIR     = Main directory for ScalES

ScalES can be compiled using `debug` or `release` mode in
by the variable `COMPILE_MODE` in `make.inc`.  This variable mainly controls the
compiling flag `-DRELEASE`.  The `debug` mode introduces tracing of call
stacks at all levels of functions, and may significantly slow down the
code.  For production runs, use `release` mode.

The `*.profile` configuration files are for debugging purpose and
can be ignored.

Build the ScalES library
------------------------

If make.inc is configured correctly,
    
    cd ${ScalES_DIR}
    cd src
    make

should produce `libscales.a` under `src/`.

Build examples
--------------

After `libpexsi.a` is built, the main routine `scales` should be readily
to be compiled.  For example, the selected inversion for a complex
matrix has the test routine

    cd ${ScalES_DIR}
    cd examples
    make scales

should produce `scales`, which can be executed with MPI.

Tests
-----

To be added.

Use of PEXSI
------------

To be added.
