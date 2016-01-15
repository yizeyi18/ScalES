TODO List   {#pageTODO}
=========
@todo
- A better way to handle various exchange correlation functionals
- Merge the treatment of XC functionals in Hamiltonian and in KohnSham
- Remove the BLOPEX dependence
- Combine PWDFT_bb etc into something called move_ions and move_ions_dg
- OpenMP does not work for the new spinor multiplication due to the
  common dependence on the Fourier structure
- Remove the unnecessary part of distinguishing coarse / fine grid, and
  document this a bit more properly. It is even possible to revamp the
  implementation by storing the wavefunction coefficients in the complex
  arithmetic
- Better way to handle error: handling function taking a message as
  input is a more versatile way for handling error messaging. callstack
  procedure is slow and does not work for openmp. The DEBUG mode is too
  slow due to push/popstacks
- Simplify the input parameters. Spinor class should be removed and
  moved to the Hamiltonian class. In the future different types of
  spinors should be treated with different classes of "Hamiltonian". The
  functions in spinor, such as preconditioners should also be moved to
  the Hamiltonian class (or KohnSham).
- SCF, eigensolver supports multiple types of classes. Details in
  different realization of the Hamiltonian class.
- The new design should be combined with the design of spin
  polarization. This design instead should leave room for k-point
  implementation.
- Refine Fourier to clean the normalization factors. Encapsulate the
  forward and backward Fourier transforms?
- The "SCALAR" design should be kept?
