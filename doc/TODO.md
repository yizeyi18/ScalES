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
- Better way to handle error: dumpcallstack should be used. Output the
  error message via cerr, and then abort for being captured by coredump.
  But if core dump is available, why callstacks? No matter what, an
  handling function taking a message as input is a more versatile way
  for handling error messaging.
- Simplify the input parameters, make Spinor a struct instead of a
  class. Move the matrix-vector multiplication to the Hamiltonian class.
- The new design should be combined with the design of spin
  polarization. This design instead should leave room for k-point
  implementation.
- Refine Fourier to clean the normalization factors. Encapsulate the
  forward and backward Fourier transforms?
