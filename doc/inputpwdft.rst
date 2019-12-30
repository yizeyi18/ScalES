Input parameters for PWDFT
--------------------------





.. list-table:: All parameters for PWDFT (listed alphabetically)
   :widths: 1 1 1 3
   :header-rows: 1

   * - Keyword
     - Type
     - Default value (unit)
     - Description


   * - Atom_Ang
     - block
     - Provided by user
     - size: (Natom,3). Unit: Angstrom. Natom is the number of atoms for a given Atom_type

   * - Atom_Bohr
     - block
     - Provided by user
     - size: (Natom,3). Unit: Bohr. Natom is the number of atoms for a given Atom_type

   * - Atom_Red
     - block
     - Provided by user
     - size: (Natom,3). Unit: relative (also called crystal coordinate). Natom is the number of atoms for a given Atom_type.

       .. note::
            For each type of the atom, there should be one (and only one) ``Atom_Ang``, ``Atom_Bohr``, ``Atom_Red`` block.

   * - Atom_Type
     - integer
     - Must be provided by user.
     - Atomic number of a given species.


   * - Atom_Types_Num
     - integer
     - Must be provided by user.
     - Number of atomic types. Should be equal to the number of ``Atom_Type``

            
   * - Ecut_Wavefunction
     - real
     - 40.0 (Ha)
     - Kinetic energy cutoff for the wavefunction in the Fourier space.

   * - Density_Grid_Factor
     - real
     - 2.0 
     - | :math:`N^{(\rho)}_{\alpha}=\text{Density_Grid_Factor}\times N^{(\psi)}_{\alpha}` 
       |   :math:`N^{(\rho)}_{\alpha} / N^{(\psi)}_{\alpha}`: number of grid points along the dimension :math:`\alpha` for the electron density / wavefunction.  :math:`\alpha=1,2,3`
       | The default choice of 2.0 aims at reducing the aliasing effect

   * - Hybrid_Mixing_Type
     - string
     - ``nested``
     - | Mixing methods for hybrid exchange-correlation functional due to the involvement of the density matrix.
       | ``nested``
       |   Two level nested SCF method as used by QuantumESPRESSO. The outer SCF loop performs fixed point iteration for the density matrix (defined using wavefunctions). The inner SCF loop performs regular density / potential mixing.
       | ``scdiis``
       |   DEPRECATED
       | ``pcdiis``
       |   Projected commutator DIIS method. Mixes a gauge transformed set of wavefunctions. 
       | 
       |   W. Hu, L. Lin and C. Yang, Projected Commutator DIIS Method for Accelerating Hybrid Functional Electronic Structure Calculations, J. Chem. Theory Comput. 13, 5458, 2017 

       .. note::
         PCDIIS method converges faster, but only works for insulating systems.


   * - Mixing_Variable
     - string
     - ``density``
     - | Mixing variable for SCF iteration
       | ``density``:
       |   Density mixing
       | ``potential``:
       |   Potential mixing


   * - Mixing_Type
     - string
     - ``anderson``
     - | ``anderson``
       |   Anderson mixing
       | ``kerker+anderson``
       |   Anderson mixing with Kerker preconditioner.
           Mainly for homogeneous metallic systems.

   * - Super_Cell
     - block
     - Must be provided by user.
     - Orthorhombic cell lengths in the unit of Bohr. Example ::
       
           begin Super_Cell
           10.0 10.0 10.0
           end Super_Cell    

   * - Temperature
     - real
     - 300.0 (K)
     - | Electronic temperature. 
       | 300 K :math:`\approx` 0.00095 Ha

   * - UPF_File
     - block
     - Must be provided by user.
     - Pseudopotential file in the UPF format. They can be downloaded from `this github link <https://github.com/pipidog/ONCVPSP>`_. Example::

            begin UPF_File
            Si_ONCV_PBE-1.1.upf
            end UPF_File


   * - XC_Type
     - string
     - ``XC_LDA_XC_TETER93``
     - Type of exchange correlation energy functionals.
       The labels come directly from the `libxc <https://www.tddft.org/programs/libxc/>`_ package. The currently supported options are:


       | ``XC_LDA_XC_TETER93``:
       |   Teter 93 parameterization. LDA.
       |   S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996)

       | ``XC_GGA_XC_PBE``:
       |   Perdew-Burke-Ernzerhof (PBE). GGA
       |   JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
       |   JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)

       | ``XC_HYB_GGA_XC_HSE06``:
       |   Heyd-Scuseria-Ernzerhof (HSE). Hybrid functional. This is the same as the ``hse`` functional in `QuantumESPRESSO <https://www.quantum-espresso.org/>`_
       |   J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003)
       |   J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006)
       |   A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006)

   * - Keyword
     - Type
     - Default value
     - Description




.. note::
    - This is actually a TODO list
    - Organize the input parameters into categories
    - Allow lower / mixed case input parameters?

.. note::
    - This is actually a TODO list
    - Change the default value of Hybrid_Mixing_Type to ``pcdiis``