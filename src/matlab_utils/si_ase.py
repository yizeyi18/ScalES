'''
Demonstrate the (complicated way) of generating a Si example using
ASE, and turn it into an orthorhombic cell.

Everything is in the unit of angstrom.

Lin Lin
9/20/2020
'''

import numpy as np
from ase.build import find_optimal_cell_shape, get_deviation_from_optimal_cell_shape  
from ase.build import make_supercell
from ase.visualize import view
from ase.io import read, write
from ase.build import bulk

print('Generate one primitive cell.')
unitcell = bulk('Si', 'diamond', a=5.43)

#print('Use find_optimal_cell_shape to turn it into an orthorhombic cell.')
#'''
#https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html#algorithm-for-finding-optimal-supercell-shapes
#

#P1 = find_optimal_cell_shape(unitcell.cell, 4, 'sc')
P1 = np.array([[-1,  1,  1],
       [ 1, -1,  1],
       [ 1,  1, -1]])


print('make a supercell')
# the supercell can also be constructed as
# unitcell.cell @ P1.T
# the cell vectors are given as the row vectors
supercell = make_supercell(unitcell, P1)

print('supercell.cell = ', supercell.cell)
print(supercell.arrays)

## generate a large supercell
#supercell_big = supercell * [2,2,2]
supercell_big = supercell * 5
view(supercell_big)

print('Dump out the supercell')
write('si.xyz', supercell_big)
#
#
#
#print('Convert to pyscf')
#import pyscf.pbc.gto as pbcgto
#import pyscf.pbc.dft as pbcdft
#import pyscf.pbc.scf as pbcscf
#from pyscf.pbc.tools import pyscf_ase
#
#cell = pbcgto.Cell()
#cell.verbose = 5
#cell.atom=pyscf_ase.ase_atoms_to_pyscf(supercell)
#cell.a=supercell.cell
#cell.basis = 'gth-dzvp'
#cell.pseudo = 'gth-pade'
#cell.build()
#
##mf=pbcdft.RKS(cell)
##
##mf.xc='lda,vwn'
#
## Note: unit = angstrom
#print('cell.unit = ', cell.unit)
##mf=pbcscf.RHF(cell)
##
##print(mf.kernel())
