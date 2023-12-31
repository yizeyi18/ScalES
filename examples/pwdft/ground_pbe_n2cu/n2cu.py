from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate
from ase.visualize import view
from ase.io import read, write

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(4, 4, 2), vacuum=10.0, orthogonal=True)

slab.calc = EMT()
e_slab = slab.get_potential_energy()

molecule = Atoms('2N', positions=[(0., 0., 0.), (0., 0., d)])
molecule.calc = EMT()
e_N2 = molecule.get_potential_energy()

add_adsorbate(slab, molecule, h, 'ontop')
print(slab.cell.array)
print(slab.positions)

constraint = FixAtoms(mask=[a.symbol != 'N' for a in slab])
slab.set_constraint(constraint)
#dyn = QuasiNewton(slab, trajectory='N2Cu.traj')
dyn = QuasiNewton(slab)
dyn.run(fmax=0.05)

print(slab.cell.array)
print(slab.positions)

print('Adsorption energy:', e_slab + e_N2 - slab.get_potential_energy())

view(slab)

write('n2cu.xyz', slab)

