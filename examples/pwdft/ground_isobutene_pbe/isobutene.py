import numpy as np
from ase.visualize import view
from ase.io import read, write
from ase.build import molecule
from ase.collections import g2

# g2.names to see which other molecules can be generated from the g2
# data set

mol= molecule('isobutene', vacuum=5.0)

view(mol)

write('isobutene.xyz', mol)
