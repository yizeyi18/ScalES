import numpy as np
from ase.build import nanotube
from ase.visualize import view
from ase.io import read, write
from ase.build import bulk

cnt = nanotube(6, 0, length=2, vacuum=5.0)

view(cnt)

write('cnt.xyz', cnt)
