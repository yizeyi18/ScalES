import numpy as np
from ase.visualize import view
from ase.io import read, write
from ase.build import bulk
from ase.build import graphene_nanoribbon


gnr = graphene_nanoribbon(3, 4, type='armchair', saturated=True,
                               vacuum=4.0)

view(gnr)

write('gnr.xyz', gnr)
