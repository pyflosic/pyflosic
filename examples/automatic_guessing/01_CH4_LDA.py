from ase.io import read 
from pycom import automatic_guessing

ase_nuclei = read('CH4.xyz')
charge = 0
spin = 0
basis = 'cc-pvtz'
xc = 'LDA,PW'

# We now have the newton flag also for the automatic guessing procedure 
automatic_guessing(ase_nuclei,charge,spin,basis,xc,method='FB',newton=True)

