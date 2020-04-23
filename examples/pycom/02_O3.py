from ase.io import read 
from pycom import automatic_guessing

# Note this xyz files only contain nuclei information. 
ase_nuclei = read('O3.xyz')
# Calculation parameters. 
charge = 0
spin = 0
basis = 'cc-pvtz'
xc = 'LDA,PW'
# Create the guess. 
automatic_guessing(ase_nuclei,charge,spin,basis,xc,method='FB',grid=9)

