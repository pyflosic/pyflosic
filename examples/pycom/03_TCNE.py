from ase.io import read 
from pycom import automatic_guessing

# Note this xyz files only contain nuclei information. 
ase_nuclei = read('TCNE.xyz')
# Calculation parameters. 
charge = 0
spin = 0
basis = 'cc-pvdz'
xc = 'LDA,PW'
# Create the guess. 
pycom_guess(ase_nuclei,charge,spin,basis,xc,method='FB')

