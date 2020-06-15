from ase.io import read 
from ase_pyflosic_calculator import * 

# Read structure (Nuclei + FODs) as xyz file. 
atoms = read('LiH.xyz')
# Create pyflosic_ase_calculator object. 
calc = PYFLOSIC(mode='flosic-os',atoms=atoms,charge=0,spin=0,xc='LDA,PW',basis='STO3G')
print(calc.get_potential_energy())
print(calc.get_forces())
print(calc.get_dipole_moment())
print(calc.get_evalues())
