from ase.io import read
from ase_pyflosic_calculator import PYFLOSIC
from nrlmol_basis import get_dfo_basis
from ase.optimize import LBFGS,FIRE 

# Set up a structure.
atoms = read('LiH.xyz')
# We want to use the NRLMOL DFO basis set. 
basis = get_dfo_basis('LiH')

# Set up ase calculator. 
calc = PYFLOSIC(atoms=atoms,charge=0,spin=0,xc='LDA,PW',basis=basis,mode='dft')
# Asign the calculator to the ase.atoms object. 
atoms.set_calculator(calc)

# Set up ase optimizer.
label = 'OPT_NUCLEI' 
maxstep = 0.1 
fmax = 0.0001 
steps = 1000 

# Perform the nuclei optimization. 
dyn = FIRE(atoms,logfile=label+'.log',trajectory=label+'.traj')
dyn.run(fmax=fmax, steps=steps)
