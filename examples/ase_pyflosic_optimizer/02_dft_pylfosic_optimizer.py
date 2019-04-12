from ase.io import read
from ase_pyflosic_calculator import PYFLOSIC
from nrlmol_basis import get_dfo_basis
from ase.optimize import LBFGS,FIRE 
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from ase_pyflosic_optimizer import flosic_optimize 

# Set up a structure.
atoms = read('LiH.xyz')
# We want to use the use the NRLMOL DFO basis set. 
basis = get_dfo_basis('LiH')
# Set up ase flosic_optimizer for nuclei optimization and performe optimization.
flosic = flosic_optimize('dft',atoms,0,0,'LDA,PW',basis,None,opt='FIRE',maxstep=0.1,label='OPT_NUCLEI')
