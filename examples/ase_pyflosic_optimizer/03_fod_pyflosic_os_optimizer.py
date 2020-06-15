from ase import Atoms
from pyscf import gto, dft
from ase.io import read
from ase_pyflosic_optimizer import flosic_optimize
import time 

# This example might took some time until it is finished. 
# You can use the command 'ase gui OPT_FRMORB.traj' to anlayze the trajectory. 
# Energy and maximum force are written in the OPT_FRMORB.xyz output file. 

# Load Nuclei+FOD starting geometry.
atoms = read("LiH_rattle.xyz")
# Start timing. 
t_start = time.time()
# The acutal pyflosic-os optimization.
# Note: We choose here a smaller basis set and the 2nd order scf cycle to speed up the calculation. 
flosic = flosic_optimize(mode='flosic-os',atoms=atoms,spin=0,charge=0,xc='LDA,PW',basis='cc-pvdz',opt='FIRE',maxstep=0.1,newton=True,verbose=4)
# End timing. 
t_end = time.time()
run_time = t_end - t_start
print('Run time [s]',run_time)
