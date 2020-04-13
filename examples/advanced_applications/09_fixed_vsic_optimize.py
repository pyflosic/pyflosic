from ase import Atoms
from pyscf import gto, dft
from ase.io import read
from ase_pyflosic_optimizer import flosic_optimize
import time 
from ase.units import Ha 

# This example might took some time until it is finished. 
# You can use the command 'ase gui OPT_FRMORB.traj' to anlayze the trajectory. 
# Energy and maximum force are written in the OPT_FRMORB.xyz output file. 

# 1) Standard optimization 

# Load Nuclei+FOD starting geometry.
ase_atoms = read("LiH_rattle.xyz")
# Start timing. 
t1_start = time.time()
# The actual pyflosic-scf optimization. 
# Note: We choose here a smaller basis set and the 2nd order scf cycle to speed up the calculation. 
flosic = flosic_optimize('flosic-scf',ase_atoms,0,0,'LDA,PW','sto3g',opt='FIRE',maxstep=0.1,use_newton=True,use_chk=False,verbose=4,steps=1)
# End timing. 
t1_end = time.time()
run_time1 = t1_end - t1_start
f = open('OPT_FRMORB.log','r')
e1 = f.readlines()[-1].split()[-2]
f.close()
if e1.find('*'):
    e1 = e1.split('*')[0]
e1 = float(e1)/Ha

# 2) Optimization using fixed VSIC option 

# Load Nuclei+FOD starting geometry.
ase_atoms = read("LiH_rattle.xyz")
# Start timing. 
t2_start = time.time()
# The actual pyflosic-scf optimization. 
# Note: We choose here a smaller basis set and the 2nd order scf cycle to speed up the calculation. 
flosic = flosic_optimize('flosic-scf',ase_atoms,0,0,'LDA,PW','sto3g',opt='FIRE',maxstep=0.1,use_newton=True,use_chk=False,verbose=4,vsic_every=8,steps=1)
# End timing. 
t2_end = time.time()
run_time2 = t2_end - t2_start
f = open('OPT_FRMORB.log','r')
e2 = f.readlines()[-1].split()[-2]
f.close()
if e2.find('*'):
    e2 = e2.split('*')[0]
e2 = float(e2)/Ha


print('Standard run Etot [Hartree]',e1)
print('Standard run time [s]',run_time1)

print('Fixed VSIC run Etot [Hartree]',e2)
print('Fixed VSIC run time [s]',run_time2)
