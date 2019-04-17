from ase.io import read
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from pyscf import dft,gto
from flosic_scf import FLOSIC
import time
import numpy as np


# This example visualizes how a second order self-consistency solver can be used to drastically speed up and stabilize calculations.
# Second order here refers to the second derivatives of the energy w.r.t. the density.
# Drawing on these, the PySCF intrinsic functions .newton() and .scanner() allow the usage of a Newton optimization algorithm.
# Such an algorithm constructs the Hessian from these derivatives and uses it to perform anoptimization that is faster than the regular SCF solver. 
# Unfortunately, it can ONLY BE USED WITH LDA OR GGA.
# PySCF currently DOES NOT SUPPORT MGGAs with this second order solver.
# To show the speedup, the calculations below will be timed.

# 1) Standard example WITHOUT usage of fixed VSIC properties 

# To use this solver, we first have to simply define a regular calculator.
# The testing system is a Li atom.
molecule = read('Li.xyz')
geo,nuclei,fod1,fod2,included =  xyz_to_nuclei_fod(molecule)
spin = 1
charge = 0
b = 'sto3g'
xc = 'LDA,PW'
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)
mf = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2)

# Perform a calculation with the regular sovler.

t1_start = time.time()
e1 = mf.kernel()
t1_end = time.time()
delta_t1 = t1_end-t1_start


# 2) Standard example WITH usage of fixed VSIC properties 

# To use this solver, we first have to simply define a regular calculator.
# The testing system is a Li atom.
molecule = read('Li.xyz')
geo,nuclei,fod1,fod2,included =  xyz_to_nuclei_fod(molecule)
spin = 1
charge = 0
b = 'sto3g'
xc = 'LDA,PW'
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)
mf = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,vsic_every=8)


# Perform a calculation with the regular sovler.

t2_start = time.time()
e2 = mf.kernel()
t2_end = time.time()
delta_t2 = t2_end-t2_start


print('Etot with regular SCF solver (Hartree) = ',e1)
print('Duration for SCF cycle with regular SCF solver (in s):',delta_t1)
print('Etot with fixed VSIC (Hartree) = ',e2)
print('Duration for SCF cycle with fixed VSIC (in s):',delta_t2)
print('The magnitudes of these values depend on the power of your CPU and the number of processes that are running on your system simultaneously. Ideally, the second value should be significantly smaller.')



