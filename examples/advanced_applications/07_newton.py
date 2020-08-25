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

# To use this solver, we first have to simply define a regular calculator.
# The testing system is a Li atom.
molecule = read('Li.xyz')
geo,nuclei,fod1,fod2,included =  xyz_to_nuclei_fod(molecule)
spin = 1
charge = 0
b = 'sto3g'
xc = 'LDA,PW'
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)
sic_object = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,ham_sic='HOO')
sic_object.max_cycle = 400


# Perform a calculation with the regular sovler.

start_regular = time.time()
sic_object.kernel()
end_regular = time.time()
duration_regular = end_regular-start_regular

# Now enable the second order solver.

sic_object = sic_object.as_scanner()
sic_object = sic_object.newton()

# And perform a calculation with the second order solver.

start_scanner = time.time()
sic_object.kernel()
end_scanner = time.time()
duration_scanner = end_scanner - start_scanner



print('Duration for SCF cycle with regular SCF solver (in s):',duration_regular)
print('Duration for SCF cycle with second order SCF solver (in s):',duration_scanner)
print('The magnitudes of these values depend on the power of your CPU and the number of processes that are running on your system simultaneously. Ideally, the second value should be significantly smaller.')



