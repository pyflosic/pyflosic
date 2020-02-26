from ase.io import read
import numpy as np 
from ase_pyflosic_calculator import PYFLOSIC
    
sysname='formic'
atoms = read(sysname+'.xyz')
charge = 0
spin = 0
xc = 'LDA,PW'
basis = 'aug-cc-pVDZ'
grid = 3
max_cycle = 300
conv_tol = 1e-6
verbose = 4
ham_sic = 'HOOOV'

    
calc = PYFLOSIC(mode='dft',atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,grid=grid,max_cycle=max_cycle,conv_tol=conv_tol,ham_sic=ham_sic,verbose=verbose)
atoms.set_calculator(calc)
pol_dft = atoms.get_polarizability()


calc = PYFLOSIC(mode='flosic-os',atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,grid=grid,max_cycle=max_cycle,conv_tol=conv_tol,ham_sic=ham_sic,verbose=verbose)
atoms.set_calculator(calc)
pol_dft = atoms.get_polarizability()

calc = PYFLOSIC(mode='flosic-scf',atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,grid=grid,max_cycle=max_cycle,conv_tol=conv_tol,ham_sic=ham_sic,verbose=verbose)
atoms.set_calculator(calc)
pol_flosic = atoms.get_polarizability()
print('pol_dft - pol_flosic: ',pol_dft-pol_flosic,'\n')

