from pyscf import gto, dft
from pyscf.grad import uks
from ase.io import read 
from nrlmol_basis import get_dfo_basis
#from berny import optimize
from pyscf.geomopt import *
from flosic_os import xyz_to_nuclei_fod,ase2pyscf

# This example shows how nuclei positions can be optimized with the pyberny solver. 
# Installation of pyberny 
# pip3 install pyberny 

# Read the structure. 
# This starting structure includes both nuclei and FOD positions. 
ase_atoms = read('LiH.xyz') 
# We want only to optimize the nuclei positions. 
[geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(ase_atoms)
ase_atoms  = nuclei 
# Calulation parameters. 
charge = 0 
spin = 0 
basis = get_dfo_basis('LiH')
xc = 'LDA,PW'
# Set up the pyscf structure object. 
mol = gto.M(atom=ase2pyscf(ase_atoms),spin=spin,charge=charge,basis=basis)
mol.verbose = 4 
# DFT pyscf calculation.
mf = dft.UKS(mol)
mf.max_cycle = 300
mf.conv_tol = 1e-6
mf.xc = xc
mf = mf.newton()
mf = mf.as_scanner()
# SCF single point for starting geometry. 
e = mf.kernel()
# Starting Gradients  
gf = uks.Gradients(mf)
gf.grid_response = True
gf.kernel()
# Optimization
optimize(mf)
