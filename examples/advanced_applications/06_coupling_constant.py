import numpy as np
from pyscf import gto,dft
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from ase.io import read
from flosic_scf import FLOSIC


# This example shows how magnetic coupling constants can be calculated with FLO-SIC. 
# As we have to apply an electric field and calculate J twice during this example, it makes sense to do that with a separate function.
# In order to calculate the magnetic coupling constant we need to calculate the broken symmetry solution for E_(LS).
# This is done by applying an electric field. Once that has been done, it is easy to calculate the coupling constant.

def apply_field(mol,mf,E):
    # This function applies an electric field to the calculation.
	# It is based on example 07.
    mol.set_common_orig([0., 0., 0.])
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf.get_hcore = lambda *args: h

def calc_J1(LS,HS,S2_LS,S2_HS):
	# This function calculates the coupling constants and gives it back in the correct unit (cm^-1). 
	# See
    # Yamaguchi et al.
	# "Calculation of magnetic coupling constants with hybrid density functionals" 
	# @  https://jyx.jyu.fi/dspace/bitstream/handle/123456789/45989/URN%3ANBN%3Afi%3Ajyu-201505211947.pdf?sequence=1, Eq.(84) 
	# for further information.
    return -2*(HS - LS)/(S2_HS-S2_LS)*219474

# First, we read the structure for the high spin configuration.

atoms = read('HHeH.xyz')

# Now, we set up the mole object.

spin = 2 
charge = 0
xc = 'LDA,PW'
b = 'cc-pVQZ'
mol = gto.M(atom=ase2pyscf(atoms), basis=b,spin=spin,charge=charge)

# Now we set up the calculator.

dft_object = dft.UKS(mol)
dft_object.max_cycle= 300
dft_object.xc = xc
dft_object.conv_tol = 1e-7

# With this, we can calculate the high-spin total ground state energy.

E_HS_DFT = dft_object.kernel()

# Next, we need the low-spin solution. The first steps are completely similar to the high-spin calculation.
# In addition we now apply a small electric field.

spin = 0
charge = 0
xc = 'LDA,PW'
b = 'cc-pVQZ'
mol = gto.M(atom=ase2pyscf(atoms), basis=b,spin=spin,charge=charge)
dft_object = dft.UKS(mol)
dft_object.max_cycle = 300
dft_object.xc = xc
dft_object.conv_tol = 1e-7

# In addition we now apply a small electric field.

h= -0.001 
apply_field(mol,dft_object,E=(0,0,0+h))

# And then calculate the low-spin total ground state energy.

E_LS_DFT = dft_object.kernel()

# First, we read the structure for the high spin configuration.

atoms = read('HHeH_s2.xyz')

# Now, we set up the mole object.

spin = 2 
charge = 0
xc = 'LDA,PW'
b = 'cc-pVQZ'
[geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)
#mol.verbose = 4

# Now we set up the calculator.

sic_object = FLOSIC(mol=mol,xc=xc,fod1=fod1,fod2=fod2)
sic_object.conv_tol = 1e-7
sic_object.max_cycle = 300

# With this, we can calculate the high-spin total ground state energy.

E_HS_SIC = sic_object.kernel()

# Next, we need the low-spin solution. The first steps are completely similar to the high-spin calculation.

atoms = read('HHeH_s0.xyz')
spin = 0
charge = 0
xc = 'LDA,PW'
b = 'cc-pVQZ'
[geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)
#mol.verbose = 4
sic_object = FLOSIC(mol=mol,xc=xc,fod1=fod1,fod2=fod2)
sic_object.conv_tol = 1e-7
sic_object.max_cycle = 300

# In addition we now apply a small electric field.

h= -0.001 
apply_field(mol,sic_object,E=(0,0,0+h))

# And then calculate the low-spin total ground state energy.

E_LS_SIC = sic_object.kernel()

# To get J we further need to specify the expectation values of S**2.
# Here, we use the ideal S**2 expectation values.

S2_LS = 1.
S2_HS = 2.

# With this we can calculate the coupling constant.  

J_SIC = calc_J1(E_LS_SIC,E_HS_SIC,S2_LS,S2_HS)
J_DFT = calc_J1(E_LS_DFT,E_HS_DFT,S2_LS,S2_HS)
print('J of HHeH (DFT):',J_DFT)
print('J of HHeH (FLO-SIC):', J_SIC)
print('These results should be:',-164.300632216,-50.72702195911)

