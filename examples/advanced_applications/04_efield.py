from ase.io import read
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from pyscf import dft,gto
from flosic_scf import FLOSIC
import numpy as np


# This example visualizes how an electric field can be applied to DFT and FLO-SIC calculations. The application of the electric field is done to calculate the polarizability alpha.
# alpha will be calculated by approximating the derivative of the dipole moment mu with a finite difference like approach.
# The testing system will be an H4 structure (two H2 molecules.)
# First, we need the calculator objects and therefore mole objects.

molecule = read('H4.xyz')
geo,nuclei,fod1,fod2,included =  xyz_to_nuclei_fod(molecule)
spin = 0
charge = 0
b = 'cc-pVQZ'
xc = 'LDA,PW'
mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)


# Now we can initiliaze the calculator objects.

dft_object = dft.UKS(mol)
dft_object.xc = xc
sic_object = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,ham_sic='HOO')

# To calculate alpha we first calculate mu WITHOUT an electric field.

dft_object.kernel()
dft_mu0 = dft_object.dip_moment()[-1]
sic_object.kernel()
sic_mu0 = sic_object.dip_moment()[-1]

# Now we apply the electric field to the DFT and FLO-SIC calculator objects.
# For this we first set the gauge origin for the dipole integral.

mol.set_common_orig([0., 0., 0.])

# Then we calculate the new external potential.

vext = (mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + np.einsum('x,xij->ij', (0,0,0-0.001), mol.intor('cint1e_r_sph', comp=3)))

# Now we update this external potential for both calculators.

dft_object.get_hcore = lambda *args: vext
sic_object.get_hcore = lambda *args: vext

# Now we determine mu WITH electric field applied.

dft_object.kernel()
dft_mu1 = dft_object.dip_moment()[-1]
sic_object.kernel()
sic_mu1 = sic_object.dip_moment()[-1]

# With this we can calculate alpha.

alpha_dft = (dft_mu1 - dft_mu0)/(-0.001)
alpha_sic = (sic_mu1 - sic_mu0)/(-0.001)

# Now we can output the results.
# NOTE: We need to convert to atomic units.


print('Polarizability of H4 (DFT): %0.1f (should be %0.1f)'  %(alpha_dft*0.393456,37.2707068494))
print('Polarizability of H4 (FLO-SIC): %0.1f (should be %0.1f)' % (alpha_sic*0.393456,34.0331661325))
print('Polarizability of H4 (Reference Value): %0.1f' %(29.5))



