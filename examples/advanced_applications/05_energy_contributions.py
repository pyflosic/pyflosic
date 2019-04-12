from ase.io import read
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from pyscf import dft,gto
from flosic_scf import FLOSIC
import numpy as np


# This example shows how the different exchange-correlation energy contributions can be visualized with FLO-SIC. The testing system is a Li atom.
# First we define the input from which the mole object will be build later.

atom = read('Li.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(atom)
spin = 1
charge = 0
b = 'cc-pvqz'

# As we will later do exchange-correlation and exchange only calculation it makes sense to define both functionals here.

xc = 'LDA,PW'
x = 'LDA,'

# Now we can build the mole object.

mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin=spin,charge=charge)

# The next part is the definition of the calculator objects.
# For both xc and x we create a separate calculator object.

dftx = dft.UKS(mol)
dftx.xc = x
dftxc = dft.UKS(mol)
dftxc.xc = xc
sicx = FLOSIC(mol,xc=x,fod1=fod1,fod2=fod2)
sicxc = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2)

# Now we need to do the ground state calculations.

etot_dftx = dftx.kernel()
etot_dftxc = dftxc.kernel()
etot_sicx = sicx.kernel()
etot_sicxc = sicxc.kernel()

# Then we can access the xc energies.

veffdftx = dftx.get_veff(mol=mol)
exc_dftx = veffdftx.__dict__['exc']
veffdftxc = dftxc.get_veff(mol=mol)
exc_dftxc = veffdftxc.__dict__['exc']
veffsicx = sicx.get_veff()
exc_sicx = veffsicx.__dict__['exc']
veffsicxc = sicxc.get_veff()
exc_sicxc = veffsicxc.__dict__['exc']

# With this data we can now analyze the energy contributions.

exsic = exc_sicxc - (exc_sicxc - exc_sicx) # SIC exchange energy.
ecsic = exc_sicxc - exc_sicx # SIC correlation energy.
e_corr_sicx = etot_sicx - etot_dftx # Correction of total energy by FLO-SIC (exchange only).
e_corr_sicxc = etot_sicxc - etot_dftxc # Correction of total energy by FLO-SIC.

# With this we can output the energy contributions.

print('SIC exchange energy:',exsic)
print('SIC correlation energy:',ecsic)
print('FLO-SIC correction to total energy (exchange only):',e_corr_sicx)
print('FLO-SIC correction to total energy (exchange-correlation):',e_corr_sicxc)
print('These results should be: ',-1.5234121806,-0.156736206467,-0.244190749868,-0.165851286945)
