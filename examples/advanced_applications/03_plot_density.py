from flosic_os import get_multiplicity,ase2pyscf,xyz_to_nuclei_fod,dynamic_rdm,flosic
from flosic_scf import FLOSIC
from ase.io import read
from pyscf import gto,dft
import numpy as np
import matplotlib.pyplot as plt

# This example shows how the density can be visualized on the numerical grid.
# The routines provided by plot_density.py are very straightforward and only need a system name in order to perform this visualization.
# The default plotting axis is the z-axis; modify the routine in whatever way you wish.

# The only input we need are a mole object, the system name (only for output purposes) and the FOD geometry.
# The example system will be an H2 molecule with spin 2.
# We first have to set up a mole object.

sysname = 'H2'
molecule = read('H2_stretched_density.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)
spin = 2
b = 'cc-pvqz'	
mol = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin)

# Set the calculation parameters.

gridlevel = 4
convtol = 1e-6
maxcycle = 50
xc = 'LDA,PW'

# Do the DFT calculation.

print('Starting DFT calculation.')
mf = dft.UKS(mol)
mf.max_cycle = maxcycle
mf.conv_tol = convtol
mf.grids.level = gridlevel
mf.xc = xc	
mf.kernel()

# Get the DFT density.

ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=0)
dm_dft = mf.make_rdm1()
rho_dft = dft.numint.eval_rho(mol, ao, dm_dft[0], None, 'LDA', 0, None)

# Do the FLOSIC OS.

print('Starting FLOSIC calculation in OS mode.')
flosic_values = flosic(mol,mf,fod1,fod2)
flo = flosic_values['flo']

# Get the FLOSIC OS density.

dm_flo = dynamic_rdm(flo,mf.mo_occ)
rho_flo_os = dft.numint.eval_rho(mol, ao, dm_flo[0], None, 'LDA', 0, None)

# Get the mesh.

mesh = mf.grids.coords

# Do the FLOSIC SCF.

print('Starting FLOSIC calculation in SCF mode.')
mf2 = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2)
mf2.max_cycle = maxcycle
mf2.conv_tol = convtol
mf2.grids.level = 4
e = mf2.kernel()

# Get the FLOSIC density. 

flo = mf2.flo
ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=0)
dm_flo = dynamic_rdm(flo,mf2.mo_occ)
rho_flo_scf = dft.numint.eval_rho(mol, ao, dm_flo[0], None, 'LDA', 0, None)

# Plot the densities.	
# Init the arrays.

rdft = []
rsic = []
rsics = []
dist = []

# For loop that makes sure only the z axis is plotted.

i = 0
for m in mesh:
	if abs(m[0]) < 0.0001 and abs(m[1]) < 0.0001:
		rdft.append(rho_dft[i])
		rsic.append(rho_flo_os[i])
		dist.append(m[2])
		rsics.append(rho_flo_scf[i])
	i = i + 1

# Configure the plot. Change this according to your choice of asthetics. 

distsort = np.sort(dist[:],axis=0)
ind = np.argsort(dist[:], axis=0)
dft = np.array(rdft)
os = np.array(rsic)
scf = np.array(rsics)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
fs = 24
fzn = fs - 4
	
fig = plt.figure()
ax = plt.gca()
ax.semilogy(distsort,dft[ind], 'o-', c='red', markeredgecolor='none',label='DFT',markersize=8)
ax.semilogy(distsort,os[ind], 's:', c='blue', markeredgecolor='none',label='FLO-SIC (one-shot mode)')
ax.semilogy(distsort,scf[ind], 'v--', c='green', markeredgecolor='none',label='FLO-SIC (self-consistent mode)')
ax.tick_params(labelsize=fzn)
plt.rcParams.update({'font.size': fs})
plt.ylabel('log($n$)',fontsize=fs)
plt.xlabel('$\mathrm{z}\,[\mathrm{Bohr}]$',fontsize=fs)

# Plot everything.

#plt.title(str(sysname))
plt.legend(fontsize=fzn)
plt.show()


# The output will appear on your screen. An example .png to how this should look is included in this folder as H2.png.
