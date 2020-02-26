from ase.io import read
from pyscf import gto
from flosic_os import xyz_to_nuclei_fod,ase2pyscf,get_multiplicity
from flosic_scf import FLOSIC,sic_occ_

# This example shows how a self-consistent FLO-SIC calculations can be done with variable spin. Please note that so far this has only been fully tested for every configuration with H2. 
# First, we have to set up SIC objects.
# We will use a H2 configuration that is rather stretched.
# For this, the spin polarization should be 2.
# To show the functionality of the variable spin routine, thisn H2 molecule will be calculated with both spin polarizations of 0, 2 and variable.

sysname = 'H2_stretched'
molecule = read(sysname+'.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)

# We also need the spin and the charge.

spin_0 = 0
spin_2 = 2
charge = 0

# Furthermore we have to pick a basis set.
# We use the minimal basis set here in order to keep computational cost low.

b = 'sto3g'

# With that we can build the mole object.

mol_0 = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin_0,charge=charge)
mol_2 = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin_2,charge=charge)

# We further need to specify the numerical grid level used in the self-consistent FLO-SIC calculation.

grid = 4

# We need to choose an exchange-correlation functional.

xc = 'LDA,PW' # Exchange-correlation functional in the form: (exchange,correlation)
# NOTE: As there exists only one way to express the exchange for LDA, there is only one identifier.
# For LDA correlation there exist several.

# Now we can initiliaze the SIC objects.

sic_0 = FLOSIC(mol_0,xc=xc,fod1=fod1,fod2=fod2,grid=grid)

# For spin = 2 we have to adjust the FOD geometry by moving the FOD for the second spin channel into the first one.

fod1_spin2 = fod1.copy()
fod2_spin2 = fod2.copy()
fod1_spin2.append(fod2_spin2[0])
del fod2_spin2[0]
sic_2 = FLOSIC(mol_2,xc=xc,fod1=fod1_spin2,fod2=fod2_spin2,grid_level=grid_level)

# To enable a variable spin configuration, we simply have to load the routine sic_occ_

sic_variable = FLOSIC(mol_0,xc=xc,fod1=fod1,fod2=fod2,grid_level=grid_level)
sic_variable = sic_occ_(sic_variable)

# Now we can run the calculation.

etot_0 = sic_0.kernel()
etot_2 = sic_2.kernel()
etot_var = sic_variable.kernel()

# Output the results.
# If everything is correct, the variable spin result should be the same as with spin 2, as sic_occ_ should have correctly identified the need to change the spin configuration.

print('Total energy of H2 with spin 0 (FLO-SIC SCF): %0.5f (should be %0.5f)' %(etot_0,-0.723760417129))
print('Total energy of H2 with spin 2 (FLO-SIC SCF): %0.5f (should be %0.5f)' %(etot_2,-0.932966725136))
print('Total energy of H2 with variable spin (FLO-SIC SCF): %0.5f (should be %0.5f)' %(etot_var,-0.932966725136))

