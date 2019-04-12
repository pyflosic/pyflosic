from ase.io import read
from pyscf import gto
from flosic_os import xyz_to_nuclei_fod,ase2pyscf,get_multiplicity
from flosic_scf import FLOSIC

# This example shows how FLO-SIC calculations can be done self-consistently with Pyflosic.
# For this, we will create an instance of the SIC class provided by Pyflosic.
# First we need to get the input from the provided .xyz file.

sysname = 'H2'
molecule = read(sysname+'.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)

# We also need the spin and the charge.

spin = get_multiplicity(sysname)
charge = 0

# Furthermore we have to pick a basis set.

b = '6-311++Gss'

# With that we can build the mole object.

mol = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin,charge=charge)

# We further need to specify the numerical grid level used in the self-consistent FLO-SIC calculation.

grid_level = 4

# We need to choose an exchange-correlation functional.

xc = 'LDA,PW' # Exchange-correlation functional in the form: (exchange,correlation)
# NOTE: As there exists only one way to express the exchange for LDA, there is only one identifier.
# For LDA correlation there exist several.

# Now we can initiliaze the SIC object.

sic_object = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,grid_level=grid_level)	

# We can the modify the SIC object now if wished.

sic_object.max_cycle = 300 # Number of SCF iterations.
sic_object.conv_tol = 1e-7 # Accuracy of the SCF cycle.
sic_object.verbose = 4 # Amount of output. 4: full output.

# Now we can start the SIC calculation. The main output, the total energy, is the direct return value.

total_energy_sic = sic_object.kernel()

# We can get further output by accessing the attributes of the SIC object.

homo_flosic = sic_object.homo_flosic

# Output the results.

print('Total energy of H2 (FLO-SIC SCF): ',total_energy_sic)
print('HOMO energy eigenvalue of H2 (FLO-SIC SCF): ',homo_flosic)
print('These results should be: ',-1.18118689724,-0.623425516328)
