from pyscf import gto, dft

# This example shows how regular DFT calculations can be done in PySCF.
# First, let's define some parameters.

# This is the molecular geometry. Later we will use ASE Atoms objects.
# In either case the positions are given in Angstrom.

molecule = 'H 0 0 0.3707; H 0 0 -0.3707'

# The spin polarization for H2 is 0.

spin_pol = 0

# The charge for H2 is zero.

charge  = 0

# We need to choose a basis set. 
# Here, we chose a basis set with medium accuracy and speed.
# Other common choices would be:
# 'sto3g' - Minimal basis set.
# 'cc-pvqz' - High accuracy basis set.

b = '6-311++Gss' 

# With this information, we can build a mole object.
# Mole objects hold the molecular geometry and further information on the molecule.
# They are the input for DFT calculations.

mol = gto.M(atom=molecule, basis={'default':b},spin=spin_pol,charge=charge)

# With this mole object, we can initiliaze the DFT calculation.
# NOTE: This does not start the DFT calculation.

dft_object = dft.UKS(mol)

# Now we can customize the DFT calculation.

dft_object.verbose = 4 # Amount of output. 4: full output.
dft_object.max_cycle = 300 # Number of SCF iterations.
dft_object.conv_tol = 1e-7 # Accuracy of the SCF cycle.
dft_object.grids.level = 3 # Level of the numerical grid. 3 is the standard value.
dft_object.xc = 'LDA,PW' # Exchange-correlation functional in the form: (exchange,correlation)
# NOTE: As there exists only one way to express the exchange for LDA, there is only one identifier.
# For LDA correlation there exist several.

# Start the DFT calculation with kernel().
# Return value is the total energy.

total_energy = dft_object.kernel()

# Output. If both lines vary a lot, something is wrong.

print('Total energy of H2 (DFT): %0.5f (should be %0.5f)' % (total_energy,-1.13634167738585))

