from ase.io import read
from flosic_os import calculate_flosic, flosic,xyz_to_nuclei_fod,ase2pyscf,get_multiplicity
from pyscf import dft,gto

# This example shows how FLO-SIC calculations can be done in the one-shot mode.
# The necessary input is a .xyz file with the molecular geometry and the FOD positions.

# The easiest way to do a FLO-SIC one-shot calculation is to call calculate_flosic.
# This is FULL FLO-SIC.
# Let's define some parameters for that.

b = '6-311++Gss' # Basis set.
verbose = 4 # Amount of output. 4: full output.
max_cycle = 300 # Number of SCF iterations.
conv_tol = 1e-7 # Accuracy of the SCF cycle.
grid = 3 # Level of the numerical grid. 3 is the standard value.
xc = 'LDA,PW' # Exchange-correlation functional in the form: (exchange,correlation)
# NOTE: As there exists only one way to express the exchange for LDA, there is only one identifier.
# For LDA correlation there exist several.

# We need the systems name (= Filename) as input.

sysname = 'H2'

# Now we can call calculate_flosic.
# calculate_flosic operates fully automatic; it performs a DFT SCF cycle and then does FLO-SIC on top of that.
# The return value is a Python dictionary.

flosic_values_1 = calculate_flosic(spin=0,fname=sysname,basis=b,verbose=verbose,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,xc=xc)

# ALTERNATIVELY: ASE Atoms object as input.
# We need an ASE Atoms object as input.
# We also need to specify the spin.

#molecule = read('H2.xyz')
#spin = 0
#flosic_values_1 = calculate_flosic(spin=0,ase_atoms=molecule,basis=b,verbose=verbose,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,xc=xc)

# Another possibility to use FLO-SIC is as an post-processing step.
# This is POST-PROCESSING one-shot.
# Here we start a regular DFT calculation and then apply FLO-SIC.
# First, set up a DFT calculation (see example 01).
# The mole object can be generated by Pyflosic routines as well.

# This routine properly parses the .xyz file.

molecule = read(sysname+'.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)

# Set spin and charge.

charge = 0
spin = get_multiplicity(sysname)

# Build the mole object.

mol = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin,charge=charge)	

# Set up the DFT calculation.

dft_object = dft.UKS(mol)
dft_object.verbose = verbose
dft_object.max_cycle = max_cycle
dft_object.conv_tol = conv_tol
dft_object.grids.level = grid
dft_object.xc = xc

# Perform the DFT calculation.

dft_energy = dft_object.kernel()

# Apply FLO-SIC to the DFT calculation.

flosic_values_2 = flosic(mol,dft_object,fod1,fod2)

# Output the results. The output for FLO-SIC is given in the form of Python dictionaries.

print("ESIC: {}".format(flosic_values_1['etot_sic']-dft_energy))

print('Total energy of H2 (DFT): %0.5f (should be %0.5f)' % (dft_energy,-1.13634167738585))
print('Total energy of H2 (FLO-SIC FULL): %0.5f (should be %0.5f) ' % (flosic_values_1['etot_sic'],-1.18032726019))
print('Total energy of H2 (FLO-SIC POST-PROCESSING): % 0.5f (should be %0.5f) ' % (flosic_values_2['etot_sic'],-1.18032726019))

