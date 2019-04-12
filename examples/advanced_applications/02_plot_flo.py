from flosic_os import xyz_to_nuclei_fod,ase2pyscf,print_flo,get_multiplicity
from ase.io import read
from pyscf import gto
from flosic_scf import FLOSIC


# This example shows how the FLO can be visualized on the numerical grid.
# The .cube files written within this process can later be viewed in e.g. VESTA.
# In order to visualize the FLO, we first need a SIC class object.

# For this we first need a mole object.
# The test system will be CH4.

sysname = 'CH4'
molecule = read(sysname+'.xyz')
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)
spin = get_multiplicity(sysname)

# We will use a smaller basis set to keep computational cost low.
# The better the basis set, the nicer the FLO will look.

b = 'sto3g'
mol = gto.M(atom=ase2pyscf(nuclei), basis={'default':b},spin=spin)

# Now we can initiliaze the SIC object.

sic_object = FLOSIC(mol,fod1=fod1,fod2=fod2,xc='LDA,PW')

# And perform a ground state calculation.

sic_object.max_cycle = 500
sic_object.kernel()

# Now we can call the visualization function. The sysname here only specifies the output name.
# Maxcell coordinates the size of the cell that can be seen in VESTA.

print_flo(sic_object, sic_object.flo, sysname, nuclei, fod1, fod2, maxcell=6.)

# The .cube files will appear in this folder. An example .png of how they should look is provided as CH4.png
