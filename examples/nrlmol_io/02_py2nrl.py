from ase_nrlmol_calculator import NRLMOL
from flosic_os import xyz_to_nuclei_fod
from ase.io import read 

def pyflosic2nrlmol(ase_atoms):
    #
    # extract all  information from a nuclei+fod xyz file 
    #
    # ase_atoms ... contains both nuclei and fod information 
    ase_atoms = read(ase_atoms)
    [geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(ase_atoms)
    e_up = len(fod1)
    e_dn = -1*len(fod2)
    atoms = nuclei 
    fods  = fod1.extend(fod2)
    return [atoms,fods,e_up,e_dn] 

# Read Nuclei+FOD xyz file. 
ase_atoms = 'CH4.xyz'
[atoms,fods,e_up,e_dn] = pyflosic2nrlmol(ase_atoms)
# The NRLOMOL calculator. 
nrl   = NRLMOL(atoms=atoms,fods=fods,basis='DFO',e_up=e_up,e_dn=e_dn,extra=0,FRMORB='FRMORB')
# Write only NRLMOL input files. 
nrl.write_input(atoms=atoms)
# The calculator can be used the perform the nrlmol calculation as well. 
# See 02_py2nrl folder for more details  
# nrl.calculate(atoms)

