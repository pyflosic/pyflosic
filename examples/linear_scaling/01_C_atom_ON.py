from ase.io import read
from pyscf import gto,dft
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from flosic_scf import FLOSIC
from onstuff import ON
from nrlmol_basis import get_dfo_basis

# Loading the structure 
ase_atoms = read('C.xyz')
pyscf_atoms,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(ase_atoms)

# Computational details 
b = get_dfo_basis('C') #'cc-pvqz'
spin = 2
charge = 0

mol = gto.M(atom=ase2pyscf(nuclei),
            basis=b,
            spin=spin,
            charge=charge)

grid  = 9
mol.verbose = 4
mol.max_memory = 2000
mol.build()
xc = 'LDA,PW'

# quick dft calculation
mdft = dft.UKS(mol)
mdft.xc = xc
mdft.kernel()

# build O(N) stuff
myon = ON(mol,[fod1.positions,fod2.positions], grid=grid)
myon.nshell = 1
myon.build()

# enable ONMSH
m = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,grid=grid, init_dm=mdft.make_rdm1(),ham_sic='HOOOV')
m.max_cycle = 40
m.set_on(myon)
m.conv_tol = 1e-5

# In-SCF-FOD optimization 
m.preopt = True
m.preopt_start_cycle=0
m.preopt_fix1s = True
m.preopt_fmin = 0.005

m.kernel()
# Some output 
print(m.fod_gradients())
print(m.get_fforces())

