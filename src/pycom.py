#   Copyright 2019 PyFLOSIC developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
# PyCOM - Automatic FOD guessing based on localized orbital density centroids 
#
# Author: 	S. Schwalbe (SS) 
# Target: 	Creates inital FOD guesses, which you will might be figure out soon, are very promising. 
#         	Please contact the author before publishing results using automatic_gussing. 
# Changes: 	27.11.2018	-	Kai Trepte (KT) Jakob Kraus (KK) 
#					as well as SS find that for some systems the FODs are shifted against the nuclei 
#					Routines are changes accordingly. The origin+vec shift is replaced by a shift from the origin of 
#					the cube file
#				- 	added option grid for automatic_guessing 
#		07.02.2018 	-	SS rename to pycom.py 

from pyscf import gto, scf, lo, tools, dft  
from pyscf.lo import boys, edmiston, pipek
from pyscf.tools.cubegen import * 
from ase.io import read,write 
from ase.atoms import Atom,Atoms
from pyscf.tools.wfn_format import write_mo
from ase.io import cube,read
import numpy as np
from ase.units import Bohr
import glob
from flosic_os import ase2pyscf

def calc_localized_orbitals(mf,mol,method='ER',jmol=False):
    # mf 		...	pyscf calculation object 
    # mol           ...     pyscf geometry object 
        # method 	...	localization method: ER, FB, PM 
        # jmol          ...     debug option to check furher 3d information files 
    
    # Localization. 
    
    # Spin 1. 
    # only occupied orbitals 
    method = method.upper()
    mo_occ = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
    if method == 'ER':
        loc1 = edmiston.Edmiston(mol, mo_occ)
    if method == 'FB':
        loc1 = boys.Boys(mol, mo_occ)
    if method == 'PM':
        loc1 = pipek.PipekMezey(mol, mo_occ)
    orb1 = loc1.kernel()
    
    # Spin 2 
    # only occupied orbitals 
    mo_occ = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
    if method == 'ER':
        loc2 = edmiston.Edmiston(mol, mo_occ) 
    if method == 'FB':
        loc2 = boys.Boys(mol, mo_occ)
    if method == 'PM':
        loc2 = pipek.PipekMezey(mol, mo_occ) 
    orb2 = loc2.kernel()
    
    # Write orbitals for jmol format. 
    if jmol == True: 
        # Spin 1.
        tools.molden.from_mo(mol, 'orb1.molden', orb1)
        with open('orb1.spt', 'w') as f:
            f.write('load orb1.molden; isoSurface MO 001;\n')
    	
    	# Spin 2.
        tools.molden.from_mo(mol, 'orb2.molden', orb2)
        with open('orb2.spt', 'w') as f:
            f.write('load orb2.molden; isoSurface MO 001;\n')
    	
    # Write orbitals in cube format.
    
    # Spin 1. 
    occ = len(mf.mo_coeff[0][mf.mo_occ[0] == 1])
    for i in range(occ):
        orbital(mol, str(method)+'_orb_'+str(i)+'spin1.cube', orb1[:,i], nx=80, ny=80, nz=80)


    # Spin 2.
    occ = len(mf.mo_coeff[1][mf.mo_occ[1] == 1])
    for i in range(occ):
        orbital(mol, str(method)+'_orb_'+str(i)+'spin2.cube', orb2[:,i], nx=80, ny=80, nz=80)
	
def get_com(f_cube,vec):
	# Calculation of COM 
	# COM	...	center of mass (COM), center of gravity (COG), centroid

	# Determine the origin of the cube file 
	#f = open(f_cube,'r') 
	#ll = f.readlines() 
	#f.close() 
	#vec_tmp = ll[2].split() 
	#vec_a = -1*float(vec_tmp[1])*Bohr
	#vec_b = -1*float(vec_tmp[2])*Bohr
	#vec_c = -1*float(vec_tmp[3])*Bohr
	#vec = [vec_a,vec_b,vec_c] 
	# cube structure in [Ang] 
    orb = cube.read(f_cube)
    # cuba data in [Bohr**3]
    data = cube.read_cube_data(f_cube)
	# cell of cube in [Ang] 
    cell= orb.get_cell()
    shape = np.array(data[0]).shape
    spacing_vec = cell/shape[0]/Bohr
    values = data[0]
    idx = 0
    unit = 1/Bohr #**3
    X = []
    Y = []
    Z = []
    V = []
    for i in range(0,shape[0]):
        for j in range(0,shape[0]):
            for k in range(0,shape[0]):
                idx+=1
                x,y,z = i*float(spacing_vec[0,0]),j*float(spacing_vec[1,1]),k*float(spacing_vec[2,2])
                # approximate fermi hole h = 2*abs(phi_i)**2 
                # Electron Pairing and Chemical Bonds:
                # see Bonding in Hypervalent Molecules from Analysis of Fermi Holes Eq(11) 
                x,y,z,v = x/unit ,y/unit ,z/unit , 2.*np.abs(values[i,j,k])**2. #2*np.abs(values[i,j,k])**2 # fermi hole approximation values[i,j,k] 
                X.append(x)
                Y.append(y)
                Z.append(z)
                V.append(v)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    V = np.array(V)
    x = sum(X*V)
    y = sum(Y*V)
    z = sum(Z*V)
    # Shifting to the origin of the cube file. 
    com = (np.array([x/sum(V),y/sum(V),z/sum(V)])-vec).tolist()
    return com 

def get_guess(atoms,spin1_cube,spin2_cube,method='ER'):
    # atoms 	...     ase atoms object 
    # spin1_cube	...	list of cubes in spin channel 1 
    # spin2_cube    ...     list of cubes in spin channel 2
    # method 	...	localization method: ER, FB, PM  
    method = method.upper()
    f1 = spin1_cube
    f2 = spin2_cube
    
    # Output file for the guess 
    f = open(str(method)+'_GUESS_COM.xyz','w')
    
    # Get nuclei information 
    # in ASE Version: 3.15.1b1
    # dcts are return 
    struct = atoms['atoms'] 
    sym = struct.get_chemical_symbols()
    pos = struct.get_positions()

    # Total number of nuclei+fods 
    f.write(str(len(sym)+len(f1)+len(f2))+'\n')
    f.write('\n')
    
    # The idea was to might shift the FODs.
    # But the FB centroids work quite well. 
    shift = 0.0 
    
    # Determine the origin of the cube file.
    # This really important. 
    # The xyz file has a different origin then the cube file.
    fv = open(f1[0],'r')
    ll = fv.readlines()
    fv.close()
    vec_tmp = ll[2].split()
    vec_a = -1*float(vec_tmp[1])*Bohr
    vec_b = -1*float(vec_tmp[2])*Bohr
    vec_c = -1*float(vec_tmp[3])*Bohr
    vec = [vec_a,vec_b,vec_c]
    
    # Nuclei positions 
    for s in range(len(sym)):
        f.write('%s %0.6f %0.6f %0.6f \n' % (sym[s],pos[s][0],pos[s][1],pos[s][2]))
    # FODs of alpha spin channel 
    for orb in f1:
        [x,y,z] = get_com(orb,vec)
        f.write('X %0.6f %0.6f %0.6f \n' % (x+shift,y+shift,z+shift))
    
    # FODs of beta spin channel
    for orb in f2:
        [x,y,z] = get_com(orb,vec)
        f.write('He %0.6f %0.6f %0.6f \n' % (x+shift,y+shift,z+shift))
    f.close()

def automatic_guessing(ase_nuclei,charge,spin,basis,xc,method='FB',ecp=None,newton=False,grid=3,BS=None,calc='UKS',symmetry=False,verbose=4):
    # ase_nuclei_atoms ...	ase.atoms.object containg only nuclei positions 
    # charge	   ...  charge of the system 
    # spin		   ...  spin state of the system 
    # basis 	   ...  basis set 
    # xc 		   ...  exchange-correlation functional 
    # method 	   ...  localization method (FB, ER, PM etc.) 
    #			Note: FB seems to give very reasonable guesses. 
    # ecp		   ...  effective core potential file 
    # newton	   ...  second order Newton Raphston scf solver works for LDA and GGA not for SCAN
    # grid 		   ...	grid level  
    # BS		   ...  broken symmetry
    # calc	 	   ,,,  UKS or UHF 
    # Performe a DFT calculation. 
    
    method = method.upper()
    calc = calc.upper()
    ase_atoms = ase_nuclei
    if ecp is None:
        mol = gto.M(atom=ase2pyscf(ase_atoms), basis=basis,spin=spin,charge=charge,symmetry=symmetry)
    if ecp is not None:
        mol = gto.M(atom=ase2pyscf(ase_atoms), basis=basis,ecp=ecp,spin=spin,charge=charge,symmetry=symmetry)
    mol.verbose = verbose 
    if calc == 'UKS':
        mf = scf.UKS(mol)
    if calc == 'UHF':
        mf = scf.UHF(mol)
    if calc == 'RHF': 
        mf = scf.RHF(mol)  
    mf.grids.level = grid
    mf.max_cycle = 3000
    mf.xc = xc
    # Broken symmetry 
    if BS != None:
        mf.kernel()
        idx_flip = mol.search_ao_label(BS)
        dma, dmb = mf.make_rdm1()
        dma_flip = dma[idx_flip.reshape(-1,1),idx_flip].copy()
        dmb_flip= dmb[idx_flip.reshape(-1,1),idx_flip].copy()
        dma[idx_flip.reshape(-1,1),idx_flip] = dmb_flip
        dmb[idx_flip.reshape(-1,1),idx_flip] = dma_flip
        dm = [dma, dmb]
        if ecp is None:
            mol = gto.M(atom=ase2pyscf(ase_atoms), basis=basis,spin=0,charge=charge,symmetry=symmetry)
        if ecp is not None:
            mol = gto.M(atom=ase2pyscf(ase_atoms), basis=basis,ecp=ecp,spin=0,charge=charge,symmetry=symmetry)
        mol.verbose = verbose
        mf = scf.UKS(mol)
        mf.grids.level = grid
        mf.max_cycle = 3000
        mf.xc = xc
    if newton == True: 
        mf = mf.as_scanner()
        mf = mf.newton()
    if BS == None:
       	mf.kernel()
    if BS != None:
        mf.run(dm)
    if calc == 'RHF':
        mf = scf.addons.convert_to_uhf(mf) 
    
    # Performe a localization calculation. 
    # Both spin channels are localized separately. 
    # The orbitals are written out as cube files. 
    calc_localized_orbitals(mf,mol,method=method)
    
    # Collect all cube files per spin channel. 
    f1 = glob.glob('*orb*spin1.cube')
    f2 = glob.glob('*orb*spin2.cube')
    
    # test for nuclei positions 
    # for ASE Version: 3.15.1b1
    # we neede the file handle and not the string 
    # new: 
    f_cube = open(f1[0])
    ase_atoms = cube.read_cube(f_cube)
    # previous: ase_atoms = cube.read_cube(f1[0])
    f_cube.close() 

    # Calculate the guess. 
    get_guess(atoms=ase_atoms,spin1_cube=f1,spin2_cube=f2,method=method)


pycom_guess = automatic_guessing


if __name__ == '__main__':
    from ase.io import read
    import os

    # Path to the xyz file 
    f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/pycom/O3.xyz'

    ase_nuclei = read(f_xyz)
    charge = 0
    spin = 0
    basis = 'dzp' #'cc-pvqz'
    xc = 'LDA,PW'
    automatic_guessing(ase_nuclei,charge,spin,basis,xc,method='FB',newton=True,symmetry=False)
