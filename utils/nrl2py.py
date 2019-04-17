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
import readline
import os
import re
from glob import glob
import sys
import numpy as np
import collections

# Updates 
# 27.06.2018: SS update added the new2old_frmorb function. 
# 11.02.2019: SS raw_input is input in python3

def new2old_frmorb(f_frmorb,path='./'):
    # Convert new frmorb to old frmorb format. 
    f = open(f_frmorb,'r')
    ll = f.readlines()
    f.close()
    for l in range(len(ll)):
        ll[l] = ll[l].split()
    new = [x for x in ll if x != []]
    f = open(path+'FRMORB_NRL2PY','w')
    f.write(' '.join(ll[0][0:2])+'\n')
    for l in range(1,len(new)):
        f.write(' '.join(new[l][0:3])+'\n')
    f.close()

# This Python program takes an existing NRLMOL FLO-SIC calculation and sets up an Pyflosic calculation.
# It will create a new folder run_pyflosic.
# Therein the files run.py / run.sh as well as nrlmol_input.xyz will be created. 
# The former two will start the PySCF calculation (it is recommended always to run run.sh).
# nrlmol_input.xyz will contain the molecular geometry together with the FOD positions in Angstrom.

def nrl_to_py(path='./'):
    print('Welcome to NRL2PY.')

    # Find out if there is a path to be specified.

    while True:
        line = input('Does your PYTHONPATH include your PySCF installation? (y/n): ')
        if line == 'y':
            pyscf_path = None
            path_needed = False
            break
        if line == 'n':
            path_needed = True
            break
    if path_needed == True:
        line = input('Please specify your PySCF path: ')
        pyscf_path = line
    while True:
        line = input('Does your PYTHONPATH your Pyflosic installation? (y/n): ')
        if line == 'y':
            pyflosic_path = None
            path_needed = False
            break
        if line == 'n':
            path_needed = True
            break
    if path_needed == True:
        line = input('Please specify your Pyflosic path: ')
        pyflosic_path = line


    while True:
        line = input('Do you use old or new FRMORB/FRMIDT format? (old/new): ')
        if line == 'old':
            frmorb_format = 'old'
            break
        if line == 'new':
            frmorb_format = 'new'
            break

    # Get the NRLMOL input.

    try:
        symbol = open(path+'SYMBOL','r')
        sline = symbol.readlines()
        symbol.close()
    except:
        print('There is no SYMBOL file. Stopping NRL2PY.')
        sys.exit()
        
    try:
        if frmorb_format == 'old':
            frmorb = open(path+'FRMORB','r')
        if frmorb_format == 'new': 
            new2old_frmorb(f_frmorb=path+'FRMORB',path=path)
            frmorb = open(path+'FRMORB_NRL2PY','r')
        fline = frmorb.readlines()
        frmorb.close()
    except:
        print('There is no FRMORB file. Stopping NRL2PY.')       
        sys.exit()

    # Process Input so that a Pyflosic calculation can be set up.

    xc_tag = sline[1].split('-')[0]
    corr = sline[1].split('*')[1]
    if xc_tag == 'LDA':
        if corr != 'NONE':
            xc = 'LDA,PW'
        else:
            xc = 'LDA,'
    if xc_tag == 'GGA':
        if corr != 'NONE':
            xc = 'PBE,PBE'
        else:
            xc = 'PBE,'

    ncores = int(sline[6].split()[0])
    cores = np.zeros((ncores,3),dtype=np.float64)
    for i in range(0,ncores):
        cores[i,0] = float(sline[8+ncores+i].split()[2])
        cores[i,1] = float(sline[8+ncores+i].split()[3])
        cores[i,2] = float(sline[8+ncores+i].split()[4])

    bohr = 0.529177210564

    cores = cores*bohr
    nfod = [0]*2
    corsym = []

    for i in range(0,ncores):
        tmp = sline[8+ncores+i].split('0')[0]
        corsym.append(tmp.split('-')[1])


    nfod[0] = int(fline[0].split()[0])
    nfod[1] = int(fline[0].split()[1])
    spin  = nfod[0] - nfod[1]
    fods = np.zeros((2,np.max(nfod),3),dtype=np.float64)

    for s in range(0,2):
        for i in range(0,nfod[s]):
            fods[s,i,0] = float(fline[1+i+s*nfod[0]].split()[0])
            fods[s,i,1] = float(fline[1+i+s*nfod[0]].split()[1])
            fods[s,i,2] = float(fline[1+i+s*nfod[0]].split()[2])
    fods = fods*bohr

    if not os.path.exists(path+'run_pyflosic'):
        os.makedirs(path+'run_pyflosic')

    elements  = {   'X':0,  'HYD':1,  'HEL':2,  'LIT':3,  'BER':4,
                     'BOR':5,  'CAR':6,  'NIT':7,   
    'OXY':8,  'FLU':9,
                     'NEO':10, 'SOD':11, 'MAG':12,  
    'ALU':13, 'SIL':14,
                     'PHO':15, 'SUL':16, 'CHL':17,  
    'ARG':18, 'POT':19,
                     'CAL':20, 'SCA':21, 'TIT':22,  
    'VAN':23, 'CHR':24,
                     'MAN':25, 'IRO':26, 'COB':27,  
    'NIC':28, 'COP':29,
                     'ZIN':30, 'GAL':31, 'GER':32,  
    'ARS':33, 'SEL':34,
                     'BRO':35, 'KRY':36, 'RUB':37,  
    'STR':38, 'YTR':39,
                     'ZIR':40, 'NIO':41, 'MOL':42,  
    'TEC':43, 'RHU':44,
                     'RHO':45, 'PAL':46, 'SLV':47,  
    'CAD':48, 'IND':49,
                     'TIN':50, 'ANT':51, 'TEL':52,  
    'IOD':53, 'XEN':54,
                     'CES':55, 'BAR':56}
    csymbols = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

    nelec_symbols = 0
    nelec_actually = nfod[0]+nfod[1]
    for i in range(0,ncores):
        nelec_symbols += elements[corsym[i]]

    charge = nelec_symbols - nelec_actually
    syssym = []
    sysname = ''
    for i in range(0,ncores):
        syssym.append(csymbols[elements[corsym[i]]])

    counter=collections.Counter(syssym)

    for elem in counter:
        if counter[elem] > 1:
            sysname += elem+str(counter[elem])
        else:
            sysname += elem



    # Write the output.

    xyzfile = open(path+'run_pyflosic/nrlmol_input.xyz','w')
    xyzfile.write(str(ncores+nfod[0]+nfod[1])+'\n')
    xyzfile.write('Pyflosic input file created from NRLMOL calculation.\n')
    for i in range(0,ncores):
        xyzfile.write(csymbols[elements[corsym[i]]]+'\t'+str(format(cores[i,0], '.16f'))+'\t'+str(format(cores[i,1], '.16f'))+'\t'+str(format(cores[i,2], '.16f'))+'\n')
    for i in range(0,nfod[0]):
        xyzfile.write('X'+'\t'+str(format(fods[0,i,0], '.16f'))+'\t'+str(format(fods[0,i,1], '.16f'))+'\t'+str(format(fods[0,i,2], '.16f'))+'\n')
    for i in range(0,nfod[1]):
        xyzfile.write('He'+'\t'+str(format(fods[1,i,0], '.16f'))+'\t'+str(format(fods[1,i,1], '.16f'))+'\t'+str(format(fods[1,i,2], '.16f'))+'\n')
    xyzfile.close()

    run_1 = open(path+'run_pyflosic/run.py','w')
    run_1.write("from pyscf import gto\n")
    run_1.write("from flosic_os import xyz_to_nuclei_fod,ase2pyscf\n")
    run_1.write("from flosic_scf import FLOSIC\n")
    run_1.write("from nrlmol_basis import get_dfo_basis\n")
    run_1.write("from ase.io import read\n")
    run_1.write("\n")
    run_1.write("molecule = read('nrlmol_input.xyz')\n")
    run_1.write("geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)\n")
    run_1.write("b = get_dfo_basis('"+sysname+"')\n")
    run_1.write("mol = gto.M(atom=ase2pyscf(nuclei), basis=b,spin="+str(spin)+",charge="+str(charge)+")\n")
    run_1.write("sic_object = FLOSIC(mol,xc='"+xc+"',fod1=fod1,fod2=fod2,grid_level=3)\n")
    run_1.write("total_energy_sic = sic_object.kernel()\n")
    run_1.write("homo_flosic = sic_object.homo_flosic\n")
    run_1.write("print('Pyflosic total energy: ',total_energy_sic)\n")
    run_1.write("print('Pyflosic HOMO energy: ',homo_flosic)\n")
    run_1.close()

    run_2 = open(path+'run_pyflosic/run.sh','w')
    if pyscf_path != None:
        run_2.write("export PYTHONPATH="+pyscf_path+":$PYTHONPATH\n")
    if pyflosic_path != None:
        run_2.write("export PYTHONPATH="+pyflosic_path+":$PYTHONPATH\n")
    run_2.write("\n")
    run_2.write("python3 run.py\n")
    run_2.close()
    print('Your PySCF / Pyflosic calculation files can be found in '+path+'run_pyflosic.')
    return

if __name__ == '__main__':
    import os
    nrl_to_py(path=os.path.dirname(os.path.realpath(__file__)))
