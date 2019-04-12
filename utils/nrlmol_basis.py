from pyscf import gto, dft
from pyscf.gto.basis.parse_gaussian import load 
import numpy as np
try:
    from ase.atoms import string2symbols
except:
    # moved in 3.17 to
    from ase.symbols import string2symbols
import os

def get_dfo_basis(ase_sym,basis='nrlmol_dfo.gbs'):
    # ase_sym	...	string of atoms/chemical symbols (e.g. 'LiH', 'H2') 
    # output	...	dict/basis format of pyscf 
    # 
    # convert string to list object 
    # 
    ase_sym = string2symbols(ase_sym)
    dfo = {}
    p_home = os.path.dirname(os.path.realpath(__file__))
    for s in ase_sym: 
        b_tmp = load(p_home+'/basis/'+basis,s)
        dfo_tmp = {s:b_tmp}
        dfo.update(dfo_tmp) 
    return dfo 

if __name__ == '__main__':
    # test NRLMOL DFO parser 
    b = get_dfo_basis(ase_sym='LiH')
    mol = gto.M(atom='Li  0  0  0.41000; H  0.0  0  -1.23000', basis=b)
    # spin = N_up - N_dn  = 2*S not 2S + 1 
    mol.spin = 0
    mf = dft.RKS(mol)
    mf.xc = ['LDA,PW','PBE,PBE','SCAN,SCAN'][0]
    mf.max_cycle = 300
    e = mf.kernel()
    print('PYSCF (DFO) = %0.9f' % e)
    eref = -7.917976334
    print('NRLMOL (DFO) = %0.9f' % eref)
