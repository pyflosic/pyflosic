# PyFLOSIC version check 
#
# author:   S. Schwalbe 
# date:     08.02.2019 

import unittest
from pyscf import gto, dft
from ase.io import read
from flosic_os import xyz_to_nuclei_fod,ase2pyscf,flosic 
from flosic_scf import FLOSIC
from nrlmol_basis import get_dfo_basis
from ase_pyflosic_optimizer import flosic_optimize
from ase.units import Ha

# Geometry
#f_xyz = '../examples/advanced_applications/CH4.xyz'
f_xyz = 'CH4.xyz'
sysname = 'CH4'
molecule = read(f_xyz)
geo,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(molecule)
spin = 0
charge  = 0
mol = gto.M(atom=ase2pyscf(nuclei), basis=get_dfo_basis(sysname),spin=spin,charge=charge)  

class KnownValues(unittest.TestCase):
    def test_dft(self):
        # DFT 
        mf = dft.UKS(mol)
        mf.verbose = 4     # Amount of output. 4: full output.
        mf.max_cycle = 300 # Number of SCF iterations.
        mf.conv_tol = 1e-6 # Accuracy of the SCF cycle.
        mf.grids.level = 7 # Level of the numerical grid. 3 is the standard value.
        mf.xc = 'LDA,PW'   # Exchange-correlation functional in the form: (exchange,correlation)
        e_ref = -40.1187154486949 
        self.assertAlmostEqual(mf.kernel(), e_ref, 5)

    def test_flosic_os(self):
        # DFT 
        mf = dft.UKS(mol)
        mf.verbose = 4     # Amount of output. 4: full output.
        mf.max_cycle = 300 # Number of SCF iterations.
        mf.conv_tol = 1e-6 # Accuracy of the SCF cycle.
        mf.grids.level = 7 # Level of the numerical grid. 3 is the standard value.
        mf.xc = 'LDA,PW'   # Exchange-correlation functional in the form: (exchange,correlation)
        mf.kernel() 
        # FLO-SIC OS
        results = flosic(mol,mf,fod1,fod2)
        e_calc = results['etot_sic']
        e_ref = -40.69057092300857
        self.assertAlmostEqual(e_calc, e_ref, 5)

    def test_flosic_scf(self):
        # FLO-SIC SCF 
        xc = 'LDA,PW'      # Exchange-correlation functional in the form: (exchange,correlation)
        grid_level = 3     # Level of the numerical grid. 3 is the standard value
        vsic_every = 1     # Calculate VSIC after 3 iterations  
        mf = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,grid_level=grid_level,vsic_every=vsic_every)
        mf.max_cycle = 300 # Number of SCF iterations.
        mf.conv_tol = 1e-6 # Accuracy of the SCF cycle.
        mf.verbose = 4     # Amount of output. 4: full output.
        e_ref =  -40.69756811131563
        self.assertAlmostEqual(mf.kernel(), e_ref, 5)

    def test_flosic_opt(self):
        # FLO-SIC SCF OPT 
        vsic_every=1
        flosic = flosic_optimize('flosic-scf',molecule,0,0,'LDA,PW',get_dfo_basis(sysname),None,opt='FIRE',maxstep=0.1,fmax=0.1,conv_tol=1e-6,grid=7,vsic_every=vsic_every,verbose=4)
        f = open('OPT_FRMORB.log','r')
        e_calc = f.readlines()[-1].split()[-2]
        f.close()
        if e_calc.find('*'):
                e_calc = e_calc.split('*')[0]
        e_calc = float(e_calc)/Ha 
        e_ref = -40.697563
        self.assertAlmostEqual(e_calc, e_ref, 4)

if __name__ == "__main__":
    print("PyFLOSIC: Full Tests")
    unittest.main()

