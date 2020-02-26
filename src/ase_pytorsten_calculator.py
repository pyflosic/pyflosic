import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms 
from ase.units import Ha, Bohr 
from pyscf import scf, gto
import copy
from flosic_scf import FLOSIC



class BasicFLOSICC(Calculator):
    '''Interface to use ase.optimize methods in an easy way within the
    pyflosic framework.
    
    This is an easy-to-use class to make the usage of pyflosic identical to other
    ase optimizer classes.
    
    For details refer to https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    
    
    Kwargs
        mf : a FLOSIC class object
            to be used to optimize the FOD's on (mandatory)
        
        atoms: an ase.Atoms object
            contains both, the spin up and down FODs
               
            fod1 and fod2 input to the FLOSIC class *must* be references
            pointing to this atoms object:
            fods = ase.Atoms('X2He2', positions=[[0,0,0],[-1,0,0],[0,0,0],[-1,0,0]])
            fodup = fod[:2]
            foddn = fod[2:]
            mf = FLOSIC(...,fod1=fodup, fod2=foddn, ...)
    
    author:
        Torsten Hahn (torstenhahn@fastmail.fm)
    '''
    
    implemented_properties = ['energy', 'forces']
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                label=os.curdir, atoms=None, **kwargs):
        
        Calculator.__init__(self, restart, ignore_bad_restart_file, \
                        label, atoms, **kwargs)
        valid_args = ('mf')
        
        
        # set any additional keyword arguments
        for arg, val in self.parameters.items():
            if arg in valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'
                                   % (arg, valid_args))
        #print("ESICC: {0}".format(atoms))
        self.atoms = atoms
        
        # find out machine precision
        # (this is pretty usefull for reliable numerics)
        #self.meps = np.finfo(np.float64).eps
        self.meps = 1e-12
        self.is_init = False

    def print_atoms(self):
        print('print_atoms', self.atoms)
        print(self.atoms.positions)
        return

    def get_forces(self, atoms=None):
        if atoms is not None:
            lpos = atoms.positions
        else:
            lpos = self.atoms.positions

        nspin = self.mf.nspin
        fposu = self.mf.fod1.positions
        nup = self.mf.fod1.positions.shape[0]
        fposd = self.mf.fod2.positions
        ndn = self.mf.fod2.positions.shape[0]
        fpos = np.zeros((nup+ndn,3), dtype=np.float64)
        fpos[:nup,:] = self.mf.fod1.positions[:,:]
        if self.mf.nspin == 2: fpos[nup:,:] = self.mf.fod2.positions[:,:]

        #print(fpos)
        pdiff = np.linalg.norm(lpos - fpos)
        #print('get_potential_energy, pdiff {}', pdiff, self.meps)
        if (pdiff > self.meps):
            self.mf.update_fpos(lpos)
            # update sic potential etupdate_fpos()
            self.mf.kernel(dm0=self.mf.make_rdm1())
        if not self.is_init:
            self.mf.kernel()
            self.is_init = True

        _ff = Ha/Bohr*self.mf.get_fforces()
        ##self.results['forces'][:,:] = self.FLO.
        return _ff



    def get_potential_energy(self, atoms=None, force_consistent=False):
        if atoms is not None:
            lpos = atoms.positions
        else:
            lpos = self.atoms.positions

        nspin = self.mf.nspin
        fposu = self.mf.fod1.positions
        nup = self.mf.fod1.positions.shape[0]
        fposd = self.mf.fod2.positions
        ndn = self.mf.fod2.positions.shape[0]
        fpos = np.zeros((nup+ndn,3), dtype=np.float64)
        fpos[:nup,:] = self.mf.fod1.positions[:,:]
        if self.mf.nspin == 2: fpos[nup:,:] = self.mf.fod2.positions[:,:]

        #print('>> lpos, fpos:', fpos.sum(), lpos.sum())
        pdiff = np.linalg.norm(lpos - fpos)
        #print('get_potential_energy, pdiff {}', pdiff, self.meps)
        if (pdiff > self.meps):
            self.mf.update_fpos(lpos)
            # update sic potential etupdate_fpos()
            self.mf.kernel(dm0=self.mf.make_rdm1())
        if not self.is_init:
            self.mf.kernel()
            self.is_init = True
        return self.mf.get_esic()*Ha


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        for p in properties:
            if p == 'energy':
                self.results['energy'] = self.get_potential_energy()
            elif p == 'forces':
                _ff = self.mf.get_fforces()
                self.results['forces'] = -units.Ha/units.Bohr*_ff.copy()
            else:
                raise PropertyNotImplementedError(\
                    'calculation of {} is not implemented'.format(p))



if __name__ == "__main__":
    from ase.io import read
    from ase.optimize import  FIRE
    
    # path to the xyz file
    #f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/ase_pyflosic_optimizer/LiH.xyz'
    #atoms = read(f_xyz)
    #calc = PYFLOSIC(atoms=atoms,charge=0,spin=0,xc='LDA,PW',basis='cc-pvqz')
    #print('pyflosic total energy: ',calc.get_energy())
    #print('pyflosic total forces: ',calc.get_forces())

    print('testing basic optimizer')
    CH3SH = '''
    C -0.04795000 +1.14952000 +0.00000000
    S -0.04795000 -0.66486000 +0.00000000
    H +1.28308000 -0.82325000 +0.00000000
    H -1.09260000 +1.46143000 +0.00000000
    H +0.43225000 +1.55121000 +0.89226000
    H +0.43225000 +1.55121000 -0.89226000
    '''
    # these are the spin-up descriptors
    fod = Atoms('X13He13', positions=[
    (-0.04795000, -0.66486000, +0.00000000),
    (-0.04795000, +1.14952000, +0.00000000),
    (-1.01954312, +1.27662578, +0.00065565),
    (+1.01316012, -0.72796570, -0.00302478),
    (+0.41874165, +1.34380502, +0.80870475),
    (+0.42024357, +1.34411742, -0.81146545),
    (-0.46764078, -0.98842277, -0.72314717),
    (-0.46848962, -0.97040067, +0.72108036),
    (+0.01320210, +0.30892333, +0.00444147),
    (-0.28022018, -0.62888360, -0.03731204),
    (+0.05389371, -0.57381853, +0.19630494),
    (+0.09262866, -0.55485889, -0.15751914),
    (-0.05807583, -0.90413106, -0.00104673),
    ( -0.04795000, -0.66486000, +0.0000000),
    ( -0.04795000, +1.14952000, +0.0000000),
    ( +1.12523084, -0.68699049, +0.0301970),
    ( +0.40996981, +1.33508869, +0.8089839),
    ( +0.40987059, +1.34148952, -0.8106910),
    ( -0.49563876, -0.99517303, +0.6829207),
    ( -0.49640020, -0.89986161, -0.6743094),
    ( +0.00073876, +0.28757089, -0.0298617),
    ( -1.03186573, +1.29783767, -0.0035536),
    ( +0.04235081, -0.54885843, +0.1924678),
    ( +0.07365725, -0.59150454, -0.1951675),
    ( -0.28422422, -0.61466396, -0.0087913),
    ( -0.02352948, -1.0425011 ,+0.01253239)])

    fodup = Atoms([a for a in fod if a.symbol == 'X'])
    foddn = Atoms([a for a in fod if a.symbol == 'He'])
    print(fodup)
    print(foddn)


    b = 'sto6g'
    spin = 0
    charge = 0

    mol = gto.M(atom=CH3SH,
                basis=b,
                spin=spin,
                charge=charge)

    grid  = 5
    mol.verbose = 4
    mol.max_memory = 2000
    mol.build()
    xc = 'LDA,PW'


    # quick dft calculation
    mdft = scf.UKS(mol)
    mdft.xc = xc
    mdft.kernel()

    # build FLOSIC object
    mflosic = FLOSIC(mol, xc=xc,
        fod1=fodup,
        fod2=foddn,
        grid=grid,
        init_dm=mdft.make_rdm1()
    )
    mflosic.max_cycle = 40
    mflosic.conv_tol = 1e-5  # just for testing

    calc = BasicFLOSICC(atoms=fod, mf=mflosic)
    #print(fod.get_potential_energy())
    #print(fod.get_forces())

    print(" >>>>>>>>>>>> OPTIMIZER TEST <<<<<<<<<<<<<<<<<<<<")

    # test the calculator by optimizing 

    dyn = FIRE(atoms=fod,
        logfile='OPT_FRMORB_FIRE_TEST.log',
        #downhill_check=True,
        maxmove=0.05
    )
    dyn.run(fmax=0.005,steps=99)
