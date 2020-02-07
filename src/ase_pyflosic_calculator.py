#2019 PyFLOSIC developers
           
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
# pyflosic-ase-calculator 
#
# author:	            Sebastian Schwalbe, Jakob Kraus
# task:  	            ase calculator for pyflosic 
# CHANGELOG 07.02.2020:     get_atoms/set_atoms
#                           get_forces-> calculate 
#                           get_energy/get_potential_energy
#                           check_state
#                           no empty fodforces in DFT forces anymore
#                           new properties: homo energy, polarizability
#                           new units: dipole (e*A), evalues (eV)
#                           efield: keyword ->  tuple (freely chosen homog. efield)
#                           efield: function -> PYFLOSIC, corrected gauge origin
#                           new keywords: dm, cart, output
#                           removed keywords: 'atoms' (superfluous)
#                           removed mode: 'both'
#                           replaced calculation example at the bottom
# FUTURELOG 07.02.2020:     remove class BasicFLOSICC by Torsten Hahn?
#                           include finite_differences polarizability?
#                           include hyperpolarizability?
#                           reintroduce mode 'both' in updated form?

import os, sys
import numpy as np
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, ReadError, Calculator, all_changes, compare_atoms
from ase.atom import Atom
from ase import Atoms 
from ase.units import Ha, Bohr, Debye 
from ase.data import atomic_numbers
from pyscf import scf, gto
from pyscf.grad import uks
from pyscf.prop.polarizability.uhf import Polarizability, polarizability
import copy
from flosic_os import xyz_to_nuclei_fod, ase2pyscf, flosic 
from flosic_scf import FLOSIC

def force_max_lij(lambda_ij):
    # calculate the RMS of the l_ij matrix 
    nspin = 2
    lijrms = 0
    for s in range(nspin):
        M = lambda_ij[s,:,:]
        e = (M-M.T)[np.triu_indices((M-M.T).shape[0])]
        e_tmp = 0.0
        for f in range(len(e)):
            e_tmp = e_tmp + e[f]**2.
        e_tmp = np.sqrt(e_tmp/(M.shape[0]*(M.shape[0]-1)))
        lijrms =  lijrms + e_tmp
    lijrms = lijrms/2.
    return  lijrms



class PYFLOSIC(FileIOCalculator):
    """ PYFLOSIC calculator for atoms and molecules.
        by Sebastian Schwalbe and Jakob Kraus
        Notes: ase      -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
    	       pyscf	-> units [Ha,Bohr,Ha/Bohr,Debye,Bohr**3] 		                
    """
    implemented_properties = ['energy', 'forces','fodforces','dipole','evalues','homo','polarizability']
    PYFLOSIC_CMD = os.environ.get('ASE_PYFLOSIC_COMMAND')
    command =  PYFLOSIC_CMD

    # Note: If you need to add keywords, please also add them in valid_args 
    default_parameters = dict(
        fod1 = None,            # ase atoms object FODs spin channel 1 
        fod2 = None,            # ase atoms objects FODs spin channnel 2 
        mol= None,              # PySCF mole object 
        charge = None,          # charge of the system 
        spin = None,            # spin of the system, equal to 2S 
        basis = None,           # basis set
        ecp = None,             # only needed if ecp basis set is used 
        xc = 'LDA,PW',          # exchange-correlation potential - must be available in libxc 
        mode = 'flosic-os',     # calculation method 
        efield=None,            # perturbative electric field
        max_cycle = 300,        # maximum number of SCF cycles 
        conv_tol = 1e-5,        # energy convergence threshold 
        grid = 3,               # numerical mesh (lowest: 0, highest: 9)
        ghost= False,           # ghost atoms at FOD positions 
        mf = None,              # PySCF calculation object
        use_newton=False,       # use the Newton SCF cycle 
        use_chk=False,          # restart from checkpoint file 
        verbose=0,              # output verbosity 
        calc_forces=False,      # calculate FOD forces 
        debug=False,            # extra ouput for debugging purpose 
        l_ij=None,              # developer option: alternative optimization target  
        ods=None,               # developer option: orbital damping sic 
        fopt='force',           # developer option: in use with l_ij, alternative optimization target 
        fixed_vsic=None,        # fixed SIC one body values Veff, Exc, Ecoul
        num_iter=0,             # SCF iteration number 
        vsic_every=1,           # calculate vsic after this number on num_iter cycles 
        ham_sic ='HOO',         # unified SIC Hamiltonian - HOO or HOOOV 
        dm = None,              # density matrix
        cart = False,           # use Cartesian GTO basis and integrals (6d,10f,15g)
        output =None            # specify an output file, if None: standard output is used
        ) 

    def __init__(self, restart=None, ignore_bad_restart_file=False,
        label=os.curdir, atoms=None, **kwargs):
        """ Constructor """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        valid_args = ('fod1','fod2','mol','charge','spin','basis','ecp','xc','mode','efield','max_cycle','conv_tol','grid','ghost','mf','use_newton','use_chk','verbose','calc_forces','debug','l_ij','ods','fopt','fixed_vsic','num_iter','vsic_every','ham_sic','dm','cart','output')
        # set any additional keyword arguments
        for arg, val in self.parameters.items():
            if arg in valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'% (arg, valid_args))
        
        self.set_atoms(atoms)
        

    def initialize(self,atoms=None,properties=['energy'],system_changes=all_changes):
        self.atoms = atoms.copy()

    def set_atoms(self, atoms):
        if self.atoms != atoms:
            self.atoms = atoms.copy()
            self.results = {}       

    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def set_label(self, label):
        self.label = label
        self.directory = label
        self.prefix = ''
        self.out = os.path.join(label, 'pyflosic.out')

    def check_state(self, atoms,tol=1e-15):
        if atoms is  None:
            system_changes = []
        else:
            system_changes = compare_atoms(self.atoms, atoms)
        
        # Ignore boundary conditions until now 
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        
        return system_changes

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
               
    def write_input(self, atoms, properties=None, system_changes=None, **kwargs):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.initialize(atoms)
        
    
    def apply_electric_field(self,mol,mf,efield):
        # based on pyscf/pyscf/prop/polarizability/uks.py and pyscf/pyscf/scf/40_apply_electric_field.py     
        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        # define gauge origin for dipole integral
    
        with mol.with_common_orig(charge_center):
        
            if mol.cart == False:
        
                ao_dip = mol.intor_symmetric('cint1e_r_sph', comp=3)
        
            else:
            
                ao_dip = mol.intor_symmetric('cint1e_r_cart',comp=3)
            
        h1 = mf.get_hcore()
        mf.get_hcore = lambda *args, **kwargs: h1 + np.einsum('x,xij->ij',efield, ao_dip)
        return mf
    
    def get_energy(self,atoms=None):
        # wrapper for get_potential_energy()
        return self.get_potential_energy(atoms) 

    def calculation_required(self, atoms, properties):
        # check if a list of properties has to be calculated or not 
        assert not isinstance(properties, str)
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False

    def get_potential_energy(self, atoms=None, force_consistent = False):
        # calculate total energy if required 
        if self.calculation_required(atoms,['energy']):
            self.calculate(atoms)
        if self.fopt == 'lij':
            # res = force_max_lij(self.lambda_ij)
            res = self.results['energy'].copy()
        if self.fopt == 'force':
            res = self.results['energy'].copy()
        if self.fopt == 'esic-force':
            res = self.results['esic'].copy()
        return res 

    def get_forces(self, atoms=None):
        # calculate forces if required
        if self.calculation_required(atoms,['forces']):
            self.calculate(atoms)
        self.forces = self.results['forces'].copy()
        return self.forces

    def get_fodforces(self, atoms=None):
        # calculate FOD forces if required
        if self.calculation_required(atoms,['fodforces']):
            self.calculate(atoms)
        self.fodforces = self.results['fodforces'].copy() 
        return self.fodforces 
	
    def get_dipole_moment(self,atoms=None):
        # calculate dipole moment if required
        if self.calculation_required(atoms,['dipole']):
            self.calculate(atoms)
        self.dipole_moment = self.results['dipole'].copy()
        return self.dipole_moment

    def get_polarizability(self,atoms=None):  
        # calculate polarizability  if required
        # only implemented for DFT
        if self.calculation_required(atoms,['polarizability']):
            self.calculate(atoms)
        self.polarizability = self.results['polarizability'].copy()
        return self.polarizability

    def get_evalues(self,atoms=None):
        # calculate eigenvalues if required
        if self.calculation_required(atoms,['evalues']):
            self.calculate(atoms)
        self.evalues = self.results['evalues'].copy()
        return self.evalues

    def get_homo(self,atoms=None):
        # calculate HOMO energy if required
        if self.calculation_required(atoms,['homo']):
            self.calculate(atoms)
        self.homo = self.results['homo'].copy()
        return self.homo

    def calculate(self, atoms = None, properties = ['energy','dipole','evalues','fodforces','forces','homo','polarizability'], system_changes = all_changes):
        self.num_iter += 1 
        if atoms is None:
            atoms = self.get_atoms()
        else:
            self.set_atoms(atoms)
        if self.mode == 'dft':
            [geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
            mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,cart=self.cart,output=self.output)
            mf = scf.UKS(mol)
            mf.xc = self.xc 
            mf.verbose = self.verbose
            mf.max_cycle = self.max_cycle
            mf.conv_tol = self.conv_tol
            mf.grids.level = self.grid
            if self.use_chk == True and self.use_newton == False:
                mf.chkfile = 'pyflosic.chk'
            if self.use_chk == True and self.use_newton == False and os.path.isfile('pyflosic.chk'):
                mf.init_guess = 'chk'
                mf.update('pyflosic.chk')
                self.dm = mf.make_rdm1()
            if self.use_newton == True and self.xc != 'SCAN,SCAN':
                mf = mf.as_scanner()
                mf = mf.newton()
            if self.efield != None:
                mf = self.apply_electric_field(mol,mf,self.efield)
            self.mf = mf
            if self.dm is None:
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            self.results['energy'] = e*Ha
            # conversion to eV to match ase
            self.results['dipole'] = self.mf.dip_moment(verbose=self.verbose)*Debye 
            # conversion to e*A to match ase
            self.results['polarizability'] = self.mf.Polarizability().polarizability()*(Bohr**3) 
            # conversion to A**3 to match ase
            self.results['evalues'] = self.mf.mo_energy*Ha
            # conversion to eV to match ase
            n_up, n_dn = self.mf.mol.nelec
            if n_up != 0 and n_dn != 0:
                e_up = np.sort(self.results['evalues'][0])
                e_dn = np.sort(self.results['evalues'][1])
                homo_up = e_up[n_up-1]
                homo_dn = e_dn[n_dn-1]
                self.results['homo'] = max(homo_up,homo_dn)
            elif n_up != 0:
                e_up = np.sort(self.results['evalues'][0])
                self.results['homo'] = e_up[n_up-1]
            elif n_dn != 0:
                e_dn = np.sort(self.results['evalues'][1])
                self.results['homo'] = e_dn[n_dn-1]
            else:
                self.results['homo'] = None

            self.results['fodforces'] = None
            if self.xc != 'SCAN,SCAN': # no gradients for meta-GGAs!
                gf = uks.Gradients(self.mf)
                gf.verbose = self.verbose
                gf.grid_response = True
                forces = gf.kernel()*(Ha/Bohr)
                # conversion to eV/A to match ase
                forces = -1*forces
                totalforces = []
                totalforces.extend(forces)
                totalforces = np.array(totalforces)
                self.results['forces'] = totalforces
            else:
                self.results['forces'] = None
            
        if self.mode == 'flosic-os':
            [geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
            if self.ecp == None:
                if self.ghost == False:
                    mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,cart=self.cart,output=self.output)
                else:
                    mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,cart=self.cart,output=self.output)
                    mol.basis ={'default':self.basis,'GHOST1':gto.basis.load('sto3g', 'H'),'GHOST2':gto.basis.load('sto3g', 'H')}
            else:
                mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,ecp=self.ecp,cart=self.cart,output=self.output)
            mf = scf.UKS(mol)
            mf.xc = self.xc 
            mf.verbose = self.verbose
            if self.use_chk == True and self.use_newton == False:
                mf.chkfile = 'pyflosic.chk'
            if self.use_chk == True and self.use_newton == False and os.path.isfile('pyflosic.chk'):
                mf.init_guess = 'chk'
                mf.update('pyflosic.chk')
                self.dm = mf.make_rdm1()
            if self.use_newton == True and self.xc != 'SCAN,SCAN':
                mf = mf.as_scanner()
                mf = mf.newton()
            mf.max_cycle = self.max_cycle
            mf.conv_tol = self.conv_tol
            mf.grids.level = self.grid
            if self.efield != None:
                mf = self.apply_electric_field(mol,mf,self.efield)
            self.mf = mf
            if self.dm is None :
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            mf = flosic(mol,self.mf,fod1,fod2,sysname=None,datatype=np.float64, print_dm_one = False, print_dm_all = False,debug=self.debug,l_ij = self.l_ij, ods = self.ods, fixed_vsic = self.fixed_vsic, calc_forces=True,ham_sic = self.ham_sic)
            self.results['energy']= mf['etot_sic']*Ha
            # conversion to eV to match ase
            self.results['fodforces'] = -1*mf['fforces']*(Ha/Bohr) 
            # conversion to eV/A to match ase
            print('Analytic FOD force [Ha/Bohr]')
            print(-1*mf['fforces'])
            print('fmax = %0.6f [Ha/Bohr]' % np.sqrt((mf['fforces']**2).sum(axis=1).max()))
            self.results['dipole'] = mf['dipole']*Debye
            # conversion to e*A to match ase
            self.results['evalues'] = mf['evalues']*Ha
            # conversion to eV to match ase
            n_up, n_dn = self.mf.mol.nelec
            if n_up != 0 and n_dn != 0:
                e_up = np.sort(self.results['evalues'][0])
                e_dn = np.sort(self.results['evalues'][1])
                homo_up = e_up[n_up-1]
                homo_dn = e_dn[n_dn-1]
                self.results['homo'] = max(homo_up,homo_dn)
            elif n_up != 0:
                e_up = np.sort(self.results['evalues'][0])
                self.results['homo'] = e_up[n_up-1]
            elif n_dn != 0:
                e_dn = np.sort(self.results['evalues'][1])
                self.results['homo'] = e_dn[n_dn-1]
            else:
                self.results['homo'] = None

        if self.mode == 'flosic-scf':
            [geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
            if self.ecp is None:
                if self.ghost == False: 
                    mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,cart=self.cart,output=self.output)	
                else: 
                    mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,cart=self.cart,output=self.output)
                    mol.basis ={'default':self.basis,'GHOST1':gto.basis.load('sto3g', 'H'),'GHOST2':gto.basis.load('sto3g', 'H')}
            else:
                mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge,ecp=self.ecp,cart=self.cart,output=self.output)
            mf = FLOSIC(mol=mol,xc=self.xc,fod1=fod1,fod2=fod2,grid_level=self.grid,calc_forces=self.calc_forces,debug=self.debug,l_ij=self.l_ij,ods=self.ods,fixed_vsic=self.fixed_vsic,num_iter=self.num_iter,vsic_every=self.vsic_every,ham_sic=self.ham_sic)
            mf.verbose = self.verbose 
            if self.use_chk == True and self.use_newton == False:
                mf.chkfile = 'pyflosic.chk'
            if self.use_chk == True and self.use_newton == False and os.path.isfile('pyflosic.chk'):
                mf.init_guess = 'chk'
                mf.update('pyflosic.chk')
                self.dm = mf.make_rdm1()
            if self.use_newton == True and self.xc != 'SCAN,SCAN':
                mf = mf.as_scanner()
                mf = mf.newton() 
            mf.max_cycle = self.max_cycle
            mf.conv_tol = self.conv_tol
            if self.efield != None:
                mf = self.apply_electric_field(mol,mf,self.efield)
            self.mf = mf
            if self.dm is None:
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            self.results['esic'] = self.mf.esic*Ha
            # conversion to eV to match ase
            self.results['energy'] = e*Ha
            # conversion to eV to match ase
            self.results['fixed_vsic'] = self.mf.fixed_vsic  
            
            if self.fopt == 'force' or self.fopt == 'esic-force':
                # default: 'force'
                fforces = self.mf.get_fforces()
                self.results['fodforces'] = fforces*(Ha/Bohr)
                # conversion to eV/A to match ase
                print('Analytical FOD force [Ha/Bohr]')
                print(fforces)
                print('fmax = %0.6f [Ha/Bohr]' % np.sqrt((fforces**2).sum(axis=1).max()))

            if self.fopt == 'lij':
                #
                # This is under development. 
                # Trying to replace the FOD forces. 
                #
                self.lambda_ij = self.mf.lambda_ij
                self.results['lambda_ij'] = self.mf.lambda_ij
                #fforces = []
                #nspin = 2  
                #for s in range(nspin):
                #	# printing the lampda_ij matrix for both spin channels 
                #	print 'lambda_ij'
                #	print lambda_ij[s,:,:]
                #	print 'RMS lambda_ij'
                #	M = lambda_ij[s,:,:]
                #	fforces_tmp =  (M-M.T)[np.triu_indices((M-M.T).shape[0])]
                #	fforces.append(fforces_tmp.tolist()) 
                #print np.array(fforces).shape
                try:	
                    # 
                    # Try to calculate the FOD forces from the differences 
                    # of SIC eigenvalues 
                    #
                    evalues_old = self.results['evalues']/Ha
                    print(evalues_old)
                    evalues_new = self.mf.evalues
                    print(evalues_new)
                    delta_evalues_up = (evalues_old[0][0:len(fod1)] - evalues_new[0][0:len(fod1)]).tolist()
                    delta_evalues_dn = (evalues_old[1][0:len(fod2)] - evalues_new[1][0:len(fod2)]).tolist()
                    print(delta_evalues_up)
                    print(delta_evalues_dn)
                    lij_force = delta_evalues_up
                    lij_force.append(delta_evalues_dn)
                    lij_force = np.array(lij_force) 
                    lij_force = np.array(lij_force,(np.shape(lij_force)[0],3))
                    print('FOD force evalued from evalues')
                    print(lij_force)
                    self.results['fodforces'] = lij_force 
                except:
                    # 
                    # If we are in the first iteration 
                    # we can still use the analystical FOD forces 
                    # as starting values 
                    # 
                    fforces = self.mf.get_fforces()
                    print(fforces)  
                    #self.results['fodforces'] = -1*fforces*(Ha/Bohr)
                    self.results['fodforces'] = fforces*(Ha/Bohr)
                    print('Analytical FOD force [Ha/Bohr]')
                    print(fforces) 
                    print('fmax = %0.6f [Ha/Bohr]' % np.sqrt((fforces**2).sum(axis=1).max()))

            self.results['dipole'] =  self.mf.dip_moment()*Debye
            # conversion to e*A to match ase
            self.results['evalues'] = self.mf.evalues*Ha
            # conversion to eV to match ase
            n_up, n_dn = self.mf.mol.nelec
            if n_up != 0 and n_dn != 0:
                e_up = np.sort(self.results['evalues'][0])
                e_dn = np.sort(self.results['evalues'][1])
                homo_up = e_up[n_up-1]
                homo_dn = e_dn[n_dn-1]
                self.results['homo'] = max(homo_up,homo_dn)
            elif n_up != 0:
                e_up = np.sort(self.results['evalues'][0])
                self.results['homo'] = e_up[n_up-1]
            elif n_dn != 0:
                e_dn = np.sort(self.results['evalues'][1])
                self.results['homo'] = e_dn[n_dn-1]
            else:
                self.results['homo'] = None


        if self.mode == 'flosic-scf' or self.mode == 'flosic-os':
            totalforces = []
            forces = np.zeros_like(nuclei.get_positions()) 
            fodforces = self.results['fodforces'].copy()
            totalforces.extend(forces)
            totalforces.extend(fodforces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces
        	

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

from ase.vibrations import Raman


# define system
atoms = Atoms('N3', [(0, 0, 0), (1, 0, 0), (0, 0, 1)])
charge = 0
spin = 0
xc = 'LDA,PW'
basis = 'aug-cc-pVQZ' 
grid = 7
max_cycle = 300
conv_tol = 1e-8
cart = False
verbose = 4
# define calculator
calc = PYFLOSIC(mode='dft',atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,grid=grid,max_cycle=max_cycle,conv_tol=conv_tol,verbose=verbose,cart=cart)
atoms.set_calculator(calc)

ram = Raman(atoms,delta=0.005)
ram.run()
ram.summary()
ram.write_spectrum(out='raman.dat',quantity='raman',intensity_unit='A^4/amu')
