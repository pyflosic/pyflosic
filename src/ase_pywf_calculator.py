#2019 PyWF developers
           
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
# author:	J. Kraus
# task:  	ase calculator for pywf

import os, sys
import numpy as np
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
    ReadError
from ase.atom import Atom
from ase import Atoms 
from ase.units import Ha, Bohr, Debye 
try:
    from ase.atoms import atomic_numbers 
except:
    # moved in 3.17 to
    from ase.data import atomic_numbers
import copy
from flosic_os import xyz_to_nuclei_fod,ase2pyscf,flosic 
from flosic_scf import FLOSIC
from ase.calculators.calculator import Calculator, all_changes, compare_atoms


def apply_field(mol,mf,E):
    # 
    # add efield to hamiltonian 
    # 
    # The gauge origin for dipole integral
    mol.set_common_orig([0., 0., 0.])
    # recalculate h1e with extra efield 
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph') + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    # update h1e with efield 
    mf.get_hcore = lambda *args: h

class PYWF(FileIOCalculator):
    
    implemented_properties = ['energy', 'forces','dipole','evalues','homo']
    PYWF_CMD = os.environ.get('ASE_PYWF_COMMAND')
    command =  PYWF_CMD

    # Note: If you need to add keywords, please also add them in valid_args 
    default_parameters = dict(
        mol= None,              # PySCF mole object 
        charge = None,          # charge of the system 
        spin = None,            # spin of the system, equal to 2S 
        basis = None,           # basis set
        ecp = None,             # only needed if ecp basis set is used 
        mode = 'hf',            # calculation method 
        efield=None,            # perturbative electric field
        max_cycle = 300,        # maximum number of SCF cycles 
        conv_tol = 1e-5,        # energy convergence threshold 
        ghost= False,           # ghost atoms at FOD positions 
        mf = None,              # PySCF calculation object
        use_newton=False,       # use the Newton SCF cycle 
        use_chk=False,          # restart from checkpoint file 
        verbose=0,              # output verbosity 
        debug=False,            # extra ouput for debugging purpose 
        num_iter=0,             # SCF iteration number 
        dm = None,              # density matrix
        mcc = None,             # PySCF post-HF calculation object
        calc_ip = True          # calculate ionization potentials
        ) 

    def __init__(self, restart=None, ignore_bad_restart_file=False,
        label=os.curdir, atoms=None, **kwargs):
        """ Constructor """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        valid_args = ('mol','charge','spin','basis','ecp','mode','efield','max_cycle','conv_tol','ghost','mf','use_newton','use_chk','verbose','debug','num_iter','dm','mcc')
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
        self.out = os.path.join(label, 'pywf.out')

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
        
    def get_energy(self,atoms=None):
        return self.get_potential_energy(atoms) 

    def calculation_required(self, atoms, properties):
        # checks of some properties need to be calculated or not 
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
        self.energy = self.results['energy'].copy()
        return self.energy

    def get_forces(self, atoms=None):
        if self.calculation_required(atoms,['forces']):
            self.calculate(atoms)
        self.forces = self.results['forces'].copy()
        return self.forces

	
    def get_dipole_moment(self,atoms=None):
        if self.calculation_required(atoms,['dipole']):
            self.calculate(atoms)
        self.dipole_moment = self.results['dipole'].copy()
        return self.dipole_moment

    def get_evalues(self,atoms=None):
        if self.calculation_required(atoms,['evalues']):
            self.calculate(atoms)
        self.evalues = self.results['evalues'].copy()
        return self.evalues

    def get_homo(self,atoms=None):
        if self.calculation_required(atoms,['homo']):
            self.calculate(atoms)
        self.homo = self.results['homo'].copy()
        return self.homo

    def calculate(self, atoms = None, properties = ['energy','dipole','evalues','forces','homo'], system_changes = all_changes):
        self.num_iter += 1 
        if atoms is None:
            atoms = self.get_atoms()
        else:
            self.set_atoms(atoms)
        if self.mode == 'hf' or self.mode == 'ccsd':
            from pyscf import gto, scf
            from pyscf.grad import uhf
            [geo,nuclei,fod1,fod2,included] =  xyz_to_nuclei_fod(atoms)
            mol = gto.M(atom=ase2pyscf(nuclei), basis=self.basis,spin=self.spin,charge=self.charge)
            mf = scf.UHF(mol)
            mf.verbose = self.verbose
            mf.max_cycle = self.max_cycle
            mf.conv_tol = self.conv_tol
            if self.use_chk == True and self.use_newton == False:
                mf.chkfile = 'pywf.chk'
            if self.use_chk == True and self.use_newton == False and os.path.isfile('pywf.chk'):
                mf.init_guess = 'chk'
                mf.update('pywf.chk')
                self.dm = mf.make_rdm1()
            if self.use_newton == True:
                mf = mf.as_scanner()
                mf = mf.newton()
            self.mf = mf
            if self.dm is None:
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            self.results['energy'] = e*Ha
            self.results['dipole'] = self.mf.dip_moment(verbose=0)*Debye 
            self.results['evalues'] = np.array(self.mf.mo_energy)*Ha
            if calc_ip == True:
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
            else:
                self.results['homo'] = None
            gf = uhf.Gradients(self.mf)
            gf.verbose = self.verbose
            forces = gf.kernel()*(Ha/Bohr)
            forces = -1.*forces
            self.results['forces'] = forces

        if self.mode == 'ccsd':
            from pyscf.cc.eom_rccsd import ipccsd
            from pyscf import cc
            from pyscf.grad import uccsd

            mcc = cc.UCCSD(self.mf)
            mcc.direct = True
            mcc.conv_tol = self.conv_tol
            mcc.max_cycle = self.max_cycle
            mcc.verbose = self.verbose
            self.mcc = mcc
            self.mcc.kernel()

            self.results['energy'] += (self.mcc.e_corr + self.mcc.ccsd_t() )*Ha
            
            if calc_ip == True:
                e_ip, c_ip = self.mcc.ipccsd()
                self.results['homo'] = -1.*e_ip*Ha
            else:
                self.results['homo'] = None
            gf = uccsd.Gradients(self.mcc)
            gf.verbose = self.verbose
            forces = gf.kernel()*(Ha/Bohr)
            forces = -1.*forces
            self.results['forces'] = forces


