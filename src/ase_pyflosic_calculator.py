# 2019 PyFLOSIC developers
#
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
#                           new keywords: dm, cart, output, solvation (only COSMO at the moment)
#                           removed keywords: 'atoms' (superfluous)
#                           removed mode: 'both'
#                           replaced calculation example at the bottom
# CHANGELOG 18.02.2020:     moved class BasicFLOSICC by Torsten Hahn to ase_pytorsten_calculator.py
# CHANGELOG 25.02.2020:     implemented polarizability for modes 'flosic-os' (just the DFT polarizability) and 'flosic-scf'
# CHANGELOG 03.03.2020:     removed valid_args, removed keyword 'mf', removed get_energy()
# CHANGELOG 01.04.2020:     use_newton -> newton, newton(default) = False, added default values for charge (0), spin (0), basis (STO-3G)
#                           removed argument mol, added argument df (default: False), changed default value for conv_tol to 1e-6, changed default value for verbose to 4
#
# FUTURELOG 01.04.2020:     include hyperpolarizability?
#                           reintroduce mode 'both' in updated form?
#                           include pbc for DFT?
#                           include PCM for DFT
#                           manage output via loggers
import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase import Atoms
from ase.units import Ha, Bohr, Debye
from pyscf import scf, gto, dft
from pyscf.prop.polarizability.uhf import polarizability, Polarizability
from flosic_os import xyz_to_nuclei_fod, ase2pyscf, flosic
from flosic_scf import FLOSIC
from pyscf.solvent.ddcosmo import DDCOSMO, ddcosmo_for_scf
from pyscf.data import radii

prune_dict = {
    'nwchem': dft.gen_grid.nwchem_prune,
    'sg1': dft.gen_grid.sg1_prune,
    'treutler': dft.gen_grid.treutler_prune,
    'no': None}

rad_dict = {
    'bragg': radii.BRAGG,
    'covalent': radii.COVALENT,
    'vdw': radii.VDW,
    'uff': radii.UFF,
    'mm3': radii.MM3}


class PYFLOSIC(Calculator):
    """ PYFLOSIC calculator for atoms and molecules.
        by Sebastian Schwalbe and Jakob Kraus
        Notes: ase      -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
               pyscf	-> units [Ha,Bohr,Ha/Bohr,Debye,Bohr**3]
    """

    implemented_properties = [
        'energy',
        'forces',
        'fodforces',
        'dipole',
        'evalues',
        'homo',
        'polarizability']

    default_parameters = dict(
        charge=0,                 # charge of the system
        spin=0,                   # spin of the system, equal to 2S
        basis='STO-3G',           # basis set
        ecp=None,                 # only needed if ecp basis set is used
        xc='LDA,PW',              # exchange-correlation potential - must be available in libxc
        # calculation method (dft,flosic-os or flosic-scf)
        mode='flosic-scf',
        efield=None,                # perturbative electric field
        max_cycle=300,            # maximum number of SCF cycles
        conv_tol=1e-6,            # energy convergence threshold
        grid=3,                   # numerical mesh (lowest: 0, highest: 9)
        newton=False,                # use the Newton second-order SCF cycle
        df=False,                    # apply density fitting
        use_chk=False,              # restart from checkpoint file
        verbose=4,                  # output verbosity
        ham_sic='HOOOV',           # choose a unified SIC Hamiltonian - HOO or HOOOV
        dm=None,                  # density matrix
        # use Cartesian GTO basis and integrals (6d,10f,15g)
        cart=False,
        output=None,              # specify an output file, if None: standard output is used
        # specify if solvation model should be applied (COSMO)
        solvation=None,
        lmax=10,                  # maximum l for basis expansion in spherical harmonics for solvation
        eta=0.1,                  # smearing parameter in solvation model
        lebedev_order=89,         # order of integration for solvation model
        rad_sol='vdw',         # radii for solvation model
        rad_sol_scal=1.0,           # scaling factor for the radii
        eps=78.3553,              # dielectric constant of solvent
        pol=False,              # calculate polarizability
        n_rad=None,               # radial grid
        n_ang=None,               # angular grid
        prune='no'              # grid pruning
    )

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PyFLOSIC', atoms=None, directory='.', **kwargs):
        """ Constructor """
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory, **kwargs)
        self.set_atoms(atoms)
        self.initialize()

    def initialize(self):
        for arg, val in self.parameters.items():
            if arg in self.default_parameters:
                setattr(self, arg, val)
            else:
                raise RuntimeError(
                    'unknown keyword arg "{}" : not in {}'.format(
                        arg, self.default_parameters))

    def set_atoms(self, atoms):
        if self.atoms != atoms:
            self.atoms = atoms.copy()
            self.results = {}

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def apply_electric_field(self, mf, efield):
        # based on pyscf/pyscf/prop/polarizability/uks.py and
        # pyscf/examples/scf/40_apply_electric_field.py
        mol = mf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        # define gauge origin for dipole integral

        with mol.with_common_orig(charge_center):

            if not mol.cart:

                ao_dip = mol.intor_symmetric('cint1e_r_sph', comp=3)

            else:

                ao_dip = mol.intor_symmetric('cint1e_r_cart', comp=3)

        h1 = mf.get_hcore()
        mf.get_hcore = lambda *args, **kwargs: h1 + \
            np.einsum('x,xij->ij', efield, ao_dip)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        # calculate total energy if required
        if force_consistent:
            name = self.__class__.__name__
            raise PropertyNotImplementedError(
                'Force consistent/free energy ("free_energy") '
                'not provided by {0} calculator'.format(name))
        else:
            if self.calculation_required(atoms, ['energy']):
                self.calculate(atoms)
            return self.results['energy']

    def get_forces(self, atoms=None):
        # calculate forces if required
        if self.calculation_required(atoms, ['forces']):
            self.calculate(atoms)
        return self.results['forces']

    def get_fodforces(self, atoms=None):
        # calculate FOD forces if required
        if self.calculation_required(atoms, ['fodforces']):
            self.calculate(atoms)
        return self.results['fodforces']

    def get_dipole_moment(self, atoms=None):
        # calculate dipole moment if required
        if self.calculation_required(atoms, ['dipole']):
            self.calculate(atoms)
        return self.results['dipole']

    def get_polarizability(self, atoms=None):
        # calculate polarizability  if required
        if self.calculation_required(atoms, ['polarizability']):
            self.calculate(atoms)
        return self.results['polarizability']

    def get_evalues(self, atoms=None):
        # calculate eigenvalues if required
        if self.calculation_required(atoms, ['evalues']):
            self.calculate(atoms)
        return self.results['evalues']

    def get_homo(self, atoms=None):
        # calculate HOMO energy if required
        if self.calculation_required(atoms, ['homo']):
            self.calculate(atoms)
        return self.results['homo']

    def calculate(
            self,
            atoms=None,
            properties=[
                'energy',
                'dipole',
                'evalues',
                'fodforces',
                'forces',
                'homo',
                'polarizability'],
            system_changes=all_changes):
        if atoms is None:
            atoms = self.get_atoms()
        else:
            self.set_atoms(atoms)
        [geo, nuclei, fod1, fod2, included] = xyz_to_nuclei_fod(atoms)
        mol = gto.M(
            atom=ase2pyscf(nuclei),
            basis=self.basis,
            ecp=self.ecp,
            spin=self.spin,
            charge=self.charge,
            cart=self.cart,
            output=self.output)
        if self.mode == 'dft':
            if self.solvation is None:
                self.mf = scf.UKS(mol)
            elif self.solvation == 'cosmo':
                cm = DDCOSMO(mol)
                cm.verbose = self.verbose
                cm.lmax = self.lmax
                cm.eta = self.eta
                cm.lebedev_order = self.lebedev_order
                cm.radii_table = rad_dict[self.rad_sol] * self.rad_sol_scal
                cm.max_cycle = self.max_cycle
                cm.conv_tol = self.conv_tol
                cm.eps = self.eps
                self.mf = ddcosmo_for_scf(scf.UKS(mol), cm)
            self.mf.xc = self.xc
            self.mf.verbose = self.verbose
            self.mf.max_cycle = self.max_cycle
            self.mf.conv_tol = self.conv_tol
            self.mf.grids.level = self.grid

            if self.n_rad is not None and self.n_ang is not None:
                self.mf.grids.atom_grid = (self.n_rad,self.n_ang)

            self.mf.grids.prune = prune_dict[self.prune]

            if self.use_chk and not self.newton:
                self.mf.chkfile = 'pyflosic.chk'
            if self.use_chk and not self.newton and os.path.isfile(
                    'pyflosic.chk'):
                self.mf.init_guess = 'chk'
                self.mf.update('pyflosic.chk')
                self.dm = self.mf.make_rdm1()
            if self.newton and self.xc != 'SCAN,SCAN':
                self.mf = self.mf.newton()
            if self.efield is not None:
                self.apply_electric_field(self.mf, self.efield)
            if self.df:
                self.mf = self.mf.density_fit()
            if self.dm is None:
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            self.results['energy'] = e * Ha
            # conversion to eV to match ase
            self.results['fodforces'] = None
            try:  # no gradients for meta-GGAs!
                gf = self.mf.nuc_grad_method()
                gf.verbose = self.verbose
                gf.grid_response = True
                forces = -1. * gf.kernel() * (Ha / Bohr)
                # conversion to eV/A to match ase
                totalforces = []
                totalforces.extend(forces)
                fod1forces = np.zeros_like(fod1.get_positions())
                fod2forces = np.zeros_like(fod2.get_positions())
                totalforces.extend(fod1forces)
                totalforces.extend(fod2forces)
                totalforces = np.array(totalforces)
                self.results['forces'] = totalforces
            except BaseException:
                self.results['forces'] = None
            self.results['evalues'] = self.mf.mo_energy * Ha
            # conversion to eV to match ase
            self.results['dipole'] = self.mf.dip_moment(
                verbose=self.verbose) * Debye
            # conversion to e*A to match ase
            if self.pol:
                self.results['polarizability'] = Polarizability(
                    self.mf).polarizability() * (Bohr**3)
                # conversion to A**3 to match ase
            else:
                self.results['polarizability'] = None

        if self.mode == 'flosic-os':
            self.mf = scf.UKS(mol)
            self.mf.xc = self.xc
            self.mf.verbose = self.verbose
            self.mf.max_cycle = self.max_cycle
            self.mf.conv_tol = self.conv_tol
            self.mf.grids.level = self.grid
            if self.n_rad is not None and self.n_ang is not None:
                self.mf.grids.atom_grid = (self.n_rad,self.n_ang)
            self.mf.grids.prune = prune_dict[self.prune]

            if self.use_chk and not self.newton:
                self.mf.chkfile = 'pyflosic.chk'
            if self.use_chk and not self.newton and os.path.isfile(
                    'pyflosic.chk'):
                self.mf.init_guess = 'chk'
                self.mf.update('pyflosic.chk')
                self.dm = self.mf.make_rdm1()
            if self.newton and self.xc != 'SCAN,SCAN':
                self.mf = self.mf.newton()
            if self.efield is not None:
                self.apply_electric_field(self.mf, self.efield)
            if self.df:
                self.mf = self.mf.density_fit()
            if self.dm is None:
                self.mf.kernel()
            else:
                self.mf.kernel(self.dm)
            self.mf = flosic(
                mol,
                self.mf,
                fod1,
                fod2,
                calc_forces=True,
                ham_sic=self.ham_sic)
            self.results['energy'] = self.mf['etot_sic'] * Ha
            # conversion to eV to match ase
            f = -1. * self.mf['fforces']
            self.results['fodforces'] = f * (Ha / Bohr)
            # conversion to eV/A to match ase
            if self.verbose >= 4:
                print('Analytic FOD force [Ha/Bohr]')
                print(f)
                print('fmax = %0.6f [Ha/Bohr]' %
                      np.sqrt((f**2).sum(axis=1).max()))
            self.results['evalues'] = self.mf['evalues'] * Ha
            # conversion to eV to match ase
            self.results['dipole'] = self.mf['dipole'] * Debye
            # conversion to e*A to match ase
            if self.pol:
                self.results['polarizability'] = Polarizability(
                    self.mf).polarizability() * (Bohr**3)
                # conversion to A**3 to match ase
            else:
                self.results['polarizability'] = None
        if self.mode == 'flosic-scf':
            self.mf = FLOSIC(
                mol=mol,
                xc=self.xc,
                fod1=fod1,
                fod2=fod2,
                grid=self.grid,
                ham_sic=self.ham_sic)
            self.mf.verbose = self.verbose
            self.mf.max_cycle = self.max_cycle
            self.mf.conv_tol = self.conv_tol
            if self.n_rad is not None and self.n_ang is not None:
                self.mf.grids.atom_grid = (self.n_rad,self.n_ang)
                self.mf.calc_uks.grids.atom_grid = (self.n_rad,self.n_ang)
            self.mf.grids.prune = prune_dict[self.prune]
            self.mf.calc_uks.grids.prune = prune_dict[self.prune]
            if self.use_chk and not self.newton:
                self.mf.chkfile = 'pyflosic.chk'
            if self.use_chk and not self.newton and os.path.isfile(
                    'pyflosic.chk'):
                self.mf.init_guess = 'chk'
                self.mf.update('pyflosic.chk')
                self.dm = self.mf.make_rdm1()
            if self.newton and self.xc != 'SCAN,SCAN':
                self.mf = self.mf.newton()
            if self.efield is not None:
                self.apply_electric_field(self.mf, self.efield)
            if self.df:
                self.mf = self.mf.density_fit()
            if self.dm is None:
                e = self.mf.kernel()
            else:
                e = self.mf.kernel(self.dm)
            self.results['esic'] = self.mf.esic * Ha
            # conversion to eV to match ase
            self.results['energy'] = e * Ha
            f = self.mf.get_fforces()
            self.results['fodforces'] = f * (Ha / Bohr)
            # conversion to eV/A to match ase
            if self.verbose >= 4:
                print('Analytic FOD force [Ha/Bohr]')
                print(f)
                print('fmax = %0.6f [Ha/Bohr]' %
                      np.sqrt((f**2).sum(axis=1).max()))
            self.results['evalues'] = self.mf.evalues * Ha
            # conversion to eV to match ase
            self.results['dipole'] = self.mf.dip_moment(
                verbose=self.verbose) * Debye
            # conversion to e*A to match ase
            if self.pol:
                p = Polarizability(self.mf).polarizability()
                self.results['polarizability'] = p * (Bohr**3)
                # conversion to A**3 to match ase
                if self.verbose >= 4:
                    print('Isotropic polarizability %.12g' %
                          ((p[0, 0] + p[1, 1] + p[2, 2]) / 3))
                    print('Polarizability anisotropy %.12g' % (
                        (.5 * ((p[0, 0] - p[1, 1])**2 + (p[1, 1] - p[2, 2])**2 + (p[2, 2] - p[0, 0])**2))**.5))
            else:
                self.results['polarizability'] = None
        if self.mode == 'flosic-scf' or self.mode == 'flosic-os':
            totalforces = []
            forces = np.zeros_like(nuclei.get_positions())
            fodforces = self.results['fodforces']
            totalforces.extend(forces)
            totalforces.extend(fodforces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces
        n_up, n_dn = self.mf.mol.nelec
        if n_up != 0 and n_dn != 0:
            e_up = np.sort(self.results['evalues'][0])
            e_dn = np.sort(self.results['evalues'][1])
            homo_up = e_up[n_up - 1]
            homo_dn = e_dn[n_dn - 1]
            self.results['homo'] = max(homo_up, homo_dn)
        elif n_up != 0:
            e_up = np.sort(self.results['evalues'][0])
            self.results['homo'] = e_up[n_up - 1]
        elif n_dn != 0:
            e_dn = np.sort(self.results['evalues'][1])
            self.results['homo'] = e_dn[n_dn - 1]
        else:
            self.results['homo'] = None


if __name__ == '__main__':

    from ase.vibrations import Raman

    # define system
    atoms = Atoms('N3', [(0, 0, 0), (1, 0, 0), (0, 0, 1)])
    basis = 'aug-cc-pVQZ'
    grid = 7
    conv_tol = 1e-8
    # define calculator
    calc = PYFLOSIC(
        mode='dft',
        atoms=atoms,
        basis=basis,
        grid=grid,
        conv_tol=conv_tol)
    atoms.set_calculator(calc)

    ram = Raman(atoms, delta=0.005)
    ram.run()
    ram.summary()
    ram.write_spectrum(
        out='raman.dat',
        quantity='raman',
        intensity_unit='A^4/amu')
