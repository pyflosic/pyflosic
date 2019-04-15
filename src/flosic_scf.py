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
# -----PYFLOSIC SIC CLASS----
#
# -----Authors-----
# main: 
#	Lenz Fiedler (LF) (fiedler.lenz@gmail.com)
# co:
#   Sebastian Schwalbe (SS) 
#   Torsten Hahn (TH) 
#   Jens Kortus (JK) 

 
# -----Imports-----
# Please note that this class imports the main SIC routine from flosic.py.
#

import time, sys
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.dft import rks, uks, UKS 
from pyscf.dft import rks, uks, UKS 
from pyscf import dft
from pyscf import scf
from pyscf.dft.uks import get_veff, energy_elec 
from pyscf.scf.uhf import get_fock
from flosic_os import flosic,xyz_to_nuclei_fod,ase2pyscf,get_multiplicity,dynamic_rdm,print_flo, write_force
from pyscf.dft import numint as ni
from pyscf.grad import rks as rks_grad
# this is needed for the O(N) stuff
import preopt as po

# this is for basic mpi support to 
# speed up veff evaluation
try:
    from mpi4py import MPI
    import mpi4py as mpi
except ImportError:
    mpi = None

# -----Notes-----
# FLO-SIC class by LF. This class allows for the self-consistent usage of the FLO-SIC 
# formalism in PySCF. It therefore calls the FLO-SIC routine given in flosic.py and 
# uses it to update the effective potential evaluated at every SCF step. The output of the 
# FLO-SIC class is twofold: the total energy value (FLO-SIC corrected) is the direct return 
# value of sic_object.kernel(). 
# Other values can be obtained by:
#		sic_object.flo -- Will hold the FLOs.	
#		sic_object.fforces -- Will hold the FOD forces. (after the get_fforces routine has 
# 								been called or if calc_forces == True.
#		sic_object.homo_flosic -- Will hold the FLO-SIC HOMO value.
#		sic_object.esic -- Will hold the total energy correction.


#-----Routines----


# This routine creates the new effective potential.
# It is: veff_dft + veff_sic 

def get_flosic_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    # pyscf standard call for scf cycle 0.
    veff = uks.get_veff(ks=ks.calc_uks,mol=ks.mol,dm=dm,dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)
    
    if mol is None: mol = ks.mol 
    # Build the hamiltonian to get the KS wave functions.
    dim = np.shape(ks.calc_uks.mo_coeff)
    s1e = ks.get_ovlp(mol)
    h1e = ks.get_hcore(mol)
    hamil = ks.get_fock(h1e, s1e, vhf_last, dm)	
    
    # Get the KSO.
    ks_new = np.zeros((2,dim[1],dim[1]), dtype=np.float64)
    
    try:
        if dm_last == 0:
            # First SCF cycle: do nothing.
            pass
    except:
        # Every other DFT cycle: Build the KS wavefunctions with the Hamiltonian, then give them to the UKS object that is the input for flosic.
        trash, ks_new = ks.eig(hamil,s1e)
        ks_inter = np.array(ks_new)
        
        # Update UKS object.
        ks.calc_uks.mo_coeff = ks_inter.copy()

    # If ldax is enabled, the xc functional is set to LDA exchange only.
    if ks.ldax == True:
        xc_sav = ks.calc_uks.xc 
        ks.calc_uks.xc = 'LDA,'

    # Call the FLOSIC routine with the UKS object.
    
    # This for the fixed Vsic modus. 

    # If Vsic values are present and the Vsic potential should not 
    # be updated use these values.
    # (THa: the outer if ... clause was added to prevent the
    # sic potentials to be calculated during initialization)
    _t0 = time.time()
    #print('>> ks.fixed_vsic', ks.fixed_vsic)
    #print('>>', ks.neval_vsic)
    #sys.exit()
    if ks.neval_vsic > -1:
        if ks.fixed_vsic != 0.0 and (ks.num_iter % ks.vsic_every) != 0:
            if ks.verbose >= 4:
                print('Use fixed Vsic (cycle = %i)' % ks.num_iter)
            flo_veff = flosic(ks.mol, ks.calc_uks, ks.fod1, ks.fod2,\
            datatype=np.float64,calc_forces=ks.calc_forces,debug=ks.debug,\
            nuclei=ks.nuclei,l_ij=ks.l_ij,ods=ks.ods,\
            fixed_vsic=ks.fixed_vsic,ham_sic=ks.ham_sic)
        
        # If no Vsic values are present or the the Vsic values should be 
        # updated calcualte new Vsic values.
        
        
        # !! THa: Possible BUG 
        # ks.fixed_vsic == 0.0 may never be 'True' because
        # a float-value is amost never exactly zero
        # better use: np.isclose(ks.fixed_vsic, 0.0)
        # !!
        if ks.fixed_vsic == 0.0 or (ks.num_iter % ks.vsic_every) == 0:
            if ks.verbose >= 4:
                print('Calculate new Vsic (cycle = %i)' %ks.num_iter) 
            flo_veff = flosic(ks.mol,ks.calc_uks,ks.fod1,ks.fod2,\
            datatype=np.float64,calc_forces=ks.calc_forces,debug=ks.debug,\
            nuclei=ks.nuclei,l_ij=ks.l_ij,ods=ks.ods,ham_sic=ks.ham_sic)
            ks.fixed_vsic = flo_veff['fixed_vsic']
    else:
        flo_veff = veff 
    
    _t1 = time. time()
    if mol.verbose > 3:
        print("FLO-SIC time for SIC potential: {0:0.1f} [s]".format(_t1-_t0))
    # increase a magic counter
    ks.num_iter = ks.num_iter + 1 


    # If ldax is enabled, the change to xc is only meant for the FLO-SIC part and 
    # therefore has to be changed back.
    if ks.ldax == True:
        ks.calc_uks.xc = xc_sav
    
    # Assign the return values.
    # The total energies of DFT and FLO-SIC
    if ks.neval_vsic > -1:
        sic_etot = flo_veff['etot_sic']
        dft_etot = flo_veff['etot_dft']
        # The FLOs.
        ks.flo = flo_veff['flo']
        # The FOD forces.
        ks.fforces = flo_veff['fforces']
        # The FLO-SIC HOMO energy eigenvalue.
        ks.homo_flosic = flo_veff['homo_sic']
        ks.evalues = flo_veff['evalues']
        ks.lambda_ij = flo_veff['lambda_ij']
        # Developer modus: atomic forces (AF) 
        if ks.debug == True:
            ks.AF = flo_veff['AF']
    else:
        sic_etot = ks.e_tot
        dft_etot = ks.e_tot
        ks.flo = ks.mo_coeff
        ks.homo_flosic = 0.0
        
    
    try:
        # First SCF cycle: veff = veff_dft and the SIC is zero.
        if dm_last == 0:
            sic_veff = veff
            sic_etot = dft_etot			
    except:
        # Every other DFT cycle: Build veff as sum of the regular veff and the SIC 
        # potential. 
        sic_veff = veff+flo_veff['hamil']
    
        # Update the density matrix.
        dm_new = dynamic_rdm(ks.flo,ks.calc_uks.mo_occ)
        dm = dm_new.copy()
        ks.mo_coeff = ks.flo
    
    # Give back the FLO-SIC energy correction and the corrected potential. This libtagarray 
    # formalism is defined by pyscf.
    sic_back = sic_etot-dft_etot
    veff_sic = lib.tag_array(sic_veff, ecoul=veff.ecoul, exc=veff.exc, vj=veff.vj, vk=veff.vk, esic=(sic_back))
    
    # Return the exchange-correlation energy and the FLO-SIC energy correction.
    ks.exc = veff.exc
    ks.esic = sic_back
    
    # increase another magic counter ;-)
    ks.neval_vsic += 1
    
    return veff_sic  


# Every DFT calculation in pyscf calls the energy_elec function multiple times. It 
# calculates the electronic energy that is then combined with the nuclei-electron 
# interaction to the total energy. 

def flosic_energy_elec(mf, dm=None, h1e=None, vhf=None):
    # Get the nuclei potential.
    h1e = mf.get_hcore()

    # This is the nuclei-electron interaction.
    e_nuc = np.einsum('ij,ji', h1e, dm[0]) + np.einsum('ij,ji', h1e, dm[1])
	
    try:
        # Every other DFT cycle: electronic energy calculated as sum of the contributions.
        e_correction = vhf.__dict__['ecoul']+vhf.__dict__['exc']+vhf.__dict__['esic']	
        e_sic = (e_correction,vhf.__dict__['ecoul'])
        
        # This part looks odd, but it is correct.
        e = (e_sic[0] + e_nuc , e_sic[1])
	
    except:
        # First SCF cycle: regular DFT energy.		
        e = energy_elec(mf, dm=dm, h1e=h1e, vhf=vhf)
	
    return e

# Every DFT calculation in PySCF calls the energy_tot function multiple times. It 
# calculates the total energy and is basically an interface to the electronic energy. This 
# function simply makes sure that the correct FLO are handed to flosic_energy_elec. All 
# the actual work is done there.

def flosic_energy_tot(mf, dm=None, h1e=None, vhf=None):
    dm = dynamic_rdm(mf.flo,mf.calc_uks.mo_occ)
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    return e_tot.real

# This is the FLO-SIC class that allows for self-consistent FLO-SIC calculations. It 
# inherits a lot of functionalities from UHF/RKS/UKS classes.

class FLOSIC(uhf.UHF):
    '''FLOSIC 
    See pyscf/dft/rks.py RKS class for the usage of the attributes
    Rewritten UKS class. ''' # This part is directly taken from the RKS class.
    def __init__(self, mol, xc, fod1, fod2, ldax=False, grid_level=3, calc_forces = False,debug=False,nuclei=None,l_ij=None,ods=None,fixed_vsic=None,num_iter=0,ham_sic='HOO',vsic_every=1, init_dm=None):
        uhf.UHF.__init__(self, mol)
        rks._dft_common_init_(self)
        
        # Give the input variables to the SIC object.
        self.mol = mol # Molecular geometry.
        self.xc = xc # Exchange-correlation functional
        self.fod1 = fod1 # FOD geometry for first spin channel.
        self.fod2 = fod2 # FOD geometry for second spin channel.
        self.nfod = [fod1.positions.shape[0],fod2.positions.shape[0]]
        self.nuclei = nuclei 
        self.ldax = ldax # If True, LDA exchange is used for FLO-SIC (debugging mainly).
        self.is_first = True # Used to determine which SCF cycle we are in.
        self.grid_level = grid_level # Grid level.
        self.grids.level = grid_level 
        self.calc_forces = calc_forces # Determines whether or not FOD forces are calculated in every step. Default: False.
        self.debug = debug # enable debugging output 		
        self.l_ij = l_ij # Lagrangian multiplier output 
        self.lambda_ij = [] # # Lagrangian multiplier matrix 
        self.ods = ods # orbital density scaling 
        
        # creation of an internal UKS object for handling FLO-SIC calculations.
        lib.logger.TIMER_LEVEL = 4
        mol.verbose = 0
        calc_uks = UKS(mol)
        calc_uks.xc = self.xc
        calc_uks.max_cycle = 0
        calc_uks.grids.level = grid_level
        
        # if an initial density matrix is given
        # initialize the subclass with it
        if init_dm is not None:
            calc_uks.kernel(init_dm)
        else:
            calc_uks.kernel()
        self.calc_uks = calc_uks
        
        # initialize var to store the FLO's
        self.calc_uks.flo_coeff = None
        self.calc_uks.flo_last = None
        self.on = None
        self.calc_uks.on = self.on
        
        # add a explicite reference to the FLOSIC class
        # to the helper class to make instances of FLOSIC
        # always accessible via calc_uks
        # (helps simplifying code)
        self.calc_uks.FLOSIC = self
        
        ## Tha added variables to ensure O(N) 
        # and in-scf fod optimization functionality
        self.preopt             = False
        self.preopt_start_cycle = 0
        self.preopt_conv_tol    = 2e-5
        self.preopt_fmin        = 0.005
        self.preopt_fix1s       = True
        self.opt_init_mxstep    = 0.0050
        self.opt_mxstep         = 0.0100
        
        #self.esic_per_cycle = [0.0]
        self.pflo               = None  # list of FLO objects
        self.cesicc             = None # list of ESICC objects
        
        
        ## /THa
        
        # Parameters to coordinate FLO-SIC output. 
        dim = np.shape(self.calc_uks.mo_coeff) # Dimensions of FLO-SIC.
        dim1 = np.shape(fod1)
        dim2 = np.shape(fod2)
        # set nspin class var (its just handy)
        self.nspin = 2
        if dim2 == 0: self.nspin = 1
        
        self.flo = np.zeros((2,dim[1],dim[1]), dtype=np.float64) # Will hold the FLOs.	
        self.fforces = np.zeros((dim1[0]+dim2[0],3), dtype=np.float64) # Will hold the FOD forces.
        if fixed_vsic is None:
            self.fixed_vsic = None 
        if fixed_vsic is not None:
            self.fixed_vsic = fixed_vsic
        
        
        print('fixed_vsic', fixed_vsic)
        #sys.exit()
        self.homo_flosic = 0.0 # Will hold the FLO-SIC HOMO value.
        self.esic = 0.0 # Will hold the total energy correction.
        self.exc = 0.0	# Will hold the FLO-SIC exchange-correlation energy.
        self.evalues = 0.0 # Will hold the FLO-SIC evalues. 	
        self.AF = 0.0
        self.num_iter = num_iter # Number of iteration
        self.vsic_every = vsic_every # Calculate the vsic after e.g 50 cycles 
        self.ham_sic = ham_sic # SIC hamiltonian 
        # This is needed that the PySCF mother class get familiar with all new variables. 
        self._keys = self._keys.union(['grid_level','fod1','homo_flosic','exc','evalues','calc_uks','esic','flo',
            'fforces','fod2','ldax','calc_forces','is_first','debug','nuclei','AF','l_ij','ods','lambda_ij',
            'num_iter','vsic_every','fixed_vsic','ham_sic','preopt_fix1s','cesicc','nspin','opt_mxstep','preopt',
            'opt_init_mxstep','pflo','preopt_start_cycle','preopt_conv_tol','on','preopt_fmin',
            'nfod','use_mpi'])

        # make sure initial mo_coeff's are in synch with the 
        # helper - subclass
        self.mo_coeff = np.asarray(calc_uks.mo_coeff)
        
        
        # this is for basic mpi support
        self.use_mpi = False
        if mpi is not None:
            wsize = MPI.COMM_WORLD.Get_size()
            #print(">>> WSIZE {}".format(wsize))
            #sys.exit()
            if wsize > 1: self.use_mpi = True
        
    
    def kernel(self, *args, **kwargs):
        # signal all slaves that wanne do update_vsic
        # with mpi
        if self.use_mpi:
            comm = MPI.COMM_WORLD
            wsize = comm.Get_size()
            for inode in range(1,wsize):
                comm.send('init', dest=inode, tag=11)
            
            info = {
                'atom'      : self.calc_uks.mol.atom,
                'basis'     : self.calc_uks.mol.basis,
                'charge'    : self.calc_uks.mol.charge,
                'spin'      : self.calc_uks.mol.spin,
                'max_memory': self.calc_uks.mol.max_memory,
                'xc'        : self.calc_uks.xc,
                'grid_level': self.calc_uks.grids.level
            }
            for inode in range(1,wsize):
                comm.send(info, dest=inode, tag=12)
            comm.Barrier()
        
        return super().kernel(*args, **kwargs)
        
        
    
    #def kernel(self, dm):
#    def pre_kernel(self, envs):
#        # start the mpi_worker if requested
#        print(">>> pre_kernel")
#        #if self.use_mpi:
#        #    rank = MPI.COMM_WORLD.Get_rank()
#        #    if rank > 0:
#        #        print('>>> starting mpi_worker on rank {}'.format(rank), flush=True)
#        #        po.mpi_worker(calc_uks)
#        #        #MPI.Finalize()
#    
#    def post_kernel(self, envs):
#        print(">>> post_kernel")
#        #if self.use_mpi:
#        #    rank = MPI.COMM_WORLD.Get_rank()
#        #    if rank == 0:
#        #        comm = MPI.COMM_WORLD
#        #        wsize = MPI.COMM_WORLD.Get_size()
#        #        # shut down mpi workers on all nodes
#        #        for inode in range(1,wsize):
#        #            comm.send('finalize', dest=inode, tag=11)
#    
    
    # Flags that might be helpful for debugging.
    def dump_flags(self):
        uhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags()
    
    def get_esic(self):
        '''Returns the SIC energy (Ha)'''
        esicspin = [p._esictot for p in self.pflo]
        return np.sum(esicspin)
        
    
    def update_fpos(self, fpos=None):
        '''Trigger an update the internal positions according to given positions
        fpos.
        If fpos is not given, then the positions stored in self.fodx will be checked
        against positions in self.pflo[s]. If they differ, the positions of pflo are updated.
        '''
        #print('>>> update_fpos called:', self.nfod)
        #print('>>> update_fpos:', np.sum(self.fod1.positions))
        if fpos is not None:
            self.fod1.positions[:,:] = fpos[:self.nfod[0],:]
            if self.nspin == 2:
                self.fod2.positions[:,:] = fpos[self.nfod[0]:self.nfod[0]+self.nfod[1],:]
            try:
                for j, flo in enumerate(self.pflo):
                    if j == 0: flo.fpos[:,:] = fod1.positions[:,:]/units.Bohr
                    if j == 1: flo.fpos[:,:] = fod2.positions[:,:]/units.Bohr
            except:
                # pflo is none
                pass
        else:
            # re-set the pflo.fpos variables (in case the data structures exist)
            try:
                for j, flo in enumerate(self.pflo):
                    if j == 0: flo.fpos[:,:] = fod1.positions[:,:]/units.Bohr
                    if j == 1: flo.fpos[:,:] = fod2.positions[:,:]/units.Bohr
            except:
                # pflo is none
                pass
        #print('<<< update_fpos:', np.sum(self.fod1.positions), flush=True)



    def set_on(self,on):
        """initialze the O(N) methods"""
        self.on = on
        self.calc_uks.on = on
        if on is not None:
            logger.info(self,'O(N) mode enabled')
            on.print_stats()
        else:
            logger.info(self,'O(N) mode disabled')

    # This routine can be called by a SIC object any time AFTER a SCF cycle has been 
    # completed. It calculates and outputs the forces.
    
    def fod_gradients(self):
        '''Return the gradients of the SIC energy with respect to the
        FOD positions (in fod1 and fod2)
        
        Return: np.array
            an array with dimensions (3,Nup+Ndn) containing the gradients in
            cartesian coordinates
        '''
        # check if we did initialize the class
        # if not, do it
        if self.pflo is None:
            logger.warn(self,'FLOSIC.get_fforces() called before kernel()')
            self.pflo = list()
            for s in range(self.nspin):
                fod = self.fod1
                if s == 1: fod = self.fod2
                self.pflo.append(po.FLO(self.calc_uks, s, fod.positions))
        
        tfodf = time.perf_counter()
        logger.info(self, 'FLO-SIC calculating FOD gradients ...')
        
        #if self.on is None:
        ff = list()
        for s in range(self.nspin):
            # ff.append(self.pflo[s].get_desic_dai())
            ff.extend(self.pflo[s].get_desic_dai().tolist())
        # self.fgrad = np.reshape(np.array(ff),(-1,3))
        self.fgrad = np.reshape(np.array(ff),(-1,3))
        
        tfodf = time.perf_counter() - tfodf
        logger.info(self, 'FOD gradients done, time: {0:0.2f}'.format(tfodf))
        
        return self.fgrad
        
    def get_fforces(self, writeout=True):
        '''Convinience function, better use self.fod_gradients()'''
        wmsg = \
        '''WARNING: FLOSIC.get_fforces() now returns forces and *NOT* gradients!
        You may want to use FLOSIC.fod_gradients() instead !
        '''
        if self.mol.verbose > 3: print(wmsg)
        
        self.fforces = -1.0*self.fod_gradients()
        
        # generate fforce.dat file,
        # unfortunately fforce.dat contains the gradients and 
        # *NOT* the forces
        if writeout:
            write_force(self.fforces)
        
        return self.fforces

    
    # Set a new potential calculator.
    get_veff = get_flosic_veff 
    vhf = get_flosic_veff
    
    # set a new energy calculator.
    energy_elec = flosic_energy_elec
    energy_tot = flosic_energy_tot
    define_xc_ = rks.define_xc_
    
    # initialize VSIC eval counter
    neval_vsic = -1
    

# New class that allows for the spin to change during the FLO-SIC calculation.
# Based on the float_occ_ routine for UHF from PySCF. It is in principle a copy of 
# float_occ_ with the correct treament of the FOD geometry added.

def sic_occ_(mf):
    # This part is directly taken from PySCF.
    from pyscf.scf import uhf
    assert(isinstance(mf, uhf.UHF))
    def get_occ(mo_energy, mo_coeff=None):
        # Spin configuration is only changed ONCE at the beginning. Elsewise, SCAN will behave very inconsistently.
        if mf.is_first == True:
            mol = mf.mol
            ee = np.sort(np.hstack(mo_energy))
            n_a = np.count_nonzero(mo_energy[0]<(ee[mol.nelectron-1]+1e-1))
            n_b = mol.nelectron - n_a
            if mf.nelec is None:
                nelec = mf.mol.nelec
            else:
                nelec = mf.nelec
            if n_a != nelec[0]:
                logger.info(mf, 'change num. alpha/beta electrons '
                                ' %d / %d -> %d / %d',
                                nelec[0], nelec[1], n_a, n_b)

            # If the spin configuration has changed, the FOD configuration needs to do as 
            # well.
            # First, initialize needed parameters.					
            dim = np.shape(mf.calc_uks.mo_coeff)
            occ = np.zeros((2,dim[1]), dtype=np.float64)								
            dim1 = np.shape(mf.fod1)
            dim2 = np.shape(mf.fod2)
		
            # Calculate new and old spin polarization.
            difforig = dim1[0] - dim2[0]
            diffnew = n_a - n_b
            diff = diffnew - difforig
			
            # If something has changed, update the FODs.
            if dim1[0] != n_a and dim2[0] != n_b:
                print('Electronic configuration has been changed, changing FOD geometry.')

                # Update the FODs.
                if diff > 0:
                    counter = diff/2
                    for i in range(0,int(counter)):
                        mf.fod1.append(mf.fod2[i])
                    del mf.fod2[0:int(counter)]
                
                if diff < 0:
                    counter = abs(diff)/2
                    for i in range(0,int(counter)):				
                        mf.fod2.append(mf.fod1[i])
                    del mf.fod1[0:int(counter)]
					
            # Update the occupation of the internal UKS object as well	
            for i in range(0,n_a):
                occ[0,i] = 1.0
            for i in range(0,n_b):
                occ[1,i] = 1.0
            mf.calc_uks.mo_occ = occ.copy()	
			
            # Taken from the UHF routine.
            mf.nelec = (n_a, n_b)
			
            # As discussed above, only for the FIRST SCF iteration the spin configuration is 
            # variable.
            mf.is_first = False
        return uhf.UHF.get_occ(mf, mo_energy, mo_coeff)
    mf.get_occ = get_occ
    return mf
dynamic_sz_ = sic_occ = sic_occ_

# This routine is supposed to replace float_occ_ to enable a more direct manipulation of 
# the parameters for the variable spin calculation. This is especially important for 
# calculations with the SCAN functional; if this is not done correctly they might crash 
# due to very small energy differences for different spin configurations. The routine is 
# in principle a copy of float_occ_ restricting it to only vary the spin for the first SCF 
# iteration.
# NOTE: To use this function, one has to add m2.is_first = True to the calculator before 
# doing m2.kernel()

def dft_occ_(mf):
    # This part is directly taken from PySCF.
    from pyscf.scf import uhf
    assert(isinstance(mf, uhf.UHF))
    def get_occ(mo_energy, mo_coeff=None):
        # Spin configuration is only changed ONCE at the beginning. Elsewise, SCAN will behave very inconsistently.
        if mf.is_first == True:
            mol = mf.mol
            ee = np.sort(np.hstack(mo_energy))
            n_a = np.count_nonzero(mo_energy[0]<(ee[mol.nelectron-1]+1e-1))
            n_b = mol.nelectron - n_a
            if mf.nelec is None:
                nelec = mf.mol.nelec
            else:
                nelec = mf.nelec
            if n_a != nelec[0]:
                logger.info(mf, 'change num. alpha/beta electrons '
                                ' %d / %d -> %d / %d',
                                nelec[0], nelec[1], n_a, n_b)
            mf.nelec = (n_a, n_b)
			
            # As discussed above, only for the FIRST SCF iteration the spin configuration is 
            # variable.
            mf.is_first = False

        return uhf.UHF.get_occ(mf, mo_energy, mo_coeff)
    mf.get_occ = get_occ
    return mf
dynamic_sz_ = dft_occ = dft_occ_


if __name__ == '__main__':
    # Test example for the FLOSIC class.
    # This simple example shows of all of the features of the SIC class.
    from ase.io import read
    import sys
    import numpy as np
    from pyscf import gto		
    import os 
    from ase import Atom, Atoms
    
    po.mpi_start()
    
    # Path to the xyz file 
    f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/ase_pyflosic_optimizer/LiH.xyz'
    f_xyz = 'SiH4_guess.xyz'
    # Read the input file.
    ase_atoms = read(f_xyz)
    
    # Split the input file.
    pyscf_atoms,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(ase_atoms)
	
    
    
    
    
    
    CH3SH = '''
    C -0.04795000 +1.14952000 +0.00000000
    S -0.04795000 -0.66486000 +0.00000000
    H +1.28308000 -0.82325000 +0.00000000
    H -1.09260000 +1.46143000 +0.00000000
    H +0.43225000 +1.55121000 +0.89226000
    H +0.43225000 +1.55121000 -0.89226000
    '''
    # this are the spin-up descriptors
    fod1 = Atoms([
    Atom('X', (-0.04795000, -0.66486000, +0.00000000)),
    Atom('X', (-0.04795000, +1.14952000, +0.00000000)),
    Atom('X', (-1.01954312, +1.27662578, +0.00065565)),
    Atom('X', (+1.01316012, -0.72796570, -0.00302478)),
    Atom('X', (+0.41874165, +1.34380502, +0.80870475)),
    Atom('X', (+0.42024357, +1.34411742, -0.81146545)),
    Atom('X', (-0.46764078, -0.98842277, -0.72314717)),
    Atom('X', (-0.46848962, -0.97040067, +0.72108036)),
    Atom('X', (+0.01320210, +0.30892333, +0.00444147)),
    Atom('X', (-0.28022018, -0.62888360, -0.03731204)),
    Atom('X', (+0.05389371, -0.57381853, +0.19630494)),
    Atom('X', (+0.09262866, -0.55485889, -0.15751914)),
    Atom('X', (-0.05807583, -0.90413106, -0.00104673))])
    
    # this are the spin-down descriptors
    fod2 = Atoms([
    Atom('He',( -0.04795000, -0.66486000, +0.0000000)),
    Atom('He',( -0.04795000, +1.14952000, +0.0000000)),
    Atom('He',( +1.12523084, -0.68699049, +0.0301970)),
    Atom('He',( +0.40996981, +1.33508869, +0.8089839)),
    Atom('He',( +0.40987059, +1.34148952, -0.8106910)),
    Atom('He',( -0.49563876, -0.99517303, +0.6829207)),
    Atom('He',( -0.49640020, -0.89986161, -0.6743094)),
    Atom('He',( +0.00073876, +0.28757089, -0.0298617)),
    Atom('He',( -1.03186573, +1.29783767, -0.0035536)),
    Atom('He',( +0.04235081, -0.54885843, +0.1924678)),
    Atom('He',( +0.07365725, -0.59150454, -0.1951675)),
    Atom('He',( -0.28422422, -0.61466396, -0.0087913)),
    Atom('He',( -0.02352948, -1.0425011 ,+0.01253239))])
    
    
    
    
    
    
    
    
    
    
    
    # Get the spin and charge.
    charge = 0
    spin = 0

    # Uncomment the basis set you want to use.
    b = 'cc-pvdz'
	
    # The ghost option enables ghost atoms at the FOD positions. Mostly obsolete.
	
    # Build the mol object.
    mol = gto.M(atom=CH3SH, basis={'default':b},spin=spin,charge=charge)
		
    # Adjust verbosity as desired. 
    mol.verbose = 4
    mol.max_memory = 8000
    
    # Calculation parameters.
    max_cycle = 40
    grid_level = 7
    conv_tol = 1e-6
    xc = 'LDA,PW'
    
    # Build the SIC calculator.
    m = FLOSIC(mol,
        xc=xc,
        fod1=fod1,
        fod2=fod2,
        grid_level=grid_level
    )
    m.max_cycle = max_cycle
    m.conv_tol = conv_tol
    
    from onstuff import ON
    myon = ON(mol,[fod1.positions,fod2.positions], grid_level=grid_level)
    myon.nshell = 2
    myon.build()
    
    m.set_on(myon)
    
    # Do the calculation.
    e1_calc = m.kernel()
    
    
    po.mpi_stop()
    sys.exit()
    
    
    print('Pyflosic total energy: ',e1_calc)
    g1 = m.fod_gradients()
    
    # change the positions of fod1
    fod1.rattle(stdev=0.2)
    
    e2_calc = m.kernel(dm0=m.make_rdm1())
    print('Pyflosic total energy: ',e2_calc)
    g2 = m.fod_gradients()
    print('Diff gradients: ', np.linalg.norm(g1-g2))
    
    
    time.sleep(2.0)
    
    po.mpi_stop()
        
    
    
