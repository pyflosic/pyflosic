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
# pyflosic-ase-optimizer 
# author:	                Sebastian Schwalbe, Jakob Kraus 
# task:  	                optimzation of FOD positions/electronic geometry 
# todo:		                separate core functions and tests
# CHANGELOG 17.02.2020:         new keywords: dm,cart,output (from ase_pyflosic_calculator)
#                               removed old keywords: force_consistent (because ase_pyflosic_calculator cannot calculate the free energy anyway)
#                               optimizer name: LineSearch->BFGSLineSearch
#                               new optimizer: LBFGSLineSearch
#                               modified imports

 
from ase.optimize import LBFGS, BFGS, BFGSLineSearch, LBFGSLineSearch, FIRE, GPMin, Berny
from ase.optimize.sciopt import SciPyFminCG
from ase_pyflosic_calculator import PYFLOSIC
from ase.constraints import FixAtoms 
from pyscf.data import radii
from ase.units import Ha,Bohr

def flosic_optimize(mode='flosic-scf',atoms,charge=0,spin=0,xc='lda,pw',basis='STO-3G',ecp=None,opt='FIRE',maxstep=0.2,label='OPT_FRMORB',fmax=1.5e-4*(Ha/Bohr),steps=1000,max_cycle=300,conv_tol=1e-6,grid=3,df=True,newton=False,use_chk=False,verbose=4,efield=None,ham_sic='HOOOV',dm=None,cart=False,output=None,solvation=None,lmax=10,eta=0.1,lebedev_order=89,radii_table=None,eps=78.3553):
    # mode 			dft only optimizes nuclei positions 
    #				flosic only optimizes FOD positions (one-shot)
    #				flosic-scf only optimizes FOD positions (self-consistent)
    # atoms 			ase atoms object 
    # charge 		 	charge 
    # spin			spin state = 2S = # alpha - # beta
    # xc			exchange-correlation functional 
    # basis	 		GTO basis set 
    # ecp			if an ECP basis set is used, you must give this extra argument 
    # opt 			optimizer (FIRE, LBFGS, ...) 
    # maxstep			stepwidth of the optimizer  
    # label			label for the outputs (logfile and trajectory file) 
    # fmax 			maximum absolute force on each atom
    # steps 		        maximum number of steps for the optimizer 
    # max_cycle     	        maximum number of SCF cycles 
    # conv_tol			energy threshold 
    # grid			numerical mesh 
    # df                        density fitting
    # newton		        use newton SCF cycle 
    # use_chk			restart from chk file 
    # verbose 			output verbosity 
    # efield 			applying an external efield 
    # ham_sic			the different unified Hamiltonians HOO and HOOOV 
    # dm                        density matrix
    # cart                      use Cartesian GTO basis and integrals (6d,10f,15g etc.)
    # output                    specify an output file, if None: standard output is used
    # solvation                 specify if solvation model should be applied (COSMO)
    # lmax                      maximum l for basis expansion in spherical harmonics for solvation
    # eta                       smearing parameter in solvation model
    # lebedev_order             order of integration for solvation model
    # radii_table               vdW radii for solvation model
    # eps                       dielectric constant of solvent
    
    
    opt = opt.lower()
    
    
	
    # set up the ase calculator
    calc = PYFLOSIC(atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,mode=mode,ecp=ecp,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,df=df,newton=newton,verbose=verbose,efield=efield,ham_sic=ham_sic,use_chk=use_chk,dm=dm,cart=cart,output=output,solvation=solvation,lmax=lmax,eta=eta,lebedev_order=lebedev_order,radii_table=radii_table,eps=eps)
    
    # assign the ase calculator to the ase atoms object
    atoms.set_calculator(calc)
	
    # select an ase optimizer 
    if opt == 'fire':
        dyn = FIRE(atoms,
                   logfile=label+'.log',
                   trajectory=label+'.traj',
                   dt=0.15,
                   maxmove=maxstep)
	
    elif opt == 'lbfgs':
        dyn = LBFGS(atoms,
                    logfile=label+'.log',
                    trajectory=label+'.traj',
                    use_line_search=False,
                    maxstep=maxstep,
                    memory=10)
	
    elif opt == 'bfgs':
        dyn = BFGS(atoms,
                   logfile=label+'.log',
                   trajectory=label+'.traj',
                   maxstep=maxstep)
	
    elif opt == 'bfgslinesearch':
        dyn = BFGSLineSearch(atoms,
                             logfile=label+'.log',
                             trajectory=label+'.traj',
                             maxstep=maxstep)
    
    elif opt == 'lbfgslinesearch':
        dyn = LBFGSLineSearch(atoms,
                             logfile=label+'.log',
                             trajectory=label+'.traj',
                             maxstep=maxstep)
	
    elif opt == 'cg':
        dyn = SciPyFminCG(atoms,
                          logfile=label+'.log',
                          trajectory=label+'.traj',
                          callback_always=False,
                          alpha=70.0,
                          master=None)
    
    elif opt == 'gpmin':
        dyn = GPMin(atoms,
                    logfile=label+'.log',
                    trajectory=label+'.traj',
                    update_prior_strategy='average',
                    update_hyperparams=True)
    

    elif opt == 'berny':
        dyn = Berny(atoms,
                    logfile=label+'.log',
                    trajectory=label+'.traj',
                    master=None)

    # run the actual optimization
    dyn.run(fmax=fmax, steps=steps)
    return atoms 

if __name__ == '__main__':
    
    from ase.io import read 
    import os

    # path to the xyz file 
    f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/ase_pyflosic_optimizer/LiH.xyz'
    atoms = read(f_xyz)

    charge = 0 
    spin = 0

    # choose some parameters 
    mode = ['dft','flosic','flosic-scf'][1]
    xc = ['LDA,PW','PBE,PBE','SCAN,SCAN'][0]
    # bfd := effective core potential/ pseudopotential need the use of ecp. 
    basis = ['sto3g','6-31G','6-311++Gss','cc-pvqz','bfd-vdz','bfd-vtz','bfd-vqz','bfd-v5z'][0]
    ecp = [None,'bfd_pp'][0]
    opt = ['FIRE','BFGS','LBFGS','CG','LineSearch'][0]

    # perform the optimization
    os_flosic = flosic_optimize('flosic-os',atoms,charge,spin,xc,basis,ecp,opt='FIRE',maxstep=0.2)	
