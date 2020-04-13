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
# author:	S. Schwalbe 
# task:  	optimzation of FOD positions/electronic geometry 
# todo:		spearate core functions and tests
 
from ase.optimize import LBFGS, BFGS, BFGSLineSearch, FIRE, MDMin
from ase.optimize.sciopt import SciPyFminCG
from ase_pyflosic_calculator import PYFLOSIC
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from ase.io import read 
#from nrlmol_basis import get_dfo_basis
from ase.constraints import FixAtoms 

def flosic_optimize(mode,atoms,charge,spin,xc,basis,ecp=None,opt='FIRE',maxstep=0.2,label='OPT_FRMORB',fmax=0.0001,steps=1000,max_cycle=300,conv_tol=1e-5,grid=7,ghost=False,use_newton=False,use_chk=False,verbose=0,debug=False,efield=None,l_ij=None,ods=None,force_consistent=False,fopt='force',fix_fods=False,ham_sic='HOO',vsic_every=1):
    # -----------------------------------------------------------------------------------
    # Input 
    # -----------------------------------------------------------------------------------
    # mode 			...	dft only optimize nuclei positions 
    #				flosic only optimize FOD positions (one-shot)
    #				flosic-scf only optimize FOD positions (self-consistent)
    # atoms 		...	ase atoms object 
    # charge 		... 	charge 
    # spin			...	spin state = alpha - beta 
    # xc			... 	exchange correlation functional 
    # basis	 		... 	GTO basis set 
    # ecp			...	if a ECP basis set is used you must give this extra argument 
    # opt 			...	optimizer (FIRE, LBFGS, ...) 
    # ----------------------------------------------------------------------------------
    # Additional/optional input 
    # ----------------------------------------------------------------------------------
    # maxstep		...	stepwidth of the optimizer  
    # label			...	label for the outputs (logfile and trajectory file) 
    # fmax 			...	maximum force 
    # steps 		...     maximum steps for the optimizer 
    # max_cycle     	...     maxium scf cycles 
    # conv_tol		...	energy threshold 
    # grid			... 	numerical mesh 
    # ghost 		...	use ghost atom at positions of FODs 
    # use_newton		...	use newton scf cycle 
    # use_chk		...	restart from chk fiels 
    # verbose 		...	output verbosity 
    # debug			...	extra output for debugging reasons 
    # efield 		...	applying a external efield 
    # l_ij			...	developer option: another optimitzation criterion, do not use for production 
    # ods			...	developer option orbital damping sic, rescale SIC, do not use for production 
    # force_cosistent	...     ase option energy consistent forces 	
    # fopt			...	optimization trarget, default FOD forces 
    # fix_fods		...	freeze FODS during the optimization, might use for 1s/2s FODs 
    # ham_sic		...	the different unified Hamiltonians HOO and HOOOV 

    if fix_fods != False:
        c = FixAtoms(fix_fods)
        atoms.set_constraint(c)
	
    # Select the wished mode.
    # DFT mode
    if mode == 'dft':
        [geo,nuclei,fod1,fod2,included] = xyz_to_nuclei_fod(atoms)
        atoms = nuclei 
        calc = PYFLOSIC(atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,mode='dft',ecp=ecp,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,ghost=ghost,use_newton=use_newton,verbose=verbose,debug=debug,efield=efield,l_ij=l_ij,ods=ods,fopt=fopt,ham_sic=ham_sic,vsic_every=vsic_every)
    # FLO-SIC one-shot (os) mode 
    if mode == 'flosic-os':
        calc = PYFLOSIC(atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,mode='flosic-os',ecp=ecp,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,ghost=ghost,use_newton=use_newton,verbose=verbose,debug=debug,efield=efield,l_ij=l_ij,ods=ods,fopt=fopt,ham_sic=ham_sic,vsic_every=vsic_every)
    # FLO-SIC scf mode 
    if mode == 'flosic-scf':
        calc = PYFLOSIC(atoms=atoms,charge=charge,spin=spin,xc=xc,basis=basis,mode='flosic-scf',ecp=ecp,max_cycle=max_cycle,conv_tol=conv_tol,grid=grid,ghost=ghost,use_newton=use_newton,verbose=verbose,debug=debug,efield=efield,l_ij=l_ij,ods=ods,fopt=fopt,ham_sic=ham_sic,vsic_every=vsic_every)
	
    # Assign the ase-calculator to the ase-atoms object. 
    atoms.set_calculator(calc)
	
    # Select the wisehd ase-optimizer. 
    if opt == 'FIRE':
        dyn = FIRE(atoms,
                   logfile=label+'.log',
                   trajectory=label+'.traj',
                   dt=0.15,
                   maxmove=maxstep)
                   #force_consistent=force_consistent)
	
    if opt == 'LBFGS':
        dyn = LBFGS(atoms,
                    logfile=label+'.log',
                    trajectory=label+'.traj',
                    use_line_search=False,
                    maxstep=maxstep,
                    memory=10)
                    #force_consistent=force_consistent)
	
    if opt == 'BFGS':
        dyn = BFGS(atoms,
                   logfile=label+'.log',
                   trajectory=label+'.traj',
                   maxstep=maxstep)
	
    if opt == 'LineSearch':
        dyn = BFGSLineSearch(atoms,
                             logfile=label+'.log',
                             trajectory=label+'.traj',
                             maxstep=maxstep)
                             #force_consistent = force_consistent)
	
    if opt == 'CG':
        dyn = SciPyFminCG(atoms,
                          logfile=label+'.log',
                          trajectory=label+'.traj',
                          callback_always=False,
                          alpha=70.0,
                          master=None)
                          #force_consistent=force_consistent)
    if opt == 'GPMin':
        from ase.optimize import GPMin 
        dyn = GPMin(atoms,
                    logfile=label+'.log',
                    trajectory=label+'.traj',
                    update_prior_strategy='average',
                    update_hyperparams=True)
        
    # Run the actuall optimization. 
    dyn.run(fmax=fmax, steps=steps)
    return atoms 

if __name__ == '__main__':
    from ase.io import read 
    import os

    # Path to the xyz file 
    f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/ase_pyflosic_optimizer/LiH.xyz'
    atoms = read(f_xyz)

    charge = 0 
    spin = 0

    # Choose parameters for your journey. 
    mode = ['dft','flosic','flosic-scf'][1]
    xc = ['LDA,PW','PBE,PBE','SCAN,SCAN'][0]
    # bfd := effective core potential/ pseudopotential need the use of ecp. 
    basis = ['sto3g','6-31G','6-311++Gss','cc-pvqz','bfd-vdz','bfd-vtz','bfd-vqz','bfd-v5z'][0]
    ecp = [None,'bfd_pp'][0]
    opt = ['FIRE','BFGS','LBFGS','CG','LineSearch'][0]

    # Do the optimization. 
    os_flosic = flosic_optimize('flosic-os',atoms,charge,spin,xc,basis,ecp,opt='FIRE',maxstep=0.2)	
