import numpy 
from pyscf import scf,gto 
from pyscf.lo import boys, edmiston, pipek, ibo
from pyscf.tools.cubegen import orbital 
from ase.io import cube
from ase.units import Bohr 
from pyscf.dft import numint
from pyscf import lib

def do_localization(mf,p):
    ''' perform the localization '''
    def pm2(mol,mo_coeff):
        return ibo.ibo(mol, mo_coeff, locmethod='PM', exponent=4)
    loc = {'FB' : boys.Boys, 
           'ER' : edmiston.Edmiston, 
           'PM' : pipek.PipekMezey,
           'PM2' : pm2}
    # Only occupied orbitals 
    pycom_orb = []
    for s in range(p.nspin):
        stop = False 
        k_iter = 0 
        # Do until every eigenvalue is positive 
        while stop == False:
            if k_iter == 0: 
                # 1st iteration starting values 
                mo_occ = mf.mo_coeff[s][:,mf.mo_occ[s]>0]
                myloc = loc[p.pycom_loc](mf.mol,mo_occ)
                myloc.verbose = p.verbose
            if k_iter != 0: 
                # not 2st iteration take the latest values 
                # as starting point 
                mo_occ = orb
                myloc = loc[p.pycom_loc](mf.mol,mo_occ)
                myloc.verbose = p.verbose
                myloc.mo_coeff = orb
            orb = myloc.kernel()
            if p.stability_analysis == False:
                stop = True
                break 
            # key not rand or atomic 
            # u0 =  myloc.get_init_guess(key='nottheotherkeys')
            u0 =  myloc.get_init_guess(key='nottheotherkeys')
            g, h_op, h_diag = myloc.gen_g_hop(u=u0)
            hessian = numpy.diag(h_diag)
            hval, hvec = numpy.linalg.eigh(hessian)
            hval_min = hval.min()
            # eigenvector of the negative eigenvector
            hvec_max = hvec[:,0]
            thres = 10e-8
            if numpy.sign(hval_min) == numpy.sign(-1) and abs(hval_min) > thres:
                stop = False
                # give some noise to the localized coefficients 
                # the rattle value might need to be optimized 
                if p.stability_analysis != False:
                    noise = numpy.random.normal(0, 0.0005, orb.shape)
                    orb = orb + noise
            if numpy.sign(hval_min) == numpy.sign(+1)  or abs(hval_min) <= thres:
                stop = True
            if p.verbose > 4: 
                print('cost function: {}'.format(myloc.cost_function(u=u0)))
                print('min(eigenvalue(hessian) : {}'.format(hval_min))
                print(hvec_max)
            k_iter += 1 
        pycom_orb.append(orb) 
    p.loc = myloc 
    p.pycom_orb = pycom_orb 
    return p

def get_com_fast(mf,p):
    ''' calculates COMS in mo_coeff space ''' 
    ao1 = numint.eval_ao(mf.mol,mf.grids.coords)
    l_com = []
    for s in range(p.nspin):
        s_com = [] 
        occ = len(p.pycom_orb[s][mf.mo_occ[s] == 1])
        for i in range(occ):
            phi = ao1.dot(p.pycom_orb[s][:,i])
            dens = numpy.conjugate(phi)*phi*mf.grids.weights
    	    # COM
            x = numpy.sum(dens*mf.grids.coords[:,0])*Bohr
            y = numpy.sum(dens*mf.grids.coords[:,1])*Bohr
            z = numpy.sum(dens*mf.grids.coords[:,2])*Bohr
            print("{} COM: {} {} {}".format(p.pycom_loc,x,y,z))
            s_com.append([x,y,z])
        l_com.append(s_com) 
    p.l_com = l_com
    return p 

def write_cube(mf,p):
    ''' write the localized orbitals as cube files '''
    l_cube = [] # list of all cube file names 
    for s in range(p.nspin):
        s_cube = []
        occ = len(mf.mo_coeff[s][mf.mo_occ[s] == 1])
        for i in range(occ):
            f_cube = '{}_orb_{}_spin{}.cube'.format(p.pycom_loc,i,s)
            s_cube.append(f_cube) 
            orbital(mf.mol, f_cube, p.pycom_orb[s][:,i], nx=p.nx, ny=p.ny, nz=p.nz)
        l_cube.append(s_cube)
    p.l_cube = l_cube 
    return p 

def calc_com(mf,p):
    ''' calculate COMs for localized orbitals '''
    l_com = []
    d_com = {}
    for s in range(p.nspin):
        s_com = []
        for f in p.l_cube[s]:
            com = get_com(f) 
            s_com.append(com)
            print('{} COM: {} {} {}'.format(f,com[0],com[1],com[2]))
            d_com.update({f : com})
        l_com.append(s_com)
    p.l_com = l_com 
    p.d_com = d_com  
    return p 

def write_guess(mf,p):
    ''' Write PyCOM guess '''
    sym = p.nuclei.get_chemical_symbols()
    pos = p.nuclei.get_positions()
    fod1 = p.l_com[0]
    fod2 = p.l_com[1]
    f_guess = '{}_GUESS_COM.xyz'.format(p.pycom_loc)
    p.f_guess = f_guess 
    f = open(f_guess,'w')
    nelem = len(sym)+len(fod1)+len(fod2)
    f.write('{}\n'.format(nelem))
    f.write("sym_fod1='{}' sym_fod2='{}'\n".format(p.sym_fod1,p.sym_fod2))
    for a in range(len(sym)):
        f.write('{} {} {} {}\n'.format(sym[a],pos[a][0],pos[a][1],pos[a][2]))
    for f1 in range(len(fod1)):
        f.write('{} {} {} {}\n'.format(p.sym_fod1,fod1[f1][0],fod1[f1][1],fod1[f1][2]))
    for f2 in range(len(fod2)):
        f.write('{} {} {} {}\n'.format(p.sym_fod2,fod2[f2][0],fod2[f2][1],fod2[f2][2]))
    f.close()
    return p 

def get_com(f_cube):
    '''' Calculation of COM '''
    orb = cube.read(f_cube)
    # cuba data in [Bohr**3]
    data = cube.read_cube_data(f_cube)
    # cell of cube in [Ang] 
    cell= orb.get_cell()
    shape = numpy.array(data[0]).shape
    spacing_vec = cell/shape[0]/Bohr
    values = data[0]
    idx = 0
    unit = 1/Bohr #**3
    X = []
    Y = []
    Z = []
    V = []
    fv = open(f_cube,'r')
    ll = fv.readlines()
    fv.close()
    vec_tmp = ll[2].split()
    vec_a = -1*float(vec_tmp[1])*Bohr
    vec_b = -1*float(vec_tmp[2])*Bohr
    vec_c = -1*float(vec_tmp[3])*Bohr
    vec = [vec_a,vec_b,vec_c]
    for i in range(0,shape[0]):
        for j in range(0,shape[0]):
            for k in range(0,shape[0]):
                idx+=1
                x,y,z = i*float(spacing_vec[0,0]),j*float(spacing_vec[1,1]),k*float(spacing_vec[2,2])
                # approximate fermi hole h = 2*abs(phi_i)**2 
                # see Bonding in Hypervalent Molecules from Analysis of Fermi Holes Eq(11) 
                x,y,z,v = x/unit ,y/unit ,z/unit , 2.*numpy.abs(values[i,j,k])**2. 
                X.append(x)
                Y.append(y)
                Z.append(z)
                V.append(v)
    X = numpy.array(X)
    Y = numpy.array(Y)
    Z = numpy.array(Z)
    V = numpy.array(V)
    x = sum(X*V)
    y = sum(Y*V)
    z = sum(Z*V)
    # Shifting to the origin of the cube file. 
    com = (numpy.array([x/sum(V),y/sum(V),z/sum(V)])-vec).tolist()
    return com

class pycom():
    ''' PyCOM - Python center of mass '''
    def __init__(self,mf,p):
        ''' Intitialization '''
        self.mf = mf 
        self.p = p 
        #self.p.stability_analysis = 'simple'
        # resolution of orbs on grid 
        self.p.nx = 80
        self.p.ny = 80 
        self.p.nz = 80
        self.p.nspin = numpy.array(mf.mo_occ).ndim
	
    def kernel(self):
        ''' Calculate FOD Guess '''
        self.p = do_localization(mf=self.mf,p=self.p)
        if self.p.write_cubes == False:
            # calculate COMS only 
            get_com_fast(mf=self.mf,p=self.p)
            self.p = write_guess(mf=self.mf,p=self.p)
            self.f_guess = self.p.f_guess

        if self.p.write_cubes == True:
            # writes CUBES and calculate COMS
            self.p = write_cube(mf=self.mf,p=self.p)
            self.p = calc_com(mf=self.mf,p=self.p)
            # dict: CUBE and COM relation 
            self.d_com = self.p.d_com 
            self.p = write_guess(mf=self.mf,p=self.p)
            self.f_guess = self.p.f_guess 

def ase2pyscf(ase_atoms):
    ''' convert coordinates from ase to pyscf '''
    return [[atom.symbol, atom.position] for atom in ase_atoms]

class parameters(): 
    ''' parameters class '''
    def __init__(self):
        self.verbose = 4 
        self.sym_fod1 = 'X'         
        self.sym_fod2 = 'He'

def pycom_guess(ase_nuclei,charge,spin,basis,xc,method='FB',ecp=None,newton=False,grid=3,BS=None,calc='UKS',symmetry=False,verbose=4,write_cubes=False,stability_analysis=True):
    ''' generate PyCOM guess 
    
    Args:
            ase_nuclei:             ase_atoms object 

            charge:                 float
                                    charge of the system 

            spin:                   integer
                                    2S of the system 

            xc:                     string 
                                    exchange-correlation functional 
            
            basis:                  string 
                                    basis set to be used 

            method:                 string, e.g., FB (prefered), ER , PM 
                                    localization method to be used 

            ecp:                    string, 
                                    for effective core potentials 

            newton:                 bool, True or False 
                                    use 2nd order scf 

            grid:                   integer, 0 - 9
                                    corresponds to pyscf grids.level 

            BS:                     bool, None or True 
                                    generate a broken symmetry ground state 

            calc:                   string, UKS, UHF, RHF 
                                    computational method 

            symmetry:               bool, True or False 
                                    use symmetry for pyscf mol 

            verbose:                integer, 4 (default) 
                                    corresponds to pyscf verbosity level 


            write_cubes:            bool, True or False 
                                    True is the original slow PyCOM method 
                                    False is the new fast PyCOM method 
            stability analysis :    bool, True or False 
                                    True performs a simple stability analysis, 
                                    tested for FB pc0

    Returns:
            a guess for Fermi-orbital descriptors as xyz file 
    '''
    method = method.upper()
    calc = calc.upper()
    mol = gto.M(atom=ase2pyscf(ase_nuclei),basis=basis,ecp=ecp,spin=spin,charge=charge,symmetry=symmetry)
    mol.verbose = verbose 
    if calc == 'UKS':
        mf = scf.UKS(mol)
    elif calc == 'UHF':
        mf = scf.UHF(mol)
    elif calc == 'RHF': 
        mf = scf.RHF(mol)  
    mf.grids.level = grid
    mf.max_cycle = 3000
    mf.xc = xc
    # Broken symmetry 
    if BS != None:
        mf.kernel()
        idx_flip = mol.search_ao_label(BS)
        dma, dmb = mf.make_rdm1()
        dma_flip = dma[idx_flip.reshape(-1,1),idx_flip].copy()
        dmb_flip= dmb[idx_flip.reshape(-1,1),idx_flip].copy()
        dma[idx_flip.reshape(-1,1),idx_flip] = dmb_flip
        dmb[idx_flip.reshape(-1,1),idx_flip] = dma_flip
        dm = [dma, dmb]
        mol = gto.M(atom=ase2pyscf(ase_nuclei),basis=basis,ecp=ecp,spin=0,charge=charge,symmetry=symmetry)
        mol.verbose = verbose
        mf = scf.UKS(mol)
        mf.grids.level = grid
        mf.max_cycle = 3000
        mf.xc = xc
    if newton == True: 
        mf = mf.as_scanner()
        mf = mf.newton()
    if BS == None:
       	mf.kernel()
    else:
        mf.run(dm)
    if calc == 'RHF':
        mf = scf.addons.convert_to_uhf(mf)
    # generate parameters object 
    p = parameters() 
    p.nuclei = ase_nuclei
    p.pycom_loc = method 
    # use simple stability analysis
    p.stability_analysis = stability_analysis 
    p.write_cubes = write_cubes
    pc = pycom(mf=mf,p=p)
    pc.kernel()



if __name__ == '__main__':
    from ase.io import read
    import os

    # path to the xyz file 
    f_xyz = os.path.dirname(os.path.realpath(__file__))+'/../examples/pycom/O3.xyz'

    ase_nuclei = read(f_xyz)
    charge = 0
    spin = 0
    basis = 'cc-pvdz' 
    xc = 'LDA,PW'
    
    pycom_guess(ase_nuclei,charge,spin,basis,xc,method='FB',newton=True,symmetry=False,write_cubes=False,stability_analysis=True)

