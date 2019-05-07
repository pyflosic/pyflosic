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
"""Program to generate an initial guess of FODs for FLOSIC calculations by density fitting of Fermi-orbitals.

The script calculates Foster-Boys (FB) orbitals upon an ordinary
DFT calculation. After doing so, Fermi-orbitals (FO) are built by fitting
the reference  position a_i to make the difference ||rho_FB - rho_FO||^2
minimal.
Contrary to other approaches, this method works suitable well with
3d-transition metals (in case PBE-DFT gives a reasonable solution).

Examples:
    Typically you run the script directly from the commandline (assuming
    you have 4 cpus available):

        $ export OMP_NUM_THREADS=4
        $ python init_guess_by_df.py mymol.xyz 0 1 ccpvdz

    Runs the code on the molecule defined in 'mymol.xyz' with total charge
    of 0 for the molecule and spin 1 using the 'ccpvdz' basis set.
    (You can use any pyscf-known basis set)

Author:
    Torsten Hahn <torstenhahn@fastmail.fm>
"""
import sys
import time
import copy
import numpy as np
from pyscf import gto
from pyscf.dft import numint
from pyscf import lo
from ase import io, units, Atoms, Atom
from ase import neighborlist as NL
from ase.utils import natural_cutoffs
from ase.symbols import atomic_numbers as ANR



def sph2cart(rtheaphi):
    """rtheaphi first index is coortinate id, second
        is r, theta, phi
    """
    r = rtheaphi[:,0]
    theta = rtheaphi[:,1]
    phi = rtheaphi[:,2]
    cart = np.zeros_like(rtheaphi)
    cart[:,0] = rtheaphi[:,0] * np.sin(theta) * np.cos(phi)
    cart[:,1] = rtheaphi[:,0] * np.sin(theta) * np.sin(phi)
    cart[:,2] = rtheaphi[:,0] * np.cos(theta)

    return cart

def cart2sph(xyz):
    """docstring for cart2sph"""
    rtheaphi = np.zeros_like(xyz)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    rtheaphi[:,0] = np.linalg.norm(xyz, axis=1)
    rtheaphi[:,1] = np.arccos(z / rtheaphi[:,0])
    rtheaphi[:,2] = np.arctan2(y,x)


    return rtheaphi

def fo(mf, fod, s=0):
    """docstring for fo"""
    ksocc = np.where(mf.mo_occ[s] > 1e-6)[0][-1] + 1
    #print("ksocc: {0}".format(ksocc))
    #print np.where(self.mf.mo_occ[self.s] > 1e-6)
    mol = mf.mol
    ao = numint.eval_ao(mol,[fod])

    # first index is nfod, second is orbital
    psi = ao.dot(mf.mo_coeff[s][:,:ksocc])
    #print psi.shape
    #sys.exit()

    # get total spin density at point self.fpos
    sd = np.sqrt(np.sum(psi**2, axis=1))
    #print sd.shape
    #print sd
    #sys.exit()
    # get value of each ks-orbital at point fpos
    _ks = mf.mo_coeff[s][:,0:ksocc]
    #print _ks.shape
    #sys.exit()
    #print np.array_str(_ks, precision=4, max_line_width=120)

    _R = np.zeros((ksocc), dtype=_ks.dtype)
    _R = psi[0,:]/sd


    #_R = np.reshape(psi/sd, (ksocc))
    fo = np.matmul(_R,_ks.transpose())

    #print fo

    return fo


def lorb2fod(mf, lo_coeff, s=0, grid_level=7):
    """
    lo_coeff[:] localized orbital
    """
    from scipy import optimize
    # get estimate for FOD'position
    # by using the COM of the orbital density
    mol = mf.mol
    ao1 = numint.eval_ao(mol,mf.grids.coords)
    phi = ao1.dot(lo_coeff)
    #print(phi.shape)

    #print('lorb2fod: s={}'.format(s))
    #print('lorb2fod: lo_coeff={}'.format(lo_coeff.sum()))


    #print(np.sum(phi**2*mf.grids.weights))
    dens = np.conjugate(phi)*phi*mf.grids.weights
    # COM
    x = np.sum(dens*mf.grids.coords[:,0])
    y = np.sum(dens*mf.grids.coords[:,1])
    z = np.sum(dens*mf.grids.coords[:,2])
    #print x


    print("  -> COM: {0:7.5f} {1:7.5f} {2:7.5f}".format(x*units.Bohr,y*units.Bohr,z*units.Bohr))
    ig_fod = np.array([x,y,z])

    #if s==1:
    #    sys.exit()

    ## build a new, smaller mesh for the density fitting
    # find nearest atom
    dists = np.linalg.norm((mol.atom_coords()-ig_fod),axis=1)
    didx = np.argsort(dists)
    #print dists
    #print didx
    nidx = -1
    for i in range(mol.natm):
        if mol.atom_pure_symbol(i) == 'H': continue
        nidx = didx[i]
        break

    if nidx == -1:
        print("ERROR")
        sys.exit()


    #print nidx, mol.atom_pure_symbol(nidx)

    # build atom object (make sure to enter ccors in Angst)
    acoord = mol.atom_coords()
    atoms = Atoms()
    for na in range(mol.natm):
        aa = Atom(symbol=mol.atom_symbol(na),
            position=acoord[na]*units.Bohr)
        atoms.extend(aa)

    cutoffs = natural_cutoffs(atoms)

    ##print cutoffs
    nl = NL.NeighborList(cutoffs,
        self_interaction=False, bothways=True)
    nl.update(atoms)

    # generate a per-atom grid (include neigbours)
    neiatm =  nl.get_neighbors(nidx)[0]
    mstr = ''
    for na in neiatm:
        sym = mol.atom_pure_symbol(na)
        pos = mol.atom_coord(na)*units.Bohr # in Angst
        #print sym, pos
        mstr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
            sym,pos[0],pos[1],pos[2])

    # also add the nearest Atom to the grid algorithm
    sym = mol.atom_pure_symbol(nidx)
    pos = mol.atom_coord(nidx)*units.Bohr # in Angst
    #print sym, pos
    mstr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
        sym,pos[0],pos[1],pos[2])

    #print ">>>"
    #print mstr

    # build a Mole object from subsystem
    b = mol.basis
    try:
        onmol =  gto.M(atom=mstr,basis=b, verbose=0)
    except RuntimeError:
        onmol =  gto.M(atom=mstr, basis=b, spin=1, verbose=0)

    _mdft = dft.UKS(onmol)
    _mdft.max_cycle = 0
    _mdft.grids.level = grid_level
    _mdft.kernel()
    ongrid = copy.copy(_mdft.grids)


    #print("Original grid size: {0}".format(mf.grids.coords.shape[0]))
    #print("  building FO, grid size: {0}"
    #    .format(ongrid.coords.shape[0]))

    ## re-calculate lo density on O(N) grid
    ao1 = numint.eval_ao(mol,ongrid.coords)
    phi = ao1.dot(lo_coeff)
    dens = np.conjugate(phi)*phi*ongrid.weights



    #sys.exit()

    # now build the corresponding fermi orbital
    #lfo = fo(mf, ig_fod, s)

    def densdiff(x0):
        #print ig_fod
        #print(s)
        _x = np.reshape(x0,(-1,3))
        lfo = fo(mf, _x, s)

        mol = mf.mol
        ao1 = numint.eval_ao(mol,ongrid.coords)
        _fo = ao1.dot(lfo)

        dens_fo = np.conjugate(_fo)*_fo*ongrid.weights

        ##print dens_fo.shape
        ##print dens.shape

        diff = np.linalg.norm(dens_fo - dens)


        return diff

    options = { 'disp' : False,
                'eps': 1e-05,
                'gtol': 1e-05,
                'maxiter': 299,
    }

    db = 1.5
    if np.linalg.norm(mol.atom_coord(nidx)-ig_fod) < 0.5:
        db = 0.75

    bounds = [(x-db,x+db),(y-db,y+db),(z-db,z+db)]


    res = optimize.minimize(densdiff, ig_fod.flatten(), method='L-BFGS-B',
        options=options, bounds=bounds)

    #print ">> done <<"
    #print res.x
    #print ig_fod
    #print ">> initial FOD moved by: {0:0.4f} [B]".format(np.linalg.norm(res.x-ig_fod))
    #print ">> density fit quality : {0:0.4f}".format(res.fun)
    print("  -> a_i: {0:7.5f} {1:7.5f} {2:7.5f}"\
      .format(res.x[0]*units.Bohr,res.x[1]*units.Bohr,res.x[2]*units.Bohr))

    return res.x


def genfofix(fname, FOFIX=False):
    """Take a xyz file and set the 1s FOD's to the core positions."""
    fn = fname
    mol_atoms = io.read(fn)
    frmorb_atoms = Atoms()
    # extract frmorbs from mol-file
    csym = mol_atoms.get_chemical_symbols()
    for i in range(len(csym)):
        if (csym[i] == 'X') or (csym[i] == 'He'):
            frmorb_atoms.extend(mol_atoms[i])

    # remove fo's from mol
    mol_atoms_sav = mol_atoms.copy()
    #print(mol_atoms_sav)
    mol_atoms = Atoms()
    for i in range(len(csym)):
        #print i, csym[i]
        if (csym[i] != 'X') and (csym[i] != 'He'):
            mol_atoms.extend(mol_atoms_sav[i])

    #
    fofix = None
    if FOFIX:
        fofix = open("FOFIX", 'w')

    #print(mol_atoms)
    #sys.exit()
    cutoff = 0.125

    for j in range(frmorb_atoms.positions.shape[0]):
        fop = frmorb_atoms.positions[j]
        for i in range(mol_atoms.positions.shape[0]):
            cs = mol_atoms.get_chemical_symbols()[i]
            if cs == 'H':
                continue
            mp = mol_atoms.positions[i]

            d = np.linalg.norm(fop - mp)
            einss = False

            _cutoff = cutoff
            if (ANR[cs] > 10):
                _cutoff = 0.1
                #print "here", cutoff
            if (ANR[cs] > 18):
                _cutoff = 0.05
                #print "here", cutoff
            if (ANR[cs] > 30):
                _cutoff = 0.025
                #print "here", cutoff

                #sys.exit()
            #print(j,i, cs, _cutoff)
            if d <= _cutoff:
                einss = True
                # set FRMORB position to core position
                frmorb_atoms.positions[j,:] = mol_atoms.positions[i,:]
                ostr = "{0}   FixPos    -1\n".format(j)
                if fofix is not None:
                    fofix.write(ostr)
                #print("{0} {1} : {2:7.5f}".format(i,j,d))
    if FOFIX:
        fofix.close()

    io.write(fn, mol_atoms.extend(frmorb_atoms), plain=True)



if __name__ == '__main__':
    from os.path import expanduser, join
    import os
    from pyscf import dft
    from ase import io
    import socket


    ifile = "JMOL.xyz"
    b = 'ccpvdz'
    #b = 'sto6g'
    spin = 0
    charge = 0
    _verbose = 3

    #print(sys.argv)
    _xc = 'PBE,PBE'  # standard functional

    try:
        ifile  = sys.argv[1]
        charge = int(sys.argv[2])
        spin   = int(sys.argv[3])
        b      = sys.argv[4]
        _xc    = sys.argv[5]
        _verbose = int(sys.argv[6])
    except:
        pass

    s = io.read(ifile)

    print("------ FLOSIC initial guess tool v1.0 ------")
    print("(c) by Torsten Hahn <torstenhahn@fastmail.fm>\n")
    print("  [No warranty, sanity check your results !]\n")

    print('{:>12s} : {}'.format('ifile', ifile))
    print('{:>12s} : {}'.format('charge', charge))
    print('{:>12s} : {}'.format('spin', spin))
    print('{:>12s} : {}'.format('basis', b))
    print('{:>12s} : {}'.format('functional', _xc))

    head, tail = os.path.split(ifile)
    #print head, tail
    ofname = tail.split('.')[0] + '_guess.' + tail.split('.')[1]

    #print ofname

    #sys.exit()

    #pyscf_atoms,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(s)

    csymb = s.get_chemical_symbols()
    cpos = s.positions
    astr = ''
    for symb,i in zip(csymb,range(len(csymb))):
        #print sym, pos
        pos = cpos[i]
        astr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
            symb,pos[0],pos[1],pos[2])


    #print nuclei


    mol = gto.M(atom=astr,
                basis=b,
                spin=spin,
                charge=charge)

    grid_level  = 6
    mol.verbose = _verbose
    fqdn = socket.getfqdn()
    # give the code some memory to breath
    mol.max_memory = 12000

    xc = _xc

    m = dft.UKS(mol)
    #m.diis_start_cycle=2
    m.diis_space = 7
    m.init_guess = 'chkfile'
    m.chkfile = ifile + '.chk'
    m.small_rho_cutoff = 1e-9
    m.grids.level = grid_level
    m.xc=xc

    print('\nPerforming DFT calculation ...\n')

    #m.level_shift = 0.2
    m.conv_tol = 2.5e-10
    m=m.newton()
    m.kernel()
    #m.analyze()

    #m.level_shift = 0
    #m.conv_tol = 1e-9
    #m.kernel(m.make_rdm1())
    #m.analyze()

    #print('Entering broken symmetry')
    ##
    ### broken symmetry, serach for the Cu atoms
    ###
    ### Flip the local spin of the first Fe atom ('0 Fe' in ao_labels)
    ###
    #idx_fe1 = mol.search_ao_label('0 N')
    #dma, dmb = m.make_rdm1()
    #dma_fe1 = dma[idx_fe1.reshape(-1,1),idx_fe1].copy()
    #dmb_fe1 = dmb[idx_fe1.reshape(-1,1),idx_fe1].copy()
    #dma[idx_fe1.reshape(-1,1),idx_fe1] = dmb_fe1
    #dmb[idx_fe1.reshape(-1,1),idx_fe1] = dma_fe1
    #dm = [dma, dmb]
    ##
    ###
    ### Change the spin and run the second pass for low-spin solution
    ###
    #mol.spin = 0
    #m = dft.UKS(mol)
    #m.xc = 'PBE,PBE'
    ## Apply large level shift to avoid big oscillation at the beginning of SCF
    ## iteration.  This is not a must for BS-DFT.  Depending on the system, this
    ## step can be omitted.
    #m.level_shift = 0.5
    #m.conv_tol = 1e-4
    #m.kernel(dm)
    #
    ##
    ## Remove the level shift and converge the low-spin state in the final pass
    ##
    #m.level_shift = 0
    #m.conv_tol = 1e-9
    #m.kernel(m.make_rdm1())

    mol.verbose = _verbose

    print("\nDFT done, starting LO generation ...\n")

    nspin = len(mol.nelec)
    #print nspin
    #sys.exit()

    fodout = s.copy()

    for spin in range(nspin):
        # find out how many orbitals to use
        # (total electrons minus 1s core electrons)
        te = mol.nelec[spin]
        ve = te
        for i in range(mol.natm):
            if mol.atom_pure_symbol(i) == 'H': continue
            ve -= 1

        ne_1s = int(te-ve)
        ve = int(ve)
        te = int(te)

        _mys = 'UP'
        if spin == 1:
            _mys = 'DN'

        print("Find Fermi-Orbitals for spin: {0} / {1} electrons".format(_mys, te))
        #print("  total e: {0} (spin {1})".format(te,spin))
        #print("1s core: {0}".format(ne_1s))
        #print("valence: {0}".format(ve))

        # define which orbitals are use for initial boys
        pz_idx = np.arange(0,te)
        nfods = len(pz_idx)
        initial_fods = np.zeros((nfods,3), dtype=np.float64)
        #print(pz_idx, len(pz_idx))

        #sys.exit()

        # build the localized orbitals
        loc_orb = lo.Boys(mol, m.mo_coeff[spin][:,pz_idx]).kernel()

        #print loc_orb.shape
        osym = 'X'
        if spin == 1:
            osym = 'He'
        for j in range(pz_idx.shape[0]):
            sstr = 'UP'
            if spin == 1:
                sstr = 'DN'
            print("  density fit for spin {0} orb #{1} ...".format(sstr,j+1), flush=True)
            #print("find initial fod: {0}".format(j))
            initial_fods[j,:] = lorb2fod(m,loc_orb[:,j], s=spin, grid_level=grid_level-1)
            fodout.extend(Atom(osym, position=initial_fods[j,:]*units.Bohr))

    print("\nWriting output file: {0}".format(ofname))
    io.write(ofname, fodout, plain=True)

    # call the code to move the 1s to the core positions
    print("Shifting 1s electrons to core positions ...")
    genfofix(ofname)
    print("\nDone, pls. check your results for sanity !!")
