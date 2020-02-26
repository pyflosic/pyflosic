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
# CHANGELOG 26.02.2020      grid_level -> grid (JaK)
import time
import copy
import numpy as np
from ase import neighborlist as NL
from flosic_os import xyz_to_nuclei_fod,ase2pyscf
from ase import io, units, Atoms, Atom
from ase.utils import natural_cutoffs
from pyscf import gto, dft
from flosic_scf import FLOSIC
from ase.constraints import FixAtoms
import sys



#        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
#        >>> grids = dft.gen_grid.Grids(mol)
#        >>> grids.level = 4
#        >>> grids.build()



# We want to have
# - a list of all neigbors of each atom
# - a list of FOD's that correspond to each atom
# - a list of all neigbors for each FOD'

# - different dft mesh's for each subgroup of a molecule

class ON(object):
    """Provides functionality to implement O(N) scaling FLOSIC"""
    def __init__(self, mol, fod, grid=9):
        super(ON, self).__init__()
        self.mol = mol
        self.nspin = len(fod)
        self._fod = fod   # list of positions, Angst
        self.grids.level = grid
        self.eps_cutoff = 1.0e-10
        self.is_init = True
        self.add_ghosts = False
        self.nshell = 2
        #print(self.grids.level)
        #sys.exit()
        
    def build(self):
        """build the required data structures (e.g. neigbor lists)"""
        # convert to Bohr as internal unit
        self.onatoms = list()
        for s in range(self.nspin):
            self.onatoms.append(dict())
        
        self.fod = list()
        self.nfod = [0,0]
        for s in range(self.nspin):
            self.fod.append( self._fod[s].copy())
            self.nfod[s] = self._fod[s].shape[0]
            #print np.array_str(self.fod[s],precision=4)
        
        # build atom object (make sure to enter ccors in Angst)
        acoord = self.mol.atom_coords()
        self.atoms = Atoms()
        for na in range(self.mol.natm):
            aa = Atom(symbol=self.mol.atom_symbol(na),
                position=acoord[na]*units.Bohr)
            self.atoms.extend(aa)
        
        cutoffs = natural_cutoffs(self.atoms)
        
        ##print cutoffs
        self.nl = NL.NeighborList(cutoffs,
            self_interaction=False, bothways=True) 
        self.nl.update(self.atoms)
        
        ##print self.nl.get_neighbors(1)
        
        # cut out the H atoms from the distances
        #non_H_ids = list()
        #for atmid in range(self.mol.natm):
        #    if self.mol.atom_pure_symbol(atmid) == 'H': continue
        #    non_H_ids.append(atmid)
        #
        #non_H_pos = np.zeros((len(non_H_ids),3), dtype=np.float64)
        #for atmid,i in zip(non_H_ids,range(len(non_H_ids))):
        #    non_H_pos[i,:] = self.atoms.positions[atmid,:]
        
        
        #self.fod_atm = [[[-1,0.0,'C']*4],[[-2,0.0,'C']*4]]
        #print self.fod_atm[0]
        #self.fod[0].shape[0],self.fod[1].shape[0]
        self.fod_atm = list()
        # find out the nearest atom for each FOD and deciced
        # if its a core FOD or not
        for s in range(self.nspin):
            lfod_atm = list()
            for i in range(self.fod[s].shape[0]):
                fpos = self.fod[s][i]
                # distance to atoms
                dists = np.linalg.norm(fpos - self.atoms.positions,
                    axis=1)
                # do not use H-Atoms as nearest atoms
                ite = 0
                nearest_atom = np.argsort(dists)[ite]
                while self.mol.atom_pure_symbol(nearest_atom) == 'H':
                    ite += 1
                    nearest_atom = np.argsort(dists)[ite]
                
                
                #print self.fod_atm
                #print self.fod_atm[s]
                #self.fod_atm[s][i][0] = nearest_atom
                #self.fod_atm[s][i][1] = dists[nearest_atom]
                #print dists[nearest_atom]
                fodtype = 'V'
                if dists[nearest_atom] < 0.175:
                    fodtype = 'C'
                if dists[nearest_atom] < 1.0e-4:
                    fodtype = 'C1s'
                #self.fod_atm[s][i][2] = fodtype
                #print fodtype
                #sys.exit()
                lfod_atm.append(
                    (nearest_atom,dists[nearest_atom],fodtype))
            self.fod_atm.append(lfod_atm)
        #print len(self.fod_atm)
        
        # make a list of all fods that correspond to a specific atom
        self.atm_fod = list()
        for na in range(self.mol.natm):
            lfodlist = [[],[]]
            for s in range(self.nspin):
                for i in range(self.nfod[s]):
                    if self.fod_atm[s][i][0] == na:
                        #print "fod", i, "belongs to atom", na
                        lfodlist[s].append(i)
            self.atm_fod.append(lfodlist)
        #print self.atm_fod
        
        # prepare the ao slices
        self.fod_ao_slc = list()
        for s in range(self.nspin):
            lfod_slc = list()
            for i in range(self.nfod[s]):
                slc = self.get_nbas_slices(s,i)
                lfod_slc.append(slc)
            self.fod_ao_slc.append(lfod_slc)
        #print len(self.fod_ao_slc[0]), len(self.fod_ao_slc)
        
        # prepare the ongrids
        print("Generating O(N) meshes: nshell={}, grid level={}"\
            .format(self.nshell, self.grids.level))
        self.fod_onmsh = list()
        self.fod_onmol = list()
        for s in range(self.nspin):
            lfod_onmsh = list()
            lfod_onmol = list()
            for i in range(self.nfod[s]):
                omol, ogrid = self.get_grid(s,i,lfod_onmsh)
                lfod_onmsh.append(ogrid)
                lfod_onmol.append(omol)
            self.fod_onmsh.append(lfod_onmsh)
            self.fod_onmol.append(lfod_onmol)
        
        print("Generating O(N) FOD lists ...")
        #print self.fod_onmsh 
        #print self.onatoms
        #sys.exit()
        
        # prepare a list that contains all fods that
        # correspond to a given fod
        self.fod_fod = [[],[]]
        for s in range(self.nspin):
            for i in range(self.nfod[s]):
                # get the atom numbers that corresponds to the fod
                #print self.fod_atm[s][i]
                #print self.atm_fod[s][i]
                atids = self.onatoms[s][i]
                lfod_fod = list()
                for atid in atids:
                    # now find out which of all the other fods
                    # correspond to these atoms
                    for j in range(self.nfod[s]):
                        if i == j: continue
                        atnr = self.fod_atm[s][j][0]
                        if atnr in atids:
                            lfod_fod.append(j)
                # remove duplicates
                lfod_fod = list(set(lfod_fod))
                #print(lfod_fod)
                #print i, len(lfod_fod)
                self.fod_fod[s].append(lfod_fod)
        #print self.fod_fod[0]
        #print self.fod_fod[1]
        #sys.exit()
        
        # finally, build a fod-grp list
        # this list orders all fod that have the same mesh
        self.fodlist2group()
        
    
    def update(self,s,fpos):
        '''update the class according to the given (new) fod positions [in Angst.]'''
        pass
        
    
    def fodlist2group(self):
        """docstring for fodlist2group"""
        print('generate fodgroup list')
        # pre-generate the data structure
        self.fodgrps = list()
        for s in range(self.nspin):
            self.fodgrps.append(list())
        
        #print(self.fodgrps)
        #print(len(self.fodgrps))
        
        #for j in range(self.nfod[0]):
        #    print(self.onatoms[0][j])
        #sys.exit()
        
        # self.onatoms hält zu jedem fod id die entsprechenden
        # atome, die das mesh definieren
        for s in range(self.nspin):
            ttable =  [False for i in range(self.nfod[s])]
            #print(ttable)
            grpcnt = 0
            for j in range(self.nfod[s]):
                if ttable[j]: continue
                self.fodgrps[s].append([j])
                jonatoms = self.onatoms[s][j]
                jonatoms.sort()
                for jj in range(j+1,self.nfod[s]):
                    if ttable[jj]: continue
                    cpmonatoms = self.onatoms[s][jj]
                    cpmonatoms.sort()
                    if jonatoms == cpmonatoms:
                        #print('fod {} and {} have same onatoms {}'.format(j, jj, jonatoms))
                        self.fodgrps[s][grpcnt].append(jj)
                        ttable[jj] = True
                grpcnt += 1
                #print(self.fodgrps[s])
                #sys.exit()
        return
    
    def get_grid(self,s,fodid,lmsh,level=None,verbose=False):
        """Generate a grid object that corresponds to the
            O(N) structure for the given FOD
        """
        print(' -> building Vxc-Grid for FOD {} ...'.format(fodid), flush=True)
        #mol.atom_pure_symbol
        if level == None:
            level = self.grids.level
        onatoms = self.onatoms[s][fodid]
        #print(onatoms)
        
        #print('onatoms:', onatoms)
        #sys.exit()
        fodtype = self.fod_atm[s][fodid][2]
        ongrid = None
        
        # check if we already have meshes available
        # with the same onatoms
        #self.mf.on.fod_onmsh[self.s][fgrp[0]]
        onatoms = list(sorted(onatoms))
        pmshid = -1
        for j in range(fodid):
            #qq = list(sorted(self.onatoms[s][j]))
            #print(j, onatoms, qq)
            #print(lmsh)
            #self.fod_onmsh
            #print(onatoms)
            #print(self.onatoms[s][j].sort())
            if onatoms == list(sorted(self.onatoms[s][j])):
                ## we already have a mesh generated
                ongrid = lmsh[j]
                pmshid = j
        #sys.exit()
        # build string to generate Mole object
        mstr = ''
        for na in onatoms:
            sym = self.mol.atom_pure_symbol(na)
            pos = self.atoms.positions[na]
            #print sym, pos
            mstr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
                sym,pos[0],pos[1],pos[2])
        
        # add ghost atom for valence descriptors
        if (fodtype == 'V') and (self.add_ghosts):
            sym = 'ghost:H'
            pos = self.fod[s][fodid]
            #print sym, pos
            mstr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
                sym,pos[0],pos[1],pos[2])
            
        print('     type {}, na {}:  atoms in msh {}'.format(fodtype, self.fod_atm[s][fodid][0], onatoms))
        
        
        # build a Mole object from subsystem
        #isinstance(o, str):
        #if type(self.mol.basis) == dict:
        #    b = self.mol.basis
        #    b['ghost'] = gto.basis.load('sto3g', 'H')
        #else:
        #    b = {'default':self.mol.basis, 'ghost': gto.basis.load('sto3g', 'H')}
        
        b = self.mol.basis
        
        #print(mstr)
        #print("{0} {1}".format(fodid,b))
        
        #sys.exit()
        
        try:
            onmol =  gto.M(atom=mstr,basis=b)
        except RuntimeError:
            onmol =  gto.M(atom=mstr, basis=b, spin=1)
        onmol.verbose=0
        onmol.max_memory=1000
        #print(onmol.atom)
        # build the meshes
        if ongrid is None:
            _mdft = dft.UKS(onmol)
            _mdft.max_cycle = 0
            _mdft.grids.level = level
            _mdft.kernel()
            ongrid = copy.copy(_mdft.grids)
        else:
            print('     (mesh was already generated for fod {})'.format(pmshid))
        #ongrid = dft.gen_grid.Grids(onmol)
        #ongrid.prune = dft.gen_grid.nwchem_prune
        #ongrid.level = level
        #ongrid.build()
        
        
        
        #print(level)
        #print(ongrid.coords.shape)
        #print onmol
        #sys.exit()
        
        return (onmol,ongrid)
        
        
    # now we want to find out the basis functions that are 
    # really needed for each FO
    # - we check the nearest atom
    # - we find the neigbors to that atom
    # - we find the nbas indices for those atoms (start, stop)
    #   and store them in a list
    def get_nbas_slices(self,s,fodid,verbose=False):
        """docstring for get_nbas_slices"""
        slices = list()
        near_atm = self.fod_atm[s][fodid]
        onatoms = [near_atm[0]]
        if (near_atm[2] == 'C') or (near_atm[2] == 'C1s'):
            if self.nshell == 2:
                nei_atm = self.get_neighbors(near_atm[0], nshell=1)
                for na in nei_atm:
                    onatoms.append(na)

        if near_atm[2] == 'V':
            nei_atm = self.get_neighbors(near_atm[0], nshell=self.nshell)
            for na in nei_atm:
                onatoms.append(na)
        
        # remove duplicates
        onatoms = list(set(onatoms))
        
        if self.nshell == -1:
            onatoms = list(range(self.mol.natm))
        
        # update class onatoms dict
        self.onatoms[s][fodid] = onatoms
        
        all_slices=self.mol.aoslice_by_atom()
        #print all_slices
        onnbas = 0
        for na in onatoms:
            lslice=(all_slices[na,2],all_slices[na,3])
            slices.append(lslice)
            onnbas += lslice[1]-lslice[0]
        
        # print all_slices[-1][-1]
        pc = 100.0 - onnbas / (all_slices[-1,-1] / 100.0)
        
        if verbose:
            print("Sparsity: {0} / {1} ({2:0.2f} %)"
                .format(onnbas,all_slices[-1,-1], pc))
        
        return slices
        
        #sys.exit()
    
    def get_on_dm(self,s,fodid,dm,flo=None):
        """Mask out all basis functions that are not required
        return FLO's and dm were unused ebtries are set to zero
        
        if condense is set to True, return dm and FLO's with only
        the size needed for the O(N) method (e.g. reduced)
        
        """
        if dm.shape[-1] != self.mol.aoslice_by_atom()[-1,-1]:
            print("DM has wrong shape")
            print(self.mol.aoslice_by_atom()[-1,-1])
            sys.exit()
        
        # find out which indices in the density matrix
        # belong to a given fod
        slcs = self.fod_ao_slc[s][fodid]
        slc = list()
        for sl in slcs:
            slc += list(range(sl[0],sl[1]))
        
        dmout = dm.copy()
        #floout = flo.copy()
        
        # zero out the dm
        for i in range(dm.shape[-1]):
            if i in slc: continue
            for j in range(dm.shape[-1]):
                if j in slc: continue
                #print "set dm zero", i, j
                dmout[i,j] = 0.0
        
        nchk = np.sum(dmout)
        if np.isnan(nchk):
            print('Got it')
            sys.exit()

        return dmout
        
    def get_neighbors(self, atmid, nshell=1):
        """
        Return the indices of the neighbor atoms to a given atom id.
        
        Args:
            atmid : int
                id of the atom for which the neigbors shall be returned
        
        Kwargs:
            nshell : int
                number of shells around the given atom that should be 
                considered as neigbors
                1 -> means include only 'nearest neigbors'
                2 -> include nearest and next-nearest neigbors
                3 -> (and larger not yet implemented!)
        
        Returns:
            List of integers which are the indices of the neigbor atoms
            to atmid.
        """
        #if (nshell < 1) or (nshell > 2):
        #    raise ValueError("nshell must be 1 or 2")
        if nshell == -1:
            ret = list(range(self.mol.natm))
            return ret
        
        if nshell == 1:
            ret = list(self.nl.get_neighbors(atmid)[0])
        else:
            base = list(self.nl.get_neighbors(atmid)[0])
            nn = []
            for aid in base:
                nn += list(self.nl.get_neighbors(aid)[0])
            base.extend(nn)
            
            # remove duplicates
            ret = list(dict.fromkeys(base))
        
        return ret
    
    def print_stats(self):
        '''Print out statistics'''
        print('    --- O(N) stats ---')
        print('{:>4} {:>4} : {:>7} {:>7} {:>12}'.format('spin','nfod', 'nmsh', 'nbas', 'onatoms'))
        _nmshs = 0
        _nbas = 0
        for s in range(self.nspin):
            for j in range(self.nfod[s]):
                onmsh = self.fod_onmsh[s][j]
                nonmsh = onmsh.coords.shape[0]
                onat = self.onatoms[s][j]
                nbas = self.fod_onmol[s][j].aoslice_by_atom()[-1,-1]
                _nbas += nbas
                _nmshs += nonmsh
                print('{:4d} {:4d} : {:7d} {:7d}    {}'.format(s,j,nonmsh,nbas, onat))
        print('    ----------------')
        _nbas = float(_nbas)/(np.sum(self.nfod))
        _tnbas = self.mol.aoslice_by_atom()[-1,-1]
        _nmshs = float(_nmshs)/(np.sum(self.nfod))
        #print(_tnbas)
        print('vsic parameters\n')
        print('nmsh: avg {:7d}'.format(int(_nmshs)))
        print('nbas: avg {:7d}'.format(int(_nbas)))
        print('spar: {:11.1f} %'.format(100.0 - _nbas/(_tnbas*0.01)))
        print('')
        print('Number of fod-groups with identical mesh: {}'.format(len(self.fodgrps[0])))
        print('    ----------------')

def C1s_FixPos(on, s):
    '''Returns an ASE constraint that fixes the 1s core descriptor positions
    
    Args:
        on : ON object
            the object that corresponds to the mol/fod structure
        
        s : spin index
    
    Returns:
        ase.constraint object if there are C1s to fix, None otherwise
    '''
    c1sidx = list()
    mol = on.mol
    for fodid in range(on.nfod[s]):
        _na = mol.atom_pure_symbol(on.fod_atm[s][fodid][0])
        _ft = on.fod_atm[s][fodid][2]
        if (_ft == 'C1s'):
            c1sidx.append(fodid)
    
    c1score = None
    if len(c1sidx) > 0:
        c1score = FixAtoms(indices=c1sidx)
        
    return c1score


if __name__ == '__main__':
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
    
    
    b = 'ccpvdz'
    spin = 0
    charge = 0
    
    mol = gto.M(atom=CH3SH, 
                basis=b,
                spin=spin,
                charge=charge)

    grid  = 7
    mol.verbose = 4
    mol.max_memory = 2000
    mol.build()
    xc = 'LDA,PW'
    
    
    # quick dft calculation
    mdft = dft.UKS(mol)
    mdft.xc = xc
    mdft.kernel()
    
    # build O(N) stuff
    myon = ON(mol,[fod1.positions,fod2.positions], grid=grid)
    myon.nshell = 2
    myon.build()
    
    # enable ONMSH
    m = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,grid=grid, init_dm=mdft.make_rdm1())
    m.max_cycle = 40
    m.set_on(myon)
    m.conv_tol = 1e-5
    
    m.preopt = False
    m.preopt_start_cycle=0
    m.preopt_fix1s = True
    m.preopt_fmin = 0.005
    
    m.kernel()
    
    print(m.fod_gradients())
    print(m.get_fforces())
    
