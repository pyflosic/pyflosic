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
import time
import os
import sys
from copy import copy, deepcopy
import numpy as np
import pyscf
from pyscf import gto
from ase import io, units, Atoms, Atom
from pyscf.dft import numint
from pyscf import dft
from pyscf import lo
from ase import neighborlist as NL
from ase.utils import natural_cutoffs
from pyscf.lib import logger
from pyscf import lib
import multiprocessing as mp

# this is for basic mpi support to
# speed up veff evaluation
try:
    from mpi4py import MPI
    import mpi4py as mpi
except ImportError:
    mpi = None
    pass

# speed up evaluation of the fod gradients
USE_NUMBA = True
try:
    from numba import jit, prange
except:
    USE_NUMBA = False
    print(">> WARNING: no NUMBA package found, \
        FOD gradient evaluation will be *VERY* slow ! <<")

@jit(nopython=True, parallel=True, cache=True)
def D3_km_outer_loop(r, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt, TdST):
    #print('do_D3km_inner_loop', m)
    #time.sleep(0.1)
    reta = np.zeros((nfod,nfod,nfod), dtype=np.float64)
    out1 = np.zeros((nfod,nfod,nfod), dtype=np.float64)
    out2 = np.zeros((nfod,nfod,nfod), dtype=np.float64)
    aab  = np.zeros((nfod,nfod,nfod), dtype=np.float64)
    Va = np.zeros((nfod,nfod), dtype=np.float64)
    for m in prange(nfod):
        for k in range(nfod):
            for l in range(nfod):
                # skip in case <phi_i|Vsic_i|phi_j> is pretty small
                if ttable[l,k]: continue
                Va[m,:] = np.ravel(T_alpha[:nfod,l]*Q_alpha_sqrt[:nfod])
                #print(Va.shape, nfod)
                #sys.exit()
                np.outer(T_alpha[:,k].ravel(), Va[m], out=out1[m])
                np.outer(Va[m], T_alpha[:,k].ravel(), out=out2[m])
                out1[m] += out2[m]
                aab[m,:,:] = 1.0/(np.outer(Q_alpha, Q_alpha_sqrt) + \
                            np.outer(Q_alpha_sqrt, Q_alpha))
                #aab[:,:] *= -0.5*TdST[:,:,m,r] * out1
                reta[m,l,k] = np.sum(-0.5*TdST[:,:,m,r] * out1[m] * aab[m])
    return reta


def D3_km_outer_loop_serial(r, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt, TdST):
    #print('do_D3km_inner_loop', m)
    #time.sleep(0.1)
    reta = np.zeros((nfod,nfod,nfod), dtype=np.float64)
    out1 = np.zeros((nfod,nfod), dtype=np.float64)
    out2 = np.zeros((nfod,nfod), dtype=np.float64)
    aab  = np.zeros((nfod,nfod), dtype=np.float64)
    for m in range(nfod):
        for k in range(nfod):
            for l in range(nfod):
                # skip in case <phi_i|Vsic_i|phi_j> is pretty small
                if ttable[l,k]: continue
                Va = np.ravel(T_alpha[:nfod,l]*Q_alpha_sqrt[:nfod])
                np.outer(T_alpha[:,k].ravel(), Va, out=out1)
                np.outer(Va, T_alpha[:,k].ravel(), out=out2)
                out1 += out2
                aab[:,:] = 1.0/(np.outer(Q_alpha, Q_alpha_sqrt) + \
                            np.outer(Q_alpha_sqrt, Q_alpha))
                #aab[:,:] *= -0.5*TdST[:,:,m,r] * out1
                reta[m,l,k] = np.sum(-0.5*TdST[:,:,m,r] * out1 * aab)
    return reta


def do_D3km_inner_loop(m, r, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt, TdST):
    #print('do_D3km_inner_loop', m)
    #time.sleep(0.1)
    reta = np.zeros((nfod,nfod), dtype=np.float64)
    out1 = np.zeros((nfod,nfod), dtype=np.float64)
    out2 = np.zeros((nfod,nfod), dtype=np.float64)
    aab  = np.zeros((nfod,nfod), dtype=np.float64)
    for k in range(nfod):
        for l in range(nfod):
            # skip in case <phi_i|Vsic_i|phi_j> is pretty small
            if ttable[l,k]: continue
            Va = np.ravel(T_alpha[:nfod,l]*Q_alpha_sqrt[:nfod])
            np.outer(T_alpha[:,k].ravel(), Va, out=out1)
            np.outer(Va, T_alpha[:,k].ravel(), out=out2)
            out1 += out2
            aab[:,:] = 1.0/(np.outer(Q_alpha, Q_alpha_sqrt) + \
                        np.outer(Q_alpha_sqrt, Q_alpha))
            #aab[:,:] *= -0.5*TdST[:,:,m,r] * out1
            reta[l,k] = np.sum(-0.5*TdST[:,:,m,r] * out1 * aab)


    # the original loop
    # take for debugging reasons
    #for k in range(nfod):
    #    for l in range(nfod):
    #        for a in range(nfod):
    #            tmp1 = T_alpha[:,k]*T_alpha[a,l]*Q_alpha_sqrt[a]
    #            tmp2 = T_alpha[a,k]*T_alpha[:,l]*Q_alpha_sqrt[:]
    #            tmp3 = ( Q_alpha_sqrt[a] + Q_alpha_sqrt[:] ) * \
    #                    np.sqrt(Q_alpha[a]*Q_alpha[:])
    #            reta[l,k] -= np.sum(0.5*TdST[:,a,m,r]* ( (tmp1+tmp2) / tmp3)
#)
                #for b in range(nfod):
                #    tmp1 = T_alpha[b,k]*T_alpha[a,l]*np.sqrt(Q_alpha[a])
                #    tmp2 = T_alpha[a,k]*T_alpha[b,l]*np.sqrt(Q_alpha[b])
                #    reta[l,k]=reta[l,k]-0.5*TdST[b,a,m,r]* ( (tmp1+tmp2) / tmp3)

    return (m,reta)


#                for k in range(nfod):
#                    for l in range(nfod):
#                        # skip in case <phi_i|Vsic_i|phi_j> is pretty small
#                        if ttable[l,k]: continue
#                        Va = np.ravel(T_alpha[:nfod,l]*Q_alpha_sqrt[:nfod])
#                        np.outer(T_alpha[:,k].ravel(), Va, out=out1)
#                        np.outer(Va, T_alpha[:,k].ravel(), out=out2)
#                        out1 += out2
#                        aab[:,:] = 1.0/(np.outer(Q_alpha, Q_alpha_sqrt) + \
#                                    np.outer(Q_alpha_sqrt, Q_alpha))
#                        #aab[:,:] *= -0.5*TdST[:,:,m,r] * out1
#                        D3_km[l,k,m,r] = np.sum(-0.5*TdST[:,:,m,r] * out1 * aab)
#

def mpi_start():
    rank = MPI.COMM_WORLD.Get_rank()
    wsize = MPI.COMM_WORLD.Get_size()
    if wsize == 1: return
    if rank > 0:
        print('>>> starting mpi_worker on rank {}'.format(rank), flush=True)
        mpi_worker()
        # once the worker returns, exit
        sys.exit()
    return


def mpi_stop():
    rank = MPI.COMM_WORLD.Get_rank()
    wsize = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
    if wsize == 1: return
    if rank == 0:
        # shut down mpi workers on all nodes
        print('>>> sending finalize')
        for inode in range(1,wsize):
            comm.send('finalize', dest=inode, tag=11)
    return

def mpi_worker():
    '''Function to be executed on the nodes except root to carry
    out some heavy work.
    At the moment, this is the calculation of veff.

    Args:

        mflosic : object
            FLOSIC object associated with the current calculation
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    wsize = comm.Get_size()

    #print('mpi_worker: my rank is {}, mpi size is {}'\
    #    .format(rank, wsize), flush=True)

    mf = None
    mol = None
    nbas = None
    #nbas = mf.mo_coeff[0].shape[0]
    #print('nbas', nbas)
    #sys.exit()

    avail = None
    while True:

        if avail is None:
            req = comm.irecv(source=0, tag=11)
        avail = req.test()
        #print('>>>> avail: ', avail)

        if not avail[0]:
            time.sleep(0.01)
            #print('... waiting ....', flush=True)
            continue
        else:
            task = avail[1]
            #print('>>> mpitask:', task)
            #sys.exit()

        if task == 'finalize':
            print('>>> Shutting down MPI')
            # may be we have to call finalize ?
            break
        elif task == 'init':
            info = comm.recv(source=0, tag=12)
            #print(info)
            mol = gto.M(atom=info['atom'],
                basis=info['basis'],
                spin=info['spin'],
                charge=info['charge'],
                verbose=0,
                max_memory=info['max_memory']
            )
            mf = dft.UKS(mol)
            mf.grids.level = info['grid_level']
            mf.xc          = info['xc']
            mf.max_cycle   = 0
            mf.kernel()
            nbas = mf.mo_coeff[0].shape[0]
            #print(">>> init done", flush=True)
            comm.Barrier()
            avail = None
            continue
            #sys.exit()
            #nbas = mf.mo_coeff[0].shape[0]
        elif task == 'vsic':
            idata = np.empty(2, dtype='i')
            comm.Bcast(idata, root=0)
            #print('Got idata: ', idata)
            nfod = idata[0]
            nmsh = idata[1]

            if nfod == -1: break

            sidx, eidx, csize = get_mpichunks(idata[0],0,comm=comm)
            #print(">>> mpi_worker: sidx, eidx, csize", sidx, eidx, csize, flush=True)

            # reserve memory for density matrices
            _dmtmp = np.zeros((2,nfod, nbas, nbas), dtype=np.float64)
            comm.Bcast(_dmtmp, root=0)
            _weights = np.zeros(nmsh, dtype='d')
            _coords = np.zeros((nmsh,3), dtype='d')
            comm.Bcast(_weights, root=0)
            comm.Bcast(_coords, root=0)

            mf.grids.coords = _coords

            # prepare the dm for use
            _dm = slice_dm4mpi(_dmtmp, sidx, eidx)
            # prepare the mesh
            mf.grids.coords = _coords.copy()
            mf.grids.weights = _weights.copy()

            #print(">>> slave: ", np.asarray(_dm).shape)

            # call the solver
            _veff = mf.get_veff(mol=mf.mol, dm=_dm)
            #print(">>> slave _veff: ", hasattr(_veff,'__dict__'), flush=True)


            # gather the results on master
            _veff_gath = comm.gather(_veff, root=0)

            avail = None
        else:
            print('>>> mpi: Unknown task:', task)
            sys.exit(-1)

        del(task)
        #print("veff", _veff.shape)

        # transfer the

        # exit for the moment
        #sys.exit()

    print('>>> mpi_worker: terminating ...')

    return

def slice_dm4mpi(dm, sidx, eidx):
    '''Slice a given density matrix to contain only the
    information needed to process sidx -> eidx entries

    dm[2,nfod,nbas,nbas]
    '''
    dmtmp = np.array(dm)

    _dma = dmtmp[0,sidx:eidx,:,:]
    _dmb = dmtmp[1,sidx:eidx,:,:]

    return [_dma, _dmb]

def get_mpichunks(psize,offset, comm=0):
    """
    Returns start- and end- indices as well as the
    respective chunk size usable for looping
    for a problem of size PSIZE based on the current
    MPI configuration on EACH NODE
    (!! needs to be called on each node !!)

    psize : size of the problem that shall become mpi-parallel
    offset : a possible offset for the loop start on ROOT
             (set to 0 if your loop starts counting at 0)

    return:
       SIDX : start index for looping on this node
       EIDX : end index for looping on this node
       CSIZE : the chunk size (basically EIDX-SIDX)
    """
    sidx = offset
    eidx = psize-1
    csize = eidx-offset+1

    try:
        irank = comm.Get_rank()
        nproc = comm.Get_size()
    except:
        irank=0
        nproc=1


    # just close your eyes, the works ! ;-)
    sidx = irank*psize/(nproc) #+1
    #eidx = (irank+1)*psize/(nproc) -1
    eidx = (irank+1)*psize/(nproc) # to fit in pythons ranmge function

    if ((irank == nproc) and ((psize % nproc)!= 0)):
      eidx = psize
    if ((irank == 0)and (offset != 0)):
        sidx = sidx+offset
    csize = eidx - sidx  + 1

    # check for more cpu's than tasks
    if (nproc > psize):
        csize = 1
        sidx = irank
        eidx = irank
        # do nothing on higher ranks
        if (irank >= psize):
            csize = 0
            sidx = 1
            eidx = 0

    return (int(sidx),int(eidx),int(csize))


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




class FO(object):
    """docstring for PreOpt"""
    def __init__(self, s, fpos, mf, mol, grid):
        super(FO, self).__init__()
        self.fpos = fpos
        self.opt_fpos = None
        self.mf = mf
        self.mol = mol
        self.grid = grid
        self.s = s
        #print("PreOpt: {0:+0.4f} {1:+0.4f} {2:+0.4f}".format(fpos[0],fpos[1],fpos[2]))
        self.make_fo()

    def make_fo(self, ai=None):
        """build the fo for the given descriptor position"""
        #print("building fo ...")

        fpos = self.fpos
        if ai is not None:
            fpos = ai

        fpos = np.array([fpos])

        ksocc = np.where(self.mf.mo_occ[self.s] > 1e-6)[0][-1]
        #print("ksocc: {0}".format(ksocc))

        ao = numint.eval_ao(self.mol,fpos)
        psi = ao.dot(self.mf.mo_coeff[self.s][:,0:ksocc])

        #print(psi.shape)

        # get total spin density at point self.fpos
        sd = np.sqrt(np.sum(np.conjugate(psi)*psi))
        #print sd
        # get value of each ks-orbital at point fpos
        _ks = self.mf.mo_coeff[self.s][:,0:ksocc]
        #print _ks.shape
        #sys.exit()
        #print np.array_str(_ks, precision=4, max_line_width=120)
        _R = np.reshape(psi/sd, (ksocc))
        #fo = np.matmul(_R.transpose(),_ks)
        #fo = np.zeros((_ks.shape[0]), dtype=np.float64)
        #print fo.shape
        #print _ks[:,0].shape
        #print _R[0]
        #for j in range(ksocc):
        #    fo += _R[j]*_ks[:,j]
        #
        #
        #print np.array_str(fo, precision=4, max_line_width=120, suppress_small=True)



        self.fo = np.matmul(_R,_ks.transpose())
        #print np.array_str(self.fo, precision=4, max_line_width=120, suppress_small=True)


    def spread(self, grid=None, order=2):
        """calculate the spread of a given orbital"""
        #print mf.grids.coords

        mf = self.mf
        mol = self.mol
        grid = self.grid
        mo_coeff = self.fo


        #coords = mf.grids.coords
        #weights = mf.grids.weights
        #if grid is not None:
        coords = grid.coords
        weights = grid.weights

        #print coords.shape
        #print weights.shape


        ao = numint.eval_ao(mol,coords)
        psi = ao.dot(mo_coeff)

        #print psi.shape

        ## calculate r^2
        #rsqr = np.linalg.norm(mf.grids.coords,axis=1)**2.0

        #print np.array_str(rsqr, precision=3)


        #lh = psi[:] * rsqr[:] * psi[:] * mf.grids.weights



        # calculate <phi | r | phi >
        rh = np.zeros_like(coords)
        rh[:,0] = psi[:] * coords[:,0] * psi[:] * weights
        rh[:,1] = psi[:] * coords[:,1] * psi[:] * weights
        rh[:,2] = psi[:] * coords[:,2] * psi[:] * weights


        r = coords
        kern = np.linalg.norm((r - rh), axis=1)
        kern = kern**order
        #print kern.shape
        #sys.exit()
        ##**order

        #qq = np.zeros_like(weights)
        spr = np.sum(psi[:] * kern[:] * psi[:] * weights)

        return spr

    def ofunc(self,x0):
        """the objective function that needs to be optimized"""
        lsrpead = 0.0
        _x=x0[0]
        _y=x0[1]
        _z=x0[2]

        self.make_fo(ai=x0)

        lspread = self.spread()

        return lspread

    def optimize_spr(self, ofodpos):
        """
        optimize the position of this fod

        ofodpos holds the positions of all fods (calculate boundaries)
        """
        from scipy import optimize

        #print np.array_str(ofodpos, precision=3)

        # find the closest fod
        dists = np.linalg.norm(ofodpos - self.fpos, axis=1)
        #print dists
        sortidx = np.argsort(dists)
        mindist = dists[sortidx[1]]*0.5



        lb = self.fpos - mindist
        ub = self.fpos + mindist

        #print self.fpos
        #print 'lb', lb
        #print 'ub', ub

        bounds = optimize.Bounds(lb,ub)

        #print self.ofunc(self.fpos)
        #print self.ofunc(lb)
        #print self.ofunc(ub)

        _x0 = self.fpos.copy()
        options = { 'disp' : -1,
                    'eps': 1e-04,
                    'gtol': 1e-03,
                    'maxiter': 20,
                    'maxls': 5
        }

        res = optimize.minimize(self.ofunc, _x0, method='L-BFGS-B',
            bounds=bounds, options=options)

        #print res.fun
        #print self.fpos
        self.opt_fpos = res.x
        self.opt_spr = res.fun

        return res.x


class FLO(object):
    """docstring for FLO"""
    def __init__(self, mf, s, fod, sinfo=None, ks_evalues=None, init_vsic=None):
        super(FLO, self).__init__()
        self.mf = mf
        self.mol = mf.mol
        self.s = s
        self.fod = fod   # in Angst !
        self.fod = self.fod/units.Bohr

        self.nfod = self.fod.shape[0]
        self.ks_idx = None
        #self.vsic_rebuild_eps = np.finfo(np.float64).eps
        self.vsic_rebuild_eps = 1e-12
        self._esictot = None
        self.gtol = 1e-3
        self.grids_coords_save = mf.grids.coords.copy()
        self.grids_weights_save = mf.grids.weights.copy()


        # this is for basic mpi support
        self.use_mpi = False
        if mpi is not None:
            wsize = MPI.COMM_WORLD.Get_size()
            #print(">>> WSIZE {}".format(wsize))
            if wsize > 1: self.use_mpi = True


        try:
            type(self.mf.on)
        except AttributeError:
            # looks like its an very plain base class
            self.mf.on = None

        ## check for pyscf version
        vv = pyscf.__version__.split('.')
        vmaj = int(vv[0])
        vmin = int(vv[1])
        try:
            vfix = int(vv[2])
        except:
            vfix = 0

        if vmin < 5:
            print("class FLO: This code requires pyscf >= 1.5.2")
            sys.exit()
        elif (vmin == 5) and (vfix == 1):
            print("class FLO: This code requires pyscf >= 1.5.2")
            sys.exit()
        else:
            pass

        logger.info(self.mf,
            "--- FOD positions (spin {0}, nfod {1}) ---"
            .format(self.s, self.nfod))

        for i in range(self.nfod):
            sym = 'X'
            if self.s == 1: sym = 'He'
            ostr = "{0}   {1:>+9.7f}  {2:>+9.7f} {3:>+9.7f}"\
                .format(sym,self.fod[i,0],self.fod[i,1],self.fod[i,2])
            logger.info(self.mf,ostr)

        #print mf.FLOSIC.on.is_init

        #print(">>> FLO mf type:", type(self.mf))

        #sys.exit()

        self.make_flos()

        self.onedm = np.zeros((self.nfod,self.nks,self.nks), dtype=np.float64)
        self.vsic = np.zeros((self.nfod,self.nks,self.nks), dtype=np.float64)
        self.vsic_init = False
        self.energies = np.zeros((self.nfod,3), dtype=np.float64)


    def set_shell_restricted(self, ks_idx):
        '''
        Enable shell restricted mode.
        (see FLOShell fro more details.)
        '''
        self.ks_idx = ks_idx

    def make_flos(self):
        """Use Lowdins method to obtain a set of orthonormalized FLOs \
        from the given FOs.
        """
        # make sure the FO's are already built()
        _fo = self.make_fo()
        # switch indicess
        fo = np.transpose(_fo)
        nfod = self.nfod

        # get the atomic overlap
        s1e = self.mf.get_ovlp(self.mol)
        self.s1e = s1e

        # Initialize everything for the Lowdin orthonormalization.
        T_lo = np.zeros((nfod,nfod), dtype=np.float64)
        Q_lo = np.zeros((nfod), dtype=np.float64)

        ovrlp_fo = np.zeros((nfod,nfod), dtype=np.float64)

        # Get the overlap.
        # The atomic overlap is directly included in sfo.
        sroot = np.linalg.cholesky(s1e)
        sfo = np.dot(np.transpose(sroot),fo[:,:])
        self.sfo = np.transpose(sfo)
        ovrlp_fo[0:nfod,0:nfod] = np.dot(np.transpose(sfo[:,0:nfod]),sfo[:,0:nfod])

        #print(np.array_str(ovrlp_fo,precision=5, max_line_width=240, suppress_small=True))
        #sys.exit()

        # This is a Lowdin symmetric orthonormalization.
        q_fo,v_fo = np.linalg.eigh(ovrlp_fo[0:nfod,0:nfod])
        T_lo[0:nfod,0:nfod] = v_fo
        Q_lo[0:nfod] = q_fo
        #print(np.array_str(Q_lo, precision=4,max_line_width=240, suppress_small=False))
        #sys.exit()
        one_div_d = (1.0 / np.sqrt(q_fo)) * np.eye(nfod)
        vinv_fo = (np.transpose(v_fo))
        tra1 = np.dot(v_fo,one_div_d)
        trafo = np.dot(tra1,vinv_fo)

        # rotate FOs into FLOs
        # in the result, first index is nfod (!)
        # that is different from the original
        # Lenz/Sebastian code, but more effective !
        flo = np.matmul(trafo,fo.T)

        # create storage for flo's
        self.flo = np.zeros((self.nks,self.nks), dtype=np.float64)

        # copy matmul result to class storage
        # this copies only the flos
        self.flo[:self.nfod,:] = flo[:self.nfod,:]

        # the rest of the array is filled with the
        # original KS orbitals
        # (be very carefull with that if you only generate FLOs
        #  per shell)
        for i in range(self.nfod,self.nks):
            self.flo[i,:] = self.mf.mo_coeff[self.s][:,i]
        #print(np.array_str(self.flo, precision=3, \
        #    suppress_small=True, max_line_width=240))

    def sumspread(self, excl_core = True, use_fo=False):
        """docstring for sumspread"""
        spr = 0.0
        for j in range(self.nfod):
            if self.is_1score(j): continue
            s = self._spreadi(j, use_fo=use_fo)
            #print("spread {0}: {1:7.5f}".format(j,s))
            spr += s

        return spr

    def make_fo(self):
        """build the fo for the given descriptor position"""
        fpos = self.fod

        # this is a critical point where we should find out how to
        # obtain a more reliable measure for the number of electrons
        try:
            ksocc = np.where(self.mf.mo_occ[self.s] > 1e-6)[0][-1] + 1
        except IndexError:
            # there are no electrons (hopefully)
            ksocc = 0
        #ksocc = len(self.mf.mo_occ[self.s][self.mf.mo_occ[self.s]==1])
        self.ksocc = ksocc
        self.nks = self.mf.mo_occ[self.s].shape[0]

        ao = numint.eval_ao(self.mol,fpos)

        # first index is nfod, second is orbital
        psi = ao.dot(self.mf.mo_coeff[self.s][:,:ksocc])

        # get total spin density at points self.fpos
        sd = np.sqrt(np.sum(psi**2, axis=1))
        # get value of each ks-orbital at point fpos
        _ks = self.mf.mo_coeff[self.s][:,0:ksocc]

        _R = np.zeros((self.nfod,ksocc))
        for m in range(0,self.nfod):
            _R[m,0:self.nfod] = psi[m,0:self.nfod]/sd[m]
        self._R = _R


        # self.fo -> first index is nr of FO
        self.fo = np.matmul(_R,_ks.transpose())
        #print(self.fo.shape)
        #sys.exit()

        #for m in range(self.nfod):
        #    ao1 = numint.eval_ao(self.mol,self.mf.grids.coords)
        #    phi = ao1.dot(self.fo[m])
        #    dens = np.sum(phi**2*self.mf.grids.weights)
        #    print(dens)
        #
        #sys.exit()
        return self.fo

    def get_dflo_dai(self, fod_id, delta=2e-4):
        """docstring for get_dflo_dai"""
        # dflo[0,:] -> dFLO/d_aix
        # dflo[1,:] -> dFLO/d_aiy
        # dflo[2,:] -> dFLO/d_aiz
        dflo = np.zeros((3,self.nfod, self.nks), dtype=np.float64)
        _flo = self.flo.copy()
        for k in range(3):
            # create new flos's with delta
            #print(pflo.fod[1,0])
            self.fod[fod_id,k] += delta
            #print(pflo.fod[1,0])

            # re-create flo's
            self.make_flos()
            flop = self.flo.copy()
            self.fod[fod_id,k] -= 2.0*delta
            self.make_flos()

            _qq = (flop - self.flo)/(2.0*delta)
            # copy only the 'occupied' FLO's
            dflo[k,0:self.nfod] = _qq[0:self.nfod].copy()

            #print(np.array_str(flo[:9],precision=4, suppress_small=True, max_line_width=120))
            #print(np.array_str(dflo_dai[:9],precision=4, suppress_small=True, max_line_width=120))
            # reset original values
            self.fod[fod_id,k] += delta
        self.flo = _flo.copy()

        return dflo

    def get_esic(self, fod_id, use_fo=False):
        """docstring for get_esic"""
        mf = self.mf
        nks = self.nks
        mol = self.mol

        occup_work =  np.array(mf.mo_occ).copy()
        for i in range(0,nks):
            if i == fod_id:
                occup_work[0][i] = 0.
                occup_work[1][i] = 0.
                occup_work[self.s][i] = 1.
            else:
                occup_work[0][i] = 0.
                occup_work[1][i] = 0.

        # Build the one electron densities.
        # shall we use the FO's instead of the FLO's
        # to calculate the SIC ?
        #print(self.fo.shape)
        #print(self.flo.shape)
        if use_fo:
            self.flo[0:self.nfod,:] = self.fo[0:self.nfod,:]
        dm_work_flo = dynamic_rdmc(self.flo,occup_work[self.s])

        #print dm_work_flo.shape
        #sys.exit()

        self.onedm[fod_id] = dm_work_flo


        # Get the SIC potential and energy for FLO.

        dm_on = dm_work_flo.copy()
        flo_on = self.flo.copy()
        #dm_on = np.zeros_like(dm_work_flo)


        # check if we want to use O(N) methodology
        if self.mf.on is not None:
            #print("ON MODE")
            #print mf.on
            #print mf.on.is_init
            #sys.exit()
            dm_on = \
                mf.on.get_on_dm(self.s,fod_id,dm_work_flo, flo_on)
            #dm_on[1] = mf.on.get_on_dm(s,j,dm_work_flo[1])
            # adjust meshes and mol
            mol_save = mol.copy()
            #print(mf.grids.coords.shape)
            gsave = mf.grids
            mf.grids = mf.on.fod_onmsh[self.s][fod_id] #.copy()

        # call the coulomb solver etc.
        _dm_on = np.zeros((2,self.nks,self.nks), dtype=dm_on.dtype)
        _dm_on[self.s] = dm_on

        #print(mf.grids.coords.shape)
        #print(np.array_str(_dm_on[self.s], precision=3, suppress_small=True, max_line_width=210))

        #mol.verbose = 4
        #chk = np.sqrt(_dm_on[self.s]**2)
        #qq = np.where(chk < 1e-9)
        #
        ##print(qq[0].shape)
        ##print(qq[1].shape)
        #if len(qq[0]) > 0:
        #    #print(chk[qq[0][0],qq[1][0]])
        #    for ii in range(len(qq[0])):
        #        _dm_on[self.s][qq[0][ii], qq[1][ii]] = 0.0

        #sys.exit()

        #if np.isnan(np.sum(_dm_on[self.s])):
        #    print("Lurg")
        #    sys.exit()

        veff_work_flo = mf.get_veff(mol=mol,dm=_dm_on)
        self.vsic[fod_id] = veff_work_flo[self.s]


        #print(">>", np.isnan(veff_work_flo.__dict__['exc']))

        if np.isnan(veff_work_flo.__dict__['exc']):
            print("EXC is NaN")
            sys.exit()

        # reset the original DFT mesh
        if self.mf.on is not None:
            #pass
            self.mf.grids = gsave


        ##
        #print(">> {0} {1}".format(s,j))
        #print np.array_str( dm_on[s], precision=4, suppress_small=False, max_line_width=280)
        #print(">")
        #print np.array_str( dm_on[s], precision=3, suppress_small=False, max_line_width=280)



        # Save the SIC potential.
        #vsics[s,j] = veff_work_flo[s]

        # Get the SIC energy parts for every FLO.

        exc_work_flo = veff_work_flo.__dict__['exc']
        ecoul_work_flo = veff_work_flo.__dict__['ecoul']
        _esic_orb = -exc_work_flo - ecoul_work_flo

        self.energies[fod_id,0] = _esic_orb
        self.energies[fod_id,1] = ecoul_work_flo
        self.energies[fod_id,2] = exc_work_flo
        if self.mol.verbose > 3:
            print(' {:>3d} {:>11.5f} {:>11.5f} {:>11.5f}'\
                .format(fod_id, exc_work_flo , ecoul_work_flo, _esic_orb))

        #sys.exit()

        return _esic_orb

    def make_onedms(self):
        """docstring for make_onedms"""
        mf = self.mf
        nks = self.nks
        mol = self.mol
        #self.onedms = list()
        for j in range(self.nfod):
            occup_work =  np.array(mf.mo_occ).copy()
            for i in range(0,nks):
                if i == j:
                    occup_work[0][i] = 0.
                    occup_work[1][i] = 0.
                    occup_work[self.s][i] = 1.
                else:
                    occup_work[0][i] = 0.
                    occup_work[1][i] = 0.
            odm = dynamic_rdmc(self.flo,occup_work[self.s])
            #print(odm.shape)
            #print(type(odm))
            #sys.exit()
            #self.onedms.append(odm)
            self.onedm[j,:,:] = odm[:,:].copy()


        #print(nks)
        #print(len(self.onedms))
        #sys.exit()

    def update_vsic(self, fod_id=None, npos=None, uall=False):
        """update sic potentials if needed"""
        #print(">> update_vsic called {0}".format(fod_id))
        # set the mpi comm
        if npos is not None:
            print("    {0} : {1} -> {2}"
                .format(fod_id,self.fod[fod_id,:],npos))
            if type(fod_id) is None:
                print("Error, wrong fod_id type in update_vsic")
                sys.exit()
            self.fod[fod_id,:] = npos[:]

        # back up current flo's
        flo_last = self.flo.copy()

        # rebuild orbitals and density matrices
        self.make_flos()
        self.make_onedms()

        # check for initialization
        if self.vsic_init == False:
            uall = True
            self.vsic_init = True

        # calculate difference to decide which
        # vsics needs to be updated
        flo_diff = np.zeros(self.nfod, dtype=np.float64)
        for j in range(self.nfod):
            flo_diff[j] = np.linalg.norm(self.flo[j,:] - flo_last[j,:])
        ##print flo_diff
        upd_ids = np.argwhere(flo_diff > self.vsic_rebuild_eps)
        ##print("    lowest flo_diff {0:.3E}".format(np.min(flo_diff)))

        upd_ids=np.reshape(upd_ids, (-1))
        ##print("    updating vsic, need {0} v_coul".format(upd_ids.shape[0]))


        # now update all sic potentials, if requested
        if uall: upd_ids = np.array(range(self.nfod))

        # now loop over the grps
        if self.mf.on is None:
            fodgrps = [list(upd_ids)]
        else:
            fodgrps = self.mf.on.fodgrps[self.s]

        #print(fodgrps)

        # to do, purge fodgrps for fodids that are not required to recalc
        ##skiptable = list()
        ##for j in enumerate(fodgrps):
        ##    skiptable.append([])
        ##for ifgrp, fgrp in enumerate(fodgrps):
        ##    for j, fodid in enumerate(fgrp):
        ##        #print('fgrp', fgrp)
        ##        #print('upids', upd_ids)
        ##        if fodid not in upd_ids: skiptable[ifgrp].append(fodid)
        #print(skiptable)
        #sys.exit()

        # loop over the group of fod's and calculate
        # veff for each group (thah has the same mesh) in a single call
        # (huge speedup !)
        if len(upd_ids) > 0 and self.mol.verbose > 3:
            print(' ---- orbital energies, spin {} ----'.format(self.s))
            print('#FLO {:>11} {:>11} {:>11} {:>11}'.format('E_xc', 'E_coul', 'E_sic', 'Nmsh'))
        ##print('nfod', self.nfod)
        for ifgrp, fgrp in enumerate(fodgrps):
            if len(fgrp) == 0:
                ## print('Empty fgrp')
                continue
            #print(' v_sic fgrp  {}'.format(fgrp))
            # prepare list of dm's for veff
            dmsize = len(fgrp)
            dma = np.zeros((dmsize, self.nks, self.nks), dtype=np.float64)
            dmb = np.zeros((dmsize, self.nks, self.nks), dtype=np.float64)
            for j, fodid in enumerate(fgrp):
                ##if fodid in skiptable[ifgrp]: continue
                #print('j,fodid', j,fodid)
                odm = self.onedm[fodid][:,:].copy()
                if self.mf.on is not None:
                    _odm = self.mf.on.get_on_dm(self.s, fodid, odm)
                    if self.s == 0:
                        dma[j,:,:] = _odm[:,:].copy()
                    else:
                        dmb[j,:,:] = _odm[:,:].copy()
                else:
                    if self.s == 0:
                        dma[j,:,:] = odm[:,:].copy()
                    else:
                        dmb[j,:,:] = odm[:,:].copy()

                #dm[j,:,:] = odm[:,:]
            _dm = [dma,dmb]
            #print(np.array(_dm).shape)

            # prepare grid for veff
            if self.mf.on is not None:
                self.mf.grids.coords = self.mf.on.fod_onmsh[self.s][fgrp[0]].coords.copy()
                self.mf.grids.weights = self.mf.on.fod_onmsh[self.s][fgrp[0]].weights.copy()
            # for debug message
            nmsh = self.mf.grids.weights.shape[0]

            # check if we want to use mpi
            if self.use_mpi and len(fgrp) > 1:
                comm = MPI.COMM_WORLD
                wsize = comm.Get_size()


                #sys.exit()
                for inode in range(1,wsize):
                    comm.send('vsic', dest=inode, tag=11)

                # send required mpi data to all nodes
                idata = np.empty(2, dtype='i')
                # send the size of the groups to the slaves
                idata[0] = len(fgrp)
                idata[1] = nmsh
                comm.Bcast(idata, root=0)
                _dmtmp = np.array(_dm, dtype=np.float64)
                comm.Bcast(_dmtmp, root=0)
                _weights = np.zeros_like(self.mf.grids.weights, dtype='d')
                _weights[:] = self.mf.grids.weights[:]
                comm.Bcast(self.mf.grids.weights, root=0)
                _coords = np.zeros_like(self.mf.grids.coords, dtype='d')
                _coords[:,:] = self.mf.grids.coords[:,:]
                comm.Bcast(self.mf.grids.coords, root=0)

                sidx, eidx, csize = get_mpichunks(len(fgrp),0,comm=comm)
                #print("sidx, eidx, csize", sidx, eidx, csize)
                #print(np.array(_dm).shape)
                _dmmpi = slice_dm4mpi(_dm, sidx, eidx)

                # call the solver
                #print(">>> master: ", np.asarray(_dmmpi).shape, flush=True)
                _veff = self.mf.get_veff(mol=self.mol, dm=_dmmpi)
                #print(">>> master _veff: ", hasattr(_veff,'__dict__'), flush=True)

                _veff_gath = comm.gather(_veff, root=0)

                #print(">>> master len(_veff_gath)", len(_veff_gath))

                # build a new _veff object to sort in all results from the
                # mpi slaves
                _veff = np.zeros_like(_dmtmp)
                #print(_veff.shape)
                offs = 0
                _exc   = list()
                _vj    = list()
                for nslave, vefftmp in enumerate(_veff_gath):
                    #print(nslave)
                    #print(np.sum(vefftmp))
                    nveff = vefftmp.shape[1]
                    #print('nveff', nslave, nveff)
                    _veff[0,offs:offs+nveff,:,:] = vefftmp[0,:,:,:]
                    _veff[1,offs:offs+nveff,:,:] = vefftmp[1,:,:,:]
                    for j in range(nveff):
                        _exc.append(vefftmp.__dict__['exc'][j])
                        _vj.append(vefftmp.__dict__['vj'][j])
                    offs += nveff
                _veff = lib.tag_array(_veff, ecoul=None,
                    exc=np.asarray(_exc),
                    vj=np.asarray(_vj),
                    vk=None)
            else:
                # call the veff code, put in all one electron dm's at once
                #_dm = [np.random.random((1,self.nks,self.nks)),
                #  np.random.random((1,self.nks,self.nks))]
                _veff = self.mf.get_veff(mol=self.mol, dm=_dm)

            # restore original grid size (if needed)
            if self.mf.on is not None:
                self.mf.grids.coords = self.grids_coords_save.copy()
                self.mf.grids.weights = self.grids_weights_save.copy()

            # now store the results
            for j, fodid in enumerate(fgrp):
                #if fodid in skiptable[ifgrp]: continue
                self.vsic[fodid] = _veff[self.s][j]
                _exc   = _veff.__dict__['exc'][j]
                #print(type(_veff.__dict__['exc']))
                #print(_veff.__dict__['exc'].shape)
                #sys.exit()
                #print(_veff.__dict__['vj'].shape)
                #sys.exit()
                _vj = _veff.__dict__['vj'][j,:,:]
                # in list-mode, veff does not calculate the coulomb energy
                # so we do this by ourself for each 1e dm
                _ecoul = np.einsum('ij,ji', _dm[self.s][j], _vj) * .5
                _esic  = -_exc - _ecoul
                self.energies[fodid,0] = _esic
                self.energies[fodid,1] = _ecoul
                self.energies[fodid,2] = _exc
                if self.mol.verbose > 3:
                    print(' {:>3d} {:>11.5f} {:>11.5f} {:>11.5f} {:>11d}'\
                        .format(fodid, _exc , _ecoul, _esic, nmsh))

        self._esictot = np.sum(self.energies[:,0])
        #print('esictot', self._esictot)

        self.lambda_ij = np.zeros((self.nfod,self.nfod), dtype=np.float64)
        ket = np.zeros((self.nks,1), dtype=np.float64)
        bra = np.zeros((1,self.nks), dtype=np.float64)

        # re-calculate lambda_ij
        for j in range(self.nfod):
            for i in range(self.nfod):
                bra[0,:] = np.transpose(self.flo[i])
                ket[:,0] = (self.flo[j])
                right = np.dot(self.vsic[j],ket)
                self.lambda_ij[i,j] = -np.dot(bra,right)

        return

    def get_desic_dai(self):
        '''
        This function returns the gradient of the sic energy
        with respect to the referenz-position of each Fermi
        orbital.

        It is essentially a re-write of Lenz/Sebastians implementation,
        but stripped down to single spin and optimized to use the data structures
        of this class.
        '''
        if not self.vsic_init:
            self.update_vsic()

        desic = np.zeros((self.nfod,3), dtype=np.float64)

        nfod = self.nfod
        s = self.s

        h_sic_input = self.lambda_ij

        ## symmetrize h_sic_input results in zero force
        #for i in range(nfod):
        #    for j in range(nfod):
        #        _q = h_sic_input[i,j] + h_sic_input[j,i]
        #        _q *= 0.5
        #        h_sic_input[i,j] = _q
        #        h_sic_input[j,i] = _q

        # check wheter or not we are going to use shell
        # restricted FLO's
        self.ksocc = np.where(self.mf.mo_occ[self.s] > 1e-6)[0][-1] + 1
        nksocc = self.ksocc
        #print(nksocc)
        if self.ks_idx is not None: nksocc = len(self.ks_idx)
        #print(nksocc)
        if self.ks_idx is None: self.ks_idx = list(range(nksocc))
        #print('nksocc: {}'.format(nksocc))
        nks = self.nks
        nbas = nks  # convinience
        occup = self.mf.mo_occ[s]


        # prepare the truth-table to be used to skip the
        # calculation of dphi/dai for very small sic potential
        # matrix elements
        eps_cutoff = 0.0
        ttable = np.eye(self.lambda_ij.shape[0], dtype=np.bool)
        if self.mf.on is not None:
            eps_cutoff = self.mf.on.eps_cutoff
            #if rank == 0: print("eps_cutoff {0}".format(eps_cutoff))
            # mask out everything that is not within the subsystem
            xidx, yidx = np.where(np.abs(h_sic_input) < eps_cutoff)
            for idx in range(xidx.shape[0]):
                ttable[xidx[idx],yidx[idx]] = True
            #for i in range(nfod):
            ##    # find the fods that correspond the the given fod
            ##    valid_fods = self.mf.on.fod_fod[s][i]
            #    for j in range(nfod):
            #        #if j == i: continue
            ##        if j not in valid_fods:
            ##            ttable[i,j] = True
        #print(np.array_str(self.lambda_ij, max_line_width=120))
        #print(np.array_str(ttable, max_line_width=120))
        #if rank == 0: print(np.array_str(ttable, max_line_width=120))
        #sys.exit()

        # define required arrays
        Psi_ai = np.zeros((nfod,nksocc), dtype=np.float64)
        gradpsi_ai = np.zeros((nfod,nksocc,3), dtype=np.float64)
        den_ai = np.zeros((nfod), dtype=np.float64)
        grad_ai = np.zeros((nfod,3), dtype=np.float64)
        gradfo = np.zeros((nbas,nfod,3), dtype=np.float64)
        sumgradpsi = np.zeros((nbas,nfod,3), dtype=np.float64)
        gradovrlp = np.zeros((nfod,nfod,3), dtype=np.float64)
        Delta1 = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        Delta3 = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        eps = h_sic_input
        # Fermi orbital overlap matrix.
        s_i_j = np.zeros((nfod,nfod), dtype=np.float64)

        # Cholesky decomposition for the atomic overlap matrix.
        sroot = np.linalg.cholesky(self.s1e)

        # Get the value of the gradients of the KSO at the FOD positions.
        # check if we have cartesian representation or not
        sphocart='GTOval_ip_sph'
        if self.mol.cart: sphocart='GTOval_ip_cart'
        ao1 = self.mol.eval_gto(sphocart, self.fod, comp=3)
        #print(ao1.shape)
        #print(self.mf.mo_coeff.shape)

        # use only the coefficients that belong to the same shell
        _coeff = np.zeros_like(self.mf.mo_coeff)
        _coeff[self.s][:,self.ks_idx] = \
                self.mf.mo_coeff[self.s][:,self.ks_idx]

        # obtain the gradients of the KS orbitals at the
        # ai positions
        gradpsi_ai_raw = [x.dot(_coeff) for x in ao1]

        # Rearrange the data to make it easier to use
        x_1 = gradpsi_ai_raw[0]
        y_1 = gradpsi_ai_raw[1]
        z_1 = gradpsi_ai_raw[2]


        #print('x_1', x_1.shape)
        #sys.exit()

        # obtain the density at the ai positions
        ao1 = numint.eval_ao(self.mol, self.fod)
        psi_ai_1 = ao1.dot(_coeff)

        #print(self.ks_idx)
        #print(psi_ai_1[:,0,:].shape)
        ##sys.exit()

        # prepare Psi and gradpsi arrays
        # (just to make the code readable)
        # copy only the required shell indexes back to
        # Psi_ai and gradpsi_ai
        for m in range(nfod):
            Psi_ai[m,:] = psi_ai_1[m,s,self.ks_idx]
            gradpsi_ai[m,:,0] = x_1[m][s][self.ks_idx]
            gradpsi_ai[m,:,1] = y_1[m][s][self.ks_idx]
            gradpsi_ai[m,:,2] = z_1[m][s][self.ks_idx]
        #print(l)
        #sys.exit()
        #print(np.array_str(gradpsi_ai[:,:,0], precision=4, suppress_small=True, max_line_width=120))

        # Calculate the density and the gradient of the density from the KS wavefunctions.
        for m in range(nfod):
            den_ai[m] = np.sum((Psi_ai[m,:])**2)

        for r in range(0,3):
            for m in range(nfod):
                grad_ai[m,r] =  np.sum(2.*Psi_ai[m,:]*gradpsi_ai[m,:,r])

        #sks = np.zeros((nbas,nksocc), dtype=np.float64)
        #sfo = np.zeros((nbas,nfod), dtype=np.float64)
        ks = self.mf.mo_coeff[s]
        fo = np.transpose(self.fo)

        # Get the gradients of the Fermi orbitals. (NOTE: NOT THE FERMI LOWDIN ORBITALS!)
        # This is dF in the usual notation.
        # Fill sks and sfo.
        sks = np.dot(np.transpose(sroot),ks[:,self.ks_idx])
        sfo = np.dot(np.transpose(sroot),fo[:,:])

        #print(sks.shape)
        #print(sfo.shape)
        #
        #print(np.array_str(sks, precision=4,\
        #     suppress_small=True, max_line_width=120))
        #
        #print(np.array_str(sfo, precision=4,\
        #     suppress_small=True, max_line_width=120))

        #sys.exit()

        # bra and ket for scalar products.
        ket = np.zeros((nfod,1), dtype=np.float64)
        bra = np.zeros((1,nfod), dtype=np.float64)

        # Get dF.
        # THa: im really not sure if the following loop is
        # entirely correct under all circuumstances,
        # especially for cases where nfod != nksocc
        for r in range(3):
            for i in range(nfod):
                sum1 = np.zeros((nbas), dtype=np.float64)
                for a in range(nksocc):
                    sum1 = gradpsi_ai[i,a,r]*sks[:,a] + sum1
                gradfo[:,i,r] = sum1[:] / np.sqrt(den_ai[i]) - (sfo[:,i]*grad_ai[i,r]) / (2.*den_ai[i])

        #sys.exit()

        s_i_j = np.zeros((nfod,nfod), dtype=np.float64)
        s_i_j[:nfod,:nfod] = np.dot(np.transpose(sfo[:,:nfod]),sfo[:,:nfod])

        ## Get the eigenvectors as done by NRLMOL.
        Q_alpha_tmp,T_alpha_tmp = np.linalg.eigh((s_i_j[0:nfod,0:nfod]))
        T_alpha = np.zeros((nfod,nfod), dtype=np.float64)
        Q_alpha = np.zeros((nfod), dtype=np.float64)

        #print(np.array_str(Q_alpha_tmp, precision=6))
        #sys.exit()


        # Resort the matrices according to NRLMOL formalism.
        for i in range(0,nfod):
            for j in range(0,nfod):
                T_alpha[j,nfod-1-i] = T_alpha_tmp[j,i]
                Q_alpha[nfod-1-i] = Q_alpha_tmp[i]

        T_alpha = np.transpose(T_alpha)

        # Temporary variables.

        TdST = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        V_tmp = np.zeros((nfod), dtype=np.float64)
        M_tmp = np.zeros((nfod,nfod), dtype=np.float64)
        D1_km = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        D1_kmd = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        D3_km = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)
        D3_kmd = np.zeros((nfod,nfod,nfod,3), dtype=np.float64)

        # Get dS.
        for r in range(0,3):
            for n in range(nfod):
                for m in range(nfod):
                    gradovrlp[n,m,r] = np.dot(np.transpose(sfo[:,n]),gradfo[:,m,r])

            # Get Matrix elements <T_j|dSdAm|T_k>.
            for m in range(nfod):
                for a in range(nfod):
                    for b in range(nfod):
                        TdST[b,a,m,r] += \
                        np.sum( gradovrlp[:,m,r]*(T_alpha[b,:]*T_alpha[a,m]+T_alpha[b,m]*T_alpha[a,:]))
                        #for i in range(nfod):
                            #TdST[b,a,m,r]=TdST[b,a,m,r]+gradovrlp[s,i,m,r]*(T_alpha[b,i]*T_alpha[a,m]+T_alpha[b,m]*T_alpha[a,i])

            # Get <phi|D1,km>
            V_tmp[0:nfod] = 1./np.sqrt(Q_alpha[0:nfod])
            M_tmp = np.zeros((nfod,nfod), dtype=np.float64)
            M_tmp2 = np.zeros((nfod,nfod), dtype=np.float64)

            for m in range(nfod):
                for k in range(nfod):
                    M_tmp[m,k] = np.sum(T_alpha[0:nfod,k]*T_alpha[0:nfod,m]*V_tmp[0:nfod])

            M_tmp2 = np.dot(M_tmp[0:nfod,0:nfod],gradovrlp[0:nfod,0:nfod,r])

            for m in range(0,nfod):
                #for kl in list(kl_iter[s]):
                for k in range(0,nfod):
                    D1_km[0:nfod,k,m,r] +=  M_tmp[m,k]*M_tmp2[0:nfod,m]
                    #D1_km[:,k,m,r] += np.sum(M_tmp[m,k]*M_tmp2[:,m])
                    #for l in range(0,nfod):
                    #k,l = kl
                    #    D1_km[l,k,m,r] = D1_km[l,k,m,r] + M_tmp[m,k]*M_tmp2[l,m]


            # Get D1_kmd (the lower case d meaning delta).
            for m in range(nfod):
                D1_kmd[0:nfod,0:nfod,m,r] = D1_km[0:nfod,0:nfod,m,r] - \
                    np.transpose(D1_km[0:nfod,0:nfod,m,r])

            # Get the first part of the forces.
            for m in range(nfod):
                for k in range(nfod):
                    desic[m,r] += np.sum(D1_kmd[:,k,m,r]*eps[:,k])


            # Get D3_km
            #
            # This is a higly optimised version to calculate D3_km
            # using a lot of numpy features to speed up the
            # calculation.
            # By using the numpy method (implicite looping etc.)
            # it is possible to bring down the scaling to O(N^3.x),
            # where x is propably < 5.
            #
            # Have fun! (Torsten Hahn)
            Q_alpha_sqrt = np.sqrt(Q_alpha)
            Q_alpha_sqrt = Q_alpha_sqrt.ravel()
            Q_alpha = Q_alpha.ravel()
            out1 = np.zeros((nfod,nfod), dtype=np.float64)
            out2 = np.zeros((nfod,nfod), dtype=np.float64)
            aab = np.zeros((nfod,nfod), dtype=np.float64)


            #t0 = time.time()
            if USE_NUMBA:
                _d3kmr = D3_km_outer_loop(r, nfod, ttable,
                    T_alpha, Q_alpha, Q_alpha_sqrt, TdST)
                #_d3kmr = D3_km_outer_loop_serial(r, nfod, ttable,
                #    T_alpha, Q_alpha, Q_alpha_sqrt, TdST)
            else:
                _d3kmr = D3_km_outer_loop_serial(r, nfod, ttable,
                    T_alpha, Q_alpha, Q_alpha_sqrt, TdST)

            #print(_d3kmr.shape)

            #print("D3kmtime", time.time()-t0)
            #sys.exit()

            ##ncpus = os.cpu_count()
            ###print("ncpus", ncpus)
            ###_iter = lib.prange(0,nfod,ncpus)
            ###print(list(_iter))
            ##mp_func = lib.background_process
            ##if self.mp_type == 'thread':
            ##    mp_func = lib.background_thread
            ##
            ##reslist = list()
            ##for i0, i1 in lib.prange(0,nfod,ncpus):
            ##    #print("process group {} {}".format(i0, i1))
            ##    thlist = list()
            ##    # spawn the processes
            ##    for m in range(i0, i1):
            ##        #print(' -> starting thread {}'.format(m))
            ##        thlist.append(mp_func(do_D3km_inner_loop,
            ##          m, r, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt, TdST))
            ##    # collect results
            ##    for j in range(len(thlist)):
            ##        #print(' <- catching result {}'.format(j))
            ##        reslist.append(thlist[j].get())
            ##
            ###print(len(reslist))
            ##
            ### store the results from the threads in
            ### D3km
            ##for i in range(len(reslist)):
            ##    res = reslist[i]
            ##    m = res[0]
            ##    D3_km[:nfod,:nfod,m,r] = res[1][:,:]
            for i in range(nfod):
                D3_km[:nfod,:nfod,i,r] = _d3kmr[i,:,:]

            ##del thlist
            #del reslist
            #
            #sys.exit()
            ##do_D3km_inner_loop(m, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt)
            #lth = lib.background_process(do_D3km_inner_loop,
            #    0, r, nfod, ttable, T_alpha, Q_alpha, Q_alpha_sqrt, TdST)
            ##def background_process(func, *args, **kwargs):
            ##    '''applying function in background'''
            ##    thread = ProcessWithReturnValue(target=func,
            ##        args=args, kwargs=kwargs)
            ##    thread.start()
            ##    return thread
            #ret = lth.get()
            ##print(ret)
            #sys.exit()
            #
            #for m in range(nfod):
            #    for k in range(nfod):
            #        for l in range(nfod):
            #            # skip in case <phi_i|Vsic_i|phi_j> is pretty small
            #            if ttable[l,k]: continue
            #            Va = np.ravel(T_alpha[:nfod,l]*Q_alpha_sqrt[:nfod])
            #            np.outer(T_alpha[:,k].ravel(), Va, out=out1)
            #            np.outer(Va, T_alpha[:,k].ravel(), out=out2)
            #            out1 += out2
            #            aab[:,:] = 1.0/(np.outer(Q_alpha, Q_alpha_sqrt) + \
            #                        np.outer(Q_alpha_sqrt, Q_alpha))
            #            #aab[:,:] *= -0.5*TdST[:,:,m,r] * out1
            #            D3_km[l,k,m,r] = np.sum(-0.5*TdST[:,:,m,r] * out1 * aab)

            # the original loop
            # take for debugging reasons
            #for m in range(nfod):
            #    for k in range(nfod):
            #        for l in range(nfod):
            #            for a in range(nfod):
            #                for b in range(nfod):
            #                    tmp1 = T_alpha[b,k]*T_alpha[a,l]*np.sqrt(Q_alpha[a])
            #                    tmp2 = T_alpha[a,k]*T_alpha[b,l]*np.sqrt(Q_alpha[b])
            #                    tmp3 = ( np.sqrt(Q_alpha[a]) +np.sqrt(Q_alpha[b]) ) * np.sqrt(Q_alpha[a]*Q_alpha[b])
            #                    D3_km[l,k,m,r]=D3_km[l,k,m,r]-0.5*TdST[b,a,m,r]* ( (tmp1+tmp2) / tmp3)

            # Get D3_kmd (the lower case d meaning delta).
            for m in range(nfod):
                D3_kmd[0:nfod,0:nfod,m,r] = D3_km[0:nfod,0:nfod,m,r] - np.transpose(D3_km[0:nfod,0:nfod,m,r])

            # Get the second part of the forces.
            for m in range(nfod):
                for k in range(nfod):
                    desic[m,r] += np.sum( D3_kmd[:,k,m,r]*eps[:,k] )

        return desic

    def get_pedcond(self):
        """docstring for get_pedcond"""

        lambda_ij = np.zeros((self.nfod,self.nfod), dtype=np.float64)
        lijp = lambda_ij
        ket = np.zeros((self.nks,1), dtype=np.float64)
        bra = np.zeros((1,self.nks), dtype=np.float64)

        err = 0.0
        for i in range(self.nfod):
            for j in range(self.nfod):
                #print "j,i", j, i
                A = self.vsic[i] - self.vsic[j]
                #A = np.dot(A,self.s1e)
                bra[0,:] = np.transpose(self.flo[i,:])
                ket[:,0] = self.flo[j,:]
                right = np.dot(A,ket[:,0])
                lijp[i,j] = np.dot(bra[0,:],right)

                ###
                A = self.vsic[j] - self.vsic[i]
                #A = np.dot(A,self.s1e)
                bra[0,:] = np.transpose(self.flo[j,:])
                ket[:,0] = self.flo[i,:]
                right = np.dot(A,ket[:,0])
                lijp[i,j] -= np.dot(bra[0,:],right)

                err += np.sqrt((lijp[j,i] - lijp[i,j])**2)
                err += lijp[j,i] - lijp[i,j]
        self._lijp = lijp

        #print np.array_str(lijp, precision=4, suppress_small=True, max_line_width=120)
        #print np.linalg.norm(lijp), err
        #sys.exit()

        return err



class FLOShell(FLO):
    """docstring for FLOShell

        ks_idx list of indices (or np.array) that defines the ks orbitals
            to use for building the FOs
    """
    def __init__(self, mf, s, fod, ks_idx):
        #super(FLO, self).__init__()
        self.mf = mf
        self.mol = mf.mol
        self.s = s
        self.fod = fod   # in Angst !
        self.fod = self.fod/units.Bohr

        self.nfod = self.fod.shape[0]
        self.ks_idx = ks_idx

        self.grids_coords_save = mf.grids.coords.copy()
        self.grids_weights_save = mf.grids.weights.copy()

        # let the parent class know we use shell-restricted mode
        super(FLOShell, self).set_shell_restricted(self.ks_idx)


        #self.vsic_rebuild_eps = np.finfo(np.float64).eps
        self.vsic_rebuild_eps = 1e-12
        self._esictot = None
        self.gtol = 1e-3

        logger.info(self.mf,
            "--- FOD positions (spin {0}, nfod {1}) ---"
            .format(self.s, self.nfod))

        for i in range(self.nfod):
            sym = 'X'
            if self.s == 1: sym = 'He'
            ostr = "{0}   {1:>+9.7f}  {2:>+9.7f} {3:>+9.7f}"\
                .format(sym,self.fod[i,0],self.fod[i,1],self.fod[i,2])
        #print mf.FLOSIC.on.is_init

        #sys.exit()

        self.make_flos()
        #self.make_fo()

        self.onedm = np.zeros((self.nfod,self.nks,self.nks), dtype=np.float64)
        self.vsic = np.zeros((self.nfod,self.nks,self.nks), dtype=np.float64)
        self.vsic_init = False
        self.energies = np.zeros((self.nfod,3), dtype=np.float64)


    def make_fo(self):
        """Build FO's from the shells given in ks_idx only"""
        fpos = self.fod
        # this is a critical point where we should find out how to
        # obtain a more reliable measure for the total
        # number of electrons in the system
        ksocc = np.where(self.mf.mo_occ[self.s] > 1e-6)[0][-1] + 1
        self.ksocc = ksocc
        self.nks = self.mf.mo_occ[self.s].shape[0]
        # nbas is better, use that from now on
        self.nbas = self.nks
        # sanity check and warn
        if self.nfod != len(self.ks_idx):
            print('WARNING: no FODs != no given KS orbitals,\
             FLOSIC may not work as expected!\n {} != {}'\
             .format(self.nfod, self.ks_idx))

        # build KS density at the positions of the FOD
        # by using only the given subset of KS orbitals from ks_idx
        ao = numint.eval_ao(self.mol, fpos)
        _coeff = np.zeros_like(self.mf.mo_coeff)
        _coeff[self.s][:,self.ks_idx] = \
                self.mf.mo_coeff[self.s][:,self.ks_idx]

        # _psi first index is nfod, second index is value
        # of subshell _psi at that position
        _psi = ao.dot(_coeff[self.s][:,:ksocc])

        # calculate the sqrt of the density at the fod fositions
        sd = np.zeros((self.nfod), dtype=np.float64)
        for j in range(self.nfod):
            sd[j] = np.sqrt(np.sum(_psi[j]**2))


        # build the rotation matrix to transform KS -> FO
        # (take care of the correct dimensions)
        _R = np.zeros((self.nfod,len(self.ks_idx)))
        for m in range(self.nfod):
            # the value of psi at the fod position
            psi_fod = _psi[m]
            # sd is already the square root of the spin dens
            _R[m,:] = psi_fod[self.ks_idx]/sd[m]
        self._R = _R

        # initialize storage for Fermi orbitals
        # FOs are store with first index to be the orbital index
        # second idx is nbas
        self.fo = np.zeros((self.nfod, self.nbas), dtype=np.float64)
        # a view of the subshell KS for code readability
        _ks = self.mf.mo_coeff[self.s][:,self.ks_idx]

        # finally, build the FO's -> apply the rotation
        # (e.g. the smart way, without unnecessary for-loops)
        for m in range(self.nfod):
            self.fo[m] = np.sum(self._R[m,:]*_ks[:,:], axis=1)

        ## check if the FO's till sum up to unity
        ## FO must ALWAYS be normalized, otherwise
        ## something is really, really broken !
        #for m in range(self.nfod):
        #    ao1 = numint.eval_ao(self.mol,self.mf.grids.coords)
        #    phi = ao1.dot(self.fo[m])
        #    dens = np.sum(phi**2*self.mf.grids.weights)
        #    print('{:>3d} : {:7.5f}'.format(m,dens))

        return self.fo



#---------------

def dynamic_rdmc(mo_coeff, mo_occ):

#    Taken from PySCF UKS class.
    #print ">>>>"
    ##print mo_coeff
    #print mo_coeff.shape
    #print mo_occ
    #print "<<<<"

    mo_a = mo_coeff
    dm_a = np.dot(mo_a.T*mo_occ, mo_a.conj())
    return np.array((dm_a))


def fo(mf, fod, s=0):
    """docstring for fo"""
    ksocc = np.where(mf.mo_occ[s] > 1e-6)[0][-1] + 1
    #print("ksocc: {0}".format(ksocc))
    #print np.where(self.mf.mo_occ[self.s] > 1e-6)
    mol = mf.mol
    ao = numint.eval_ao(mol,[fod])

    # first index is nfod, second is orbital
    psi = ao.dot(mf.mo_coeff[s][:,0:ksocc])
    #print psi.shape
    #sys.exit()

    # get total spin density at point self.fpos
    sd = np.sqrt(np.sum(np.conjugate(psi)*psi, axis=1))
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
    print(phi.shape)

    print('lorb2fod: s={}'.format(s))
    print('lorb2fod: lo_coeff={}'.format(lo_coeff.sum()))


    print(np.sum(phi**2*mf.grids.weights))
    dens = np.conjugate(phi)*phi*mf.grids.weights
    # COM
    x = np.sum(dens*mf.grids.coords[:,0])
    y = np.sum(dens*mf.grids.coords[:,1])
    z = np.sum(dens*mf.grids.coords[:,2])
    #print x


    print("COM: {0:7.5f} {1:7.5f} {2:7.5f}".format(x*units.Bohr,y*units.Bohr,z*units.Bohr))
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
        onmol =  gto.M(atom=mstr,basis=b)
    except RuntimeError:
        onmol =  gto.M(atom=mstr, basis=b, spin=1)

    _mdft = dft.UKS(onmol)
    _mdft.max_cycle = 0
    _mdft.grids.level = grid_level
    _mdft.kernel()
    ongrid = copy(_mdft.grids)


    #print("Original grid size: {0}".format(mf.grids.coords.shape[0]))
    print("  building FO, grid size: {0}"
        .format(ongrid.coords.shape[0]))

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

    return res.x


def initial_guess(m, grid_level):
    """docstring for initial_guess"""
    #print nspin
    #sys.exit()

    mol = m.mol
    nspin = len(mol.nelec)


    fodup = Atoms()
    foddn = Atoms()

    for spin in range(nspin):
        fodout = fodup
        if spin == 1: fodout = foddn
        print("Find Fermi-Orbitals for spin: {0}".format(spin))
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

        print("  total e: {0} (spin {1})".format(te,spin))
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

    return (fodup, foddn)


from ase.test import NotAvailable
from ase.calculators.calculator import Calculator, all_changes
import os, time

class ESICC(Calculator):
    '''Interface to use ase.optimize methods'''

    implemented_properties = ['energy', 'forces']


    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                label=os.curdir, atoms=None, **kwargs):

        Calculator.__init__(self, restart, ignore_bad_restart_file, \
                        label, atoms, **kwargs)
        valid_args = ('mf', 'spin', 'ks_idx')
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
        self.meps = np.finfo(np.float64).eps
        #print("ESICC",self.__fodatoms)
        #print(self.spin)
        if not hasattr(self,'ks_idx'): self.ks_idx = None
        if self.ks_idx is None:
            self.FLO = FLO(self.mf,
                        self.spin,
                        self.atoms.positions)
        else:
            self.FLO = FLOShell(self.mf,
                        self.spin,
                        self.atoms.positions,
                        ks_idx=self.ks_idx)

        if atoms is not None:
            self.results = {'energy': 0.0,
                'free_energy' : 0.0,
                'forces': np.zeros((len(self.atoms), 3))}
        else:
            self.results = {'energy': 0.0,
                'free_energy' : 0.0,
                'forces': np.zeros((1,3))}


        if 'energy' not in self.results:
            print("Woaaaa ")
            sys.exit()

        self.time_vsic     = 0.0
        self.time_desicdai = 0.0
        self.FLO.update_vsic()

        #print(self.implemented_properties)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if atoms is not None:
            lpos = atoms.positions
        else:
            lpos = self.atoms.positions
        _t0 = time.time()
        pdiff = np.linalg.norm(lpos - self.FLO.fod * units.Bohr)
        #print('get_potential_energy, pdiff {}', pdiff, self.meps)
        if (pdiff > self.meps):
            self.FLO.fod = lpos / units.Bohr
            # update sic potential etc.
            self.FLO.update_vsic()

            self.time_vsic += time.time() - _t0

        ##self.results['forces'][:,:] = self.FLO.

        return self.FLO._esictot*units.Ha


    def get_forces(self, atoms=None):
        if atoms is not None:
            lpos = atoms.positions
        else:
            lpos = self.atoms.positions
        _t0 = time.time()
        pdiff = np.linalg.norm(lpos - self.FLO.fod * units.Bohr)
        #print('get_potential_energy, pdiff {}', pdiff, self.meps)
        if (pdiff > self.meps):
            self.FLO.fod = lpos / units.Bohr
            # update sic potential etc.
            self.FLO.update_vsic()
            self.time_vsic += time.time() - _t0

        _t0 = time.time()
        _ff = -units.Ha/units.Bohr*self.FLO.get_desic_dai()
        self.time_desicdai += time.time() - _t0
        ##self.results['forces'][:,:] = self.FLO.
        return _ff

    def print_atoms(self):
        print('print_atoms', self.atoms)
        print(self.atoms.positions)
        return

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        #print(self.results)
        #sys.exit()
        #pdiff = np.linalg.norm(self.atoms.positions - self.FLO.fod * units.Bohr)
        #if (pdiff > self.meps):
        #    self.FLO.fod = self.atoms.positions / units.Bohr

        for p in properties:
            if p == 'energy':
                self.results['energy'] = self.get_potential_energy()
            elif p == 'forces':
                _ff = self.FLO.get_desic_dai()
                self.results['forces'] = -units.Ha/units.Bohr*_ff.copy()
            else:
                raise PropertyNotImplementedError(\
                    'calculation of {} is not implemented'.format(p))


if __name__ == '__main__':
    from flosic_os import flosic,xyz_to_nuclei_fod,ase2pyscf
    from os.path import expanduser, join
    from flosic_scf import FLOSIC
    from pyscf import dft
    from onstuff import ON

    from ase import io

    home = expanduser("~")

    ifile = "/Users/hahn/Jobs/NSCF/SiH4_guess.xyz"

    s = io.read(join(home,ifile))
    pyscf_atoms,nuclei,fod1,fod2,included = xyz_to_nuclei_fod(s)


    s = nuclei

    csymb = s.get_chemical_symbols()
    cpos = s.positions
    astr = ''
    for symb,i in zip(csymb,range(len(csymb))):
        #print sym, pos
        pos = cpos[i]
        astr += "{0} {1:0.12f} {2:0.12f} {3:0.12f};".format(
            symb,pos[0],pos[1],pos[2])

    b = 'ccpvdz'
    #b = 'sto6g'
    spin = 0
    charge = 0

    #print nuclei


    mol = gto.M(atom=astr,
                basis={'default':b},
                spin=spin,
                charge=charge)

    grid_level  = 6
    mol.verbose = 2
    mol.max_memory = 1000
    mol.build()

    xc = 'LDA,PW'

    m = dft.UKS(mol)
    m.diis_start_cycle=2
    m.diis_space = 7
    m.small_rho_cutoff = 1e-9
    m.grids.level = grid_level
    m.xc=xc
    m.kernel()
    #m.analyze()


    m = FLOSIC(mol,xc=xc,fod1=fod1,fod2=fod2,grid_level=grid_level)

    myon = ON(mol,[fod1.positions,fod2.positions], grid_level=grid_level)
    myon.build()

    #myon.print_stats()


    # enable ONMSH
    #m.max_cycle = 2
    m.set_on(myon)

    #print(fod1.positions)
    calc = ESICC(atoms=fod1, mf=m.calc_uks, spin=0)
    #print('Etot 1: {:9.6f}'.format(fod1.get_potential_energy()))
    #print(fod1.get_forces())
    #fod1.positions[2,1] += 0.2
    #print('Etot 2: {:9.6f}'.format(fod1.get_potential_energy()))
    #print(fod1.get_forces())

    #print('>>')
    #print(calc.FLO.fod)


    from ase.optimize import  BFGS, GPMin, FIRE, LBFGS
    from ase.constraints import FixAtoms

    c1sidx = list()
    # fix the C1s fod's
    for fodid in range(myon.nfod[0]):
        ftype = myon.fod_atm[0][fodid][2]
        print("{} {}".format(fodid, ftype))
        if ftype == 'C1s':
            c1sidx.append(fodid)

    c1score = FixAtoms(indices=c1sidx)
    fod1.set_constraint(c1score)


    #sys.exit()

    #precon = Exp(A=2)
    dyn = BFGS(atoms=fod1,
        #trajectory='OPT_FRMORB.traj',
        #precon=None, use_armijo=False,
        logfile='OPT_FRMORB.log')

    dyn.run(fmax=2.5e-4,steps=1000)


    io.write('fodu.xyz',fod1)

    sys.exit()




    print("\nDFT done, starting LO generation ...\n")

    nspin = len(mol.nelec)
    #print nspin
    #sys.exit()

    fodout = s.copy()

    for spin in range(nspin):
        print("Find Fermi-Orbitals for spin: {0}".format(spin))
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

        print("  total e: {0} (spin {1})".format(te,spin))
        #print("1s core: {0}".format(ne_1s))
        #print("valence: {0}".format(ve))

        # define which orbitals are use for initial boys
        pz_idx = np.arange(0,te)
        nfods = len(pz_idx)
        initial_fods = np.zeros((nfods,3), dtype=np.float64)
        #print pz_idx, len(pz_idx)

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
            print("  density fit for spin {0} orb #{1} ...".format(sstr,j+1))
            #print("find initial fod: {0}".format(j))
            initial_fods[j,:] = lorb2fod(m,loc_orb[:,j], grid_level=grid_level)
            fodout.extend(Atom(osym, position=initial_fods[j,:]*units.Bohr))


    io.write('fodout.xyz', fodout)

    #print np.array_str(initial_fods*units.Bohr, precision=4, max_line_width=120)



    sys.exit()

    from pyscf.tools import cubegen
    for j in range(pz_idx.shape[0]):
        cubegen.orbital(mol, 'boys_{0}.cube'.format(j), loc_orb[:,j],
            nx=50,ny=50,nz=50)
