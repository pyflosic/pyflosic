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
import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, ReadError
from ase.atom import Atom
from ase import Atoms 
from ase.units import Ha, Bohr, Debye 
try:
    from ase.atoms import atomic_numbers
except:
    # moved in 3.17 to
    from ase.data import atomic_numbers
import copy

class NRLMOL(FileIOCalculator):
    """ NRLMOL calculator.
            by Sebastian Schwalbe
        Notes: ase.calculators -> units [eV,Angstroem,eV/Angstroem]
           NRLMOL       -> units [Ha,Bohr,Ha/Bohr]                         
    """
    implemented_properties = ['energy', 'forces']
    NRLMOL_CMD = os.environ.get('ASE_NRLMOL_COMMAND')
    # command for NRLMOL executable 
    command =  NRLMOL_CMD
    # 
    default_parameters = dict(atoms = None,
                              fods = None,
                              SYMBOL = None,
                              ISYMGEN = None,
                              FRMORB = 'FRMORB',
                              basis = None,
                              e_up = None,
                              e_dn = None,
                              calc = 'ALL',
                              mode = 'SCF-ONLY',
                              xc = 'lda',
                              extra=0)

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=os.curdir, atoms=None, **kwargs):
        """ Constructor """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)
        valid_args = ('atoms','fods','SYMBOL','ISYMGEN','basis','e_up','e_dn','calc','mode','xc','extra','FRMORB')
        # set any additional keyword arguments
        for arg, val in self.parameters.items():
            if arg in valid_args:
                setattr(self, arg, val)
            else: 
                raise RuntimeError('unknown keyword arg "%s" : not in %s'% (arg, valid_args))
        # SYMBOL to atoms.object 
        #if atoms == None:
        #    self.read_symbol(SYMBOL=SYMBOL)
        #if atoms != None:
        #    self.atoms = atoms

    def set_atoms(self, atoms):
        self.atoms = copy.deepcopy(atoms)

    def set_label(self, label):
        self.label = label
        self.directory = label
        self.prefix = ''
        self.out = os.path.join(label, 'NRLMOL.OUT')

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore boundary conditions 
        # because NRLMOL has no pbc 
        if 'pbc' in system_changes:
            system_changes.remove('pbc')
        return system_changes

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
    
    def nrlmol2elements(self,element):
        # converts atomic numbers to nrlmol atomic symbols 
        elements = {'X':0,  'HYD':1,  'HEL':2,  'LIT':3,  'BER':4,
                    'BOR':5,  'CAR':6,  'NIT':7,  'OXY':8,  'FLU':9,
                    'NEO':10, 'SOD':11, 'MAG':12, 'ALU':13, 'SIL':14,
                    'PHO':15, 'SUL':16, 'CHL':17, 'ARG':18, 'POT':19,
                    'CAL':20, 'SCA':21, 'TIT':22, 'VAN':23, 'CHR':24,
                    'MAN':25, 'IRO':26, 'COB':27, 'NIC':28, 'COP':29,
                    'ZIN':30, 'GAL':31, 'GER':32, 'ARS':33, 'SEL':34,
                    'BRO':35, 'KRY':36, 'RUB':37, 'STR':38, 'YTR':39,
                    'ZIR':40, 'NIO':41, 'MOL':42, 'TEC':43, 'RHU':44,
                    'RHO':45, 'PAL':46, 'SLV':47, 'CAD':48, 'IND':49,
                    'TIN':50, 'ANT':51, 'TEL':52, 'IOD':53, 'XEN':54,
                    'CES':55, 'BAR':56}
        return elements[element]

    def element2nrlmol(self,element):
        # converts nrlmol atomic symbols to atomic numbers
        elements  = {'0':'X'   ,  '1':'HYD',  '2':'HEL',  '3':'LIT',  '4':'BER',
                     '5':'BOR' ,  '6':'CAR',  '7':'NIT',  '8':'OXY', ' 9':'FLU',
                     '10':'NEO', '11':'SOD', '12':'MAG', '13':'ALU', '14':'SIL',
                     '15':'PHO', '16':'SUL', '17':'CHL', '18':'ARG', '19':'POT',
                     '20':'CAL', '21':'SCA', '22':'TIT', '23':'VAN', '24':'CHR',
                     '25':'MAN', '26':'IRO', '27':'COB', '28':'NIC', '29':'COP',
                     '30':'ZIN', '31':'GAL', '32':'GER', '33':'ARS', '34':'SEL',
                     '35':'BRO', '36':'KRY', '37':'RUB', '38':'STR', '39':'YTR',
                     '40':'ZIR', '41':'NIO', '42':'MOL', '43':'TEC', '44':'RHU',
                     '45':'RHO', '46':'PAL', '47':'SLV', '48':'CAD', '49':'IND',
                     '50':'TIN', '51':'ANT', '52':'TEL', '53':'IOD', '54':'XEN',
                     '55':'CES', '56':'BAR'}
        return elements[element]

    def read_symbol(self,SYMBOL):
        # read NRLMOL SYMBOL input and convert it to ase.atoms object 
        f = open(SYMBOL,'r')
        ll = f.readlines()
        f.close()
        atoms = Atoms()
        for l in range(len(ll)):
            if ll[l].find('ALL') != -1 or ll[l].find('BHS') != -1:
                tmp = ll[l].split()
                # atom = px py pz s
                atom = tmp[0].split('-')[-1][0:3]
                px = float(tmp[2])*Bohr
                py = float(tmp[3])*Bohr
                pz = float(tmp[4])*Bohr 
                s  = tmp[5]
                a = Atom(self.nrlmol2elements(atom),[px,py,pz])
                atoms.append(a)
        self.atoms = atoms     
                 
    def nrlmol_xc(self,xc):
        # convert xc tags to nrlmol xc functionals 
        name_xc = {'lda':'LDA-PW91*LDA-PW91',
                   'pbe':'GGA-PBE*GGA-PBE',
                   'scan':'GGA-SCAN*GGA-SCAN'}
        return name_xc[xc]
        
    def write_symbol(self,atoms,e_up,e_dn,calc='ALL',mode='SCF-ONLY',xc='lda',extra=0):
        # write NRLMOL SYMBOL file based on ase.atoms object 
        xc_name = self.nrlmol_xc(xc)
        natoms  = len(atoms.get_chemical_symbols())
        o = open('SYMBOL','w')
        o.write(mode+'\n')
        o.write(xc_name+'\n')
        o.write('OLDMESH'+'\n')
        o.write('1    NUMBER OF SYMBOLIC FILES\n')
        o.write('ISYMGEN = INPUT\n')
        o.write('  %d  NUMBER OF SYMBOLS IN LIST\n' % (int(natoms + 2))) 
        o.write('  %d  NUMBER OF NUCLEI\n' % (int(natoms)))
        for i in range(natoms):
            o.write('1.0  1 1 1\n')
        o.write('  1  CALCULATION SETUP BY ISETUP\n')
        count = dict()
        for n in range(natoms):
            a = atoms[n]
            atom_name = str(atomic_numbers[a.symbol])
            pos = a.position
            # NRLMOL uses atomic units  
            px = pos[0]/Bohr
            py = pos[1]/Bohr
            pz = pos[2]/Bohr
            #if count.has_key(atom_name) == True:
            # has_key is not available in python3
            if atom_name in count:
                e_tmp = {atom_name : count[atom_name] + 1}
            #if count.has_key(atom_name) == False:
            # has_key is not availanle in python3
            if atom_name not in count:
                e_tmp = {atom_name : 1} 
            count.update(e_tmp)            
            count_str = '%d' %  count[atom_name]
            count_str = count_str.zfill(3)
            #o.write(calc+'-'+self.element2nrlmol(atom_name)+count_str+' = '+str('%0.7f' % px)+' '+str('%0.7f' % py)+' '+str('%0.7f' % pz)+' UPO\n') 
            o.write(calc+'-'+self.element2nrlmol(atom_name)+count_str+' = '+str('%.8s' % ('%.7f' % float(px)))+' '+str('%.8s' % ('%.7f' % float(py)))+' '+str('%.8s' % ('%.7f' % float(pz)))+' UPO\n')
        o.write('ELECTRONS  = '+str('%0.6f' % e_up)+' '+str('%0.6f' % e_dn)+'\n')
        o.write('EXTRABASIS = %d\n' % (extra))     
        o.close()
        self.elements = count

    def write_input(self, atoms, properties=None, system_changes=None, **kwargs):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        # writes NRLMOL input for each calculation 
        # WARNING: OS operation 
        #       We use the SCF-ONLY mode of NRLMOL, because all other operations are 
        #       controlled by ase or this calculator. Thus, we have every time 
        #       only one geometry in the SYMBOL file and start from this. 
        #       Thus the RUNS and GEOCNVRG need to be deleted.  
        files = ['RUNS','GEOCNVRG','FRMIDT','FRMORB','FRMGRP']
        for f in files:
            if os.path.exists(f) == True:
                os.remove(f)
        #try:
        #    os.remove('RUNS')
        #    os.remove('GEOCNVRG')
        #    # this is a hack, if we want to calculate NUC forces for the 
        #    # flosic optimizer we need to delete FRMIDT and FRMGRP 
        #    if self.fods == None:
        #        os.remove('FRMIDT')
        #        os.remove('FRMORB')
        #        os.remove('FRMGRP')
        #except: 'Nothing'
        #self.read_symbol(SYMBOL)
        self.write_symbol(atoms=atoms,e_up=self.e_up,e_dn=self.e_dn,calc=self.calc,mode=self.mode,xc=self.xc,extra=self.extra)
        self.write_isymgen(calc=self.calc,elements=self.elements,basis=self.basis)
        if self.fods != None:
            self.write_fods(fods=self.fods,e_up=self.e_up,e_dn=self.e_dn)
        self.initialize(atoms)
        
    def initialize(self, atoms):
        tmp =  'Somehow needs to there.'

        #def calculate_energy(self, atoms):
        #        s = '# execute command'

    def read_energy(self,version=2):
        if version == 1:
            # read total energy from NRLMOL FRCOUT 
            f = open(os.path.join(self.directory, 'FRCOUT'), 'r')
            ll = f.readlines()
            # Unit conversion NRLMOL [Ha] to ase.calculator [eV] 
            etot = float(ll[0].split()[0])*Ha
            f.close()
            self.results['energy'] = etot
        if version == 2: 
            # read total energy from NRLMOL SUMMARY 
            f = open(os.path.join(self.directory, 'SUMMARY'), 'r')
            ll = f.readlines()
            # Unit conversion NRLMOL [Ha] to ase.calculator [eV] 
            etot = float(ll[-1].split()[-2])*Ha
            f.close()
            self.results['energy'] = etot

    # SS until here clean?
    def read_forces(self):
        # read forces from general NRLMOL.OUT output 
        # because easier to grep 
        lines = open(os.path.join(self.directory, 'NRLMOL.OUT'), 'r').readlines()
        forces = []
        for line in range(len(lines)):
            if lines[line].rfind('TOTAL:  ') > -1:
                f = lines[line].split()
        # unit conversion from NRLMOL forces [Ha/Bohr] -> ase forces [eV/Ang]
        fx = float(f[1].replace('D','E'))*Ha/Bohr
        fy = float(f[2].replace('D','E'))*Ha/Bohr
        fz = float(f[3].replace('D','E'))*Ha/Bohr
        forces.append(np.array([fx,fy,fz]))
        self.results['forces'] = np.array(forces)

    def get_energy(self):
        self.read_energy()
        return self.results['energy']

    def get_forces(self,atoms):
        # calculates forces if required 
        if self.calculation_required(atoms,'energy'):
            self.calculate(atoms)
        self.read_forces()
        return self.results['forces']

    def read_convergence(self):
        # check if the calculation is converged or not 
        converged = False
        text = open(os.path.join(self.directory, 'NRLMOL.OUT'), 'r').read()
        if ('SELF-CONSISTENCY REACHED, CALCULATING FORCES' in text):
            converged = True
        return converged

        #def get_magnetic_moment(self, atoms=None):
        #        self.magnetic_moment = 0

        #def get_magnetic_moments(self, atoms):
                # not implemented yet, so
                # so set the total magnetic moment on the atom no. 0 and fill with 0.0
        #        magmoms = [0.0 for a in range(len(atoms))]
        #        magmoms[0] = self.get_magnetic_moment(atoms)
        #        return np.array(magmoms)
    
    def calculation_required(self, atoms, properties):
        # checks of some properties need to be calculated or not 
        system_changes = self.check_state(atoms)
        if system_changes:
            return True
        for name in properties:
            if name not in self.results:
                return True
        return False

    def get_potential_energy(self, atoms, force_consistent=False):
        # calculate total energy if required 
        if self.calculation_required(atoms,'energy'):
            self.calculate(atoms)
        self.read_energy()
        return self.results['energy']

    def write_isymgen(self,calc,elements,basis):
        # writes NRLMOL ISYMGEN file 
        # search tags 
        tag1 = 'BEGIN-'
        tag2 = 'END-'
        f = open(basis,'r')
        o = open('ISYMGEN','w')
        ll = f.readlines()
        f.close()
        START = []
        END = []
        #name1 = elements.keys()
        # in python3 dct.keys() returns no list 
        name1 = list(elements.keys())
        name  = []
        elements_new = dict()
        # in the input elements dict there only numbers (e.g. {7:2})
        # for this function we need (e.g. {'ALL-NIT':2}). 
        # Thus this loop creates this dict. 
        for n1 in range(len(name1)):
            tmp_n = calc+'-'+self.element2nrlmol(str(name1[n1]))
            tmp_v = elements[name1[n1]]
            name.append(tmp_n)
            tmp_d ={tmp_n : tmp_v}
            elements_new.update(tmp_d)
        elements = elements_new
        for e in range(len(name)):
            for l in range(len(ll)):
                if ll[l].find(tag1+name[e]) != -1:
                    idx_start = l + 1
                    START.append(idx_start)
                if ll[l].find(tag2+name[e]) != -1:
                    idx_end = l #-1 
                    END.append(idx_end)
        o.write('   %d          TOTAL NUMBER OF ATOM TYPES\n' % int(len(name)))
        for s in range(len(START)):
            n = int(self.nrlmol2elements(name[s].split('-')[-1]))
            ele  = name[s].split('-')[-1]
            calc = name[s].split('-')[0]
            o.write('   %d   %d      ELECTRONIC AND NUCLEAR CHARGE\n' %(n,n))
            o.write('%s           %s-ELECTRON ATOM TYPE\n' % (calc,calc))
            o.write('   %d          NUMBER OF ATOMS OF TYPE %s\n' % (elements[name[s]],ele))
            for c in range(elements[name[s]]):
                count = str(c+1)
                count = count.zfill(3)
                o.write(name[s]+count+'\n')
            for b in range(START[s],END[s]):
                o.write(ll[b])
        o.write('ELECTRONS\n')
        o.write('WFOUT\n')
        o.close()

    def write_fods(self,fods,e_up,e_dn):
        # fods ... ase.atoms object containing only spin up and spin down fods
        # Note: No summetry supported! 
        o1 = open('FRMGRP','w')
        o1.write('           1\n')
        o1.write('         1.0000000000        0.0000000000        0.0000000000\n')
        o1.write('         0.0000000000        1.0000000000        0.0000000000\n')
        o1.write('         0.0000000000        0.0000000000        1.0000000000\n')
        o1.close()
        o2 = open(self.FRMORB,'w')
        o2.write('         %d           %d\n' %(int(abs(e_up)),int(abs(e_dn))))        
        for f in fods: 
            pos = f.position
            px = pos[0]/Bohr
            py = pos[1]/Bohr
            pz = pos[2]/Bohr
            o2.write(str(px)+' '+str(py)+' '+str(pz)+'\n')
        o2.close() 
    
    def read_fodforces(self):        
        f = open('fforce.dat','r')
        ll = f.readlines()
        f.close()    
        fodforces = []
        for l in range(len(ll)):
            tmp_f = ll[l].split()
            fx = float(tmp_f[0])*Ha/Bohr
            fy = float(tmp_f[1])*Ha/Bohr
            fz = float(tmp_f[2])*Ha/Bohr
            fodforces.append(np.array([fx,fy,fz]))
        self.results['fodforces'] = np.array(fodforces)
    
    def get_fodforces(self, atoms):
        # calculate fodforces if required 
        if self.calculation_required(atoms,'energy'):
            self.calculate(atoms)
        self.read_fodforces()
        return self.results['fodforces']
    
    def read_dipole_moment(self):
        f = open('DIPOLE','r')
        ll = f.readlines() 
        f.close()
        tmp_D = ll[0].split()
        Dx = float(tmp_D[0].replace('D','E'))*Bohr #/0.393456 #*Debye
        Dy = float(tmp_D[1].replace('D','E'))*Bohr #/0.393456 #*Debye
        Dz = float(tmp_D[2].replace('D','E'))*Bohr #/0.393456 #*Debye
        D = [Dx,Dy,Dz]
        D = np.array(D)
        self.results['dipole'] = D 

    def get_dipole_moment(self,atoms):
        # calculate dipole moment if required 
        if self.calculation_required(atoms,'energy'):
            self.calculate(atoms)
        self.read_dipole_moment()
        return self.results['dipole']

if __name__ == "__main__":
    
    atoms = Atoms('LiH',[[0,0,0.5],[0,0,-0.5]])
    e_up =  2.0 
    e_dn = -2.0
    SYMBOL = './SYMBOL'
    ISYMGEN = './ISYMGEN'
    a = NRLMOL(SYMBOL=SYMBOL,ISYMGEN=ISYMGEN,e_up=e_up,e_dn=e_dn)
    a.calculate() 
    print(a.get_energy())
    print(a.get_forces())
    a = NRLMOL(atoms=atoms,ISYMGEN=ISYMGEN,e_up=e_up,e_dn=e_dn)
    a.write_input(atoms=atoms)
    a.calculate()
    print(a.get_energy())
    print(a.get_forces())
    

