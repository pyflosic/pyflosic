from ase.io import read,write
import argparse
import numpy as np
from ase.atoms import Atoms
def xyz_to_nuclei_fod(ase_atoms):

    # Get the number of atoms and fods.

    syssize = len(ase_atoms)

    # First the preparations for enabling the usage of GHOST atoms are done.
    # We need to determine if a FOD is included in the creation of the GHOST atoms or 
    # not. If two GHOST atoms are created very close to each other (or other atoms) this 
    # will result in the DFT calculation crashing. A FOD is included if it is no closer 
    # then fodeps to another FOD or a nuclei.

    fodeps = 0.1
    included = []
    numbernotincluded = 0
    dist = -1.0

    # Iterate over FODs and nuclei positions.

    for i in range(0,syssize):

        # Nuclei are always included, obviously. NOTE: FODs for spin0 are given with an 
        # X, the for spin1 with an He symbol.

        if ase_atoms[i].symbol != 'X' and ase_atoms[i].symbol != 'He':
            included.append(1)
        if ase_atoms[i].symbol == 'X' or ase_atoms[i].symbol == 'He':

        # For every FOD the distance to every included FOD and the nuclei is calculated 
        # in order to determine whether or not it has to be included.

            distold = fodeps
            for j in range(0,i):
                dist = np.sqrt((ase_atoms[i].position[0]-ase_atoms[j].position[0])**2+(ase_atoms[i].position[1]-ase_atoms[j].position[1])**2+(ase_atoms[i].position[2]-ase_atoms[j].position[2])**2)
                if dist < distold and included[j]==1:
                    distold = dist

        # If the smallest distance is smaller than fodeps, the FOD will not be included. 
        # Elsewise it is included.

            if distold < fodeps:
                included.append(0)
                numbernotincluded += 1
            else:
                included.append(1)

    # Now the actual splitting is done.
    # These arrays will hold nuclei and FODs (for each spin channel separately).


    nuclei = Atoms([])
    fod1 = Atoms([])
    fod2 = Atoms([])
    nrofnuclei = 0

    # tmp will be used to create the list that can be used to enable GHOST atoms.

    tmp = []

    # Determine where to split.

    for i in range(0,syssize):
        if ase_atoms[i].symbol != 'X':
            nrofnuclei = nrofnuclei + 1

        # If the first FOD is found, nuclei assigning will be done.

        if ase_atoms[i].symbol == 'X':
            break

    # Split everything. 

    for i in range(0,syssize):

        # Assign the nuclei.

        if i < nrofnuclei:
            tmp.append(ase_atoms[i].symbol+' '+str(ase_atoms[i].position[0])+' '+str(ase_atoms[i].position[1])+' '+str(ase_atoms[i].position[2])+'\n')
            nuclei.append(ase_atoms[i])

        # Assing FODs for spin0.    

        elif ase_atoms[i].symbol == 'X':
            fod1.append(ase_atoms[i])
            if included[i] == 1:
                tmp.append('ghost1'+' '+str(ase_atoms[i].position[0])+' '+str(ase_atoms[i].position[1])+' '+str(ase_atoms[i].position[2])+'\n')

        # Assign FODs for spin1.        

        elif ase_atoms[i].symbol == 'He':
            fod2.append(ase_atoms[i])
            if included[i] == 1:
                tmp.append('ghost2'+' '+str(ase_atoms[i].position[0])+' '+str(ase_atoms[i].position[1])+' '+str(ase_atoms[i].position[2])+'\n')

    # geo holds both nuclei and GHOST atoms at FOD positions.        

    geo = ''.join(tmp)
    # geo can now be used for calculation with GHOST atoms, nuclei and fod1/fod2 for 
    # every other calculation.

    return geo,nuclei,fod1,fod2,included


def get_on_opt(f_nuc,f_fod1,f_fod2):
    nuc = read(f_nuc)
    fod1 = read(f_fod1)
    fod2 = read(f_fod2)
    geo,nuclei,init_fod1,init_fod2,included =xyz_to_nuclei_fod(nuc)

    #write('final0.xyz',fod1,'xyz')
    #write('final1.xyz',fod2,'xyz')
    #print(nuclei.get_positions())
    #print(fod1.get_positions())

    comb = Atoms([])
    for n in nuclei:
        comb.append(n)
    for f1 in fod1:
        comb.append(f1)
    for f2 in fod2:
        comb.append(f2)
    write('final_ON.xyz',comb,'xyz')


def on_argparser():
        # Command line parsing for the todo package. 
        parser = argparse.ArgumentParser(description='Commandline argparser for ON optimized structure')
        parser.add_argument('-nuc', metavar='nuc',type=str, nargs='+', help='nuclei geometry')
        parser.add_argument('-fod1', metavar='fod1',type=str, nargs='+', help='fod1 geometry')
        parser.add_argument('-fod2', metavar='fod2',type=str, nargs='+', help='fod2 geometry')

        # Parse all command line arguments      
        args = parser.parse_args()
        print(args)
        # Load a toto class instance 
        if args.nuc is not None:
                f_nuc = args.nuc[0]
        if args.fod1 is not None:
                f_fod1 = args.fod1[0]
        if args.fod2 is not None:
                f_fod2 = args.fod2[0]
        get_on_opt(f_nuc,f_fod1,f_fod2)

if __name__ == "__main__":

        on_argparser()
