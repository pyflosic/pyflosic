from ase.units import Hartree

# Links 
# http://cccbdb.nist.gov/introx.asp

def bonding_energy(elements,system,input_unit='Hartree'):
        eb = 0
        for e in elements:
                eb = eb + e
        eb = eb - system
        # units conversion from Hartree to eV 
        if input_unit == 'Hartree':
                eb = eb *Hartree
        return eb

# SYSTEM   IT          TRACE              EDFT                  EKIN+ENONLOC          CHARGE             EDFT+SIC          LOWEST
# Si	   25       -175.876332706       -289.224747603        288.352899978         13.999999999       -289.097783990       -289.097783155
# H	    4         -0.499658860         -0.499359706          0.499680555          0.999999999         -0.499921851         -0.499921856
# Si2H6	    3       -355.252726935       -582.269513361        580.695534111         34.000000830       -582.054252014       -582.054252301
	
atom1 = -289.097783990 
atom2 = -0.499921851 
mol = -582.054252014 

elements = [atom1,atom1,atom2,atom2,atom2,atom2,atom2,atom2]
system = mol
eb = bonding_energy(elements,system)
print 'E_{B,PBE} = '+str(eb)+' eV'
e_ref = 22.05206
print 'E_{B,CCCBDB} = '+str(e_ref)+' eV'

