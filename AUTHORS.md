# main developers: 

Core routines, code structure and working examples.   

#### Sebastian Schwalbe, M. Sc.  
email: theonov13@gmail.com 
- different SIC Hamiltonians (different unified Hamiltonians)   
- object-oriented aspects (e.g. class design for flosic_scf.py)   
- ase-pyflosic framework (ase_pyflosic_calculator.py, ase_pyflosic_optimizer.py)   
- PyCOM 
- G2-1 benchmark post-processing (see pyG21 code)    
- DFO basis set (pyscf interface and testing with various codes)    
- pyflosic2NRLMOL routine (see pyNRLMOL code)   

#### Kai Trepte, PhD  
email: kai.trepte1987@gmail.com
- intensive code testing  
- FOD guessing   
- benchmarking against NRLMOL code   
- porting to python3   

#### Jakob Kraus, M. Sc.
email: jakob.kraus@physik.tu-freiberg.de
- code testing
- application to large molecules
- combination with solvation models   
- ase-pyflosic framework (ase_pyflosic_calculator.py, ase_pyflosic_optimizer.py)     

#### Susi Lehtola, PhD 
email: susi.lehtola@helsinki.fi
- DFO/DFO+ basis testing, debugging 
- new concepts 
- theoretical guiding 
- code layout 
- recommendations for default settings (grid, unified Hamiltonian)


# co-developers: 

Additional features, code testing and improvements.  

#### Prof. Jens Kortus    
- DFO basis set in gbs format (see utils/basis/nrlmol_dfo.gbs)  
- theoretical concepts   

#### Simon Liebing, M. Sc.    
- DFO basis set in gbs format (see utils/basis/nrlmol_dfo.gbs)  
- licence issues  


# former developers:

#### Torsten Hahn, PhD     
email: torstenhahn@fastmail.fm    
- FOD forces   
- linearly scaling FLO-SIC SCF cycle and FOD forces   
- MPI paralellization   

#### Lenz Fiedler,  M. Sc.  
email: fiedler.lenz@gmail.com  
- all original core FLO-SIC rountines (flosic_os.py, flosic_scf.py)   
- FOD visualization (e.g. cube files and density plots)  
- basic and advanced tests   
- documentation  
- NRLMOL2pyflosic routine    
