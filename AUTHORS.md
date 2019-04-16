# main developers: 

Core routines, code structure and working examples.   

Lenz Fiedler,  M. Sc.  
email: fiedler.lenz@gmail.com  
- all original core FLO-SIC rountines (flosic_os.py, flosic_scf.py)   
- FOD visualization (e.g. cube files and density plots)  
- basic and advanced tests   
- documentation  
- NRLMOL2pyflosic routine    

Sebastian Schwalbe, M. Sc.  
email: theonov13@gmail.com 
- different SIC Hamiltonians (different unified Hamiltonians)   
- object-oriented aspects (e.g. class design for flosic_scf.py)   
- ase-pyflosic framework (ase_pyflosic_calculator.py, ase_pyflosic_optimizer.py)   
- automatic guessing  
- G2-1 benchmark post-processing (see pyG21 code)    
- DFO basis set (pyscf interface and testing with various codes)    
- pyflosic2NRLMOL routine (see pyNRLMOL code)   

Dr. Torsten Hahn      
email: hahn@physik.tu-freiberg.de     
- FOD forces   
- linearly scaling FLO-SIC SCF cycle and FOD forces   
- MPI paralellization   

# co-developers: 

Additional features, code testing and improvements.  

Prof. Dr. Jens Kortus    
- DFO basis set in gbs format (see utils/basis/nrlmol_dfo.gbs)  
- theoretical concepts   

Simon Liebing, M.Sc.    
- DFO basis set in gbs format (see utils/basis/nrlmol_dfo.gbs)  
- licence issues  

Dr. Kai Trepte  
email: trept1k@cmich.edu  
- intensive code testing  
- FOD guessing   
- benchmarking again NRLMOL code   
- porting to python3   
