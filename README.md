![GitHub Logo](/images/pyflosic_logo.png)


# pyflosic 
Python-based Fermi-Löwdin orbital self-interaction-correction  
Coding language: python3   
Licence: apache2   

The self-interaction error (SIE) is one of the mayor drawbacks of one of the most widley used electronic structure methods - density functional theory (DFT). Pederson et al. proposed a unitarily invariant and numerically feasible method based on Fermi-Löwdin orbital self-interaction correction (FLO-SIC). We implemented this method using the modern PySCF electronic structure code as basis.   

#### Theoretical foundation  
* Mark R. Pederson, Adrienn Ruzsinszky, and John P. Perdew. Communication: Self-interaction correction with unitary invariance in density functional theory. The Journal of Chemical Physics, 140(12):121103, March 2014.
* Zeng-hui Yang, Mark R. Pederson, and John P. Perdew. Full self-consistency in the Fermi-orbital self-interaction correction. Physical Review A, 95(5):052505, May 2017.  

## Installation 
You need a working pyscf installation on your system. 

## Dependencies (required)

* pyscf 
* h5py 
* scipy 
* numpy 
* ase 
* numba 

## Dependencies (optional)

* pyberny


## Note:heavy_exclamation_mark:
Please note that the code is in the open beta testing phase now. If you discover any problem while working with the code, please do not hesitate to contact one of the developers.      

## Authors 

* Lenz Fiedler 
* Sebastian Schwalbe (theonov13@gmail.com)  
* Torsten Hahn  
* Kai Trepte (trept1k@cmich.edu) 
* Jakob Kraus (jakob.kraus@student.tu-freiberg.de) 
* Jens Kortus 

The development of pyflosic started with the master thesis of Lenz Fiedler. Over the last year, we had many updates and complete code re-writes by Sebastian Schwalbe (ase-backends,classes etc.) and Torsten Hahn (various speed-up techniques). Our main testers are Kai Trepte and Sebastian Schwalbe. Our overall theoretical guide and head of decisions is Prof. Jens Kortus.

## Citation
If you use the pyflosic code for a scientific article or contribution, please cite the following article. 
