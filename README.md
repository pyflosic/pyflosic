[![DOI](https://zenodo.org/badge/185756742.svg)](https://zenodo.org/badge/latestdoi/185756742)
![GitHub Logo](/images/pyflosic_logo.png)


# PyFLOSIC 
Python-based Fermi-Löwdin orbital self-interaction-correction  
Coding language: python3   
Licence: APACHE2   

The following document will guide you through the setup and show you how to get started with this Python-based FLO-SIC code.

#### Contents

- *INSTALL.md*: Installation guide in Markdown (md) language. For older installations see INSTALL_OLD.  
- *AUTHORS.md*: The authors of PyFLOSIC along with contact information.
- *VERSION.md*: The version of PyFLOSIC you have acquired.
- *doc/*: It is strongly recommended to read this before using PyFLOSIC.
- *src/*: Contains the source files for PyFLOSIC.
- *examples/*: Contains useful examples to get started with PyFLOSIC. 
- *utils/*: Utilities that are useful for PyFLOSIC.
- *test/*: Contains a version control test.
- *LICENCE*: APACHE2 licence file.

Please see the PyFLOSIC manual in the folder doc for a detailed introduction.

The self-interaction error (SIE) is one of the mayor drawbacks of one of the most widley used electronic structure methods - density functional theory (DFT). Pederson et al. proposed a unitarily invariant and numerically feasible method based on Fermi-Löwdin orbital self-interaction correction (FLO-SIC). We implemented this method using the modern PySCF electronic structure code as basis.   

#### Theoretical foundations  
* Mark R. Pederson, Adrienn Ruzsinszky, and John P. Perdew. Communication: Self-interaction correction with unitary invariance in density functional theory. The Journal of Chemical Physics, 140(12):121103, March 2014.
* Zeng-hui Yang, Mark R. Pederson, and John P. Perdew. Full self-consistency in the Fermi-orbital self-interaction correction. Physical Review A, 95(5):052505, May 2017.  

## Installation 
You need a working pyscf installation on your system. 

## Dependencies (required)
We recommend the following package versions. 

* ase 3.17.0
* h5py 2.9.0
* numba 0.43.1
* scipy 1.2.1
* numpy 1.16.2
* pyscf 1.6.1

## Dependencies (optional)

* matplotlib 
* python3-tk
* pyberny


## Note:heavy_exclamation_mark:
Please note that the code is in the open beta testing phase now. If you discover any problem while working with the code, please do not hesitate to contact one of the developers.      

## Authors 

* Lenz Fiedler (fiedler.lenz@gmail.com)
* Sebastian Schwalbe (theonov13@gmail.com)  
* Torsten Hahn (torstenhahn@fastmail.fm)
* Kai Trepte (trept1k@cmich.edu) 
* Jakob Kraus (jakob.kraus@physik.tu-freiberg.de) 
* Jens Kortus (Jens.Kortus@physik.tu-freiberg.de)

The development of PyFLOSIC started with the master thesis of Lenz Fiedler. Over the last year, we had many updates and complete code re-writes by Sebastian Schwalbe (ase-backends,classes etc.) and Torsten Hahn (various speed-up techniques). Our main testers are Kai Trepte, Sebastian Schwalbe and Jakob Kraus. Our overall theoretical guide and head of decisions is Prof. Jens Kortus.

## Citation
If you use the PyFLOSIC code for a scientific article or contribution, please cite the following article. 

* **PyFLOSIC - Python based Fermi-Löwdin orbital self-interaction correction**  
  S. Sebastian, L. Fiedler. T. Hahn, K. Trepte, J. Kraus, J. Kortus  
  arXiv e-prints, Physics - Computational Physics, 2019, arXiv:1905.02631  
