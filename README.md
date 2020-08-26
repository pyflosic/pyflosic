[![DOI](https://zenodo.org/badge/185756742.svg)](https://zenodo.org/badge/latestdoi/185756742)
![GitHub Logo](/images/pyflosic_logo.png)


# PyFLOSIC   
### Python-based Fermi-Löwdin orbital self-interaction correction (FLO-SIC)

  
[![license](https://img.shields.io/badge/license-APACHE2-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![language](https://img.shields.io/badge/language-Python3-blue)](https://www.python.org/)
[![version](https://img.shields.io/badge/version-1.0.2-lightgrey)]()  


[![researchgate](https://img.shields.io/static/v1?label=researchgate&message=OpenSIC&style=social&logo=researchgate)](https://www.researchgate.net/project/Fermi-Loewdin-orbital-self-interaction-correction-developed-in-Freiberg-FLO-SICFG)
[![youtube](https://img.shields.io/static/v1?label=YouTube&message=OpenSIC&logo=youtube&style=social)](https://www.youtube.com/watch?v=-1bxmCwn7Sw)
[![twitter](https://img.shields.io/static/v1?label=twitter&message=OpenSIC&style=social&logo=twitter)](https://twitter.com/OpenSIC_project)

### PyFLOSIC structure

- *INSTALL.md*: Installation guide. 
- *CONTRIBUTING.md*: The main developers and co-developers of PyFLOSIC along with contact information.
- *ALPHA_VERSIONS.md*: A list of pre-GitHub PyFLOSIC code versions.
- *LICENSE*: APACHE2 license file.
- *doc/*: It is strongly recommended to read this before using PyFLOSIC.
- *src/*: Contains the source files for PyFLOSIC.
- *examples/*: Contains useful examples to get started with PyFLOSIC. 
- *utils/*: Contains utilities that are useful for PyFLOSIC.
- *testing/*: Contains a version control test.


The self-interaction error (SIE) is one of the mayor drawbacks of one of the most widley used electronic structure methods - density functional theory (DFT). Pederson et al. proposed a unitarily invariant and numerically feasible method based on Fermi-Löwdin orbital self-interaction correction (FLO-SIC). We implemented this method using the highly modular PySCF electronic structure and quantum chemistry code as basis.   

### Theoretical foundations of FLO-SIC
* Mark R. Pederson, Adrienn Ruzsinszky, and John P. Perdew. Communication: Self-interaction correction with unitary invariance in density functional theory. The Journal of Chemical Physics, 140(12):121103, March 2014.
* Zeng-hui Yang, Mark R. Pederson, and John P. Perdew. Full self-consistency in the Fermi-orbital self-interaction correction. Physical Review A, 95(5):052505, May 2017.  

## PyFLOSIC dependencies (required)
We recommend the following package versions: 

* ase 3.17.0
* h5py 2.10.0
* numba 0.48.0
* scipy 1.5.2
* numpy 1.19.1
* pyscf 1.7.1

## PyFLOSIC dependencies (optional)

* matplotlib 
* python3-tk
* pyberny

## PyFLOSIC installation
See [INSTALL.md](/INSTALL.md). 

## PyFLOSIC documentation
See [doc/pyflosic_manual.pdf](/doc/pyflosic_manual.pdf).

## PyFLOSIC authors 

### current
* Sebastian Schwalbe (theonov13@gmail.com) 
* Lenz Fiedler (fiedler.lenz@gmail.com)
* Jakob Kraus (jakob.kraus@physik.tu-freiberg.de) 
* Kai Trepte (kai.trepte1987@gmail.com)
* Susi Lehtola (susi.lehtola@helsinki.fi)
* Jens Kortus (jens.kortus@physik.tu-freiberg.de)

### former 
* Torsten Hahn (torstenhahn@fastmail.fm)

The development of PyFLOSIC started with the master's thesis of Lenz Fiedler.  
Over the last year, we had many updates and complete code rewrites by Sebastian Schwalbe (ase-backends, classes etc.) and Torsten Hahn (various speed-up techniques). Our main testers are Kai Trepte, Sebastian Schwalbe, and Jakob Kraus. Jens Kortus is our overall theoretical guide and head of decisions, whereas Susi Lehtola has recently joined the team and provides new perspectives.  
For more details on the individual contributions of the PyFLOSIC main developers and co-developers, see [here](CONTRIBUTING.md).

## Citing PyFLOSIC
If you use the PyFLOSIC code for a scientific article or contribution, please cite the following article: 

* **PyFLOSIC: Python-based Fermi-Löwdin orbital self-interaction correction**  
  Sebastian Schwalbe, Lenz Fiedler, Jakob Kraus, Jens Kortus, Kai Trepte, and Susi Lehtola,    
  Journal of Chemical Physics 153:084104, August 2020. [DOI:10.1063/5.0012519](https://doi.org/10.1063/5.0012519) [arXiv:1905.02631](https://arxiv.org/abs/1905.02631)
  
Applications of the PyFLOSIC code:     
* [Interpretation and Automatic Generation of Fermi‐Orbital Descriptors](https://onlinelibrary.wiley.com/doi/full/10.1002/jcc.26062)
* [Stretched or noded orbital densities and self-interaction correction in density functional theory](https://aip.scitation.org/doi/10.1063/1.5087065)
  
## PyFLOSIC Milestones 
  * **April 30, 2020**   
Thanks to all the people who have read our article! We are very happy that our Journal of Computational Chemistry (JCC) article "Interpretation and Automatic Generation of Fermi‐Orbital Descriptors", in which every result was produced using our PyFLOSIC code, was promoted to one of the top 10% downloaded articles. [twitter](https://twitter.com/theonov13) 
  * **August 24, 2020**  
We are glad to announce that our PyFLOSIC article (see above) is now finally published in the Journal of Chemical Physics (JCP)! 

## Some remarks for PyFLOSIC users
PyFLOSIC is still a relatively young code subject to regular significant changes.  
If you discover any issues while working with PyFLOSIC, feel free to contact us or raise an issue [here](https://github.com/pyflosic/pyflosic/issues). 
