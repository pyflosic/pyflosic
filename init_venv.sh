#!/bin/bash
pip3 install virtualenv
virtualenv -p /usr/bin/python3.7 venv
source venv/bin/activate
pip3 install ase==3.17.0
pip3 install h5py==2.10.0
pip3 install numba==0.48.0
pip3 install scipy==1.5.2
pip3 install numpy==1.19.1
pip3 install pyscf==1.7.1
echo 'To deactivate the virtualenv environment, simply type "deactivate" in your command line.'
