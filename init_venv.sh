#!/bin/bash
pip3 install virtualenv
virtualenv -p /usr/bin/python3.7 venv
source venv/bin/activate
pip3 install ase==3.17.0
pip3 install pyscf==1.7.1
pip3 install numba==0.48.0
