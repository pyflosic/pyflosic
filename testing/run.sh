
# Specify your PySCF path here.

#export PYTHONPATH=/home/SHARE/for_sebastian/pyscf_sebastian/pyscf/:$PYTHONPATH
#export PYTHONPATH=/home/schwalbe/__Programms__/pyscf_v3/pyscf/:$PYTHONPATH

# Pyflosic path.
cd ..
pyflosic="$(pwd)"
cd testing

export PYTHONPATH=$pyflosic/src/:$PYTHONPATH
export PYTHONPATH=$pyflosic/utils/:$PYTHONPATH

python3 run_test.py
