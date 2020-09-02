# Run script for Pyflosic examples by LF.
# Type in the example file you want to run.

# Specify your PySCF path here.

#export PYTHONPATH=/home/SHARE/for_sebastian/pyscf_sebastian/pyscf/:$PYTHONPATH
#export PYTHONPATH=/home/schwalbe/__Programms__/pyscf_v3/pyscf/:$PYTHONPATH

# Pyflosic path.
pyflosic="$(dirname "$(pwd)")"
echo $pyflosic
export PYTHONPATH=$pyflosic/src/:$PYTHONPATH
export PYTHONPATH=$pyflosic/utils/:$PYTHONPATH

python3 run_test.py
