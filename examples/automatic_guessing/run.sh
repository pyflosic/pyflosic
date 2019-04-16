# Run script for Pyflosic examples by LF.
# Type in the example file you want to run.

# Specify your PySCF path here.

export PYTHONPATH=/home/SHARE/for_sebastian/pyscf_sebastian/pyscf/:$PYTHONPATH

# Pyflosic path.
cd ..
pyflosic="$(dirname "$(pwd)")"
cd automatic_guessing
export PYTHONPATH=$pyflosic/src/:$PYTHONPATH
export PYTHONPATH=$pyflosic/utils/:$PYTHONPATH

python3 $1
