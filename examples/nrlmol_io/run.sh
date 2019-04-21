# Run script for Pyflosic examples by LF.
# Type in the example file you want to run.

# Specify your PySCF path here.

export PYTHONPATH=/home/SHARE/for_sebastian/pyscf_sebastian/pyscf/:$PYTHONPATH

# for ase nrlmol calculator 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/sw/intel/intel2015/composer_xe_2015.3.187/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/sw/gfortran/lib64
export LD_LIBRARY_PATH=/global/sw/libxc/lib/:$LD_LIBRARY_PATH
# NRLMOL path 
JOB_FILE_SCF=/home/schwalbe/00_RUNNING/G2_1/code/perfectGGA_lenz/mpnrlmol.mpi

export ASE_NRLMOL_COMMAND="mpirun -n 2 $JOB_FILE_SCF 1> NRLMOL.OUT 2> NRLMOL.OUT"


# Pyflosic path.
cd ..
pyflosic="$(dirname "$(pwd)")"
cd nrlmol_io
export PYTHONPATH=$pyflosic/src/:$PYTHONPATH
export PYTHONPATH=$pyflosic/utils/:$PYTHONPATH

python3 $1
