module load  modenv/scs5 pyscf/1.6.0-intel-2018a-Python-3.6.4
export PYTHONPATH=/home/sschwalb/__Programms__/ase/:$PYTHONPATH
#export PYTHONPATH=/home/jakr872a/src/ase/:$PYTHONPATH
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_sequential.so:$MKLROOT/lib/intel64/libmkl_core.so
python3 ../nrlmol_basis.py
