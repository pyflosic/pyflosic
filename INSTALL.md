Date: August 25th, 2020      
To install the code with all its dependencies, simply type    

```bash
$ source init_venv.sh
```

All Python packages. including PySCF, are installed using pip.      
For the deactivation of the environment simply type    

```bash
$ deactivate
```

After the first installation, the environment can alternativly started     
```bash
$ source venv/bin/activate 
```
while the ending procedure is the same. 

Date: 16.04.2019  
Installation checked on:  

- Ubuntu 18.10 (Cosmic Cuttlefish)
- Elementary OS 0.4.1 Loki
- Debian GNU/Linux 9.8 (stretch)
- Debian 10 (buster)

# PySCF installation 

```bash
 $ git clone https://github.com/sunqm/pyscf  
 $ cd pyscf/pyscf/lib   
 $ mkdir build   
 $ cd build 
 $ vi ../CMakeLists.txt
 ```

We do not recommend using libxc versions older than 4.X.Y if you are interested in 
the SCAN meta-GGA. 
Search for libxc in CMakeLists.txt (it should be the 2nd entry) and comment the line "include libxc version 3.0.1" out
and remove the comment symbol from the line with the newer libxc version (e.g. 4.2.3).

```bash 
$ cmake .. 
$ make 
```
You need to add pyscf to your PYTHONPATH enviroment variable. 

```bash
export PYTHONPATH=[path_to_pyscf]/pyscf/:$PYTHONPATH
```
Note: The variable path_to_pyscf describes the absolute path pointing towards the pyscf folder (not including the pyscf folder itself). 

With newer versions of PySCF, e.g. pyscf-1.6.4, it might be necessary to add the following library paths in your job script:

```bash
export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_avx2.so
export LD_PRELOAD=$LD_PRELOAD:$MKLROOT/lib/intel64/libmkl_sequential.so
export LD_PRELOAD=$LD_PRELOAD:$MKLROOT/lib/intel64/libmkl_core.so
```

The value of $MKLROOT depends on your specific system.

# Python packages (required)  

```bash 
$ pip3 install numpy 
$ pip3 install scipy 
$ pip3 install ase 
$ pip3 install numba 
$ pip3 install h5py 
```

To check which versions you have installed you might can use 

```bash 
$ python3 python_package_versions.py
```

We recommand the following versions 

- ase 3.17.0
- h5py 2.9.0
- numba 0.43.1
- scipy 1.2.1
- numpy 1.16.2
- pyscf 1.6.1-1

If your installed versions are different, you might want to update to newer versions. 
Otherwise, some python WARNINGS may appear in the screen log or output files. 

```bash 
$ pip3 install numpy --upgrade
$ pip3 install scipy --upgrade
$ pip3 install ase --upgrade
$ pip3 install numba --upgrade
$ pip3 install h5py --upgrade
```

# Python packages (optional) 

For visualization and plotting, you might want to have the following packages. 
```bash 
$ pip3 install matplotlib 
$ apt-get install python3-tk
```

Further, if you want to use the pyberny geometry optimizer, you need to install it. 
```bash 
$ pip3 install -U pyberny
```

# PyFLOSIC installation 

```bash 
$ git clone https://github.com/pyflosic/pyflosic.git
```

You need to add pyflosic to your PYTHONPATH environment variable globally or in bash/job scripts.

```bash
export PYTHONPATH=[path_to_pyflosic]/pyflosic/src/:$PYTHONPATH
export PYTHONPATH=[path_to_pyflosic]/pyflosic/utils/:$PYTHONPATH
```
Note: The variable path_to_pyflosic describes the absolute path pointing towards the pyflosic folder (not including the pyflosic folder itself). 
