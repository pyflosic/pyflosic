Date: 16.04.2019  
Installation checked on:  

- Ubuntu 18.10 (Cosmic Cuttlefish)
- Elementary OS 0.4.1 Lok

(Older) implemenation on different operating systems (os) can be find in the INSTALL file. 

# PySCF installation 

```bash
 $ git clone https://github.com/sunqm/pyscf  
 $ cd pyscf/lib   
 $ mkdir build   
 $ cd build 
 $ vi ../CMakeLists.txt
 ```

We do not recommend to use older libxc version then 4.X.Y, for instance if you are interested in 
the SCAN meta-GGA. 
Search for libxc in CMakeLists.txt (should be the 2nd entry) and comment out the line include libxc version 3.0.1 
and comment in the line with the newer libxc version (e.g. 4.2.3). 

```bash 
$ cmake .. 
$ make 
```
You need to add pyscf to your PYTHONPATH enviroment variable. 

```bash
export PYTHONPATH=[path_to_pyscf]/pyscf/:$PYTHONPATH
```

# Python packages (required)  

```bash 
$ pip3 install numpy 
$ pip3 install scipy 
$ pip3 install ase 
$ pip3 install numba 
$ pip3 install h5py 
```

If you have installed some of these packages, you might to update to newer versions. 


```bash 
$ pip3 install numpy --upgrade
$ pip3 install scipy --upgrade
$ pip3 install ase --upgrade
$ pip3 install numba --upgrade
$ pip3 install h5py --upgrade
```

# Python packages (optional) 

For visualization and plotting you might want to have the following packages. 
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

You need to add pyflosic to your PYTHONPATH enviroment variable globally or in bash/job scripts.

```bash
export PYTHONPATH=[path_to_pyflosic]/pyflosic/src/:$PYTHONPATH
export PYTHONPATH=[path_to_pyflosic]/pyflosic/utils/:$PYTHONPATH
```
