Date: 16.04.2019  
Installation checked on:  

- Ubuntu 18.10 (Cosmic Cuttlefish)
- Elementary OS 0.4.1 Lok

Another (older) implementation on different operating systems (OS) can be found in the INSTALL file. 

# PySCF installation 

```bash
 $ git clone https://github.com/sunqm/pyscf  
 $ cd pyscf/lib   
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

# Python packages (required)  

```bash 
$ pip3 install numpy 
$ pip3 install scipy 
$ pip3 install ase 
$ pip3 install numba 
$ pip3 install h5py 
```

If you have installed some of these packages, you might want to update to newer versions. 


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
