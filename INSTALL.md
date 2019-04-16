Date: 16.04.2019 
Checked on: 

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
# Python packages 

```bash 
$ pip3 install numpy 
$ pip3 install scipy 
$ pip3 install ase 
$ pip3 install numba 
$ pip3 install h5py 
```
