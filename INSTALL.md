 ```bash
 $ git clone https://github.com/sunqm/pyscf  
 $ cd pyscf/lib   
 $ mkdir build   
 $ cd build 
 $ vi ../CMakeLists.txt
 ```
Search for libxc in CMakeLists.txt (should be the 2nd entry) and comment out the line include libxc version 3.0.1 
and comment in the line with the newer libxc version. 

```bash 
$ cmake .. 
$ make 
```
