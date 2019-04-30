# Note: O(N) method, linear scaling 

This method is different from the original Pederson FLO-SIC method.  
The O(N) method is based on the sparsity of the density matrix based on localization.   
The FODs are optimized within the SCF cycle (FOD In-SCF optimization).   

* FODs are optimized for spin up/down separately
  * trajectory for spin up:   optout_0.xyz
  * trajectory for spin down: optout_1.xyz 
* In the OPT_FRMORB.log the SIC energy itself is monitored and not the total SIC corrected energy.


This calculation might take several minutes (5 - 10 mins depending on your local machine).   
To perform the calculation use the following command

```bash 
bash run.sh 01_C_atom_ON.py
```
Note: If your are not using the recommended run.sh file, please make sure you added 
the pyflosic/utils and pyflosic/src to your PYTHONPATH environment variable. E.g.
```bash 
export PYTHONPATH=$pyflosic/src/:$PYTHONPATH
export PYTHONPATH=$pyflosic/utils/:$PYTHONPATH
```

To collect the final O(N) optimized FODs in one .xyz file (named final_ON.xyz), use the following command 
```bash
python3 get_final_ON_xyz.py -nuc C.xyz -fod1 optout_0.xyz -fod2 optout_1.xyz 
```
To look at the final O(N) optimized FODs, perform the previous command and then type
```bash
ase gui final_ON.xyz
```
