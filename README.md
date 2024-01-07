## CompPhysClass
This is a python code for calculating bending modulus (Kc) of a membrane. 

Try to look at file `Kc.cfg` for parameter configuration.

Try to look at file `Kc.ipynb` for step-by-step running and changing the algorithm. Versions of Kc calculation are generated from this python notebook file.

Try to look at folder `ref/` for references and documents.

## Quick start
Any `Kc_new_ver?.py` or `Kc.ipynb` can be used for Kc calculation. Remember to setup `Kc.cfg` before running!

## Input
Input requires files `.tpr` and `.xtc` that can be conventionally obtained from GROMACS. Besides, file `dictionary.dat` defines tilt vector for molecule types.

## Output
Output is stored in file `Kc.log`. In addition, there are some figures and other log files that can be easily obtained by changing the code a little bit!

## Note
Because of the large sizes of `.tpr` and `.xtc`, these files are not uploaded here!

Important paper: "*Calculating the Bending Modulus for Multicomponent Lipid Membranes in Different Thermodynamic Phases*".
