# $\omega_{\rm GDD}^{\rm ML}$

This machine learning model enable fast prediction of the range-separation parameter $\omega_{\rm GDD}^{\rm ML}$ for the LRC-&omega;PBE functional. Given Cartesian coordinates in the form of a .xyz file, this program will generate the features used by the model and return the predicted value for the parameter.

The details of this code can be found in the work of DOI:

## Running the code.

This code requires a conda installation. then install the environment:
```
conda create -n YOUR_ENV_NAME -f environment.yml
```

Once the environment is installed, activate it and run the code:
 ```
 conda activate YOUR_ENV_NAME
 python3 main.py C60.xyz
 ```
 
 Note that the C60 molecule is given as example, but any xyz file will do.
