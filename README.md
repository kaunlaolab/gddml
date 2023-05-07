# $\omega_{\rm GDD}^{\rm ML}$

This machine learning model enable fast prediction of the range-separation parameter $\omega_{\rm GDD}^{\rm ML}$ for the LRC-&omega;PBE functional. Given Cartesian coordinates in the form of a .xyz or .sdf file, this program will generate the features used by the model and return the predicted value for the parameter.

The details of this code can be found in the work of DOI:

To ensure this program performs to the best of its ability, using a SDF file is recommended as this format provides the best descriptions of the system and ensure proper placement of charges or double bonds. If provided a system in the XYZ format, the model will rely on xyz2mol (https://github.com/jensengroup/xyz2mol) to obtain a mol object from its coordinate. In case of a charged system, the second line of the XYZ file must be formatted as to explcititly state it. For example, the second line of the XYZ file containing a positively charged system would be formatted as:
```
charge=1=
```
For more information on xyz2mol, please see the related article at DOI:10.1002/bkcs.10334

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
