import csv
import os
import pandas as pd

from openbabel import openbabel
# import xyz2mol

from rdkit import Chem
import rdkit.Chem.GraphDescriptors as GD
import rdkit.Chem.EState.EState_VSA as estate
import rdkit.Chem.Lipinski as LP
import rdkit.Chem.MolSurf as MS
import rdkit.Chem.Crippen as CP
import rdkit.Chem.Descriptors as D1


from padelpy import padeldescriptor

import xgboost as xgb


# def get_mol(xyz):
#     """
#     Obtain rdkit mol object from xyz file using xyz2mol
#     :param xyz: xyz file
#     :return: rdkit mol object
#     """
#     atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(xyz)
#     mol = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge)
#     return mol[0]


def mu(N):
    """
    Obtain my value as feature.
    Function fitted on a subset of QM7b
    :param N: number of atoms in molecule
    :return: mu
    """
    return 308.14 * (N ** (-0.52))




def rdkit_features(mol):
    """
    obtain rdkit feature necessary for gdd prediction by XGBoost model
    :param mol: RDKit mol object
    :return: dictionary containing RDKit features
    """
    desc_d = {'GD': ['Chi0', 'Chi0v', 'Chi1', 'Chi1v', 'Chi4v'
        , 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3']

        , 'Estate': ['EState_VSA10', 'EState_VSA2', 'EState_VSA5', 'EState_VSA7']

        , 'CP': ['MolMR']
        , 'D1': ['MolWt']
        , 'LP': ['FractionCSP3', 'NHOHCount', 'NOCount'
            , 'NumAliphaticHeterocycles', 'NumAromaticHeterocycles'
            , 'NumHAcceptors', 'NumRotatableBonds'
            , 'NumSaturatedCarbocycles', 'NumSaturatedRings'
                 ]

        , 'MS': ['PEOE_VSA1', 'PEOE_VSA13', 'PEOE_VSA14'
            , 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA7'
            , 'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA7'
            , 'SlogP_VSA1', 'SlogP_VSA12', 'SlogP_VSA3'
            , 'SlogP_VSA4', 'SlogP_VSA8', 'SlogP_VSA9',
                 ]
              }

    res_d = {}
    for i in desc_d:
        for j in desc_d[i]:
            if i == 'GD':
                f = getattr(GD, j)
                res_d[j] = f(mol)
            elif i == 'Estate':
                f = getattr(estate, j)
                res_d[j] = f(mol)
            elif i == 'CP':
                f = getattr(CP, j)
                res_d[j] = f(mol)
            elif i == 'D1':
                f = getattr(D1, j)
                res_d[j] = f(mol)
            elif i == 'LP':
                f = getattr(LP, j)
                res_d[j] = f(mol)
            elif i == 'MS':
                f = getattr(MS, j)
                res_d[j] = f(mol)
    return res_d


def sub_feature(smi, type):
    """
    Obtain Substructures fingerprint from padelpy
    :param smi: str, SMILES of molecule
    :param type: str, Sub pr SubC fpr Substructures and Substrucures Count respecively
    :return: dict, Sub feature
    """

    with open("tmp.smi","w") as smi_file:
        smi_file.write(smi)

    if type == 'Sub':
        fingerprint_descriptortypes = './xml_pdp/SubstructureFingerprinter.xml'
    elif type == 'SubC':
        fingerprint_descriptortypes = './xml_pdp/SubstructureFingerprintCount.xml'
    else:
        print(type, 'not supported')

    padeldescriptor(mol_dir='tmp.smi',
                  d_file='tmp.csv',  # 'Substructure.csv'
                  descriptortypes=fingerprint_descriptortypes,
                  detectaromaticity=True,
                  standardizenitro=True,
                  standardizetautomers=True,
                  threads=4,
                  removesalt=True,
                  log=False,
                  fingerprints=True)

    with open('tmp.csv', 'r') as f:
        l = list(csv.reader(f))
        fingerprint = {header: list(map(float, values))[0] for header, *values in zip(*l) if header != 'Name'}

    os.remove('tmp.smi')
    os.remove('tmp.csv')

    return fingerprint


def gddml(xyz, name=None):
    """
    Obtain gdd value predicted by XGBoost model
    :param xyz: str, path to xyz file in format required by xyz2mol
    :param name: str, optional, name of molecule
    :return: predicted gdd value
    """

    # name molecule
    if name is not None:
        all_feat = {'name': str(name)}
    else:
        all_feat = {'name': str(xyz)}

    # get mu feature
    # mol = get_mol(xyz)
    # smi='[SiH2]1C=Cc2c1c1c3cocc3c3cc(cnc3c1c1cocc21)-c1cccc2nsnc12'
    # mol = Chem.MolFromMolFile(xyz)
    # # mol =  Chem.MolFromSmiles(smi)
    # # mol = Chem.AddHs(mol)
    # mol = Chem.RemoveHs(mol)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, xyz)
    natom = mol.NumAtoms()
    obConversion.WriteFile(mol, 'test.sdf')
    obConversion.SetInAndOutFormats("xyz", "smi")
    smi = obConversion.WriteString(mol)
    print(smi)
    # print(outMDL)

    mol = Chem.MolFromMolFile('test.sdf')
    # mol = Chem.RemoveHs(mol)
    # natom = int(mol.NumAtoms())
    all_feat['mu'] = mu(natom)

    # get RDKit feature
    tmp = rdkit_features(mol)
    all_feat = all_feat | tmp

    # get Sub fingerprint
    # smi = Chem.MolToSmiles(mol)
    # print(smi)
    subfp = sub_feature(smi, 'Sub')
    subfpc = sub_feature(smi, 'SubC')

    all_feat = all_feat | subfp | subfpc
    for i in all_feat:
        all_feat[i] = [all_feat[i]]
    df = pd.DataFrame.from_dict(all_feat).set_index('name')

    model = xgb.XGBRegressor()
    model.load_model("gddml-xgb.json")
    print(all_feat)
    gdd = model.predict(df)[0]
    return gdd


from pathlib import Path

resource_path = Path(__file__).parent
# test = '/home/corev/project_gdd/xyz_all/QM7b-molecule0014.xyz'
# test = '/home/corev/project_gdd/xyz_all/showcase1-0001.xyz'
# test = '/home/corev/project_gdd/xyz_all/test-1734.xyz'
# test = '/home/corev/project_gdd/tmp/test.sd34'

tmp = gddml('test.xyz', 'test.xyz')
print(tmp)
# df = {x: [] for x in tmp}

# geom_dir='/home/corev/project_gdd/xyz_all/'
# # geom_dir='/home/corev/project_gdd/gddML/extra_test_xyz/'
# for i in os.listdir(geom_dir):
#     if i.endswith('.xyz'):
#         name = i[:-4]
#         tmp = gddml(geom_dir+i, name)
#         for j in df:
#             df[j].append(tmp[j])
#     print(len(df['name']))
# df = pd.DataFrame.from_dict(df)
# df.to_csv('feature-all.csv', index=False)