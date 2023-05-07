import sys
from pathlib import Path

import numpy as np
import math

from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
import xyz2mol

import joblib
import warnings

# Remove sklearn warning about features names (model was fitted using pandas DataFrame)
warnings.filterwarnings("ignore", category=UserWarning)

path = str(Path(__file__).parent)


def mu(n):
    """
    Obtain mu as described in paper.
    :param n: number of atoms in molecule
    :return: mu
    """

    return 308.14 * (n ** (-0.52))


def rdkit_features(mol):
    """
    obtain rdkit feature necessary for gdd prediction by XGBoost model
    :param mol: RDKit mol object
    :return: dictionary containing RDKit features
    """

    descs = ['MinEStateIndex', 'qed', 'MinPartialCharge',
             'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan2',
             'FpDensityMorgan3', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'BCUT2D_MRLOW',
             'BalabanJ', 'BertzCT', 'Chi0n', 'Chi1n', 'Chi3v', 'Chi4v', 'Kappa3',
             'PEOE_VSA1', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA2', 'PEOE_VSA3',
             'PEOE_VSA9', 'SMR_VSA10', 'SMR_VSA5', 'SMR_VSA7', 'SMR_VSA9',
             'SlogP_VSA1', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
             'SlogP_VSA8', 'TPSA', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
             'EState_VSA3', 'EState_VSA5', 'VSA_EState10', 'VSA_EState2',
             'VSA_EState3', 'VSA_EState4', 'VSA_EState6', 'VSA_EState7',
             'VSA_EState8', 'HeavyAtomCount', 'NHOHCount', 'NumAliphaticCarbocycles',
             'NumAliphaticHeterocycles', 'NumAromaticHeterocycles',
             'NumAromaticRings', 'NumHAcceptors', 'NumHeteroatoms',
             'NumRotatableBonds', 'NumSaturatedCarbocycles', 'MolMR', 'fr_Al_COO',
             'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_NH', 'fr_COO2',
             'fr_C_O', 'fr_C_S', 'fr_Imine', 'fr_NH0', 'fr_NH2', 'fr_N_O',
             'fr_Ndealkylation1', 'fr_aryl_methyl', 'fr_dihydropyridine',
             'fr_epoxide', 'fr_ester', 'fr_furan', 'fr_guanido', 'fr_halogen',
             'fr_hdrzone', 'fr_imidazole', 'fr_ketone_Topliss', 'fr_lactone',
             'fr_methoxy', 'fr_nitro', 'fr_nitro_arom', 'fr_oxazole', 'fr_phenol',
             'fr_phenol_noOrthoHbond', 'fr_piperdine', 'fr_priamide', 'fr_pyridine',
             'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_thiazole']
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descs_val = desc_calc.CalcDescriptors(mol)
    res_d = {}
    for i, name in enumerate(descs):
        res_d[name] = descs_val[i]
    return res_d


def get_SubFPC(mol):
    """
    Obtain Substructure Count Fingerprint
    :param mol: RDKit mol object
    :return: Dictionary of Subtructure fingerprint
    """
    subfp = {}
    fingerprint = path + "/Substructure_fingerprint.csv"

    with open(fingerprint, 'r') as fp:
        for sub in fp:
            name, group, smarts = sub.split('\t')
            val = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts.replace('\n', ''))))
            subfp[name.replace('FP', 'FPC')] = val
    return subfp


def get_EstateFP(mol):
    """
    Obtain Estate0 fingerprint
    :param mol: RDKit mol object
    :return: Dictionary of Estate0 fingerprint
    """
    fp = {}
    list_estate0 = FingerprintMol(mol)[0]
    for i, val in enumerate(list_estate0):
        if val != 0:
            list_estate0[i] = 1
    for i, val in enumerate(list_estate0):
        fp['EStateFP' + str(i + 1)] = val
    return fp


def get_MACCS(mol):
    """
    Obtain MACCS Fingerprint for gddML
    :param mol: RDKit mol object
    :return: Dictionary of MACCS fingerprint
    """
    fp_l = list(MACCSkeys.GenMACCSKeys(mol))
    fp = {}
    for j, val in enumerate(fp_l):
        j = j + 1
        fp['MACCSkeys' + str(j)] = val
    return fp


def get_distance(a, b):
    """
    Get distance between two 3D points.
    a: [x, y, z] of first point
    b: [x, y, z] of second point
    """
    distance = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2))
    return distance


def get_mol(mol):
    """
    generate dictionary containing the atoms and coordinate of a molecule.
    :param mol: RDKit mol object
    :return: dict_A (dictionary with atoms and coordinate of molecule)

    """
    mol_d = {'atom': [], 'coord': []}
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        mol_d['atom'].append(atom.GetSymbol())
        mol_d['coord'].append([positions.x, positions.y, positions.z])
    return mol_d


def get_pair_hod(m, x=0, y=10, n=2, atom_p='O-O'):
    """
    Obtain histogram of distance for given atom pair. Bins are divided equally among the given range.
    :param m: RDKit mol object (see get_mol())
    :param x: start of histogram range
    :param y: end of histogram range
    :param n: number of bins
    :param atom_p: atom pairs investigated, in 'X-Y' format
    :return: Histogram of distance in dictionary form
    """
    a1, a2 = atom_p.split('-')
    hd_dict = {a1 + '-' + a2: []}

    mol = get_mol(m)

    for i, atomi in enumerate(mol['atom']):
        for j, atomj in enumerate(mol['atom']):
            if ((atomi == a1 and atomj == a2) or (atomi == a2 and atomj == a1)) and (i > j):
                dis = get_distance(mol['coord'][i], mol['coord'][j])
                hd_dict[a1 + '-' + a2].append(dis)

    bins = np.linspace(x, y, n)
    for pair in hd_dict:
        data = []
        bins_d = {}
        for i in enumerate(bins[:-1]):
            bin1 = i[1]
            bin2 = bins[i[0] + 1]

            ind = np.where((hd_dict[pair] >= bin1) & (hd_dict[pair] <= bin2))

            bindis = bin2 - bin1

            if bin1 not in bins_d:
                bins_d[bin1] = []
            if bin2 not in bins_d:
                bins_d[bin2] = []
            for j in ind[0]:
                c1 = 1 - ((np.array(hd_dict[pair][j]) - bin1) / bindis)
                c2 = ((np.array(hd_dict[pair][j]) - bin1) / bindis)
                bins_d[bin1].append(c1)
                bins_d[bin2].append(c2)
        for i in bins_d:
            bins_d[i] = round(sum(bins_d[i]), 3)
            data.append(bins_d[i])
        hd_dict[pair] = data

    return hd_dict


def get_hod(m):
    """
    Obtain Histogram of Distances for given molecule m. See paper for parameter fitting procedure
    :param m: RDKit mol object (see get_mol())
    :return: dictionary containing the molecule's Histogam of Distances
    """
    features = {}
    pair_hd = {'C-C': [3, 46, 15], 'C-O': [3, 17, 6], 'O-N': [9, 18, 58]}
    for atom_p in pair_hd:
        x = pair_hd[atom_p][0]
        y = pair_hd[atom_p][1]
        n = pair_hd[atom_p][2]
        hod = get_pair_hod(m, x, y, n, atom_p)
        for i, val in enumerate([round(x, 3) for x in np.linspace(x, y, n)]):
            features[atom_p + '_' + str(val)] = hod[atom_p][i]

    return features

def gddml(xyz):
    """
    Obtain gdd value predicted by gddML.
    If file given as input is in xyz format, will rely on xyz2mol for convertion to RDKit mol object.
    For some molecule, this may be inefficent or xyz2mol may fail to describe it correctly.
    In such case, it is recommended to provide a sdf file instead.
    :param xyz: str, path to xyz file in format required by xyz2mol or path to sdf
    :return: name of file, predicted gdd value
    """

    # Check if file is in xyz or sdf format via extension, generate mol object
    if xyz.endswith('.xyz'):
        name = xyz.split('/')[-1].split('.')[0]
        all_feat = {'name': str(name)}

        atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(xyz)
        mol = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge)[0]

    elif xyz.endswith('sdf'):
        name = xyz.split('/')[-1].split('.')[0]
        all_feat = {'name': str(name)}
        mol = Chem.MolFromMolFile(xyz, removeHs=False)

    else:
        print('Please provide mol file in xyz or sdf format')

    # Get features
    rdkitfp = rdkit_features(mol)
    mu_feat = {'mu': mu(len(mol.GetAtoms()))}
    hodfp = get_hod(mol)
    estatefp = get_EstateFP(mol)
    maccsfp = get_MACCS(mol)
    subfpc = get_SubFPC(mol)

    all_feat = rdkitfp | mu_feat | hodfp | estatefp | maccsfp | subfpc
    df = np.array([[all_feat[x] for x in all_feat]])

    #predict values
    model = joblib.load(path + '/gddML.pkl')
    gdd = int(round(model.predict(df)[0], 0)) * 0.001

    return name, gdd


if __name__ == '__main__':
    res = gddml(str(sys.argv[1]))
    print(res[0], res[1])
