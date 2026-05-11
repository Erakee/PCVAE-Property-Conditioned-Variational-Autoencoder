import sys
import os
import contextlib
import random
import numpy as np
import torch
# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from util.config_loader import (
    default_config_path,
    load_training_config,
    project_root,
    resolve_config_path,
)
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.DataStructs as DataStructs
from util.tokens import Tokenizer, getTokenizer
from dscribe.descriptors import SOAP
from ase import Atoms
# suppress rdkit warnings
from rdkit import rdBase, RDLogger
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.*')


def __load_config(fyaml=None, model='vae_h'):
    """Alias for :func:`util.config_loader.load_training_config`."""
    return load_training_config(fyaml, model=model)


config = load_training_config()


def p_cfg(model):
    return load_training_config(model=model)

def mkdir_multi(path_str):
    if os.path.isdir(path_str) or path_str == '':
        return
    else:
        dir_namme = os.path.dirname(path_str)
        mkdir_multi(dir_namme)
        os.mkdir(path_str)

def get_tokenizer(model='vae_h'):
    if os.path.exists(config['fname_tokenizer']):
        print('read tokenizer from pickle file "%s"' % (config['fname_tokenizer']))
        with open(config['fname_tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = getTokenizer(config['token_file'], handleBraces=True)
        with open(config['fname_tokenizer'], 'wb') as f:
            pickle.dump(tokenizer, f)
    config['vae_param']['num_vocabs'] = tokenizer.getTokensSize()# - 2
    return tokenizer

def ori_get_tokenizer(model='vae_h'):
    if os.path.exists(config['fname_tokenizer']):
        print('read tokenizer from pickle file "%s"' % (config['fname_tokenizer']))
        with open(config['fname_tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = getTokenizer(config['token_file'], handleBraces=True)
        with open(config['fname_tokenizer'], 'wb') as f:
            pickle.dump(tokenizer, f)
    config['vae_param']['num_vocabs'] = tokenizer.getTokensSize() - 2
    return tokenizer

def isValidSmiles(smiles):
    if smiles == '':
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def getSimpleScore(smiles):
    if smiles == '':
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    else:
        for ma in mol.GetAtoms():
            if ma.GetAtomicNum() not in (1, 6, 7, 8, 9, 17):
                return -1
        return 1

class SimilEvaluator(object):
    def __init__(self, smi_fname, pkl_fname, maxLength):
        self.maxLength = maxLength
        if os.path.isfile(pkl_fname):
            with open(pkl_fname, 'rb') as f:
                self.fps = pickle.load(f)
        else:
            self.fps = []
            with open(smi_fname, 'r') as f:
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    mol = Chem.MolFromSmiles(line.strip())
                    if mol is not None:
                        self.fps.append(self.genFP(mol))
            with open(pkl_fname, 'wb') as f:
                pickle.dump(self.fps, f)

    @staticmethod
    def genFP(mol):
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True)
        # return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True)

    def getSimil(self, smiles):
        if len(smiles) < 1 or len(smiles) >= self.maxLength:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = self.genFP(mol)
        simil = max(DataStructs.BulkTanimotoSimilarity(fp, self.fps))
        return self.rescaleScores(simil)
    
    def rescaleScores(self, simil):
        if simil < 0.2:
            return -1
        elif simil > 0.7:
            return 1
        else:
            return (simil - 0.2) * 4 - 1.0

class temp_seed:
    def __init__(self, seed):
        self.seed = seed
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()

    def __enter__(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)


def get_soap_descriptors(smiles_list):
    valid_mols = []
    valid_smiles = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                if mol.GetNumConformers() == 0:
                    result = AllChem.EmbedMolecule(mol)
                    if result == -1:
                        continue
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                positions = mol.GetConformer().GetPositions()
                ase_atoms = Atoms(numbers=atomic_numbers, positions=positions)
                valid_mols.append(ase_atoms)
                valid_smiles.append(smi)
                valid_indices.append(i)
            except (ValueError, Exception) as e:
                print(f"Error processing SMILES {smi}: {e}")
                continue

    species = set()
    for mol in valid_mols:
        species.update(mol.get_chemical_symbols())
    cutoff = 6.0
    nmax = 8
    lmax = 6
    soap = SOAP(species=species, r_cut=cutoff, n_max=nmax, l_max=lmax, periodic=False)

    descriptors = []
    for mol in valid_mols:
        desc = soap.create(mol)
        mol_desc = np.mean(desc, axis=0)
        descriptors.append(mol_desc)
    return np.array(descriptors), valid_smiles
