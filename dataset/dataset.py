import hashlib
import os
import pickle

import torch
import pandas as pd
import numpy as np
from ase import Atoms
from rdkit import Chem

from util.tokens import Tokenizer
import util.utils as utils
import torch
from util import utils
from dscribe.descriptors import SOAP
from rdkit.Chem import AllChem


config = utils.__load_config()

def preprocess(df):
    # 生成焓数据归一化，将上下限扩大到设定阈值
    enthalpy = np.array(df)
    threshold = utils.config['threshold']
    min_enthalpy = np.min(enthalpy)
    max_enthalpy = np.max(enthalpy)
    diff = max_enthalpy - min_enthalpy
    lower_bound = min_enthalpy - threshold * diff
    upper_bound = max_enthalpy + threshold * diff
    max_diff = upper_bound - lower_bound
    normalized_enthalpy = (enthalpy - lower_bound) / (upper_bound - lower_bound)
    # normalized_enthalpy = normalized_enthalpy.to(torch.float32)
    normalized_enthalpy = torch.tensor(normalized_enthalpy, dtype=torch.float32)
    return normalized_enthalpy.tolist(), lower_bound, upper_bound

def get_soap_descriptors(smiles_list, enthalpy_list, smiles_indices):
    valid_mols = []
    valid_smiles = []
    valid_indices = []
    valid_enthalpy = []
    valid_smiles_indices = []
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
                valid_enthalpy.append(enthalpy_list[i])
                valid_smiles_indices.append(smiles_indices[i])
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
    return np.array(descriptors), valid_smiles, valid_smiles_indices, valid_enthalpy

class ori_SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['heat_of_formation'].tolist()
        self.enthalpy, self.lb, self.ub = preprocess(self.enthalpy)
        # 存储 one-hot 编码结果
        self.smiles_hot = self.one_hot_collate_fn(self.smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles_hot[i], self.enthalpy[i]

    def _getbound(self):
        return self.lb, self.ub

    def collate_fn(self, smilesStrs):
        tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
        return torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addStart=True)], padding_value=self.tokenizer.getTokensNum('<pad>'), batch_first=True), torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in self.tokenizer.getNumVector(tokenVectors, addEnd=True)], padding_value=self.tokenizer.getTokensNum('<pad>'), batch_first=True)

    def one_hot_collate_fn(self, smilesStrs):
        skip_vocab = 2
        tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
        numVectors = self.tokenizer.getNumVector(tokenVectors)
        one_hot_code = torch.zeros((len(smilesStrs), self.maxLength, self.tokenizer.getTokensSize() - skip_vocab), dtype=torch.float32)
        for i, vec in enumerate(numVectors):
            last_j = -1
            for j, n in enumerate(vec):
                one_hot_code[i, j, n - 2] = 1
                last_j = j
            if last_j + 1 < self.maxLength:
                one_hot_code[i, last_j + 1:, 0] = 1
        return one_hot_code


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['heat_of_formation'].tolist()
        self.enthalpy, self.lb, self.ub = preprocess(self.enthalpy)

        # 将collate_fn绑定到当前实例（关键修改！）
        self.collate_fn = self._one_hot_collate_fn

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.enthalpy[i]

    def _getbound(self):
        return self.lb, self.ub

    def _one_hot_collate_fn(self, batch):
        """实例方法版本，自动获取self的tokenizer和maxLength"""
        skip_vocab = 2
        smilesStrs = [item[0] for item in batch]
        enthalpies = torch.tensor([item[1] for item in batch], dtype=torch.float32)

        tokenVectors = self.tokenizer.tokenize(smilesStrs, useTokenDict=True)
        numVectors = self.tokenizer.getNumVector(tokenVectors)

        one_hot_code = torch.zeros(
            (len(batch), self.maxLength, self.tokenizer.getTokensSize() - skip_vocab),
            dtype=torch.float32
        )

        for i, vec in enumerate(numVectors):
            last_j = -1
            for j, n in enumerate(vec):
                one_hot_code[i, j, n - 2] = 1
                last_j = j
            if last_j + 1 < self.maxLength:
                one_hot_code[i, last_j + 1:, 0] = 1

        return one_hot_code, enthalpies


class SmilesDictDataset(torch.utils.data.Dataset):
    def __init__(self, fname, tokenizer, maxLength):
        super().__init__()
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.pad_idx = tokenizer.getTokensNum('<pad>')

        self.data = pd.read_csv(fname)
        self.smiles = self.data['smiles'].tolist()
        self.enthalpy = self.data['heat_of_formation'].tolist()
        self.enthalpy, self.lb, self.ub = preprocess(self.enthalpy)
        self.smiles_indices = self._preprocess_smiles()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles_indices[i], torch.tensor(self.enthalpy[i])

    def _getbound(self):
        return self.lb, self.ub

    def _preprocess_smiles(self):
        """将 SMILES 转换为整数索引序列并填充"""
        processed = []
        for smi in self.smiles:
            # Step 1: Tokenize SMILES
            token_vector = self.tokenizer.tokenize(
                [smi],
                useTokenDict=True
            )[0]  # 获取第一个（唯一）SMILES的token列表

            # Step 2: 转换为整数索引
            num_vector = self.tokenizer.getNumVector(
                [token_vector],
                addStart=True,
                addEnd=True
            )[0]  # 假设需要添加 <start> 和 <end>

            # Step 3: 截断/填充到 maxLength
            # 检查索引不超过词表大小
            if max(num_vector) >= self.tokenizer.getTokensSize():
                print(f"Warning: token index {max(num_vector)} exceeds vocabulary size {self.tokenizer.getTokensSize()}")
            
            # 截断/填充到 maxLength
            if len(num_vector) > self.maxLength:
                # 保留 <start> 和 <end> 的情况下截断中间部分
                truncated = [num_vector[0]] + num_vector[1:-1][:self.maxLength - 2] + [num_vector[-1]]
                num_vector = truncated
            else:
                # 填充到 maxLength（在末尾添加 <pad>）
                padding = [self.pad_idx] * (self.maxLength - len(num_vector))
                num_vector = num_vector + padding

            processed.append(torch.tensor(num_vector))
        return processed

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        smiles_indices, enthalpies = zip(*batch)
        # 转换为张量 (已经预先填充到相同长度)
        smiles_tensor = torch.stack(smiles_indices)  # [batch_size, maxLength]
        enthalpies_tensor = torch.stack(enthalpies)  # [batch_size]
        return smiles_tensor, enthalpies_tensor


class backup_HybridDataset(SmilesDictDataset):
    def __init__(self, fname, tokenizer, maxLength, soap_config):
        super().__init__(fname, tokenizer, maxLength)

        self.processed_soap, self.valid_smiles, self.valid_smiles_indices, self.valid_enthalpy = \
            self._get_soap_descriptors(self.smiles, self.enthalpy, self.smiles_indices)
        # 这里返回的valid smiles要返回字典索引

    def _get_soap_descriptors(smiles_list, enthalpy_list, smiles_indices):
        valid_mols = []
        valid_smiles = []
        valid_indices = []
        valid_enthalpy = []
        valid_smiles_indices = []
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
                    valid_enthalpy.append(enthalpy_list[i])
                    valid_smiles_indices.append(smiles_indices[i])
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
        return np.array(descriptors), valid_smiles, valid_smiles_indices, valid_enthalpy

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        return {
            'smiles': self.valid_smiles[idx],
            'smiles_indices': self.valid_smiles_indices[idx],
            'soap': self.processed_soap[idx],
            'enthalpy': self.valid_enthalpy[idx]
        }

    @staticmethod
    def collate_fn(batch):
        soap, enthalpy = zip(*batch)
        return torch.stack(soap), torch.stack(enthalpy)

class HybridDataset(SmilesDictDataset):
    def __init__(self, fname, tokenizer, maxLength, soap_config):
        super().__init__(fname, tokenizer, maxLength)

        # self.processed_soap, self.valid_smiles, self.valid_smiles_indices, self.valid_enthalpy = \
        #     self._get_soap_descriptors(self.smiles, self.enthalpy, self.smiles_indices)

        # 生成唯一缓存文件名
        self.cache_file = self._generate_cache_name(fname, soap_config)

        # 尝试加载缓存
        if not self._load_cache():
            # 缓存不存在时重新计算
            self.processed_soap, self.valid_smiles, self.valid_smiles_indices, self.valid_enthalpy = \
                get_soap_descriptors(self.smiles, self.enthalpy, self.smiles_indices)
            self._save_cache()

    def _generate_cache_name(self, fname, config):
        """生成包含配置哈希值的唯一文件名"""
        config_str = f"{config['rcut']}_{config['nmax']}_{config['lmax']}"
        data_hash = hashlib.md5(open(fname, 'rb').read()).hexdigest()[:8]
        return f"{os.path.splitext(fname)[0]}_soap_{config_str}_{data_hash}.pkl"

    def _load_cache(self):
        """加载缓存数据"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                # 验证数据完整性
                assert len(cache_data['smiles']) == len(cache_data['indices'])
                assert len(cache_data['smiles']) == len(cache_data['enthalpy'])
                assert cache_data['soap'].shape[0] == len(cache_data['smiles'])

                self.processed_soap = cache_data['soap']
                self.valid_smiles = cache_data['smiles']
                self.valid_smiles_indices = cache_data['indices']
                self.valid_enthalpy = cache_data['enthalpy']
                print(f"Loaded cached SOAP descriptors from {self.cache_file}")
                return True
            except Exception as e:
                print(f"Cache loading failed: {str(e)}")
                return False
        return False

    def _save_cache(self):
        """保存缓存数据"""
        cache_data = {
            'soap': self.processed_soap,
            'smiles': self.valid_smiles,
            'indices': self.valid_smiles_indices,
            'enthalpy': self.valid_enthalpy
        }

        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved SOAP descriptors to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {str(e)}")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        return {
            # 'smiles': self.valid_smiles[idx],
            'smiles_indices': self.valid_smiles_indices[idx],
            'soap': torch.FloatTensor(self.processed_soap[idx]),
            'enthalpy': torch.tensor(self.valid_enthalpy[idx], dtype=torch.float)
        }

    @staticmethod
    def collate_fn(batch):
        # 从每个样本的字典中提取所需字段
        soap = torch.stack([item['soap'] for item in batch])
        enthalpy = torch.stack([item['enthalpy'] for item in batch])
        smiles_indices = torch.stack([item['smiles_indices'] for item in batch])

        # 根据模型需要返回相应字段
        return {
            'soap': soap,
            'enthalpy': enthalpy,
            'smiles_indices': smiles_indices
        }

# 使用方法
# tokenizer = utils.get_tokenizer()
# print(tokenizer.tokensDict)
# smilesDataset = SmilesDictDataset(
#         utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
# # smiles_dataset = SmilesDataset(fname=utils.config['fname_dataset'], tokenizer=tokenizer, maxLength=utils.config['maxLength'])
# dataloader = torch.utils.data.DataLoader(
#         smilesDataset, batch_size=utils.config['batch_size'],
#         shuffle=True, num_workers=4,
#         collate_fn=smilesDataset.collate_fn)
# for batch in dataloader:
#    X, enthalpy = batch