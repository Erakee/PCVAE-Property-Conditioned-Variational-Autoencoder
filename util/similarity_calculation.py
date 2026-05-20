from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors, RDKFingerprint
from scipy.spatial.distance import cosine
from rdkit.DataStructs import FingerprintSimilarity


def get_mol_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol)
    ], dtype=float)

def get_morgan_fingerprint(mol, radius=2, nBits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def descriptor_similarity_with_fingerprint(target_smiles, generated_smi_file, alpha=0.5):
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol:
        raise ValueError("Invalid target SMILES")

    target_desc = get_mol_descriptors(target_mol)
    target_fp = get_morgan_fingerprint(target_mol)

    gen_descs, gen_fps = [], []
    with open(generated_smi_file, 'r') as f:
        for line in f:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            gen_descs.append(get_mol_descriptors(mol))
            gen_fps.append(get_morgan_fingerprint(mol))

    if not gen_descs:
        return 0.0

    # 所有描述符统一归一化
    all_descs = np.vstack([target_desc] + gen_descs)
    mean = all_descs.mean(axis=0)
    std = all_descs.std(axis=0) + 1e-8
    norm_descs = (all_descs - mean) / std

    target_norm = norm_descs[0]
    gen_norms = norm_descs[1:]

    # 融合相似度计算
    similarities = []
    for desc, fp in zip(gen_norms, gen_fps):
        desc_sim = 1 / (1 + euclidean(target_norm, desc))
        fp_sim = tanimoto_similarity(target_fp, fp)
        combined_sim = alpha * desc_sim + (1 - alpha) * fp_sim
        similarities.append(combined_sim)

    return np.mean(similarities)



def get_mol_fingerprint(mol):
    """计算分子的RDK指纹"""
    return RDKFingerprint(mol)


def descriptor_cosine_similarity(target_smiles, generated_smi_file):
    """计算描述符的余弦相似度"""
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol:
        raise ValueError("Invalid target SMILES")

    target_desc = get_mol_descriptors(target_mol)
    gen_descs = []

    with open(generated_smi_file, 'r') as f:
        for line in f:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            gen_descs.append(get_mol_descriptors(mol))

    if not gen_descs:
        return 0.0

    # 标准化描述符（z-score）
    scaler = StandardScaler()
    all_descs = np.vstack([target_desc] + gen_descs)
    norm_descs = scaler.fit_transform(all_descs)
    target_norm = norm_descs[0]
    gen_norms = norm_descs[1:]

    # 计算描述符的余弦相似度
    similarities = []
    for desc in gen_norms:
        sim = 1 - cosine(target_norm, desc)  # 余弦相似度 = 1 - 余弦距离
        similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def fingerprint_similarity(target_smiles, generated_smi_file):
    """计算分子指纹的Tanimoto相似度"""
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol:
        raise ValueError("Invalid target SMILES")

    target_fp = get_mol_fingerprint(target_mol)
    similarities = []

    with open(generated_smi_file, 'r') as f:
        for line in f:
            smi = line.strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            gen_fp = get_mol_fingerprint(mol)
            similarity = FingerprintSimilarity(target_fp, gen_fp)
            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0.0


def combined_similarity(target_smiles, generated_smi_file, weight_desc=0.5, weight_fp=0.5):
    """融合描述符和指纹的相似度"""
    # 计算余弦相似度（描述符）
    desc_sim = descriptor_cosine_similarity(target_smiles, generated_smi_file)
    # 计算Tanimoto相似度（指纹）
    fp_sim = fingerprint_similarity(target_smiles, generated_smi_file)

    # 根据权重加权融合相似度
    combined_sim = weight_desc * desc_sim + weight_fp * fp_sim
    return combined_sim


def main():
    score = descriptor_similarity_with_fingerprint(
        # target_smiles="C1(=NC(=O)NN1)[N+](=O)[O-]",
        target_smiles="CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        generated_smi_file=r"D:\Project\VAE_Related\ConVAE\generate_smi\422_generated_smiles.smi",
        alpha=1  # 你也可以试试 0.7 或 0.3
    )
    print("平均融合相似度分数：", score)
    score2 = combined_similarity(
        # target_smiles="C1(=NC(=O)NN1)[N+](=O)[O-]",
        target_smiles="CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        generated_smi_file=r"D:\Project\VAE_Related\ConVAE\generate_smi\422_generated_smiles.smi",
        weight_desc = 1, weight_fp = 0
    )
    print("平均cosine相似度分数：", score2)

if __name__ == '__main__':
    main()
