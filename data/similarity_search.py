import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs


def smiles_to_fp(smi, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)


def find_similar_smiles(query_smi, smiles_list, top_k=10):
    query_fp = smiles_to_fp(query_smi)
    fps = [smiles_to_fp(smi) for smi in smiles_list]

    sims = []
    for smi, fp in zip(smiles_list, fps):
        if fp is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            sims.append((smi, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def main():
    # 路径 & 查询结构
    csv_path = "em_train.csv"
    query_smi = "[O-][N+](=O)C1=NC(=O)NN1"

    # 读取数据
    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].dropna().tolist()

    # 查找相似结构
    similar = find_similar_smiles(query_smi, smiles_list, top_k=20)
    print("Top 10 similar structures:\n")
    for rank, (smi, score) in enumerate(similar, 1):
        print(f"{rank}. {smi} \t similarity = {score:.3f}")

    # 可视化前几个结构（可选）
    mols = [Chem.MolFromSmiles(smi) for smi, _ in similar]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
    img.save("top_similar_structures.png")


if __name__ == "__main__":
    main()
