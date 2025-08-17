import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
import seaborn as sns

# === 可调节的全局字体设置 ===
FONT_BASE = 20  # 可调节的字体大小主值

sns.set_theme(style="white")  # 白底无网格
plt.rcParams.update({
    'font.size': FONT_BASE,
    'axes.labelsize': FONT_BASE + 2,
    'axes.titlesize': FONT_BASE + 4,
    'xtick.labelsize': FONT_BASE,
    'ytick.labelsize': FONT_BASE,
    'legend.fontsize': FONT_BASE,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# === 数据读取 ===
csv_path = r"D:\Project\EVAE_paper\data\em_train.csv"
df = pd.read_csv(csv_path)

# === 方法1：生成焓分布 ===
plt.figure(figsize=(8, 5))
sns.histplot(df['heat_of_formation'], bins=20, kde=True, color='#0072B2')

# plt.title("Distribution of EoF")
plt.xlabel("EoF (kcal/mol)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"D:\Project\EVAE_paper\visualization\dataset\hist_heat_of_formation.png", dpi=300)
plt.show()

# === 方法2：元素频率统计（含氢）===
element_counter = Counter()
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mol = Chem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        element_counter.update(atoms)

element_df = pd.DataFrame(element_counter.items(), columns=['Element', 'Count'])
element_df.sort_values(by="Count", ascending=False, inplace=True)

plt.figure(figsize=(8, 5))
sns.barplot(data=element_df, x="Element", y="Count", palette="dark:#5A9_r")
plt.title("Element Frequency (with Hydrogens)")
plt.xlabel("Element")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"D:\Project\EVAE_paper\visualization\dataset\element_frequency_with_H.png", dpi=300)
plt.show()

# === 方法3：原子数分布（含氢）===
atom_counts = []
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mol = Chem.AddHs(mol)
        atom_counts.append(mol.GetNumAtoms())

plt.figure(figsize=(8, 5))
sns.histplot(atom_counts, bins=20, kde=True, color='#D55E00')

# plt.title("Molecular Size Distribution (with Hydrogens)")
plt.xlabel("Number of Atoms per Molecule")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"D:\Project\EVAE_paper\visualization\dataset\atom_count_distribution_with_H.png", dpi=300)
plt.show()
