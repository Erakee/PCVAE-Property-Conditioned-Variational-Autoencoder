"""焓预测推理 — MPNN 模型加载与图特征提取。"""
from __future__ import annotations
import warnings

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data

from ._config import load_config
from .mpnn_model import MPNNModel

warnings.filterwarnings('ignore')


# ── 单例模型加载（首次调用时初始化）──────────────────────────────────────────

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL: MPNNModel | None = None


def _get_model() -> MPNNModel:
    global _MODEL
    if _MODEL is None:
        cfg = load_config()
        m = MPNNModel(**cfg['mpnn_param']).to(_DEVICE)
        m.load_state_dict(torch.load(cfg['fname_enthalpy_mpnn'], map_location=_DEVICE))
        m.eval()
        _MODEL = m
    return _MODEL


# ── 原子 / 键特征 ─────────────────────────────────────────────────────────────

_PERMITTED_ATOMS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'Unknown']
_PERMITTED_BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]


def _one_hot(x, allowed_list):
    if x not in allowed_list:
        x = allowed_list[-1]
    return [int(x == s) for s in allowed_list]


def _atom_features(atom) -> np.ndarray:
    atom_type   = _one_hot(atom.GetSymbol(), _PERMITTED_ATOMS)
    formal_chg  = _one_hot(int(atom.GetFormalCharge()),
                           [-3, -2, -1, 0, 1, 2, 3, 'Extreme'])
    hyb_type    = _one_hot(str(atom.GetHybridization()),
                           ['S', 'SP', 'SP2', 'SP3', 'OTHER'])
    degree      = _one_hot(int(atom.GetTotalDegree()),
                           [0, 1, 2, 3, 4, 'MoreThanFour'])
    in_ring     = [int(atom.IsInRing())]
    aromatic    = [int(atom.GetIsAromatic())]
    mass_scaled = [(atom.GetMass() - 10.812) / 116.092]
    pt          = Chem.GetPeriodicTable()
    vdw_scaled  = [(pt.GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6]
    cov_scaled  = [(pt.GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76]
    chirality   = _one_hot(str(atom.GetChiralTag()),
                           ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                            'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'])
    n_hs        = _one_hot(int(atom.GetTotalNumHs()),
                           [0, 1, 2, 3, 4, 'MoreThanFour'])

    feat = (atom_type + formal_chg + hyb_type + degree + in_ring +
            aromatic + mass_scaled + vdw_scaled + cov_scaled +
            chirality + n_hs)
    return np.array(feat)


def _bond_features(bond) -> np.ndarray:
    bond_type = _one_hot(bond.GetBondType(), _PERMITTED_BONDS)
    is_conj   = [int(bond.GetIsConjugated())]
    in_ring   = [int(bond.IsInRing())]
    stereo    = _one_hot(str(bond.GetStereo()),
                         ['STEREOZ', 'STEREOE', 'STEREOANY', 'STEREONONE'])
    return np.array(bond_type + is_conj + in_ring + stereo)


def _smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()

    ref_mol = Chem.MolFromSmiles('O=O')
    n_node_feat = len(_atom_features(ref_mol.GetAtomWithIdx(0)))
    n_edge_feat = len(_bond_features(ref_mol.GetBondBetweenAtoms(0, 1)))

    X = np.zeros((n_nodes, n_node_feat))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = _atom_features(atom)
    X = torch.tensor(X, dtype=torch.float)

    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    E = torch.stack([
        torch.from_numpy(rows.astype(np.int64)),
        torch.from_numpy(cols.astype(np.int64)),
    ], dim=0)

    EF = np.zeros((n_edges, n_edge_feat))
    for k, (i, j) in enumerate(zip(rows, cols)):
        EF[k] = _bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
    EF = torch.tensor(EF, dtype=torch.float)

    return Data(x=X, edge_index=E, edge_attr=EF, y=torch.zeros(1))


# ── 公共 API ──────────────────────────────────────────────────────────────────

def is_valid_smiles(smiles: str) -> bool:
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def predict_one(smiles: str) -> float:
    """对单个 SMILES 预测生成焓（kcal/mol）。无效结构返回 0.0。"""
    if not is_valid_smiles(smiles) or len(smiles) <= 2:
        return 0.0
    data = _smiles_to_graph(smiles).to(_DEVICE)
    with torch.no_grad():
        pred = _get_model()(data)
    return float(pred.item())
