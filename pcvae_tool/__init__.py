"""
pcvae_tool — PCVAE molecular generation & enthalpy prediction toolkit
=====================================================================

四个 tool 模块：

  pcvae_tool.tools.pcvae_tool      → 分子生成（4 种条件模式）
  pcvae_tool.tools.enthalpy_tool   → 生成焓预测（MPNN）
  pcvae_tool.tools.smiles_utils_tool → SMILES 验证 / 规范化 / 属性
  pcvae_tool.tools.similarity_tool   → 分子相似度 / 多样性 / 新颖性

常用函数在顶层暴露（同 env 内调用）：

    import pcvae_tool as pt

    # 焓预测
    pt.predict_enthalpy('CC1=CC=CC=C1')
    pt.predict_enthalpies(['C', 'CC', 'CCO'])

    # 分子生成
    pt.get_enthalpy_range()
    pt.generate_random(num_samples=50)
    pt.generate_by_enthalpy(enthalpy=-50.0, num_samples=100)
    pt.generate_by_smiles(smiles='C1CCCCC1')
    pt.generate_by_smiles_and_enthalpy(smiles='...', enthalpy=52.8)

    # SMILES 工具
    pt.validate('CCO')
    pt.canonicalize('c1ccccc1')
    pt.get_properties('c1ccccc1')
    pt.batch_validate(['CCO', 'invalid'])
    pt.batch_canonicalize(['c1ccccc1', 'CCO'])
    pt.batch_properties(['CCO', 'C1CCCCC1'])
    pt.deduplicate(['CCO', 'OCC', 'c1ccccc1'])

    # 相似度工具
    pt.tanimoto('CCO', 'CCCO')
    pt.most_similar('CCO', ['CC', 'CCO', 'CCCC'], top_n=3)
    pt.diversity_score(['CCO', 'c1ccccc1', 'CC(=O)O'])
    pt.novelty_score(generated=[...], reference=[...])
    pt.filter_by_similarity(['CCO', 'CCCO'], reference='CCO', min_sim=0.3)
"""
from .tools.enthalpy_tool import (
    predict_enthalpy,
    predict_enthalpies,
)
from .tools.pcvae_tool import (
    generate_random,
    generate_by_enthalpy,
    generate_by_smiles,
    generate_by_smiles_and_enthalpy,
    generate_molecules,
    get_enthalpy_range,
)
from .tools.smiles_utils_tool import (
    validate,
    canonicalize,
    get_properties,
    batch_validate,
    batch_canonicalize,
    batch_properties,
    deduplicate,
)
from .tools.similarity_tool import (
    tanimoto,
    most_similar,
    similarity_matrix,
    diversity_score,
    novelty_score,
    filter_by_similarity,
)

__version__ = '1.1.0'

__all__ = [
    # enthalpy
    'predict_enthalpy', 'predict_enthalpies',
    # generation
    'generate_random', 'generate_by_enthalpy', 'generate_by_smiles',
    'generate_by_smiles_and_enthalpy', 'generate_molecules', 'get_enthalpy_range',
    # smiles utils
    'validate', 'canonicalize', 'get_properties',
    'batch_validate', 'batch_canonicalize', 'batch_properties', 'deduplicate',
    # similarity
    'tanimoto', 'most_similar', 'similarity_matrix',
    'diversity_score', 'novelty_score', 'filter_by_similarity',
]
