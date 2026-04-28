"""
pcvae_tool — PCVAE molecular generation & enthalpy prediction toolkit
=====================================================================

两个对外 tool 模块：

  pcvae_tool.tools.pcvae_tool      → 分子生成（4 种条件模式）
  pcvae_tool.tools.enthalpy_tool   → 生成焓预测（MPNN）

为方便调用，常用函数也直接在顶层暴露：

    import pcvae_tool

    pcvae_tool.predict_enthalpy('CC1=CC=CC=C1')
    pcvae_tool.generate_random(num_samples=50)
    pcvae_tool.generate_by_enthalpy(enthalpy=-50.0, num_samples=100)
    pcvae_tool.generate_by_smiles(smiles='C1CCCCC1')
    pcvae_tool.generate_by_smiles_and_enthalpy(smiles='...', enthalpy=52.8)
    pcvae_tool.get_enthalpy_range()
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

__version__ = '1.0.0'

__all__ = [
    # enthalpy
    'predict_enthalpy',
    'predict_enthalpies',
    # generation
    'generate_random',
    'generate_by_enthalpy',
    'generate_by_smiles',
    'generate_by_smiles_and_enthalpy',
    'generate_molecules',
    'get_enthalpy_range',
]
