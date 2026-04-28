# pcvae_tool

PCVAE 分子生成 + 生成焓预测的独立工具包，**已包含全部预训练权重**，可直接作为
agent tool 调用。

```
pcvae_tool/
├── README.md              ← 本文件
├── __init__.py            ← 顶层暴露常用接口（同 env 内调用）
├── config.yaml            ← 推理配置
├── requirements.txt       ← 依赖清单
│
├── cli.py                 ← JSON-stdout 命令行入口（跨进程/跨语言）
├── client.py              ← 跨环境客户端（agent 项目复制此文件即可）
│
├── weights/               ← 预训练权重
│   ├── pcvae_encoder.pt
│   ├── pcvae_decoder.pt
│   └── enthalpy_mpnn.pt
│
├── data/                  ← 推理所需最小数据
│   ├── em_train.csv       焓值归一化范围
│   ├── em_train.smi       tokenizer 重建
│   └── .tokenizer.pkl     训练时保存的 tokenizer（必须随仓库一起带走）
│
├── core/                  ← 内部实现（一般不要直接用）
│   ├── _config.py
│   ├── tokenizer.py
│   ├── pcvae_model.py
│   ├── mpnn_model.py
│   └── enthalpy_predictor.py
│
└── tools/                 ← 同 env 内调用的 tool 接口
    ├── enthalpy_tool.py
    └── pcvae_tool.py
```

> 同环境调用 → `import pcvae_tool`；
> 跨 conda 环境调用 → 把 `client.py` 拷到 agent 项目，见下文集成指南。

---

## Quick Start

```python
import pcvae_tool as pt

# ── Tool A: 生成焓预测 ────────────────────────────────────────────────────────
pt.predict_enthalpy('CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]')
# {'smiles': 'CC1=C...', 'valid': True, 'enthalpy': -16.78}

pt.predict_enthalpies(['C', 'CC', 'invalid_smi'])
# {'results': [...], 'n_total': 3, 'n_valid': 2, 'n_invalid': 1}


# ── Tool B: 分子生成（4 种模式）───────────────────────────────────────────────
pt.get_enthalpy_range()
# {'lower_bound': -556.03, 'upper_bound': 421.28, 'unit': 'kcal/mol'}

# 模式 1: 完全随机
pt.generate_random(num_samples=50)

# 模式 2: 仅焓值条件
pt.generate_by_enthalpy(enthalpy=-50.0, num_samples=100)

# 模式 3: 仅 SMILES 骨架条件
pt.generate_by_smiles(smiles='C1CCCCC1', num_samples=100)

# 模式 4: SMILES + 焓值联合条件（如 RDX）
pt.generate_by_smiles_and_enthalpy(
    smiles='C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]',
    enthalpy=52.8,
    num_samples=100,
)

# 统一接口（自动判断模式）
pt.generate_molecules(smiles=None, enthalpy=-50.0, num_samples=100)
```

---

## API

### 焓预测

| 函数 | 输入 | 输出 |
|------|------|------|
| `predict_enthalpy(smiles)` | `str` | `{smiles, valid, enthalpy}` |
| `predict_enthalpies(smiles_list)` | `list[str]` | `{results, n_total, n_valid, n_invalid}` |

> `valid` 表示 RDKit 是否可解析；`enthalpy=None` 表示无法预测
> （结构非法 **或** 长度 ≤ 2 字符——MPNN 在过小分子上不稳定）。

### 分子生成

所有生成函数都返回相同结构的 dict：

```python
{
    'valid_smiles':         [...],          # list[str]
    'predicted_enthalpies': [...],          # list[float], kcal/mol
    'stats': {
        'mode':            'smi_h52p8',     # 模式标签
        'smiles_input':    '...' | None,
        'enthalpy_target': 52.8 | None,
        'n_attempted':     100,
        'n_valid':         87,
        'valid_rate':      0.87,
        'device':          'cuda',
        'timestamp':       '20260427_153012',
        'output_paths':    {...}            # 仅 save=True 时存在
    },
    'output_paths': {                       # 仅 save=True 时非空
        'smiles':   '.../smi_h52p8_n100_153012_smiles.smi',
        'info':     '.../smi_h52p8_n100_153012_info.txt',
        'enthalpy': '.../smi_h52p8_n100_153012_enthalpy.txt',
        'summary':  '.../smi_h52p8_n100_153012_summary.json',
    }
}
```

| 函数 | 必需参数 | 可选参数 |
|------|---------|---------|
| `generate_random` | — | `num_samples`, `save`, `output_dir` |
| `generate_by_enthalpy` | `enthalpy` | `num_samples`, `save`, `output_dir` |
| `generate_by_smiles` | `smiles` | `num_samples`, `save`, `output_dir` |
| `generate_by_smiles_and_enthalpy` | `smiles`, `enthalpy` | `num_samples`, `save`, `output_dir` |
| `generate_molecules` | — | 全部可选；自动根据传入参数选择模式 |
| `get_enthalpy_range` | — | — |

### 输出文件

`save=True`（默认）时，每次生成会在 `output_dir/{YYYYMMDD}/` 下产出 4 份文件：

| 后缀 | 内容 | 用途 |
|------|------|------|
| `_smiles.smi` | 每行一个有效 SMILES | 直接供下游消费 |
| `_info.txt`   | `SMILES,焓值` | 人工查看 |
| `_enthalpy.txt` | 每行一个焓值 | 批量统计 |
| `_summary.json` | 结构化元数据 | **agent 解析入口** |

---

## 跨 conda 环境调用（agent 项目集成）

PCVAE 工具依赖较重（torch / PyG / rdkit），通常 agent 项目不会在同一个 conda
环境里运行。本工具内置了 **subprocess + JSON** 的跨环境调用方案：

### 工作原理

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│  Agent 项目（任意 env）     │ ──────> │ moleculeCVAE env                 │
│  - 只依赖标准库             │ subproc │   python -m pcvae_tool.cli ...   │
│  - 拷一份 client.py 即可    │ <────── │   stdout = 单行 JSON             │
└─────────────────────────────┘   JSON  └──────────────────────────────────┘
```

### 步骤 1：把 `client.py` 拷到 agent 项目

```bash
cp pcvae_tool/client.py /path/to/agent_project/pcvae_client.py
```

`client.py` **只依赖 Python 标准库**（`subprocess`, `json`, `os`, `pathlib`），
不需要在 agent 环境里装任何额外包。

### 步骤 2：在 agent 代码里调用

```python
from pcvae_client import PCVAEClient, PCVAEToolError

client = PCVAEClient(
    python_exe = r'D:\Anaconda3\envs\moleculeCVAE\python.exe',  # PCVAE 环境的 Python
    tool_root  = r'D:\Project\EVAE_paper',                       # pcvae_tool/ 父目录
)

# Tool A: 焓预测
client.predict_enthalpy('CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]')
client.predict_enthalpies(['C', 'CC', 'CCO'])

# Tool B: 分子生成（4 种模式）
client.get_enthalpy_range()
client.generate_random(num_samples=50)
client.generate_by_enthalpy(enthalpy=-50.0, num_samples=100)
client.generate_by_smiles(smiles='C1CCCCC1', num_samples=100)
client.generate_by_smiles_and_enthalpy(smiles='...', enthalpy=52.8, num_samples=100)

# 业务错误（如焓值超范围）以异常形式抛出
try:
    client.generate_by_enthalpy(enthalpy=99999.0, num_samples=10)
except PCVAEToolError as e:
    print(e.payload['type'], e.payload['error'])
```

### 备选：用 conda env 名（不推荐，启动较慢）

```python
client = PCVAEClient(
    conda_env = 'moleculeCVAE',
    tool_root = r'D:\Project\EVAE_paper',
)
```

> 如果你不想每次写 `tool_root`，可以在系统里设置环境变量
> `PCVAE_TOOL_ROOT=D:\Project\EVAE_paper`，client 会自动读取。

### 直接调用 CLI（语言无关）

如果 agent 不是 Python，也可以直接 subprocess + 解析 JSON：

```bash
"D:\Anaconda3\envs\moleculeCVAE\python.exe" -m pcvae_tool.cli generate_random --num_samples 50
"D:\Anaconda3\envs\moleculeCVAE\python.exe" -m pcvae_tool.cli predict_enthalpy --smiles "CCO"
```

stdout 末行为单行 JSON；失败时 exit code = 1 且 JSON 含 `error/type/traceback` 字段。

---

## 注意事项

1. **焓值范围**：调用 `generate_by_enthalpy*` 之前先用 `get_enthalpy_range()` 检查
   合法范围。超出范围会抛 `ValueError`。
2. **首次调用**：会把 `data/em_train.smi` 重建为 tokenizer 缓存（仓库已自带训练时
   保存好的 `.tokenizer.pkl`，直接用即可——**千万不要删掉它**，否则重建后的
   token-id 顺序与权重不匹配会输出乱码）。
3. **设备**：自动检测 CUDA；无 GPU 时 fallback 到 CPU，仅速度较慢。
4. **与原项目独立**：本目录不依赖 `D:/Project/EVAE_paper/` 下任何外部文件，可整目录拷走使用。
5. **单位**：训练数据单位为 **kcal/mol**，所有焓值参数和返回值都按 kcal/mol 处理。
