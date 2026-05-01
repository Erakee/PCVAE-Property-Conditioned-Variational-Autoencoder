"""
pcvae_tool.client  ─  跨 conda 环境的薄客户端

设计目标
--------
- 仅使用 Python 标准库（subprocess / json / os / pathlib）
- agent 项目把本文件单独拷过去即可，**无需安装 torch / PyG / rdkit**
- 通过子进程调用 PCVAE 环境里的 `python -m pcvae_tool.cli`

使用方式
--------
    from pcvae_tool_client import PCVAEClient   # 重命名后置于 agent 项目

    client = PCVAEClient(
        python_exe = r'D:\\Anaconda3\\envs\\moleculeCVAE\\python.exe',
        tool_root  = r'D:\\Project\\EVAE_paper',  # pcvae_tool/ 所在目录
    )

    client.get_enthalpy_range()
    client.predict_enthalpy('CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]')
    client.predict_enthalpies(['C', 'CC', 'invalid'])
    client.generate_random(num_samples=50)
    client.generate_by_enthalpy(enthalpy=-50.0, num_samples=50)
    client.generate_by_smiles(smiles='C1CCCCC1', num_samples=50)
    client.generate_by_smiles_and_enthalpy(smiles='...', enthalpy=52.8, num_samples=50)
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


class PCVAEToolError(RuntimeError):
    """子进程返回的业务错误，包含 message / type / traceback。"""

    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__(payload.get('error', 'PCVAE tool error'))


class PCVAEClient:
    """通过子进程调用 PCVAE 环境里的 pcvae_tool.cli。"""

    def __init__(
        self,
        python_exe: Optional[str] = None,
        tool_root:  Optional[str] = None,
        conda_env:  Optional[str] = None,
        timeout:    Optional[float] = 600.0,
    ):
        """
        Parameters
        ----------
        python_exe : str
            PCVAE 环境的 Python 可执行文件绝对路径
            （例如 ``D:\\Anaconda3\\envs\\moleculeCVAE\\python.exe``）。
            优先级最高，**推荐使用此方式**。
        tool_root : str
            pcvae_tool/ 父目录的绝对路径（即包含 pcvae_tool 文件夹的那一层）。
            若为 None 则自动读取 PCVAE_TOOL_ROOT 环境变量。
        conda_env : str
            如果不想指定 python_exe，可改用 conda 环境名（要求系统能找到
            ``conda`` 命令）。会通过 ``conda run -n <env> --no-capture-output``
            调用。当 python_exe 提供时本字段被忽略。
        timeout : float
            子进程超时秒数，None 表示无限制。
        """
        self.python_exe = python_exe
        self.conda_env = conda_env
        self.timeout = timeout

        self.tool_root = (
            tool_root
            or os.environ.get('PCVAE_TOOL_ROOT')
            or self._guess_tool_root()
        )
        if self.tool_root is None:
            raise ValueError(
                'tool_root not provided. Set the PCVAE_TOOL_ROOT environment '
                'variable or pass tool_root=<dir containing pcvae_tool/>.')

        if not (Path(self.tool_root) / 'pcvae_tool' / 'cli.py').is_file():
            raise FileNotFoundError(
                f'pcvae_tool.cli not found under {self.tool_root}. '
                f'Make sure tool_root is the directory that *contains* '
                f'the pcvae_tool/ folder.')

        if not self.python_exe and not self.conda_env:
            raise ValueError(
                'Either python_exe or conda_env must be provided.')

    @staticmethod
    def _guess_tool_root() -> Optional[str]:
        """尝试从本文件所在位置推断 tool_root（pcvae_tool/ 的父目录）。"""
        here = Path(__file__).resolve()
        if here.parent.name == 'pcvae_tool':
            return str(here.parent.parent)
        return None

    def _build_cmd(self, sub_args: Sequence[str]) -> list:
        if self.python_exe:
            return [self.python_exe, '-m', 'pcvae_tool.cli', *sub_args]
        conda = shutil.which('conda') or 'conda'
        return [conda, 'run', '-n', self.conda_env, '--no-capture-output',
                'python', '-m', 'pcvae_tool.cli', *sub_args]

    def _run(self, sub_args: Sequence[str]) -> dict:
        cmd = self._build_cmd(sub_args)
        proc = subprocess.run(
            cmd,
            cwd=self.tool_root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=self.timeout,
        )

        stdout = (proc.stdout or '').strip()
        if not stdout:
            raise PCVAEToolError({
                'error': f'subprocess returned empty stdout (rc={proc.returncode})',
                'type':  'EmptyOutput',
                'stderr': proc.stderr,
            })

        last_line = stdout.splitlines()[-1]
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError as e:
            raise PCVAEToolError({
                'error':  f'failed to parse JSON: {e}',
                'type':   'JSONDecodeError',
                'stdout': stdout,
                'stderr': proc.stderr,
            }) from None

        if proc.returncode != 0 or 'error' in payload:
            raise PCVAEToolError(payload)
        return payload

    def get_enthalpy_range(self) -> dict:
        return self._run(['get_enthalpy_range'])

    def predict_enthalpy(self, smiles: str) -> dict:
        return self._run(['predict_enthalpy', '--smiles', smiles])

    def predict_enthalpies(self, smiles_list: Iterable[str]) -> dict:
        smiles_list = list(smiles_list)
        if not smiles_list:
            return {'results': [], 'n_total': 0, 'n_valid': 0, 'n_invalid': 0}
        return self._run(['predict_enthalpies', '--smiles', *smiles_list])

    @staticmethod
    def _gen_args(num_samples: int, save: bool, output_dir: Optional[str]) -> list:
        args = ['--num_samples', str(num_samples)]
        if not save:
            args.append('--no_save')
        if output_dir:
            args += ['--output_dir', output_dir]
        return args

    def generate_random(self, num_samples: int = 100, save: bool = True,
                        output_dir: Optional[str] = None) -> dict:
        return self._run(['generate_random',
                          *self._gen_args(num_samples, save, output_dir)])

    def generate_by_enthalpy(self, enthalpy: float, num_samples: int = 100,
                             save: bool = True,
                             output_dir: Optional[str] = None) -> dict:
        return self._run(['generate_by_enthalpy',
                          '--enthalpy', str(enthalpy),
                          *self._gen_args(num_samples, save, output_dir)])

    def generate_by_smiles(self, smiles: str, num_samples: int = 100,
                           save: bool = True,
                           output_dir: Optional[str] = None) -> dict:
        return self._run(['generate_by_smiles',
                          '--smiles', smiles,
                          *self._gen_args(num_samples, save, output_dir)])

    def generate_by_smiles_and_enthalpy(self, smiles: str, enthalpy: float,
                                        num_samples: int = 100,
                                        save: bool = True,
                                        output_dir: Optional[str] = None) -> dict:
        return self._run(['generate_by_smiles_and_enthalpy',
                          '--smiles',   smiles,
                          '--enthalpy', str(enthalpy),
                          *self._gen_args(num_samples, save, output_dir)])

    # ── SMILES 工具 ───────────────────────────────────────────────────────────

    def smiles_validate(self, smiles: str) -> dict:
        """验证单个 SMILES。返回 {smiles, valid, reason}。"""
        return self._run(['smiles_validate', '--smiles', smiles])

    def smiles_batch_validate(self, smiles_list: Iterable[str]) -> dict:
        """批量验证 SMILES，返回有效/无效分组。"""
        return self._run(['smiles_batch_validate', '--smiles', *list(smiles_list)])

    def smiles_canonicalize(self, smiles: str) -> dict:
        """规范化单个 SMILES。返回 {smiles_input, smiles_canonical, valid}。"""
        return self._run(['smiles_canonicalize', '--smiles', smiles])

    def smiles_batch_canonicalize(self, smiles_list: Iterable[str]) -> dict:
        """批量规范化 SMILES，返回规范列表及统计。"""
        return self._run(['smiles_batch_canonicalize', '--smiles', *list(smiles_list)])

    def smiles_properties(self, smiles: str) -> dict:
        """获取单个分子属性（分子式、分子量、环数、元素等）。"""
        return self._run(['smiles_properties', '--smiles', smiles])

    def smiles_batch_properties(self, smiles_list: Iterable[str],
                                keep_invalid: bool = False) -> dict:
        """批量计算分子属性，默认跳过无效 SMILES。"""
        args = ['smiles_batch_properties', '--smiles', *list(smiles_list)]
        if keep_invalid:
            args.append('--keep_invalid')
        return self._run(args)

    def smiles_deduplicate(self, smiles_list: Iterable[str],
                           keep_invalid: bool = False) -> dict:
        """按规范 SMILES 对列表去重，返回去重后列表及统计。"""
        args = ['smiles_deduplicate', '--smiles', *list(smiles_list)]
        if keep_invalid:
            args.append('--keep_invalid')
        return self._run(args)

    # ── 相似度工具 ────────────────────────────────────────────────────────────

    def tanimoto(self, smiles_a: str, smiles_b: str) -> dict:
        """计算两个分子的 Tanimoto 相似度（ECFP4）。"""
        return self._run(['tanimoto', '--smiles_a', smiles_a, '--smiles_b', smiles_b])

    def most_similar(self, query: str, smiles_list: Iterable[str],
                     top_n: int = 5) -> dict:
        """从列表中找出与 query 最相似的 top_n 个分子（降序）。"""
        return self._run(['most_similar',
                          '--query', query,
                          '--smiles', *list(smiles_list),
                          '--top_n', str(top_n)])

    def similarity_matrix(self, smiles_list: Iterable[str]) -> dict:
        """计算列表内 N×N Tanimoto 相似度矩阵（建议 ≤200 个分子）。"""
        return self._run(['similarity_matrix', '--smiles', *list(smiles_list)])

    def diversity_score(self, smiles_list: Iterable[str]) -> dict:
        """计算列表内部多样性（0~1，越高越多样）。"""
        return self._run(['diversity_score', '--smiles', *list(smiles_list)])

    def novelty_score(self, generated: Iterable[str], reference: Iterable[str],
                      threshold: float = 0.4) -> dict:
        """计算生成集对参考集的新颖性（最大相似度 < threshold 的比例）。"""
        return self._run(['novelty_score',
                          '--generated', *list(generated),
                          '--reference', *list(reference),
                          '--threshold', str(threshold)])

    def filter_by_similarity(self, smiles_list: Iterable[str], reference: str,
                              min_sim: float = 0.0,
                              max_sim: float = 1.0) -> dict:
        """保留与参考分子相似度在 [min_sim, max_sim] 内的分子。"""
        return self._run(['filter_by_similarity',
                          '--smiles',    *list(smiles_list),
                          '--reference', reference,
                          '--min_sim',   str(min_sim),
                          '--max_sim',   str(max_sim)])


if __name__ == '__main__':
    print('This module is a client. Import PCVAEClient and call its methods.',
          file=sys.stderr)
    sys.exit(2)
