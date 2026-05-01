"""
pcvae_tool.cli  ─  JSON-stdout CLI 入口（供跨环境 agent 通过 subprocess 调用）

用法
----
    python -m pcvae_tool.cli <command> [options...]

命令 — 焓预测
--------------
    predict_enthalpy            --smiles SMI
    predict_enthalpies          --smiles SMI [SMI ...]
    get_enthalpy_range

命令 — 分子生成
---------------
    generate_random             [--num_samples N] [--no_save] [--output_dir D]
    generate_by_enthalpy        --enthalpy E [--num_samples N] ...
    generate_by_smiles          --smiles SMI [--num_samples N] ...
    generate_by_smiles_and_enthalpy
                                --smiles SMI --enthalpy E [--num_samples N] ...

命令 — SMILES 工具
------------------
    smiles_validate             --smiles SMI
    smiles_batch_validate       --smiles SMI [SMI ...]
    smiles_canonicalize         --smiles SMI
    smiles_batch_canonicalize   --smiles SMI [SMI ...]
    smiles_properties           --smiles SMI
    smiles_batch_properties     --smiles SMI [SMI ...] [--keep_invalid]
    smiles_deduplicate          --smiles SMI [SMI ...] [--keep_invalid]

命令 — 相似度
-------------
    tanimoto                    --smiles_a SMI --smiles_b SMI
    most_similar                --query SMI --smiles SMI [SMI ...] [--top_n N]
    similarity_matrix           --smiles SMI [SMI ...]
    diversity_score             --smiles SMI [SMI ...]
    novelty_score               --generated SMI [SMI ...] --reference SMI [SMI ...]
                                [--threshold F]
    filter_by_similarity        --smiles SMI [SMI ...] --reference SMI
                                [--min_sim F] [--max_sim F]

约定
----
- 成功：stdout 输出一行 JSON（dict），exit code = 0
- 失败：stdout 输出 {"error": "<message>", "type": "<ExceptionClass>"}, exit code = 1
- 所有 RDKit / 其他 stderr 噪声不会污染 stdout（已禁用日志）。
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback


def _silence_rdkit():
    """关闭 RDKit C++ 日志，避免污染 stdout JSON。"""
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass


def _emit(payload: dict, code: int = 0):
    """把 dict 序列化为单行 JSON 写到 stdout，并以指定状态码退出。"""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=float))
    sys.stdout.write('\n')
    sys.stdout.flush()
    sys.exit(code)


def _emit_error(exc: BaseException):
    _emit({
        'error': str(exc),
        'type':  type(exc).__name__,
        'traceback': traceback.format_exc(),
    }, code=1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='pcvae_tool.cli')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('get_enthalpy_range')

    sp = sub.add_parser('predict_enthalpy')
    sp.add_argument('--smiles', required=True)

    sp = sub.add_parser('predict_enthalpies')
    sp.add_argument('--smiles', required=True, nargs='+')

    def _gen_common(sp):
        sp.add_argument('--num_samples', type=int, default=100)
        sp.add_argument('--no_save', action='store_true',
                        help='Do not write output files (only return JSON).')
        sp.add_argument('--output_dir', default=None,
                        help='Override default output directory.')

    sp = sub.add_parser('generate_random');                _gen_common(sp)

    sp = sub.add_parser('generate_by_enthalpy');           _gen_common(sp)
    sp.add_argument('--enthalpy', type=float, required=True)

    sp = sub.add_parser('generate_by_smiles');             _gen_common(sp)
    sp.add_argument('--smiles', required=True)

    sp = sub.add_parser('generate_by_smiles_and_enthalpy'); _gen_common(sp)
    sp.add_argument('--smiles',   required=True)
    sp.add_argument('--enthalpy', type=float, required=True)

    # ── SMILES 工具 ───────────────────────────────────────────────────────────
    sp = sub.add_parser('smiles_validate')
    sp.add_argument('--smiles', required=True)

    sp = sub.add_parser('smiles_batch_validate')
    sp.add_argument('--smiles', required=True, nargs='+')

    sp = sub.add_parser('smiles_canonicalize')
    sp.add_argument('--smiles', required=True)

    sp = sub.add_parser('smiles_batch_canonicalize')
    sp.add_argument('--smiles', required=True, nargs='+')

    sp = sub.add_parser('smiles_properties')
    sp.add_argument('--smiles', required=True)

    sp = sub.add_parser('smiles_batch_properties')
    sp.add_argument('--smiles', required=True, nargs='+')
    sp.add_argument('--keep_invalid', action='store_true')

    sp = sub.add_parser('smiles_deduplicate')
    sp.add_argument('--smiles', required=True, nargs='+')
    sp.add_argument('--keep_invalid', action='store_true')

    # ── 相似度工具 ────────────────────────────────────────────────────────────
    sp = sub.add_parser('tanimoto')
    sp.add_argument('--smiles_a', required=True)
    sp.add_argument('--smiles_b', required=True)

    sp = sub.add_parser('most_similar')
    sp.add_argument('--query',  required=True)
    sp.add_argument('--smiles', required=True, nargs='+')
    sp.add_argument('--top_n',  type=int, default=5)

    sp = sub.add_parser('similarity_matrix')
    sp.add_argument('--smiles', required=True, nargs='+')

    sp = sub.add_parser('diversity_score')
    sp.add_argument('--smiles', required=True, nargs='+')

    sp = sub.add_parser('novelty_score')
    sp.add_argument('--generated', required=True, nargs='+')
    sp.add_argument('--reference', required=True, nargs='+')
    sp.add_argument('--threshold', type=float, default=0.4)

    sp = sub.add_parser('filter_by_similarity')
    sp.add_argument('--smiles',    required=True, nargs='+')
    sp.add_argument('--reference', required=True)
    sp.add_argument('--min_sim',   type=float, default=0.0)
    sp.add_argument('--max_sim',   type=float, default=1.0)

    return p


def _gen_kwargs(args) -> dict:
    kwargs = {'num_samples': args.num_samples, 'save': not args.no_save}
    if args.output_dir:
        kwargs['output_dir'] = args.output_dir
    return kwargs


def main(argv=None):
    _silence_rdkit()
    args = _build_parser().parse_args(argv)

    try:
        import pcvae_tool as pt
        from pcvae_tool.tools import smiles_utils_tool as su
        from pcvae_tool.tools import similarity_tool as sim
    except Exception as e:
        _emit_error(e)

    try:
        # ── 焓预测 ────────────────────────────────────────────────────────────
        if args.cmd == 'get_enthalpy_range':
            _emit(pt.get_enthalpy_range())
        elif args.cmd == 'predict_enthalpy':
            _emit(pt.predict_enthalpy(args.smiles))
        elif args.cmd == 'predict_enthalpies':
            _emit(pt.predict_enthalpies(args.smiles))

        # ── 分子生成 ──────────────────────────────────────────────────────────
        elif args.cmd == 'generate_random':
            _emit(pt.generate_random(**_gen_kwargs(args)))
        elif args.cmd == 'generate_by_enthalpy':
            _emit(pt.generate_by_enthalpy(enthalpy=args.enthalpy,
                                          **_gen_kwargs(args)))
        elif args.cmd == 'generate_by_smiles':
            _emit(pt.generate_by_smiles(smiles=args.smiles,
                                        **_gen_kwargs(args)))
        elif args.cmd == 'generate_by_smiles_and_enthalpy':
            _emit(pt.generate_by_smiles_and_enthalpy(
                smiles=args.smiles, enthalpy=args.enthalpy,
                **_gen_kwargs(args)))

        # ── SMILES 工具 ───────────────────────────────────────────────────────
        elif args.cmd == 'smiles_validate':
            _emit(su.validate(args.smiles))
        elif args.cmd == 'smiles_batch_validate':
            _emit(su.batch_validate(args.smiles))
        elif args.cmd == 'smiles_canonicalize':
            _emit(su.canonicalize(args.smiles))
        elif args.cmd == 'smiles_batch_canonicalize':
            _emit(su.batch_canonicalize(args.smiles))
        elif args.cmd == 'smiles_properties':
            _emit(su.get_properties(args.smiles))
        elif args.cmd == 'smiles_batch_properties':
            _emit(su.batch_properties(args.smiles,
                                      skip_invalid=not args.keep_invalid))
        elif args.cmd == 'smiles_deduplicate':
            _emit(su.deduplicate(args.smiles,
                                 keep_invalid=args.keep_invalid))

        # ── 相似度工具 ────────────────────────────────────────────────────────
        elif args.cmd == 'tanimoto':
            _emit(sim.tanimoto(args.smiles_a, args.smiles_b))
        elif args.cmd == 'most_similar':
            _emit(sim.most_similar(args.query, args.smiles, top_n=args.top_n))
        elif args.cmd == 'similarity_matrix':
            _emit(sim.similarity_matrix(args.smiles))
        elif args.cmd == 'diversity_score':
            _emit(sim.diversity_score(args.smiles))
        elif args.cmd == 'novelty_score':
            _emit(sim.novelty_score(args.generated, args.reference,
                                    threshold=args.threshold))
        elif args.cmd == 'filter_by_similarity':
            _emit(sim.filter_by_similarity(args.smiles, args.reference,
                                           min_sim=args.min_sim,
                                           max_sim=args.max_sim))
        else:
            _emit_error(ValueError(f'Unknown command: {args.cmd}'))

    except SystemExit:
        raise
    except BaseException as e:
        _emit_error(e)


if __name__ == '__main__':
    main()
