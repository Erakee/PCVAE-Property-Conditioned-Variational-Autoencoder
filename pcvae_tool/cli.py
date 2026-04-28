"""
pcvae_tool.cli  ─  JSON-stdout CLI 入口（供跨环境 agent 通过 subprocess 调用）

用法
----
    python -m pcvae_tool.cli <command> [options...]

命令
----
    predict_enthalpy            --smiles SMI
    predict_enthalpies          --smiles SMI [SMI ...]
    get_enthalpy_range
    generate_random             [--num_samples N] [--no_save] [--output_dir D]
    generate_by_enthalpy        --enthalpy E [--num_samples N] ...
    generate_by_smiles          --smiles SMI [--num_samples N] ...
    generate_by_smiles_and_enthalpy
                                --smiles SMI --enthalpy E [--num_samples N] ...

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
    except Exception as e:
        _emit_error(e)

    try:
        if args.cmd == 'get_enthalpy_range':
            _emit(pt.get_enthalpy_range())

        elif args.cmd == 'predict_enthalpy':
            _emit(pt.predict_enthalpy(args.smiles))

        elif args.cmd == 'predict_enthalpies':
            _emit(pt.predict_enthalpies(args.smiles))

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

        else:
            _emit_error(ValueError(f'Unknown command: {args.cmd}'))

    except SystemExit:
        raise
    except BaseException as e:
        _emit_error(e)


if __name__ == '__main__':
    main()
