#!/usr/bin/env python3
"""Portable entry point for frozen-record and external-raw paper reproduction.

This wrapper does not refit ETRI models, generate ETRI masks, or repair data.
Original scientific sources are hash-verified and executed without modification.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from private_reference import REFERENCE
import platform
import runpy
import shutil
import sys
import time
import types

PACKAGE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def io_path(path):
    """Windows long-path spelling for IO; ordinary Paths on POSIX."""
    path = Path(path).absolute()
    text = str(path)
    if os.name != 'nt' or text.startswith('\\\\?\\'):
        return path
    if text.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + text[2:])
    return Path('\\\\?\\' + text)


def plain_path(path):
    text = str(path)
    if text.startswith('\\\\?\\UNC\\'):
        text = '\\\\' + text[8:]
    elif text.startswith('\\\\?\\'):
        text = text[4:]
    return Path(text)


def digest(path):
    result = hashlib.sha256()
    with io_path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def resolve_file(root, relative):
    """Interpret manifest paths identically on Windows and POSIX; reject escape."""
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('manifest paths must be nonempty slash-separated relative paths')
    portable = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if portable.is_absolute() or windows.drive or windows.root or '..' in portable.parts:
        raise ValueError('manifest path is absolute or traverses its input root')
    root = io_path(root).resolve(strict=True)
    target = root.joinpath(*portable.parts).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError('input symlink escapes its root') from exc
    if not target.is_file():
        raise ValueError('manifest entry is not a regular file')
    return target if len(str(plain_path(target))) >= 240 else plain_path(target)


def verify_entries(entries, roots):
    checked = []
    seen = set()
    for entry in entries:
        key = (entry['root'], entry['path'])
        if key in seen:
            raise ValueError('duplicate manifest input')
        seen.add(key)
        path = resolve_file(roots[entry['root']], entry['path'])
        actual_size = path.stat().st_size
        actual_hash = digest(path)
        if actual_size != entry['bytes'] or actual_hash != entry['sha256']:
            raise ValueError(f'input byte/hash mismatch: {entry["root"]}/{entry["path"]}')
        checked.append({'root': entry['root'], 'path': entry['path'],
                        'bytes': actual_size, 'sha256': actual_hash})
    return checked


def create_output(output, input_roots):
    output = plain_path(io_path(output).resolve())
    if output.exists():
        raise FileExistsError(f'output already exists; choose a new directory: {output}')
    for root in input_roots:
        root = plain_path(io_path(root).resolve())
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('output and input roots must be disjoint')
    output.mkdir(parents=True, exist_ok=False)
    return output


def compare_csv(actual, expected, *, rel_tol=1e-12, abs_tol=1e-15):
    def read(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.reader(handle))
    a, b = read(actual), read(expected)
    mismatches = []
    if len(a) != len(b):
        mismatches.append({'reason': 'row_count', 'actual': len(a), 'expected': len(b)})
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            mismatches.append({'row': i, 'reason': 'column_count'})
            continue
        for j, (va, vb) in enumerate(zip(ra, rb)):
            if va == vb:
                continue
            try:
                fa, fb = float(va), float(vb)
                same = (math.isnan(fa) and math.isnan(fb)) or math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)
            except ValueError:
                same = False
            if not same:
                mismatches.append({'row': i, 'column': j, 'actual': va, 'expected': vb})
    return {'name': Path(actual).name, 'byte_identical': digest(actual) == digest(expected),
            'numeric_equal': not mismatches, 'mismatches': mismatches[:20],
            'relative_tolerance': rel_tol, 'absolute_tolerance': abs_tol}


def save_json(path, value):
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, 'item'):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert) + '\n', encoding='utf-8')


def verify_vendor():
    doc = json.loads((PACKAGE / 'vendor_manifest.json').read_text(encoding='utf-8'))
    entries = [dict(entry, root='package') for entry in doc['files']]
    return verify_entries(entries, {'package': PACKAGE})


def resolve_roots(args):
    data = args.data_root.resolve()
    roots = {'etri_sdd': data / 'etri/sdd',
             'etri_source': PACKAGE / 'vendor/etri_source',
             'external_project': data / 'external'}
    if args.inputs_json:
        raw = json.loads(args.inputs_json.read_text(encoding='utf-8-sig'))
        if not set(raw) <= set(roots):
            raise ValueError('unknown input-root mapping')
        for key, value in raw.items():
            path = Path(value)
            if not path.is_absolute():
                raise ValueError('explicit local input roots must be absolute')
            roots[key] = path.resolve()
    return roots


def load_vendor(name):
    path = PACKAGE / 'vendor' / name
    spec = importlib.util.spec_from_file_location('paper_vendor_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_audit_paths(roots, run):
    """Replace machine-specific path configuration, never scientific code/guards."""
    import pandas as pd
    module = types.ModuleType('audit_paths')
    module.Path, module.hashlib, module.json, module.sys = Path, hashlib, json, sys
    module.RUN = run
    module.SDD = io_path(roots['etri_sdd'])
    module.DIAG = module.SDD / 'task-12-etri-production_20260810_230202'
    module.DOWN = module.SDD / 'task-12-downstream-etri-production-v1'
    module.xp = io_path
    module.digest = digest
    module.read_text = lambda path: io_path(path).read_text(encoding='utf-8-sig')
    module.read_json = lambda path: json.loads(module.read_text(path))
    module.save_json = lambda name, value: save_json(run / name, value)
    module.parquet = lambda path, columns=None: pd.read_parquet(io_path(path), columns=columns)

    def arrow(path, columns=None):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        with pa.memory_map(str(io_path(path)), 'r') as source:
            try:
                table = ipc.open_file(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                table = ipc.open_stream(source).read_all()
            if columns is not None:
                table = table.select(columns)
            return table.to_pandas()
    module.arrow = arrow
    previous = sys.modules.get('audit_paths')
    sys.modules['audit_paths'] = module
    return previous


def records(args, roots, output):
    audit = output / 'recomputed'
    audit.mkdir()
    previous = install_audit_paths(roots, audit)
    try:
        for name in ['recompute_primary.py', 'recompute_downstream.py']:
            with (output / (name + '.log')).open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                runpy.run_path(str(PACKAGE / 'vendor/audit' / name), run_name='__main__')
    finally:
        if previous is None:
            sys.modules.pop('audit_paths', None)
        else:
            sys.modules['audit_paths'] = previous
    checks = []
    for name in ['primary_recomputation.json', 'downstream_recomputation.json']:
        doc = json.loads((audit / name).read_text(encoding='utf-8'))
        checks.extend(doc['consistency_checks'])
        checks.extend({'check': row['claim'], 'pass': row['match_at_printed_precision']} for row in doc['paper_claims'])
    if not all(row['pass'] for row in checks):
        save_json(output / 'failed_calculation_checks.json', checks)
        raise ValueError('independent recomputation disagrees with frozen records/paper')
    evidence = output / 'evidence'
    old_argv = sys.argv
    sys.argv = ['extract_etri_paper_evidence.py', '--sdd-dir', str(io_path(roots['etri_sdd'])),
                '--source-root', str(io_path(roots['etri_source'])), '--output-dir', str(evidence)]
    try:
        with (output / 'reporting.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            runpy.run_path(str(PACKAGE / 'vendor/extract_etri_paper_evidence.py'), run_name='__main__')
    finally:
        sys.argv = old_argv
    comparisons = compare_results(evidence, REFERENCE / 'etri')
    if not comparisons['scientific_equal']:
        save_json(output / 'failed_reporting_comparison.json', comparisons)
        raise ValueError('reporting output disagrees with frozen reference')
    return {'execution': 'independent calculation from archived feature replay values and saved probabilities, followed by original reporting extraction',
            'new_etri_masks': False, 'new_etri_feature_extraction': False, 'new_model_fits': 0,
            'checks': len(checks), 'all_calculation_checks_passed': True,
            'comparison': comparisons,
            'historical_report_note': 'The unchanged manuscript_evidence.md says external analyses were pending at its generation time. The current paper classifies completed external results as post-plan exploratory directional replication.'}


def compare_results(actual, expected):
    rows = []
    for reference in sorted(expected.iterdir()):
        if not reference.is_file() or reference.name.endswith('_manifest.json') or reference.name == 'evidence_manifest.json':
            continue
        result = actual / reference.name
        if not result.is_file():
            rows.append({'name': reference.name, 'equal': False, 'missing': True})
        elif reference.suffix == '.csv':
            comparison = compare_csv(result, reference)
            comparison['equal'] = comparison['numeric_equal']
            rows.append(comparison)
        elif reference.suffix == '.json':
            rows.append({'name': reference.name, 'equal': json.loads(result.read_text()) == json.loads(reference.read_text()),
                         'byte_identical': digest(result) == digest(reference)})
        else:
            rows.append({'name': reference.name, 'equal': result.read_text(encoding='utf-8-sig') == reference.read_text(encoding='utf-8-sig'),
                         'byte_identical': digest(result) == digest(reference)})
    return {'scientific_equal': bool(rows) and all(row['equal'] for row in rows), 'files': rows,
            'path_bearing_manifest_note': 'Original source-file manifests are generated and retained; absolute path keys change when inputs are relocated.'}


def external(args, roots, output):
    module = load_vendor('external_transfer.py')
    with (output / 'external.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        manifest = module.run(io_path(roots['external_project']), output / 'raw_results')
    comparisons = compare_results(output / 'raw_results', REFERENCE / 'external')
    # This manifest has no absolute source paths, so equality is meaningful here.
    comparisons['manifest_byte_identical'] = digest(output / 'raw_results/external_transfer_manifest.json') == digest(REFERENCE / 'external/external_transfer_manifest.json')
    if not comparisons['scientific_equal']:
        save_json(output / 'failed_external_comparison.json', comparisons)
        raise ValueError('external raw output differs from the frozen scientific reference')
    return {'execution': 'unchanged external source: validate all pinned source files, reload raw data, regenerate days/masks/features/participant intervals',
            'external_predictive_models_fitted': 0, 'external_labels_used': False,
            'historical_code_label': manifest['decision'],
            'interpretation': 'post-plan exploratory directional replication; no formal transfer or predictive-stability conclusion',
            'comparison': comparisons}


def displays(args, roots, output):
    evidence = args.evidence.resolve(strict=True)
    external_dir = args.external_results.resolve(strict=True)
    # The original renderer has fixed annotations. Only its frozen evidence is accepted.
    etri_compare = compare_results(evidence, REFERENCE / 'etri')
    external_compare = compare_results(external_dir, REFERENCE / 'external')
    if not etri_compare['scientific_equal'] or not external_compare['scientific_equal']:
        raise ValueError('fixed-layout display source is not the verified frozen evidence')
    stage = output / 'display_inputs/paper_ictc2026/02_experiments'
    for source, dest in [(evidence, stage / 'etri_paper_evidence'),
                         (external_dir, stage / 'external_transfer/results')]:
        dest.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copyfile(path, dest / path.name)
    module = load_vendor('generate_final_displays.py')
    manifest = module.build(output / 'display_inputs', output / 'figures', output / 'tables')
    comparison = []
    for directory, name in [('figures','fig2_results.png'), ('tables','table1_feature_contracts.tex'), ('tables','table2_external_transfer.tex')]:
        actual, reference = output / directory / name, REFERENCE / 'displays' / name
        row = {'name': name, 'byte_identical': digest(actual) == digest(reference)}
        if name.endswith('.tex'):
            row['content_equal'] = actual.read_text(encoding='utf-8-sig') == reference.read_text(encoding='utf-8-sig')
        else:
            from PIL import Image
            with Image.open(actual) as a, Image.open(reference) as b:
                row['content_equal'] = a.size == b.size and a.convert('RGBA').tobytes() == b.convert('RGBA').tobytes()
        comparison.append(row)
    from pypdf import PdfReader
    a = PdfReader(output / 'figures/fig2_results.pdf')
    b = PdfReader(REFERENCE / 'displays/fig2_results.pdf')
    pdf_equal = len(a.pages) == len(b.pages) and all(x.get_contents().get_data() == y.get_contents().get_data() for x,y in zip(a.pages,b.pages))
    corrected = args.table_wording == 'corrected'
    if corrected:
        table = output / 'tables/table1_feature_contracts.tex'
        old = 'Mean valid nonnegative duration or transformed intensity'
        new = r'Sum of valid usage durations; mean of $\log(1+\mathrm{lux})$'
        text = table.read_text(encoding='utf-8')
        if text.count(old) != 1:
            raise ValueError('historical contract wording anchor is missing')
        table.write_text(text.replace(old, new), encoding='utf-8')
    manifest['table_wording'] = args.table_wording
    manifest['corrected_wording_note'] = 'usage is summed; each valid raw lux value is log1p transformed before its mean; empirical numeric entries unchanged'
    for directory, name in [('figures','fig2_results.pdf'),('figures','fig2_results.png'),('tables','table1_feature_contracts.tex'),('tables','table2_external_transfer.tex')]:
        p = output / directory / name
        manifest['generated_files'][name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    save_json(output / 'display_manifest.json', manifest)
    return {'execution': 'original display build from regenerated input CSVs; copied lightweight display inputs are not new calculations',
            'comparison_before_optional_wording_edit': comparison, 'pdf_page_streams_identical': pdf_equal,
            'all_png_tex_content_equal': all(row['content_equal'] for row in comparison),
            'table_wording': args.table_wording, 'manuscript_tex_compiled': False,
            'portability_note': 'PNG pixels and PDF streams can differ with fonts/Matplotlib/platform. Scientific CSV equality and normalized TeX text are checked separately.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-site', type=Path, help='explicit local optional dependency directory; unnecessary in the requirements environment')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['verify-inputs', 'records', 'external', 'displays']:
        p = sub.add_parser(name)
        p.add_argument('--data-root', type=Path, default=Path('/data'))
        p.add_argument('--inputs-json', type=Path, help='local-only absolute root mapping; do not redistribute private paths')
        p.add_argument('--manifest', type=Path, default=PACKAGE / 'input_manifest.json')
        if name == 'verify-inputs':
            p.add_argument('--group', choices=['records','external'], required=True)
        else:
            p.add_argument('--output', type=Path, required=True)
        if name == 'displays':
            p.add_argument('--evidence', type=Path, required=True)
            p.add_argument('--external-results', type=Path, required=True)
            p.add_argument('--table-wording', choices=['submitted','corrected'], default='submitted')
    args = parser.parse_args(argv)
    if args.extra_site:
        sys.path.append(str(args.extra_site.resolve(strict=True)))
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    verified_source = verify_vendor()
    roots = resolve_roots(args)
    group = args.group if args.command == 'verify-inputs' else args.command
    checked = []
    if group in ['records','external']:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        checked = verify_entries(manifest['groups'][group], roots)
    if args.command == 'verify-inputs':
        print(json.dumps({'status':'INPUTS_VERIFIED','group':group,'files':len(checked),'source_files':len(verified_source)}))
        return 0
    relevant_roots = [roots['etri_sdd'], roots['etri_source']] if args.command == 'records' else [roots['external_project']]
    if args.command == 'displays':
        relevant_roots = [args.evidence, args.external_results]
    output = create_output(args.output, relevant_roots + [PACKAGE / 'vendor', REFERENCE])
    # Isolate Matplotlib's writable cache from the archived source and user profile.
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib')
    started = time.monotonic()
    base_report = {'command':args.command, 'python':sys.version, 'platform':platform.platform(),
                   'source_files_verified':verified_source, 'input_files_verified':checked,
                   'status':'RUNNING','new_imputation_experiment':False}
    save_json(output / 'run_manifest.json', base_report)
    try:
        result = {'records':records,'external':external,'displays':displays}[args.command](args, roots, output)
    except Exception as exc:
        base_report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        save_json(output / 'run_manifest.json', base_report)
        raise
    base_report.update(status='COMPLETE_WITH_STATED_SCOPE', elapsed_seconds=time.monotonic()-started, result=result)
    save_json(output / 'run_manifest.json', base_report)
    print(json.dumps({'status':base_report['status'],'command':args.command,'output':str(output),'elapsed_seconds':base_report['elapsed_seconds']},ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
