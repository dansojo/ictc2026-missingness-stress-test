"""Portable stage launcher. Running a stage is explicit; check never fits models."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))

import portable_paths as p
from release_support import verify_sources

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'extensions/2026-09-11'
PAPER = ROOT / 'original_paper_reproduction/paper'
LEGACY = ROOT / 'extensions/2026-09-10/reviewer_extension'
STAGES = {
    'base': PAPER / 'full_reproduce.py',
    'm0': LEGACY / 'run_recovery_v2.py',
    'legacy': LEGACY / 'run_recovery_v2.py',
    'same-day': EXT / 'imputation_extension/run_extension.py',
    'cross-day': EXT / 'crossday_imputation_v1/run_crossday.py',
    'donors': EXT / 'crossday_imputation_v1/donor_inventory/build_inventory_serial_partial.py',
    'donors-finalize': EXT / 'crossday_imputation_v1/donor_inventory/finalize_inventory.py',
    'external-inventory': EXT / 'external_inventory/audit_external_inventory.py',
    'canonicalize': EXT / 'external_inventory/canonicalize_studentlife.py',
    'verify-external': EXT / 'external_inventory/verify_inventory.py',
    'prediction': EXT / 'external_prediction/run_prediction.py',
}


def check(stage):
    """Validate input gates, not numerical results. Never create experiment outputs."""
    if stage == 'same-day':
        importlib.import_module('run_extension').check_pins()
    elif stage == 'cross-day':
        importlib.import_module('run_crossday').check_pins()
    elif stage in ('m0', 'legacy', 'donors', 'donors-finalize'):
        importlib.import_module('run_recovery_v2').pin_inputs()
        if stage == 'legacy':
            m = json.loads((p.role('m0') / 'manifest.json').read_text(encoding='utf8'))
            for name,h in m['source_hashes'].items():
                if p.digest(LEGACY / name) != h: p.check_source(LEGACY / name,h)
        if stage in ('donors', 'donors-finalize'):
            importlib.import_module('run_extension').check_pins()
            # Import the packaged donor extraction API, not an archive module.
            importlib.import_module('inventory_core')
    elif stage == 'prediction':
        protocol = json.loads(p.role('prediction_protocol').read_text(encoding='utf8'))
        for name, sha in protocol['sources'].items(): p.check_source(STAGES[stage].parent / name, sha)
        windows = p.resolve_input(protocol['windows_path'])
        if p.digest(windows) != protocol['windows_sha256']: raise ValueError('Prepared windows changed')
        import pandas as pd
        importlib.import_module('run_prediction').validate_windows(pd.read_parquet(windows))
    elif stage in ('external-inventory', 'verify-external'):
        # Preserve ALL source provenance pins, including the provider fold archive.
        records = json.loads((p.role('external_inventory_inputs') / 'source_manifest.json').read_text(encoding='utf8'))
        failures = []
        for record in records:
            try:
                file = p.resolve_input(record['path'])
                historical = record['path'].replace('\\', '/')
                for marker in ('/StudentLife/', '/ExtraSensory.per_uuid_features_labels/', '/02_experiments/external_data/'):
                    if marker in historical:
                        file = p.role('external_provider') / marker.strip('/') / historical.split(marker,1)[1]
                        break
                if '/paper/reference/external/' in historical:
                    file = p.role('reference') / 'external' / historical.split('/paper/reference/external/',1)[1]
                if file.stat().st_size != record['bytes'] or p.digest(file) != record['sha256']:
                    raise ValueError('Source hash mismatch')
            except (OSError, ValueError) as error: failures.append(type(error).__name__)
        if failures: raise ValueError(f'{len(failures)} external source provenance checks unavailable or mismatched')
    elif stage == 'canonicalize':
        if not (p.role('external_inventory_inputs') / 'eligible_prediction_windows.parquet').is_file():
            raise FileNotFoundError('Original prepared windows unavailable')
    elif stage == 'base':
        for key in ('raw', 'data', 'reference', 'external_inputs'):
            if not p.role(key).exists(): raise FileNotFoundError('Configured base input unavailable: ' + key)
        for folder in ('etri', 'external'):
            if not (p.role('reference') / folder).is_dir(): raise FileNotFoundError('Private comparison reference unavailable')
        from full_reproduce import external_context
        from types import SimpleNamespace
        external_context(SimpleNamespace(data_root=p.role('data'), external_inputs_json=p.role('external_inputs')))
    scopes = {
        'base': 'configured roots exist; original external input/vendor pins',
        'm0': 'base upstream stage outputs and canonical hashes',
        'legacy': 'base upstream hashes and preflight source compatibility',
        'donors': 'upstream hashes, frozen M0/legacy inputs, donor API import',
        'donors-finalize': 'upstream availability only; no generated donor output inspected',
        'same-day': 'frozen source, M0 and legacy inputs',
        'cross-day': 'all frozen files plus actually consumed M0, same-day and donor trees',
        'prediction': 'source and window pins; all prepared window contracts; no model fits',
        'external-inventory': 'all recorded provider/reference source hashes at actual consumption paths',
        'verify-external': 'source provenance only; no prepared-window verification',
        'canonicalize': 'original prepared-window file presence only',
    }
    return {'stage': stage, 'input_check': 'passed', 'checked_scope': scopes[stage],
            'models_fitted': 0, 'full_reproduction': False, 'entire_stage_verified': False}


def main():
    if sys.flags.optimize:
        raise RuntimeError('Assertions must remain enabled; do not use -O or PYTHONOPTIMIZE')
    os.environ['PYTHONOPTIMIZE'] = '0'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run'])
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, help='Original M0/same-day/cross-day bounded cell limit')
    parser.add_argument('--resume', action='store_true', help='Base pipeline completed-stage resume only')
    args = parser.parse_args()
    os.environ['ICTC_RUN_CONFIG'] = str(args.config.resolve(strict=True))
    cfg = p.configuration()
    os.environ['ICTC_PRIVATE_REFERENCE'] = str(p.role('reference'))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    paths = [ROOT / 'scripts/_support', ROOT / 'scripts', ROOT, PAPER, LEGACY, *[STAGES[k].parent for k in ('same-day','cross-day','donors','external-inventory','prediction')]]
    sys.path[:0] = list(map(str, paths))
    os.environ['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    verify_sources(ROOT)
    if args.action == 'check':
        print(json.dumps(check(args.stage), indent=2)); return 0
    if args.output is None: parser.error('run requires a private --output')
    if args.limit is not None and (args.limit < 1 or args.stage not in ('m0','same-day','cross-day')):
        parser.error('--limit is supported only for M0/same-day/cross-day and must be positive')
    if args.resume and args.stage != 'base': parser.error('--resume is available only for base')
    existing = args.stage == 'donors-finalize' or args.resume
    output = p.validate_output(args.output, existing=existing)
    if existing and not output.is_dir(): raise FileNotFoundError('Prior private stage output required')
    # Required external provenance is checked before any inventory output is written.
    if args.stage == 'external-inventory': check(args.stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg['roles']['work_output'] = str(output)
    cfg['read_roots'] = [*cfg.get('read_roots', []), str(output)]
    argv = []
    if args.stage == 'base':
        argv = ['--raw-root', str(p.role('raw')), '--data-root', str(p.role('data')),
                '--external-inputs-json', str(p.role('external_inputs')), '--output-dir', str(output)]
        if args.resume: argv += ['--resume']
    elif args.stage == 'm0': argv = ['preflight', '--output', str(output)]
    elif args.stage == 'legacy': argv = ['execute', '--preflight', str(p.role('m0')), '--output', str(output)]
    elif args.stage == 'same-day': argv = ['run', '--output', str(output)]
    elif args.stage in ('cross-day','prediction'): argv = ['--output', str(output)]
    elif args.stage == 'verify-external': argv = ['--deduplicated']
    if args.limit: argv += ['--limit', str(args.limit)]
    with tempfile.TemporaryDirectory(prefix='ictc-routing-', dir=output.parent) as temporary:
        resolved = Path(temporary) / 'paths.json'
        resolved.write_text(json.dumps(cfg, indent=2), encoding='utf8')
        os.environ['ICTC_RUN_CONFIG'] = str(resolved)
        if args.stage in ('donors','external-inventory','canonicalize','verify-external'): output.mkdir()
        result = subprocess.run([sys.executable, '-u', '-B', str(STAGES[args.stage]), *argv])
        if output.is_dir():
            lineage = {'stage': args.stage, 'exit_code': result.returncode, 'configuration_sha256': p.digest(resolved),
                       'packaged_source_manifest_sha256': p.digest(ROOT / 'docs/provenance/SOURCE_MANIFEST.json'),
                       'input_route': 'explicit private paths; original frozen protocols retained',
                       'cell_limit': args.limit, 'strict_full_comparison_passed': False}
            record = output / ('portable-lineage-' + args.stage + '.json')
            # A resumed attempt must retain the previous launcher record.
            counter = 1
            while record.exists():
                record = output / ('portable-lineage-' + args.stage + '-' + str(counter) + '.json'); counter += 1
            record.write_text(json.dumps(lineage, indent=2), encoding='utf8')
        return result.returncode


if __name__ == '__main__': raise SystemExit(main())
