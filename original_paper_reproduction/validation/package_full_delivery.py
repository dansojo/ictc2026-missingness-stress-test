#!/usr/bin/env python3
"""Prepare a private-path-free full delivery only after the complete experiment passes.

Default mode checks evidence and temporary staging without creating a ZIP. An
explicit --create and a new --version are required to publish a local archive.
Raw data, fit states and the original machine-specific proof documents stay
outside the package. No scientific run, upload or historical ZIP rewrite occurs.

Explicit --review-differences creates a separately named REVIEW archive of a
fully executed run whose strict comparison failed. It preserves every failure
and never declares strict reproduction successful or the archive submission-ready.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_STAGES = ('prepare', 'g1', 'g2_scenarios', 'g2', 'stress', 'downstream')
G2_CHECKPOINT_COMMIT = '808a6f6063026b5c4928d191cf1a480ed2886955'
G2_CHECKPOINT_TREE = 'c183d9c36b0bde077ebddcdc51f801241c880e1f868026c5ed0c910f53a97c76'
SANITIZED_MANIFESTS = {'paper/full_vendor_manifest.json', 'leaderboard/preprocessing_manifest.json'}
VALIDATION_SCRIPTS = {'validation/package_full_delivery.py', 'validation/compare_full_displays.py'}
DISPLAY_FILES = ('fig2_results.png', 'fig2_results.pdf', 'table1_feature_contracts.tex', 'table2_external_transfer.tex')
ROOT_DOCUMENTS = {'README.md', 'MODEL_DESCRIPTION.md', 'MODEL_DESCRIPTION.pdf', 'VERIFICATION_REPORT.md'}
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
SAFE_EXTENSIONS = {'.py', '.json', '.md', '.txt', '.csv', '.tex', '.png', '.pdf', '.toml', '.yaml', '.yml'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(document) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relative_path(value: str) -> PurePosixPath:
    """Use the same safe archive spelling on Windows and POSIX."""
    if not isinstance(value, str) or '\\' in value or '\x00' in value or ':' in value:
        raise ValueError('Invalid portable relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('Invalid portable relative path')
    if any(part.rstrip(' .') != part or part.split('.')[0].upper() in
           {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)),
            *('LPT' + str(i) for i in range(1, 10))} for part in path.parts):
        raise ValueError('Unsafe portable path component')
    return path


def under(root: Path, relative: str) -> Path:
    rel = relative_path(relative)
    root = root.resolve(strict=True)
    path = root.joinpath(*rel.parts)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, 'is_junction') and ancestor.is_junction()):
            raise ValueError('Linked delivery/input paths are not allowed')
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError('Path escapes root')
    return resolved


def verify_pin(root: Path, pin: dict) -> Path:
    if (not isinstance(pin, dict) or not isinstance(pin.get('path'), str)
            or not isinstance(pin.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', pin['sha256']) is None):
        raise ValueError('Required file digest pin is missing or malformed')
    path = under(root, pin['path'])
    if not path.is_file() or sha256(path) != pin.get('sha256'):
        raise ValueError('Pinned file missing or digest changed: ' + pin['path'])
    if 'bytes' in pin and pin['bytes'] != path.stat().st_size:
        raise ValueError('Pinned file size changed: ' + pin['path'])
    return path


def require_complete_proof(pipeline: dict, comparison: dict, downstream: dict) -> None:
    if (pipeline.get('schema_name') != 'independent_etri_paper_pipeline'
            or pipeline.get('status') != 'complete' or pipeline.get('new_model_fits') != 160
            or pipeline.get('scientific_experiment_executed') is not True):
        raise ValueError('Packaging requires a completed fresh full pipeline')
    if (comparison.get('schema_name') != 'independent_etri_detailed_comparison_v1'
            or comparison.get('status') != 'complete' or comparison.get('full_scope') is not True
            or comparison.get('all_scientific_equal') is not True
            or len(comparison.get('requested_stages', [])) != len(COMPARISON_STAGES)
            or set(comparison.get('requested_stages', [])) != set(COMPARISON_STAGES)
            or set(comparison.get('stages', {})) != set(COMPARISON_STAGES)
            or any(item.get('scientific_equal') is not True for item in comparison.get('stages', {}).values())):
        raise ValueError('Packaging requires every full-scope scientific comparison to pass')
    if (downstream.get('schema_name') != 'independent_etri_reproduction'
            or downstream.get('stage') != 'downstream' or downstream.get('status') != 'complete'
            or downstream.get('fresh_execution') is not True or downstream.get('new_model_fits') != 160
            or downstream.get('reused_historical_model_fits') != 0 or downstream.get('prediction_rows') != 943392
            or downstream.get('expected_primary_key_count') != 314464):
        raise ValueError('Packaging requires 160 new fits, zero reused fits and all 943392 predictions')


def require_matching_fit_report(report: dict) -> list[dict]:
    fits = report.get('fits', [])
    if (report.get('both_packs_validated') is not True or report.get('actual_fit_count') != 160
            or report.get('reference_fit_count') != 160 or report.get('scientific_equal') is not True
            or len(fits) != 160 or [item.get('fit_seq') for item in fits] != list(range(160))
            or any(item.get('scientific_equal') is not True for item in fits)):
        raise ValueError('Detailed comparison lacks all 160 validated matching fits')
    return fits


def require_review_proof(pipeline: dict, comparison: dict, downstream: dict) -> None:
    """Require completed generation and an actual failed full comparison, not a pass."""
    if (pipeline.get('schema_name') != 'independent_etri_paper_pipeline'
            or pipeline.get('status') != 'comparison_failed'
            or pipeline.get('failed_step') != 'detailed_comparison'
            or pipeline.get('new_model_fits') != 160
            or pipeline.get('scientific_experiment_executed') is not True):
        raise ValueError('Review requires complete generation with failure only at detailed comparison')
    if (comparison.get('schema_name') != 'independent_etri_detailed_comparison_v1'
            or comparison.get('status') != 'failed' or comparison.get('full_scope') is not True
            or comparison.get('all_scientific_equal') is not False
            or len(comparison.get('requested_stages', [])) != len(COMPARISON_STAGES)
            or set(comparison.get('requested_stages', [])) != set(COMPARISON_STAGES)
            or set(comparison.get('stages', {})) != set(COMPARISON_STAGES)
            or not any(item.get('scientific_equal') is False for item in comparison['stages'].values())):
        raise ValueError('Review requires an explicit full-scope comparison failure')
    if (downstream.get('schema_name') != 'independent_etri_reproduction'
            or downstream.get('stage') != 'downstream' or downstream.get('status') != 'complete'
            or downstream.get('fresh_execution') is not True or downstream.get('new_model_fits') != 160
            or downstream.get('reused_historical_model_fits') != 0 or downstream.get('prediction_rows') != 943392
            or downstream.get('expected_primary_key_count') != 314464):
        raise ValueError('Review requires every fresh fit and prediction to have completed')


def sanitize_manifest(relative: str, document):
    """The only authorized transformations; never alter the source document."""
    result = copy.deepcopy(document)
    if relative == 'paper/full_vendor_manifest.json':
        for row in result:
            row.pop('source', None)
    elif relative == 'leaderboard/preprocessing_manifest.json':
        for row in result['sources']:
            row.pop('source_path', None)
    else:
        raise ValueError('No staging transformation authorized for ' + relative)
    return result


def assert_no_private_paths(name: str, data: bytes) -> None:
    # Normalize JSON escaping and percent escaping before scanning. Code is
    # rejected, never rewritten. Generic caller-selected /data examples are valid.
    from urllib.parse import unquote
    text = unquote(data.decode('utf-8', errors='replace'))
    text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda match: chr(int(match.group(1), 16)), text)
    text = text.replace('\\', '/')
    text = re.sub('/+', '/', text)
    private = (r'(?i)(?:[a-z]:)?/' + r'users/[^/\s"<>]+(?:/|\b)',
               r'(?i)/' + r'home/[^/\s"<>]+(?:/|\b)', r'(?i)/' + r'documents and settings/',
               r'(?i)(?:^|[\s"=:])(?:[a-z]:)?/[^\n"<>]*onedrive(?:/|\b)',
               r'(?i)file:/+(?:[a-z]:/)?(?:users|home)/')
    if any(re.search(pattern, text) for pattern in private):
        raise ValueError('A private local path appears in ' + name)
    if name.lower().endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Cannot inspect an encrypted delivery PDF')
        extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
        extracted += '\n' + str(reader.metadata)
        assert_no_private_paths(name + '.extracted_text', extracted.encode('utf-8'))
        # Include annotations and compressed PDF stream text, not just visible
        # pages. A source-path string can otherwise survive in PDF metadata.
        visited = set()

        def inspect_pdf(value):
            value = value.get_object() if hasattr(value, 'get_object') else value
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, (str, bytes)):
                assert_no_private_paths(name + '.object', value.encode('utf-8') if isinstance(value, str) else value)
            elif isinstance(value, dict):
                if hasattr(value, 'get_data'):
                    assert_no_private_paths(name + '.stream', value.get_data())
                for child in value.values():
                    inspect_pdf(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    inspect_pdf(child)

        inspect_pdf(reader.trailer)
    elif name.lower().endswith('.png'):
        from PIL import Image
        with Image.open(io.BytesIO(data)) as picture:
            assert_no_private_paths(name + '.metadata', str(picture.info).encode('utf-8'))


def delivery_allowed(relative: str) -> bool:
    try:
        path = relative_path(relative)
    except ValueError:
        return False
    if any(part.startswith('local_') or part in {'.venv', 'venv', '__pycache__', '.git', '.codex',
           'raw', 'processed', 'results', 'output', 'outputs', 'node_modules'} for part in path.parts):
        return False
    if any(part.startswith('.') and part != '.gitkeep' for part in path.parts):
        return False
    if relative.startswith('paper/g2_vendor/original_runner/'):
        return False
    if path.name in {'IMPLEMENTATION_PLAN.md', 'PACKAGE_MANIFEST.json'}:
        return False
    if relative in ROOT_DOCUMENTS or relative in VALIDATION_SCRIPTS:
        return True
    if path.parts[0] not in {'paper', 'leaderboard', 'tests'}:
        return False
    if path.suffix in {'.parquet', '.arrow'}:
        return any(relative.startswith(prefix) for prefix in ('paper/tests/fixtures/', 'leaderboard/tests/fixtures/', 'tests/fixtures/'))
    return path.suffix.lower() in SAFE_EXTENSIONS or path.name == '.gitkeep'


def verify_source_manifests(root: Path) -> dict:
    full = read_json(root / 'paper/full_vendor_manifest.json')
    if len(full) != 24 or len({row['path'] for row in full}) != 24:
        raise ValueError('Exactly 24 immutable scientific sources are required')
    for pin in full:
        verify_pin(root / 'paper/full_vendor', pin)
    checkpoint = read_json(root / 'paper/g2_vendor_manifest.json')
    if (checkpoint.get('schema_name') != 'ictc_g2_checkpoint_sources_v1'
            or checkpoint.get('commit') != G2_CHECKPOINT_COMMIT
            or checkpoint.get('code_tree_sha256') != G2_CHECKPOINT_TREE
            or len(checkpoint.get('files', [])) != 7
            or len({pin['original_path'] for pin in checkpoint.get('files', [])}) != 7):
        raise ValueError('The exact seven-file G2 execution checkpoint is required')
    physical = [pin for pin in checkpoint['files'] if pin.get('role') == 'scientific_module']
    provenance = [pin for pin in checkpoint['files'] if pin.get('role') == 'provenance_only']
    if len(physical) != 4 or len(provenance) != 3 or any(pin.get('path') is not None for pin in provenance):
        raise ValueError('G2 checkpoint requires four physical modules and three provenance-only digest records')
    for pin in checkpoint['files']:
        if pin['path'] is None:
            continue
        if not pin['path'].startswith('g2_vendor/semantic_indicators/'):
            raise ValueError('G2 source is outside the checkpoint vendor tree')
        verify_pin(root / 'paper', pin)
    assert_no_private_paths('paper/g2_vendor_manifest.json', json_bytes(checkpoint))
    reporting = read_json(root / 'paper/vendor_manifest.json')['files']
    if len(reporting) != 11 or len({row['path'] for row in reporting}) != 11:
        raise ValueError('Reporting/external source set is incomplete')
    for pin in reporting:
        verify_pin(root / 'paper', pin)
    preprocessing = read_json(root / 'leaderboard/preprocessing_manifest.json')
    sources = preprocessing['sources']
    if (len(sources) != 8 or len({row['vendor_relative_path'] for row in sources}) != 8
            or preprocessing['stages'] != [row['vendor_relative_path'] for row in sources]
            or preprocessing.get('scientific_source_changes') != []):
        raise ValueError('The eight unchanged preprocessing sources are required')
    for row in sources:
        if row['vendor_sha256'] != row['sha256']:
            raise ValueError('Preprocessing source pin no longer matches original')
        verify_pin(root / 'leaderboard/vendor/preprocessing', dict(row, path=row['vendor_relative_path']))
    # Parse the training source declaration without executing training/imports.
    tree = ast.parse((root / 'leaderboard/reproduce.py').read_bytes())
    declared = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'VENDOR_SHA' for target in node.targets)]
    if len(declared) != 1:
        raise ValueError('Leaderboard model source declaration is missing')
    model_path = 'leaderboard/vendor/modeling/tree_model_reliability_submit.py'
    verify_pin(root, {'path': model_path, 'sha256': declared[0]})
    return {'full_scientific_sources': [{'path': row['path'], 'sha256': row['sha256']} for row in full],
            'g2_checkpoint_sources': checkpoint,
            'reporting_sources': [{key: row[key] for key in ('path', 'sha256', 'bytes')} for row in reporting],
            'preprocessing_sources': [{'path': row['vendor_relative_path'], 'sha256': row['sha256'], 'bytes': row['bytes']} for row in sources],
            'leaderboard_model_source': {'path': model_path, 'sha256': declared[0]}}


def comparison_bindings(run: Path, report: dict, artifacts: dict, *, expected_roles: dict | None = None,
                        expected_artifacts: dict | None = None, review_differences: bool = False) -> dict:
    """Bind comparison to actual files; explicit review preserves failed results."""
    result = {}
    for stage, item in report['stages'].items():
        tables = item.get('tables', {})
        if expected_roles is not None and set(tables) != set(expected_roles[stage]):
            raise ValueError('Scientific comparison table scope is incomplete: ' + stage)
        if stage != 'g1' and not tables:
            raise ValueError('Comparison has no scientific tables for ' + stage)
        portable_tables = {}
        for role, table in tables.items():
            if table.get('scientific_equal') is not True and not review_differences:
                raise ValueError('A failed table is hidden in the comparison')
            if table.get('declared_empty') is True:
                a, b = table.get('actual_declaration', {}), table.get('reference_declaration', {})
                if any(pin.get('path') is not None or pin.get('rows') != 0 or pin.get('columns') != [] for pin in (a, b)):
                    raise ValueError('Invalid declared-empty comparison')
                if expected_artifacts is not None and expected_artifacts[stage][role] is not None:
                    raise ValueError('A materialized stage artifact was incorrectly compared as empty')
                portable_tables[role] = {'declared_empty': True, 'scientific_equal': True}
                continue
            actual = Path(table.get('actual_path', '')).resolve()
            if not actual.is_relative_to(run.resolve()):
                raise ValueError('Comparison points to another experiment')
            relative = actual.relative_to(run.resolve()).as_posix()
            if expected_artifacts is not None and relative != expected_artifacts[stage][role]:
                raise ValueError('Scientific comparison role points to the wrong stage artifact')
            pin = artifacts.get(relative)
            if pin is None or table.get('actual_sha256') != pin['sha256']:
                raise ValueError('Comparison artifact digest differs from completed stages')
            if table.get('actual_rows') != pin.get('rows') or table.get('actual_columns') != pin.get('columns'):
                raise ValueError('Comparison artifact topology differs from completed stages')
            portable_tables[role] = {'artifact': relative, 'actual_sha256': pin['sha256'],
                                    'reference_sha256': table['reference_sha256'], 'rows': table['actual_rows'],
                                    'scientific_equal': table['scientific_equal']}
            if table.get('scientific_equal') is False:
                portable_tables[role]['differing_columns'] = {
                    column: details for column, details in table.get('columns', {}).items()
                    if details.get('exact_equal') is not True}
        result[stage] = {'scientific_equal': item['scientific_equal'], 'tables': portable_tables}
    return result


def verify_display_comparison(run: Path, displays: dict, reference_root: Path) -> dict:
    report_path = run / 'display_comparison.json'
    report = read_json(report_path)
    if (report.get('schema_name') != 'fresh_paper_display_comparison_v1'
            or report.get('status') != 'complete' or report.get('all_content_equal') is not True
            or set(report.get('files', {})) != set(DISPLAY_FILES)
            or report.get('fresh_display_manifest_sha256') != sha256(run / 'displays/manifest.json')):
        raise ValueError('Fresh display content comparison is missing, partial or crosswired')
    files = {}
    for name in DISPLAY_FILES:
        row, pin = report['files'][name], displays['outputs'][name]
        if row.get('content_equal') is not True or row.get('actual_sha256') != pin['sha256']:
            raise ValueError('Fresh display comparison contains a failed or changed artifact')
        verify_pin(run / 'displays', pin)
        if row.get('reference_sha256') != sha256(under(reference_root, name)):
            raise ValueError('Display comparison reference changed')
        files[name] = {'actual_sha256': row['actual_sha256'], 'reference_sha256': row['reference_sha256'],
                       'content_equal': True, 'byte_identical': row.get('byte_identical')}
    return {'report_sha256': sha256(report_path), 'fresh_display_manifest_sha256': report['fresh_display_manifest_sha256'],
            'status': 'complete', 'all_content_equal': True, 'files': files}


def verify_completed_run(root: Path, run: Path, *, review_differences: bool = False) -> dict:
    """Read-only gate: reuse the exact orchestration validators, never its execute()."""
    root, run = root.resolve(strict=True), run.resolve(strict=True)
    pipeline_path = run / 'pipeline_manifest.json'
    pipeline_digest = sha256(pipeline_path)
    pipeline = read_json(pipeline_path)
    if pipeline.get('status') != 'complete' and not review_differences:
        raise ValueError('The full pipeline is not complete; packaging refused')
    comparison_pin = dict(pipeline.get('detailed_comparison', {}))
    if review_differences and 'sha256' not in comparison_pin:
        # Failed original orchestration does not append a success digest.
        # Bind the unchanged failed file now; never edit the pipeline to add one.
        comparison_pin['sha256'] = sha256(under(run, comparison_pin['path']))
    comparison_path = verify_pin(run, comparison_pin)
    comparison = read_json(comparison_path)
    downstream = read_json(run / 'downstream/manifest.json')
    (require_review_proof if review_differences else require_complete_proof)(pipeline, comparison, downstream)
    if (Path(comparison.get('actual_root', '')).resolve() != run
            or comparison.get('archive_read_only') is not True
            or comparison.get('runner_sha256') != sha256(root / 'paper/full_compare.py')):
        raise ValueError('Detailed comparison run/source binding changed')
    sources = verify_source_manifests(root)
    drivers = {path.name: sha256(path) for path in sorted((root / 'paper').glob('full_*.py'))}
    if (pipeline.get('driver_sources') != drivers
            or pipeline.get('scientific_sources') != sources['full_scientific_sources']):
        raise ValueError('Pipeline generation source hashes changed')
    sys.path.insert(0, str(root / 'paper'))
    runtime = importlib.import_module('full_reproduce')
    if Path(runtime.__file__).resolve() != root / 'paper/full_reproduce.py':
        raise ValueError('Wrong orchestration validator imported')
    source_runtime = importlib.import_module('full_sources')
    if (Path(source_runtime.__file__).resolve() != root / 'paper/full_sources.py'
            or source_runtime.verify_g2_sources() != sources['g2_checkpoint_sources']):
        raise ValueError('G2 checkpoint verifier or code-tree binding changed')
    config = pipeline['config']
    pre = runtime.verify_preprocessing(Path(config['preprocessed_root']), Path(config['raw_root']))
    if pre != pipeline.get('preprocessing'):
        raise ValueError('Raw/canonical preprocessing provenance changed')
    external_args = argparse.Namespace(data_root=Path(config['data_root']),
        external_inputs_json=Path(config['external_inputs_json']) if config.get('external_inputs_json') else None)
    if external_args.external_inputs_json and sha256(external_args.external_inputs_json) != config.get('external_inputs_json_sha256'):
        raise ValueError('External input mapping changed')
    external = runtime.verify_external(run / 'external', runtime.external_context(external_args))
    if external != pipeline.get('external'):
        raise ValueError('External raw result/source pins changed')
    runtime.verify_display_science(run)
    expected = set(runtime.STAGE_NAMES) | {'g2_chunks/' + (scenario if replicate == 1 else scenario + '-run2')
        for group in runtime.G2_GROUPS for scenario, replicate in group}
    if set(pipeline.get('completed', {})) != expected:
        raise ValueError('The full seven stages and nine G2 executions are not all complete')
    context = {'prepare_sha256': sha256(run / 'prepare/manifest.json'),
               'canonical_files': pre['canonical_files'], 'g1_file_sha256': sha256(runtime.G1_PATH),
               'labels': (Path(config['raw_root']) / 'ch2026_metrics_train.csv').resolve()}
    stages, artifacts, documents = {}, {}, {}
    for name in sorted(expected):
        directory = under(run, name)
        manifest_digest = sha256(directory / 'manifest.json')
        if manifest_digest != pipeline['completed'][name].get('manifest_sha256'):
            raise ValueError('Completed stage manifest changed: ' + name)
        kind = 'g2_scenario' if name.startswith('g2_chunks/') else name
        document = runtime.verify_stage(directory, kind)
        documents[name] = document
        runtime.validate_stage_lineage(run, kind, document, context)
        if document.get('scientific_sources', sources['full_scientific_sources']) != sources['full_scientific_sources']:
            raise ValueError('Stage scientific source pins differ: ' + name)
        if kind in ('g2_scenario', 'g2') and document.get('g2_execution_sources') != sources['g2_checkpoint_sources']:
            raise ValueError('G2 execution is not bound to its exact checkpoint: ' + name)
        if kind == 'g1' and document.get('runner_sha256') != drivers['full_g1.py']:
            raise ValueError('G1 execution source pin changed')
        for source_name, digest in document.get('driver_sources', {}).items():
            if source_name not in drivers or digest != drivers[source_name]:
                raise ValueError('Stage execution source pin changed: ' + name)
        for pin in document.get('adapter_sources', []):
            declared = Path(pin['path'])
            path = declared.resolve() if declared.is_absolute() else under(root / 'paper', pin['path'])
            if not path.is_relative_to(root / 'paper') or sha256(path) != pin['sha256']:
                raise ValueError('Stage adapter source pin changed: ' + name)
        if kind == 'g2_scenario':
            scenario = directory.name.removesuffix('-run2')
            replicate = 2 if directory.name.endswith('-run2') else 1
            extra = scenario == 'contiguous_20pct'
            if (document.get('scenario') != scenario or document.get('replicate') != replicate
                    or document.get('parameters') != {'root_seed': 42, 'draws': 50, 'max_days': 10,
                         'run_calibration': extra, 'run_modality': extra}):
                raise ValueError('G2 execution settings differ')
        pins = {}
        for role, pin in document['outputs'].items():
            if pin.get('path') is None:
                continue
            path = verify_pin(directory, pin)
            relative = path.relative_to(run).as_posix()
            clean = {key: pin[key] for key in ('sha256', 'bytes', 'rows', 'columns') if key in pin}
            clean['path'] = relative
            pins[role] = clean
            artifacts[relative] = clean
        stages[name] = {'manifest_sha256': manifest_digest, 'status': 'complete', 'artifacts': pins}
    fit_report = comparison['stages']['downstream'].get('fit_states', {})
    fits = require_matching_fit_report(fit_report)
    # Reopen only the newly produced pack. The original loader checks its content
    # digests, index, ledger and estimator serialization; it does not train models.
    from semantic_indicators.stress_downstream_runner import load_downstream_fit_state_pack
    observed_manifest = read_json(run / 'downstream/run/manifest.json')
    pack = load_downstream_fit_state_pack(run / 'downstream/run/fit',
        expected_content_digest=observed_manifest['fit_authority_digest'])
    identities = [(model.fold, model.held_out_subject, model.arm, model.model_name, model.target) for model in pack.models]
    if len(identities) != 160 or len(set(identities)) != 160 or identities != [tuple(item['identity']) for item in fits]:
        raise ValueError('Actual fit state topology differs from the comparison')
    compare_runtime = importlib.import_module('full_compare')
    if Path(compare_runtime.__file__).resolve() != root / 'paper/full_compare.py':
        raise ValueError('Wrong comparison validator imported')
    expected_roles = {'prepare': ('primitives', 'baselines', 'representations'), 'g1': (),
        'g2': documents['g2']['outputs'], 'stress': compare_runtime.STRESS_TABLES,
        'downstream': compare_runtime.DOWNSTREAM_TABLES,
        'g2_scenarios': tuple(name.removeprefix('g2_chunks/') + '/' + role
             for name, document in documents.items() if name.startswith('g2_chunks/') for role in document['outputs'])}
    expected_artifacts = {name: {} for name in COMPARISON_STAGES}
    for comparison_stage, roles in expected_roles.items():
        for role in roles:
            if comparison_stage == 'g2_scenarios':
                scenario, table = role.split('/', 1)
                stage_name, output_role = 'g2_chunks/' + scenario, table
            elif comparison_stage == 'downstream':
                stage_name, output_role = 'downstream', 'run/' + role
            else:
                stage_name, output_role = comparison_stage, role
            path = documents[stage_name]['outputs'][output_role]['path']
            expected_artifacts[comparison_stage][role] = stage_name + '/' + path if path is not None else None
    g1_comparison = comparison['stages']['g1']
    if (g1_comparison.get('new_table_count') != 19
            or g1_comparison.get('reviewed_pair_comparison', {}).get('all_within_reference_precision') is not True):
        raise ValueError('Detailed comparison lacks the complete reviewed G1 result')
    comparisons = comparison_bindings(run, comparison, artifacts, expected_roles=expected_roles,
                                      expected_artifacts=expected_artifacts, review_differences=review_differences)
    display_comparison = verify_display_comparison(run, documents['displays'], root / 'paper/reference/displays')
    if sha256(pipeline_path) != pipeline_digest or sha256(comparison_path) != comparison_pin['sha256']:
        raise ValueError('Pipeline or detailed comparison changed while checking')
    return {'schema_name': 'independent_etri_portable_delivery_verification_v1',
            'status': 'review_with_comparison_differences' if review_differences else 'verified',
            'strict_full_comparison_passed': not review_differences,
            'submission_ready': False if review_differences else None,
            'pipeline_manifest_sha256': pipeline_digest, 'detailed_comparison_sha256': sha256(comparison_path),
            'scientific_experiment_executed': True, 'full_comparison_scope': True,
            'comparison_policy': 'Exact scientific values; G1 uses the explicitly reviewed JSON reporting precision. Diagnostic tolerances cannot turn a failure into a pass.',
            'new_model_fits': 160, 'reused_historical_model_fits': 0, 'prediction_rows': 943392,
            'expected_primary_key_count': 314464, 'fit_states_reopened_and_validated': 160,
            'stage_verification': stages, 'comparison': comparisons, 'source_verification': sources,
            'display_comparison': display_comparison,
            'driver_sources': drivers, 'preprocessing_manifest_sha256': pre['manifest_sha256'],
            'packager_sha256': sha256(root / 'validation/package_full_delivery.py'),
            'display_comparison_runner_sha256': sha256(root / 'validation/compare_full_displays.py'),
            'canonical_file_sha256': pre['canonical_files'], 'raw_file_sha256': pre['raw_files'],
            'external_manifest_sha256': external['manifest_sha256'], 'external_output_sha256': external['outputs'],
            'proof_document_policy': 'Original local manifests remain private; this summary exports checked facts and digest pins only.'}


def file_manifest(staging: Path) -> list[dict]:
    records = []
    for path in sorted(staging.rglob('*')):
        relative = path.relative_to(staging).as_posix()
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise ValueError('Linked staging files are not allowed')
        if not path.is_file():
            continue
        relative_path(relative)
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    if len({row['path'].casefold() for row in records}) != len(records):
        raise ValueError('Case-insensitive delivery name collision')
    return records


def require_staging_unchanged(staging: Path, records: list[dict]) -> None:
    if file_manifest(staging) != records:
        raise ValueError('Staging file set or digest changed')


def write_verified_archive(staging: Path, target: Path, records: list[dict]) -> dict:
    """Verify a temporary ZIP and real extraction before exclusive final creation."""
    if target.exists():
        raise FileExistsError('Never overwrite an existing delivery')
    require_staging_unchanged(staging, records)
    if any(row['path'] == 'PACKAGE_MANIFEST.json' for row in records):
        raise ValueError('Package manifest name is reserved')
    with tempfile.TemporaryDirectory(prefix='ictc-full-archive-') as temp:
        temp_root = Path(temp)
        pending = temp_root / 'verified.zip'
        manifest = json_bytes({'schema_name': 'ictc2026_full_delivery_files_v1', 'files': records})
        with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for pin in records:
                archive.write(verify_pin(staging, pin), pin['path'])
            archive.writestr('PACKAGE_MANIFEST.json', manifest)
        extraction = temp_root / 'extracted'
        extraction.mkdir()
        expected = {row['path']: row for row in records}
        expected['PACKAGE_MANIFEST.json'] = {'path': 'PACKAGE_MANIFEST.json', 'bytes': len(manifest),
                                            'sha256': hashlib.sha256(manifest).hexdigest()}
        with zipfile.ZipFile(pending) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) or archive.testzip() is not None:
                raise ValueError('ZIP member inventory or CRC changed')
            for entry in archive.infolist():
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.is_dir():
                    raise ValueError('Unexpected ZIP member kind')
                destination = under(extraction, entry.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
                verify_pin(extraction, expected[entry.filename])
        if read_json(extraction / 'PACKAGE_MANIFEST.json')['files'] != records:
            raise ValueError('Extracted file manifest changed')
        if (extraction / 'paper/full_prepare.py').is_file():
            if verify_source_manifests(extraction) != verify_source_manifests(staging):
                raise ValueError('Extracted scientific source hashes changed')
            verify_staged_loaders(extraction)
        require_staging_unchanged(staging, records)
        target.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with target.open('xb') as output, pending.open('rb') as source:
                created = True
                shutil.copyfileobj(source, output)
            if sha256(target) != sha256(pending):
                raise ValueError('Final archive digest changed')
        except BaseException:
            if created:
                target.unlink()  # Only this call's exclusively created partial file.
            raise
    return {'file': target.name, 'bytes': target.stat().st_size, 'sha256': sha256(target),
            'verified_payload_files': len(records), 'contents_verified': True, 'extraction_verified': True}


def stage_delivery(root: Path, staging: Path, verification: dict, *, run: Path | None = None) -> dict:
    if any(staging.iterdir()):
        raise FileExistsError('Delivery staging must start empty')
    transforms = {}
    candidates = [root / name for name in ROOT_DOCUMENTS | VALIDATION_SCRIPTS]
    for name in ('paper', 'leaderboard', 'tests'):
        for directory, child_dirs, filenames in os.walk(root / name, followlinks=False):
            child_dirs[:] = [child for child in child_dirs if not child.startswith('.') and child not in
                {'__pycache__', 'raw', 'processed', 'results', 'output', 'outputs', 'venv', 'node_modules'}]
            candidates.extend(Path(directory) / filename for filename in filenames)
    for path in sorted(candidates):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or not delivery_allowed(relative):
            continue
        source = under(root, relative)
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Delivery file exceeds the small-artifact limit: ' + relative)
        data = source.read_bytes()
        if relative in ROOT_DOCUMENTS and b'FULL_PAPER_REPRODUCTION_IN_PROGRESS' in data:
            raise ValueError('Delivery documentation still records an unfinished full experiment: ' + relative)
        if relative in SANITIZED_MANIFESTS:
            original = json.loads(data.decode('utf-8-sig'))
            cleaned = sanitize_manifest(relative, original)
            staged_bytes = json_bytes(cleaned)
            transforms[relative] = {'original_sha256': hashlib.sha256(data).hexdigest(),
                'staged_sha256': hashlib.sha256(staged_bytes).hexdigest(),
                'removed_field': '[*].source' if relative.startswith('paper/') else 'sources[*].source_path',
                'all_other_fields_preserved': True,
                'scientific_file_sha256_preserved': [row['sha256'] for row in
                     (cleaned if isinstance(cleaned, list) else cleaned['sources'])]}
            data = staged_bytes
        assert_no_private_paths(relative, data)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    required = {*ROOT_DOCUMENTS, *SANITIZED_MANIFESTS, *VALIDATION_SCRIPTS,
        'paper/g2_vendor_manifest.json', 'paper/full_sources.py',
        'tests/test_full_g1.py', 'tests/test_full_compare.py',
        'tests/test_package_full_delivery.py', 'validation/package_full_delivery.py',
        'paper/protocol/task-4a-reviewed-g1-pair-summary.json', 'paper/full_reproduce.py',
        'paper/requirements.txt', 'leaderboard/requirements.txt'}
    if any(not (staging / name).is_file() for name in required):
        raise ValueError('Required code, protocol, root tests or delivery documentation is missing')
    if verify_source_manifests(staging) != verification['source_verification']:
        raise ValueError('Staged scientific source hashes changed')
    for relative, record in transforms.items():
        if sha256(root / relative) != record['original_sha256']:
            raise ValueError('Original source manifest changed during staging')
        if sanitize_manifest(relative, read_json(root / relative)) != read_json(staging / relative):
            raise ValueError('Unapproved manifest field transformation')
    loader_check = verify_staged_loaders(staging)
    fresh_exports = []
    if run is not None:
        for stage in ('evidence', 'displays'):
            for pin in verification['stage_verification'][stage]['artifacts'].values():
                if stage == 'evidence' and not pin['path'].endswith('.csv'):
                    continue
                source = verify_pin(run, pin)
                relative = 'verified_results/' + pin['path']
                data = source.read_bytes()
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError('Fresh paper display/export is unexpectedly large')
                assert_no_private_paths(relative, data)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                fresh_exports.append({'path': relative, 'sha256': pin['sha256'], 'bytes': len(data)})
        if len(fresh_exports) != 12:
            raise ValueError('Exactly eight fresh scientific CSVs and four displays must be exported')
    summary = dict(verification, manifest_transformations=transforms, sanitized_loader_verification=loader_check,
                   fresh_paper_exports=fresh_exports)
    data = json_bytes(summary)
    assert_no_private_paths('FULL_DELIVERY_VERIFICATION.json', data)
    (staging / 'FULL_DELIVERY_VERIFICATION.json').write_bytes(data)
    records = file_manifest(staging)
    if sum(row['bytes'] for row in records) > MAX_PAYLOAD_BYTES:
        raise ValueError('Code delivery unexpectedly includes large data or intermediate outputs')
    return {'payload_files': len(records), 'payload_bytes': sum(row['bytes'] for row in records),
            'manifest_transformations': transforms, 'sanitized_loader_verification': loader_check}


def verify_staged_loaders(staging: Path) -> dict:
    """Import only staged validators to demonstrate removed metadata is not needed."""
    script = """import json, pathlib, sys
root = pathlib.Path.cwd()
sys.path[:0] = [str(root / 'paper'), str(root / 'leaderboard')]
import full_prepare, full_sources, data_preparation
assert pathlib.Path(full_prepare.__file__).resolve() == root / 'paper/full_prepare.py'
assert pathlib.Path(data_preparation.__file__).resolve() == root / 'leaderboard/data_preparation.py'
assert pathlib.Path(full_sources.__file__).resolve() == root / 'paper/full_sources.py'
assert len(full_prepare.verify_vendor()) == 24
assert len(data_preparation.verify_vendor()) == 8
g2_records = full_sources.verify_g2_sources()['files']
assert len(g2_records) == 7
assert sum(row['path'] is not None for row in g2_records) == 4
for path in root.rglob('*.py'):
    compile(path.read_bytes(), path.relative_to(root).as_posix(), 'exec')
print(json.dumps({'staged_full_scientific_sources_verified': 24,
                  'staged_g2_checkpoint_records_verified': 7, 'staged_g2_physical_sources_verified': 4,
                  'staged_preprocessing_sources_verified': 8, 'all_python_sources_compiled': True}))
"""
    environment = {key: value for key, value in os.environ.items() if key not in
                   {'PYTHONPATH', 'FULL_DOWNSTREAM_ORIGINAL_TEST_ROOT'}}
    environment.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    checked = subprocess.run([sys.executable, '-I', '-B', '-c', script], cwd=staging, env=environment,
                             capture_output=True, text=True, encoding='utf-8', timeout=60)
    if checked.returncode:
        raise ValueError('Staged portable source loaders or compilation failed: ' + checked.stderr)
    return json.loads(checked.stdout)


def native_artifact(path: Path | None, root: Path) -> dict | None:
    if path is None:
        return None
    entries = read_json(root / 'PACKAGES.json')['packages']
    expected = [row for row in entries if row.get('file') == path.name]
    if len(expected) != 1 or expected[0].get('contents_verified') is not True:
        raise ValueError('Native model ZIP lacks a prior verified package record')
    record = expected[0]
    if path.name != 'LEADERBOARD_NATIVE_MODELS.zip' or sha256(path) != record['sha256'] or path.stat().st_size != record['bytes']:
        raise ValueError('Existing verified native model ZIP changed')
    return {'file': path.name, 'sha256': record['sha256'], 'bytes': record['bytes'],
            'role': 'Existing separately supplied native model artifact; unchanged and not embedded.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--version', required=True, help='New delivery identifier; existing archives are never overwritten')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'deliveries')
    parser.add_argument('--native-model-zip', type=Path)
    parser.add_argument('--create', action='store_true', help='Create a new local versioned ZIP after all checks; default checks only')
    parser.add_argument('--review-differences', action='store_true', help='Explicit REVIEW archive: preserve a failed strict comparison after complete generation; not a final pass')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', args.version) or '..' in args.version:
        parser.error('Version must be a short safe filename identifier')
    prefix = 'ICTC2026_FULL_CODE_REVIEW_' if args.review_differences else 'ICTC2026_FULL_CODE_'
    target = args.output_dir.resolve() / (prefix + args.version + '.zip')
    if target.exists():
        raise FileExistsError('Requested full delivery version already exists')
    run = args.run_dir.resolve(strict=True)
    if target.is_relative_to(run) or any(target.is_relative_to(ROOT / part) for part in ('paper', 'leaderboard', 'tests')):
        raise ValueError('Delivery destination overlaps experiment or source inputs')
    verification = verify_completed_run(ROOT, run, review_differences=args.review_differences)
    verification['native_model_external_artifact'] = native_artifact(args.native_model_zip, ROOT)
    with tempfile.TemporaryDirectory(prefix='ictc-full-delivery-') as temp:
        staging = Path(temp)
        prepared = stage_delivery(ROOT, staging, verification, run=run)
        # Verify again after staging, so a changing experiment/source cannot be
        # presented as the already-reviewed proof. Original proof files stay local.
        repeated = verify_completed_run(ROOT, run, review_differences=args.review_differences)
        expected = {key: value for key, value in verification.items() if key != 'native_model_external_artifact'}
        if repeated != expected:
            raise ValueError('Full experiment proof changed during staging')
        result = {'status': 'review_archive_checks_passed' if args.review_differences else 'checks_passed',
                  'strict_full_comparison_passed': not args.review_differences, 'zip_created': False, **prepared}
        if args.create:
            result.update(write_verified_archive(staging, target, file_manifest(staging)), zip_created=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
