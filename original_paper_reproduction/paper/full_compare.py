#!/usr/bin/env python3
"""Read-only comparison of fresh full-study outputs with explicit archived references.

No generation API is invoked. Numeric tolerance options are diagnostic only: an
exact numeric mismatch always fails validation, except the separately documented
precision of the reviewed G1 JSON. Input files are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from full_prepare import PAPER, sha256, verify_vendor, write_json

STAGES = ('prepare', 'g1', 'g2_scenarios', 'g2', 'stress', 'downstream')
SCENARIOS = ('contiguous_10pct', 'contiguous_20pct', 'contiguous_40pct',
             'outage_1h', 'outage_3h', 'outage_6h', 'evening_critical', 'empirical_gap')
STRESS_REFERENCE = 'task-12-etri-production_20260810_230202'
DOWNSTREAM_REFERENCE = 'task-12-downstream-etri-production-v1'
STRESS_TABLES = ('target_ledger', 'official_current_crosslink', 'mask_ledger',
                 'deletion_audit', 'cell_replays', 'primary_comparison',
                 'participant_deltas', 'denominators', 'family_inference',
                 'dose_response_participant', 'dose_response_family', 'decision')
DOWNSTREAM_TABLES = (
    'contiguous_provenance.arrow', 'ineligible_target_ledger.arrow',
    'observed_predictions.arrow', 'prediction_universe.arrow', 'window_exclusion_ledger.arrow',
    'fit/fit_ledger.arrow', 'fit/fit_state_index.arrow',
    'statistics/bootstrap_indices.parquet', 'statistics/bootstrap_statistics.parquet',
    'statistics/condition_group_descriptives.parquet', 'statistics/decision.parquet',
    'statistics/global_condition_descriptives.parquet', 'statistics/group_cells.parquet',
    'statistics/participant_condition_descriptives.parquet', 'statistics/participant_summaries.parquet',
    'statistics/primary_cell_contrasts.parquet', 'statistics/sign_ledger.parquet',
    'statistics/sign_statistics.parquet')
G2_JSON_DTYPES = {'level': 'str', 'primitive': 'object', 'pair': 'object',
                  'domain': 'str', 'gate': 'str', 'observed': 'object',
                  'comparator': 'str', 'threshold': 'object', 'passed': 'bool',
                  'can_change_domain_decision': 'bool', 'status': 'str',
                  'production_eligible': 'bool', 'root_seed': 'str', 'draws': 'int64'}
KEY_COLUMNS = ('row_seq', 'fit_seq', 'subject_id', 'participant', 'sensor_day_id',
               'primitive', 'feature_family', 'family', 'scenario', 'draw', 'condition',
               'arm', 'model_name', 'target', 'pair', 'bootstrap_draw', 'sign_index')


def fs(path: Path) -> Path:
    """Extended Windows paths are needed for the long read-only archive tree."""
    path = Path(path)
    text = str(path)
    if os.name == 'nt' and not text.startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path.resolve()))
    return path


def read_json(path: Path) -> dict:
    return json.loads(fs(path).read_text(encoding='utf-8-sig'))


def file_sha(path: Path) -> str:
    return sha256(fs(path))


def json_value(value):
    if isinstance(value, np.generic):
        return json_value(value.item())
    if value is pd.NA or value is pd.NaT or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + '... [sample truncated]'
    return value


def scalar_equal(a, b) -> bool:
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(scalar_equal(a[k], b[k]) for k in a))
    sequence = (list, tuple, np.ndarray)
    if isinstance(a, sequence) or isinstance(b, sequence):
        return (isinstance(a, sequence) and isinstance(b, sequence) and len(a) == len(b)
                and all(scalar_equal(x, y) for x, y in zip(a, b)))
    null_a, null_b = bool(pd.isna(a)), bool(pd.isna(b))
    if null_a or null_b:
        return null_a and null_b
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    return bool(a == b)


def compare_frames(actual: pd.DataFrame, reference: pd.DataFrame, *, exceptions: dict | None = None,
                   diagnostic_atol: float = 0.0, diagnostic_rtol: float = 0.0,
                   row_offset: int = 0) -> dict:
    """Compare in stored order; never align/sort away a changed key or coerce IDs to floats."""
    exceptions = exceptions or {}
    columns_equal = list(actual.columns) == list(reference.columns)
    dtypes_equal = list(map(str, actual.dtypes)) == list(map(str, reference.dtypes))
    result = {'actual_rows': len(actual), 'reference_rows': len(reference),
              'actual_columns': list(actual.columns), 'reference_columns': list(reference.columns),
              'actual_dtypes': list(map(str, actual.dtypes)), 'reference_dtypes': list(map(str, reference.dtypes)),
              'columns_equal': columns_equal, 'dtypes_equal': dtypes_equal,
              'index_equal': actual.index.equals(reference.index), 'columns': {},
              'exceptions': dict(exceptions), 'exact_values_order': False,
              'within_diagnostic_tolerance': False, 'scientific_equal': False}
    if len(actual) != len(reference) or not columns_equal:
        result['reason'] = 'row count or column order differs'
        return result
    keys = [key for key in KEY_COLUMNS if key in actual.columns]
    for column in actual.columns:
        a, b = actual[column], reference[column]
        numeric_float = pd.api.types.is_float_dtype(a.dtype) and pd.api.types.is_float_dtype(b.dtype)
        max_abs = max_rel = None
        finite_changes = 0
        if numeric_float:
            x, y = a.to_numpy(dtype=float, na_value=np.nan), b.to_numpy(dtype=float, na_value=np.nan)
            equal = (x == y) | (np.isnan(x) & np.isnan(y))
            finite = np.isfinite(x) & np.isfinite(y)
            finite_changes = int(np.count_nonzero(np.isfinite(x) != np.isfinite(y)))
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                difference = np.abs(x[finite] - y[finite])
                relative = np.divide(difference, np.abs(y[finite]),
                                     out=np.where(difference == 0, 0.0, np.inf), where=y[finite] != 0)
            max_abs = float(difference.max(initial=0.0))
            max_rel = float(relative.max(initial=0.0))
            diagnostic = equal.copy()
            diagnostic[finite] = difference <= diagnostic_atol + diagnostic_rtol * np.abs(y[finite])
        else:
            nested = any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in a.head(5))
            nested |= any(isinstance(v, (dict, list, tuple, np.ndarray)) for v in b.head(5))
            if nested or a.dtype == object or b.dtype == object:
                equal = np.fromiter((scalar_equal(x, y) for x, y in zip(a, b)), dtype=bool, count=len(a))
            else:
                equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False).to_numpy(dtype=bool)
            diagnostic = equal.copy()
            if pd.api.types.is_integer_dtype(a.dtype) and pd.api.types.is_integer_dtype(b.dtype):
                differences = [abs(int(a.iloc[i]) - int(b.iloc[i])) for i in np.flatnonzero(~equal)
                               if not pd.isna(a.iloc[i]) and not pd.isna(b.iloc[i])]
                max_abs = max(differences, default=0)
            elif a.dtype == object or b.dtype == object:
                differences, relatives = [], []
                numeric_types = (int, float, np.integer, np.floating)
                for position in np.flatnonzero(~equal):
                    x, y = a.iloc[position], b.iloc[position]
                    if (isinstance(x, numeric_types) and isinstance(y, numeric_types)
                            and not isinstance(x, (bool, np.bool_)) and not isinstance(y, (bool, np.bool_))):
                        xf, yf = math.isfinite(x), math.isfinite(y)
                        finite_changes += int(xf != yf)
                        if xf and yf:
                            # Keep object-valued integer identities exact as well.
                            difference = abs(int(x) - int(y)) if isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)) else abs(float(x) - float(y))
                            differences.append(difference)
                            relatives.append(difference / abs(y) if y else float('inf'))
                            diagnostic[position] = difference <= diagnostic_atol + diagnostic_rtol * abs(y)
                if differences:
                    max_abs, max_rel = max(differences), max(relatives)
        positions = np.flatnonzero(~equal)
        examples = []
        for position in positions[:3]:
            i = int(position)
            examples.append({'row_position': row_offset + i,
                             'key': {key: json_value(actual[key].iloc[i]) for key in keys},
                             'actual': json_value(a.iloc[i]), 'reference': json_value(b.iloc[i])})
        result['columns'][column] = {
            'different_count': int(len(positions)), 'finite_status_changes': finite_changes,
            'max_absolute_difference': json_value(max_abs), 'max_relative_difference': json_value(max_rel),
            'exact_equal': not len(positions), 'within_diagnostic_tolerance': bool(diagnostic.all()),
            'exception_reason': exceptions.get(column), 'examples': examples}
    result['exact_values_order'] = bool(result['index_equal'] and all(x['exact_equal'] for x in result['columns'].values()))
    result['within_diagnostic_tolerance'] = bool(result['index_equal'] and all(
        x['within_diagnostic_tolerance'] for name, x in result['columns'].items() if name not in exceptions))
    result['scientific_equal'] = bool(columns_equal and dtypes_equal and result['index_equal'] and all(
        x['exact_equal'] for name, x in result['columns'].items() if name not in exceptions))
    return result


def table_exceptions(stage: str, table: str) -> dict[str, str]:
    if stage == 'g2_scenarios' and table == 'run_manifest':
        return {'runtime_seconds': 'Measured wall-clock runtime, reliability.py:4141-4163; retained in report.'}
    if stage == 'downstream' and table == 'contiguous_provenance':
        return {
            'official_g2_authority_digest': 'NEW independent G2 manifest replaces the historical authority root.',
            'official_evidence_digest': 'OfficialCellEvidence binds G2 artifact hashes; all semantic fields compared.',
            'deletion_audit_digest': 'OfficialDeletionAudit binds G2 artifact hash; counts/record identity compared.',
            'content_digest': 'This row seal includes the three independently rebound provenance fields above.'}
    return {}


@contextmanager
def arrow_reader(path: Path):
    with fs(path).open('rb') as stream:
        try:
            reader = ipc.open_file(stream)
        except pa.ArrowInvalid:
            stream.seek(0)
            reader = ipc.open_stream(stream)
        yield reader


def table_metadata(path: Path) -> dict:
    if path.suffix == '.parquet':
        parquet = pq.ParquetFile(fs(path))
        schema, rows = parquet.schema_arrow, parquet.metadata.num_rows
    elif path.suffix in {'.arrow', '.ipc'}:
        with arrow_reader(path) as reader:
            schema = reader.schema
            if hasattr(reader, 'num_record_batches'):
                rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
            else:
                rows = sum(batch.num_rows for batch in reader)
    else:
        frame = pd.read_json(fs(path), lines=True, dtype=G2_JSON_DTYPES)
        return {'rows': len(frame), 'columns': list(frame.columns), 'arrow_schema': None,
                'logical_dtypes': list(map(str, frame.dtypes))}
    return {'rows': rows, 'columns': schema.names,
            'logical_dtypes': list(map(str, pa.Table.from_batches([], schema=schema).to_pandas().dtypes)),
            'arrow_schema': [[field.name, str(field.type), field.nullable] for field in schema]}


def table_batches(path: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    if path.suffix == '.parquet':
        yield from pq.ParquetFile(fs(path)).iter_batches(batch_size=batch_size)
    elif path.suffix in {'.arrow', '.ipc'}:
        with arrow_reader(path) as reader:
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches)) if hasattr(reader, 'num_record_batches') else reader
            for batch in batches:
                for offset in range(0, batch.num_rows, batch_size):
                    yield batch.slice(offset, batch_size)
    else:
        frame = pd.read_json(fs(path), lines=True, dtype=G2_JSON_DTYPES)
        yield from pa.Table.from_pandas(frame, preserve_index=False).to_batches(max_chunksize=batch_size)


def merge_chunk(report: dict, chunk: dict) -> None:
    for flag in ('columns_equal', 'dtypes_equal', 'index_equal', 'exact_values_order',
                 'within_diagnostic_tolerance', 'scientific_equal'):
        report[flag] = report.get(flag, True) and chunk[flag]
    for key in ('actual_dtypes', 'reference_dtypes'):
        report.setdefault(key, chunk[key])
    for name, incoming in chunk['columns'].items():
        if name not in report['columns']:
            report['columns'][name] = incoming
            continue
        existing = report['columns'][name]
        for key in ('different_count', 'finite_status_changes'):
            existing[key] += incoming[key]
        for key in ('exact_equal', 'within_diagnostic_tolerance'):
            existing[key] &= incoming[key]
        for key in ('max_absolute_difference', 'max_relative_difference'):
            values = [value for value in (existing[key], incoming[key]) if value is not None]
            existing[key] = ('+Infinity' if '+Infinity' in values else max(values)) if values else None
        existing['examples'] = (existing['examples'] + incoming['examples'])[:3]


def compare_table(actual: Path, reference: Path, *, exceptions: dict | None = None,
                  diagnostic_atol: float = 0.0, diagnostic_rtol: float = 0.0,
                  batch_size: int = 32768, table_name: str = '') -> dict:
    """Stream IPC/Parquet in aligned row order without loading943392 predictions at once."""
    am, rm = table_metadata(actual), table_metadata(reference)
    ah, rh = file_sha(actual), file_sha(reference)
    report = {'actual_path': str(actual), 'reference_path': str(reference),
              'actual_sha256': ah, 'reference_sha256': rh, 'byte_identical': ah == rh,
              'actual_rows': am['rows'], 'reference_rows': rm['rows'],
              'actual_columns': am['columns'], 'reference_columns': rm['columns'],
              'actual_dtypes': am['logical_dtypes'], 'reference_dtypes': rm['logical_dtypes'],
              'dtypes_equal': am['logical_dtypes'] == rm['logical_dtypes'],
              'actual_arrow_schema': am['arrow_schema'], 'reference_arrow_schema': rm['arrow_schema'],
              'arrow_schema_equal': am['arrow_schema'] == rm['arrow_schema'],
              'columns': {}, 'exceptions': exceptions or {}, 'rows_compared': 0}
    if am['rows'] != rm['rows'] or am['columns'] != rm['columns']:
        report.update(scientific_equal=False, exact_values_order=False, dtypes_equal=False,
                      columns_equal=am['columns'] == rm['columns'], reason='row count or column order differs')
        return report
    # JSON decisions require their source logical dtype overrides; converting mixed
    # bool/numeric objects through Arrow would coerce the very types being audited.
    if actual.suffix == '.jsonl' or reference.suffix == '.jsonl':
        a = pd.read_json(fs(actual), lines=True, dtype=G2_JSON_DTYPES)
        b = pd.read_json(fs(reference), lines=True, dtype=G2_JSON_DTYPES)
        chunk = compare_frames(a, b, exceptions=exceptions, diagnostic_atol=diagnostic_atol, diagnostic_rtol=diagnostic_rtol)
        merge_chunk(report, chunk)
        report['rows_compared'] = len(a)
    else:
        ai, ri = iter(table_batches(actual, batch_size)), iter(table_batches(reference, batch_size))
        a, b = next(ai, None), next(ri, None)
        apos = bpos = offset = 0
        while a is not None and b is not None:
            count = min(a.num_rows - apos, b.num_rows - bpos)
            af, bf = a.slice(apos, count).to_pandas(), b.slice(bpos, count).to_pandas()
            if table_name == 'confidence_ablation':
                af['sensor_day_id'] = af['sensor_day_id'].astype(object)
                bf['sensor_day_id'] = bf['sensor_day_id'].astype(object)
            chunk = compare_frames(af, bf, exceptions=exceptions, diagnostic_atol=diagnostic_atol,
                                   diagnostic_rtol=diagnostic_rtol, row_offset=offset)
            merge_chunk(report, chunk)
            offset += count
            apos += count
            bpos += count
            if apos == a.num_rows:
                a, apos = next(ai, None), 0
            if bpos == b.num_rows:
                b, bpos = next(ri, None), 0
        report['rows_compared'] = offset
        if offset != am['rows'] or a is not None or b is not None:
            report.update(scientific_equal=False, reason='stream row topology differs')
    for key in ('columns_equal', 'dtypes_equal', 'index_equal', 'exact_values_order', 'within_diagnostic_tolerance', 'scientific_equal'):
        report.setdefault(key, True)
    report['scientific_equal'] &= report['arrow_schema_equal'] and report['dtypes_equal']
    return report


def model_state_document(state_format: str, model_bytes: bytes) -> object:
    """Decode coefficient JSON or all LightGBM text fields, retaining every tree threshold."""
    if state_format != 'lightgbm-model-string-v1':
        return json.loads(model_bytes.decode('utf-8'))

    def token(value):
        if re.fullmatch(r'[+-]?\d+', value):
            return int(value)
        if re.fullmatch(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', value):
            return float(value)
        return value

    lines = []
    for line in model_bytes.decode('utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            lines.append([key, [token(part) for part in value.split()]])
        elif line.startswith('[') and line.endswith(']') and ':' in line:
            key, value = line[1:-1].split(':', 1)
            lines.append([key, [token(part) for part in value.split()]])
        else:
            lines.append([line])
    return lines


def _flatten(value, prefix='') -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in _flatten(value[key], f'{prefix}/{key}')]
    if isinstance(value, (list, tuple)):
        return [item for i, child in enumerate(value) for item in _flatten(child, f'{prefix}/{i}')]
    return [(prefix, value)]


def compare_fit_packs(actual_run: Path, reference_run: Path, *, diagnostic_atol=0.0, diagnostic_rtol=0.0) -> dict:
    """Original read-only pack validator reconstructs states; no fit/prediction is called."""
    verify_vendor()
    from semantic_indicators import outcome
    from semantic_indicators.stress_downstream_runner import load_downstream_fit_state_pack
    actual_manifest, reference_manifest = read_json(actual_run / 'manifest.json'), read_json(reference_run / 'manifest.json')
    a = load_downstream_fit_state_pack(fs(actual_run / 'fit'), expected_content_digest=actual_manifest['fit_authority_digest'])
    b = load_downstream_fit_state_pack(fs(reference_run / 'fit'), expected_content_digest=reference_manifest['fit_authority_digest'])
    result = {'actual_fit_count': len(a.models), 'reference_fit_count': len(b.models),
              'both_packs_validated': True, 'new_fits_required': 160,
              'pack_byte_identical': file_sha(actual_run / 'fit/fit_states.pack') == file_sha(reference_run / 'fit/fit_states.pack'),
              'reporting_threshold': 0.5,
              'frozen_model_parameters': {'elasticnet': outcome.ELASTICNET_PARAMETERS,
                                          'lightgbm': outcome.LIGHTGBM_PARAMETERS},
              'fits': [], 'scientific_equal': len(a.models) == len(b.models) == 160}
    for i, (am, rm) in enumerate(zip(a.models, b.models)):
        identity = (am.fold, am.held_out_subject, am.arm, am.model_name, am.target)
        reference_identity = (rm.fold, rm.held_out_subject, rm.arm, rm.model_name, rm.target)
        state_checks = {}
        for field in fields(am.preprocessor_state):
            av, rv = getattr(am.preprocessor_state, field.name), getattr(rm.preprocessor_state, field.name)
            state_checks[field.name] = scalar_equal(av, rv)
        numbers_a, numbers_b = [], []
        for component in ('medians', 'centers', 'scales'):
            for position, value in enumerate(getattr(am.preprocessor_state, component)):
                numbers_a.append({'component': component, 'position': position, 'value': value})
            for position, value in enumerate(getattr(rm.preprocessor_state, component)):
                numbers_b.append({'component': component, 'position': position, 'value': value})
        preprocessing = compare_frames(pd.DataFrame(numbers_a), pd.DataFrame(numbers_b),
                                       diagnostic_atol=diagnostic_atol, diagnostic_rtol=diagnostic_rtol)
        model_bytes_equal = am.model_state_bytes == rm.model_state_bytes
        model_semantics_equal = model_state_document(am.model_state_format, am.model_state_bytes) == model_state_document(rm.model_state_format, rm.model_state_bytes)
        item = {'fit_seq': i, 'identity': identity, 'identity_equal': identity == reference_identity,
                'declared_features_equal': am.declared_feature_names == rm.declared_feature_names,
                'preprocessor_fields_equal': state_checks,
                'preprocessor_numeric_comparison': preprocessing,
                'model_state_format_equal': am.model_state_format == rm.model_state_format,
                'model_state_byte_identical': model_bytes_equal, 'model_semantics_equal': model_semantics_equal,
                'preprocessor_digest_equal': am.preprocessor_digest == rm.preprocessor_digest,
                'model_digest_equal': am.model_digest == rm.model_digest,
                'fit_content_digest_equal': am.content_digest == rm.content_digest}
        if not model_semantics_equal:
            # Flatten only differing models for compact field-level numeric diagnostics.
            av = _flatten(model_state_document(am.model_state_format, am.model_state_bytes))
            rv = _flatten(model_state_document(rm.model_state_format, rm.model_state_bytes))
            item['model_field_differences'] = [
                {'actual_path': ak, 'reference_path': rk, 'actual': json_value(x), 'reference': json_value(y)}
                for (ak, x), (rk, y) in zip(av, rv) if ak != rk or not scalar_equal(x, y)][:10]
            numeric_differences = [abs(float(x)-float(y)) for (ak,x),(rk,y) in zip(av,rv)
                                   if ak == rk and isinstance(x,(int,float)) and isinstance(y,(int,float))]
            item['model_max_absolute_numeric_difference'] = json_value(max(numeric_differences, default=0.0))
        item['scientific_equal'] = bool(item['identity_equal'] and item['declared_features_equal']
            and all(state_checks.values()) and preprocessing['scientific_equal']
            and item['model_state_format_equal'] and model_semantics_equal
            and item['preprocessor_digest_equal'] and item['model_digest_equal'] and item['fit_content_digest_equal'])
        result['fits'].append(item)
        result['scientific_equal'] &= item['scientific_equal']
    return result


def fresh_manifest(root: Path, stage: str) -> dict:
    document = read_json(root / 'manifest.json')
    if (document.get('schema_name') != 'independent_etri_reproduction' or document.get('stage') != stage
            or document.get('status') != 'complete' or document.get('fresh_execution') is not True):
        raise ValueError(f'{root}: comparison requires completed fresh {stage} evidence')
    for name, item in document['outputs'].items():
        if item.get('path') is None:
            continue
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path) != item['sha256']:
            raise ValueError(f'Fresh output missing, changed or escapes stage: {name}')
    return document


def compare_record(actual_root: Path, reference_root: Path, actual: dict, reference: dict,
                   *, stage: str, name: str, **options) -> dict:
    if actual.get('path') is None or reference.get('path') is None:
        equal = (actual.get('path') is None and reference.get('path') is None
                 and actual.get('rows') == reference.get('rows') == 0
                 and actual.get('columns', []) == reference.get('columns', []) == [])
        return {'declared_empty': True, 'actual_declaration': actual, 'reference_declaration': reference,
                'scientific_equal': equal, 'exact_values_order': equal, 'byte_identical': None}
    a, r = actual_root / actual['path'], reference_root / reference['path']
    if reference.get('sha256') and file_sha(r) != reference['sha256']:
        raise ValueError(f'Archived reference changed from its manifest: {r}')
    result = compare_table(a, r, exceptions=table_exceptions(stage, name), table_name=name, **options)
    for side, declaration in (('actual', actual), ('reference', reference)):
        if 'rows' in declaration and result[f'{side}_rows'] != declaration['rows']:
            result.update(scientific_equal=False, reason=f'{side} manifest row declaration differs')
    return result


def compare_g2_scenario(actual_dir: Path, reference_dir: Path, **options) -> dict:
    """Compare one completed fresh scenario without requiring the other eight runs."""
    actual_dir, reference_dir = Path(actual_dir), Path(reference_dir)
    document = fresh_manifest(actual_dir, 'g2_scenario')
    original = read_json(reference_dir / 'summary.json')['artifacts']
    result = {'actual_dir': str(actual_dir), 'reference_dir': str(reference_dir),
              'tables': {}, 'scientific_equal': True,
              'output_roles_equal': set(document['outputs']) == set(original)}
    if not result['output_roles_equal']:
        result.update(scientific_equal=False, actual_roles=list(document['outputs']),
                      reference_roles=list(original), reason='G2 scenario output roles differ')
    for name, reference in original.items():
        try:
            report = compare_record(actual_dir, reference_dir, document['outputs'][name], reference,
                                    stage='g2_scenarios', name=name, **options)
        except Exception as error:
            report = {'scientific_equal': False, 'error_type': type(error).__name__, 'error': str(error)}
        result['tables'][name] = report
        result['scientific_equal'] &= report['scientific_equal']
    return result


def compare_stage(actual_root: Path, reference_sdd: Path, stage: str, **options) -> dict:
    result = {'tables': {}, 'scientific_equal': True}

    def one(label, aroot, rroot, ar, rr, name):
        try:
            report = compare_record(aroot, rroot, ar, rr, stage=stage, name=name, **options)
        except Exception as error:
            report = {'scientific_equal': False, 'error_type': type(error).__name__, 'error': str(error)}
        result['tables'][label] = report
        result['scientific_equal'] &= report['scientific_equal']

    if stage == 'g1':
        from full_g1 import compare_g1_pairs
        root = actual_root / 'g1'
        document = fresh_manifest(root, 'g1')
        reference = reference_sdd / 'task-4a-reviewed-g1-pair-summary.json'
        summary = pd.read_parquet(fs(root / document['outputs']['pair_summary']['path']))
        result['reviewed_pair_comparison'] = compare_g1_pairs(summary, fs(reference))
        result['scientific_equal'] = result['reviewed_pair_comparison']['all_within_reference_precision']
        result['new_table_count'] = len(document['outputs'])
        result['scientific_equal'] &= result['new_table_count'] == 19
        result['uncompared_reference_tables'] = [name for name in document['outputs'] if name != 'pair_summary']
        result['reference_limit'] = 'Archive stores reviewed pair JSON and report, not full original G1 table files; all19 new hashes verified, reviewed pair precision explicitly retained.'
        return result
    if stage == 'prepare':
        root = actual_root / 'prepare'
        document = fresh_manifest(root, 'prepare')
        for name in ('primitives', 'baselines', 'representations'):
            one(name, root, reference_sdd, document['outputs'][name],
                {'path': f'task-4c2-real-{name}.parquet'}, name)
    elif stage == 'g2_scenarios':
        for scenario in (*SCENARIOS, 'empirical_gap-run2'):
            root, reference = actual_root / 'g2_chunks' / scenario, reference_sdd / f'task-4c2-v2-{scenario}'
            comparison = compare_g2_scenario(root, reference, **options)
            for name, table in comparison['tables'].items():
                result['tables'][f'{scenario}/{name}'] = table
            result['scientific_equal'] &= comparison['scientific_equal']
    elif stage == 'g2':
        root, reference = actual_root / 'g2', reference_sdd / 'task-4c2-v2-official'
        document = fresh_manifest(root, 'g2')
        original = read_json(reference / 'manifest.json')['official_artifacts']
        if set(document['outputs']) != set(original):
            raise ValueError('G2 aggregate output roles differ')
        for name, ref in original.items():
            one(name, root, reference, document['outputs'][name], ref, name)
    elif stage == 'stress':
        root, reference = actual_root / 'stress', reference_sdd / STRESS_REFERENCE
        document = fresh_manifest(root, 'stress')
        original = {item['name']: item for item in read_json(reference / 'manifest.json')['artifacts']}
        if set(document['outputs']) != set(STRESS_TABLES) or set(original) != set(STRESS_TABLES):
            raise ValueError('Stress output roles differ from all12 original tables')
        for name in STRESS_TABLES:
            one(name, root, reference, document['outputs'][name], original[name], name)
    elif stage == 'downstream':
        root, reference = actual_root / 'downstream', reference_sdd / DOWNSTREAM_REFERENCE
        document = fresh_manifest(root, 'downstream')
        if (document.get('new_model_fits') != 160 or document.get('prediction_rows') != 943392
                or document.get('reused_historical_model_fits') != 0):
            raise ValueError('Downstream comparison requires160 fresh fits and943392 predictions with zero historical fit reuse')
        original = {item['relative_path']: item for item in read_json(reference / 'manifest.json')['artifacts']}
        for relative in DOWNSTREAM_TABLES:
            ref = original[relative]
            one(relative, root, reference, document['outputs']['run/' + relative],
                {'path': relative, 'sha256': ref['physical_sha256']}, Path(relative).stem)
        result['fit_states'] = compare_fit_packs(root / 'run', reference,
            diagnostic_atol=options.get('diagnostic_atol', 0.0), diagnostic_rtol=options.get('diagnostic_rtol', 0.0))
        result['scientific_equal'] &= result['fit_states']['scientific_equal']
    else:
        raise ValueError(f'Unknown comparison stage: {stage}')
    return result


def run_comparisons(actual_root: Path, reference_sdd: Path, report_path: Path,
                    *, stages: tuple[str, ...] = STAGES, diagnostic_atol: float = 0.0,
                    diagnostic_rtol: float = 0.0, batch_size: int = 32768) -> dict:
    if not stages or len(set(stages)) != len(stages) or any(stage not in STAGES for stage in stages):
        raise ValueError('Comparison stages must be a nonempty unique subset of the full study')
    if not math.isfinite(diagnostic_atol) or not math.isfinite(diagnostic_rtol) or diagnostic_atol < 0 or diagnostic_rtol < 0 or batch_size <= 0:
        raise ValueError('Invalid comparison diagnostics or batch size')
    actual_root, reference_sdd, report_path = actual_root.resolve(), reference_sdd.resolve(), report_path.resolve()
    if actual_root == reference_sdd or actual_root.is_relative_to(reference_sdd) or reference_sdd.is_relative_to(actual_root):
        raise ValueError('Fresh output and archived reference roots must be separate')
    if report_path.is_relative_to(reference_sdd):
        raise ValueError('Comparison report cannot be written inside the read-only archive')
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite an earlier comparison report: {report_path}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'schema_name': 'independent_etri_detailed_comparison_v1', 'status': 'running',
              'actual_root': str(actual_root), 'reference_sdd': str(reference_sdd),
              'requested_stages': list(stages), 'full_scope': set(stages) == set(STAGES),
              'scientific_generation_performed': False, 'archive_read_only': True,
              'policy': {'numeric_acceptance': 'exact; tolerances diagnose only and never convert failure to pass',
                         'diagnostic_atol': diagnostic_atol, 'diagnostic_rtol': diagnostic_rtol,
                         'g1_exception': 'Reviewed JSON reported precision, see full_g1 comparison'},
              'runner_sha256': file_sha(Path(__file__)), 'stages': {}, 'all_scientific_equal': False}
    write_json(report_path, report)
    for stage in stages:
        print(json.dumps({'stage': 'comparison', 'comparing': stage, 'status': 'running'}), flush=True)
        try:
            result = compare_stage(actual_root, reference_sdd, stage, diagnostic_atol=diagnostic_atol,
                                   diagnostic_rtol=diagnostic_rtol, batch_size=batch_size)
        except Exception as error:
            result = {'scientific_equal': False, 'error_type': type(error).__name__, 'error': str(error)}
        report['stages'][stage] = result
        write_json(report_path, report)
    report['all_scientific_equal'] = all(item['scientific_equal'] for item in report['stages'].values())
    report.update(status='complete' if report['all_scientific_equal'] else 'failed',
                  seconds=time.perf_counter() - started)
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-root', type=Path, required=True)
    parser.add_argument('--reference-sdd', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--stages', default=','.join(STAGES), help='Comma-separated stage subset; default full study')
    parser.add_argument('--diagnostic-atol', type=float, default=0.0)
    parser.add_argument('--diagnostic-rtol', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=32768)
    args = parser.parse_args()
    result = run_comparisons(args.actual_root, args.reference_sdd, args.report,
        stages=tuple(args.stages.split(',')), diagnostic_atol=args.diagnostic_atol,
        diagnostic_rtol=args.diagnostic_rtol, batch_size=args.batch_size)
    print(json.dumps({'stage': 'comparison', 'status': result['status'], 'full_scope': result['full_scope'],
                      'all_scientific_equal': result['all_scientific_equal'], 'report': str(args.report)}), flush=True)
    if not result['all_scientific_equal']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
