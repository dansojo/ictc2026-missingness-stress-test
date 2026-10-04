"""Separate, byte-bound provenance for a bounded fresh execution graph.

Historical freezes are verified independently; their hashes are never replaced
with these manifests. Every stage remains explicitly ineligible as a full run.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
import json
from datetime import datetime, timezone

from portable_paths import configuration, digest
import os

PRIMITIVES = ('screen_load_24h', 'phone_activity_load_24h', 'usage_load_24h',
              'mobile_light_exposure_24h', 'wearable_light_exposure_24h')
PARAMETERS = {'root_seed': 42, 'draws': 50, 'max_selected_days': 10,
              'selection': 'first subject/day per recovery primitive after full original selection',
              'normalization': 'full original selected-day MAD and other-participant population baseline',
              'mask_geometry': 'original contiguous20 time window and matched valid-record random mask'}
PARENTS = {'raw': (), 'prepare': ('raw',), 'g2': ('prepare',),
           'stress': ('prepare', 'g2'), 'm0': ('prepare', 'g2', 'stress'),
           'legacy': ('m0',), 'donors': ('raw', 'prepare', 'm0'),
           'same-day': ('m0', 'legacy'), 'cross-day': ('m0', 'same-day', 'donors')}


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def enabled():
    return (configuration() or {}).get('execution_mode') == 'fresh_bounded'


def configured_contract():
    cfg = configuration() or {}
    if not enabled() or not isinstance(cfg.get('fresh_contract'), dict):
        raise ValueError('An explicit fresh bounded execution contract is required')
    return cfg['fresh_contract']


def historical_role(name, current):
    """Keep archive freeze verification separate from fresh consumed inputs."""
    if not enabled(): return current
    from portable_paths import resolve_input
    value = (configuration() or {}).get('historical_roles', {}).get(name)
    if value is None: raise ValueError('Missing separate historical provenance role: ' + name)
    return resolve_input(value)


def verify_active_config(path, expected):
    """Bind the generated role map as well as its immutable parent contract."""
    active = os.environ.get('ICTC_RUN_CONFIG')
    if active is None or Path(active).resolve() != Path(path).resolve() or digest(path) != expected:
        raise ValueError('Active fresh execution configuration changed')


def create_contract(root, raw_inputs, code_inputs, frozen_inputs, *, cell_limit):
    if type(cell_limit) is not int or not 1 <= cell_limit <= 5:
        raise ValueError('The fresh chain supports one to five cells only')
    path = Path(root).resolve() / 'run-contract.json'
    if path.exists(): raise FileExistsError('Run contract is immutable')
    record = lambda mapping: {name: {'path': str(Path(file).resolve()), 'sha256': sha}
                              for name, (file, sha) in mapping.items()}
    value = {'schema': 'ictc-fresh-bounded-contract-v1', 'scope': 'bounded',
             'full_reproduction': False, 'cell_limit': cell_limit, 'parameters': PARAMETERS,
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'raw_inputs': record(raw_inputs), 'execution_sources': record(code_inputs),
             'historical_provenance': record(frozen_inputs)}
    write_json(path, value)
    result = {'path': str(path), 'sha256': digest(path)}
    verify_contract(result)
    return result


def verify_contract(reference):
    path = Path(reference['path']).resolve(strict=True)
    if digest(path) != reference['sha256']: raise ValueError('Fresh run contract changed')
    value = json.loads(path.read_text(encoding='utf8'))
    if (value.get('schema') != 'ictc-fresh-bounded-contract-v1' or value.get('scope') != 'bounded'
            or value.get('full_reproduction') is not False or value.get('parameters') != PARAMETERS
            or type(value.get('cell_limit')) is not int or not 1 <= value['cell_limit'] <= 5):
        raise ValueError('Fresh scope or scientific parameters changed')
    for group in ('raw_inputs', 'execution_sources', 'historical_provenance'):
        records = value.get(group)
        if not isinstance(records, dict) or not records: raise ValueError('Missing run-contract input group')
        for record in records.values():
            if digest(record['path']) != record['sha256']:
                raise ValueError('Pinned raw, source or historical-provenance bytes changed')
    return value


def safe_output(root, name):
    normalized = name.replace('\\', '/')
    if (PurePosixPath(normalized).is_absolute() or PureWindowsPath(name).is_absolute()
            or '..' in PurePosixPath(normalized).parts): raise ValueError('Output path escapes stage')
    unresolved = root / normalized
    if any(part.is_symlink() for part in (unresolved, *unresolved.parents)):
        raise ValueError('Symlink output is not a sealed stage artifact')
    path = unresolved.resolve()
    if not path.is_relative_to(root): raise ValueError('Output path escapes stage')
    return path


def _keys(stage, keys, parents, contract):
    if stage in ('raw', 'prepare'):
        if keys: raise ValueError('Upstream population stage cannot claim bounded target keys')
        return
    if (not isinstance(keys, list) or len(keys) != contract['cell_limit']
            or len({tuple(k) for k in keys}) != len(keys)):
        raise ValueError('Fresh selected-cell topology is incomplete or duplicated')
    for key in keys:
        if (not isinstance(key, list) or len(key) != 3 or not isinstance(key[0], str)
                or type(key[1]) is not int or key[2] not in PRIMITIVES):
            raise ValueError('Invalid fresh target identity')
    if len({key[2] for key in keys}) != len(keys): raise ValueError('At most one target per primitive')
    for parent in parents.values():
        if parent['stage'] not in ('raw', 'prepare') and parent['cell_keys'] != keys:
            raise ValueError('Fresh stage is crosswired to different parent cells')


def seal_stage(directory, stage, contract_ref, parents, *, cell_keys=None):
    directory = Path(directory).resolve(); keys = cell_keys or []
    contract = verify_contract(contract_ref)
    if directory != Path(contract_ref['path']).resolve().parent / stage:
        raise ValueError('Stage directory is outside its immutable run root')
    if stage not in PARENTS or set(parents) != set(PARENTS[stage]):
        raise ValueError('Missing or incorrect fresh parent stage set')
    if (directory / 'lineage.json').exists(): raise FileExistsError('Completed stage lineage is immutable')
    checked = {name: verify_stage(path, name, contract_ref) for name,path in parents.items()}
    _keys(stage, keys, checked, contract)
    outputs = {p.relative_to(directory).as_posix(): digest(safe_output(directory, p.relative_to(directory).as_posix()))
               for p in sorted(directory.rglob('*')) if p.is_file()}
    if not outputs: raise ValueError('Empty stage cannot be complete')
    value = {'schema': 'ictc-fresh-bounded-stage-v1', 'stage': stage, 'status': 'complete',
             'scope': 'bounded', 'full_reproduction': False, 'contract_sha256': contract_ref['sha256'],
             'cell_keys': keys, 'outputs': outputs,
             'parents': {name: {'path': str(Path(path).resolve()), 'sha256': digest(Path(path) / 'lineage.json')}
                         for name,path in parents.items()}}
    write_json(directory / 'lineage.json', value)
    return value


def verify_stage(directory, stage, contract_ref=None, _seen=None):
    contract_ref = contract_ref or configured_contract()
    contract = verify_contract(contract_ref)
    directory = Path(directory).resolve(strict=True)
    if stage not in PARENTS or directory != Path(contract_ref['path']).resolve().parent / stage:
        raise ValueError('Fresh parent graph is crosswired to a different run')
    seen = set() if _seen is None else set(_seen)
    if directory in seen: raise ValueError('Cyclic fresh lineage')
    seen.add(directory)
    value = json.loads((directory / 'lineage.json').read_text(encoding='utf8'))
    if (value.get('schema') != 'ictc-fresh-bounded-stage-v1' or value.get('stage') != stage
            or value.get('status') != 'complete' or value.get('scope') != 'bounded'
            or value.get('full_reproduction') is not False or value.get('contract_sha256') != contract_ref['sha256']):
        raise ValueError('Fresh manifest identity, scope or contract differs')
    parents = value.get('parents', {})
    if set(parents) != set(PARENTS[stage]): raise ValueError('Fresh stage parent set differs')
    checked = {}
    for name, record in parents.items():
        parent = Path(record['path']).resolve()
        if digest(parent / 'lineage.json') != record['sha256']: raise ValueError('Parent manifest changed')
        checked[name] = verify_stage(parent, name, contract_ref, seen)
    _keys(stage, value.get('cell_keys'), checked, contract)
    outputs = value.get('outputs')
    if not isinstance(outputs, dict) or not outputs: raise ValueError('Fresh outputs are missing')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(outputs) | {'lineage.json'}: raise ValueError('Unsealed extra or missing output file')
    for name, sha in outputs.items():
        if name == 'lineage.json' or digest(safe_output(directory, name)) != sha:
            raise ValueError('Fresh output changed')
    return value


def validate_selection(selected, g2_directory):
    value = verify_stage(g2_directory, 'g2')
    keys = [[r.subject_id, int(r.sensor_day_id), r.primitive] for r in selected.itertuples(index=False)]
    if keys != value['cell_keys']: raise ValueError('Selected rows differ from bound fresh cells')
    return len(keys)


def matching_inputs(m0, legacy=None, same_day=None, donors=None):
    base = verify_stage(m0, 'm0')
    for path, stage in [(legacy, 'legacy'), (same_day, 'same-day'), (donors, 'donors')]:
        if path is not None and verify_stage(path,stage)['cell_keys'] != base['cell_keys']:
            raise ValueError('Recovery inputs carry different fresh cell identities')
    return len(base['cell_keys'])
