"""Verified stage-specific scientific sources for the historical G2 execution.

G2 was executed before Task8A changed prepared source identity. Its complete
seven-file fingerprint is recovered from checkpoint 808a6f6. Four numerical
modules load under a separate package alias; three historical metadata records
complete the fingerprint. Legacy runners are not distributed or executed.
The later, separately pinned full_vendor modules remain the Task12 authority.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import threading
import types

PAPER = Path(__file__).resolve().parent
G2_COMMIT = '808a6f6063026b5c4928d191cf1a480ed2886955'
G2_CODE_TREE_SHA256 = 'c183d9c36b0bde077ebddcdc51f801241c880e1f868026c5ed0c910f53a97c76'
G2_ALIAS = '_ictc_g2_20260809'
G2_MODULES = ('contracts', 'personalization', 'primitives', 'reliability')
_LOAD_LOCK = threading.RLock()
_MODULE_ROOT = 'ETRI_Human_AI_v2/src/semantic_indicators/'
_RUNNER_ROOT = '.superpowers/sdd/'
_EXPECTED = {
    **{_MODULE_ROOT + name + '.py': ('g2_vendor/semantic_indicators/' + name + '.py', 'scientific_module')
       for name in G2_MODULES},
    **{_RUNNER_ROOT + name: (None, 'provenance_only')
       for name in ('run_task4c2_real.py', 'aggregate_task4c2.py', 'task-4a-reviewed-g1-pair-summary.json')},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_g2_sources(package_root: Path | None = None) -> dict:
    """Verify four physical modules and all seven historically proven tree records."""
    root = (Path(package_root) if package_root is not None else PAPER).resolve()
    document = json.loads((root / 'g2_vendor_manifest.json').read_text(encoding='utf-8-sig'))
    if (set(document) != {'schema_name', 'commit', 'code_tree_sha256', 'files'}
            or document['schema_name'] != 'ictc_g2_checkpoint_sources_v1'
            or document['commit'] != G2_COMMIT or document['code_tree_sha256'] != G2_CODE_TREE_SHA256):
        raise ValueError('G2 checkpoint manifest identity differs from the executed scientific source')
    records = document['files']
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError('G2 checkpoint must contain exactly seven source/provenance files')
    if {row.get('original_path') for row in records} != set(_EXPECTED):
        raise ValueError('G2 checkpoint original source roles are incomplete or duplicated')
    tree = []
    for row in records:
        if set(row) != {'path', 'original_path', 'bytes', 'sha256', 'role'}:
            raise ValueError('G2 checkpoint file record schema changed')
        expected_path, expected_role = _EXPECTED[row['original_path']]
        relative = row['path']
        if relative != expected_path or row['role'] != expected_role:
            raise ValueError('G2 checkpoint path or role escapes the exact portable source map')
        if relative is not None:
            if (not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
                    or PureWindowsPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts):
                raise ValueError('G2 checkpoint file path is not relative')
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('G2 checkpoint file missing or outside the package')
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError('G2 checkpoint physical source differs: ' + relative)
        tree.append({'path': row['original_path'], 'bytes': row['bytes'], 'sha256': row['sha256']})
    tree.sort(key=lambda row: row['path'])
    # Original runner _file_tree_sha256/_canonical_sha256: relative paths,
    # byte counts, byte hashes; never use deployment paths or JSON file layout.
    value = json.dumps(tree, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False, default=str).encode('utf-8')
    if hashlib.sha256(value).hexdigest() != G2_CODE_TREE_SHA256:
        raise ValueError('G2 checkpoint code tree does not match the original execution fingerprint')
    return document


def load_g2_modules() -> types.SimpleNamespace:
    """Load original relative imports without replacing semantic_indicators modules."""
    with _LOAD_LOCK:
        provenance = verify_g2_sources()
        directory = (PAPER / 'g2_vendor/semantic_indicators').resolve()
        prefix = G2_ALIAS + '.'
        if G2_ALIAS not in sys.modules:
            package = types.ModuleType(G2_ALIAS)
            package.__package__ = G2_ALIAS
            package.__path__ = [str(directory)]
            package.__spec__ = ModuleSpec(G2_ALIAS, loader=None, is_package=True)
            sys.modules[G2_ALIAS] = package
        elif getattr(sys.modules[G2_ALIAS], '__path__', None) != [str(directory)]:
            raise ValueError('G2 alias is already bound to a different source directory')
        modules = {}
        try:
            for name in G2_MODULES:
                module = importlib.import_module(prefix + name)
                if Path(module.__file__).resolve() != directory / (name + '.py'):
                    raise ValueError('G2 numerical module resolved outside its checkpoint')
                modules[name] = module
        except BaseException:
            for name in list(sys.modules):
                if name == G2_ALIAS or name.startswith(prefix):
                    del sys.modules[name]
            raise
        if verify_g2_sources() != provenance:
            raise ValueError('G2 source manifest changed during numerical module loading')
        return types.SimpleNamespace(**modules, provenance=provenance)
