"""Explicit private paths; historical hashes are never substituted or rewritten.

Use portable_run.py. Configuration and run outputs belong outside this release.
No network access, data acquisition, or recursive copy is performed here.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os

RELEASE = Path(__file__).resolve().parents[2]


def configuration():
    name = os.environ.get('ICTC_RUN_CONFIG')
    return json.loads(Path(name).read_text(encoding='utf-8-sig')) if name else None


def role(name, default=None):
    cfg = configuration()
    if cfg is None:
        if default is None: raise ValueError('Run through portable_run.py with private configuration')
        return Path(default)
    value = cfg.get('roles', {}).get(name)
    if value is None:
        if default is not None: return Path(default)
        raise ValueError('Missing configured role: ' + name)
    path = Path(value).resolve()
    if not any(path.is_relative_to(Path(root).resolve()) for root in cfg.get('read_roots', [])):
        raise ValueError('Configured role is outside explicit read roots: ' + name)
    return path


def resolve_input(value):
    cfg = configuration()
    if cfg is None: return Path(value)
    text = str(value).replace('\\', '/').rstrip('/')
    for mapping in sorted(cfg.get('mappings', []), key=lambda x: len(x['from']), reverse=True):
        old = mapping['from'].replace('\\', '/').rstrip('/')
        if text.casefold() == old.casefold() or text.casefold().startswith(old.casefold() + '/'):
            relative = text[len(old):].lstrip('/')
            if '..' in PurePosixPath(relative).parts: raise ValueError('Historical path escapes mapped root')
            root = Path(mapping['to']).resolve()
            if not any(root.is_relative_to(Path(allowed).resolve()) for allowed in cfg.get('read_roots', [])):
                raise ValueError('Mapped input is outside explicit read roots')
            result = (root / relative).resolve()
            if not result.is_relative_to(root): raise ValueError('Historical path escapes mapped root')
            return result
    path = Path(value).resolve()
    if not path.exists(): raise FileNotFoundError('No configured mapping for historical input')
    if not any(path.is_relative_to(Path(root).resolve()) for root in cfg.get('read_roots', [])):
        raise ValueError('Input is outside configured read roots')
    return path


def mapped_pins(pins):
    result = {}
    for path, expected in pins.items():
        target = str(resolve_input(path))
        if target in result and result[target] != expected: raise ValueError('Conflicting mapped hashes')
        result[target] = expected
    return result


def check_consumed_tree(root, pins, marker):
    """Bind the actual configured tree to historical pins, including donor days."""
    root = Path(root).resolve(); checked = 0
    for historical, expected in pins.items():
        name = historical.replace('\\', '/')
        if marker not in name: continue
        relative = name.split(marker, 1)[1]
        if '..' in PurePosixPath(relative).parts: raise ValueError('Invalid frozen relative path')
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or digest(file) != expected:
            raise ValueError('Consumed input differs from frozen bytes')
        checked += 1
    if not checked: raise ValueError('No frozen hashes bind the consumed input tree')
    return checked


def original_source(code):
    cfg = configuration()
    if cfg is None: return Path(code)
    code = Path(code).resolve()
    for item in sorted(cfg.get('source_roots', []), key=lambda x: len(x['code']), reverse=True):
        root = Path(item['code']).resolve()
        if code.is_relative_to(root): return resolve_input(Path(item['original']) / code.relative_to(root))
    raise ValueError('No original-source mapping for packaged code')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def check_source(code, expected, original=None):
    """Check old frozen bytes AND reviewed packaged bytes, independently."""
    code = Path(code).resolve()
    if configuration() is None:
        if digest(code) != expected: raise ValueError('Frozen source changed')
        return
    source = original_source(code) if original is None else resolve_input(original)
    if digest(source) != expected: raise ValueError('Original frozen source changed')
    entries = json.loads((RELEASE / 'docs/provenance/SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    name = code.relative_to(RELEASE).as_posix()
    entry = next((e for e in entries if e['destination'] == name), None)
    if entry is None:
        # Historical protocol notes and execution logs need not be public assets.
        if code.exists(): raise ValueError('Unlisted packaged source')
        return
    if entry.get('original_sha256', entry['sha256']) != expected or digest(code) != entry['sha256']:
        raise ValueError('Packaged source differs from reviewed adaptation')


def validate_output(output, *, existing=False):
    path = Path(output).resolve()
    cfg = configuration() or {}
    protected = [RELEASE, *map(Path, cfg.get('read_roots', [])),
                 *[Path(v) for k,v in cfg.get('roles', {}).items() if k != 'work_output'],
                 *[Path(m['to']) for m in cfg.get('mappings', [])],
                 *[Path(m['original']) for m in cfg.get('source_roots', [])]]
    for root in protected:
        root = root.resolve()
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError('Output overlaps protected inputs or release')
    if path.exists() and not existing: raise FileExistsError('Output must be new')
    return path
