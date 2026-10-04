"""Portable read-only inputs and private output boundaries."""
from pathlib import Path
import hashlib,json,math

def failure_summary(error):
    """Only error type is safe for an aggregate report; console logs stay private."""
    return {'status':'failed','error_type':type(error).__name__}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fresh_output(output,inputs,code_root):
    output=Path(output).resolve()
    for root in [*inputs,code_root]:
        root=Path(root).resolve()
        if output==root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Output overlaps protected input or release code')
    if output.exists():raise FileExistsError('Output must be new')
    output.mkdir(parents=True)
    return output

def checked_file(root,relative,expected):
    root=Path(root).resolve()
    target=(root/relative).resolve()
    if not target.is_relative_to(root):raise ValueError('Manifest path escapes root')
    if sha256(target)!=expected:raise ValueError('Artifact SHA-256 mismatch')
    return target

def compare_scalar(actual,expected,rtol=1e-12,atol=1e-12):
    actual,expected=float(actual),float(expected)
    both_nan=math.isnan(actual) and math.isnan(expected)
    exact=both_nan or actual==expected
    if not exact and not math.isclose(actual,expected,rel_tol=rtol,abs_tol=atol):
        raise AssertionError('Numerical comparison failed')
    return {'exact':exact,'absdiff':0. if both_nan else abs(actual-expected)}

def verify_sources(root):
    root=Path(root)
    entries=json.loads((root/'docs/provenance/SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    for entry in entries:checked_file(root,entry['destination'],entry['sha256'])
    return len(entries)

def verify_selected_pins(root,pins):
    root=Path(root)
    entries=json.loads((Path(__file__).resolve().parents[2]/'docs/provenance/SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    release=Path(__file__).resolve().parents[2]
    checked=0
    for relative,expected in pins.items():
        file=Path(root)/relative
        if file.suffix=='.py' or file.name=='config.json':
            path=file.resolve()
            if not path.is_relative_to(release):
                checked_file(root,relative,expected);checked+=1;continue
            name=path.relative_to(release).as_posix()
            entry=next((e for e in entries if e['destination']==name),None)
            if entry is None or entry.get('original_sha256',entry['sha256'])!=expected:
                raise ValueError('Historical source pin differs from original source ledger')
            checked_file(root,relative,entry['sha256']);checked+=1
    return checked

def prediction_index(frame,masked=False):
    keys=['dataset','participant','window_id']+(['draw','geometry'] if masked else [])
    if frame.duplicated(keys).any():raise ValueError('Duplicate prediction key')
    value='p' if masked else 'p_observed'
    return {tuple(getattr(row,k) for k in keys):getattr(row,value) for row in frame.itertuples(index=False)}

class Inputs:
    """Keep input hashes for before/after verification; never write into input roots."""
    def __init__(self,hub):
        self.hub=Path(hub).resolve(strict=True);self.read={}
    def check(self,path,expected=None):
        path=Path(path).resolve(strict=True)
        if not path.is_relative_to(self.hub):raise ValueError('Input escapes configured archive')
        actual=sha256(path)
        if expected and actual!=expected:raise ValueError('Historical input hash mismatch')
        self.read[path]=actual
        return path
    def json(self,path,expected=None):
        return json.loads(self.check(path,expected).read_text(encoding='utf-8-sig'))
    def output(self,folder,relative,manifest):
        name=next((k for k in manifest['outputs'] if k.replace('\\','/')==relative.replace('\\','/')),None)
        if name is None:raise ValueError('Requested input missing from historical manifest')
        value=manifest['outputs'][name]
        expected=value if isinstance(value,str) else value['sha256']
        return self.check(checked_file(folder,relative,expected),expected)
    def finish(self):
        for path,expected in self.read.items():
            if sha256(path)!=expected:raise ValueError('Input changed during execution')
        return len(self.read)
