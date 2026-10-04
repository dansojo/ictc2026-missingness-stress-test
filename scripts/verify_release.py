"""Static public-candidate audit; no data reads or experiment execution."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse,ast,hashlib,json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))

from release_support import verify_sources

PROHIBITED={'.parquet','.pkl','.pickle','.npy','.npz','.db','.sqlite','.zip','.gz','.pdf','.png','.jpg'}
SECRET=re.compile(r'(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?<![\w-])ghp_[A-Za-z0-9]{30,}|(?<![\w-])sk-(?:proj-)?[A-Za-z0-9_-]{35,}|(?<!\w)AKIA[A-Z0-9]{16})')
PERSONAL_PATH=re.compile(r'(?:[A-Za-z]:[/\\]Users[/\\](?!PrivatePerson)|/home/(?!private_user))',re.I)

def reviewed_posters(root, problems):
    """Only explicitly reviewed poster bytes may bypass the binary-asset ban."""
    root=Path(root).resolve();manifest=root/'docs/assets/posters.json';allowed=set()
    if not manifest.exists():return allowed
    try:
        document=json.loads(manifest.read_text(encoding='utf8'))
        if not isinstance(document,dict):raise ValueError('Poster manifest must be an object')
        if document.get('status')=='pending' and document.get('assets')==[]:return allowed
        assets=document.get('assets')
        if document.get('status')!='approved' or not isinstance(assets,list) or not assets:
            raise ValueError('Poster approval manifest is incomplete')
        roles=set()
        formats={'a0_pdf':'.pdf','custom_pdf':'.pdf','a0_preview':'.png','custom_preview':'.png'}
        for item in assets:
            if not isinstance(item,dict):raise ValueError('Poster asset must be an object')
            name=item['path'];role=item['role'];relative=PurePosixPath(name)
            if (role not in formats or role in roles or name in allowed or '\\' in name
                    or relative.is_absolute() or PureWindowsPath(name).is_absolute()
                    or len(relative.parts)!=3 or relative.parts[:2]!=('docs','assets')
                    or '..' in relative.parts or relative.suffix!=formats[role]):
                raise ValueError('Invalid or duplicate poster asset')
            path=root/name
            if path.is_symlink() or not path.resolve().is_relative_to(root/'docs/assets'):
                raise ValueError('Poster path escapes its directory')
            content=path.read_bytes()
            signature=b'%PDF-' if relative.suffix=='.pdf' else b'\x89PNG\r\n\x1a\n'
            if (not content.startswith(signature) or len(content)!=item['bytes']
                    or hashlib.sha256(content).hexdigest()!=item['sha256']):
                raise ValueError('Poster bytes differ from reviewed asset')
            roles.add(role);allowed.add(name)
    except (OSError,KeyError,TypeError,ValueError) as error:
        problems.append('poster manifest: '+str(error));return set()
    return allowed

def scan(root):
    problems=[]
    poster_assets=reviewed_posters(root,problems)
    for path in Path(root).rglob('*'):
        if path.is_symlink():problems.append('symlink: '+str(path.relative_to(root)));continue
        if not path.is_file():continue
        rel=path.relative_to(root).as_posix()
        if (path.suffix.lower() in PROHIBITED and rel not in poster_assets) or path.name=='.env':problems.append('prohibited asset: '+rel)
        if path.suffix in ('.py','.json','.md','.txt','.csv'):
            text=path.read_text(encoding='utf-8-sig')
            if SECRET.search(text):problems.append('secret pattern: '+rel)
            for line in text.splitlines():
                synthetic=path.name.startswith('test_') and ('PrivatePerson' in line or 'private_user' in line)
                own_detector=path.name=='verify_release.py' and line.startswith('PERSONAL_PATH=')
                if PERSONAL_PATH.search(line) and not synthetic and not own_detector:
                    problems.append('personal path: '+rel);break
            if path.suffix=='.py':
                try:ast.parse(text,filename=rel)
                except SyntaxError:problems.append('invalid Python: '+rel)
        if path.suffix=='.csv' and path.name not in ('paper-targets.csv',):
            problems.append('unreviewed CSV: '+rel)
    return problems

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    problems=scan(a.root)
    try:count=verify_sources(a.root)
    except (ValueError,OSError) as error:problems.append('source manifest: '+str(error));count=0
    result=dict(status='passed' if not problems else 'failed',source_files_checked=count,files=sum(x.is_file() for x in a.root.rglob('*')),issues=problems,scope='static selected-content scan, not a proof of legal redistribution rights')
    print(json.dumps(result,indent=2));return int(bool(problems))

if __name__=='__main__':raise SystemExit(main())
