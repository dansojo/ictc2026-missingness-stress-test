from pathlib import Path
import argparse,subprocess,sys
h=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
out=h/f'tests_{a.label}.txt'
assert not out.exists(),'Preserve earlier test logs; use a new label'
r=subprocess.run([sys.executable,'-X','utf8','-m','unittest','-v','test_crossday_core'],cwd=h,capture_output=True,text=True,encoding='utf8')
out.write_text(r.stdout+r.stderr,encoding='utf8')
print((r.stdout+r.stderr)[-4500:])
raise SystemExit(r.returncode)
