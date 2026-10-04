"""Run public synthetic/unit checks; never requires provider data."""
from pathlib import Path
import os,subprocess,sys,tempfile

if sys.flags.optimize:
    raise RuntimeError('Assertions must remain enabled; do not use -O or PYTHONOPTIMIZE')

ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=os.pathsep.join([str(ROOT/'scripts'),str(ROOT/'scripts/_support'),str(ROOT),str(ROOT/'original_paper_reproduction/paper'),str(ROOT/'extensions/2026-09-10/reviewer_extension')])
groups=[(ROOT,['discover','-s','tests','-p','test_*.py','-v']),
 (ROOT/'original_paper_reproduction/paper',['discover','-s','tests','-v']),
 (ROOT/'original_paper_reproduction',['discover','-s','tests','-p','test_full_*.py','-v']),
 (ROOT/'original_paper_reproduction/leaderboard',['discover','-s','tests','-p','test_data_preparation.py','-v']),
 (ROOT/'extensions/2026-09-11/imputation_extension',['-v','test_imputation','test_summary']),
 (ROOT/'extensions/2026-09-11/crossday_imputation_v1',['-v','test_crossday_core','test_crossday_adapter','test_crossday_summary','test_crossday_audit_precision']),
 (ROOT/'extensions/2026-09-11/crossday_imputation_v1/donor_inventory',['-v','test_inventory']),
 (ROOT/'extensions/2026-09-11/external_prediction',['-v','test_prediction_core','test_window_contracts'])]
with tempfile.TemporaryDirectory(prefix='ictc-tests-') as temp:
    env['MPLCONFIGDIR']=temp
    results=[subprocess.run([sys.executable,'-B','-m','unittest',*args],cwd=cwd,env=env).returncode for cwd,args in groups]
raise SystemExit(int(any(results)))
