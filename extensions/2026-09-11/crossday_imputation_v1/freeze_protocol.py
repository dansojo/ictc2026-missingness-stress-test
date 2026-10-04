"""Freeze actual inventory, settings and all execution/verification sources before outcomes."""
from pathlib import Path
import json
from run_crossday import HERE,OLD,PRE,PRIOR,sha,stamp,write

def freeze():
    destination=HERE/'FROZEN_PROTOCOL.json'
    if destination.exists():raise ValueError('Existing freeze must never be overwritten')
    if list(HERE.glob('*/recovery_rows.parquet')):raise ValueError('Cannot assert before-outcome freeze after results exist')
    inventory=HERE/'donor_inventory'
    manifest=json.loads((inventory/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='complete' or len(manifest['partitions'])!=50:raise ValueError('Full donor inventory required')
    files={}
    def add(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected and actual!=expected:raise ValueError('Historical manifest hash differs: '+str(path))
        files[str(path)]=actual
    source_names=['DESIGN_DRAFT.md','PROTOCOL_FINAL.md','config.json','crossday_core.py','run_crossday.py','freeze_protocol.py',
        'summarize_crossday.py','verify_crossday.py','independent_code_review.md','test_crossday_core.py',
        'test_crossday_adapter.py','test_crossday_summary.py','tests_red_01.txt','tests_green_02.txt',
        'adapter_tests_green_01.txt','summary_tests_green_04.txt','all_tests_final_01.txt']
    for name in source_names:add(HERE/name)
    for p in inventory.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.log','.pyc'):add(p)
    for path,h in manifest['input_hashes'].items():add(path,h)
    for name in ['run_recovery_v2.py','recovery_core.py','recovery_support.py']:add(OLD/name)
    add(PRE/'manifest.json')
    pm=json.loads((PRE/'manifest.json').read_text(encoding='utf8'))
    for name,h in pm['outputs'].items():add(PRE/name,h)
    for path,h in pm['input_hashes'].items():add(path,h)
    add(PRIOR/'recovery_rows.parquet','90007efab549a4de69e6fdeb9e67fad731e0ca5b49f722b6a0b660043a17645e')
    add(PRIOR/'manifest.json','0fa49cb485e787eb45337c7d5d0c111750b8620095871f79a0c18d8f47ee0791')
    add(PRIOR.parent/'FROZEN_PROTOCOL.json','17d489b789021cad15008d6235dcb28bb5c9c32567980c5cc5f01007dbd1222a')
    revision4=HERE.parent/'READY_FOR_REVIEW/ICTC2026_TAE_HO_KIM_REVIEWER_REVISION4.pdf'
    add(revision4,'1ccb12b0c5e20636e7335cb09915916341bd7e0c6dc7aca1297f61db994e1db7')
    add(HERE.parent/'camera_ready_revision4/paper_ictc2026/05_manuscript/main_expanded.tex')
    # Protect complete prior revision/recovery artifacts, beyond the main PDF and row table.
    for folder in ['imputation_extension','camera_ready_revision4','READY_FOR_REVIEW']:
        for p in (HERE.parent/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:add(p)
    completion=HERE.parent/'COMPLETION_VERIFICATION.json'
    add(completion)
    for relative in json.loads(completion.read_text(encoding='utf8'))['experiment_output_paths']:
        add(HERE.parent/relative)
    write(destination,dict(schema='crossday-frozen-protocol-v1',frozen_utc=stamp(),
        new_crossday_results_observed=False,prior_revision4_results_known=True,
        data_inventory_complete=True,configuration_chosen_without_crossday_scores=True,
        files=files,protocol_source_names=source_names))
    print(json.dumps(dict(frozen_path=str(destination),sha256=sha(destination),pinned_files=len(files)),indent=2))

if __name__=='__main__':freeze()
