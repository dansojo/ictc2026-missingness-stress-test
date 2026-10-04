"""Bounded raw-to-recovery execution, explicitly separate from a full study.

All study calculations are original functions. This driver limits target cells,
retains the full population for selection/scales and the complete within-person
donor calendar, and seals every consumed parent independently of archive pins.
Outputs contain sensitive records: use a private directory outside the release.
"""
from pathlib import Path
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))

from portable_paths import RELEASE, digest, validate_output, role
import fresh_lineage as lineage


def setup_imports():
    paths = [RELEASE / 'original_paper_reproduction/paper',
             RELEASE / 'original_paper_reproduction/leaderboard',
             RELEASE / 'extensions/2026-09-10/reviewer_extension',
             RELEASE / 'extensions/2026-09-11/imputation_extension',
             RELEASE / 'extensions/2026-09-11/crossday_imputation_v1',
             RELEASE / 'extensions/2026-09-11/crossday_imputation_v1/donor_inventory']
    sys.path[:0] = [str(p) for p in paths]


def build_donors(root, keys):
    """Original inventory packing for ALL dates in each selected partition."""
    import numpy as np
    import pandas as pd
    import run_recovery_v2 as legacy
    import inventory_core as core
    out = root / 'donors'
    (out / 'cache').mkdir(parents=True, exist_ok=False)
    calendar_columns = ['subject_id','lifelog_date','sleep_date','sensor_day_id',
                        'collection_start_date','elapsed_day_index','any_source_record_observed',
                        'any_measurement_support_observed','any_sensor_observed']
    calendar = pd.read_parquet(root / 'prepare/primitives.parquet', columns=calendar_columns)
    if len(calendar) != 853 or calendar.subject_id.nunique() != 10:
        raise ValueError('Fresh donor population calendar is incomplete')
    canonical = root / 'raw/canonical'
    target_cells = {}
    for path in sorted((root / 'm0/cells').glob('*.pkl')):
        with path.open('rb') as stream: cell = pickle.load(stream)
        target_cells[(cell['subject_id'],cell['sensor_day_id'],cell['primitive'])] = cell
    if set(target_cells) != {tuple(k) for k in keys}: raise ValueError('Fresh donor targets differ')
    records = []
    for subject, day, primitive in keys:
        indexes = {}; pool = {}
        for row in calendar.loc[calendar.subject_id.eq(subject)].sort_values('sensor_day_id').itertuples(index=False):
            series = pd.Series(row._asdict())
            prepared = legacy._prepare_selected_cell(primitive, series, canonical_root=canonical, sensor_indexes=indexes)
            cell = core.pack_view(legacy.primitives.export_prepared_phase_view(prepared),
                                  legacy.reliability._calendar_row_for_replay(series))
            cell['original_source_hash'] = digest(canonical / legacy.DEFS[primitive].sensor_file)
            pool[(subject, primitive, int(row.sensor_day_id))] = cell
        target = pool[(subject,primitive,day)]; original = target_cells[(subject,day,primitive)]
        for field in ['raw','valid','times']:
            np.testing.assert_array_equal(target[field],original[field],strict=True)
        for field in ['raw_frame','calendar']:
            pd.testing.assert_frame_equal(target[field],original[field],check_exact=True)
        for field in ['original_source_hash','source_record_digest']:
            if target[field] != original[field]: raise ValueError('Fresh target identity differs')
        others = [c for c in pool.values() if c['day_start_ns'] != target['day_start_ns']]
        overlap = sum(core.target_overlap_count(c,target) for c in others)
        shared = sum(len(np.intersect1d(c['times'],target['times'])) for c in others)
        if overlap or shared: raise ValueError('Donor records overlap target date')
        path = out / 'cache' / f'{subject}__{primitive}.pkl'
        with path.open('wb') as stream: pickle.dump(pool, stream, protocol=5)
        records.append(dict(subject_id=subject, primitive=primitive, calendar_dates=len(pool),
                            target_exact=True, donor_overlap_records=overlap, shared_timestamps=shared,
                            path=path.relative_to(out).as_posix(),sha256=digest(path)))
        print(json.dumps({'stage':'donors','partitions':len(records),'calendar_dates':len(pool)}),flush=True)
    lineage.write_json(out / 'manifest.json', dict(status='complete',scope='bounded',
                       full_reproduction=False,partitions=records,new_model_fits=0))


def run(config_path, output, cell_limit):
    if sys.flags.optimize: raise ValueError('Optimized Python disables original scientific assertions')
    if not 1 <= cell_limit <= 5: raise ValueError('Only one to five target cells are authorized by this driver')
    config_path = Path(config_path).resolve(strict=True)
    os.environ['ICTC_RUN_CONFIG'] = str(config_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if cfg.get('execution_mode') == 'fresh_bounded': raise ValueError('Use the original private paths configuration')
    output = validate_output(output)
    output.mkdir(parents=True,exist_ok=False)
    setup_imports()
    from release_support import verify_sources
    verify_sources(RELEASE)
    import bounded_upstream as upstream
    raw_root = role('raw')
    preprocess = role('base_inputs') / 'validation/preprocessing/raw_run/preprocessing_run.json'
    raw_pins = json.loads(preprocess.read_text(encoding='utf8'))['input_sha256']
    code = {p.relative_to(RELEASE).as_posix():(p,digest(p)) for p in RELEASE.rglob('*')
            if p.is_file() and (p.suffix == '.py' or p.name == 'config.json' or p.name == 'SOURCE_MANIFEST.json')}
    frozen = {name:(role(name),digest(role(name))) for name in ('same_day_protocol','cross_day_protocol')}
    frozen.update(preprocessing=(preprocess,digest(preprocess)),private_config=(config_path,digest(config_path)))
    ref = lineage.create_contract(output, {name:(raw_root/name,sha) for name,sha in raw_pins.items()},
                                  code, frozen, cell_limit=cell_limit)
    cfg['historical_roles'] = dict(cfg['roles'])
    cfg['roles'].update(base_fresh=str(output),m0=str(output/'m0'),legacy_results=str(output/'legacy'),
                        same_day_results=str(output/'same-day'),donor_inputs=str(output/'donors'))
    cfg['read_roots'] = [*cfg['read_roots'],str(output)]
    cfg.update(execution_mode='fresh_bounded',fresh_contract=ref)
    lineage.write_json(output / 'execution-paths.json', cfg)
    os.environ['ICTC_RUN_CONFIG'] = str(output / 'execution-paths.json')
    active_config_sha256 = digest(output / 'execution-paths.json')
    started = time.perf_counter(); keys = []
    manifest = dict(status='running',scope='bounded',full_reproduction=False,cell_limit=cell_limit,
                    run_contract=ref,execution_config_sha256=active_config_sha256,stages=[],new_model_fits=0)
    def execute(stage, operation):
        lineage.verify_active_config(output/'execution-paths.json',active_config_sha256)
        parents = {name:output/name for name in lineage.PARENTS[stage]}
        lineage.verify_contract(ref)
        for name,path in parents.items(): lineage.verify_stage(path,name,ref)
        print(json.dumps(dict(stage=stage,status='running')),flush=True)
        value = operation()
        lineage.verify_active_config(output/'execution-paths.json',active_config_sha256)
        lineage.seal_stage(output/stage, stage, ref, parents, cell_keys=keys if stage not in ('raw','prepare') else [])
        manifest['stages'].append(dict(stage=stage,lineage_sha256=digest(output/stage/'lineage.json')))
        lineage.write_json(output/'chain-manifest.json',manifest)
        print(json.dumps(dict(stage=stage,status='complete')),flush=True)
        return value
    try:
        execute('raw',lambda:upstream.run_raw(raw_root,output/'raw'))
        canonical = output/'raw/canonical'
        execute('prepare',lambda:upstream.run_prepare(canonical,output/'prepare'))
        def g2():
            nonlocal keys
            keys = [list(k) for k in upstream.run_g2(output/'prepare',canonical,output/'g2',cell_limit)]
        execute('g2',g2)
        execute('stress',lambda:upstream.run_stress(output/'prepare',canonical,output/'g2',output/'stress'))
        import run_recovery_v2 as legacy
        execute('m0',lambda:legacy.preflight(output/'m0',cell_limit))
        execute('legacy',lambda:legacy.execute(output/'m0',output/'legacy'))
        execute('donors',lambda:build_donors(output,keys))
        import run_extension
        execute('same-day',lambda:run_extension.run(output/'same-day',cell_limit))
        import run_crossday
        execute('cross-day',lambda:run_crossday.run(output/'cross-day',cell_limit))
        manifest.update(status='complete',cell_keys=keys,seconds=time.perf_counter()-started)
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc),seconds=time.perf_counter()-started)
        raise
    finally:
        lineage.write_json(output/'chain-manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells',type=int,choices=range(1,6),default=5)
    args = parser.parse_args()
    run(args.config,args.output,args.cells)
