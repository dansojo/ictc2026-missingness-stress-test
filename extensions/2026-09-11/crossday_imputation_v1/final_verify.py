"""Fresh final gates: immutable experiment, amended audit, author tables and figures."""
from pathlib import Path
import json
from datetime import datetime,timezone
import pandas as pd
from run_crossday import HERE,sha,check_pins
from build_author_report import METHODS,LABELS,PRIMITIVES

def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def run():
    output=HERE/'FINAL_VERIFICATION.json'
    if output.exists():raise ValueError('Preserve final verification history')
    frozen=check_pins()
    count=0
    for folder in ['full_01','summary_01']:
        m=load(HERE/folder/'manifest.json')
        assert m['status']=='complete'
        for name,h in m['outputs'].items():assert sha(HERE/folder/name)==h;count+=1
    audit=load(HERE/'independent_audit_02/verification.json')
    assert audit['status']=='passed'
    assert sha(HERE/'verify_crossday_v2.py')==audit['verifier_sha256']
    assert sha(HERE/'full_01/manifest.json')==audit['run_manifest_sha256']
    assert sha(HERE/'summary_01/manifest.json')==audit['summary_manifest_sha256']
    for name,h in audit['audit_outputs'].items():assert sha(HERE/'independent_audit_02'/name)==h
    amendment=load(HERE/'AUDIT_AMENDMENT_01.json')
    assert not amendment['production_or_results_changed']
    for path,h in amendment['files'].items():assert sha(path)==h
    assert load(HERE/'independent_audit_01/verification.json')['status']=='failed'
    boundary=load(HERE/'independent_audit_01/binary_aggregate_scan.json')
    assert boundary['aggregates']==39200 and boundary['production_vote_mismatches']==0
    inp=load(HERE/'PROTOCOL_NUMERIC_CHECK.json');finding=load(HERE/'FINDINGS_NUMERIC_CHECK.json')
    assert inp['status']=='passed' and not inp['errors']
    assert finding['status']=='passed_provisional_report_consistency' and not finding['errors']
    assert sha(HERE/'FINDINGS_KO.md')==finding['report_sha256']
    assert sha(HERE/'PROTOCOL_FINAL.md')==inp['protocol_sha256']
    visual=load(HERE/'figures_02/VISUAL_REVIEW.json')
    assert visual['status']=='passed' and visual['source_summary_sha256']==sha(HERE/'summary_01/summary.csv')
    assert visual['renderer_sha256']==sha(HERE/'plot_effects.py')
    for name,h in visual['outputs'].items():assert sha(HERE/'figures_02'/name)==h
    report=(HERE/'AUTHOR_REPORT.md').read_text(encoding='utf8')
    assert (HERE/'FINDINGS_KO.md').read_text(encoding='utf8') in report
    summaries=pd.read_csv(HERE/'summary_01/summary.csv',float_precision='round_trip')
    fallback=pd.read_csv(HERE/'summary_01/fallback_totals.csv',float_precision='round_trip')
    checked=0
    def actual_table(heading):
        section=report.split(heading+'\n',1)[1].split('\n##',1)[0]
        return [line for line in section.splitlines() if any(line.startswith('| '+label+' |') for label in LABELS)]
    for comp,clabel in [('M0','대치하지 않은 경우'),('CD_ALL_DAY','다른 날 전체 평균')]:
        for geometry,glabel in [('contiguous_20pct','연속 결측'),('scattered_random_20pct','무작위 분산 결측')]:
            rows=actual_table('### '+glabel+': '+clabel+' 대비');assert len(rows)==8
            for line,method,label in zip(rows,METHODS,LABELS):
                fields=[x.strip() for x in line.strip('|').split('|')];assert fields[0]==label
                for actual,primitive in zip(fields[1:],PRIMITIVES):
                    q=summaries.loc[(summaries.primitive==primitive)&(summaries.geometry==geometry)&(summaries.comparator==comp)&(summaries.method==method)].iloc[0]
                    expected=format(q.median_gain,'+.4f') if bool(q.applicable) else 'N/A'
                    assert actual==expected,(line,expected);checked+=1
    for geometry,glabel in [('contiguous_20pct','연속 결측'),('scattered_random_20pct','무작위 분산 결측')]:
        rows=actual_table('### '+glabel+' fallback 비율');assert len(rows)==8
        for line,method in zip(rows,METHODS):
            fields=[x.strip() for x in line.strip('|').split('|')]
            for actual,primitive in zip(fields[1:],PRIMITIVES):
                q=fallback.loc[(fallback.primitive==primitive)&(fallback.geometry==geometry)&(fallback.method==method)]
                numerator=q.fallback_n.sum();denominator=q.attempted_n.sum()
                expected=format(100*numerator/denominator,'.2f')+'%' if denominator else 'N/A'
                assert actual==expected,(line,expected);checked+=1
    result=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
        original_frozen_protocol_sha256=sha(HERE/'FROZEN_PROTOCOL.json'),original_frozen_files=len(frozen['files']),
        original_experiment_outputs_checked=count,amended_audit_status=audit['status'],
        audit_amendment_sha256=sha(HERE/'AUDIT_AMENDMENT_01.json'),
        original_audit_failure_preserved=True,production_or_results_changed=False,
        binary_archive_aggregates_exact_vote_checked=39200,
        independent_raw_values=audit['independent_raw_values'],independent_raw_scope='draw0, all500cells, bothgeometries, allapplicablemethods',
        independent_all_row_scope='all400000newrows, allsummaryeffects/intervals/counts',
        protocol_numeric_statements=inp['statement_count'],findings_numeric_statements=finding['statement_count'],
        author_table_cells_checked=checked,figures_visually_reviewed=2,manuscript_uploaded=False,
        deliverable_hashes={name:sha(HERE/name) for name in ['AUTHOR_REPORT.md','FINDINGS_KO.md','README.md','PROTOCOL_FINAL.md',
            'PROTOCOL_NUMERIC_CHECK.json','FINDINGS_NUMERIC_CHECK.json','AUDIT_AMENDMENT_01.md',
            'independent_audit_02/verification.json','figures_02/VISUAL_REVIEW.json','build_author_report.py','final_verify.py']})
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='deliverable_hashes'},indent=2))

if __name__=='__main__':run()
